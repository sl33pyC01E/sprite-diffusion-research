"""Train start/middle/start latent interpolation on the dense MUGEN corpus."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np

from spritelab.latent_keypose_train import build_keypose_action_bundles
from spritelab.latent_motion_train import (
    LatentMotionTrainingCorpus,
    _ema_update,
    _file_sha256,
    _load_frozen_decoder,
    _runtime_facts,
    _target_directed_motion_floor_loss,
    load_latent_motion_training_corpus,
)
from spritelab.models.anchored_latent_motion_dit import (
    AnchoredActionConditionedLatentMotionDiT,
    apply_latent_anchors,
    masked_velocity_mse,
)
from spritelab.models.latent_motion_dit import LatentMotionDiTConfig
from spritelab.models.sprite_autoencoder import sprite_reconstruction_loss
from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency boundary
    torch = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None

Precision = Literal["float32", "bfloat16"]


class AnchoredMotionTrainingError(ValueError):
    """Raised when anchored interpolation training is invalid."""


@dataclass(frozen=True, slots=True)
class AnchoredMotionTrainingConfig:
    """Fixed eight-frame start/middle/start interpolation contract."""

    anchor_frame_indices: tuple[int, ...] = (0, 4, 7)
    canonical_middle_frame_index: int = 4
    gradient_accumulation: int = 2
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.9995
    latent_weight: float = 1.0
    pixel_weight: float = 2.0
    pixel_action_contrast_weight: float = 0.5
    target_directed_motion_weight: float = 2.0
    minimum_target_motion_progress: float = 0.8
    action_token_count: int = 4
    action_condition_scale: float = 2.0
    steps: int = 30_000
    log_every: int = 25
    validate_every: int = 500
    checkpoint_every: int = 2_500
    validation_identities: int = 8
    seed: int = 20260831
    device: str = "cuda"
    precision: Precision = "bfloat16"
    model: LatentMotionDiTConfig = LatentMotionDiTConfig(
        latent_size=64,
        num_frames=8,
        latent_channels=8,
        patch_size=4,
        model_dim=384,
        depth=12,
        num_heads=6,
        condition_dim=384,
    )

    def __post_init__(self) -> None:
        if self.anchor_frame_indices != (0, 4, 7):
            raise ValueError("v1 anchored interpolation requires frames (0,4,7)")
        if self.canonical_middle_frame_index != 4:
            raise ValueError("v1 canonical middle frame must be frame 4")
        if self.model.num_frames != 8:
            raise ValueError("anchored motion model must contain eight frames")
        for name in (
            "gradient_accumulation",
            "action_token_count",
            "steps",
            "log_every",
            "validate_every",
            "checkpoint_every",
            "validation_identities",
            "seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0,steps)")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "latent_weight",
            "pixel_weight",
            "pixel_action_contrast_weight",
            "target_directed_motion_weight",
            "action_condition_scale",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        if not 0 < self.minimum_target_motion_progress <= 1:
            raise ValueError("minimum_target_motion_progress must be in (0,1]")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")


@dataclass(frozen=True, slots=True)
class AnchoredMotionTrainingResult:
    output_directory: Path
    report_path: Path
    checkpoint_path: Path
    report_sha256: str


def _anchored_batch(
    runtime: Any,
    corpus: LatentMotionTrainingCorpus,
    selection: tuple[int, ...],
    *,
    config: AnchoredMotionTrainingConfig,
    device: Any,
    mean: Any,
    std: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Build normalized authored interiors with exact start/middle/start anchors."""

    indices = list(selection)
    target = runtime.from_numpy(corpus.target_latents[indices].astype(np.float32)).to(device)
    reference = runtime.from_numpy(corpus.reference_latents[indices].astype(np.float32)).to(device)
    target = (target - mean) / std
    reference = (reference - mean[:, 0]) / std[:, 0]
    clean = target - reference.unsqueeze(1)
    clean = clean.clone()
    clean[:, 0] = 0
    clean[:, 7] = 0
    anchors = runtime.zeros_like(clean)
    anchors[:, config.canonical_middle_frame_index] = clean[:, config.canonical_middle_frame_index]
    mask = runtime.zeros((len(indices), 8), dtype=runtime.bool, device=device)
    mask[:, list(config.anchor_frame_indices)] = True
    rgba = (
        runtime.from_numpy(corpus.target_rgba[indices])
        .to(device=device, dtype=runtime.float32)
        .permute(0, 1, 4, 2, 3)
        .div(255)
    )
    phases = runtime.from_numpy(corpus.phases[indices]).to(device)
    actions = runtime.tensor([corpus.rows[index].action_index for index in indices], device=device)
    return clean, reference, rgba, phases, actions, anchors, mask


def _trajectory_bundle_metrics(runtime: Any, *, predicted_rgba: Any, target_rgba: Any) -> dict:
    """Measure missing-frame fidelity and action separation without anchor leakage."""

    expected = tuple(target_rgba.shape)
    if len(expected) != 5 or expected[0] < 3 or expected[2] != 4:
        raise ValueError("trajectory bundles must have shape [B>=3,T,4,H,W]")
    if tuple(predicted_rgba.shape) != expected:
        raise ValueError("trajectory bundles must share one shape")
    predicted_pm = _premultiplied(runtime, predicted_rgba)
    target_pm = _premultiplied(runtime, target_rgba)
    own = runtime.stack(
        [runtime.nn.functional.l1_loss(predicted_pm[i], target_pm[i]) for i in range(expected[0])]
    )
    wrong = runtime.stack(
        [
            runtime.stack(
                [
                    runtime.nn.functional.l1_loss(predicted_pm[i], target_pm[j])
                    for j in range(expected[0])
                    if j != i
                ]
            ).min()
            for i in range(expected[0])
        ]
    )
    target_distances = runtime.stack(
        [
            runtime.nn.functional.l1_loss(target_pm[left], target_pm[right])
            for left, right in combinations(range(expected[0]), 2)
        ]
    )
    generated_distances = runtime.stack(
        [
            runtime.nn.functional.l1_loss(predicted_pm[left], predicted_pm[right])
            for left, right in combinations(range(expected[0]), 2)
        ]
    )
    return {
        "correct_target_preference_rate": (own < wrong).to(runtime.float32).mean(),
        "generated_action_separation": generated_distances.mean(),
        "premultiplied_rgba_mae": runtime.nn.functional.l1_loss(predicted_pm, target_pm),
        "target_action_separation": target_distances.mean(),
    }


def _pixel_action_bundle_loss(runtime: Any, *, predicted_rgba: Any, target_rgba: Any) -> Any:
    predicted_pm = _premultiplied(runtime, predicted_rgba)
    target_pm = _premultiplied(runtime, target_rgba)
    own = [
        runtime.nn.functional.l1_loss(predicted_pm[i], target_pm[i]) for i in range(len(target_pm))
    ]
    losses = []
    for left, right in combinations(range(len(target_pm)), 2):
        distance = runtime.nn.functional.l1_loss(target_pm[left], target_pm[right])
        if float(distance.detach().cpu()) <= 1e-4:
            continue
        denominator = distance.clamp_min(1e-4)
        losses.extend(
            (
                runtime.nn.functional.l1_loss(
                    predicted_pm[left] - predicted_pm[right],
                    target_pm[left] - target_pm[right],
                )
                / denominator,
                runtime.relu(
                    own[left]
                    - runtime.nn.functional.l1_loss(predicted_pm[left], target_pm[right])
                    + 0.1 * distance
                )
                / denominator,
                runtime.relu(
                    own[right]
                    - runtime.nn.functional.l1_loss(predicted_pm[right], target_pm[left])
                    + 0.1 * distance
                )
                / denominator,
            )
        )
    if not losses:
        return predicted_rgba.sum() * 0
    return runtime.stack(losses).mean()


def sample_anchored_motion_residual(
    runtime: Any,
    model: Any,
    *,
    noise: Any,
    reference: Any,
    actions: Any,
    phases: Any,
    anchor_residuals: Any,
    anchor_mask: Any,
) -> Any:
    """One-step endpoint sample with anchors clamped before and after prediction."""

    state = apply_latent_anchors(noise, anchor_residuals, anchor_mask)
    times = runtime.ones((state.shape[0],), device=state.device)
    velocity = model(
        state,
        reference,
        times,
        actions,
        frame_phase=phases,
        anchor_residuals=anchor_residuals,
        anchor_mask=anchor_mask,
    )
    return apply_latent_anchors(state - velocity, anchor_residuals, anchor_mask)


def run_anchored_motion_training(
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    config: AnchoredMotionTrainingConfig | None = None,
    resume_checkpoint_path: Path | str | None = None,
    expected_resume_sha256: str | None = None,
    disk_guard: DiskGuard | None = None,
) -> AnchoredMotionTrainingResult:
    """Train the hard-anchored interpolation stage without copying corpus arrays."""

    runtime = _require_torch()
    experiment = config or AnchoredMotionTrainingConfig()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace anchored-motion output: {output}")
    if (resume_checkpoint_path is None) != (expected_resume_sha256 is None):
        raise ValueError("resume checkpoint and expected SHA-256 must be supplied together")
    device = runtime.device(experiment.device)
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if experiment.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 anchored-motion training requires CUDA")
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(18 * 1024**3, label="MUGEN anchored interpolation training")
    corpus = load_latent_motion_training_corpus(
        manifest_path, verify_hashes=True, array_loading="lazy"
    )
    output.mkdir(parents=True, exist_ok=False)
    with (output / "training-history.jsonl").open("x", encoding="utf-8", newline="\n") as history:
        return _train(
            runtime,
            corpus=corpus,
            output=output,
            history=history,
            config=experiment,
            device=device,
            resume_checkpoint_path=(
                Path(resume_checkpoint_path).resolve()
                if resume_checkpoint_path is not None
                else None
            ),
            expected_resume_sha256=expected_resume_sha256,
            disk_guard=guard,
        )


def _train(
    runtime: Any,
    *,
    corpus: LatentMotionTrainingCorpus,
    output: Path,
    history: Any,
    config: AnchoredMotionTrainingConfig,
    device: Any,
    resume_checkpoint_path: Path | None,
    expected_resume_sha256: str | None,
    disk_guard: DiskGuard,
) -> AnchoredMotionTrainingResult:
    runtime.manual_seed(config.seed)
    if device.type == "cuda":
        runtime.cuda.manual_seed_all(config.seed)
    model = AnchoredActionConditionedLatentMotionDiT(
        config.model,
        len(corpus.action_vocabulary),
        action_token_count=config.action_token_count,
        action_condition_scale=config.action_condition_scale,
    ).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = runtime.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    decoder = _load_frozen_decoder(runtime, corpus, device=device)
    sampler = runtime.Generator(device="cpu").manual_seed(config.seed + 1)
    noise_generator = runtime.Generator(device=device).manual_seed(config.seed + 2)
    start_step = 0
    lineage = None
    if resume_checkpoint_path is not None:
        assert expected_resume_sha256 is not None
        parent = _load_resume(
            runtime,
            resume_checkpoint_path,
            expected_sha256=expected_resume_sha256,
            corpus=corpus,
            config=config,
        )
        start_step = int(parent["step"])
        if start_step >= config.steps:
            raise AnchoredMotionTrainingError("resume step must be below cumulative steps")
        model.load_state_dict(parent["raw_model"], strict=True)
        ema.load_state_dict(parent["ema_model"], strict=True)
        optimizer.load_state_dict(parent["optimizer"])
        sampler.set_state(parent["rng_state"]["sampler"])
        noise_generator.set_state(parent["rng_state"]["noise"])
        runtime.set_rng_state(parent["rng_state"]["torch_cpu"])
        if device.type == "cuda":
            runtime.cuda.set_rng_state(parent["rng_state"]["cuda"], device=device)
        lineage = {
            "initialization": "exact_resume_with_optimizer_and_rng",
            "parent_checkpoint_path": str(resume_checkpoint_path),
            "parent_checkpoint_sha256": expected_resume_sha256,
            "parent_step": start_step,
        }
    train_bundles = build_keypose_action_bundles(corpus, corpus.train_indices)
    validation_bundles = build_keypose_action_bundles(corpus, corpus.validation_indices)[
        : config.validation_identities
    ]
    mean = runtime.tensor(corpus.channel_mean, device=device).view(1, 1, 8, 1, 1)
    std = runtime.tensor(corpus.channel_standard_deviation, device=device).view(1, 1, 8, 1, 1)
    dtype = runtime.bfloat16 if config.precision == "bfloat16" else runtime.float32
    autocast = config.precision == "bfloat16"
    missing_indices = (1, 2, 3, 5, 6)
    latest_validation = None
    model.train()
    for step_index in range(start_step, config.steps):
        step = step_index + 1
        learning_rate = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        totals = {"latent": 0.0, "pixel": 0.0, "contrast": 0.0, "motion": 0.0, "loss": 0.0}
        for _ in range(config.gradient_accumulation):
            bundle_index = int(runtime.randint(len(train_bundles), (1,), generator=sampler).item())
            selection = train_bundles[bundle_index]
            clean, reference, target_rgba, phases, actions, anchors, mask = _anchored_batch(
                runtime,
                corpus,
                selection,
                config=config,
                device=device,
                mean=mean,
                std=std,
            )
            shared_noise = runtime.randn(
                (1, *clean.shape[1:]), device=device, generator=noise_generator
            ).expand_as(clean)
            noisy = apply_latent_anchors(shared_noise, anchors, mask)
            times = runtime.ones((len(selection),), device=device)
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                predicted = model(
                    noisy,
                    reference,
                    times,
                    actions,
                    frame_phase=phases,
                    anchor_residuals=anchors,
                    anchor_mask=mask,
                )
                latent_loss = masked_velocity_mse(predicted, shared_noise - clean, mask)
                generated_residual = apply_latent_anchors(noisy - predicted, anchors, mask)
                generated_latent = (reference.unsqueeze(1) + generated_residual) * std + mean
                logits = decoder.decode_logits(generated_latent.reshape(-1, 8, 64, 64)).reshape(
                    len(selection), 8, 4, 128, 128
                )
                predicted_rgba = runtime.sigmoid(logits)
                pixel_loss = sprite_reconstruction_loss(
                    logits[:, missing_indices].reshape(-1, 4, 128, 128),
                    target_rgba[:, missing_indices].reshape(-1, 4, 128, 128),
                ).total
                contrast_loss = _pixel_action_bundle_loss(
                    runtime,
                    predicted_rgba=predicted_rgba[:, missing_indices],
                    target_rgba=target_rgba[:, missing_indices],
                )
                reference_latent = reference * std[:, 0] + mean[:, 0]
                reference_rgba = runtime.sigmoid(decoder.decode_logits(reference_latent))
                loop_target = target_rgba.clone()
                loop_target[:, 0] = reference_rgba
                loop_target[:, 7] = reference_rgba
                motion_loss = _target_directed_motion_floor_loss(
                    runtime,
                    predicted_rgba=predicted_rgba,
                    target_rgba=loop_target,
                    minimum_progress=config.minimum_target_motion_progress,
                )
                loss = (
                    config.latent_weight * latent_loss
                    + config.pixel_weight * pixel_loss
                    + config.pixel_action_contrast_weight * contrast_loss
                    + config.target_directed_motion_weight * motion_loss
                )
                scaled = loss / config.gradient_accumulation
            if not bool(runtime.isfinite(scaled)):
                raise RuntimeError(f"non-finite anchored-motion loss at step {step}")
            scaled.backward()
            totals["latent"] += float(latent_loss.detach().cpu())
            totals["pixel"] += float(pixel_loss.detach().cpu())
            totals["contrast"] += float(contrast_loss.detach().cpu())
            totals["motion"] += float(motion_loss.detach().cpu())
            totals["loss"] += float(loss.detach().cpu())
        gradient_norm = float(
            runtime.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            .detach()
            .cpu()
        )
        optimizer.step()
        _ema_update(runtime, ema, model, 0 if step <= config.warmup_steps else config.ema_decay)
        if step == 1 or step % config.validate_every == 0 or step == config.steps:
            latest_validation = _validate(
                runtime,
                corpus,
                validation_bundles,
                ema,
                decoder,
                config=config,
                device=device,
                dtype=dtype,
                autocast=autocast,
                mean=mean,
                std=std,
            )
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            row = {
                "gradient_norm_before_clip": gradient_norm,
                "latent_endpoint_loss": totals["latent"] / config.gradient_accumulation,
                "learning_rate": learning_rate,
                "loss": totals["loss"] / config.gradient_accumulation,
                "pixel_action_contrast_loss": totals["contrast"] / config.gradient_accumulation,
                "pixel_endpoint_loss": totals["pixel"] / config.gradient_accumulation,
                "step": step,
                "target_directed_motion_loss": totals["motion"] / config.gradient_accumulation,
                "validation": latest_validation
                if step == 1 or step % config.validate_every == 0
                else None,
            }
            history.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            history.flush()
            os.fsync(history.fileno())
        if step % config.checkpoint_every == 0 or step == config.steps:
            _write_checkpoint(
                runtime,
                output / f"training-step-{step:07d}.pt",
                corpus=corpus,
                config=config,
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                sampler=sampler,
                noise_generator=noise_generator,
                validation=latest_validation,
                lineage=lineage,
                disk_guard=disk_guard,
            )
    checkpoint = output / f"training-step-{config.steps:07d}.pt"
    report = {
        "action_vocabulary": list(corpus.action_vocabulary),
        "anchor_contract": {
            "anchor_frames": list(config.anchor_frame_indices),
            "canonical_middle_frame": config.canonical_middle_frame_index,
            "endpoint_frames_are_exact_reference": True,
            "loss_frames": [1, 2, 3, 5, 6],
        },
        "artifact_kind": "mugen_start_middle_start_anchored_motion_training_report",
        "checkpoint_file_sha256": _file_sha256(checkpoint),
        "checkpoint_path": str(checkpoint),
        "claim": "teacher-forced true middle anchors; predicted-keypose robustness not yet claimed",
        "config": asdict(config),
        "corpus": corpus.contract,
        "lineage": lineage,
        "runtime": _runtime_facts(runtime, device),
        "schema_version": 1,
        "step": config.steps,
        "validation": latest_validation,
    }
    payload = canonical_json_bytes(report)
    report_path = output / "training-report.json"
    _atomic_bytes(report_path, payload, disk_guard=disk_guard)
    return AnchoredMotionTrainingResult(
        output_directory=output,
        report_path=report_path,
        checkpoint_path=checkpoint,
        report_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _validate(
    runtime: Any,
    corpus: LatentMotionTrainingCorpus,
    bundles: tuple[tuple[int, ...], ...],
    model: Any,
    decoder: Any,
    *,
    config: AnchoredMotionTrainingConfig,
    device: Any,
    dtype: Any,
    autocast: bool,
    mean: Any,
    std: Any,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    generator = runtime.Generator(device=device).manual_seed(config.seed + 20_000)
    missing_indices = (1, 2, 3, 5, 6)
    with runtime.no_grad():
        for selection in bundles:
            clean, reference, target_rgba, phases, actions, anchors, mask = _anchored_batch(
                runtime,
                corpus,
                selection,
                config=config,
                device=device,
                mean=mean,
                std=std,
            )
            noise = runtime.randn(
                (1, *clean.shape[1:]), device=device, generator=generator
            ).expand_as(clean)
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                residual = sample_anchored_motion_residual(
                    runtime,
                    model,
                    noise=noise,
                    reference=reference,
                    actions=actions,
                    phases=phases,
                    anchor_residuals=anchors,
                    anchor_mask=mask,
                )
                latent = (reference.unsqueeze(1) + residual) * std + mean
                predicted_rgba = runtime.sigmoid(
                    decoder.decode_logits(latent.reshape(-1, 8, 64, 64))
                ).reshape(len(selection), 8, 4, 128, 128)
            metrics = _trajectory_bundle_metrics(
                runtime,
                predicted_rgba=predicted_rgba[:, missing_indices].float(),
                target_rgba=target_rgba[:, missing_indices],
            )
            metrics["anchor_latent_max_abs_error"] = (
                (residual - anchors).abs() * mask.view(len(selection), 8, 1, 1, 1)
            ).max()
            predicted_pm = _premultiplied(runtime, predicted_rgba.float())
            reference_latent = reference * std[:, 0] + mean[:, 0]
            reference_rgba = runtime.sigmoid(decoder.decode_logits(reference_latent))
            loop_target = target_rgba.clone()
            loop_target[:, 0] = reference_rgba
            loop_target[:, 7] = reference_rgba
            target_pm = _premultiplied(runtime, loop_target)
            generated_motion = (predicted_pm[:, 1:] - predicted_pm[:, :-1]).abs().mean()
            target_motion = (target_pm[:, 1:] - target_pm[:, :-1]).abs().mean()
            metrics["generated_temporal_magnitude"] = generated_motion
            metrics["target_temporal_magnitude"] = target_motion
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.cpu())
    model.train()
    count = len(bundles)
    output = {key: value / count for key, value in totals.items()}
    output["generated_to_target_action_separation_ratio"] = output[
        "generated_action_separation"
    ] / max(output["target_action_separation"], 1e-8)
    output["temporal_motion_ratio"] = output["generated_temporal_magnitude"] / max(
        output["target_temporal_magnitude"], 1e-8
    )
    return output


def _premultiplied(runtime: Any, rgba: Any) -> Any:
    alpha = rgba[:, :, 3:4]
    return runtime.cat((rgba[:, :, :3] * alpha, alpha), dim=2)


def _learning_rate(step: int, config: AnchoredMotionTrainingConfig) -> float:
    if step <= config.warmup_steps and config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
    return (
        config.minimum_learning_rate
        + (config.learning_rate - config.minimum_learning_rate) * cosine
    )


def _write_checkpoint(
    runtime: Any,
    path: Path,
    *,
    corpus: LatentMotionTrainingCorpus,
    config: AnchoredMotionTrainingConfig,
    step: int,
    model: Any,
    ema: Any,
    optimizer: Any,
    sampler: Any,
    noise_generator: Any,
    validation: dict[str, float] | None,
    lineage: dict[str, Any] | None,
    disk_guard: DiskGuard,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace anchored checkpoint: {path}")
    disk_guard.require_capacity(2 * 1024**3, label=f"anchored checkpoint step {step}")
    payload = {
        "action_vocabulary": list(corpus.action_vocabulary),
        "artifact_kind": "mugen_start_middle_start_anchored_motion_resume_checkpoint",
        "config": asdict(config),
        "corpus": corpus.contract,
        "ema_model": ema.state_dict(),
        "lineage": lineage,
        "optimizer": optimizer.state_dict(),
        "raw_model": model.state_dict(),
        "rng_state": {
            "cuda": runtime.cuda.get_rng_state(device=model.action_embedding.weight.device)
            if model.action_embedding.weight.device.type == "cuda"
            else None,
            "noise": noise_generator.get_state(),
            "sampler": sampler.get_state(),
            "torch_cpu": runtime.get_rng_state(),
        },
        "schema_version": 1,
        "step": step,
        "validation": validation,
    }
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        runtime.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_resume(
    runtime: Any,
    path: Path,
    *,
    expected_sha256: str,
    corpus: LatentMotionTrainingCorpus,
    config: AnchoredMotionTrainingConfig,
) -> dict[str, Any]:
    if _file_sha256(path) != expected_sha256:
        raise AnchoredMotionTrainingError("resume checkpoint SHA-256 mismatch")
    try:
        value = runtime.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise AnchoredMotionTrainingError("resume checkpoint failed safe load") from error
    if not isinstance(value, dict) or value.get("artifact_kind") != (
        "mugen_start_middle_start_anchored_motion_resume_checkpoint"
    ):
        raise AnchoredMotionTrainingError("resume checkpoint has the wrong artifact kind")
    if value.get("corpus") != corpus.contract:
        raise AnchoredMotionTrainingError("resume corpus contract differs")
    if value.get("action_vocabulary") != list(corpus.action_vocabulary):
        raise AnchoredMotionTrainingError("resume action vocabulary differs")
    parent_config = value.get("config")
    if not isinstance(parent_config, dict):
        raise AnchoredMotionTrainingError("resume config is missing")
    current = asdict(config)
    for key, parent_value in parent_config.items():
        if key != "steps" and current.get(key) != parent_value:
            raise AnchoredMotionTrainingError(f"resume config differs at {key!r}")
    return value


def _atomic_bytes(path: Path, payload: bytes, *, disk_guard: DiskGuard) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace anchored artifact: {path}")
    disk_guard.require_capacity(len(payload) + 1024**2, label=path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("anchored-motion training requires PyTorch") from _TORCH_IMPORT_ERROR
    return torch
