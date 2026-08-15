"""Fixed-middle key-pose training for the anchored MUGEN animation pipeline."""

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

from spritelab.latent_motion_train import (
    LatentMotionTrainingCorpus,
    _ema_update,
    _file_sha256,
    _load_frozen_decoder,
    _runtime_facts,
    build_matched_action_index,
    load_latent_motion_training_corpus,
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
KeyposePredictionMode = Literal["endpoint_flow", "direct_residual"]


class LatentKeyposeTrainingError(ValueError):
    """Raised when the key-pose training contract is invalid."""


@dataclass(frozen=True, slots=True)
class LatentKeyposeTrainingConfig:
    """Training contract for reference sprite + verb -> canonical middle pose."""

    keypose_frame_index: int = 4
    prediction_mode: KeyposePredictionMode = "endpoint_flow"
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
    action_token_count: int = 4
    action_condition_scale: float = 2.0
    steps: int = 30_000
    log_every: int = 25
    validate_every: int = 500
    checkpoint_every: int = 2_500
    validation_identities: int = 16
    seed: int = 20260830
    device: str = "cuda"
    precision: Precision = "bfloat16"
    model: LatentMotionDiTConfig = LatentMotionDiTConfig(
        latent_size=64,
        num_frames=1,
        latent_channels=8,
        patch_size=4,
        model_dim=384,
        depth=12,
        num_heads=6,
        condition_dim=384,
    )

    def __post_init__(self) -> None:
        for name in (
            "keypose_frame_index",
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
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not 0 <= self.keypose_frame_index < 8:
            raise ValueError("keypose_frame_index must be in [0,8)")
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
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0,steps)")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "latent_weight",
            "pixel_weight",
            "pixel_action_contrast_weight",
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
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if self.model.num_frames != 1:
            raise ValueError("key-pose model must contain exactly one frame token plane")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if self.prediction_mode not in {"endpoint_flow", "direct_residual"}:
            raise ValueError("prediction_mode must be endpoint_flow or direct_residual")


@dataclass(frozen=True, slots=True)
class LatentKeyposeTrainingResult:
    output_directory: Path
    report_path: Path
    checkpoint_path: Path
    report_sha256: str


def build_keypose_action_bundles(
    corpus: LatentMotionTrainingCorpus, indices: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """Return one complete, canonically ordered action bundle per identity."""

    index = build_matched_action_index(corpus.rows, indices)
    required = set(corpus.action_vocabulary)
    bundles = []
    for identity in sorted(index, key=str.encode):
        actions = index[identity]
        if set(actions) != required:
            continue
        bundles.append(tuple(actions[action] for action in corpus.action_vocabulary))
    if not bundles:
        raise LatentKeyposeTrainingError("split contains no complete action bundles")
    return tuple(bundles)


def _build_keypose_model(config: LatentKeyposeTrainingConfig, action_count: int) -> Any:
    from spritelab.latent_motion_train import _ActionConditionedMotionModel

    return _ActionConditionedMotionModel(
        config.model,
        action_count,
        conditioning_mode="expanded",
        action_token_count=config.action_token_count,
        action_condition_scale=config.action_condition_scale,
    )


def _keypose_batch(
    runtime: Any,
    corpus: LatentMotionTrainingCorpus,
    selection: tuple[int, ...],
    *,
    frame_index: int,
    device: Any,
    mean: Any,
    std: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    indices = list(selection)
    target_full = corpus.target_latents[indices].astype(np.float32)
    target = runtime.from_numpy(target_full[:, frame_index]).to(device)
    reference = runtime.from_numpy(corpus.reference_latents[indices].astype(np.float32)).to(device)
    target = (target - mean) / std
    reference = (reference - mean) / std
    rgba_full = corpus.target_rgba[indices]
    rgba = (
        runtime.from_numpy(rgba_full[:, frame_index])
        .to(device=device, dtype=runtime.float32)
        .permute(0, 3, 1, 2)
        .div(255)
    )
    phase = runtime.full((len(indices), 1), frame_index / 8, device=device)
    actions = runtime.tensor([corpus.rows[index].action_index for index in indices], device=device)
    return target, reference, rgba, phase, actions


def _keypose_bundle_metrics(runtime: Any, *, predicted_rgba: Any, target_rgba: Any) -> dict:
    """Measure exact action-pose fidelity and six-way target preference."""

    expected = tuple(target_rgba.shape)
    if len(expected) != 5 or expected[0] < 3 or expected[1:3] != (1, 4):
        raise ValueError("key-pose bundles must have shape [B>=3,1,4,H,W]")
    if tuple(predicted_rgba.shape) != expected:
        raise ValueError("key-pose bundles must share one shape")
    predicted_alpha = predicted_rgba[:, :, 3:4]
    target_alpha = target_rgba[:, :, 3:4]
    predicted_pm = runtime.cat((predicted_rgba[:, :, :3] * predicted_alpha, predicted_alpha), dim=2)
    target_pm = runtime.cat((target_rgba[:, :, :3] * target_alpha, target_alpha), dim=2)
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
    predicted_foreground = predicted_alpha >= 0.5
    target_foreground = target_alpha >= 0.5
    intersection = (predicted_foreground & target_foreground).sum()
    union = (predicted_foreground | target_foreground).sum()
    return {
        "alpha_iou_127": intersection / union.clamp_min(1),
        "correct_target_preference_rate": (own < wrong).to(runtime.float32).mean(),
        "generated_action_separation": generated_distances.mean(),
        "premultiplied_rgba_mae": runtime.nn.functional.l1_loss(predicted_pm, target_pm),
        "target_action_separation": target_distances.mean(),
    }


def _keypose_prediction_contract(
    runtime: Any,
    *,
    clean_residual: Any,
    noise: Any,
    prediction_mode: KeyposePredictionMode,
) -> tuple[Any, Any]:
    """Return model input and supervised prediction for one key-pose mode."""

    if tuple(noise.shape) != tuple(clean_residual.shape):
        raise ValueError("key-pose noise and clean residual must share one shape")
    if prediction_mode == "endpoint_flow":
        return noise, noise - clean_residual
    if prediction_mode == "direct_residual":
        return runtime.zeros_like(clean_residual), -clean_residual
    raise ValueError("prediction_mode must be endpoint_flow or direct_residual")


def _pixel_action_bundle_loss(runtime: Any, *, predicted_rgba: Any, target_rgba: Any) -> Any:
    predicted_alpha = predicted_rgba[:, :, 3:4]
    target_alpha = target_rgba[:, :, 3:4]
    predicted_pm = runtime.cat((predicted_rgba[:, :, :3] * predicted_alpha, predicted_alpha), dim=2)
    target_pm = runtime.cat((target_rgba[:, :, :3] * target_alpha, target_alpha), dim=2)
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


def run_latent_keypose_training(
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    config: LatentKeyposeTrainingConfig | None = None,
    resume_checkpoint_path: Path | str | None = None,
    expected_resume_sha256: str | None = None,
    disk_guard: DiskGuard | None = None,
) -> LatentKeyposeTrainingResult:
    """Train a no-clobber fixed-slot action-pose model on all six MUGEN verbs."""

    runtime = _require_torch()
    experiment = config or LatentKeyposeTrainingConfig()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace latent key-pose output: {output}")
    if (resume_checkpoint_path is None) != (expected_resume_sha256 is None):
        raise ValueError("resume checkpoint and expected SHA-256 must be supplied together")
    device = runtime.device(experiment.device)
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if experiment.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 key-pose training requires CUDA")
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(18 * 1024**3, label="MUGEN fixed-middle key-pose training")
    corpus = load_latent_motion_training_corpus(
        manifest_path, verify_hashes=True, array_loading="lazy"
    )
    output.mkdir(parents=True, exist_ok=False)
    history_path = output / "training-history.jsonl"
    with history_path.open("x", encoding="utf-8", newline="\n") as history:
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
    config: LatentKeyposeTrainingConfig,
    device: Any,
    resume_checkpoint_path: Path | None,
    expected_resume_sha256: str | None,
    disk_guard: DiskGuard,
) -> LatentKeyposeTrainingResult:
    runtime.manual_seed(config.seed)
    if device.type == "cuda":
        runtime.cuda.manual_seed_all(config.seed)
    model = _build_keypose_model(config, len(corpus.action_vocabulary)).to(device)
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
            raise LatentKeyposeTrainingError("resume step must be below cumulative steps")
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
    mean = runtime.tensor(corpus.channel_mean, device=device).view(1, 8, 1, 1)
    std = runtime.tensor(corpus.channel_standard_deviation, device=device).view(1, 8, 1, 1)
    dtype = runtime.bfloat16 if config.precision == "bfloat16" else runtime.float32
    autocast = config.precision == "bfloat16"
    latest_validation = None
    model.train()
    for step_index in range(start_step, config.steps):
        step = step_index + 1
        learning_rate = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        totals = {"latent": 0.0, "pixel": 0.0, "contrast": 0.0, "loss": 0.0}
        for _ in range(config.gradient_accumulation):
            bundle_index = int(runtime.randint(len(train_bundles), (1,), generator=sampler).item())
            selection = train_bundles[bundle_index]
            target, reference, target_rgba, phases, actions = _keypose_batch(
                runtime,
                corpus,
                selection,
                frame_index=config.keypose_frame_index,
                device=device,
                mean=mean,
                std=std,
            )
            clean = target - reference
            shared_noise = runtime.randn(
                (1, *clean.shape[1:]), device=device, generator=noise_generator
            ).expand_as(clean)
            model_input, prediction_target = _keypose_prediction_contract(
                runtime,
                clean_residual=clean,
                noise=shared_noise,
                prediction_mode=config.prediction_mode,
            )
            times = runtime.ones((len(selection),), device=device)
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                predicted = model(
                    model_input.unsqueeze(1),
                    reference,
                    times,
                    actions,
                    frame_phase=phases,
                )[:, 0]
                latent_loss = runtime.nn.functional.mse_loss(
                    predicted.float(), prediction_target.float()
                )
                generated_residual = model_input - predicted
                generated_latent = (reference + generated_residual) * std + mean
                logits = decoder.decode_logits(generated_latent)
                pixel_loss = sprite_reconstruction_loss(logits, target_rgba).total
                predicted_rgba = runtime.sigmoid(logits).reshape(len(selection), 1, 4, 128, 128)
                target_rgba_5d = target_rgba.reshape(len(selection), 1, 4, 128, 128)
                contrast_loss = _pixel_action_bundle_loss(
                    runtime,
                    predicted_rgba=predicted_rgba,
                    target_rgba=target_rgba_5d,
                )
                loss = (
                    config.latent_weight * latent_loss
                    + config.pixel_weight * pixel_loss
                    + config.pixel_action_contrast_weight * contrast_loss
                )
                scaled = loss / config.gradient_accumulation
            if not bool(runtime.isfinite(scaled)):
                raise RuntimeError(f"non-finite key-pose loss at step {step}")
            scaled.backward()
            totals["latent"] += float(latent_loss.detach().cpu())
            totals["pixel"] += float(pixel_loss.detach().cpu())
            totals["contrast"] += float(contrast_loss.detach().cpu())
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
        "artifact_kind": "mugen_fixed_middle_latent_keypose_training_report",
        "checkpoint_file_sha256": _file_sha256(checkpoint),
        "checkpoint_path": str(checkpoint),
        "claim": "reference sprite plus verb to fixed frame-4 action pose; no text-to-image claim",
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
    return LatentKeyposeTrainingResult(
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
    config: LatentKeyposeTrainingConfig,
    device: Any,
    dtype: Any,
    autocast: bool,
    mean: Any,
    std: Any,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    generator = runtime.Generator(device=device).manual_seed(config.seed + 20_000)
    with runtime.no_grad():
        for selection in bundles:
            target, reference, target_rgba, phases, actions = _keypose_batch(
                runtime,
                corpus,
                selection,
                frame_index=config.keypose_frame_index,
                device=device,
                mean=mean,
                std=std,
            )
            clean = target - reference
            noise = runtime.randn(
                (1, *clean.shape[1:]), device=device, generator=generator
            ).expand_as(clean)
            model_input, prediction_target = _keypose_prediction_contract(
                runtime,
                clean_residual=clean,
                noise=noise,
                prediction_mode=config.prediction_mode,
            )
            times = runtime.ones((len(selection),), device=device)
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                velocity = model(
                    model_input.unsqueeze(1),
                    reference,
                    times,
                    actions,
                    frame_phase=phases,
                )[:, 0]
                shifted_actions = runtime.roll(actions, 1)
                shifted_velocity = model(
                    model_input.unsqueeze(1),
                    reference,
                    times,
                    shifted_actions,
                    frame_phase=phases,
                )[:, 0]
                generated = (reference + model_input - velocity) * std + mean
                shifted = (reference + model_input - shifted_velocity) * std + mean
                decoded = runtime.sigmoid(decoder.decode_logits(generated)).reshape(
                    len(selection), 1, 4, 128, 128
                )
                shifted_decoded = runtime.sigmoid(decoder.decode_logits(shifted)).reshape(
                    len(selection), 1, 4, 128, 128
                )
            target_5d = target_rgba.reshape(len(selection), 1, 4, 128, 128)
            metrics = _keypose_bundle_metrics(
                runtime, predicted_rgba=decoded.float(), target_rgba=target_5d
            )
            predicted_pm = _premultiplied(runtime, decoded.float())
            shifted_pm = _premultiplied(runtime, shifted_decoded.float())
            target_pm = _premultiplied(runtime, target_5d)
            replacement = runtime.roll(target_pm, 1, dims=0)
            before = (predicted_pm - replacement).abs().mean(dim=(1, 2, 3, 4))
            after = (shifted_pm - replacement).abs().mean(dim=(1, 2, 3, 4))
            metrics["action_swap_replacement_movement_rate"] = (
                (after < before).to(runtime.float32).mean()
            )
            metrics["action_token_endpoint_loss_delta"] = runtime.nn.functional.mse_loss(
                shifted_velocity.float(), prediction_target.float()
            ) - runtime.nn.functional.mse_loss(velocity.float(), prediction_target.float())
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value.cpu())
    model.train()
    count = len(bundles)
    output = {key: value / count for key, value in totals.items()}
    output["generated_to_target_action_separation_ratio"] = output[
        "generated_action_separation"
    ] / max(output["target_action_separation"], 1e-8)
    return output


def _premultiplied(runtime: Any, rgba: Any) -> Any:
    alpha = rgba[:, :, 3:4]
    return runtime.cat((rgba[:, :, :3] * alpha, alpha), dim=2)


def _learning_rate(step: int, config: LatentKeyposeTrainingConfig) -> float:
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
    config: LatentKeyposeTrainingConfig,
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
        raise FileExistsError(f"Refusing to replace key-pose checkpoint: {path}")
    disk_guard.require_capacity(2 * 1024**3, label=f"key-pose checkpoint step {step}")
    payload = {
        "action_vocabulary": list(corpus.action_vocabulary),
        "artifact_kind": "mugen_fixed_middle_latent_keypose_resume_checkpoint",
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
    config: LatentKeyposeTrainingConfig,
) -> dict[str, Any]:
    if _file_sha256(path) != expected_sha256:
        raise LatentKeyposeTrainingError("resume checkpoint SHA-256 mismatch")
    try:
        value = runtime.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise LatentKeyposeTrainingError("resume checkpoint failed safe load") from error
    if not isinstance(value, dict) or value.get("artifact_kind") != (
        "mugen_fixed_middle_latent_keypose_resume_checkpoint"
    ):
        raise LatentKeyposeTrainingError("resume checkpoint has the wrong artifact kind")
    if value.get("corpus") != corpus.contract:
        raise LatentKeyposeTrainingError("resume corpus contract differs")
    if value.get("action_vocabulary") != list(corpus.action_vocabulary):
        raise LatentKeyposeTrainingError("resume action vocabulary differs")
    parent_config = value.get("config")
    if not isinstance(parent_config, dict):
        raise LatentKeyposeTrainingError("resume config is missing")
    current = asdict(config)
    for key, parent_value in parent_config.items():
        if key != "steps" and current.get(key) != parent_value:
            raise LatentKeyposeTrainingError(f"resume config differs at {key!r}")
    return value


def _atomic_bytes(path: Path, payload: bytes, *, disk_guard: DiskGuard) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace key-pose artifact: {path}")
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
        raise RuntimeError("latent key-pose training requires PyTorch") from _TORCH_IMPORT_ERROR
    return torch
