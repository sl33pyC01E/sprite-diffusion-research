"""Bounded reference-conditioned latent-motion overfit for MUGEN research."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.evaluation import compare_matched_sequences
from spritelab.models.latent_motion_dit import (
    LatentMotionDiTConfig,
    ReferenceConditionedLatentMotionDiT,
)
from spritelab.models.sprite_autoencoder import (
    SpriteAutoencoderConfig,
    SpriteRGBAAutoencoder,
    sprite_reconstruction_loss,
)
from spritelab.mugen_motion_dataset import _array_sha256
from spritelab.previews import export_rgba_clip_preview
from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional runtime boundary
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class LatentMotionOverfitError(RuntimeError):
    """Raised when an input, runtime, or output contract differs."""


@dataclass(frozen=True, slots=True)
class LatentMotionOverfitConfig:
    steps: int = 1_500
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    base_weight: float = 1.0
    endpoint_weight: float = 1.0
    pixel_endpoint_weight: float = 0.0
    gradient_clip_norm: float = 1.0
    log_every: int = 25
    seed: int = 20260824
    device: str = "cuda"
    precision: str = "bfloat16"
    patch_size: int = 4
    model_dim: int = 128
    depth: int = 4
    num_heads: int = 4

    def __post_init__(self) -> None:
        for name in ("steps", "log_every", "seed", "patch_size", "model_dim", "depth", "num_heads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "weight_decay",
            "base_weight",
            "endpoint_weight",
            "pixel_endpoint_weight",
            "gradient_clip_norm",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")
        if self.base_weight == 0 and self.endpoint_weight == 0 and self.pixel_endpoint_weight == 0:
            raise ValueError("at least one training objective weight must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")


@dataclass(frozen=True, slots=True)
class LoadedMotionOverfitCorpus:
    plan_path: Path
    plan_file_sha256: str
    latent_manifest_path: Path
    latent_manifest_file_sha256: str
    autoencoder_checkpoint_path: Path
    autoencoder_checkpoint_sha256: str
    identity_id: str
    verbs: tuple[str, ...]
    records: tuple[dict[str, Any], ...]
    target_latents: np.ndarray
    reference_latents: np.ndarray
    target_rgba: np.ndarray
    phases: np.ndarray
    durations_ms: tuple[tuple[float, ...], ...]
    loop_modes: tuple[str, ...]
    channel_mean: np.ndarray
    channel_std: np.ndarray
    autoencoder_architecture: dict[str, Any]


def load_motion_overfit_corpus(
    plan_path: Path | str,
    *,
    identity_id: str,
    verbs: Sequence[str],
) -> LoadedMotionOverfitCorpus:
    """Load one exact sequence per requested verb with complete hash verification."""

    plan_file = Path(plan_path).resolve()
    plan_bytes = plan_file.read_bytes()
    plan = _json_object(plan_bytes, "motion plan")
    if plan.get("artifact_kind") != "mugen_reference_conditioned_latent_motion_plan":
        raise LatentMotionOverfitError("motion plan has the wrong artifact kind")
    records = plan.get("records")
    counts = plan.get("counts")
    if (
        not isinstance(records, list)
        or not isinstance(counts, dict)
        or counts.get("sequences") != len(records)
    ):
        raise LatentMotionOverfitError("motion plan sequence count differs")
    normalized_verbs = tuple(_unique_nonempty(verbs, "verbs"))
    chosen = []
    for verb in normalized_verbs:
        candidates = sorted(
            (
                row
                for row in records
                if isinstance(row, dict)
                and row.get("identity_id") == identity_id
                and isinstance(row.get("conditioning"), dict)
                and row["conditioning"].get("verb") == verb
            ),
            key=lambda row: str(row.get("sequence_id")).encode(),
        )
        if not candidates:
            raise LatentMotionOverfitError(f"identity lacks requested verb: {verb}")
        chosen.append(candidates[0])
    source = plan.get("source")
    if not isinstance(source, dict):
        raise LatentMotionOverfitError("motion plan source is absent")
    latent_source = source.get("latent_manifest")
    if not isinstance(latent_source, dict):
        raise LatentMotionOverfitError("latent manifest source is absent")
    latent_manifest_path = Path(_text(latent_source, "path")).resolve()
    latent_manifest_bytes = latent_manifest_path.read_bytes()
    latent_manifest_sha256 = hashlib.sha256(latent_manifest_bytes).hexdigest()
    if latent_manifest_sha256 != latent_source.get("file_sha256"):
        raise LatentMotionOverfitError("latent manifest hash differs")
    latent_manifest = _json_object(latent_manifest_bytes, "latent manifest")
    normalization = latent_manifest.get("normalization")
    codec = latent_manifest.get("codec")
    if not isinstance(normalization, dict) or not isinstance(codec, dict):
        raise LatentMotionOverfitError("latent normalization/codec is absent")
    mean = np.asarray(normalization.get("channel_mean"), dtype=np.float32)
    std = np.asarray(normalization.get("channel_standard_deviation"), dtype=np.float32)
    if (
        mean.shape != (8,)
        or std.shape != (8,)
        or not np.isfinite(mean).all()
        or not (np.isfinite(std).all() and (std > 0).all())
    ):
        raise LatentMotionOverfitError("latent normalization is invalid")
    checkpoint_path = Path(_text(codec, "checkpoint_path")).resolve()
    checkpoint_sha256 = _file_sha256(checkpoint_path)
    if checkpoint_sha256 != codec.get("checkpoint_file_sha256"):
        raise LatentMotionOverfitError("autoencoder checkpoint hash differs")
    architecture = codec.get("architecture")
    if not isinstance(architecture, dict):
        raise LatentMotionOverfitError("autoencoder architecture is absent")

    latent_root = latent_manifest_path.parent
    materialization_source = source.get("materialization")
    if not isinstance(materialization_source, dict):
        raise LatentMotionOverfitError("materialization source is absent")
    materialization_path = Path(_text(materialization_source, "path")).resolve()
    if _file_sha256(materialization_path) != materialization_source.get("file_sha256"):
        raise LatentMotionOverfitError("materialization hash differs")
    materialization_root = materialization_path.parent
    target_latents = []
    reference_latents = []
    target_rgba = []
    phases = []
    durations = []
    loop_modes = []
    for row in chosen:
        target = _dict(row, "target")
        target_latent = _load_latent(latent_root, _dict(target, "latent"))
        reference = _dict(row, "reference")
        reference_record = _dict(reference, "latent")
        reference_full = _load_latent(latent_root, reference_record)
        frame_index = reference.get("frame_index")
        if not isinstance(frame_index, int) or not 0 <= frame_index < 8:
            raise LatentMotionOverfitError("reference frame index is invalid")
        reference_frame = np.ascontiguousarray(reference_full[frame_index])
        if _array_sha256(reference_frame) != reference_record.get("frame_array_content_sha256"):
            raise LatentMotionOverfitError("reference latent frame hash differs")
        source_pixels = _dict(target, "source_pixels")
        rgba = _load_rgba(materialization_root, source_pixels)
        phase = np.asarray(target.get("phase"), dtype=np.float32)
        duration = target.get("duration_ms")
        if phase.shape != (8,) or not np.isfinite(phase).all():
            raise LatentMotionOverfitError("target phase is invalid")
        if (
            not isinstance(duration, list)
            or len(duration) != 8
            or not all(isinstance(value, (int, float)) and value > 0 for value in duration)
        ):
            raise LatentMotionOverfitError("target duration is invalid")
        target_latents.append(target_latent.astype(np.float32))
        reference_latents.append(reference_frame.astype(np.float32))
        target_rgba.append(rgba)
        phases.append(phase)
        durations.append(tuple(float(value) for value in duration))
        loop_modes.append(str(target.get("loop_mode")))
    return LoadedMotionOverfitCorpus(
        plan_path=plan_file,
        plan_file_sha256=hashlib.sha256(plan_bytes).hexdigest(),
        latent_manifest_path=latent_manifest_path,
        latent_manifest_file_sha256=latent_manifest_sha256,
        autoencoder_checkpoint_path=checkpoint_path,
        autoencoder_checkpoint_sha256=checkpoint_sha256,
        identity_id=identity_id,
        verbs=normalized_verbs,
        records=tuple(chosen),
        target_latents=np.stack(target_latents),
        reference_latents=np.stack(reference_latents),
        target_rgba=np.stack(target_rgba),
        phases=np.stack(phases),
        durations_ms=tuple(durations),
        loop_modes=tuple(loop_modes),
        channel_mean=mean,
        channel_std=std,
        autoencoder_architecture=architecture,
    )


if torch is not None and nn is not None:

    class _VerbConditioner(nn.Module):
        def __init__(self, count: int, width: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(count, width)
            self.norm = nn.LayerNorm(width)

        def forward(self, indices: torch.Tensor) -> torch.Tensor:
            return self.norm(self.embedding(indices)).unsqueeze(1)


def run_motion_overfit(
    plan_path: Path | str,
    output_directory: Path | str,
    *,
    identity_id: str,
    verbs: Sequence[str],
    config: LatentMotionOverfitConfig | None = None,
    initial_checkpoint_path: Path | str | None = None,
    expected_initial_checkpoint_sha256: str | None = None,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Train and publish one no-clobber latent-motion causal overfit."""

    runtime = _require_torch()
    experiment = config or LatentMotionOverfitConfig()
    corpus = load_motion_overfit_corpus(plan_path, identity_id=identity_id, verbs=verbs)
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace latent-motion experiment: {output}")
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(1024**3, label="latent-motion overfit")
    if experiment.device == "cuda" and not runtime.cuda.is_available():
        raise LatentMotionOverfitError("CUDA was requested but is unavailable")
    device = runtime.device(experiment.device)
    runtime.manual_seed(experiment.seed)
    if device.type == "cuda":
        runtime.cuda.manual_seed_all(experiment.seed)
    model_config = LatentMotionDiTConfig(
        latent_size=64,
        num_frames=8,
        latent_channels=8,
        patch_size=experiment.patch_size,
        model_dim=experiment.model_dim,
        depth=experiment.depth,
        num_heads=experiment.num_heads,
        condition_dim=experiment.model_dim,
    )
    model = ReferenceConditionedLatentMotionDiT(model_config).to(device)
    conditioner = _VerbConditioner(len(corpus.verbs), experiment.model_dim).to(device)
    initialization = _load_initial_checkpoint(
        runtime,
        initial_checkpoint_path,
        expected_sha256=expected_initial_checkpoint_sha256,
        model=model,
        conditioner=conditioner,
        corpus=corpus,
        model_config=model_config,
    )
    optimizer = runtime.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()),
        lr=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
    )
    mean = runtime.from_numpy(corpus.channel_mean).to(device).view(1, 1, 8, 1, 1)
    std = runtime.from_numpy(corpus.channel_std).to(device).view(1, 1, 8, 1, 1)
    target = runtime.from_numpy(corpus.target_latents).to(device)
    reference = runtime.from_numpy(corpus.reference_latents).to(device)
    target_normalized = (target - mean) / std
    reference_normalized = (reference - mean[:, 0]) / std[:, 0]
    clean = target_normalized - reference_normalized.unsqueeze(1)
    phases = runtime.from_numpy(corpus.phases).to(device)
    target_rgba_unit = (
        runtime.from_numpy(corpus.target_rgba)
        .to(device=device, dtype=runtime.float32)
        .permute(0, 1, 4, 2, 3)
        .reshape(-1, 4, 128, 128)
        .div(255)
    )
    pixel_decoder = None
    if experiment.pixel_endpoint_weight > 0:
        pixel_decoder = _load_frozen_decoder(runtime, corpus, device=device)
    action_indices = runtime.arange(len(corpus.verbs), device=device)
    generator = runtime.Generator(device=device).manual_seed(experiment.seed + 1)
    endpoint_generator = runtime.Generator(device=device).manual_seed(experiment.seed + 2)
    fixed_noise = (
        runtime.randn((1, *clean.shape[1:]), device=device, generator=endpoint_generator)
        .expand_as(clean)
        .clone()
    )

    def endpoint_prediction(condition_tokens: torch.Tensor) -> torch.Tensor:
        return model(
            fixed_noise,
            reference_normalized,
            runtime.ones(len(corpus.verbs), device=device),
            condition_tokens,
            frame_phase=phases,
        )

    def endpoint_loss(condition_tokens: torch.Tensor) -> torch.Tensor:
        return runtime.nn.functional.mse_loss(
            endpoint_prediction(condition_tokens), fixed_noise - clean
        )

    model.train()
    conditioner.train()
    with runtime.no_grad():
        initial_endpoint_loss = float(endpoint_loss(conditioner(action_indices)).cpu())
    history = []
    use_autocast = device.type == "cuda" and experiment.precision == "bfloat16"
    for step in range(1, experiment.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        autocast = runtime.autocast(
            device_type=device.type,
            dtype=runtime.bfloat16,
            enabled=use_autocast,
        )
        with autocast:
            tokens = conditioner(action_indices)
            if experiment.base_weight > 0:
                noise = runtime.randn(clean.shape, device=device, generator=generator)
                timesteps = runtime.rand((clean.shape[0],), device=device, generator=generator)
                expanded = timesteps.view(clean.shape[0], 1, 1, 1, 1)
                noisy = (1 - expanded) * clean + expanded * noise
                target_velocity = noise - clean
                predicted = model(
                    noisy,
                    reference_normalized,
                    timesteps,
                    tokens,
                    frame_phase=phases,
                )
                base_loss = runtime.nn.functional.mse_loss(predicted, target_velocity)
            else:
                base_loss = runtime.zeros((), device=device)
            matched_prediction = endpoint_prediction(tokens)
            matched_loss = runtime.nn.functional.mse_loss(matched_prediction, fixed_noise - clean)
            pixel_endpoint_loss = runtime.zeros((), device=device)
            if pixel_decoder is not None:
                generated_residual = fixed_noise - matched_prediction
                generated_normalized = reference_normalized.unsqueeze(1) + generated_residual
                generated_latent = generated_normalized * std + mean
                generated_logits = pixel_decoder.decode_logits(
                    generated_latent.reshape(-1, 8, 64, 64)
                )
                pixel_endpoint_loss = sprite_reconstruction_loss(
                    generated_logits, target_rgba_unit
                ).total
            loss = (
                experiment.base_weight * base_loss
                + experiment.endpoint_weight * matched_loss
                + experiment.pixel_endpoint_weight * pixel_endpoint_loss
            )
        if not bool(runtime.isfinite(loss)):
            raise LatentMotionOverfitError(f"training became non-finite at step {step}")
        loss.backward()
        gradient_norm = runtime.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(conditioner.parameters()),
            experiment.gradient_clip_norm,
        )
        optimizer.step()
        if step == 1 or step % experiment.log_every == 0 or step == experiment.steps:
            history_row = {
                "base_loss": float(base_loss.detach().cpu()),
                "endpoint_loss": float(matched_loss.detach().cpu()),
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "pixel_endpoint_loss": float(pixel_endpoint_loss.detach().cpu()),
                "step": step,
            }
            history.append(history_row)
            print(json.dumps(history_row, sort_keys=True), flush=True)

    model.eval()
    conditioner.eval()
    with runtime.no_grad():
        correct_tokens = conditioner(action_indices)
        final_endpoint_loss = float(endpoint_loss(correct_tokens).cpu())
        permuted_tokens = conditioner(runtime.roll(action_indices, shifts=1))
        permuted_endpoint_loss = float(endpoint_loss(permuted_tokens).cpu())
        predicted_velocity = model(
            fixed_noise,
            reference_normalized,
            runtime.ones(len(corpus.verbs), device=device),
            correct_tokens,
            frame_phase=phases,
        )
        generated_residual = fixed_noise - predicted_velocity
        generated_normalized = reference_normalized.unsqueeze(1) + generated_residual
        generated_latent = generated_normalized * std + mean
        generated_rgba = _decode_latents(runtime, corpus, generated_latent, device=device)

    stage_parent = output.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=stage_parent))
    try:
        preview_dir = stage / "previews"
        metrics = []
        preview_rows = []
        for index, verb in enumerate(corpus.verbs):
            target_array = corpus.target_rgba[index]
            generated_array = generated_rgba[index]
            target_path = stage / f"target-{index:02d}-{verb}.npy"
            generated_path = stage / f"generated-{index:02d}-{verb}.npy"
            _save_npy(target_path, target_array)
            _save_npy(generated_path, generated_array)
            target_preview = export_rgba_clip_preview(
                target_array,
                preview_dir,
                artifact_stem=f"{index:02d}-{verb}-target",
                duration_ms=corpus.durations_ms[index],
                loop_mode=_preview_loop_mode(corpus.loop_modes[index]),
                integer_scale=2,
                preserve_frame_slots=True,
                disk_guard=guard,
            )
            generated_preview = export_rgba_clip_preview(
                generated_array,
                preview_dir,
                artifact_stem=f"{index:02d}-{verb}-generated",
                duration_ms=corpus.durations_ms[index],
                loop_mode=_preview_loop_mode(corpus.loop_modes[index]),
                integer_scale=2,
                preserve_frame_slots=True,
                disk_guard=guard,
            )
            comparison = compare_matched_sequences(
                _images(generated_array),
                _images(target_array),
                loop_mode=_preview_loop_mode(corpus.loop_modes[index]),
            )
            metrics.append({"verb": verb, **asdict(comparison)})
            preview_rows.append(
                {
                    "generated_animated_sha256": generated_preview.animated_png_sha256,
                    "generated_sheet_sha256": generated_preview.contact_sheet_sha256,
                    "sequence_id": corpus.records[index]["sequence_id"],
                    "target_animated_sha256": target_preview.animated_png_sha256,
                    "target_sheet_sha256": target_preview.contact_sheet_sha256,
                    "verb": verb,
                }
            )
        checkpoint_path = stage / "checkpoint.pt"
        runtime.save(
            {
                "artifact_kind": "mugen_reference_latent_motion_overfit_checkpoint",
                "conditioner": conditioner.state_dict(),
                "config": asdict(experiment),
                "identity_id": corpus.identity_id,
                "initialization": initialization,
                "model": model.state_dict(),
                "model_config": asdict(model_config),
                "normalization": {
                    "channel_mean": corpus.channel_mean.tolist(),
                    "channel_standard_deviation": corpus.channel_std.tolist(),
                },
                "plan_file_sha256": corpus.plan_file_sha256,
                "step": experiment.steps,
                "verbs": list(corpus.verbs),
            },
            checkpoint_path,
        )
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        report = {
            "artifact_kind": "mugen_reference_latent_motion_overfit_report",
            "autoencoder_checkpoint_sha256": corpus.autoencoder_checkpoint_sha256,
            "checkpoint": {"file_sha256": checkpoint_sha256, "path": "checkpoint.pt"},
            "claim": (
                "same-identity bounded latent-motion memorization and action-token sensitivity only"
            ),
            "config": asdict(experiment),
            "endpoint": {
                "action_permuted_final_loss": permuted_endpoint_loss,
                "action_token_loss_delta": permuted_endpoint_loss - final_endpoint_loss,
                "final_loss": final_endpoint_loss,
                "initial_loss": initial_endpoint_loss,
                "shared_noise": True,
            },
            "history": history,
            "identity_id": corpus.identity_id,
            "initialization": initialization,
            "latent_manifest_file_sha256": corpus.latent_manifest_file_sha256,
            "metrics": metrics,
            "model_config": asdict(model_config),
            "plan_file_sha256": corpus.plan_file_sha256,
            "previews": preview_rows,
            "schema_version": 1,
            "verbs": list(corpus.verbs),
        }
        report_bytes = canonical_json_bytes(report)
        report_path = stage / "report.json"
        report_path.write_bytes(report_bytes)
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output / "report.json", hashlib.sha256(report_bytes).hexdigest()


def _decode_latents(
    runtime: Any, corpus: LoadedMotionOverfitCorpus, latent: Any, *, device: Any
) -> np.ndarray:
    decoder = _load_frozen_decoder(runtime, corpus, device=device)
    flat = latent.reshape(-1, 8, 64, 64)
    chunks = []
    with runtime.no_grad():
        for start in range(0, flat.shape[0], 8):
            decoded = decoder.decode(flat[start : start + 8]).clamp(0, 1)
            chunks.append(decoded.mul(255).round().to(runtime.uint8).cpu())
    rgba = runtime.cat(chunks).numpy().transpose(0, 2, 3, 1)
    return np.ascontiguousarray(rgba.reshape(len(corpus.verbs), 8, 128, 128, 4))


def _load_frozen_decoder(runtime: Any, corpus: LoadedMotionOverfitCorpus, *, device: Any) -> Any:
    checkpoint = runtime.load(
        corpus.autoencoder_checkpoint_path, map_location="cpu", weights_only=True
    )
    architecture = dict(corpus.autoencoder_architecture)
    if isinstance(architecture.get("channel_multipliers"), list):
        architecture["channel_multipliers"] = tuple(architecture["channel_multipliers"])
    decoder = SpriteRGBAAutoencoder(SpriteAutoencoderConfig(**architecture)).to(device).eval()
    decoder.load_state_dict(checkpoint["ema"], strict=True)
    decoder.requires_grad_(False)
    return decoder


def _load_initial_checkpoint(
    runtime: Any,
    path: Path | str | None,
    *,
    expected_sha256: str | None,
    model: Any,
    conditioner: Any,
    corpus: LoadedMotionOverfitCorpus,
    model_config: LatentMotionDiTConfig,
) -> dict[str, Any] | None:
    if (path is None) != (expected_sha256 is None):
        raise LatentMotionOverfitError(
            "initial checkpoint path and expected SHA-256 must be provided together"
        )
    if path is None:
        return None
    checkpoint_path = Path(path).resolve()
    actual_sha256 = _file_sha256(checkpoint_path)
    if actual_sha256 != expected_sha256:
        raise LatentMotionOverfitError("initial checkpoint hash differs")
    payload = runtime.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("artifact_kind") != (
        "mugen_reference_latent_motion_overfit_checkpoint"
    ):
        raise LatentMotionOverfitError("initial checkpoint has the wrong artifact kind")
    expected_contract = {
        "identity_id": corpus.identity_id,
        "model_config": asdict(model_config),
        "normalization": {
            "channel_mean": corpus.channel_mean.tolist(),
            "channel_standard_deviation": corpus.channel_std.tolist(),
        },
        "plan_file_sha256": corpus.plan_file_sha256,
        "verbs": list(corpus.verbs),
    }
    for key, expected in expected_contract.items():
        if payload.get(key) != expected:
            raise LatentMotionOverfitError(f"initial checkpoint {key} differs")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise LatentMotionOverfitError("initial checkpoint step is invalid")
    try:
        model.load_state_dict(payload["model"], strict=True)
        conditioner.load_state_dict(payload["conditioner"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise LatentMotionOverfitError("initial checkpoint state differs") from error
    return {
        "checkpoint_file_sha256": actual_sha256,
        "optimizer_restored": False,
        "parent_step": step,
        "policy": "weights_only_fresh_optimizer_refinement",
    }


def _load_latent(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = _resolve_under(root, _text(record, "relative_path"))
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record.get("file_sha256"):
        raise LatentMotionOverfitError("latent file hash differs")
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if array.dtype != np.float16 or array.shape != (8, 8, 64, 64):
        raise LatentMotionOverfitError("latent geometry differs")
    if _array_sha256(array) != record.get("array_content_sha256"):
        raise LatentMotionOverfitError("latent array hash differs")
    return np.ascontiguousarray(array)


def _load_rgba(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = _resolve_under(root, _text(record, "relative_path"))
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record.get("file_sha256"):
        raise LatentMotionOverfitError("RGBA file hash differs")
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    if array.dtype != np.uint8 or array.shape != (8, 128, 128, 4):
        raise LatentMotionOverfitError("RGBA geometry differs")
    if _array_sha256(array) != record.get("array_content_sha256"):
        raise LatentMotionOverfitError("RGBA array hash differs")
    return np.ascontiguousarray(array)


def _images(array: np.ndarray) -> tuple[Any, ...]:
    from PIL import Image

    return tuple(Image.fromarray(frame) for frame in array)


def _save_npy(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    path.write_bytes(buffer.getvalue())


def _preview_loop_mode(value: str) -> str:
    return value if value in {"loop", "one_shot", "ping_pong"} else "one_shot"


def _unique_nonempty(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must contain non-empty text")
        normalized = value.strip()
        if normalized in result:
            raise ValueError(f"{label} must not contain duplicates")
        result.append(normalized)
    if len(result) < 2:
        raise ValueError(f"{label} must contain at least two values")
    return tuple(result)


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LatentMotionOverfitError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise LatentMotionOverfitError(f"{label} must be an object")
    return value


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise LatentMotionOverfitError(f"{key} must be non-empty text")
    return result


def _dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise LatentMotionOverfitError(f"{key} must be an object")
    return result


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise LatentMotionOverfitError("artifact path escapes its root")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise LatentMotionOverfitError(
            "latent-motion overfit requires PyTorch"
        ) from _TORCH_IMPORT_ERROR
    return torch
