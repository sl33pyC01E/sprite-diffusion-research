"""Identity/action-balanced reconstruction training for sprite RGBA latents."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.broad_train import PreparedBroadCorpus, PreparedBroadRow, prepare_broad_corpus
from spritelab.models.sprite_autoencoder import (
    SpriteAutoencoderConfig,
    SpriteReconstructionLossConfig,
    SpriteRGBAAutoencoder,
    sprite_reconstruction_loss,
)
from spritelab.storage import DiskGuard

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch-free environment
    torch = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingAutoencoderTrainingTorchError(RuntimeError):
    """Raised when training is requested without PyTorch."""


@dataclass(frozen=True, slots=True)
class SpriteAutoencoderTrainingConfig:
    architecture: SpriteAutoencoderConfig = field(default_factory=SpriteAutoencoderConfig)
    reconstruction: SpriteReconstructionLossConfig = field(
        default_factory=SpriteReconstructionLossConfig
    )
    batch_size: int = 32
    gradient_accumulation: int = 1
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    horizontal_flip_probability: float = 0.5
    steps: int = 20_000
    log_every: int = 25
    validate_every: int = 500
    checkpoint_every: int = 5_000
    validation_frames: int = 512
    seed: int = 20260822
    device: str = "cuda"
    precision: str = "bfloat16"

    def __post_init__(self) -> None:
        for name in (
            "batch_size",
            "gradient_accumulation",
            "steps",
            "log_every",
            "validate_every",
            "checkpoint_every",
            "validation_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise ValueError("warmup_steps must be an integer")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if (
            not math.isfinite(self.horizontal_flip_probability)
            or not 0 <= self.horizontal_flip_probability <= 1
        ):
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty")


@dataclass(frozen=True, slots=True)
class AutoencoderTrainingResult:
    output_directory: Path
    report_path: Path
    checkpoint_path: Path
    report_sha256: str
    checkpoint_sha256: str


FrameIndex = dict[str, dict[str, tuple[int, ...]]]


def identity_action_frame_index(rows: tuple[PreparedBroadRow, ...]) -> FrameIndex:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        grouped[row.identity_id][row.action].append(index)
    return {
        identity: {
            action: tuple(indices)
            for action, indices in sorted(actions.items(), key=lambda item: item[0].encode())
        }
        for identity, actions in sorted(grouped.items(), key=lambda item: item[0].encode())
    }


def sample_balanced_frames(
    rows: tuple[PreparedBroadRow, ...],
    index: FrameIndex,
    *,
    batch_size: int,
    generator: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    """Draw identity, action, clip, then authored-duration-resampled frame uniformly."""

    if not index:
        raise ValueError("frame index cannot be empty")
    identities = tuple(index)
    output: list[tuple[int, int]] = []
    for _ in range(batch_size):
        identity = identities[int(generator.integers(len(identities)))]
        actions = tuple(index[identity])
        action = actions[int(generator.integers(len(actions)))]
        candidates = index[identity][action]
        row_index = candidates[int(generator.integers(len(candidates)))]
        frame_index = int(generator.integers(rows[row_index].rgba.shape[0]))
        output.append((row_index, frame_index))
    return tuple(output)


def validation_frame_plan(
    rows: tuple[PreparedBroadRow, ...], *, maximum_frames: int
) -> tuple[tuple[int, int], ...]:
    """Select stable action/identity/frame coverage without random validation drift."""

    if maximum_frames <= 0:
        raise ValueError("maximum_frames must be positive")
    candidates: list[tuple[bytes, int, int]] = []
    for row_index, row in enumerate(rows):
        for frame_index in range(row.rgba.shape[0]):
            digest = hashlib.sha256(
                f"{row.identity_id}\0{row.action}\0{row.sequence_id}\0{frame_index}".encode()
            ).digest()
            candidates.append((digest, row_index, frame_index))
    candidates.sort(key=lambda value: value[0])
    return tuple((row, frame) for _, row, frame in candidates[:maximum_frames])


def run_autoencoder_training(
    materialization_manifest: Path | str,
    output_directory: Path | str,
    *,
    config: SpriteAutoencoderTrainingConfig | None = None,
    disk_guard: DiskGuard | None = None,
) -> AutoencoderTrainingResult:
    runtime = _require_torch()
    if config is None:
        config = SpriteAutoencoderTrainingConfig()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace autoencoder training output: {output}")
    corpus = prepare_broad_corpus(
        materialization_manifest,
        target_size=config.architecture.image_size,
        target_frames=8,
    )
    if not corpus.train or not corpus.validation:
        raise ValueError("autoencoder training requires train and validation rows")
    if disk_guard is not None:
        disk_guard.require_capacity(512 * 1024**2, label="sprite autoencoder training artifacts")
    device = runtime.device(config.device)
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise RuntimeError("CUDA autoencoder training requested but unavailable")
    runtime.manual_seed(config.seed)
    if device.type == "cuda":
        runtime.cuda.manual_seed_all(config.seed)
    sampler = np.random.default_rng(config.seed)
    model = SpriteRGBAAutoencoder(config.architecture).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = runtime.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_index = identity_action_frame_index(corpus.train)
    validation_plan = validation_frame_plan(
        corpus.validation, maximum_frames=config.validation_frames
    )
    dtype = runtime.bfloat16 if config.precision == "bfloat16" else runtime.float32
    autocast = config.precision == "bfloat16"

    output.mkdir(parents=True)
    history_path = output / "training-history.jsonl"
    with history_path.open("x", encoding="utf-8", newline="\n") as history:
        model.train()
        for step_index in range(config.steps):
            step = step_index + 1
            learning_rate = _learning_rate(step, config)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            accumulated = defaultdict(float)
            for _ in range(config.gradient_accumulation):
                selection = sample_balanced_frames(
                    corpus.train,
                    train_index,
                    batch_size=config.batch_size,
                    generator=sampler,
                )
                target = _frame_tensor(runtime, corpus.train, selection, device=device)
                if config.horizontal_flip_probability:
                    mask = runtime.from_numpy(
                        sampler.random(target.shape[0]) < config.horizontal_flip_probability
                    ).to(device=device)
                    target = runtime.where(mask[:, None, None, None], target.flip(-1), target)
                with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                    losses = sprite_reconstruction_loss(
                        model(target), target, config=config.reconstruction
                    )
                    loss = losses.total / config.gradient_accumulation
                if not bool(runtime.isfinite(loss)):
                    raise RuntimeError(f"non-finite autoencoder loss at step {step}")
                loss.backward()
                for name in losses.__dataclass_fields__:
                    accumulated[name] += float(getattr(losses, name).detach().cpu())
            gradient_norm = float(
                runtime.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
                .detach()
                .cpu()
            )
            optimizer.step()
            _ema_update(runtime, ema, model, config.ema_decay)
            validation = None
            if step == 1 or step % config.validate_every == 0 or step == config.steps:
                validation = _validate(
                    runtime,
                    ema,
                    corpus,
                    validation_plan,
                    config=config,
                    device=device,
                    dtype=dtype,
                    autocast=autocast,
                )
            if step == 1 or step % config.log_every == 0 or step == config.steps:
                record = {
                    "gradient_norm_before_clip": gradient_norm,
                    "learning_rate": learning_rate,
                    "loss": {
                        name: value / config.gradient_accumulation
                        for name, value in accumulated.items()
                    },
                    "step": step,
                    "validation": validation,
                }
                history.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                history.flush()
                os.fsync(history.fileno())
            if step % config.checkpoint_every == 0 or step == config.steps:
                _write_checkpoint(
                    runtime,
                    output / f"training-step-{step:07d}.pt",
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    corpus=corpus,
                    config=config,
                    step=step,
                    sampler=sampler,
                    disk_guard=disk_guard,
                )

    final_validation = _validate(
        runtime,
        ema,
        corpus,
        validation_plan,
        config=config,
        device=device,
        dtype=dtype,
        autocast=autocast,
        output=output / "validation-reconstructions",
    )
    checkpoint = output / f"training-step-{config.steps:07d}.pt"
    checkpoint_sha = _file_sha256(checkpoint)
    report = {
        "artifact_kind": "sprite_rgba_autoencoder_training",
        "checkpoint": {"file_sha256": checkpoint_sha, "path": checkpoint.name},
        "config": asdict(config),
        "corpus": _corpus_record(corpus),
        "final_validation": final_validation,
        "latent_contract": {
            "channels": config.architecture.latent_channels,
            "continuous": True,
            "downsample_factor": config.architecture.downsample_factor,
            "height": config.architecture.latent_size,
            "width": config.architecture.latent_size,
        },
        "schema_version": 1,
    }
    report_payload = _canonical_json(report)
    report_path = output / "training-report.json"
    report_path.write_bytes(report_payload)
    return AutoencoderTrainingResult(
        output,
        report_path,
        checkpoint,
        hashlib.sha256(report_payload).hexdigest(),
        checkpoint_sha,
    )


def _frame_tensor(
    runtime: Any,
    rows: tuple[PreparedBroadRow, ...],
    selection: tuple[tuple[int, int], ...],
    *,
    device: Any,
) -> Any:
    rgba = np.stack([rows[row].rgba[frame] for row, frame in selection], axis=0)
    value = np.ascontiguousarray(rgba.transpose(0, 3, 1, 2), dtype=np.float32) / 255
    return runtime.from_numpy(value).to(device=device)


def _validate(
    runtime: Any,
    model: Any,
    corpus: PreparedBroadCorpus,
    plan: tuple[tuple[int, int], ...],
    *,
    config: SpriteAutoencoderTrainingConfig,
    device: Any,
    dtype: Any,
    autocast: bool,
    output: Path | None = None,
) -> dict[str, Any]:
    totals = defaultdict(float)
    count = 0
    saved: list[dict[str, Any]] = []
    if output is not None:
        output.mkdir(parents=True)
    model.eval()
    with runtime.no_grad():
        for start in range(0, len(plan), config.batch_size):
            selection = plan[start : start + config.batch_size]
            target = _frame_tensor(runtime, corpus.validation, selection, device=device)
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                logits = model(target)
                losses = sprite_reconstruction_loss(logits, target, config=config.reconstruction)
            batch = target.shape[0]
            for name in losses.__dataclass_fields__:
                totals[name] += float(getattr(losses, name).detach().cpu()) * batch
            predicted = runtime.sigmoid(logits).float()
            target_alpha = target[:, 3] >= 0.5
            predicted_alpha = predicted[:, 3] >= 0.5
            totals["alpha_intersection"] += int((target_alpha & predicted_alpha).sum().cpu())
            totals["alpha_union"] += int((target_alpha | predicted_alpha).sum().cpu())
            count += batch
            if output is not None and len(saved) < 64:
                predicted_u8 = (
                    predicted.clamp(0, 1).mul(255).round().to(runtime.uint8).cpu().numpy()
                )
                target_u8 = target.mul(255).round().to(runtime.uint8).cpu().numpy()
                for local, (row_index, frame_index) in enumerate(selection):
                    if len(saved) >= 64:
                        break
                    row = corpus.validation[row_index]
                    for role, array in (
                        ("target", target_u8[local]),
                        ("reconstructed", predicted_u8[local]),
                    ):
                        path = (
                            output / f"{len(saved):04d}-{row.sequence_id}-{frame_index}-{role}.npy"
                        )
                        np.save(
                            path, np.ascontiguousarray(array.transpose(1, 2, 0)), allow_pickle=False
                        )
                    saved.append(
                        {
                            "action": row.action,
                            "description": row.request.description,
                            "frame_index": frame_index,
                            "sequence_id": row.sequence_id,
                        }
                    )
    model.train()
    return {
        "alpha_iou_127": (
            totals["alpha_intersection"] / totals["alpha_union"] if totals["alpha_union"] else 1.0
        ),
        "frames": count,
        "loss": {
            name: totals[name] / count
            for name in (
                "total",
                "premultiplied_rgba_l1",
                "alpha_bce",
                "visible_rgb_l1",
                "edge_l1",
            )
        },
        "saved_reconstruction_frames": saved,
    }


def _write_checkpoint(
    runtime: Any,
    path: Path,
    *,
    model: Any,
    ema: Any,
    optimizer: Any,
    corpus: PreparedBroadCorpus,
    config: SpriteAutoencoderTrainingConfig,
    step: int,
    sampler: np.random.Generator,
    disk_guard: DiskGuard | None,
) -> None:
    payload = {
        "artifact_kind": "sprite_rgba_autoencoder_resume_checkpoint",
        "config": asdict(config),
        "corpus": _corpus_record(corpus),
        "ema": ema.state_dict(),
        "model": model.state_dict(),
        "numpy_sampler_state_json": json.dumps(
            sampler.bit_generator.state, separators=(",", ":"), sort_keys=True
        ),
        "optimizer": optimizer.state_dict(),
        "runtime": {"torch": runtime.__version__},
        "schema_version": 1,
        "step": step,
        "torch_cpu_rng_state": runtime.get_rng_state(),
    }
    buffer = io.BytesIO()
    runtime.save(payload, buffer)
    data = buffer.getvalue()
    if disk_guard is not None:
        disk_guard.require_capacity(len(data), label="sprite autoencoder checkpoint")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _learning_rate(step: int, config: SpriteAutoencoderTrainingConfig) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return (
        config.minimum_learning_rate
        + (config.learning_rate - config.minimum_learning_rate) * cosine
    )


def _ema_update(runtime: Any, ema: Any, model: Any, decay: float) -> None:
    with runtime.no_grad():
        for target, source in zip(ema.parameters(), model.parameters(), strict=True):
            target.mul_(decay).add_(source, alpha=1 - decay)
        for target, source in zip(ema.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def _corpus_record(corpus: PreparedBroadCorpus) -> dict[str, Any]:
    return {
        "corpus_sha256": corpus.corpus_sha256,
        "materialization_manifest_sha256": corpus.materialization_manifest_sha256,
        "source_snapshot_canonical_sha256": corpus.source_snapshot_canonical_sha256,
        "source_snapshot_manifest_sha256": corpus.source_snapshot_manifest_sha256,
        "train_rows": len(corpus.train),
        "validation_rows": len(corpus.validation),
    }


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_torch() -> Any:
    if torch is None:
        raise MissingAutoencoderTrainingTorchError(
            "sprite autoencoder training requires PyTorch"
        ) from _TORCH_IMPORT_ERROR
    return torch
