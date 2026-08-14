"""Resumable identity-disjoint minibatch training for materialized sprite corpora."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from spritelab.models import ConditioningSchema, PixelDiTConfig
from spritelab.models.conditioning import (
    EncodedConditionBatch,
    SpriteConditionEncoder,
    encode_generation_conditions,
)
from spritelab.models.flow import sample_rectified_flow_batch
from spritelab.models.pixeldit import FactorizedSpriteDiT
from spritelab.models.semantic_conditioning import SemanticSpriteConditionEncoder
from spritelab.semantic_text import SemanticEmbeddingTable, load_semantic_embedding_table
from spritelab.storage import DiskGuard
from spritelab.training_data import (
    MaterializedTrainingClip,
    load_materialized_training_clips,
    model_to_rgba_uint8,
    rgba_uint8_to_model,
)

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised in the torch-free venv
    torch = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None

Precision = Literal["float32", "bfloat16"]


class MissingBroadTrainingTorchError(RuntimeError):
    """Raised when broad training is requested without PyTorch."""


class BroadTrainingContractError(ValueError):
    """Raised when an immutable resume or corpus contract does not match."""


@dataclass(frozen=True, slots=True)
class BroadTrainingConfig:
    """Explicit first-stage generalization experiment configuration."""

    target_size: int = 128
    target_frames: int = 8
    patch_size: int = 8
    model_dim: int = 192
    depth: int = 6
    num_heads: int = 6
    condition_dim: int = 192
    max_text_bytes: int = 48
    batch_size: int = 2
    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    foreground_weight: float = 2.0
    alpha_channel_weight: float = 4.0
    endpoint_weight: float = 1.0
    horizontal_flip_probability: float = 0.0
    ema_decay: float = 0.999
    gradient_clip_norm: float = 1.0
    steps: int = 10_000
    log_every: int = 25
    validate_every: int = 250
    checkpoint_every: int = 1_000
    semantic_embedding_table: str | None = None
    seed: int = 20260817
    device: str = "cuda"
    precision: Precision = "bfloat16"

    def __post_init__(self) -> None:
        positive_ints = (
            "target_size",
            "target_frames",
            "patch_size",
            "model_dim",
            "depth",
            "num_heads",
            "condition_dim",
            "max_text_bytes",
            "batch_size",
            "gradient_accumulation",
            "steps",
            "log_every",
            "validate_every",
            "checkpoint_every",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise ValueError("warmup_steps must be an integer")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0, steps)")
        finite_nonnegative = (
            "minimum_learning_rate",
            "weight_decay",
            "foreground_weight",
            "alpha_channel_weight",
            "endpoint_weight",
        )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        for name in finite_nonnegative:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(self.horizontal_flip_probability)
            or not 0 <= self.horizontal_flip_probability <= 1
        ):
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        if self.target_size % self.patch_size:
            raise ValueError("target_size must be divisible by patch_size")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError(f"unsupported precision: {self.precision!r}")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be non-empty")
        if self.semantic_embedding_table is not None and (
            not isinstance(self.semantic_embedding_table, str)
            or not self.semantic_embedding_table.strip()
        ):
            raise ValueError("semantic_embedding_table must be non-empty text or None")


@dataclass(frozen=True, slots=True)
class PreparedBroadRow:
    sequence_id: str
    identity_id: str
    action: str
    split: str
    request: Any
    rgba: np.ndarray
    frame_phases: tuple[float, ...]
    source_size: tuple[int, int]
    source_file_sha256: str
    normalized_array_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedBroadCorpus:
    train: tuple[PreparedBroadRow, ...]
    validation: tuple[PreparedBroadRow, ...]
    materialization_manifest_sha256: str
    source_snapshot_canonical_sha256: str
    source_snapshot_manifest_sha256: str
    corpus_sha256: str
    spatial_transform: str


@dataclass(frozen=True, slots=True)
class BroadTrainingResult:
    output_directory: Path
    report_path: Path
    training_checkpoint_path: Path
    inference_checkpoint_path: Path
    report_sha256: str


def export_broad_ema_inference_checkpoint(
    training_checkpoint_path: Path | str,
    output_path: Path | str,
    *,
    expected_training_checkpoint_sha256: str,
    disk_guard: DiskGuard | None = None,
) -> str:
    """Export one periodic EMA state through the standard safe inference envelope."""

    runtime = _require_torch()
    source = Path(training_checkpoint_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace EMA inference checkpoint: {output}")
    actual_sha256 = _sha256_file(source)
    if actual_sha256 != expected_training_checkpoint_sha256:
        raise BroadTrainingContractError(
            "training checkpoint SHA-256 mismatch: "
            f"expected {expected_training_checkpoint_sha256}, got {actual_sha256}"
        )
    try:
        payload = runtime.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise BroadTrainingContractError("training checkpoint failed safe load") from error
    if not isinstance(payload, dict) or payload.get("artifact_kind") != (
        "broad_training_resume_checkpoint"
    ):
        raise BroadTrainingContractError("source is not a broad-training resume checkpoint")
    broad_config = payload.get("broad_config")
    step = payload.get("step")
    if not isinstance(broad_config, dict) or not isinstance(step, int) or step <= 0:
        raise BroadTrainingContractError("training checkpoint config/step is invalid")
    inference_config = _inference_config(broad_config, step=step)
    semantic = payload.get("semantic_embedding_table")
    exported = {
        "artifact_kind": (
            "broad_training_semantic_ema_inference_checkpoint"
            if semantic is not None
            else "broad_training_ema_inference_checkpoint"
        ),
        "broad_config": broad_config,
        "condition_encoder": payload["ema_condition_encoder"],
        "config": inference_config,
        "corpus_sha256": payload["corpus_sha256"],
        "denoiser": payload["ema_denoiser"],
        "materialization_manifest_sha256": payload["materialization_manifest_sha256"],
        "model_config": payload["model_config"],
        "runtime": payload["runtime"],
        "semantic_embedding_table": semantic,
        "source_training_checkpoint_sha256": actual_sha256,
        "step": step,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(runtime, output, exported, disk_guard=disk_guard)
    return _sha256_file(output)


def prepare_broad_corpus(
    manifest_path: Path | str,
    *,
    target_size: int,
    target_frames: int,
    usage: Literal["conditional_generation", "autoencoder"] = "conditional_generation",
) -> PreparedBroadCorpus:
    """Verify train/validation clips and normalize them without filtering identities."""

    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")
    if isinstance(target_frames, bool) or not isinstance(target_frames, int) or target_frames <= 0:
        raise ValueError("target_frames must be a positive integer")
    if usage not in {"conditional_generation", "autoencoder"}:
        raise ValueError("usage must be 'conditional_generation' or 'autoencoder'")
    _require_manifest_usage(manifest_path, usage)
    train_clips = load_materialized_training_clips(
        manifest_path,
        split="train",
        target_frames=target_frames,
    )
    validation_clips = load_materialized_training_clips(
        manifest_path,
        split="validation",
        target_frames=target_frames,
    )
    train = tuple(_prepare_row(clip, target_size=target_size) for clip in train_clips)
    validation = tuple(_prepare_row(clip, target_size=target_size) for clip in validation_clips)
    train_ids = {row.identity_id for row in train}
    validation_ids = {row.identity_id for row in validation}
    overlap = train_ids.intersection(validation_ids)
    if overlap:
        raise BroadTrainingContractError(
            f"train and validation identities overlap: {sorted(overlap)!r}"
        )
    first = train_clips[0]
    all_clips = (*train_clips, *validation_clips)
    contracts = {
        (
            clip.materialization_manifest_sha256,
            clip.source_snapshot_canonical_sha256,
            clip.source_snapshot_manifest_sha256,
        )
        for clip in all_clips
    }
    if len(contracts) != 1:
        raise BroadTrainingContractError("clips do not share one materialization contract")
    rows = [_row_contract(row) for row in (*train, *validation)]
    corpus_payload = {
        "materialization_manifest_sha256": first.materialization_manifest_sha256,
        "rows": rows,
        "source_snapshot_canonical_sha256": first.source_snapshot_canonical_sha256,
        "source_snapshot_manifest_sha256": first.source_snapshot_manifest_sha256,
        "spatial_transform": "left_aligned_floor_index_nearest_rgba_v1",
        "target_frames": target_frames,
        "target_size": target_size,
    }
    corpus_sha256 = hashlib.sha256(_canonical_json_bytes(corpus_payload)).hexdigest()
    return PreparedBroadCorpus(
        train=train,
        validation=validation,
        materialization_manifest_sha256=first.materialization_manifest_sha256,
        source_snapshot_canonical_sha256=first.source_snapshot_canonical_sha256,
        source_snapshot_manifest_sha256=first.source_snapshot_manifest_sha256,
        corpus_sha256=corpus_sha256,
        spatial_transform="left_aligned_floor_index_nearest_rgba_v1",
    )


def _require_manifest_usage(
    manifest_path: Path | str,
    usage: Literal["conditional_generation", "autoencoder"],
) -> None:
    path = Path(manifest_path).resolve()
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BroadTrainingContractError(
            f"cannot inspect materialization eligibility: {path}"
        ) from error
    if not isinstance(value, dict):
        raise BroadTrainingContractError("materialization eligibility root must be an object")
    eligibility = value.get("model_eligibility")
    if eligibility is None:
        return
    if not isinstance(eligibility, dict):
        raise BroadTrainingContractError("materialization model_eligibility must be an object")
    eligibility_key = (
        "autoencoder_reconstruction" if usage == "autoencoder" else "conditional_generation"
    )
    if eligibility.get(eligibility_key) is not True:
        reason = eligibility.get("reason")
        raise BroadTrainingContractError(
            f"materialization is not eligible for {usage}: {reason or 'no reason recorded'}"
        )


def resize_rgba_nearest(rgba: np.ndarray, target_size: int) -> np.ndarray:
    """Resize ``[T,H,W,4]`` using explicit left-aligned floor-index nearest."""

    if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8:
        raise TypeError("rgba must be a uint8 NumPy array")
    if rgba.ndim != 4 or rgba.shape[-1] != 4:
        raise ValueError(f"rgba must have shape [T,H,W,4]; got {rgba.shape!r}")
    if isinstance(target_size, bool) or not isinstance(target_size, int) or target_size <= 0:
        raise ValueError("target_size must be a positive integer")
    height, width = rgba.shape[1:3]
    y = np.floor(np.arange(target_size, dtype=np.float64) * height / target_size).astype(np.int64)
    x = np.floor(np.arange(target_size, dtype=np.float64) * width / target_size).astype(np.int64)
    resized = rgba[:, y[:, None], x[None, :], :]
    return np.ascontiguousarray(resized, dtype=np.uint8)


def identity_action_index(
    rows: tuple[PreparedBroadRow, ...],
) -> dict[str, dict[str, tuple[int, ...]]]:
    """Return a stable hierarchical index for identity/action-balanced sampling."""

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


def sample_balanced_indices(
    index: dict[str, dict[str, tuple[int, ...]]],
    *,
    batch_size: int,
    generator: Any,
) -> tuple[int, ...]:
    """Draw identity, then action, then sequence uniformly with replacement."""

    runtime = _require_torch()
    identities = tuple(index)
    if not identities:
        raise ValueError("identity/action index cannot be empty")
    output: list[int] = []
    for _ in range(batch_size):
        identity = identities[
            int(runtime.randint(len(identities), (1,), generator=generator).item())
        ]
        actions = tuple(index[identity])
        action = actions[int(runtime.randint(len(actions), (1,), generator=generator).item())]
        candidates = index[identity][action]
        output.append(
            candidates[int(runtime.randint(len(candidates), (1,), generator=generator).item())]
        )
    return tuple(output)


def run_broad_training(
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    config: BroadTrainingConfig | None = None,
    resume_checkpoint_path: Path | str | None = None,
    expected_resume_sha256: str | None = None,
    disk_guard: DiskGuard | None = None,
) -> BroadTrainingResult:
    """Train or resume into a new no-clobber directory with periodic checkpoints."""

    runtime = _require_torch()
    experiment = config or BroadTrainingConfig()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace broad-training output: {output}")
    if (resume_checkpoint_path is None) != (expected_resume_sha256 is None):
        raise ValueError("resume checkpoint and expected SHA-256 must be supplied together")
    device = runtime.device(experiment.device)
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if experiment.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 broad training currently requires CUDA")
    if disk_guard is not None:
        disk_guard.require_capacity(label="broad training output")

    corpus = prepare_broad_corpus(
        manifest_path,
        target_size=experiment.target_size,
        target_frames=experiment.target_frames,
    )
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "training-history.jsonl"
    log_handle = log_path.open("x", encoding="utf-8", newline="\n")
    try:
        result = _train(
            runtime,
            corpus=corpus,
            manifest=Path(manifest_path).resolve(),
            output=output,
            log_handle=log_handle,
            config=experiment,
            device=device,
            resume_checkpoint_path=(
                Path(resume_checkpoint_path).resolve()
                if resume_checkpoint_path is not None
                else None
            ),
            expected_resume_sha256=expected_resume_sha256,
            disk_guard=disk_guard,
        )
    except BaseException:
        log_handle.flush()
        os.fsync(log_handle.fileno())
        raise
    finally:
        log_handle.close()
    return result


def _train(
    runtime: Any,
    *,
    corpus: PreparedBroadCorpus,
    manifest: Path,
    output: Path,
    log_handle: Any,
    config: BroadTrainingConfig,
    device: Any,
    resume_checkpoint_path: Path | None,
    expected_resume_sha256: str | None,
    disk_guard: DiskGuard | None,
) -> BroadTrainingResult:
    runtime.manual_seed(config.seed)
    if device.type == "cuda":
        runtime.cuda.manual_seed_all(config.seed)
    schema = replace(ConditioningSchema(), phase_bins=config.target_frames)
    model_config = PixelDiTConfig(
        height=config.target_size,
        width=config.target_size,
        num_frames=config.target_frames,
        patch_size=config.patch_size,
        model_dim=config.model_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        condition_dim=config.condition_dim,
        conditioning=schema,
    )
    denoiser = FactorizedSpriteDiT(model_config).to(device)
    semantic_table = _load_training_semantic_table(config, corpus)
    if semantic_table is None:
        encoder = SpriteConditionEncoder(
            schema,
            condition_dim=config.condition_dim,
            max_text_bytes=config.max_text_bytes,
        ).to(device)
    else:
        encoder = SemanticSpriteConditionEncoder(
            schema,
            condition_dim=config.condition_dim,
            semantic_dim=semantic_table.descriptor.embedding_dim,
            max_text_bytes=config.max_text_bytes,
        ).to(device)
    ema_denoiser = copy.deepcopy(denoiser).eval().requires_grad_(False)
    ema_encoder = copy.deepcopy(encoder).eval().requires_grad_(False)
    optimizer = runtime.optim.AdamW(
        (*denoiser.parameters(), *encoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sampler_generator = runtime.Generator(device="cpu").manual_seed(config.seed + 1)
    flow_generator = runtime.Generator(device=device).manual_seed(config.seed + 2)
    endpoint_generator = runtime.Generator(device=device).manual_seed(config.seed + 3)
    start_step = 0
    parent: dict[str, Any] | None = None
    if resume_checkpoint_path is not None:
        assert expected_resume_sha256 is not None
        parent = _load_resume_checkpoint(
            runtime,
            resume_checkpoint_path,
            expected_sha256=expected_resume_sha256,
            corpus=corpus,
            config=config,
            model_config=model_config,
        )
        start_step = int(parent["step"])
        if start_step >= config.steps:
            raise BroadTrainingContractError("resume step must be below requested cumulative steps")
        denoiser.load_state_dict(parent["raw_denoiser"], strict=True)
        encoder.load_state_dict(parent["raw_condition_encoder"], strict=True)
        ema_denoiser.load_state_dict(parent["ema_denoiser"], strict=True)
        ema_encoder.load_state_dict(parent["ema_condition_encoder"], strict=True)
        optimizer.load_state_dict(parent["optimizer"])
        sampler_generator.set_state(parent["rng_state"]["sampler"])
        flow_generator.set_state(parent["rng_state"]["flow"])
        endpoint_generator.set_state(parent["rng_state"]["endpoint"])
        runtime.set_rng_state(parent["rng_state"]["torch_cpu"])
        if device.type == "cuda":
            runtime.cuda.set_rng_state(parent["rng_state"]["cuda"], device=device)

    train_encoded = encode_generation_conditions(
        [row.request for row in corpus.train], schema, max_text_bytes=config.max_text_bytes
    )
    validation_encoded = encode_generation_conditions(
        [row.request for row in corpus.validation],
        schema,
        max_text_bytes=config.max_text_bytes,
    )
    train_semantic = _semantic_rows(semantic_table, corpus.train)
    validation_semantic = _semantic_rows(semantic_table, corpus.validation)
    sampler_index = identity_action_index(corpus.train)
    dtype = runtime.bfloat16 if config.precision == "bfloat16" else runtime.float32
    autocast_enabled = config.precision == "bfloat16"

    denoiser.train()
    encoder.train()
    for step_index in range(start_step, config.steps):
        step = step_index + 1
        learning_rate = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_base = 0.0
        accumulated_endpoint = 0.0
        for _ in range(config.gradient_accumulation):
            indices = sample_balanced_indices(
                sampler_index,
                batch_size=config.batch_size,
                generator=sampler_generator,
            )
            clean, phases = _tensor_batch(runtime, corpus.train, indices, device=device)
            if config.horizontal_flip_probability > 0:
                flip_mask = (
                    runtime.rand((clean.shape[0],), generator=sampler_generator, device="cpu")
                    < config.horizontal_flip_probability
                )
                if bool(flip_mask.any()):
                    flip_mask = flip_mask.to(device=device)
                    clean = runtime.where(
                        flip_mask[:, None, None, None, None],
                        clean.flip(dims=(-1,)),
                        clean,
                    )
            encoded = _slice_encoded(train_encoded, indices)
            semantic = _semantic_tensor(runtime, train_semantic, indices, device=device)
            context, context_mask = _encode_context(encoder, encoded, semantic)
            flow = sample_rectified_flow_batch(clean, generator=flow_generator)
            endpoint_flow = sample_rectified_flow_batch(
                clean,
                timesteps=runtime.ones((clean.shape[0],), device=device, dtype=clean.dtype),
                generator=endpoint_generator,
            )
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                predicted = denoiser(
                    flow.noisy,
                    flow.timesteps,
                    context,
                    conditioning_mask=context_mask,
                    frame_phase=phases,
                )
                endpoint_predicted = denoiser(
                    endpoint_flow.noisy,
                    endpoint_flow.timesteps,
                    context,
                    conditioning_mask=context_mask,
                    frame_phase=phases,
                )
            base_loss = _weighted_flow_loss(runtime, predicted, flow.target_velocity, clean, config)
            endpoint_loss = _weighted_flow_loss(
                runtime,
                endpoint_predicted,
                endpoint_flow.target_velocity,
                clean,
                config,
            )
            loss = (base_loss + config.endpoint_weight * endpoint_loss) / (
                config.gradient_accumulation
            )
            if not bool(runtime.isfinite(loss)):
                raise RuntimeError(f"non-finite training loss at step {step}")
            loss.backward()
            accumulated_base += float(base_loss.detach().cpu())
            accumulated_endpoint += float(endpoint_loss.detach().cpu())
        gradient_norm = float(
            runtime.nn.utils.clip_grad_norm_(
                (*denoiser.parameters(), *encoder.parameters()),
                config.gradient_clip_norm,
            )
            .detach()
            .cpu()
        )
        optimizer.step()
        _ema_update(runtime, ema_denoiser, denoiser, config.ema_decay)
        _ema_update(runtime, ema_encoder, encoder, config.ema_decay)

        should_validate = step == 1 or step % config.validate_every == 0 or step == config.steps
        validation = (
            _validate(
                runtime,
                rows=corpus.validation,
                encoded=validation_encoded,
                denoiser=ema_denoiser,
                encoder=ema_encoder,
                config=config,
                device=device,
                dtype=dtype,
                autocast_enabled=autocast_enabled,
                semantic_vectors=validation_semantic,
            )
            if should_validate
            else None
        )
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            row = {
                "base_loss": accumulated_base / config.gradient_accumulation,
                "endpoint_loss": accumulated_endpoint / config.gradient_accumulation,
                "gradient_norm_before_clip": gradient_norm,
                "learning_rate": learning_rate,
                "step": step,
                "validation": validation,
            }
            log_handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            log_handle.flush()
            os.fsync(log_handle.fileno())
        if step % config.checkpoint_every == 0 or step == config.steps:
            _write_training_checkpoint(
                runtime,
                output / f"training-step-{step:07d}.pt",
                corpus=corpus,
                config=config,
                model_config=model_config,
                step=step,
                denoiser=denoiser,
                encoder=encoder,
                ema_denoiser=ema_denoiser,
                ema_encoder=ema_encoder,
                optimizer=optimizer,
                sampler_generator=sampler_generator,
                flow_generator=flow_generator,
                endpoint_generator=endpoint_generator,
                device=device,
                disk_guard=disk_guard,
            )

    final_validation = _validate(
        runtime,
        rows=corpus.validation,
        encoded=validation_encoded,
        denoiser=ema_denoiser,
        encoder=ema_encoder,
        config=config,
        device=device,
        dtype=dtype,
        autocast_enabled=autocast_enabled,
        semantic_vectors=validation_semantic,
        generate=True,
        output=output / "validation-samples",
    )
    training_checkpoint = output / f"training-step-{config.steps:07d}.pt"
    inference_checkpoint = output / "checkpoint-ema.pt"
    _write_inference_checkpoint(
        runtime,
        inference_checkpoint,
        corpus=corpus,
        config=config,
        model_config=model_config,
        step=config.steps,
        denoiser=ema_denoiser,
        encoder=ema_encoder,
        device=device,
        disk_guard=disk_guard,
    )
    lineage = None
    if resume_checkpoint_path is not None:
        lineage = {
            "parent_checkpoint_path": str(resume_checkpoint_path),
            "parent_checkpoint_sha256": expected_resume_sha256,
            "parent_step": start_step,
        }
    report = {
        "artifact_kind": "identity_disjoint_minibatch_sprite_training",
        "claim": (
            "validation identities are disjoint from training identities; results measure "
            "held-out reconstruction for the indexed TMWA distribution, not open-vocabulary "
            "text-to-sprite generation"
        ),
        "config": asdict(config),
        "corpus": _corpus_report(corpus),
        "ema_inference_checkpoint": {
            "file_sha256": _sha256_file(inference_checkpoint),
            "path": inference_checkpoint.name,
        },
        "final_validation": final_validation,
        "history": {
            "file_sha256": _sha256_file(output / "training-history.jsonl"),
            "path": "training-history.jsonl",
        },
        "lineage": lineage,
        "materialization_manifest_path": str(manifest),
        "semantic_embedding_table": _semantic_table_report(semantic_table),
        "runtime": _runtime_facts(runtime, device),
        "step": config.steps,
        "training_checkpoint": {
            "file_sha256": _sha256_file(training_checkpoint),
            "path": training_checkpoint.name,
        },
    }
    report_path = output / "broad-training-report.json"
    report_bytes = _canonical_json_bytes(report)
    _atomic_bytes(report_path, report_bytes, disk_guard=disk_guard)
    return BroadTrainingResult(
        output_directory=output,
        report_path=report_path,
        training_checkpoint_path=training_checkpoint,
        inference_checkpoint_path=inference_checkpoint,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
    )


def _prepare_row(clip: MaterializedTrainingClip, *, target_size: int) -> PreparedBroadRow:
    rgba = resize_rgba_nearest(clip.rgba, target_size)
    return PreparedBroadRow(
        sequence_id=clip.sequence_id,
        identity_id=clip.identity_id,
        action=clip.request.action,
        split=clip.split,
        request=clip.request,
        rgba=rgba,
        frame_phases=clip.frame_phases,
        source_size=(clip.rgba.shape[2], clip.rgba.shape[1]),
        source_file_sha256=clip.source_file_sha256,
        normalized_array_sha256=_array_sha256(rgba),
    )


def _row_contract(row: PreparedBroadRow) -> dict[str, Any]:
    return {
        "action": row.action,
        "identity_id": row.identity_id,
        "normalized_array_sha256": row.normalized_array_sha256,
        "sequence_id": row.sequence_id,
        "source_file_sha256": row.source_file_sha256,
        "source_size": list(row.source_size),
        "split": row.split,
    }


def _corpus_report(corpus: PreparedBroadCorpus) -> dict[str, Any]:
    def split(rows: tuple[PreparedBroadRow, ...]) -> dict[str, Any]:
        return {
            "actions": _counts(row.action for row in rows),
            "identities": len({row.identity_id for row in rows}),
            "sequences": len(rows),
            "source_sizes": _counts(f"{row.source_size[0]}x{row.source_size[1]}" for row in rows),
        }

    return {
        "corpus_sha256": corpus.corpus_sha256,
        "identity_overlap": 0,
        "materialization_manifest_sha256": corpus.materialization_manifest_sha256,
        "source_snapshot_canonical_sha256": corpus.source_snapshot_canonical_sha256,
        "source_snapshot_manifest_sha256": corpus.source_snapshot_manifest_sha256,
        "spatial_transform": corpus.spatial_transform,
        "train": split(corpus.train),
        "validation": split(corpus.validation),
    }


def _counts(values: Any) -> dict[str, int]:
    output: dict[str, int] = defaultdict(int)
    for value in values:
        output[value] += 1
    return dict(sorted(output.items(), key=lambda item: item[0].encode()))


def _load_training_semantic_table(
    config: BroadTrainingConfig,
    corpus: PreparedBroadCorpus,
) -> SemanticEmbeddingTable | None:
    if config.semantic_embedding_table is None:
        return None
    table = load_semantic_embedding_table(config.semantic_embedding_table)
    missing = sorted(
        {
            row.request.description
            for row in (*corpus.train, *corpus.validation)
            if row.request.description not in table.descriptions
        },
        key=str.encode,
    )
    if missing:
        raise BroadTrainingContractError(
            f"semantic embedding table is missing descriptions: {missing!r}"
        )
    return table


def _semantic_rows(
    table: SemanticEmbeddingTable | None,
    rows: tuple[PreparedBroadRow, ...],
) -> np.ndarray | None:
    if table is None:
        return None
    return table.lookup_many([row.request.description for row in rows])


def _semantic_tensor(
    runtime: Any,
    values: np.ndarray | None,
    indices: tuple[int, ...],
    *,
    device: Any,
) -> Any | None:
    if values is None:
        return None
    selected = np.ascontiguousarray(values[np.asarray(indices, dtype=np.int64)])
    return runtime.from_numpy(selected).to(device=device)


def _encode_context(
    encoder: Any,
    encoded: EncodedConditionBatch,
    semantic_vectors: Any | None,
) -> tuple[Any, Any]:
    if semantic_vectors is None:
        return encoder(encoded)
    return encoder(encoded, semantic_vectors)


def _semantic_table_report(table: SemanticEmbeddingTable | None) -> dict[str, Any] | None:
    if table is None:
        return None
    return {
        "artifact_directory": str(table.artifact_directory),
        "description_count": len(table.descriptions),
        "embedding_dim": table.descriptor.embedding_dim,
        "embeddings_array_sha256": table.embeddings_array_sha256,
        "embeddings_file_sha256": table.embeddings_file_sha256,
        "manifest_sha256": table.manifest_sha256,
        "model_id": table.descriptor.model_id,
        "model_revision": table.descriptor.model_revision,
        "snapshot_tree_sha256": table.descriptor.snapshot_tree_sha256,
    }


def _tensor_batch(
    runtime: Any,
    rows: tuple[PreparedBroadRow, ...],
    indices: tuple[int, ...],
    *,
    device: Any,
) -> tuple[Any, Any]:
    clean_np = np.stack([rgba_uint8_to_model(rows[index].rgba) for index in indices])
    phases_np = np.asarray([rows[index].frame_phases for index in indices], dtype=np.float32)
    return (
        runtime.from_numpy(clean_np).to(device=device),
        runtime.from_numpy(phases_np).to(device=device),
    )


def _slice_encoded(batch: EncodedConditionBatch, indices: tuple[int, ...]) -> EncodedConditionBatch:
    def pick(values: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(values[index] for index in indices)

    return EncodedConditionBatch(
        descriptions=pick(batch.descriptions),
        text_token_ids=pick(batch.text_token_ids),
        text_attention_mask=pick(batch.text_attention_mask),
        entity_ids=pick(batch.entity_ids),
        action_ids=pick(batch.action_ids),
        view_ids=pick(batch.view_ids),
        direction_ids=pick(batch.direction_ids),
        loop_mode_ids=pick(batch.loop_mode_ids),
        max_text_bytes=batch.max_text_bytes,
    )


def _weighted_flow_loss(
    runtime: Any,
    predicted: Any,
    target: Any,
    clean: Any,
    config: BroadTrainingConfig,
) -> Any:
    squared = (predicted.float() - target.float()).square()
    if config.alpha_channel_weight != 1:
        weights = runtime.ones((1, 1, 4, 1, 1), device=squared.device, dtype=squared.dtype)
        weights[:, :, 3] = config.alpha_channel_weight
        squared = squared * weights
    if config.foreground_weight:
        alpha = ((clean[:, :, 3:4].float() + 1) * 0.5).clamp(0, 1)
        squared = squared * (1 + config.foreground_weight * alpha)
    return squared.mean()


def _learning_rate(step: int, config: BroadTrainingConfig) -> float:
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.steps - config.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
    return (
        config.minimum_learning_rate
        + (config.learning_rate - config.minimum_learning_rate) * cosine
    )


def _ema_update(runtime: Any, ema: Any, source: Any, decay: float) -> None:
    with runtime.no_grad():
        for ema_value, source_value in zip(ema.parameters(), source.parameters(), strict=True):
            ema_value.lerp_(source_value.detach(), 1 - decay)
        for ema_value, source_value in zip(ema.buffers(), source.buffers(), strict=True):
            ema_value.copy_(source_value)


def _validate(
    runtime: Any,
    *,
    rows: tuple[PreparedBroadRow, ...],
    encoded: EncodedConditionBatch,
    denoiser: Any,
    encoder: Any,
    config: BroadTrainingConfig,
    device: Any,
    dtype: Any,
    autocast_enabled: bool,
    semantic_vectors: np.ndarray | None,
    generate: bool = False,
    output: Path | None = None,
) -> dict[str, Any]:
    denoiser.eval()
    encoder.eval()
    base_total = 0.0
    endpoint_total = 0.0
    count = 0
    sample_records: list[dict[str, Any]] = []
    generator = runtime.Generator(device=device).manual_seed(config.seed + 100)
    if generate:
        assert output is not None
        output.mkdir(parents=True, exist_ok=False)
    with runtime.no_grad():
        for start in range(0, len(rows), config.batch_size):
            indices = tuple(range(start, min(start + config.batch_size, len(rows))))
            clean, phases = _tensor_batch(runtime, rows, indices, device=device)
            conditions = _slice_encoded(encoded, indices)
            semantic = _semantic_tensor(runtime, semantic_vectors, indices, device=device)
            context, context_mask = _encode_context(encoder, conditions, semantic)
            noise = runtime.randn(
                clean.shape, device=device, dtype=clean.dtype, generator=generator
            )
            timesteps = runtime.rand(
                (clean.shape[0],), device=device, dtype=clean.dtype, generator=generator
            )
            flow = sample_rectified_flow_batch(clean, noise=noise, timesteps=timesteps)
            endpoint_flow = sample_rectified_flow_batch(
                clean,
                noise=noise,
                timesteps=runtime.ones_like(timesteps),
            )
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                predicted = denoiser(
                    flow.noisy,
                    flow.timesteps,
                    context,
                    conditioning_mask=context_mask,
                    frame_phase=phases,
                )
                endpoint_predicted = denoiser(
                    endpoint_flow.noisy,
                    endpoint_flow.timesteps,
                    context,
                    conditioning_mask=context_mask,
                    frame_phase=phases,
                )
            base = _weighted_flow_loss(runtime, predicted, flow.target_velocity, clean, config)
            endpoint = _weighted_flow_loss(
                runtime, endpoint_predicted, endpoint_flow.target_velocity, clean, config
            )
            base_total += float(base.cpu()) * len(indices)
            endpoint_total += float(endpoint.cpu()) * len(indices)
            count += len(indices)
            if generate:
                samples = (noise - endpoint_predicted.float()).cpu().numpy()
                for local, index in enumerate(indices):
                    rgba = model_to_rgba_uint8(samples[local])
                    path = output / f"{rows[index].sequence_id}.npy"
                    _atomic_numpy(path, rgba)
                    sample_records.append(
                        {
                            "file_sha256": _sha256_file(path),
                            "normalized_target_array_sha256": rows[index].normalized_array_sha256,
                            "path": path.name,
                            "sequence_id": rows[index].sequence_id,
                        }
                    )
    denoiser.train(False)
    encoder.train(False)
    return {
        "base_loss": base_total / count,
        "endpoint_loss": endpoint_total / count,
        "fixed_noise_seed": config.seed + 100,
        "sample_count": len(sample_records) if generate else None,
        "samples": sample_records if generate else None,
    }


def _write_training_checkpoint(
    runtime: Any,
    path: Path,
    *,
    corpus: PreparedBroadCorpus,
    config: BroadTrainingConfig,
    model_config: PixelDiTConfig,
    step: int,
    denoiser: Any,
    encoder: Any,
    ema_denoiser: Any,
    ema_encoder: Any,
    optimizer: Any,
    sampler_generator: Any,
    flow_generator: Any,
    endpoint_generator: Any,
    device: Any,
    disk_guard: DiskGuard | None,
) -> None:
    payload = {
        "artifact_kind": "broad_training_resume_checkpoint",
        "broad_config": asdict(config),
        "corpus_sha256": corpus.corpus_sha256,
        "encoder_kind": (
            "frozen_semantic_plus_byte_structured_v1"
            if config.semantic_embedding_table is not None
            else "byte_structured_v1"
        ),
        "ema_condition_encoder": ema_encoder.state_dict(),
        "ema_denoiser": ema_denoiser.state_dict(),
        "materialization_manifest_sha256": corpus.materialization_manifest_sha256,
        "model_config": asdict(model_config),
        "optimizer": optimizer.state_dict(),
        "raw_condition_encoder": encoder.state_dict(),
        "raw_denoiser": denoiser.state_dict(),
        "rng_state": {
            "cuda": runtime.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "endpoint": endpoint_generator.get_state(),
            "flow": flow_generator.get_state(),
            "sampler": sampler_generator.get_state(),
            "torch_cpu": runtime.get_rng_state(),
        },
        "runtime": _runtime_facts(runtime, device),
        "semantic_embedding_table": _semantic_table_report(
            _load_training_semantic_table(config, corpus)
        ),
        "step": step,
    }
    _atomic_torch_save(runtime, path, payload, disk_guard=disk_guard)


def _write_inference_checkpoint(
    runtime: Any,
    path: Path,
    *,
    corpus: PreparedBroadCorpus,
    config: BroadTrainingConfig,
    model_config: PixelDiTConfig,
    step: int,
    denoiser: Any,
    encoder: Any,
    device: Any,
    disk_guard: DiskGuard | None,
) -> None:
    inference_config = _inference_config(asdict(config), step=step)
    payload = {
        "artifact_kind": (
            "broad_training_semantic_ema_inference_checkpoint"
            if config.semantic_embedding_table is not None
            else "broad_training_ema_inference_checkpoint"
        ),
        "broad_config": asdict(config),
        "condition_encoder": encoder.state_dict(),
        "config": inference_config,
        "corpus_sha256": corpus.corpus_sha256,
        "denoiser": denoiser.state_dict(),
        "materialization_manifest_sha256": corpus.materialization_manifest_sha256,
        "model_config": asdict(model_config),
        "runtime": _runtime_facts(runtime, device),
        "semantic_embedding_table": _semantic_table_report(
            _load_training_semantic_table(config, corpus)
        ),
        "step": step,
    }
    _atomic_torch_save(runtime, path, payload, disk_guard=disk_guard)


def _inference_config(config: dict[str, Any], *, step: int) -> dict[str, Any]:
    return {
        "alpha_channel_weight": config["alpha_channel_weight"],
        "condition_dim": config["condition_dim"],
        "depth": config["depth"],
        "device": config["device"],
        "foreground_weight": config["foreground_weight"],
        "learning_rate": config["learning_rate"],
        "log_every": config["log_every"],
        "matched_endpoint_weight": config["endpoint_weight"],
        "max_text_bytes": config["max_text_bytes"],
        "model_dim": config["model_dim"],
        "num_heads": config["num_heads"],
        "patch_size": config["patch_size"],
        "precision": config["precision"],
        "sample_steps": 1,
        "seed": config["seed"],
        "steps": step,
        "target_bucket": config["target_size"],
        "target_frames": config["target_frames"],
        "weight_decay": config["weight_decay"],
    }


def _load_resume_checkpoint(
    runtime: Any,
    path: Path,
    *,
    expected_sha256: str,
    corpus: PreparedBroadCorpus,
    config: BroadTrainingConfig,
    model_config: PixelDiTConfig,
) -> dict[str, Any]:
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise BroadTrainingContractError(
            f"resume checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    try:
        payload = runtime.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise BroadTrainingContractError("resume checkpoint failed safe load") from error
    if not isinstance(payload, dict):
        raise BroadTrainingContractError("resume checkpoint root must be a mapping")
    if payload.get("artifact_kind") != "broad_training_resume_checkpoint":
        raise BroadTrainingContractError("resume checkpoint has the wrong artifact kind")
    if payload.get("corpus_sha256") != corpus.corpus_sha256:
        raise BroadTrainingContractError("resume corpus SHA-256 differs")
    parent_config = payload.get("broad_config")
    if not isinstance(parent_config, dict):
        raise BroadTrainingContractError("resume broad_config is missing")
    expected_config = asdict(config)
    for key, value in parent_config.items():
        if key != "steps" and expected_config.get(key) != value:
            raise BroadTrainingContractError(f"resume config differs at {key!r}")
    if parent_config.get("steps", 0) > config.steps:
        raise BroadTrainingContractError("resume parent planned steps exceed child steps")
    if payload.get("model_config") != asdict(model_config):
        raise BroadTrainingContractError("resume model configuration differs")
    current_semantic = _semantic_table_report(_load_training_semantic_table(config, corpus))
    if payload.get("semantic_embedding_table") != current_semantic:
        raise BroadTrainingContractError("resume semantic embedding table differs")
    return payload


def _runtime_facts(runtime: Any, device: Any) -> dict[str, Any]:
    return {
        "cuda_version": runtime.version.cuda,
        "cudnn_version": runtime.backends.cudnn.version(),
        "deterministic_algorithms_enabled": runtime.are_deterministic_algorithms_enabled(),
        "device": str(device),
        "device_name": runtime.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(runtime.__version__),
    }


def _array_sha256(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes, *, disk_guard: DiskGuard | None) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing artifact: {path}")
    if disk_guard is not None:
        disk_guard.require_capacity(additional_bytes=len(payload), label=path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing sample: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as handle:
        temporary = Path(handle.name)
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_torch_save(
    runtime: Any,
    path: Path,
    payload: dict[str, Any],
    *,
    disk_guard: DiskGuard | None,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint: {path}")
    if disk_guard is not None:
        disk_guard.require_capacity(label=path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        runtime.save(payload, temporary)
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _require_torch() -> Any:
    if torch is None:
        raise MissingBroadTrainingTorchError(
            "broad training requires a platform-appropriate PyTorch installation"
        ) from _TORCH_IMPORT_ERROR
    return torch
