"""Verified, no-clobber inference for tiny PixelDiT overfit checkpoints.

This module deliberately treats the current checkpoints as memorization diagnostics.
It accepts arbitrary description text for replay experiments, but it does not turn an
overfit checkpoint into evidence of open-vocabulary generalization.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np

from spritelab.captions import SpriteGenerationRequest
from spritelab.models.conditioning import (
    EncodedConditionBatch,
    SpriteConditionEncoder,
    encode_generation_conditions,
)
from spritelab.models.config import ConditioningSchema, PixelDiTConfig, SpriteClipCondition
from spritelab.models.flow import endpoint_sample_velocity_model, euler_sample_velocity_model
from spritelab.models.pixeldit import FactorizedSpriteDiT
from spritelab.models.semantic_conditioning import SemanticSpriteConditionEncoder
from spritelab.semantic_text import SemanticEmbeddingTable, load_semantic_embedding_table
from spritelab.storage import DiskGuard, HashMismatch
from spritelab.training_data import model_to_rgba_uint8

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised in the torch-free venv
    torch = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


NoiseStrategy = Literal["independent", "shared"]
SamplerAlgorithm = Literal["euler", "endpoint"]


class MissingInferenceTorchError(RuntimeError):
    """Raised when checkpoint inference is requested without PyTorch."""


class CheckpointContractError(ValueError):
    """Raised when a verified checkpoint does not satisfy the inference contract."""


@dataclass(frozen=True, slots=True)
class CheckpointInferenceConfig:
    """Runtime controls that are independent of the stored training configuration."""

    seed: int
    sample_steps: int = 32
    sampler_algorithm: SamplerAlgorithm = "euler"
    noise_strategy: NoiseStrategy = "independent"
    device: str = "cpu"
    deterministic_algorithms: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        _positive_integer("sample_steps", self.sample_steps)
        if self.sampler_algorithm not in {"euler", "endpoint"}:
            raise ValueError(
                f"sampler_algorithm must be 'euler' or 'endpoint'; got {self.sampler_algorithm!r}"
            )
        if self.sampler_algorithm == "endpoint" and self.sample_steps != 1:
            raise ValueError("endpoint sampling requires sample_steps=1")
        if self.noise_strategy not in {"independent", "shared"}:
            raise ValueError(
                f"noise_strategy must be 'independent' or 'shared'; got {self.noise_strategy!r}"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if not isinstance(self.deterministic_algorithms, bool):
            raise TypeError("deterministic_algorithms must be a bool")


@dataclass(frozen=True, slots=True)
class CheckpointInferenceResult:
    """Immutable paths and hashes returned after a successful inference run."""

    output_directory: Path
    report_path: Path
    sample_paths: tuple[Path, ...]
    checkpoint_sha256: str
    report_sha256: str
    noise_sha256: str


@dataclass(frozen=True, slots=True)
class _LoadedCheckpoint:
    model_config: PixelDiTConfig
    max_text_bytes: int
    denoiser: Any
    condition_encoder: Any
    stored_training_config: dict[str, Any]
    stored_runtime: dict[str, Any]
    step: int
    semantic_table: SemanticEmbeddingTable | None


_BASE_TRAINING_CONFIG_FIELDS = (
    "target_bucket",
    "target_frames",
    "patch_size",
    "model_dim",
    "depth",
    "num_heads",
    "condition_dim",
    "max_text_bytes",
    "learning_rate",
    "weight_decay",
    "foreground_weight",
    "matched_endpoint_weight",
    "steps",
    "log_every",
    "sample_steps",
    "seed",
    "device",
    "precision",
)
_OPTIONAL_TRAINING_CONFIG_FIELDS = ("alpha_channel_weight",)


def run_checkpoint_inference(
    checkpoint_path: Path | str,
    output_directory: Path | str,
    requests: Sequence[SpriteGenerationRequest],
    frame_phases: Sequence[Sequence[float]],
    *,
    expected_checkpoint_sha256: str,
    config: CheckpointInferenceConfig,
    source_report_path: Path | str | None = None,
    expected_source_report_sha256: str | None = None,
    disk_guard: DiskGuard | None = None,
) -> CheckpointInferenceResult:
    """Generate uint8 RGBA clips from a hash-verified overfit checkpoint.

    The checkpoint is hashed before it is deserialized and is loaded only through
    ``torch.load(..., weights_only=True)``. There is intentionally no unsafe pickle
    fallback. Every output is atomically published and existing targets are refused.
    """

    runtime = _require_torch()
    inference = config
    request_rows = _validate_requests(requests)
    phase_rows = _normalize_phase_rows(frame_phases, expected_batch=len(request_rows))
    expected_digest = _required_digest(expected_checkpoint_sha256, "expected_checkpoint_sha256")

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    checkpoint_digest = _sha256_file(checkpoint)
    if checkpoint_digest != expected_digest:
        raise HashMismatch(
            f"Checkpoint SHA-256 mismatch: expected {expected_digest}, received {checkpoint_digest}"
        )

    source_report = _verify_source_report(
        source_report_path,
        expected_sha256=expected_source_report_sha256,
        checkpoint_sha256=checkpoint_digest,
    )
    output = Path(output_directory).resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"inference output path is not a directory: {output}")
    report_path = output / "inference-report.json"
    sample_paths = _planned_sample_paths(output, request_rows, phase_rows)
    _preflight_no_clobber((report_path, *sample_paths))

    loaded = _load_verified_checkpoint(runtime, checkpoint)
    _validate_phases_against_schema(
        request_rows,
        phase_rows,
        loaded.model_config,
    )
    encoded = encode_generation_conditions(
        request_rows,
        loaded.model_config.conditioning,
        max_text_bytes=loaded.max_text_bytes,
    )
    semantic_vectors = None
    if loaded.semantic_table is not None:
        try:
            semantic_array = loaded.semantic_table.lookup_many(
                [request.description for request in request_rows]
            )
        except KeyError as error:
            raise CheckpointContractError(
                "semantic checkpoint accepts only descriptions in its hash-bound table"
            ) from error
        semantic_vectors = runtime.from_numpy(semantic_array)
    device = _validated_device(runtime, inference.device)
    estimated_output_bytes = _estimated_output_bytes(
        loaded.model_config,
        batch_size=len(request_rows),
    )
    if disk_guard is not None:
        disk_guard.require_capacity(
            estimated_output_bytes,
            label="checkpoint inference outputs",
        )

    denoiser = loaded.denoiser.to(device=device)
    condition_encoder = loaded.condition_encoder.to(device=device)
    if semantic_vectors is not None:
        semantic_vectors = semantic_vectors.to(device=device)
    denoiser.eval()
    condition_encoder.eval()
    phase_array = np.ascontiguousarray(phase_rows, dtype=np.float32)
    phases = runtime.from_numpy(phase_array).to(device=device)
    generator = runtime.Generator(device=device).manual_seed(inference.seed)
    generator_initial_sha256 = _tensor_sha256(runtime, generator.get_state())
    noise = _generate_noise(
        runtime,
        loaded.model_config,
        batch_size=len(request_rows),
        strategy=inference.noise_strategy,
        generator=generator,
        device=device,
    )
    generator_final_sha256 = _tensor_sha256(runtime, generator.get_state())
    noise_sha256 = _tensor_sha256(runtime, noise)
    noise_row_sha256 = [_tensor_sha256(runtime, row) for row in noise]

    previous_deterministic = runtime.are_deterministic_algorithms_enabled()
    try:
        runtime.use_deterministic_algorithms(inference.deterministic_algorithms)
        with runtime.no_grad():
            if semantic_vectors is None:
                context, context_mask = condition_encoder(encoded)
            else:
                context, context_mask = condition_encoder(encoded, semantic_vectors)
            if inference.sampler_algorithm == "endpoint":
                sampled = endpoint_sample_velocity_model(
                    denoiser,
                    noise,
                    conditioning=context,
                    conditioning_mask=context_mask,
                    frame_phase=phases,
                )
            else:
                sampled = euler_sample_velocity_model(
                    denoiser,
                    noise,
                    steps=inference.sample_steps,
                    conditioning=context,
                    conditioning_mask=context_mask,
                    frame_phase=phases,
                )
    finally:
        runtime.use_deterministic_algorithms(previous_deterministic)

    sampled_cpu = sampled.float().cpu().numpy()
    arrays = tuple(model_to_rgba_uint8(row) for row in sampled_cpu)
    encoded_rows = _encoded_condition_rows(encoded)
    sample_records: list[dict[str, Any]] = []
    for index, (path, array, request, requested_phases, phases_row, encoded_row) in enumerate(
        zip(
            sample_paths,
            arrays,
            request_rows,
            phase_rows,
            phase_array,
            encoded_rows,
            strict=True,
        )
    ):
        payload = _numpy_bytes(array)
        _atomic_bytes(
            path,
            payload,
            disk_guard=disk_guard,
            label="checkpoint inference sample",
        )
        sample_records.append(
            {
                "array_sha256": _numpy_array_sha256(array),
                "dtype": str(array.dtype),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "index": index,
                "path": path.relative_to(output).as_posix(),
                "shape": list(array.shape),
                "size_bytes": len(payload),
                "request": asdict(request),
                "request_sha256": _request_sha256(request, requested_phases),
                "rendered_prompt": request.text,
                "model_text_prompt": request.description.strip(),
                "structured_labels": {
                    "action": request.action,
                    "direction": request.direction,
                    "entity_class": request.entity_class,
                    "loop_mode": request.loop_mode,
                    "view": request.view,
                },
                "frame_phases_requested": list(requested_phases),
                "frame_phases_float32": [float(value) for value in phases_row],
                "encoded_condition": encoded_row,
                "noise_row_sha256": noise_row_sha256[index],
            }
        )

    current_runtime = _runtime_facts(runtime, device)
    report = {
        "artifact_kind": "pixeldit_overfit_checkpoint_inference",
        "schema_version": 1,
        "claim_scope": {
            "accepted_input": (
                "hash-bound semantic-table descriptions plus checkpoint-vocabulary structured "
                "labels"
                if loaded.semantic_table is not None
                else "arbitrary description text plus checkpoint-vocabulary structured labels"
            ),
            "limit": (
                "checkpoint replay and conditioning diagnostic only; accepted text is not "
                "evidence of open-vocabulary generalization"
            ),
        },
        "checkpoint": {
            "file_sha256": checkpoint_digest,
            "load_contract": "torch.load(weights_only=True,map_location='cpu')",
            "path": str(checkpoint),
            "step": loaded.step,
            "stored_training_config": loaded.stored_training_config,
            "stored_training_runtime": loaded.stored_runtime,
        },
        "source_report": source_report,
        "model_config": asdict(loaded.model_config),
        "condition_encoder": {
            "class": "SpriteConditionEncoder",
            "max_text_bytes": loaded.max_text_bytes,
            "text_contract": (
                "model receives the stripped description through deterministic NFC UTF-8 "
                "byte tokenization; rendered_prompt is documentation, not a second text input"
            ),
            "semantic_embedding_table": (
                {
                    "embedding_dim": loaded.semantic_table.descriptor.embedding_dim,
                    "embeddings_array_sha256": loaded.semantic_table.embeddings_array_sha256,
                    "manifest_sha256": loaded.semantic_table.manifest_sha256,
                    "model_id": loaded.semantic_table.descriptor.model_id,
                    "model_revision": loaded.semantic_table.descriptor.model_revision,
                }
                if loaded.semantic_table is not None
                else None
            ),
        },
        "sampler": {
            "algorithm": (
                "direct_t1_endpoint_velocity"
                if inference.sampler_algorithm == "endpoint"
                else "backward_euler_rectified_flow"
            ),
            "path_direction": "t=1_noise_to_t=0_data",
            "sample_steps": inference.sample_steps,
        },
        "rng": {
            "generator": "torch.Generator",
            "generator_device": str(device),
            "generator_final_state_sha256": generator_final_sha256,
            "generator_initial_state_sha256": generator_initial_sha256,
            "noise_batch_sha256": noise_sha256,
            "noise_row_sha256": noise_row_sha256,
            "noise_strategy": inference.noise_strategy,
            "seed": inference.seed,
        },
        "runtime": {
            **current_runtime,
            "deterministic_algorithms_requested": inference.deterministic_algorithms,
            "deterministic_algorithms_enabled_during_sampling": (
                inference.deterministic_algorithms
            ),
            "replay_limit": (
                "same-seed byte equality is expected only on a compatible deterministic "
                "Torch/device stack; the recorded hashes make divergence observable"
            ),
        },
        "sample_count": len(sample_records),
        "samples": sample_records,
    }
    report_bytes = _canonical_json_bytes(report)
    _atomic_bytes(
        report_path,
        report_bytes,
        disk_guard=disk_guard,
        label="checkpoint inference report",
    )
    return CheckpointInferenceResult(
        output_directory=output,
        report_path=report_path,
        sample_paths=sample_paths,
        checkpoint_sha256=checkpoint_digest,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        noise_sha256=noise_sha256,
    )


def _load_verified_checkpoint(runtime: Any, path: Path) -> _LoadedCheckpoint:
    try:
        payload = runtime.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise CheckpointContractError(
            "checkpoint could not be loaded with safe weights_only=True semantics; "
            "unsafe pickle loading is intentionally unsupported"
        ) from error
    if not isinstance(payload, Mapping):
        raise CheckpointContractError("checkpoint root must be a mapping")

    model_config = _reconstruct_model_config(_required_mapping(payload, "model_config"))
    training_config = _validate_training_config(_required_mapping(payload, "config"), model_config)
    stored_runtime = _validate_stored_runtime(_required_mapping(payload, "runtime"))
    step = _required_positive_integer(payload.get("step"), "checkpoint.step")
    if step != training_config["steps"]:
        raise CheckpointContractError(
            f"checkpoint.step must equal config.steps; got {step} and {training_config['steps']}"
        )

    denoiser_state = _validate_state_dict(runtime, payload.get("denoiser"), "checkpoint.denoiser")
    encoder_state = _validate_state_dict(
        runtime,
        payload.get("condition_encoder"),
        "checkpoint.condition_encoder",
    )
    denoiser = FactorizedSpriteDiT(model_config)
    semantic_table = _load_checkpoint_semantic_table(payload)
    if semantic_table is None:
        condition_encoder = SpriteConditionEncoder(
            model_config.conditioning,
            condition_dim=model_config.condition_dim,
            max_text_bytes=training_config["max_text_bytes"],
        )
    else:
        condition_encoder = SemanticSpriteConditionEncoder(
            model_config.conditioning,
            condition_dim=model_config.condition_dim,
            semantic_dim=semantic_table.descriptor.embedding_dim,
            max_text_bytes=training_config["max_text_bytes"],
        )
    try:
        denoiser.load_state_dict(denoiser_state, strict=True)
        condition_encoder.load_state_dict(encoder_state, strict=True)
    except RuntimeError as error:
        raise CheckpointContractError(
            "checkpoint state does not exactly match its stored model/encoder configuration"
        ) from error
    return _LoadedCheckpoint(
        model_config=model_config,
        max_text_bytes=training_config["max_text_bytes"],
        denoiser=denoiser,
        condition_encoder=condition_encoder,
        stored_training_config=training_config,
        stored_runtime=stored_runtime,
        step=step,
        semantic_table=semantic_table,
    )


def _load_checkpoint_semantic_table(
    payload: Mapping[str, Any],
) -> SemanticEmbeddingTable | None:
    artifact_kind = payload.get("artifact_kind")
    semantic_kind = "broad_training_semantic_ema_inference_checkpoint"
    raw = payload.get("semantic_embedding_table")
    if artifact_kind != semantic_kind:
        if raw is not None:
            raise CheckpointContractError(
                "non-semantic checkpoint cannot declare a semantic embedding table"
            )
        return None
    if not isinstance(raw, Mapping):
        raise CheckpointContractError("semantic checkpoint table record is missing")
    directory = raw.get("artifact_directory")
    if not isinstance(directory, str) or not directory:
        raise CheckpointContractError("semantic checkpoint table path is invalid")
    try:
        table = load_semantic_embedding_table(directory)
    except (OSError, ValueError) as error:
        raise CheckpointContractError("semantic checkpoint table failed verification") from error
    expected = {
        "embeddings_array_sha256": table.embeddings_array_sha256,
        "embeddings_file_sha256": table.embeddings_file_sha256,
        "manifest_sha256": table.manifest_sha256,
        "model_id": table.descriptor.model_id,
        "model_revision": table.descriptor.model_revision,
        "snapshot_tree_sha256": table.descriptor.snapshot_tree_sha256,
    }
    mismatches = {
        key: (raw.get(key), value) for key, value in expected.items() if raw.get(key) != value
    }
    if mismatches:
        raise CheckpointContractError(
            f"semantic checkpoint table record differs from artifact: {mismatches!r}"
        )
    return table


def _reconstruct_model_config(raw: Mapping[str, Any]) -> PixelDiTConfig:
    model_fields = {field.name for field in fields(PixelDiTConfig)}
    if set(raw) != model_fields:
        raise CheckpointContractError(
            "checkpoint.model_config fields do not exactly match PixelDiTConfig; "
            f"missing={sorted(model_fields.difference(raw))!r}, "
            f"extra={sorted(set(raw).difference(model_fields))!r}"
        )
    schema_raw = raw.get("conditioning")
    if not isinstance(schema_raw, Mapping):
        raise CheckpointContractError("checkpoint.model_config.conditioning must be a mapping")
    schema_fields = {field.name for field in fields(ConditioningSchema)}
    if set(schema_raw) != schema_fields:
        raise CheckpointContractError(
            "stored conditioning fields do not exactly match ConditioningSchema; "
            f"missing={sorted(schema_fields.difference(schema_raw))!r}, "
            f"extra={sorted(set(schema_raw).difference(schema_fields))!r}"
        )
    vocabulary_fields = {
        "entity_classes",
        "action_classes",
        "view_classes",
        "direction_classes",
        "loop_modes",
        "condition_tokens",
    }
    schema_values: dict[str, Any] = {}
    for name in schema_fields:
        value = schema_raw[name]
        if name in vocabulary_fields:
            if not isinstance(value, list | tuple) or not all(
                isinstance(label, str) for label in value
            ):
                raise CheckpointContractError(f"stored conditioning.{name} must be strings")
            value = tuple(value)
        schema_values[name] = value
    try:
        schema = ConditioningSchema(**schema_values)
        model_values = dict(raw)
        model_values["conditioning"] = schema
        return PixelDiTConfig(**model_values)
    except (TypeError, ValueError) as error:
        raise CheckpointContractError(f"invalid stored PixelDiT configuration: {error}") from error


def _validate_training_config(
    raw: Mapping[str, Any], model_config: PixelDiTConfig
) -> dict[str, Any]:
    allowed_fields = set(_BASE_TRAINING_CONFIG_FIELDS) | set(_OPTIONAL_TRAINING_CONFIG_FIELDS)
    missing = set(_BASE_TRAINING_CONFIG_FIELDS).difference(raw)
    extra = set(raw).difference(allowed_fields)
    if missing or extra:
        raise CheckpointContractError(
            "checkpoint.config fields do not match TinyOverfitConfig; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    positive = {
        "target_bucket",
        "target_frames",
        "patch_size",
        "model_dim",
        "depth",
        "num_heads",
        "condition_dim",
        "max_text_bytes",
        "steps",
        "log_every",
        "sample_steps",
    }
    normalized: dict[str, Any] = {}
    for name in (*_BASE_TRAINING_CONFIG_FIELDS, *_OPTIONAL_TRAINING_CONFIG_FIELDS):
        value = raw.get(name, 1.0) if name == "alpha_channel_weight" else raw[name]
        if name in positive:
            value = _required_positive_integer(value, f"checkpoint.config.{name}")
        elif name == "seed":
            if isinstance(value, bool) or not isinstance(value, int):
                raise CheckpointContractError("checkpoint.config.seed must be an integer")
        elif name in {
            "learning_rate",
            "weight_decay",
            "foreground_weight",
            "alpha_channel_weight",
            "matched_endpoint_weight",
        }:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise CheckpointContractError(f"checkpoint.config.{name} must be numeric")
            value = float(value)
            if not math.isfinite(value):
                raise CheckpointContractError(f"checkpoint.config.{name} must be finite")
        elif name == "device":
            if not isinstance(value, str) or not value:
                raise CheckpointContractError("checkpoint.config.device must be non-empty")
        elif name == "precision" and value not in {"float32", "bfloat16"}:
            raise CheckpointContractError(f"unsupported stored training precision: {value!r}")
        normalized[name] = value
    nonnegative = (
        "weight_decay",
        "foreground_weight",
        "alpha_channel_weight",
        "matched_endpoint_weight",
    )
    if any(normalized[name] < 0 for name in nonnegative):
        raise CheckpointContractError("stored non-negative training weights cannot be negative")
    if normalized["learning_rate"] <= 0:
        raise CheckpointContractError("stored learning_rate must be positive")
    expected = {
        "target_bucket": model_config.height,
        "target_frames": model_config.num_frames,
        "patch_size": model_config.patch_size,
        "model_dim": model_config.model_dim,
        "depth": model_config.depth,
        "num_heads": model_config.num_heads,
        "condition_dim": model_config.condition_dim,
    }
    if model_config.height != model_config.width:
        raise CheckpointContractError("TinyOverfitConfig requires a square model bucket")
    mismatches = {
        name: (normalized[name], expected_value)
        for name, expected_value in expected.items()
        if normalized[name] != expected_value
    }
    if mismatches:
        raise CheckpointContractError(
            f"stored training and model configurations disagree: {mismatches!r}"
        )
    return normalized


def _validate_stored_runtime(raw: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "cuda_version",
        "cudnn_version",
        "deterministic_algorithms_enabled",
        "device",
        "device_name",
        "torch_version",
    }
    missing = required.difference(raw)
    if missing:
        raise CheckpointContractError(
            f"checkpoint.runtime is missing required facts: {sorted(missing)!r}"
        )
    string_or_none = ("cuda_version", "device_name")
    normalized: dict[str, Any] = {}
    for name in string_or_none:
        value = raw[name]
        if value is not None and not isinstance(value, str):
            raise CheckpointContractError(f"checkpoint.runtime.{name} must be text or null")
        normalized[name] = value
    cudnn = raw["cudnn_version"]
    if cudnn is not None and (isinstance(cudnn, bool) or not isinstance(cudnn, int)):
        raise CheckpointContractError("checkpoint.runtime.cudnn_version must be integer or null")
    normalized["cudnn_version"] = cudnn
    deterministic = raw["deterministic_algorithms_enabled"]
    if not isinstance(deterministic, bool):
        raise CheckpointContractError(
            "checkpoint.runtime.deterministic_algorithms_enabled must be bool"
        )
    normalized["deterministic_algorithms_enabled"] = deterministic
    for name in ("device", "torch_version"):
        value = raw[name]
        if not isinstance(value, str) or not value:
            raise CheckpointContractError(f"checkpoint.runtime.{name} must be non-empty text")
        normalized[name] = value
    return normalized


def _validate_state_dict(runtime: Any, raw: object, name: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not raw:
        raise CheckpointContractError(f"{name} must be a non-empty state mapping")
    state: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, runtime.Tensor):
            raise CheckpointContractError(f"{name} must map string keys only to tensors")
        if (value.is_floating_point() or value.is_complex()) and not bool(
            runtime.isfinite(value).all()
        ):
            raise CheckpointContractError(f"{name}.{key} contains non-finite values")
        state[key] = value
    return state


def _validate_requests(
    requests: Sequence[SpriteGenerationRequest],
) -> tuple[SpriteGenerationRequest, ...]:
    if isinstance(requests, str | bytes) or not isinstance(requests, Sequence):
        raise TypeError("requests must be a sequence of SpriteGenerationRequest")
    normalized = tuple(requests)
    if not normalized:
        raise ValueError("at least one SpriteGenerationRequest is required")
    for index, request in enumerate(normalized):
        if not isinstance(request, SpriteGenerationRequest):
            raise TypeError(f"requests[{index}] must be a SpriteGenerationRequest")
    return normalized


def _normalize_phase_rows(
    rows: Sequence[Sequence[float]], *, expected_batch: int
) -> tuple[tuple[float, ...], ...]:
    if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
        raise TypeError("frame_phases must be a sequence of per-request sequences")
    if len(rows) != expected_batch:
        raise ValueError(
            f"frame_phases must contain one row per request; got {len(rows)} and {expected_batch}"
        )
    output: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        if isinstance(row, str | bytes) or not isinstance(row, Sequence):
            raise TypeError(f"frame_phases[{row_index}] must be a sequence")
        values: list[float] = []
        for column, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(
                value, int | float | np.integer | np.floating
            ):
                raise TypeError(f"frame_phases[{row_index}][{column}] must be a real number")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"frame_phases[{row_index}][{column}] must be finite")
            values.append(converted)
        output.append(tuple(values))
    return tuple(output)


def _validate_phases_against_schema(
    requests: tuple[SpriteGenerationRequest, ...],
    phases: tuple[tuple[float, ...], ...],
    model_config: PixelDiTConfig,
) -> None:
    for index, (request, row) in enumerate(zip(requests, phases, strict=True)):
        condition = SpriteClipCondition(
            identity_id=f"inference-request-{index}",
            entity_class=request.entity_class,
            action=request.action,
            view=request.view,
            direction=request.direction,
            loop_mode=request.loop_mode,
            frame_phases=row,
        )
        try:
            model_config.conditioning.validate_clip(
                condition,
                expected_frames=model_config.num_frames,
            )
        except ValueError as error:
            raise ValueError(f"invalid request/frame_phases row {index}: {error}") from error


def _generate_noise(
    runtime: Any,
    model_config: PixelDiTConfig,
    *,
    batch_size: int,
    strategy: NoiseStrategy,
    generator: Any,
    device: Any,
) -> Any:
    tail = (
        model_config.num_frames,
        model_config.channels,
        model_config.height,
        model_config.width,
    )
    if strategy == "shared":
        row = runtime.randn(
            (1, *tail),
            device=device,
            dtype=runtime.float32,
            generator=generator,
        )
        return row.repeat(batch_size, 1, 1, 1, 1)
    return runtime.randn(
        (batch_size, *tail),
        device=device,
        dtype=runtime.float32,
        generator=generator,
    )


def _encoded_condition_rows(batch: EncodedConditionBatch) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    categorical = zip(
        batch.entity_ids,
        batch.action_ids,
        batch.view_ids,
        batch.direction_ids,
        batch.loop_mode_ids,
        strict=True,
    )
    for index, ids in enumerate(categorical):
        output.append(
            {
                "categorical_ids": {
                    "action": ids[1],
                    "direction": ids[3],
                    "entity_class": ids[0],
                    "loop_mode": ids[4],
                    "view": ids[2],
                },
                "description_after_strip": batch.descriptions[index],
                "max_text_bytes": batch.max_text_bytes,
                "text_attention_mask": list(batch.text_attention_mask[index]),
                "text_token_ids": list(batch.text_token_ids[index]),
            }
        )
    return output


def _verify_source_report(
    path_value: Path | str | None,
    *,
    expected_sha256: str | None,
    checkpoint_sha256: str,
) -> dict[str, Any] | None:
    if path_value is None:
        if expected_sha256 is not None:
            raise ValueError("expected_source_report_sha256 requires source_report_path")
        return None
    path = Path(path_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"source report does not exist: {path}")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        expected = _required_digest(expected_sha256, "expected_source_report_sha256")
        if actual_sha256 != expected:
            raise HashMismatch(
                f"Source report SHA-256 mismatch: expected {expected}, received {actual_sha256}"
            )
    try:
        report = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointContractError("source report must be valid UTF-8 JSON") from error
    if not isinstance(report, Mapping):
        raise CheckpointContractError("source report root must be an object")
    checkpoint_record = report.get("checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise CheckpointContractError("source report must contain a checkpoint object")
    declared = _required_digest(
        checkpoint_record.get("file_sha256"),
        "source_report.checkpoint.file_sha256",
    )
    if declared != checkpoint_sha256:
        raise HashMismatch(
            "Source report names a different checkpoint SHA-256: "
            f"expected {checkpoint_sha256}, received {declared}"
        )
    report_config = report.get("config")
    return {
        "artifact_kind": report.get("artifact_kind"),
        "file_sha256": actual_sha256,
        "path": str(path),
        "checkpoint_file_sha256": declared,
        "config_present": isinstance(report_config, Mapping),
    }


def _planned_sample_paths(
    output: Path,
    requests: tuple[SpriteGenerationRequest, ...],
    phases: tuple[tuple[float, ...], ...],
) -> tuple[Path, ...]:
    return tuple(
        output / "samples" / f"{index:04d}-{_request_sha256(request, phase_row)[:20]}.npy"
        for index, (request, phase_row) in enumerate(zip(requests, phases, strict=True))
    )


def _request_sha256(request: SpriteGenerationRequest, phases: tuple[float, ...]) -> str:
    payload = {
        "frame_phases": list(phases),
        "request": asdict(request),
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _preflight_no_clobber(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to replace existing inference artifact: {path}")


def _numpy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def _numpy_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = f"{contiguous.dtype.str}\0{'x'.join(map(str, contiguous.shape))}\0".encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _tensor_sha256(runtime: Any, value: Any) -> str:
    contiguous = value.detach().contiguous()
    raw = contiguous.view(runtime.uint8).cpu().numpy().tobytes(order="C")
    header = (
        f"{contiguous.dtype}\0{'x'.join(str(dimension) for dimension in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + raw).hexdigest()


def _atomic_bytes(
    path: Path,
    payload: bytes,
    *,
    disk_guard: DiskGuard | None,
    label: str,
) -> None:
    if disk_guard is not None:
        disk_guard.require_capacity(len(payload), label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to replace existing inference artifact: {path}"
            ) from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _estimated_output_bytes(model_config: PixelDiTConfig, *, batch_size: int) -> int:
    array_bytes = (
        batch_size
        * model_config.num_frames
        * model_config.height
        * model_config.width
        * model_config.channels
    )
    return array_bytes + (batch_size + 1) * 128 * 1024


def _runtime_facts(runtime: Any, device: Any) -> dict[str, Any]:
    return {
        "cuda_version": getattr(runtime.version, "cuda", None),
        "cudnn_version": (
            runtime.backends.cudnn.version() if runtime.backends.cudnn.is_available() else None
        ),
        "deterministic_algorithms_enabled_at_report": (
            runtime.are_deterministic_algorithms_enabled()
        ),
        "device": str(device),
        "device_name": (runtime.cuda.get_device_name(device) if device.type == "cuda" else None),
        "numpy_version": str(np.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(runtime.__version__),
    }


def _validated_device(runtime: Any, value: str) -> Any:
    try:
        device = runtime.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid torch device: {value!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("checkpoint inference currently supports only CPU and CUDA devices")
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _require_torch() -> Any:
    if torch is None:
        raise MissingInferenceTorchError(
            "checkpoint inference requires a platform-appropriate PyTorch installation"
        ) from _TORCH_IMPORT_ERROR
    return torch


def _required_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise CheckpointContractError(f"checkpoint.{name} must be a mapping")
    return value


def _required_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CheckpointContractError(f"{name} must be a positive integer; got {value!r}")
    return value


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _required_digest(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
