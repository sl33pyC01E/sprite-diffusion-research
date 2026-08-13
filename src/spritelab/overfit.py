"""Optional-PyTorch tiny-corpus overfit runner for pipeline verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from spritelab.models import ConditioningSchema, PixelDiTConfig
from spritelab.models.conditioning import (
    EncodedConditionBatch,
    SpriteConditionEncoder,
    encode_generation_conditions,
)
from spritelab.models.flow import (
    RectifiedFlowBatch,
    euler_sample_velocity_model,
    sample_rectified_flow_batch,
)
from spritelab.models.pixeldit import FactorizedSpriteDiT
from spritelab.storage import DiskGuard, HashMismatch
from spritelab.training_data import (
    collate_materialized_clips,
    load_materialized_training_clips,
    model_to_rgba_uint8,
)

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised in the torch-free venv
    torch = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None

Precision = Literal["float32", "bfloat16"]


class MissingOverfitTorchError(RuntimeError):
    """Raised when the overfit runner is used without PyTorch."""


class OverfitContinuationContractError(ValueError):
    """Raised when a parent overfit bundle cannot be resumed truthfully."""


@dataclass(frozen=True, slots=True)
class TinyOverfitConfig:
    """Small, explicit experiment configuration suitable for one 64-pixel bucket."""

    target_bucket: int = 64
    target_frames: int = 8
    patch_size: int = 4
    model_dim: int = 128
    depth: int = 4
    num_heads: int = 4
    condition_dim: int = 128
    max_text_bytes: int = 48
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    foreground_weight: float = 2.0
    alpha_channel_weight: float = 1.0
    matched_endpoint_weight: float = 1.0
    steps: int = 1_000
    log_every: int = 25
    sample_steps: int = 32
    seed: int = 0
    device: str = "cuda"
    precision: Precision = "float32"

    def __post_init__(self) -> None:
        for name in (
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
        ):
            _positive_integer(name, getattr(self, name))
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.foreground_weight) or self.foreground_weight < 0:
            raise ValueError("foreground_weight must be finite and non-negative")
        if not math.isfinite(self.alpha_channel_weight) or self.alpha_channel_weight < 0:
            raise ValueError("alpha_channel_weight must be finite and non-negative")
        if not math.isfinite(self.matched_endpoint_weight) or self.matched_endpoint_weight < 0:
            raise ValueError("matched_endpoint_weight must be finite and non-negative")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.target_bucket % self.patch_size:
            raise ValueError("target_bucket must be divisible by patch_size")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError(f"unsupported precision: {self.precision!r}")


@dataclass(frozen=True, slots=True)
class TinyOverfitResult:
    output_directory: Path
    report_path: Path
    checkpoint_path: Path
    sample_paths: tuple[Path, ...]
    initial_loss: float
    final_loss: float
    minimum_loss: float
    report_sha256: str


@dataclass(frozen=True, slots=True)
class _ContinuationContext:
    parent_checkpoint_path: Path
    parent_checkpoint_sha256: str
    parent_report_path: Path
    parent_report_sha256: str
    parent_checkpoint: dict[str, Any]
    parent_report: dict[str, Any]
    parent_step: int
    additional_steps: int


@dataclass(frozen=True, slots=True)
class EndpointContrastExclusion:
    """A row omitted from causal endpoint supervision without leaving base training."""

    sequence_id: str
    identity_id: str
    action: str
    target_sha256: str
    reason: str


@dataclass(frozen=True, slots=True)
class EndpointContrastGroup:
    """Rows whose only varying generation-request field is action."""

    key_sha256: str
    identity_id: str
    description: str
    entity_class: str
    view: str
    direction: str
    loop_mode: str
    frame_phases: tuple[float, ...]
    selected_indices: tuple[int, ...]
    sequence_ids: tuple[str, ...]
    actions: tuple[str, ...]
    target_sha256: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EndpointContrastPlan:
    """Deterministic unambiguous same-condition, multi-action endpoint plan."""

    groups: tuple[EndpointContrastGroup, ...]
    exclusions: tuple[EndpointContrastExclusion, ...]

    @property
    def selected_indices(self) -> tuple[int, ...]:
        return tuple(index for group in self.groups for index in group.selected_indices)


def run_tiny_overfit(
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    config: TinyOverfitConfig | None = None,
    sequence_ids: tuple[str, ...] | None = None,
    overwrite: bool = False,
    disk_guard: DiskGuard | None = None,
) -> TinyOverfitResult:
    return _run_tiny_overfit(
        manifest_path,
        output_directory,
        config=config,
        sequence_ids=sequence_ids,
        overwrite=overwrite,
        disk_guard=disk_guard,
        continuation=None,
    )


def continue_tiny_overfit(
    manifest_path: Path | str,
    parent_checkpoint_path: Path | str,
    parent_report_path: Path | str,
    output_directory: Path | str,
    *,
    expected_parent_checkpoint_sha256: str,
    expected_parent_report_sha256: str,
    additional_steps: int,
    config: TinyOverfitConfig | None = None,
    sequence_ids: tuple[str, ...] | None = None,
    disk_guard: DiskGuard | None = None,
) -> TinyOverfitResult:
    """Resume a verified overfit checkpoint into a new immutable experiment.

    ``additional_steps`` is deliberately required to be positive; a zero-step
    request is rejected instead of publishing a misleading duplicate lineage.
    The parent bundle is read only, and the child directory must not exist.
    """

    runtime = _require_torch()
    _positive_integer("additional_steps", additional_steps)
    manifest = Path(manifest_path).resolve()
    parent_checkpoint = Path(parent_checkpoint_path).resolve()
    parent_report = Path(parent_report_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing continuation output: {output}")
    if output == parent_checkpoint.parent or output == parent_report.parent:
        raise ValueError("continuation output must differ from the parent experiment directory")

    continuation, experiment, ordered_sequence_ids = _prepare_continuation(
        runtime,
        manifest=manifest,
        parent_checkpoint_path=parent_checkpoint,
        parent_report_path=parent_report,
        expected_parent_checkpoint_sha256=expected_parent_checkpoint_sha256,
        expected_parent_report_sha256=expected_parent_report_sha256,
        additional_steps=additional_steps,
        expected_config=config,
        expected_sequence_ids=sequence_ids,
    )
    if disk_guard is not None:
        disk_guard.require_capacity(label="overfit continuation staging")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".continuation.tmp",
            dir=output.parent,
        )
    ).resolve()
    try:
        staged_result = _run_tiny_overfit(
            manifest,
            staging,
            config=experiment,
            sequence_ids=ordered_sequence_ids,
            overwrite=False,
            disk_guard=disk_guard,
            continuation=continuation,
        )
    except BaseException:
        _remove_private_staging_directory(staging, output.parent)
        raise
    try:
        os.rename(staging, output)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to replace existing continuation output: {output}; "
            f"completed staging bundle remains at {staging}"
        ) from error
    except OSError as error:
        if output.exists():
            raise FileExistsError(
                f"Refusing to replace existing continuation output: {output}; "
                f"completed staging bundle remains at {staging}"
            ) from error
        raise
    return TinyOverfitResult(
        output_directory=output,
        report_path=output / staged_result.report_path.relative_to(staging),
        checkpoint_path=output / staged_result.checkpoint_path.relative_to(staging),
        sample_paths=tuple(
            output / path.relative_to(staging) for path in staged_result.sample_paths
        ),
        initial_loss=staged_result.initial_loss,
        final_loss=staged_result.final_loss,
        minimum_loss=staged_result.minimum_loss,
        report_sha256=staged_result.report_sha256,
    )


def _run_tiny_overfit(
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    config: TinyOverfitConfig | None,
    sequence_ids: tuple[str, ...] | None,
    overwrite: bool,
    disk_guard: DiskGuard | None,
    continuation: _ContinuationContext | None,
) -> TinyOverfitResult:
    """Train and sample a deliberately tiny memorization diagnostic.

    This is not an evaluation of general prompt generation. It verifies that the
    materialized corpus, conditioning encoder, rectified-flow objective, denoiser,
    optimizer, checkpoint, sampler, and RGBA export work together end to end.
    """

    runtime = _require_torch()
    experiment = config or TinyOverfitConfig()
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if continuation is not None and overwrite:
        raise ValueError("continuation artifacts are always no-clobber")
    start_step = continuation.parent_step if continuation is not None else 0
    if start_step >= experiment.steps:
        raise ValueError(
            f"cumulative steps must exceed parent step; got {experiment.steps} and {start_step}"
        )
    device = runtime.device(experiment.device)
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise ValueError("CUDA was requested but torch.cuda.is_available() is false")
    if experiment.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 overfit mode currently requires CUDA")

    manifest = Path(manifest_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"overfit output path is not a directory: {output}")
    report_path = output / "overfit-report.json"
    checkpoint_path = output / "checkpoint.pt"
    if not overwrite and (report_path.exists() or checkpoint_path.exists()):
        raise FileExistsError(f"Refusing to replace an existing overfit artifact in {output}")

    clips = load_materialized_training_clips(
        manifest,
        sequence_ids=sequence_ids,
        split="train",
        target_bucket=experiment.target_bucket,
        target_frames=experiment.target_frames,
    )
    batch = collate_materialized_clips(clips)
    sample_paths = tuple(_sample_path(output, clip.sequence_id) for clip in clips)
    if not overwrite:
        existing = tuple(path for path in sample_paths if path.exists())
        if existing:
            raise FileExistsError(f"Refusing to replace existing sample: {existing[0]}")
    _seed_all(experiment.seed, runtime, device=device)
    schema = replace(ConditioningSchema(), phase_bins=experiment.target_frames)
    model_config = PixelDiTConfig(
        height=experiment.target_bucket,
        width=experiment.target_bucket,
        num_frames=experiment.target_frames,
        patch_size=experiment.patch_size,
        model_dim=experiment.model_dim,
        depth=experiment.depth,
        num_heads=experiment.num_heads,
        condition_dim=experiment.condition_dim,
        phase_harmonics=4,
        conditioning=schema,
    )
    denoiser = FactorizedSpriteDiT(model_config).to(device=device)
    condition_encoder = SpriteConditionEncoder(
        schema,
        condition_dim=experiment.condition_dim,
        max_text_bytes=experiment.max_text_bytes,
    ).to(device=device)
    encoded = encode_generation_conditions(
        batch.requests,
        schema,
        max_text_bytes=experiment.max_text_bytes,
    )
    clean = runtime.from_numpy(batch.clean).to(device=device)
    phases = runtime.from_numpy(batch.frame_phases).to(device=device)
    endpoint_plan = _build_endpoint_contrast_plan(clips, batch.clean)
    endpoint_indices = endpoint_plan.selected_indices
    endpoint_clips = tuple(clips[index] for index in endpoint_indices)
    endpoint_encoded = (
        _slice_encoded_conditions(encoded, endpoint_indices) if endpoint_indices else None
    )
    if endpoint_indices:
        endpoint_index = runtime.as_tensor(endpoint_indices, dtype=runtime.long, device=device)
        endpoint_clean = clean.index_select(0, endpoint_index)
        endpoint_phases = phases.index_select(0, endpoint_index)
    else:
        endpoint_clean = None
        endpoint_phases = None
    parameters = tuple(denoiser.parameters()) + tuple(condition_encoder.parameters())
    optimizer = runtime.optim.AdamW(
        parameters,
        lr=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
    )
    generator = runtime.Generator(device=device).manual_seed(experiment.seed + 1)
    dtype = runtime.bfloat16 if experiment.precision == "bfloat16" else runtime.float32
    autocast_enabled = experiment.precision == "bfloat16"
    history: list[dict[str, float | int]] = (
        [dict(row) for row in continuation.parent_report["history"]]
        if continuation is not None
        else []
    )
    initial_training_loss: float | None = (
        float(continuation.parent_report["initial_training_loss"])
        if continuation is not None
        else None
    )
    final_training_loss = (
        float(continuation.parent_report["final_training_loss"])
        if continuation is not None
        else math.inf
    )
    minimum_loss = (
        float(continuation.parent_report["minimum_loss"]) if continuation is not None else math.inf
    )
    diagnostic_generator = runtime.Generator(device=device).manual_seed(experiment.seed + 3)
    diagnostic_flow = sample_rectified_flow_batch(clean, generator=diagnostic_generator)
    endpoint_diagnostic_generator = runtime.Generator(device=device).manual_seed(
        experiment.seed + 4
    )
    endpoint_training_generator = runtime.Generator(device=device).manual_seed(experiment.seed + 5)
    if continuation is not None:
        _restore_continuation_state(
            runtime,
            continuation,
            denoiser=denoiser,
            condition_encoder=condition_encoder,
            optimizer=optimizer,
            training_generator=generator,
            endpoint_training_generator=endpoint_training_generator,
            device=device,
        )
    if endpoint_clean is not None:
        endpoint_diagnostic_noise = _contrast_grouped_noise(
            runtime,
            endpoint_clean,
            endpoint_plan,
            generator=endpoint_diagnostic_generator,
        )
        endpoint_diagnostic_flow = sample_rectified_flow_batch(
            endpoint_clean,
            noise=endpoint_diagnostic_noise,
            timesteps=runtime.ones(
                (endpoint_clean.shape[0],),
                device=device,
                dtype=endpoint_clean.dtype,
            ),
        )
    else:
        endpoint_diagnostic_noise = None
        endpoint_diagnostic_flow = None
    if continuation is not None:
        _verify_continuation_diagnostics(
            runtime,
            continuation,
            diagnostic_flow=diagnostic_flow,
            diagnostic_generator=diagnostic_generator,
            endpoint_diagnostic_noise=endpoint_diagnostic_noise,
            endpoint_diagnostic_generator=endpoint_diagnostic_generator,
        )

    def current_loss(
        flow: Any,
        conditions: EncodedConditionBatch = encoded,
        *,
        reference_clean: Any = clean,
        frame_phases: Any = phases,
    ) -> Any:
        context, context_mask = condition_encoder(conditions)
        with runtime.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=autocast_enabled,
        ):
            predicted = denoiser(
                flow.noisy,
                flow.timesteps,
                context,
                conditioning_mask=context_mask,
                frame_phase=frame_phases,
            )
            squared = (predicted - flow.target_velocity).float().square()
            if experiment.alpha_channel_weight != 1.0:
                channel_weights = runtime.ones(
                    (1, 1, squared.shape[2], 1, 1),
                    device=squared.device,
                    dtype=squared.dtype,
                )
                channel_weights[:, :, 3] = experiment.alpha_channel_weight
                squared = squared * channel_weights
            alpha = ((reference_clean[:, :, 3:4] + 1.0) * 0.5).clamp(0.0, 1.0)
            return (squared * (1.0 + experiment.foreground_weight * alpha)).mean()

    denoiser.eval()
    condition_encoder.eval()
    with runtime.no_grad():
        initial_loss = float(current_loss(diagnostic_flow).detach().cpu())
        initial_endpoint_loss = (
            float(
                current_loss(
                    endpoint_diagnostic_flow,
                    endpoint_encoded,
                    reference_clean=endpoint_clean,
                    frame_phases=endpoint_phases,
                )
                .detach()
                .cpu()
            )
            if endpoint_diagnostic_flow is not None and endpoint_encoded is not None
            else None
        )
    if not math.isfinite(initial_loss):
        raise RuntimeError("initial fixed diagnostic loss is non-finite")
    if initial_endpoint_loss is not None and not math.isfinite(initial_endpoint_loss):
        raise RuntimeError("initial matched-endpoint diagnostic loss is non-finite")
    resume_initial_loss = initial_loss if continuation is not None else None
    resume_initial_endpoint_loss = initial_endpoint_loss if continuation is not None else None
    if continuation is not None:
        _require_replayed_metric(
            "parent final_loss",
            initial_loss,
            continuation.parent_report["final_loss"],
        )
        _require_replayed_optional_metric(
            "parent matched_endpoint_final_loss",
            initial_endpoint_loss,
            continuation.parent_report["matched_endpoint_final_loss"],
        )
        initial_loss = float(continuation.parent_report["initial_loss"])
        initial_endpoint_loss = _optional_finite_float(
            continuation.parent_report["matched_endpoint_initial_loss"],
            "parent_report.matched_endpoint_initial_loss",
        )
    denoiser.train()
    condition_encoder.train()
    for step in range(start_step, experiment.steps):
        optimizer.zero_grad(set_to_none=True)
        flow = sample_rectified_flow_batch(clean, generator=generator)
        base_loss = current_loss(flow)
        endpoint_loss = None
        if (
            endpoint_clean is not None
            and endpoint_encoded is not None
            and experiment.matched_endpoint_weight > 0
        ):
            endpoint_noise = _contrast_grouped_noise(
                runtime,
                endpoint_clean,
                endpoint_plan,
                generator=endpoint_training_generator,
            )
            endpoint_flow = sample_rectified_flow_batch(
                endpoint_clean,
                noise=endpoint_noise,
                timesteps=runtime.ones(
                    (endpoint_clean.shape[0],),
                    device=device,
                    dtype=endpoint_clean.dtype,
                ),
            )
            endpoint_loss = current_loss(
                endpoint_flow,
                endpoint_encoded,
                reference_clean=endpoint_clean,
                frame_phases=endpoint_phases,
            )
        loss = (
            base_loss
            if endpoint_loss is None
            else base_loss + experiment.matched_endpoint_weight * endpoint_loss
        )
        if not bool(runtime.isfinite(loss)):
            raise RuntimeError(f"training loss became non-finite at step {step + 1}")
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        initial_training_loss = value if initial_training_loss is None else initial_training_loss
        final_training_loss = value
        minimum_loss = min(minimum_loss, value)
        if step == 0 or (step + 1) % experiment.log_every == 0 or step + 1 == experiment.steps:
            history_row: dict[str, float | int] = {
                "base_loss": float(base_loss.detach().cpu()),
                "loss": value,
                "step": step + 1,
            }
            if endpoint_loss is not None:
                history_row["matched_endpoint_loss"] = float(endpoint_loss.detach().cpu())
            history.append(history_row)

    denoiser.eval()
    condition_encoder.eval()
    with runtime.no_grad():
        final_loss = float(current_loss(diagnostic_flow).detach().cpu())
        final_endpoint_loss = (
            float(
                current_loss(
                    endpoint_diagnostic_flow,
                    endpoint_encoded,
                    reference_clean=endpoint_clean,
                    frame_phases=endpoint_phases,
                )
                .detach()
                .cpu()
            )
            if endpoint_diagnostic_flow is not None and endpoint_encoded is not None
            else None
        )
    if not math.isfinite(final_loss):
        raise RuntimeError("final fixed diagnostic loss is non-finite")
    if final_endpoint_loss is not None and not math.isfinite(final_endpoint_loss):
        raise RuntimeError("final matched-endpoint diagnostic loss is non-finite")
    action_permuted_final_loss: float | None = None
    endpoint_action_permuted_final_loss: float | None = None
    permutation_rows = _endpoint_action_permutation_rows(endpoint_plan)
    permuted = _permute_actions_for_endpoint_plan(encoded, endpoint_plan)
    action_swap_matrix: list[dict[str, Any]] = []
    if permuted is not None:
        with runtime.no_grad():
            action_permuted_final_loss = float(
                current_loss(diagnostic_flow, permuted).detach().cpu()
            )
            if endpoint_diagnostic_flow is not None:
                endpoint_permuted = _slice_encoded_conditions(permuted, endpoint_indices)
                endpoint_action_permuted_final_loss = float(
                    current_loss(
                        endpoint_diagnostic_flow,
                        endpoint_permuted,
                        reference_clean=endpoint_clean,
                        frame_phases=endpoint_phases,
                    )
                    .detach()
                    .cpu()
                )
                action_swap_matrix = _endpoint_action_swap_matrix(
                    current_loss=current_loss,
                    flow=endpoint_diagnostic_flow,
                    conditions=endpoint_encoded,
                    reference_clean=endpoint_clean,
                    frame_phases=endpoint_phases,
                    endpoint_plan=endpoint_plan,
                    clips=endpoint_clips,
                )
    if action_permuted_final_loss is not None and not math.isfinite(action_permuted_final_loss):
        raise RuntimeError("action-permuted fixed diagnostic loss is non-finite")
    if endpoint_action_permuted_final_loss is not None and not math.isfinite(
        endpoint_action_permuted_final_loss
    ):
        raise RuntimeError("action-permuted matched-endpoint loss is non-finite")
    sample_generator = runtime.Generator(device=device).manual_seed(experiment.seed + 2)
    noise = _identity_grouped_noise(
        runtime,
        clean,
        clips,
        generator=sample_generator,
    )
    if continuation is not None and not runtime.equal(
        sample_generator.get_state(),
        continuation.parent_checkpoint["rng_state"]["sample_generator"],
    ):
        raise OverfitContinuationContractError(
            "recreated matched sample generator state differs from parent"
        )
    with runtime.no_grad():
        context, context_mask = condition_encoder(encoded)
        sampled = euler_sample_velocity_model(
            denoiser,
            noise,
            steps=experiment.sample_steps,
            conditioning=context,
            conditioning_mask=context_mask,
            frame_phase=phases,
        )
    sampled_rgba = np.stack(
        [model_to_rgba_uint8(clip) for clip in sampled.float().cpu().numpy()],
        axis=0,
    )
    output.mkdir(parents=True, exist_ok=True)
    for path, array in zip(sample_paths, sampled_rgba, strict=True):
        _atomic_numpy(path, array, overwrite=overwrite, disk_guard=disk_guard)

    input_clips = _input_clip_rows(clips)
    runtime_facts = _runtime_facts(runtime, device)
    continuation_lineage = (
        {
            "additional_steps": continuation.additional_steps,
            "cumulative_step": experiment.steps,
            "parent_checkpoint_path": str(continuation.parent_checkpoint_path),
            "parent_checkpoint_sha256": continuation.parent_checkpoint_sha256,
            "parent_report_path": str(continuation.parent_report_path),
            "parent_report_sha256": continuation.parent_report_sha256,
            "parent_step": continuation.parent_step,
            "resume_fixed_diagnostic_loss": resume_initial_loss,
            "resume_matched_endpoint_diagnostic_loss": resume_initial_endpoint_loss,
            "state_contract": (
                "model, condition encoder, optimizer, Python/NumPy/Torch global RNG, "
                "and dedicated training RNG states restored from the hash-verified parent"
            ),
        }
        if continuation is not None
        else None
    )
    checkpoint = {
        "action_permutation_rows": permutation_rows,
        "action_permuted_final_loss": action_permuted_final_loss,
        "action_swap_matrix": action_swap_matrix,
        "condition_encoder": condition_encoder.state_dict(),
        "config": asdict(experiment),
        "continuation": continuation_lineage,
        "denoiser": denoiser.state_dict(),
        "input_clips": input_clips,
        "matched_endpoint_contrast_plan": asdict(endpoint_plan),
        "matched_endpoint_action_permuted_final_loss": endpoint_action_permuted_final_loss,
        "matched_endpoint_final_loss": final_endpoint_loss,
        "matched_endpoint_initial_loss": initial_endpoint_loss,
        "matched_endpoint_sequence_ids": [clip.sequence_id for clip in endpoint_clips],
        "materialization_manifest_sha256": clips[0].materialization_manifest_sha256,
        "model_config": asdict(model_config),
        "optimizer": optimizer.state_dict(),
        "rng_state": {
            "cuda": runtime.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "diagnostic_generator": diagnostic_generator.get_state(),
            "endpoint_diagnostic_generator": endpoint_diagnostic_generator.get_state(),
            "endpoint_training_generator": endpoint_training_generator.get_state(),
            "numpy": _numpy_rng_state(),
            "python": random.getstate(),
            "sample_generator": sample_generator.get_state(),
            "torch_cpu": runtime.get_rng_state(),
            "training_generator": generator.get_state(),
        },
        "runtime": runtime_facts,
        "diagnostic_tensor_sha256": {
            "ordinary_noise": _tensor_sha256(runtime, diagnostic_flow.noise),
            "ordinary_timesteps": _tensor_sha256(runtime, diagnostic_flow.timesteps),
            "endpoint_noise": (
                _tensor_sha256(runtime, endpoint_diagnostic_noise)
                if endpoint_diagnostic_noise is not None
                else None
            ),
        },
        "sample_noise_contract": (
            "one deterministic initial-noise tensor per identity; all actions for an "
            "identity share it so action comparisons use matched stochastic input"
        ),
        "training_conditioning_contract": (
            "ordinary stochastic rectified-flow loss plus matched pure-noise t=1 loss "
            "for target-distinct, unambiguous rows whose identity and every non-action "
            "condition match; "
            "endpoint noise is shared within each causal action contrast group; "
            f"alpha-channel residual multiplier is {experiment.alpha_channel_weight:g}"
        ),
        "sequence_ids": batch.sequence_ids,
        "source_snapshot_canonical_sha256": clips[0].source_snapshot_canonical_sha256,
        "source_snapshot_manifest_sha256": clips[0].source_snapshot_manifest_sha256,
        "step": experiment.steps,
    }
    _atomic_torch_save(
        checkpoint_path,
        checkpoint,
        overwrite=overwrite,
        disk_guard=disk_guard,
    )
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    assert initial_training_loss is not None
    report = {
        "action_permutation_rows": permutation_rows,
        "artifact_kind": "tiny_corpus_memorization_diagnostic",
        "action_condition_loss_delta": (
            action_permuted_final_loss - final_loss
            if action_permuted_final_loss is not None
            else None
        ),
        "action_permuted_final_loss": action_permuted_final_loss,
        "action_swap_matrix": action_swap_matrix,
        "checkpoint": {
            "file_sha256": checkpoint_sha256,
            "path": checkpoint_path.relative_to(output).as_posix(),
        },
        "config": asdict(experiment),
        "continuation": continuation_lineage,
        "diagnostic_loss_contract": (
            "initial_loss and final_loss use the same fixed noise/timestep batch; "
            "history losses use fresh stochastic flow batches; continued runs preserve "
            "the original step-zero initial_loss and record the parent replay value in "
            "continuation.resume_fixed_diagnostic_loss"
        ),
        "diagnostic_tensor_sha256": {
            "ordinary_noise": _tensor_sha256(runtime, diagnostic_flow.noise),
            "ordinary_timesteps": _tensor_sha256(runtime, diagnostic_flow.timesteps),
            "endpoint_noise": (
                _tensor_sha256(runtime, endpoint_diagnostic_noise)
                if endpoint_diagnostic_noise is not None
                else None
            ),
        },
        "final_loss": final_loss,
        "final_training_loss": final_training_loss,
        "history": history,
        "initial_loss": initial_loss,
        "initial_training_loss": initial_training_loss,
        "input_clips": input_clips,
        "matched_endpoint_contrast_plan": asdict(endpoint_plan),
        "matched_endpoint_action_loss_delta": (
            endpoint_action_permuted_final_loss - final_endpoint_loss
            if endpoint_action_permuted_final_loss is not None and final_endpoint_loss is not None
            else None
        ),
        "matched_endpoint_action_permuted_final_loss": endpoint_action_permuted_final_loss,
        "matched_endpoint_final_loss": final_endpoint_loss,
        "matched_endpoint_initial_loss": initial_endpoint_loss,
        "matched_endpoint_sequence_ids": [clip.sequence_id for clip in endpoint_clips],
        "materialization_manifest_file_sha256": clips[0].materialization_manifest_sha256,
        "materialization_manifest_path": str(manifest),
        "minimum_loss": minimum_loss,
        "reproducibility_claim": (
            "Python, NumPy, Torch, and dedicated generators are seeded and their "
            "post-run states are checkpointed; bit-exact replay is claimed only when "
            "deterministic algorithms are enabled on the same torch/CUDA stack"
        ),
        "runtime": runtime_facts,
        "sample_files": [
            {
                "file_sha256": _sha256_file(path),
                "path": path.relative_to(output).as_posix(),
                "sequence_id": clip.sequence_id,
            }
            for path, clip in zip(sample_paths, clips, strict=True)
        ],
        "sequence_ids": list(batch.sequence_ids),
        "source_snapshot_canonical_sha256": clips[0].source_snapshot_canonical_sha256,
        "source_snapshot_manifest_sha256": clips[0].source_snapshot_manifest_sha256,
        "training_claim": (
            "pipeline overfit/memorization check only; not evidence of open-vocabulary "
            "text-to-sprite generalization"
        ),
        "training_conditioning_contract": (
            "ordinary stochastic rectified-flow loss plus a weighted matched pure-noise "
            "endpoint loss only for target-distinct, unambiguous groups with identical "
            "non-action conditions; zero weight retains fixed endpoint diagnostics but skips the "
            f"training forward. Alpha-channel residual multiplier is "
            f"{experiment.alpha_channel_weight:g}. Report action steering only if "
            "per-swap deltas and "
            "matched-noise samples support it"
        ),
    }
    report_bytes = (
        json.dumps(report, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_bytes(
        report_path,
        report_bytes,
        overwrite=overwrite,
        disk_guard=disk_guard,
        label="overfit report",
    )
    return TinyOverfitResult(
        output_directory=output,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        sample_paths=sample_paths,
        initial_loss=initial_loss,
        final_loss=final_loss,
        minimum_loss=minimum_loss,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
    )


def _prepare_continuation(
    runtime: Any,
    *,
    manifest: Path,
    parent_checkpoint_path: Path,
    parent_report_path: Path,
    expected_parent_checkpoint_sha256: str,
    expected_parent_report_sha256: str,
    additional_steps: int,
    expected_config: TinyOverfitConfig | None,
    expected_sequence_ids: tuple[str, ...] | None,
) -> tuple[_ContinuationContext, TinyOverfitConfig, tuple[str, ...]]:
    expected_checkpoint_digest = _required_digest(
        expected_parent_checkpoint_sha256,
        "expected_parent_checkpoint_sha256",
    )
    expected_report_digest = _required_digest(
        expected_parent_report_sha256,
        "expected_parent_report_sha256",
    )
    if not parent_checkpoint_path.is_file():
        raise FileNotFoundError(f"parent checkpoint does not exist: {parent_checkpoint_path}")
    if not parent_report_path.is_file():
        raise FileNotFoundError(f"parent report does not exist: {parent_report_path}")
    checkpoint_digest = _sha256_file(parent_checkpoint_path)
    if checkpoint_digest != expected_checkpoint_digest:
        raise HashMismatch(
            "Parent checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_digest}, received {checkpoint_digest}"
        )
    report_bytes = parent_report_path.read_bytes()
    report_digest = hashlib.sha256(report_bytes).hexdigest()
    if report_digest != expected_report_digest:
        raise HashMismatch(
            "Parent report SHA-256 mismatch: "
            f"expected {expected_report_digest}, received {report_digest}"
        )
    try:
        parent_report_raw = json.loads(report_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OverfitContinuationContractError("parent report is not valid UTF-8 JSON") from error
    if not isinstance(parent_report_raw, Mapping):
        raise OverfitContinuationContractError("parent report root must be an object")
    parent_report = dict(parent_report_raw)
    try:
        parent_checkpoint_raw = runtime.load(
            parent_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise OverfitContinuationContractError(
            "parent checkpoint could not be loaded with torch.load(weights_only=True)"
        ) from error
    if not isinstance(parent_checkpoint_raw, Mapping):
        raise OverfitContinuationContractError("parent checkpoint root must be a mapping")
    parent_checkpoint = dict(parent_checkpoint_raw)

    checkpoint_record = _required_mapping(
        parent_report.get("checkpoint"), "parent_report.checkpoint"
    )
    declared_checkpoint_digest = _required_digest(
        checkpoint_record.get("file_sha256"),
        "parent_report.checkpoint.file_sha256",
    )
    if declared_checkpoint_digest != checkpoint_digest:
        raise HashMismatch(
            "parent report names a different checkpoint SHA-256: "
            f"{declared_checkpoint_digest} != {checkpoint_digest}"
        )
    relative_checkpoint = Path(
        _required_nonempty_string(
            checkpoint_record.get("path"),
            "parent_report.checkpoint.path",
        )
    )
    if relative_checkpoint.is_absolute() or ".." in relative_checkpoint.parts:
        raise OverfitContinuationContractError(
            "parent report checkpoint path must be safe and relative"
        )
    declared_checkpoint_path = (parent_report_path.parent / relative_checkpoint).resolve()
    if declared_checkpoint_path != parent_checkpoint_path:
        raise OverfitContinuationContractError(
            "parent report checkpoint path does not identify the supplied checkpoint"
        )

    checkpoint_config = _tiny_overfit_config(
        parent_checkpoint.get("config"),
        "parent_checkpoint.config",
    )
    report_config = _tiny_overfit_config(
        parent_report.get("config"),
        "parent_report.config",
    )
    if checkpoint_config != report_config:
        raise OverfitContinuationContractError("parent checkpoint/report configs differ")
    parent_step = _required_positive_integer(
        parent_checkpoint.get("step"),
        "parent_checkpoint.step",
    )
    if parent_step != checkpoint_config.steps:
        raise OverfitContinuationContractError(
            "parent checkpoint step must equal parent config.steps; "
            f"got {parent_step} and {checkpoint_config.steps}"
        )
    cumulative_steps = parent_step + additional_steps
    experiment = replace(checkpoint_config, steps=cumulative_steps)
    if expected_config is not None and expected_config != experiment:
        differences = {
            field.name: (getattr(experiment, field.name), getattr(expected_config, field.name))
            for field in fields(TinyOverfitConfig)
            if getattr(experiment, field.name) != getattr(expected_config, field.name)
        }
        raise OverfitContinuationContractError(
            f"requested continuation config differs from the derived parent config: {differences!r}"
        )

    checkpoint_sequence_ids = _required_sequence_ids(
        parent_checkpoint.get("sequence_ids"),
        "parent_checkpoint.sequence_ids",
    )
    report_sequence_ids = _required_sequence_ids(
        parent_report.get("sequence_ids"),
        "parent_report.sequence_ids",
    )
    if checkpoint_sequence_ids != report_sequence_ids:
        raise OverfitContinuationContractError("parent checkpoint/report sequence order differs")
    if (
        expected_sequence_ids is not None
        and tuple(expected_sequence_ids) != checkpoint_sequence_ids
    ):
        raise OverfitContinuationContractError(
            "requested sequence IDs must exactly match the ordered parent sequence IDs"
        )

    manifest_bytes = manifest.read_bytes()
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    checkpoint_manifest_digest = _required_digest(
        parent_checkpoint.get("materialization_manifest_sha256"),
        "parent_checkpoint.materialization_manifest_sha256",
    )
    report_manifest_digest = _required_digest(
        parent_report.get("materialization_manifest_file_sha256"),
        "parent_report.materialization_manifest_file_sha256",
    )
    if manifest_digest != checkpoint_manifest_digest or manifest_digest != report_manifest_digest:
        raise HashMismatch("materialization manifest SHA-256 does not match the parent bundle")
    reported_manifest = Path(
        _required_nonempty_string(
            parent_report.get("materialization_manifest_path"),
            "parent_report.materialization_manifest_path",
        )
    ).resolve()
    if reported_manifest != manifest:
        raise OverfitContinuationContractError(
            "supplied materialization manifest path differs from the parent report"
        )

    clips = load_materialized_training_clips(
        manifest,
        sequence_ids=checkpoint_sequence_ids,
        split="train",
        target_bucket=checkpoint_config.target_bucket,
        target_frames=checkpoint_config.target_frames,
    )
    batch = collate_materialized_clips(clips)
    if batch.sequence_ids != checkpoint_sequence_ids:
        raise OverfitContinuationContractError("materialized clip order differs from the parent")
    input_clips = _input_clip_rows(clips)
    if parent_checkpoint.get("input_clips") != input_clips:
        raise OverfitContinuationContractError(
            "parent checkpoint input clip hashes or metadata differ from materialization"
        )
    report_input_clips = json.loads(
        json.dumps(input_clips, ensure_ascii=False, separators=(",", ":"))
    )
    if parent_report.get("input_clips") != report_input_clips:
        raise OverfitContinuationContractError(
            "parent report input clip hashes or metadata differ from materialization"
        )
    canonical_sha = clips[0].source_snapshot_canonical_sha256
    snapshot_manifest_sha = clips[0].source_snapshot_manifest_sha256
    for source, label in (
        (parent_checkpoint, "parent_checkpoint"),
        (parent_report, "parent_report"),
    ):
        if (
            _required_digest(
                source.get("source_snapshot_canonical_sha256"),
                f"{label}.source_snapshot_canonical_sha256",
            )
            != canonical_sha
        ):
            raise HashMismatch("source snapshot canonical SHA-256 differs from materialization")
        if (
            _required_digest(
                source.get("source_snapshot_manifest_sha256"),
                f"{label}.source_snapshot_manifest_sha256",
            )
            != snapshot_manifest_sha
        ):
            raise HashMismatch("source snapshot manifest SHA-256 differs from materialization")

    schema = replace(ConditioningSchema(), phase_bins=checkpoint_config.target_frames)
    model_config = PixelDiTConfig(
        height=checkpoint_config.target_bucket,
        width=checkpoint_config.target_bucket,
        num_frames=checkpoint_config.target_frames,
        patch_size=checkpoint_config.patch_size,
        model_dim=checkpoint_config.model_dim,
        depth=checkpoint_config.depth,
        num_heads=checkpoint_config.num_heads,
        condition_dim=checkpoint_config.condition_dim,
        phase_harmonics=4,
        conditioning=schema,
    )
    if parent_checkpoint.get("model_config") != asdict(model_config):
        raise OverfitContinuationContractError(
            "parent checkpoint architecture does not match its TinyOverfitConfig"
        )
    endpoint_plan = _build_endpoint_contrast_plan(clips, batch.clean)
    endpoint_plan_payload = asdict(endpoint_plan)
    if parent_checkpoint.get("matched_endpoint_contrast_plan") != endpoint_plan_payload:
        raise OverfitContinuationContractError(
            "parent checkpoint endpoint contrast plan differs from materialization"
        )
    report_endpoint_plan = json.loads(
        json.dumps(endpoint_plan_payload, ensure_ascii=False, separators=(",", ":"))
    )
    if parent_report.get("matched_endpoint_contrast_plan") != report_endpoint_plan:
        raise OverfitContinuationContractError(
            "parent report endpoint contrast plan differs from materialization"
        )
    expected_endpoint_ids = tuple(
        clips[index].sequence_id for index in endpoint_plan.selected_indices
    )
    if (
        _required_sequence_ids(
            parent_checkpoint.get("matched_endpoint_sequence_ids"),
            "parent_checkpoint.matched_endpoint_sequence_ids",
            allow_empty=True,
        )
        != expected_endpoint_ids
    ):
        raise OverfitContinuationContractError("parent checkpoint endpoint sequence IDs differ")
    if (
        _required_sequence_ids(
            parent_report.get("matched_endpoint_sequence_ids"),
            "parent_report.matched_endpoint_sequence_ids",
            allow_empty=True,
        )
        != expected_endpoint_ids
    ):
        raise OverfitContinuationContractError("parent report endpoint sequence IDs differ")

    _validate_parent_states(
        runtime,
        checkpoint=parent_checkpoint,
        report=parent_report,
        config=checkpoint_config,
        model_config=model_config,
    )
    context = _ContinuationContext(
        parent_checkpoint_path=parent_checkpoint_path,
        parent_checkpoint_sha256=checkpoint_digest,
        parent_report_path=parent_report_path,
        parent_report_sha256=report_digest,
        parent_checkpoint=parent_checkpoint,
        parent_report=parent_report,
        parent_step=parent_step,
        additional_steps=additional_steps,
    )
    return context, experiment, checkpoint_sequence_ids


def _validate_parent_states(
    runtime: Any,
    *,
    checkpoint: dict[str, Any],
    report: dict[str, Any],
    config: TinyOverfitConfig,
    model_config: PixelDiTConfig,
) -> None:
    checkpoint_runtime = dict(
        _required_mapping(checkpoint.get("runtime"), "parent_checkpoint.runtime")
    )
    report_runtime = dict(_required_mapping(report.get("runtime"), "parent_report.runtime"))
    if checkpoint_runtime != report_runtime:
        raise OverfitContinuationContractError("parent checkpoint/report runtime facts differ")
    device = runtime.device(config.device)
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise ValueError("CUDA was requested by the parent but torch.cuda.is_available() is false")
    if config.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 continuation requires the parent's CUDA device")
    current_runtime = _runtime_facts(runtime, device)
    for name in ("cuda_version", "cudnn_version", "device", "device_name", "torch_version"):
        if checkpoint_runtime.get(name) != current_runtime[name]:
            raise OverfitContinuationContractError(
                f"current runtime {name} differs from parent: "
                f"{current_runtime[name]!r} != {checkpoint_runtime.get(name)!r}"
            )
    if not isinstance(checkpoint_runtime.get("deterministic_algorithms_enabled"), bool):
        raise OverfitContinuationContractError(
            "parent runtime deterministic_algorithms_enabled must be a bool"
        )

    denoiser = FactorizedSpriteDiT(model_config)
    encoder = SpriteConditionEncoder(
        model_config.conditioning,
        condition_dim=config.condition_dim,
        max_text_bytes=config.max_text_bytes,
    )
    denoiser_state = _required_mapping(checkpoint.get("denoiser"), "parent_checkpoint.denoiser")
    encoder_state = _required_mapping(
        checkpoint.get("condition_encoder"),
        "parent_checkpoint.condition_encoder",
    )
    try:
        denoiser.load_state_dict(denoiser_state, strict=True)
        encoder.load_state_dict(encoder_state, strict=True)
    except (RuntimeError, ValueError, TypeError) as error:
        raise OverfitContinuationContractError(
            "parent model state does not exactly match its stored architecture"
        ) from error
    optimizer = runtime.optim.AdamW(
        tuple(denoiser.parameters()) + tuple(encoder.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_state = _required_mapping(
        checkpoint.get("optimizer"),
        "parent_checkpoint.optimizer",
    )
    try:
        optimizer.load_state_dict(optimizer_state)
    except (RuntimeError, ValueError, TypeError, KeyError) as error:
        raise OverfitContinuationContractError(
            "parent optimizer state does not match the model parameter groups"
        ) from error

    rng = _required_mapping(checkpoint.get("rng_state"), "parent_checkpoint.rng_state")
    required_rng = {
        "cuda",
        "diagnostic_generator",
        "endpoint_diagnostic_generator",
        "endpoint_training_generator",
        "numpy",
        "python",
        "sample_generator",
        "torch_cpu",
        "training_generator",
    }
    if set(rng) != required_rng:
        raise OverfitContinuationContractError(
            "parent RNG state fields differ: "
            f"expected {sorted(required_rng)!r}, got {sorted(rng)!r}"
        )
    _python_rng_state(rng["python"])
    _numpy_rng_tuple(rng["numpy"])
    _validated_generator_state(runtime, rng["torch_cpu"], device=runtime.device("cpu"))
    for name in (
        "diagnostic_generator",
        "endpoint_diagnostic_generator",
        "endpoint_training_generator",
        "sample_generator",
        "training_generator",
    ):
        _validated_generator_state(runtime, rng[name], device=device)
    if device.type == "cuda":
        _validated_generator_state(runtime, rng["cuda"], device=device)
    elif rng["cuda"] is not None:
        raise OverfitContinuationContractError("CPU parent checkpoint must store cuda RNG as null")

    diagnostic_hashes = _required_mapping(
        checkpoint.get("diagnostic_tensor_sha256"),
        "parent_checkpoint.diagnostic_tensor_sha256",
    )
    if set(diagnostic_hashes) != {"endpoint_noise", "ordinary_noise", "ordinary_timesteps"}:
        raise OverfitContinuationContractError("parent diagnostic tensor hash fields differ")
    for name in ("ordinary_noise", "ordinary_timesteps"):
        _required_digest(diagnostic_hashes.get(name), f"parent diagnostic {name}")
    endpoint_hash = diagnostic_hashes.get("endpoint_noise")
    if endpoint_hash is not None:
        _required_digest(endpoint_hash, "parent diagnostic endpoint_noise")
    if report.get("diagnostic_tensor_sha256") != diagnostic_hashes:
        raise OverfitContinuationContractError(
            "parent checkpoint/report diagnostic tensor hashes differ"
        )
    history = report.get("history")
    if not isinstance(history, list) or not history:
        raise OverfitContinuationContractError("parent report history must be nonempty")
    history_steps = [
        _required_positive_integer(row.get("step"), f"parent_report.history[{index}].step")
        if isinstance(row, Mapping)
        else (_raise_contract(f"parent_report.history[{index}] must be an object"))
        for index, row in enumerate(history)
    ]
    if history_steps != sorted(set(history_steps)) or history_steps[-1] != config.steps:
        raise OverfitContinuationContractError(
            "parent history steps must be unique, increasing, and end at config.steps"
        )
    for name in (
        "initial_loss",
        "final_loss",
        "initial_training_loss",
        "final_training_loss",
        "minimum_loss",
    ):
        _finite_float(report.get(name), f"parent_report.{name}")
    _optional_finite_float(
        report.get("matched_endpoint_initial_loss"),
        "parent_report.matched_endpoint_initial_loss",
    )
    _optional_finite_float(
        report.get("matched_endpoint_final_loss"),
        "parent_report.matched_endpoint_final_loss",
    )


def _restore_continuation_state(
    runtime: Any,
    continuation: _ContinuationContext,
    *,
    denoiser: Any,
    condition_encoder: Any,
    optimizer: Any,
    training_generator: Any,
    endpoint_training_generator: Any,
    device: Any,
) -> None:
    checkpoint = continuation.parent_checkpoint
    try:
        denoiser.load_state_dict(checkpoint["denoiser"], strict=True)
        condition_encoder.load_state_dict(checkpoint["condition_encoder"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
    except (RuntimeError, ValueError, TypeError, KeyError) as error:
        raise OverfitContinuationContractError("failed to restore parent training state") from error
    rng = checkpoint["rng_state"]
    training_generator.set_state(rng["training_generator"])
    endpoint_training_generator.set_state(rng["endpoint_training_generator"])
    random.setstate(_python_rng_state(rng["python"]))
    np.random.set_state(_numpy_rng_tuple(rng["numpy"]))
    runtime.set_rng_state(rng["torch_cpu"])
    if device.type == "cuda":
        runtime.cuda.set_rng_state(rng["cuda"], device=device)
    runtime.use_deterministic_algorithms(
        continuation.parent_checkpoint["runtime"]["deterministic_algorithms_enabled"]
    )


def _verify_continuation_diagnostics(
    runtime: Any,
    continuation: _ContinuationContext,
    *,
    diagnostic_flow: Any,
    diagnostic_generator: Any,
    endpoint_diagnostic_noise: Any,
    endpoint_diagnostic_generator: Any,
) -> None:
    checkpoint = continuation.parent_checkpoint
    expected = checkpoint["diagnostic_tensor_sha256"]
    actual = {
        "ordinary_noise": _tensor_sha256(runtime, diagnostic_flow.noise),
        "ordinary_timesteps": _tensor_sha256(runtime, diagnostic_flow.timesteps),
        "endpoint_noise": (
            _tensor_sha256(runtime, endpoint_diagnostic_noise)
            if endpoint_diagnostic_noise is not None
            else None
        ),
    }
    if actual != expected:
        raise OverfitContinuationContractError(
            f"recreated diagnostic tensors differ from parent hashes: {actual!r} != {expected!r}"
        )
    rng = checkpoint["rng_state"]
    if not runtime.equal(diagnostic_generator.get_state(), rng["diagnostic_generator"]):
        raise OverfitContinuationContractError("ordinary diagnostic generator state differs")
    if not runtime.equal(
        endpoint_diagnostic_generator.get_state(),
        rng["endpoint_diagnostic_generator"],
    ):
        raise OverfitContinuationContractError("endpoint diagnostic generator state differs")


def _input_clip_rows(clips: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "identity_id": clip.identity_id,
            "intro_loop_projection": (
                asdict(clip.intro_loop_projection)
                if clip.intro_loop_projection is not None
                else None
            ),
            "materialized_array_sha256": clip.source_array_sha256,
            "materialized_file_sha256": clip.source_file_sha256,
            "request": asdict(clip.request),
            "sequence_id": clip.sequence_id,
            "source_blob_sha256": list(clip.source_blob_sha256),
            "source_duration_ms": list(clip.source_duration_ms),
            "source_id": clip.source_id,
            "source_loop_mode": clip.source_loop_mode,
            "temporal_duration_method": clip.temporal_duration_method,
            "temporal_selection": (
                asdict(clip.temporal_selection) if clip.temporal_selection is not None else None
            ),
            "training_duration_ms": list(clip.duration_ms),
        }
        for clip in clips
    ]


def _tiny_overfit_config(value: object, name: str) -> TinyOverfitConfig:
    raw = _required_mapping(value, name)
    expected_fields = {field.name for field in fields(TinyOverfitConfig)}
    if set(raw) != expected_fields:
        raise OverfitContinuationContractError(
            f"{name} fields differ: expected {sorted(expected_fields)!r}, got {sorted(raw)!r}"
        )
    try:
        config = TinyOverfitConfig(**dict(raw))
    except (TypeError, ValueError) as error:
        raise OverfitContinuationContractError(f"{name} is invalid") from error
    if asdict(config) != dict(raw):
        raise OverfitContinuationContractError(f"{name} does not round-trip exactly")
    return config


def _required_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OverfitContinuationContractError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise OverfitContinuationContractError(f"{name} keys must be strings")
    return value


def _required_sequence_ids(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise OverfitContinuationContractError(f"{name} must be a sequence")
    rows = tuple(value)
    if not rows and not allow_empty:
        raise OverfitContinuationContractError(f"{name} cannot be empty")
    if any(not isinstance(row, str) or not row for row in rows):
        raise OverfitContinuationContractError(f"{name} must contain nonempty strings")
    if len(set(rows)) != len(rows):
        raise OverfitContinuationContractError(f"{name} cannot contain duplicates")
    return rows


def _required_digest(value: object, name: str) -> str:
    text = _required_nonempty_string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise OverfitContinuationContractError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _required_nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise OverfitContinuationContractError(f"{name} must be a nonempty string")
    return value


def _required_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OverfitContinuationContractError(f"{name} must be a positive integer")
    return value


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OverfitContinuationContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OverfitContinuationContractError(f"{name} must be finite")
    return result


def _optional_finite_float(value: object, name: str) -> float | None:
    return None if value is None else _finite_float(value, name)


def _python_rng_state(value: object) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise OverfitContinuationContractError("parent Python RNG state must be a tuple")
    probe = random.Random()
    try:
        probe.setstate(value)
    except (TypeError, ValueError) as error:
        raise OverfitContinuationContractError("parent Python RNG state is invalid") from error
    return value


def _numpy_rng_tuple(value: object) -> tuple[str, np.ndarray, int, int, float]:
    raw = _required_mapping(value, "parent NumPy RNG state")
    if set(raw) != {"bit_generator", "cached_gaussian", "has_gauss", "keys", "position"}:
        raise OverfitContinuationContractError("parent NumPy RNG state fields differ")
    bit_generator = _required_nonempty_string(raw["bit_generator"], "NumPy bit_generator")
    keys = np.asarray(raw["keys"], dtype=np.uint32)
    if keys.ndim != 1 or not len(keys):
        raise OverfitContinuationContractError("parent NumPy RNG keys must be one-dimensional")
    position = raw["position"]
    has_gauss = raw["has_gauss"]
    cached = raw["cached_gaussian"]
    if isinstance(position, bool) or not isinstance(position, int):
        raise OverfitContinuationContractError("parent NumPy RNG position must be an integer")
    if isinstance(has_gauss, bool) or not isinstance(has_gauss, int):
        raise OverfitContinuationContractError("parent NumPy RNG has_gauss must be an integer")
    cached_float = _finite_float(cached, "parent NumPy RNG cached_gaussian")
    state = (bit_generator, keys, position, has_gauss, cached_float)
    probe = np.random.RandomState()
    try:
        probe.set_state(state)
    except (TypeError, ValueError) as error:
        raise OverfitContinuationContractError("parent NumPy RNG state is invalid") from error
    return state


def _validated_generator_state(runtime: Any, value: object, *, device: Any) -> None:
    if not runtime.is_tensor(value) or value.dtype != runtime.uint8 or value.ndim != 1:
        raise OverfitContinuationContractError(
            "parent Torch generator state must be a uint8 vector"
        )
    try:
        runtime.Generator(device=device).set_state(value)
    except (RuntimeError, ValueError, TypeError) as error:
        raise OverfitContinuationContractError(
            f"parent Torch generator state is invalid for device {device}"
        ) from error


def _require_replayed_metric(name: str, actual: float, expected: object) -> None:
    expected_float = _finite_float(expected, name)
    if not math.isclose(actual, expected_float, rel_tol=1e-6, abs_tol=1e-7):
        raise OverfitContinuationContractError(
            f"replayed {name} differs from parent: {actual} != {expected_float}"
        )


def _require_replayed_optional_metric(name: str, actual: float | None, expected: object) -> None:
    expected_float = _optional_finite_float(expected, name)
    if actual is None or expected_float is None:
        if actual is not expected_float:
            raise OverfitContinuationContractError(f"replayed {name} presence differs from parent")
        return
    _require_replayed_metric(name, actual, expected_float)


def _raise_contract(message: str) -> Any:
    raise OverfitContinuationContractError(message)


def _remove_private_staging_directory(staging: Path, parent: Path) -> None:
    resolved_parent = parent.resolve()
    resolved_staging = staging.resolve()
    if resolved_staging.parent != resolved_parent:
        raise RuntimeError("refusing to remove continuation staging outside its parent")
    if not resolved_staging.name.startswith(".") or not resolved_staging.name.endswith(
        ".continuation.tmp"
    ):
        raise RuntimeError("refusing to remove a non-private continuation staging directory")
    shutil.rmtree(resolved_staging)


def _require_torch() -> Any:
    if torch is None:
        raise MissingOverfitTorchError(
            "run_tiny_overfit requires a platform-appropriate PyTorch installation"
        ) from _TORCH_IMPORT_ERROR
    return torch


def _seed_all(seed: int, runtime: Any, *, device: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    runtime.manual_seed(seed)
    if device.type == "cuda":
        runtime.cuda.manual_seed(seed)


def _sample_path(output: Path, sequence_id: str) -> Path:
    digest = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()
    return output / "samples" / f"{digest}.npy"


def _runtime_facts(runtime: Any, device: Any) -> dict[str, Any]:
    cuda_version = getattr(runtime.version, "cuda", None)
    cudnn_version = (
        runtime.backends.cudnn.version() if runtime.backends.cudnn.is_available() else None
    )
    return {
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "deterministic_algorithms_enabled": runtime.are_deterministic_algorithms_enabled(),
        "device": str(device),
        "device_name": runtime.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(runtime.__version__),
    }


def _identity_grouped_noise(
    runtime: Any,
    clean: Any,
    clips: tuple[Any, ...],
    *,
    generator: Any,
) -> Any:
    by_identity: dict[str, Any] = {}
    rows: list[Any] = []
    for clip in clips:
        if clip.identity_id not in by_identity:
            by_identity[clip.identity_id] = runtime.randn(
                clean.shape[1:],
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        rows.append(by_identity[clip.identity_id])
    return runtime.stack(rows, dim=0)


def _build_endpoint_contrast_plan(
    clips: tuple[Any, ...],
    clean: np.ndarray,
) -> EndpointContrastPlan:
    """Select target-distinct causal contrasts and quarantine unusable rows."""

    if len(clips) != len(clean):
        raise ValueError("clip and clean-array counts must match")
    candidates: dict[tuple[Any, ...], list[int]] = {}
    for index, clip in enumerate(clips):
        request = clip.request
        key = (
            clip.identity_id,
            request.description,
            request.entity_class,
            request.view,
            request.direction,
            request.loop_mode,
            tuple(float(value) for value in clip.frame_phases),
        )
        candidates.setdefault(key, []).append(index)

    groups: list[EndpointContrastGroup] = []
    exclusions: list[EndpointContrastExclusion] = []
    ordered_candidates = sorted(candidates.items(), key=lambda item: _condition_key_bytes(item[0]))
    for key, indices in ordered_candidates:
        by_action: dict[str, list[int]] = {}
        for index in indices:
            by_action.setdefault(clips[index].request.action, []).append(index)
        if len(by_action) < 2:
            continue
        representatives: list[tuple[str, int, str]] = []
        for action, action_indices in sorted(
            by_action.items(), key=lambda item: item[0].encode("utf-8")
        ):
            ordered = sorted(action_indices, key=lambda index: clips[index].sequence_id.encode())
            digests = {_numpy_array_sha256(clean[index]) for index in ordered}
            if len(digests) > 1:
                exclusions.extend(
                    EndpointContrastExclusion(
                        sequence_id=clips[index].sequence_id,
                        identity_id=clips[index].identity_id,
                        action=action,
                        target_sha256=_numpy_array_sha256(clean[index]),
                        reason="conflicting_targets_for_identical_action_and_non_action_conditions",
                    )
                    for index in ordered
                )
                continue
            representative = ordered[0]
            digest = next(iter(digests))
            representatives.append((action, representative, digest))
            exclusions.extend(
                EndpointContrastExclusion(
                    sequence_id=clips[index].sequence_id,
                    identity_id=clips[index].identity_id,
                    action=action,
                    target_sha256=digest,
                    reason="byte_identical_duplicate_target_uses_one_representative",
                )
                for index in ordered[1:]
            )
        by_target: dict[str, list[tuple[str, int, str]]] = {}
        for representative in representatives:
            by_target.setdefault(representative[2], []).append(representative)
        target_distinct_representatives: list[tuple[str, int, str]] = []
        for target_sha256, target_rows in sorted(by_target.items()):
            ordered_target_rows = sorted(target_rows, key=lambda item: item[0].encode("utf-8"))
            target_distinct_representatives.append(ordered_target_rows[0])
            exclusions.extend(
                EndpointContrastExclusion(
                    sequence_id=clips[index].sequence_id,
                    identity_id=clips[index].identity_id,
                    action=action,
                    target_sha256=target_sha256,
                    reason="byte_identical_cross_action_target_uses_one_representative",
                )
                for action, index, _digest in ordered_target_rows[1:]
            )
        target_distinct_representatives.sort(key=lambda item: item[0].encode("utf-8"))
        if len(target_distinct_representatives) < 2:
            exclusions.extend(
                EndpointContrastExclusion(
                    sequence_id=clips[index].sequence_id,
                    identity_id=clips[index].identity_id,
                    action=action,
                    target_sha256=digest,
                    reason=(
                        "no_target_distinct_multi_action_contrast_after_conflict_and_alias_filter"
                    ),
                )
                for action, index, digest in target_distinct_representatives
            )
            continue
        representatives = target_distinct_representatives
        condition_payload = _condition_key_payload(key)
        groups.append(
            EndpointContrastGroup(
                key_sha256=hashlib.sha256(
                    json.dumps(
                        condition_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                identity_id=condition_payload["identity_id"],
                description=condition_payload["description"],
                entity_class=condition_payload["entity_class"],
                view=condition_payload["view"],
                direction=condition_payload["direction"],
                loop_mode=condition_payload["loop_mode"],
                frame_phases=tuple(condition_payload["frame_phases"]),
                selected_indices=tuple(item[1] for item in representatives),
                sequence_ids=tuple(clips[item[1]].sequence_id for item in representatives),
                actions=tuple(item[0] for item in representatives),
                target_sha256=tuple(item[2] for item in representatives),
            )
        )
    groups.sort(key=lambda group: group.key_sha256)
    exclusions.sort(key=lambda row: (row.sequence_id.encode("utf-8"), row.reason))
    return EndpointContrastPlan(groups=tuple(groups), exclusions=tuple(exclusions))


def _condition_key_payload(key: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "description": key[1],
        "direction": key[4],
        "entity_class": key[2],
        "frame_phases": list(key[6]),
        "identity_id": key[0],
        "loop_mode": key[5],
        "view": key[3],
    }


def _condition_key_bytes(key: tuple[Any, ...]) -> bytes:
    return json.dumps(
        _condition_key_payload(key),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _numpy_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _contrast_grouped_noise(
    runtime: Any,
    clean: Any,
    plan: EndpointContrastPlan,
    *,
    generator: Any,
) -> Any:
    if clean.shape[0] != len(plan.selected_indices):
        raise ValueError("endpoint tensor and contrast-plan row counts must match")
    rows: list[Any] = []
    for group in plan.groups:
        noise = runtime.randn(
            clean.shape[1:],
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        rows.extend(noise for _index in group.selected_indices)
    if not rows:
        raise ValueError("endpoint contrast plan contains no selected rows")
    return runtime.stack(rows, dim=0)


def _slice_encoded_conditions(
    batch: EncodedConditionBatch,
    indices: tuple[int, ...],
) -> EncodedConditionBatch:
    if not indices:
        raise ValueError("condition indices cannot be empty")
    if any(index < 0 or index >= batch.batch_size for index in indices):
        raise IndexError("condition index is out of range")

    def selected(values: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(values[index] for index in indices)

    return replace(
        batch,
        descriptions=selected(batch.descriptions),
        text_token_ids=selected(batch.text_token_ids),
        text_attention_mask=selected(batch.text_attention_mask),
        entity_ids=selected(batch.entity_ids),
        action_ids=selected(batch.action_ids),
        view_ids=selected(batch.view_ids),
        direction_ids=selected(batch.direction_ids),
        loop_mode_ids=selected(batch.loop_mode_ids),
    )


def _permute_actions_for_endpoint_plan(
    batch: EncodedConditionBatch,
    plan: EndpointContrastPlan,
) -> EncodedConditionBatch | None:
    """Cyclically swap action tokens only inside matched causal contrast groups."""

    if any(index >= batch.batch_size for index in plan.selected_indices):
        raise ValueError("endpoint contrast plan index exceeds condition batch")
    permuted = list(batch.action_ids)
    for group in plan.groups:
        for position, index in enumerate(group.selected_indices):
            replacement = group.selected_indices[(position + 1) % len(group.selected_indices)]
            permuted[index] = batch.action_ids[replacement]
    if not plan.groups:
        return None
    return replace(batch, action_ids=tuple(permuted))


def _endpoint_action_permutation_rows(plan: EndpointContrastPlan) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for group in plan.groups:
        for position, sequence_id in enumerate(group.sequence_ids):
            replacement = (position + 1) % len(group.sequence_ids)
            rows.append(
                {
                    "contrast_group_sha256": group.key_sha256,
                    "original_action": group.actions[position],
                    "replacement_action": group.actions[replacement],
                    "sequence_id": sequence_id,
                }
            )
    return rows


def _endpoint_action_swap_matrix(
    *,
    current_loss: Any,
    flow: RectifiedFlowBatch,
    conditions: EncodedConditionBatch,
    reference_clean: Any,
    frame_phases: Any,
    endpoint_plan: EndpointContrastPlan,
    clips: tuple[Any, ...],
) -> list[dict[str, Any]]:
    if conditions.batch_size != len(clips) or len(clips) != len(endpoint_plan.selected_indices):
        raise ValueError("endpoint diagnostic rows do not match the contrast plan")
    rows: list[dict[str, Any]] = []
    cursor = 0
    for group in endpoint_plan.groups:
        positions = tuple(range(cursor, cursor + len(group.selected_indices)))
        for position in positions:
            one_flow = _slice_flow(flow, position)
            one_condition = _slice_encoded_conditions(conditions, (position,))
            one_clean = reference_clean[position : position + 1]
            one_phases = frame_phases[position : position + 1]
            correct_loss = float(
                current_loss(
                    one_flow,
                    one_condition,
                    reference_clean=one_clean,
                    frame_phases=one_phases,
                )
                .detach()
                .cpu()
            )
            for replacement_position in positions:
                if replacement_position == position:
                    continue
                swapped = replace(
                    one_condition,
                    action_ids=(conditions.action_ids[replacement_position],),
                )
                swapped_loss = float(
                    current_loss(
                        one_flow,
                        swapped,
                        reference_clean=one_clean,
                        frame_phases=one_phases,
                    )
                    .detach()
                    .cpu()
                )
                if not math.isfinite(correct_loss) or not math.isfinite(swapped_loss):
                    raise RuntimeError("endpoint action swap loss is non-finite")
                rows.append(
                    {
                        "contrast_group_sha256": group.key_sha256,
                        "correct_loss": correct_loss,
                        "delta": swapped_loss - correct_loss,
                        "original_action": clips[position].request.action,
                        "replacement_action": clips[replacement_position].request.action,
                        "sequence_id": clips[position].sequence_id,
                        "swapped_loss": swapped_loss,
                    }
                )
        cursor += len(group.selected_indices)
    return rows


def _slice_flow(flow: RectifiedFlowBatch, position: int) -> RectifiedFlowBatch:
    selection = slice(position, position + 1)
    return RectifiedFlowBatch(
        clean=flow.clean[selection],
        noise=flow.noise[selection],
        noisy=flow.noisy[selection],
        timesteps=flow.timesteps[selection],
        target_velocity=flow.target_velocity[selection],
    )


def _tensor_sha256(runtime: Any, value: Any) -> str:
    contiguous = value.detach().contiguous()
    raw = contiguous.view(runtime.uint8).cpu().numpy().tobytes(order="C")
    header = (
        f"{contiguous.dtype}\0{'x'.join(str(dimension) for dimension in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + raw).hexdigest()


def _numpy_rng_state() -> dict[str, Any]:
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": bit_generator,
        "cached_gaussian": float(cached_gaussian),
        "has_gauss": int(has_gauss),
        "keys": keys.tolist(),
        "position": int(position),
    }


def _atomic_numpy(
    path: Path,
    array: np.ndarray,
    *,
    overwrite: bool,
    disk_guard: DiskGuard | None,
) -> None:
    estimated = int(array.nbytes) + 65_536
    if disk_guard is not None:
        disk_guard.require_capacity(estimated, label="overfit sample")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_temporary(
            temporary,
            path,
            overwrite=overwrite,
            refusal=f"Refusing to replace existing sample: {path}",
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_torch_save(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
    disk_guard: DiskGuard | None,
) -> None:
    runtime = _require_torch()
    if disk_guard is not None:
        disk_guard.require_capacity(
            _estimate_torch_save_bytes(runtime, payload),
            label="overfit checkpoint",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        runtime.save(payload, temporary)
        if disk_guard is not None:
            disk_guard.require_capacity(label="overfit checkpoint promotion")
        _publish_temporary(
            temporary,
            path,
            overwrite=overwrite,
            refusal=f"Refusing to replace existing checkpoint: {path}",
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_bytes(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool,
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
        _publish_temporary(
            temporary,
            path,
            overwrite=overwrite,
            refusal=f"Refusing to replace existing artifact: {path}",
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _publish_temporary(
    temporary: Path,
    path: Path,
    *,
    overwrite: bool,
    refusal: str,
) -> None:
    if overwrite:
        os.replace(temporary, path)
        return
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(refusal) from error
    temporary.unlink()


def _estimate_torch_save_bytes(runtime: Any, value: object) -> int:
    """Conservatively estimate serialized tensor payload before touching the disk."""

    raw_bytes = _torch_payload_bytes(runtime, value)
    return raw_bytes + max(8 * 1024 * 1024, raw_bytes // 10)


def _torch_payload_bytes(runtime: Any, value: object) -> int:
    if runtime.is_tensor(value):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, Mapping):
        return sum(
            _torch_payload_bytes(runtime, key) + _torch_payload_bytes(runtime, nested)
            for key, nested in value.items()
        )
    if isinstance(value, list | tuple):
        return sum(_torch_payload_bytes(runtime, nested) for nested in value)
    if isinstance(value, str):
        return len(value.encode("utf-8")) + 64
    if isinstance(value, bytes | bytearray):
        return len(value) + 64
    return 64


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
