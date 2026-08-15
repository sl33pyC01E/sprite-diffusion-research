"""Resumable matched-action training for reference-conditioned MUGEN motion."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

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
except ImportError as exc:  # pragma: no cover - optional dependency boundary
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None

Precision = Literal["float32", "bfloat16"]
TimeSampling = Literal["endpoint", "uniform"]
FlowSampler = Literal["euler", "heun"]
ArrayLoading = Literal["eager", "lazy"]
ActionBatchMode = Literal["pair", "bundle"]
ActionConditioningMode = Literal["single", "expanded"]


class LatentMotionTrainingError(ValueError):
    """Raised when a broad motion training contract is invalid."""


@dataclass(frozen=True, slots=True)
class LatentMotionTrainingConfig:
    """Quality-first matched-action contract for one 4090-class GPU."""

    gradient_accumulation: int = 4
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.9995
    latent_endpoint_weight: float = 1.0
    pixel_endpoint_weight: float = 1.0
    action_contrast_weight: float = 0.0
    pixel_action_contrast_weight: float = 0.0
    temporal_motion_weight: float = 0.0
    target_directed_motion_weight: float = 0.0
    minimum_target_motion_progress: float = 0.8
    action_batch_mode: ActionBatchMode = "pair"
    action_conditioning_mode: ActionConditioningMode = "single"
    action_token_count: int = 1
    action_condition_scale: float = 1.0
    time_sampling: TimeSampling = "uniform"
    endpoint_sample_probability: float = 0.25
    inference_steps: int = 16
    sampler_algorithm: FlowSampler = "heun"
    steps: int = 50_000
    log_every: int = 25
    validate_every: int = 500
    checkpoint_every: int = 1_000
    validation_pairs: int = 8
    preview_pairs: int = 4
    seed: int = 20260825
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
        for name in (
            "gradient_accumulation",
            "action_token_count",
            "inference_steps",
            "steps",
            "log_every",
            "validate_every",
            "checkpoint_every",
            "validation_pairs",
            "preview_pairs",
            "seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise ValueError("warmup_steps must be an integer")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0,steps)")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.action_condition_scale) or self.action_condition_scale <= 0:
            raise ValueError("action_condition_scale must be finite and positive")
        for name in (
            "minimum_learning_rate",
            "weight_decay",
            "latent_endpoint_weight",
            "pixel_endpoint_weight",
            "action_contrast_weight",
            "pixel_action_contrast_weight",
            "temporal_motion_weight",
            "target_directed_motion_weight",
            "endpoint_sample_probability",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        if self.latent_endpoint_weight == 0 and self.pixel_endpoint_weight == 0:
            raise ValueError("at least one denoising objective must be positive")
        if self.endpoint_sample_probability > 1:
            raise ValueError("endpoint_sample_probability must be in [0,1]")
        if (
            not math.isfinite(self.minimum_target_motion_progress)
            or not 0 < self.minimum_target_motion_progress <= 1
        ):
            raise ValueError("minimum_target_motion_progress must be in (0,1]")
        if self.time_sampling not in {"endpoint", "uniform"}:
            raise ValueError("time_sampling must be endpoint or uniform")
        if self.action_batch_mode not in {"pair", "bundle"}:
            raise ValueError("action_batch_mode must be pair or bundle")
        if self.action_conditioning_mode not in {"single", "expanded"}:
            raise ValueError("action_conditioning_mode must be single or expanded")
        if self.action_conditioning_mode == "single" and self.action_token_count != 1:
            raise ValueError("single action conditioning requires one token")
        if self.action_batch_mode == "bundle" and self.action_contrast_weight:
            raise ValueError("bundle action batches require pixel-space action contrast")
        if self.sampler_algorithm not in {"euler", "heun"}:
            raise ValueError("sampler_algorithm must be euler or heun")
        if self.time_sampling == "endpoint" and (
            self.endpoint_sample_probability != 0
            or self.inference_steps != 1
            or self.sampler_algorithm != "euler"
        ):
            raise ValueError("endpoint training requires its one-step endpoint sampler")
        if self.time_sampling == "uniform" and self.inference_steps < 2:
            raise ValueError("uniform flow training requires at least two inference steps")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")


@dataclass(frozen=True, slots=True)
class LatentMotionTrainingRow:
    sequence_id: str
    identity_id: str
    verb: str
    action_index: int
    split: str
    duration_ms: tuple[float, ...]
    loop_mode: str


@dataclass(frozen=True, slots=True)
class _LazyArrayEntry:
    path: Path
    file_sha256: str
    array_content_sha256: str
    source_shape: tuple[int, ...]
    item_index: int | None = None
    item_array_content_sha256: str | None = None


class _LazyVerifiedArrayStack:
    """Index exact NPY payloads without materializing the full corpus in RAM."""

    def __init__(
        self,
        entries: tuple[_LazyArrayEntry, ...],
        *,
        dtype: Any,
        item_shape: tuple[int, ...],
        verify_hashes: bool,
        cache_size: int = 8,
    ) -> None:
        if not entries:
            raise ValueError("lazy array stack cannot be empty")
        self._entries = entries
        self.dtype = np.dtype(dtype)
        self.shape = (len(entries), *item_shape)
        self._item_shape = item_shape
        self._verify_hashes = verify_hashes
        self._cache_size = cache_size
        self._cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self._verified_paths: set[Path] = set()
        self._verified_items: set[tuple[Path, int]] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += len(self)
            if not 0 <= index < len(self):
                raise IndexError(index)
            return self._load(index)
        if isinstance(key, slice):
            indices = tuple(range(*key.indices(len(self))))
        elif isinstance(key, np.ndarray):
            indices = tuple(int(value) for value in key.tolist())
        else:
            indices = tuple(int(value) for value in key)
        if not indices:
            return np.empty((0, *self._item_shape), dtype=self.dtype)
        return np.stack(tuple(self._load(index) for index in indices))

    def array_content_sha256(self, index: int) -> str:
        entry = self._entries[index]
        return entry.item_array_content_sha256 or entry.array_content_sha256

    def _load(self, index: int) -> np.ndarray:
        entry = self._entries[index]
        value = self._cache.get(entry.path)
        if value is None:
            value = self._load_source(entry)
            self._cache[entry.path] = value
            self._cache.move_to_end(entry.path)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(entry.path)
        if entry.item_index is None:
            item = value
        else:
            item = np.ascontiguousarray(value[entry.item_index])
            item_key = (entry.path, entry.item_index)
            if self._verify_hashes and item_key not in self._verified_items:
                if _array_sha256(item) != entry.item_array_content_sha256:
                    raise LatentMotionTrainingError(f"lazy array item hash differs: {entry.path}")
                self._verified_items.add(item_key)
        if item.dtype != self.dtype or item.shape != self._item_shape:
            raise LatentMotionTrainingError(f"lazy array geometry differs: {entry.path}")
        return np.ascontiguousarray(item)

    def _load_source(self, entry: _LazyArrayEntry) -> np.ndarray:
        if self._verify_hashes and entry.path not in self._verified_paths:
            payload = entry.path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry.file_sha256:
                raise LatentMotionTrainingError(f"lazy array file hash differs: {entry.path}")
            try:
                value = np.load(io.BytesIO(payload), allow_pickle=False)
            except (OSError, ValueError) as error:
                raise LatentMotionTrainingError(
                    f"lazy array is unreadable: {entry.path}"
                ) from error
            if _array_sha256(value) != entry.array_content_sha256:
                raise LatentMotionTrainingError(f"lazy array content hash differs: {entry.path}")
            self._verified_paths.add(entry.path)
        else:
            try:
                value = np.load(entry.path, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError) as error:
                raise LatentMotionTrainingError(
                    f"lazy array is unreadable: {entry.path}"
                ) from error
        if (
            value.dtype != self.dtype
            or value.shape != entry.source_shape
            or not bool(np.isfinite(value).all())
        ):
            raise LatentMotionTrainingError(f"lazy array content differs: {entry.path}")
        return value


@dataclass(frozen=True, slots=True)
class LatentMotionTrainingCorpus:
    rows: tuple[LatentMotionTrainingRow, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    action_vocabulary: tuple[str, ...]
    target_latents: np.ndarray | _LazyVerifiedArrayStack
    reference_latents: np.ndarray | _LazyVerifiedArrayStack
    target_rgba: np.ndarray | _LazyVerifiedArrayStack
    phases: np.ndarray
    channel_mean: tuple[float, ...]
    channel_standard_deviation: tuple[float, ...]
    autoencoder_checkpoint_path: Path
    autoencoder_architecture: dict[str, Any]
    contract: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LatentMotionTrainingResult:
    output_directory: Path
    report_path: Path
    training_checkpoint_path: Path
    inference_checkpoint_path: Path
    report_sha256: str


@dataclass(frozen=True, slots=True)
class LatentMotionEvaluationResult:
    output_directory: Path
    report_path: Path
    report_sha256: str


if torch is not None and nn is not None:

    class _ActionConditionedMotionModel(nn.Module):
        def __init__(
            self,
            config: LatentMotionDiTConfig,
            action_count: int,
            *,
            conditioning_mode: ActionConditioningMode = "single",
            action_token_count: int = 1,
            action_condition_scale: float = 1.0,
        ) -> None:
            super().__init__()
            self.config = config
            self.conditioning_mode = conditioning_mode
            self.action_token_count = action_token_count
            self.action_condition_scale = action_condition_scale
            self.dit = ReferenceConditionedLatentMotionDiT(config)
            self.action_embedding = nn.Embedding(action_count, config.condition_dim)
            self.action_norm = nn.LayerNorm(config.condition_dim)
            if conditioning_mode == "expanded":
                phase_width = 1 + 2 * config.phase_harmonics
                self.action_token_projection = nn.Linear(
                    config.condition_dim,
                    action_token_count * config.condition_dim,
                )
                self.action_token_norm = nn.LayerNorm(config.condition_dim)
                self.action_frame_mlp = nn.Sequential(
                    nn.Linear(config.condition_dim + phase_width, config.model_dim * 2),
                    nn.SiLU(),
                    nn.Linear(config.model_dim * 2, config.model_dim),
                )
                self._initialize_expanded_conditioning()

        def _initialize_expanded_conditioning(self) -> None:
            with torch.no_grad():
                self.action_token_projection.weight.zero_()
                self.action_token_projection.bias.zero_()
                identity = torch.eye(
                    self.config.condition_dim,
                    dtype=self.action_token_projection.weight.dtype,
                )
                for token_index in range(self.action_token_count):
                    start = token_index * self.config.condition_dim
                    self.action_token_projection.weight[
                        start : start + self.config.condition_dim
                    ].copy_(identity)
                nn.init.normal_(self.action_frame_mlp[-1].weight, std=0.02)
                self.action_frame_mlp[-1].bias.zero_()

        def _expanded_action_conditioning(
            self, action_indices: torch.Tensor, frame_phase: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            base = self.action_norm(self.action_embedding(action_indices))
            context = self.action_token_projection(base).reshape(
                base.shape[0], self.action_token_count, self.config.condition_dim
            )
            context = self.action_token_norm(context) * self.action_condition_scale
            frequencies = torch.arange(
                1,
                self.config.phase_harmonics + 1,
                device=frame_phase.device,
                dtype=torch.float32,
            )
            phase = frame_phase.to(dtype=torch.float32)
            angles = 2 * math.pi * phase.unsqueeze(-1) * frequencies
            phase_features = torch.cat(
                (phase.unsqueeze(-1), torch.sin(angles), torch.cos(angles)), dim=-1
            )
            frame_input = torch.cat(
                (
                    base.unsqueeze(1).expand(-1, self.config.num_frames, -1),
                    phase_features,
                ),
                dim=-1,
            )
            frame_conditioning = self.action_frame_mlp(frame_input) * self.action_condition_scale
            return context, frame_conditioning

        def forward(
            self,
            video: torch.Tensor,
            reference: torch.Tensor,
            timesteps: torch.Tensor,
            action_indices: torch.Tensor,
            *,
            frame_phase: torch.Tensor,
        ) -> torch.Tensor:
            if self.conditioning_mode == "expanded":
                context, frame_conditioning = self._expanded_action_conditioning(
                    action_indices, frame_phase
                )
            else:
                context = (
                    self.action_norm(self.action_embedding(action_indices)).unsqueeze(1)
                    * self.action_condition_scale
                )
                frame_conditioning = None
            return self.dit(
                video,
                reference,
                timesteps,
                context,
                frame_phase=frame_phase,
                frame_conditioning=frame_conditioning,
            )


def _build_action_conditioned_motion_model(
    config: LatentMotionTrainingConfig, action_count: int
) -> Any:
    return _ActionConditionedMotionModel(
        config.model,
        action_count,
        conditioning_mode=config.action_conditioning_mode,
        action_token_count=config.action_token_count,
        action_condition_scale=config.action_condition_scale,
    )


def build_matched_action_index(
    rows: tuple[LatentMotionTrainingRow, ...], indices: tuple[int, ...]
) -> dict[str, dict[str, int]]:
    """Build identity -> verb -> canonical row for causal matched sampling."""

    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for index in indices:
        row = rows[index]
        if row.verb in grouped[row.identity_id]:
            raise ValueError(f"duplicate canonical identity/verb: {row.identity_id}/{row.verb}")
        grouped[row.identity_id][row.verb] = index
    return {
        identity: dict(sorted(verbs.items(), key=lambda item: item[0].encode()))
        for identity, verbs in sorted(grouped.items(), key=lambda item: item[0].encode())
        if len(verbs) >= 2
    }


def sample_matched_action_pair(
    index: dict[str, dict[str, int]], *, generator: Any
) -> tuple[int, int]:
    """Uniformly sample one identity and two distinct action targets."""

    runtime = _require_torch()
    if not index:
        raise ValueError("matched action index cannot be empty")
    identities = tuple(index)
    identity = identities[int(runtime.randint(len(identities), (1,), generator=generator))]
    verbs = tuple(index[identity])
    order = runtime.randperm(len(verbs), generator=generator)[:2].tolist()
    return index[identity][verbs[order[0]]], index[identity][verbs[order[1]]]


def _target_distinct_pairs_from_index(
    index: dict[str, dict[str, int]], target_digests: dict[int, str]
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Keep named action contrasts only when their exact target tensors differ."""

    output = {}
    for identity, actions in index.items():
        verbs = tuple(actions)
        pairs = tuple(
            (actions[verbs[left]], actions[verbs[right]])
            for left, right in combinations(range(len(verbs)), 2)
            if target_digests[actions[verbs[left]]] != target_digests[actions[verbs[right]]]
        )
        if pairs:
            output[identity] = pairs
    return output


def _sample_target_distinct_pair(
    pairs: dict[str, tuple[tuple[int, int], ...]], *, generator: Any
) -> tuple[int, int]:
    runtime = _require_torch()
    if not pairs:
        raise ValueError("target-distinct pair index cannot be empty")
    identities = tuple(pairs)
    identity = identities[int(runtime.randint(len(identities), (1,), generator=generator))]
    candidates = pairs[identity]
    return candidates[int(runtime.randint(len(candidates), (1,), generator=generator))]


def _target_distinct_bundles_from_index(
    index: dict[str, dict[str, int]], target_digests: dict[int, str]
) -> dict[str, tuple[int, ...]]:
    """Keep each dense identity's full action bundle when two targets differ."""

    output = {}
    for identity, actions in index.items():
        bundle = tuple(actions[verb] for verb in sorted(actions, key=str.encode))
        if len({target_digests[row] for row in bundle}) >= 2:
            output[identity] = bundle
    return output


def _sample_target_distinct_bundle(
    bundles: dict[str, tuple[int, ...]], *, generator: Any
) -> tuple[int, ...]:
    runtime = _require_torch()
    if not bundles:
        raise ValueError("target-distinct bundle index cannot be empty")
    identities = tuple(bundles)
    identity = identities[int(runtime.randint(len(identities), (1,), generator=generator))]
    return bundles[identity]


def _sample_training_times(
    runtime: Any,
    *,
    batch: int,
    config: LatentMotionTrainingConfig,
    device: Any,
    generator: Any,
) -> Any:
    """Sample one shared noise level for a matched causal action pair."""

    if config.time_sampling == "endpoint":
        return runtime.ones((batch,), device=device)
    sampled = runtime.rand((1,), device=device, generator=generator)
    if config.endpoint_sample_probability:
        choose_endpoint = runtime.rand((1,), device=device, generator=generator)
        sampled = runtime.where(choose_endpoint < config.endpoint_sample_probability, 1, sampled)
    return sampled.expand(batch)


def _time_batch_view(times: Any, *, batch: int) -> Any:
    """Broadcast scalar flow times across a variable-size action batch."""

    if tuple(times.shape) != (batch,):
        raise ValueError(f"times must have shape {(batch,)!r}")
    return times.view(batch, 1, 1, 1, 1)


def _sample_motion_residual(
    runtime: Any,
    model: Any,
    *,
    noise: Any,
    reference: Any,
    actions: Any,
    phases: Any,
    inference_steps: int,
    sampler_algorithm: FlowSampler,
) -> Any:
    """Integrate the learned clean-to-noise velocity field from t=1 to t=0."""

    if inference_steps <= 0:
        raise ValueError("inference_steps must be positive")
    if sampler_algorithm not in {"euler", "heun"}:
        raise ValueError("sampler_algorithm must be euler or heun")
    state = noise
    batch = int(noise.shape[0])
    step_size = -1.0 / inference_steps
    for step in range(inference_steps):
        time_value = 1.0 - step / inference_steps
        times = runtime.full((batch,), time_value, device=noise.device)
        velocity = model(state, reference, times, actions, frame_phase=phases)
        proposal = state + step_size * velocity
        if sampler_algorithm == "heun" and step + 1 < inference_steps:
            next_times = runtime.full((batch,), time_value + step_size, device=noise.device)
            next_velocity = model(
                proposal,
                reference,
                next_times,
                actions,
                frame_phase=phases,
            )
            state = state + 0.5 * step_size * (velocity + next_velocity)
        else:
            state = proposal
    return state


def load_latent_motion_training_corpus(
    manifest_path: Path | str,
    *,
    verify_hashes: bool = True,
    array_loading: ArrayLoading = "eager",
) -> LatentMotionTrainingCorpus:
    """Load the complete canonical manifest and exact latent/RGBA closure."""

    if array_loading not in {"eager", "lazy"}:
        raise ValueError("array_loading must be eager or lazy")

    manifest_file = Path(manifest_path).resolve()
    manifest_bytes = manifest_file.read_bytes()
    manifest = _json_object(manifest_bytes, "training manifest")
    if manifest.get("artifact_kind") != (
        "mugen_reference_conditioned_primary_motion_training_manifest"
    ):
        raise LatentMotionTrainingError("training manifest has the wrong artifact kind")
    config = manifest.get("config")
    if not isinstance(config, dict) or config.get("one_sequence_per_identity_verb") is not True:
        raise LatentMotionTrainingError("training manifest is not canonical identity/action data")
    records = _counted_records(manifest, "training manifest")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise LatentMotionTrainingError("training manifest source is missing")
    plan_path = Path(_required_text(source, "motion_plan_path")).resolve()
    plan_bytes = plan_path.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != source.get("motion_plan_file_sha256"):
        raise LatentMotionTrainingError("motion-plan hash differs")
    plan = _json_object(plan_bytes, "motion plan")
    plan_source = plan.get("source")
    if not isinstance(plan_source, dict):
        raise LatentMotionTrainingError("motion-plan source is missing")
    latent_source = plan_source.get("latent_manifest")
    materialization_source = plan_source.get("materialization")
    if not isinstance(latent_source, dict) or not isinstance(materialization_source, dict):
        raise LatentMotionTrainingError("latent/materialization source is missing")
    latent_manifest_path = Path(_required_text(latent_source, "path")).resolve()
    latent_manifest_bytes = latent_manifest_path.read_bytes()
    if hashlib.sha256(latent_manifest_bytes).hexdigest() != latent_source.get("file_sha256"):
        raise LatentMotionTrainingError("latent-manifest hash differs")
    latent_manifest = _json_object(latent_manifest_bytes, "latent manifest")
    materialization_path = Path(_required_text(materialization_source, "path")).resolve()
    if _file_sha256(materialization_path) != materialization_source.get("file_sha256"):
        raise LatentMotionTrainingError("materialization hash differs")
    normalization = latent_manifest.get("normalization")
    codec = latent_manifest.get("codec")
    if not isinstance(normalization, dict) or not isinstance(codec, dict):
        raise LatentMotionTrainingError("normalization/codec is missing")
    mean = _float_tuple(normalization.get("channel_mean"), "channel mean")
    std = _float_tuple(normalization.get("channel_standard_deviation"), "channel std")
    autoencoder_path = Path(_required_text(codec, "checkpoint_path")).resolve()
    if _file_sha256(autoencoder_path) != codec.get("checkpoint_file_sha256"):
        raise LatentMotionTrainingError("autoencoder checkpoint hash differs")
    architecture = codec.get("architecture")
    if not isinstance(architecture, dict):
        raise LatentMotionTrainingError("autoencoder architecture is missing")

    actions = tuple(sorted({_record_verb(record) for record in records}, key=str.encode))
    action_to_index = {action: index for index, action in enumerate(actions)}
    latent_root = latent_manifest_path.parent
    materialization_root = materialization_path.parent
    latent_cache: dict[str, np.ndarray] = {}
    rows = []
    target_latents: list[np.ndarray | _LazyArrayEntry] = []
    reference_latents: list[np.ndarray | _LazyArrayEntry] = []
    target_rgba: list[np.ndarray | _LazyArrayEntry] = []
    phases = []
    for record in records:
        sequence_id = _required_text(record, "sequence_id")
        target = _required_dict(record, "target")
        reference = _required_dict(record, "reference")
        target_latent_record = _required_dict(target, "latent")
        target_latent = (
            _lazy_array_entry(
                latent_root,
                target_latent_record,
                shape=(8, 8, 64, 64),
                label=f"target latent {sequence_id}",
            )
            if array_loading == "lazy"
            else _load_array(
                latent_root,
                target_latent_record,
                dtype=np.float16,
                shape=(8, 8, 64, 64),
                label=f"target latent {sequence_id}",
                verify_hashes=verify_hashes,
                cache=latent_cache,
            )
        )
        reference_record = _required_dict(reference, "latent")
        frame_index = reference.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < 8
        ):
            raise LatentMotionTrainingError(f"reference frame index differs for {sequence_id}")
        if array_loading == "lazy":
            reference_frame = _lazy_array_entry(
                latent_root,
                reference_record,
                shape=(8, 8, 64, 64),
                label=f"reference latent {sequence_id}",
                item_index=frame_index,
                item_array_content_sha256=_required_hash(
                    reference_record,
                    "frame_array_content_sha256",
                    f"reference frame {sequence_id}",
                ),
            )
        else:
            reference_full = _load_array(
                latent_root,
                reference_record,
                dtype=np.float16,
                shape=(8, 8, 64, 64),
                label=f"reference latent {sequence_id}",
                verify_hashes=verify_hashes,
                cache=latent_cache,
            )
            reference_frame = np.ascontiguousarray(reference_full[frame_index])
            if verify_hashes and _array_sha256(reference_frame) != reference_record.get(
                "frame_array_content_sha256"
            ):
                raise LatentMotionTrainingError(f"reference frame hash differs for {sequence_id}")
        source_pixels = _required_dict(target, "source_pixels")
        rgba = (
            _lazy_array_entry(
                materialization_root,
                source_pixels,
                shape=(8, 128, 128, 4),
                label=f"target RGBA {sequence_id}",
            )
            if array_loading == "lazy"
            else _load_array(
                materialization_root,
                source_pixels,
                dtype=np.uint8,
                shape=(8, 128, 128, 4),
                label=f"target RGBA {sequence_id}",
                verify_hashes=verify_hashes,
                cache=None,
            )
        )
        phase = np.asarray(target.get("phase"), dtype=np.float32)
        if phase.shape != (8,) or not np.isfinite(phase).all() or np.any((phase < 0) | (phase > 1)):
            raise LatentMotionTrainingError(f"target phase differs for {sequence_id}")
        durations = target.get("duration_ms")
        if (
            not isinstance(durations, list)
            or len(durations) != 8
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in durations
            )
        ):
            raise LatentMotionTrainingError(f"target durations differ for {sequence_id}")
        verb = _record_verb(record)
        rows.append(
            LatentMotionTrainingRow(
                sequence_id=sequence_id,
                identity_id=_required_text(record, "identity_id"),
                verb=verb,
                action_index=action_to_index[verb],
                split=_required_text(record, "split"),
                duration_ms=tuple(float(value) for value in durations),
                loop_mode=_required_text(target, "loop_mode"),
            )
        )
        target_latents.append(target_latent)
        reference_latents.append(reference_frame)
        target_rgba.append(rgba)
        phases.append(phase)
    del latent_cache
    row_tuple = tuple(rows)
    split_indices = {
        split: tuple(index for index, row in enumerate(row_tuple) if row.split == split)
        for split in ("train", "validation", "test")
    }
    if any(not values for values in split_indices.values()):
        raise LatentMotionTrainingError("train/validation/test splits must be non-empty")
    identity_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in row_tuple:
        identity_splits[row.identity_id].add(row.split)
    if any(len(values) != 1 for values in identity_splits.values()):
        raise LatentMotionTrainingError("identities cross dataset splits")
    contract = {
        "action_vocabulary": list(actions),
        "autoencoder_checkpoint_sha256": _file_sha256(autoencoder_path),
        "latent_manifest_file_sha256": hashlib.sha256(latent_manifest_bytes).hexdigest(),
        "manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "motion_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "record_count": len(rows),
        "split_identities": {
            split: len({row_tuple[index].identity_id for index in indices})
            for split, indices in split_indices.items()
        },
        "split_rows": {split: len(indices) for split, indices in split_indices.items()},
    }
    contract["canonical_sha256"] = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    if array_loading == "lazy":
        target_latent_stack: np.ndarray | _LazyVerifiedArrayStack = _LazyVerifiedArrayStack(
            tuple(target_latents),
            dtype=np.float16,
            item_shape=(8, 8, 64, 64),
            verify_hashes=verify_hashes,
        )
        reference_latent_stack: np.ndarray | _LazyVerifiedArrayStack = _LazyVerifiedArrayStack(
            tuple(reference_latents),
            dtype=np.float16,
            item_shape=(8, 64, 64),
            verify_hashes=verify_hashes,
        )
        target_rgba_stack: np.ndarray | _LazyVerifiedArrayStack = _LazyVerifiedArrayStack(
            tuple(target_rgba),
            dtype=np.uint8,
            item_shape=(8, 128, 128, 4),
            verify_hashes=verify_hashes,
        )
    else:
        target_latent_stack = np.ascontiguousarray(np.stack(target_latents))
        reference_latent_stack = np.ascontiguousarray(np.stack(reference_latents))
        target_rgba_stack = np.ascontiguousarray(np.stack(target_rgba))
    return LatentMotionTrainingCorpus(
        rows=row_tuple,
        train_indices=split_indices["train"],
        validation_indices=split_indices["validation"],
        test_indices=split_indices["test"],
        action_vocabulary=actions,
        target_latents=target_latent_stack,
        reference_latents=reference_latent_stack,
        target_rgba=target_rgba_stack,
        phases=np.ascontiguousarray(np.stack(phases)),
        channel_mean=mean,
        channel_standard_deviation=std,
        autoencoder_checkpoint_path=autoencoder_path,
        autoencoder_architecture=architecture,
        contract=contract,
    )


def run_latent_motion_training(
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    config: LatentMotionTrainingConfig | None = None,
    resume_checkpoint_path: Path | str | None = None,
    expected_resume_sha256: str | None = None,
    warm_start_checkpoint_path: Path | str | None = None,
    expected_warm_start_sha256: str | None = None,
    disk_guard: DiskGuard | None = None,
) -> LatentMotionTrainingResult:
    """Train a no-clobber, resumable, held-out-identity motion model."""

    runtime = _require_torch()
    experiment = config or LatentMotionTrainingConfig()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace latent-motion output: {output}")
    if (resume_checkpoint_path is None) != (expected_resume_sha256 is None):
        raise ValueError("resume checkpoint and expected SHA-256 must be supplied together")
    if (warm_start_checkpoint_path is None) != (expected_warm_start_sha256 is None):
        raise ValueError("warm-start checkpoint and expected SHA-256 must be supplied together")
    if resume_checkpoint_path is not None and warm_start_checkpoint_path is not None:
        raise ValueError("resume and warm-start checkpoints are mutually exclusive")
    device = runtime.device(experiment.device)
    if device.type == "cuda" and not runtime.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if experiment.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 latent-motion training requires CUDA")
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(12 * 1024**3, label="broad latent-motion training")
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
            warm_start_checkpoint_path=(
                Path(warm_start_checkpoint_path).resolve()
                if warm_start_checkpoint_path is not None
                else None
            ),
            expected_warm_start_sha256=expected_warm_start_sha256,
            disk_guard=guard,
        )


def evaluate_latent_motion_checkpoint(
    manifest_path: Path | str,
    checkpoint_path: Path | str,
    output_directory: Path | str,
    *,
    expected_checkpoint_sha256: str,
    maximum_test_pairs: int = 64,
    preview_pairs: int = 6,
    seed: int = 20260826,
    device: str = "cuda",
    disk_guard: DiskGuard | None = None,
) -> LatentMotionEvaluationResult:
    """Safe-load and evaluate one EMA checkpoint on untouched test identities."""

    runtime = _require_torch()
    for name, value in (
        ("maximum_test_pairs", maximum_test_pairs),
        ("preview_pairs", preview_pairs),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace latent-motion evaluation: {output}")
    checkpoint_file = Path(checkpoint_path).resolve()
    actual_sha256 = _file_sha256(checkpoint_file)
    if actual_sha256 != expected_checkpoint_sha256:
        raise LatentMotionTrainingError("inference checkpoint SHA-256 mismatch")
    try:
        checkpoint = runtime.load(checkpoint_file, map_location="cpu", weights_only=True)
    except Exception as error:
        raise LatentMotionTrainingError("inference checkpoint failed safe load") from error
    checkpoint_kind = _ema_checkpoint_artifact_kind(checkpoint)
    corpus = load_latent_motion_training_corpus(
        manifest_path, verify_hashes=True, array_loading="lazy"
    )
    if checkpoint.get("corpus") != corpus.contract:
        raise LatentMotionTrainingError("inference checkpoint corpus differs")
    if _checkpoint_action_vocabulary(checkpoint, checkpoint_kind) != list(corpus.action_vocabulary):
        raise LatentMotionTrainingError("inference checkpoint action vocabulary differs")
    checkpoint_config = _config_from_dict(checkpoint.get("config"))
    runtime_device = runtime.device(device)
    if runtime_device.type == "cuda" and not runtime.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = _build_action_conditioned_motion_model(
        checkpoint_config, len(corpus.action_vocabulary)
    ).to(runtime_device)
    try:
        model.load_state_dict(checkpoint["ema_model"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise LatentMotionTrainingError("inference checkpoint model state differs") from error
    model.eval()
    decoder = _load_frozen_decoder(runtime, corpus, device=runtime_device)
    selection = _balanced_matched_pairs(corpus, corpus.test_indices, maximum_test_pairs)
    training_selection = _balanced_matched_pairs(corpus, corpus.train_indices, maximum_test_pairs)
    mean = runtime.tensor(corpus.channel_mean, device=runtime_device).view(1, 1, 8, 1, 1)
    std = runtime.tensor(corpus.channel_standard_deviation, device=runtime_device).view(
        1, 1, 8, 1, 1
    )
    dtype = runtime.bfloat16 if runtime_device.type == "cuda" else runtime.float32
    autocast = runtime_device.type == "cuda"
    metrics = _validate(
        runtime,
        corpus,
        selection,
        model,
        decoder,
        device=runtime_device,
        dtype=dtype,
        autocast=autocast,
        mean=mean,
        std=std,
        seed=seed,
        inference_steps=checkpoint_config.inference_steps,
        sampler_algorithm=checkpoint_config.sampler_algorithm,
    )
    direct_endpoint_metrics = None
    training_metrics = _validate(
        runtime,
        corpus,
        training_selection,
        model,
        decoder,
        device=runtime_device,
        dtype=dtype,
        autocast=autocast,
        mean=mean,
        std=std,
        seed=seed,
        inference_steps=checkpoint_config.inference_steps,
        sampler_algorithm=checkpoint_config.sampler_algorithm,
    )
    training_direct_endpoint_metrics = None
    if checkpoint_config.time_sampling == "uniform":
        direct_endpoint_metrics = _validate(
            runtime,
            corpus,
            selection,
            model,
            decoder,
            device=runtime_device,
            dtype=dtype,
            autocast=autocast,
            mean=mean,
            std=std,
            seed=seed,
            inference_steps=1,
            sampler_algorithm="euler",
        )
        training_direct_endpoint_metrics = _validate(
            runtime,
            corpus,
            training_selection,
            model,
            decoder,
            device=runtime_device,
            dtype=dtype,
            autocast=autocast,
            mean=mean,
            std=std,
            seed=seed,
            inference_steps=1,
            sampler_algorithm="euler",
        )
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(512 * 1024**2, label="latent-motion test evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        previews = _export_validation_previews(
            runtime,
            corpus,
            selection[:preview_pairs],
            model,
            decoder,
            output=stage / "previews",
            device=runtime_device,
            dtype=dtype,
            autocast=autocast,
            mean=mean,
            std=std,
            seed=seed,
            inference_steps=checkpoint_config.inference_steps,
            sampler_algorithm=checkpoint_config.sampler_algorithm,
            disk_guard=guard,
        )
        direct_endpoint_previews = None
        training_previews = _export_validation_previews(
            runtime,
            corpus,
            training_selection[:preview_pairs],
            model,
            decoder,
            output=stage / "training-previews",
            device=runtime_device,
            dtype=dtype,
            autocast=autocast,
            mean=mean,
            std=std,
            seed=seed,
            inference_steps=checkpoint_config.inference_steps,
            sampler_algorithm=checkpoint_config.sampler_algorithm,
            disk_guard=guard,
        )
        training_direct_endpoint_previews = None
        if direct_endpoint_metrics is not None:
            direct_endpoint_previews = _export_validation_previews(
                runtime,
                corpus,
                selection[:preview_pairs],
                model,
                decoder,
                output=stage / "direct-endpoint-previews",
                device=runtime_device,
                dtype=dtype,
                autocast=autocast,
                mean=mean,
                std=std,
                seed=seed,
                inference_steps=1,
                sampler_algorithm="euler",
                disk_guard=guard,
            )
            training_direct_endpoint_previews = _export_validation_previews(
                runtime,
                corpus,
                training_selection[:preview_pairs],
                model,
                decoder,
                output=stage / "training-direct-endpoint-previews",
                device=runtime_device,
                dtype=dtype,
                autocast=autocast,
                mean=mean,
                std=std,
                seed=seed,
                inference_steps=1,
                sampler_algorithm="euler",
                disk_guard=guard,
            )
        report = {
            "artifact_kind": "mugen_reference_latent_motion_test_evaluation",
            "checkpoint": {
                "artifact_kind": checkpoint_kind,
                "file_sha256": actual_sha256,
                "step": checkpoint.get("step"),
            },
            "claim": "untouched identity-disjoint test pairs; no open-domain generalization claim",
            "corpus": corpus.contract,
            "direct_endpoint_control": (
                {
                    "claim": (
                        "same checkpoint, requests, phases, order, and fixed noise; "
                        "one-step t=1 endpoint diagnostic"
                    ),
                    "metrics": direct_endpoint_metrics,
                    "previews": direct_endpoint_previews,
                    "sampling": {"algorithm": "euler", "steps": 1},
                }
                if direct_endpoint_metrics is not None
                else None
            ),
            "metrics": metrics,
            "pairs": [
                {
                    "identity_id": corpus.rows[left].identity_id,
                    "left_sequence_id": corpus.rows[left].sequence_id,
                    "left_verb": corpus.rows[left].verb,
                    "right_sequence_id": corpus.rows[right].sequence_id,
                    "right_verb": corpus.rows[right].verb,
                }
                for left, right in selection
            ],
            "previews": previews,
            "training_distribution_control": {
                "claim": (
                    "exact training identities with target-distinct balanced action pairs; "
                    "memorization/optimization evidence only"
                ),
                "direct_endpoint_control": (
                    {
                        "claim": (
                            "same training requests, phases, order, and fixed noise; "
                            "one-step t=1 endpoint diagnostic"
                        ),
                        "metrics": training_direct_endpoint_metrics,
                        "previews": training_direct_endpoint_previews,
                        "sampling": {"algorithm": "euler", "steps": 1},
                    }
                    if training_direct_endpoint_metrics is not None
                    else None
                ),
                "metrics": training_metrics,
                "pairs": [
                    {
                        "identity_id": corpus.rows[left].identity_id,
                        "left_sequence_id": corpus.rows[left].sequence_id,
                        "left_verb": corpus.rows[left].verb,
                        "right_sequence_id": corpus.rows[right].sequence_id,
                        "right_verb": corpus.rows[right].verb,
                    }
                    for left, right in training_selection
                ],
                "previews": training_previews,
                "sampling": {
                    "algorithm": checkpoint_config.sampler_algorithm,
                    "steps": checkpoint_config.inference_steps,
                },
            },
            "runtime": _runtime_facts(runtime, runtime_device),
            "sampling": {
                "algorithm": checkpoint_config.sampler_algorithm,
                "steps": checkpoint_config.inference_steps,
            },
            "schema_version": 2,
            "seed": seed,
        }
        payload = canonical_json_bytes(report)
        (stage / "evaluation-report.json").write_bytes(payload)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return LatentMotionEvaluationResult(
        output_directory=output,
        report_path=output / "evaluation-report.json",
        report_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _ema_checkpoint_artifact_kind(checkpoint: Any) -> str:
    if not isinstance(checkpoint, dict):
        raise LatentMotionTrainingError("EMA checkpoint must be an object")
    artifact_kind = checkpoint.get("artifact_kind")
    allowed = {
        "mugen_reference_latent_motion_ema_inference_checkpoint",
        "mugen_reference_latent_motion_resume_checkpoint",
    }
    if artifact_kind not in allowed:
        raise LatentMotionTrainingError("EMA checkpoint has the wrong artifact kind")
    return artifact_kind


def _checkpoint_action_vocabulary(checkpoint: dict[str, Any], artifact_kind: str) -> Any:
    vocabulary = checkpoint.get("action_vocabulary")
    if vocabulary is None and artifact_kind == "mugen_reference_latent_motion_resume_checkpoint":
        corpus = checkpoint.get("corpus")
        if isinstance(corpus, dict):
            vocabulary = corpus.get("action_vocabulary")
    return vocabulary


def _train(
    runtime: Any,
    *,
    corpus: LatentMotionTrainingCorpus,
    output: Path,
    history: Any,
    config: LatentMotionTrainingConfig,
    device: Any,
    resume_checkpoint_path: Path | None,
    expected_resume_sha256: str | None,
    warm_start_checkpoint_path: Path | None,
    expected_warm_start_sha256: str | None,
    disk_guard: DiskGuard,
) -> LatentMotionTrainingResult:
    runtime.manual_seed(config.seed)
    if device.type == "cuda":
        runtime.cuda.manual_seed_all(config.seed)
    model = _build_action_conditioned_motion_model(config, len(corpus.action_vocabulary)).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = runtime.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        fused=device.type == "cuda",
    )
    decoder = _load_frozen_decoder(runtime, corpus, device=device)
    sampler_generator = runtime.Generator(device="cpu").manual_seed(config.seed + 1)
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
            raise LatentMotionTrainingError("resume step must be below cumulative steps")
        model.load_state_dict(parent["raw_model"], strict=True)
        ema.load_state_dict(parent["ema_model"], strict=True)
        optimizer.load_state_dict(parent["optimizer"])
        sampler_generator.set_state(parent["rng_state"]["sampler"])
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
    elif warm_start_checkpoint_path is not None:
        assert expected_warm_start_sha256 is not None
        parent = _load_warm_start(
            runtime,
            warm_start_checkpoint_path,
            expected_sha256=expected_warm_start_sha256,
            corpus=corpus,
            config=config,
        )
        parent_config = _config_from_dict(parent["config"])
        _load_warm_start_model_state(
            model,
            parent["ema_model"],
            parent_config=parent_config,
            current_config=config,
        )
        _load_warm_start_model_state(
            ema,
            parent["ema_model"],
            parent_config=parent_config,
            current_config=config,
        )
        lineage = {
            "initialization": (
                "ema_weights_plus_new_expanded_action_conditioning"
                if parent_config.action_conditioning_mode != config.action_conditioning_mode
                else "ema_weights_only_fresh_optimizer_and_rng"
            ),
            "parent_checkpoint_path": str(warm_start_checkpoint_path),
            "parent_checkpoint_sha256": expected_warm_start_sha256,
            "parent_step": int(parent["step"]),
        }
    train_index = build_matched_action_index(corpus.rows, corpus.train_indices)
    target_digests = _target_rgba_digests(corpus, corpus.train_indices)
    train_pairs = _target_distinct_pairs_from_index(train_index, target_digests)
    train_bundles = _target_distinct_bundles_from_index(train_index, target_digests)
    training_evaluation_selection = _balanced_matched_pairs(
        corpus, corpus.train_indices, config.validation_pairs
    )
    validation_selection = _validation_pairs(corpus, config.validation_pairs)
    mean = runtime.tensor(corpus.channel_mean, device=device).view(1, 1, 8, 1, 1)
    std = runtime.tensor(corpus.channel_standard_deviation, device=device).view(1, 1, 8, 1, 1)
    dtype = runtime.bfloat16 if config.precision == "bfloat16" else runtime.float32
    autocast = config.precision == "bfloat16"
    model.train()
    latest_validation = None
    for step_index in range(start_step, config.steps):
        step = step_index + 1
        learning_rate = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated_latent = 0.0
        accumulated_pixel = 0.0
        accumulated_action_contrast = 0.0
        accumulated_pixel_action_contrast = 0.0
        accumulated_temporal_motion = 0.0
        accumulated_target_directed_motion = 0.0
        accumulated_loss = 0.0
        for _ in range(config.gradient_accumulation):
            selection = (
                _sample_target_distinct_bundle(train_bundles, generator=sampler_generator)
                if config.action_batch_mode == "bundle"
                else _sample_target_distinct_pair(train_pairs, generator=sampler_generator)
            )
            target, reference, target_rgba, phases, actions = _batch(
                runtime, corpus, selection, device=device, mean=mean, std=std
            )
            clean = target - reference.unsqueeze(1)
            shared_noise = runtime.randn(
                (1, *clean.shape[1:]), device=device, generator=noise_generator
            ).expand_as(clean)
            times = _sample_training_times(
                runtime,
                batch=len(selection),
                config=config,
                device=device,
                generator=noise_generator,
            )
            time_view = _time_batch_view(times, batch=len(selection))
            noisy = (1 - time_view) * clean + time_view * shared_noise
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                predicted = model(
                    noisy,
                    reference,
                    times,
                    actions,
                    frame_phase=phases,
                )
                latent_loss = runtime.nn.functional.mse_loss(
                    predicted.float(), (shared_noise - clean).float()
                )
                generated_residual = noisy - time_view * predicted
                action_contrast_loss = (
                    _matched_action_contrast_loss(
                        runtime,
                        estimated_clean=generated_residual.float(),
                        target_clean=clean.float(),
                    )
                    if config.action_contrast_weight
                    else latent_loss.new_zeros(())
                )
                generated_latent = (reference.unsqueeze(1) + generated_residual) * std + mean
                logits = decoder.decode_logits(generated_latent.reshape(-1, 8, 64, 64))
                pixel_loss = sprite_reconstruction_loss(logits, target_rgba).total
                predicted_rgba = runtime.sigmoid(logits).reshape(len(selection), 8, 4, 128, 128)
                target_rgba_5d = target_rgba.reshape(len(selection), 8, 4, 128, 128)
                pixel_action_contrast_loss = (
                    (
                        _matched_pixel_action_bundle_loss(
                            runtime,
                            predicted_rgba=predicted_rgba,
                            target_rgba=target_rgba_5d,
                        )
                        if config.action_batch_mode == "bundle"
                        else _matched_pixel_action_contrast_loss(
                            runtime,
                            predicted_rgba=predicted_rgba,
                            target_rgba=target_rgba_5d,
                        )
                    )
                    if config.pixel_action_contrast_weight
                    else latent_loss.new_zeros(())
                )
                temporal_motion_loss = (
                    _temporal_motion_anti_collapse_loss(
                        runtime,
                        predicted_rgba=predicted_rgba,
                        target_rgba=target_rgba_5d,
                    )
                    if config.temporal_motion_weight
                    else latent_loss.new_zeros(())
                )
                target_directed_motion_loss = (
                    _target_directed_motion_floor_loss(
                        runtime,
                        predicted_rgba=predicted_rgba,
                        target_rgba=target_rgba_5d,
                        minimum_progress=config.minimum_target_motion_progress,
                    )
                    if config.target_directed_motion_weight
                    else latent_loss.new_zeros(())
                )
                loss = (
                    config.latent_endpoint_weight * latent_loss
                    + config.pixel_endpoint_weight * pixel_loss
                    + config.action_contrast_weight * action_contrast_loss
                    + config.pixel_action_contrast_weight * pixel_action_contrast_loss
                    + config.temporal_motion_weight * temporal_motion_loss
                    + config.target_directed_motion_weight * target_directed_motion_loss
                )
                scaled = loss / config.gradient_accumulation
            if not bool(runtime.isfinite(scaled)):
                raise RuntimeError(f"non-finite latent-motion loss at step {step}")
            scaled.backward()
            accumulated_latent += float(latent_loss.detach().cpu())
            accumulated_pixel += float(pixel_loss.detach().cpu())
            accumulated_action_contrast += float(action_contrast_loss.detach().cpu())
            accumulated_pixel_action_contrast += float(pixel_action_contrast_loss.detach().cpu())
            accumulated_temporal_motion += float(temporal_motion_loss.detach().cpu())
            accumulated_target_directed_motion += float(target_directed_motion_loss.detach().cpu())
            accumulated_loss += float(loss.detach().cpu())
        gradient_norm = float(
            runtime.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
            .detach()
            .cpu()
        )
        optimizer.step()
        _ema_update(
            runtime,
            ema,
            model,
            0.0 if step <= config.warmup_steps else config.ema_decay,
        )
        if step == 1 or step % config.validate_every == 0 or step == config.steps:
            latest_validation = _validate(
                runtime,
                corpus,
                validation_selection,
                ema,
                decoder,
                device=device,
                dtype=dtype,
                autocast=autocast,
                mean=mean,
                std=std,
                seed=config.seed + 20_000,
                inference_steps=config.inference_steps,
                sampler_algorithm=config.sampler_algorithm,
            )
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            row = {
                "action_contrast_loss": (
                    accumulated_action_contrast / config.gradient_accumulation
                ),
                "gradient_norm_before_clip": gradient_norm,
                "latent_endpoint_loss": accumulated_latent / config.gradient_accumulation,
                "learning_rate": learning_rate,
                "loss": accumulated_loss / config.gradient_accumulation,
                "pixel_endpoint_loss": accumulated_pixel / config.gradient_accumulation,
                "pixel_action_contrast_loss": (
                    accumulated_pixel_action_contrast / config.gradient_accumulation
                ),
                "temporal_motion_loss": (
                    accumulated_temporal_motion / config.gradient_accumulation
                ),
                "target_directed_motion_loss": (
                    accumulated_target_directed_motion / config.gradient_accumulation
                ),
                "step": step,
                "validation": latest_validation
                if step == 1 or step % config.validate_every == 0
                else None,
            }
            history.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            history.flush()
            os.fsync(history.fileno())
        if step % config.checkpoint_every == 0 or step == config.steps:
            _write_training_checkpoint(
                runtime,
                output / f"training-step-{step:07d}.pt",
                corpus=corpus,
                config=config,
                step=step,
                model=model,
                ema=ema,
                optimizer=optimizer,
                sampler_generator=sampler_generator,
                noise_generator=noise_generator,
                device=device,
                disk_guard=disk_guard,
            )
    final_checkpoint = output / f"training-step-{config.steps:07d}.pt"
    inference_checkpoint = output / "checkpoint-ema.pt"
    _atomic_torch_save(
        runtime,
        inference_checkpoint,
        {
            "action_vocabulary": list(corpus.action_vocabulary),
            "artifact_kind": "mugen_reference_latent_motion_ema_inference_checkpoint",
            "config": asdict(config),
            "corpus": corpus.contract,
            "ema_policy": _ema_policy(config),
            "ema_model": ema.state_dict(),
            "normalization": {
                "channel_mean": list(corpus.channel_mean),
                "channel_standard_deviation": list(corpus.channel_standard_deviation),
            },
            "step": config.steps,
        },
        disk_guard=disk_guard,
    )
    preview_rows = _export_validation_previews(
        runtime,
        corpus,
        validation_selection[: config.preview_pairs],
        ema,
        decoder,
        output=output / "previews",
        device=device,
        dtype=dtype,
        autocast=autocast,
        mean=mean,
        std=std,
        seed=config.seed + 20_000,
        inference_steps=config.inference_steps,
        sampler_algorithm=config.sampler_algorithm,
        disk_guard=disk_guard,
    )
    final_training_evaluation = _validate(
        runtime,
        corpus,
        training_evaluation_selection,
        ema,
        decoder,
        device=device,
        dtype=dtype,
        autocast=autocast,
        mean=mean,
        std=std,
        seed=config.seed + 30_000,
        inference_steps=config.inference_steps,
        sampler_algorithm=config.sampler_algorithm,
    )
    training_preview_rows = _export_validation_previews(
        runtime,
        corpus,
        training_evaluation_selection[: config.preview_pairs],
        ema,
        decoder,
        output=output / "training-previews",
        device=device,
        dtype=dtype,
        autocast=autocast,
        mean=mean,
        std=std,
        seed=config.seed + 30_000,
        inference_steps=config.inference_steps,
        sampler_algorithm=config.sampler_algorithm,
        disk_guard=disk_guard,
    )
    report = {
        "artifact_kind": "mugen_reference_latent_motion_training",
        "claim": (
            "matched in-distribution training replay plus identity-disjoint validation "
            "of canonical reference and action-token motion"
        ),
        "config": asdict(config),
        "corpus": corpus.contract,
        "ema_policy": _ema_policy(config),
        "final_training_evaluation": final_training_evaluation,
        "final_validation": latest_validation,
        "history": {
            "file_sha256": _file_sha256(output / "training-history.jsonl"),
            "path": "training-history.jsonl",
        },
        "inference_checkpoint": {
            "file_sha256": _file_sha256(inference_checkpoint),
            "path": inference_checkpoint.name,
        },
        "lineage": lineage,
        "previews": preview_rows,
        "runtime": _runtime_facts(runtime, device),
        "step": config.steps,
        "training_previews": training_preview_rows,
        "training_checkpoint": {
            "file_sha256": _file_sha256(final_checkpoint),
            "path": final_checkpoint.name,
        },
    }
    report_path = output / "training-report.json"
    report_payload = canonical_json_bytes(report)
    _atomic_bytes(report_path, report_payload, disk_guard=disk_guard)
    return LatentMotionTrainingResult(
        output_directory=output,
        report_path=report_path,
        training_checkpoint_path=final_checkpoint,
        inference_checkpoint_path=inference_checkpoint,
        report_sha256=hashlib.sha256(report_payload).hexdigest(),
    )


def _batch(
    runtime: Any,
    corpus: LatentMotionTrainingCorpus,
    selection: tuple[int, ...],
    *,
    device: Any,
    mean: Any,
    std: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    indices = list(selection)
    target = runtime.from_numpy(corpus.target_latents[indices].astype(np.float32)).to(device)
    reference = runtime.from_numpy(corpus.reference_latents[indices].astype(np.float32)).to(device)
    target = (target - mean) / std
    reference = (reference - mean[:, 0]) / std[:, 0]
    rgba = (
        runtime.from_numpy(corpus.target_rgba[indices])
        .to(device=device, dtype=runtime.float32)
        .permute(0, 1, 4, 2, 3)
        .reshape(-1, 4, 128, 128)
        .div(255)
    )
    phases = runtime.from_numpy(corpus.phases[indices]).to(device)
    actions = runtime.tensor([corpus.rows[index].action_index for index in indices], device=device)
    return target, reference, rgba, phases, actions


def _validation_pairs(
    corpus: LatentMotionTrainingCorpus, maximum_pairs: int
) -> tuple[tuple[int, int], ...]:
    return _balanced_matched_pairs(corpus, corpus.validation_indices, maximum_pairs)


def _matched_pairs(
    corpus: LatentMotionTrainingCorpus,
    indices: tuple[int, ...],
    maximum_pairs: int,
) -> tuple[tuple[int, int], ...]:
    index = build_matched_action_index(corpus.rows, indices)
    digests = _target_rgba_digests(corpus, indices)
    distinct = _target_distinct_pairs_from_index(index, digests)
    output = []
    for identity in sorted(index, key=str.encode):
        candidates = distinct.get(identity)
        if not candidates:
            continue
        output.append(candidates[0])
        if len(output) == maximum_pairs:
            break
    if not output:
        raise LatentMotionTrainingError("split has no matched action pairs")
    return tuple(output)


def _balanced_matched_pairs(
    corpus: LatentMotionTrainingCorpus,
    indices: tuple[int, ...],
    maximum_pairs: int,
) -> tuple[tuple[int, int], ...]:
    """Select action contrasts across identities and verb-pair families.

    Every multi-action identity contributes before any receives a second pair.
    Within that constraint, selection prefers unused and underrepresented verb
    pairs so lexical action order cannot dominate broad held-out evidence.
    """

    index = build_matched_action_index(corpus.rows, indices)
    return _balanced_pairs_from_index(
        index,
        maximum_pairs,
        target_digests=_target_rgba_digests(corpus, indices),
    )


def _balanced_pairs_from_index(
    index: dict[str, dict[str, int]],
    maximum_pairs: int,
    *,
    target_digests: dict[int, str] | None = None,
) -> tuple[tuple[int, int], ...]:
    if isinstance(maximum_pairs, bool) or not isinstance(maximum_pairs, int):
        raise ValueError("maximum_pairs must be an integer")
    if maximum_pairs <= 0:
        raise ValueError("maximum_pairs must be positive")
    candidates: dict[str, list[tuple[str, str, int, int]]] = {}
    pair_frequency: defaultdict[tuple[str, str], int] = defaultdict(int)
    for identity, actions in index.items():
        verbs = tuple(actions)
        identity_candidates = []
        for left, right in combinations(range(len(verbs)), 2):
            left_verb, right_verb = verbs[left], verbs[right]
            if target_digests is not None and (
                target_digests[actions[left_verb]] == target_digests[actions[right_verb]]
            ):
                continue
            identity_candidates.append(
                (left_verb, right_verb, actions[left_verb], actions[right_verb])
            )
            pair_frequency[(left_verb, right_verb)] += 1
        candidates[identity] = identity_candidates

    output: list[tuple[int, int]] = []
    selected_per_identity: defaultdict[str, int] = defaultdict(int)
    selected_per_verb: defaultdict[str, int] = defaultdict(int)
    selected_pairs: set[tuple[str, str]] = set()
    remaining = {
        identity: list(identity_candidates) for identity, identity_candidates in candidates.items()
    }

    def candidate_key(identity: str, candidate: tuple[str, str, int, int]) -> tuple[Any, ...]:
        left_verb, right_verb, _left_index, _right_index = candidate
        verb_pair = (left_verb, right_verb)
        return (
            selected_per_identity[identity],
            verb_pair in selected_pairs,
            selected_per_verb[left_verb] + selected_per_verb[right_verb],
            pair_frequency[verb_pair],
            left_verb.encode(),
            right_verb.encode(),
            identity.encode(),
        )

    while len(output) < maximum_pairs:
        choices = [
            (candidate_key(identity, candidate), identity, candidate)
            for identity, identity_candidates in remaining.items()
            for candidate in identity_candidates
        ]
        if not choices:
            break
        _key, identity, candidate = min(choices, key=lambda choice: choice[0])
        left_verb, right_verb, left_index, right_index = candidate
        output.append((left_index, right_index))
        selected_per_identity[identity] += 1
        selected_per_verb[left_verb] += 1
        selected_per_verb[right_verb] += 1
        selected_pairs.add((left_verb, right_verb))
        remaining[identity].remove(candidate)
    if not output:
        raise LatentMotionTrainingError("split has no matched action pairs")
    return tuple(output)


def _target_rgba_digests(
    corpus: LatentMotionTrainingCorpus, indices: tuple[int, ...]
) -> dict[int, str]:
    if isinstance(corpus.target_rgba, _LazyVerifiedArrayStack):
        return {index: corpus.target_rgba.array_content_sha256(index) for index in indices}
    return {index: _array_sha256(corpus.target_rgba[index]) for index in indices}


def _validate(
    runtime: Any,
    corpus: LatentMotionTrainingCorpus,
    selection: tuple[tuple[int, int], ...],
    model: Any,
    decoder: Any,
    *,
    device: Any,
    dtype: Any,
    autocast: bool,
    mean: Any,
    std: Any,
    seed: int,
    inference_steps: int,
    sampler_algorithm: FlowSampler,
) -> dict[str, float]:
    model.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    verb_totals: defaultdict[str, float] = defaultdict(float)
    verb_counts: defaultdict[str, int] = defaultdict(int)
    generator = runtime.Generator(device=device).manual_seed(seed)
    with runtime.no_grad():
        for pair in selection:
            target, reference, target_rgba, phases, actions = _batch(
                runtime, corpus, pair, device=device, mean=mean, std=std
            )
            clean = target - reference.unsqueeze(1)
            noise = runtime.randn(
                (1, *clean.shape[1:]), device=device, generator=generator
            ).expand_as(clean)
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                endpoint_times = runtime.ones((2,), device=device)
                correct_endpoint = model(
                    noise, reference, endpoint_times, actions, frame_phase=phases
                )
                permuted_endpoint = model(
                    noise,
                    reference,
                    endpoint_times,
                    runtime.flip(actions, (0,)),
                    frame_phase=phases,
                )
                target_velocity = noise - clean
                correct_mse = runtime.nn.functional.mse_loss(
                    correct_endpoint.float(), target_velocity.float()
                )
                permuted_mse = runtime.nn.functional.mse_loss(
                    permuted_endpoint.float(), target_velocity.float()
                )
                generated_residual = _sample_motion_residual(
                    runtime,
                    model,
                    noise=noise,
                    reference=reference,
                    actions=actions,
                    phases=phases,
                    inference_steps=inference_steps,
                    sampler_algorithm=sampler_algorithm,
                )
                generated_permuted_residual = _sample_motion_residual(
                    runtime,
                    model,
                    noise=noise,
                    reference=reference,
                    actions=runtime.flip(actions, (0,)),
                    phases=phases,
                    inference_steps=inference_steps,
                    sampler_algorithm=sampler_algorithm,
                )
                generated = (reference.unsqueeze(1) + generated_residual) * std + mean
                generated_permuted = (
                    reference.unsqueeze(1) + generated_permuted_residual
                ) * std + mean
                logits = decoder.decode_logits(generated.reshape(-1, 8, 64, 64))
                permuted_logits = decoder.decode_logits(generated_permuted.reshape(-1, 8, 64, 64))
                reconstruction = sprite_reconstruction_loss(logits, target_rgba)
            predicted_rgba = runtime.sigmoid(logits.float()).reshape(2, 8, 4, 128, 128)
            permuted_rgba = runtime.sigmoid(permuted_logits.float()).reshape(2, 8, 4, 128, 128)
            target_5d = target_rgba.reshape(2, 8, 4, 128, 128)
            predicted_alpha = predicted_rgba[:, :, 3:4]
            target_alpha = target_5d[:, :, 3:4]
            predicted_pm = runtime.cat(
                (predicted_rgba[:, :, :3] * predicted_alpha, predicted_alpha), dim=2
            )
            permuted_alpha = permuted_rgba[:, :, 3:4]
            permuted_pm = runtime.cat(
                (permuted_rgba[:, :, :3] * permuted_alpha, permuted_alpha), dim=2
            )
            target_pm = runtime.cat((target_5d[:, :, :3] * target_alpha, target_alpha), dim=2)
            causal = _paired_action_metrics(
                runtime,
                predicted_pm=predicted_pm,
                permuted_pm=permuted_pm,
                target_pm=target_pm,
            )
            appearance = _paired_appearance_metrics(
                runtime,
                predicted_pm=predicted_pm,
                target_pm=target_pm,
            )
            motion = _paired_temporal_motion_metrics(
                runtime,
                predicted_pm=predicted_pm,
                target_pm=target_pm,
            )
            temporal = runtime.nn.functional.l1_loss(
                predicted_pm[:, 1:] - predicted_pm[:, :-1],
                target_pm[:, 1:] - target_pm[:, :-1],
            )
            totals["latent_endpoint_mse"] += float(correct_mse.cpu())
            totals["action_permuted_mse"] += float(permuted_mse.cpu())
            totals["premultiplied_rgba_mae"] += float(
                runtime.nn.functional.l1_loss(predicted_pm, target_pm).cpu()
            )
            totals["temporal_delta_mae"] += float(temporal.cpu())
            totals["decoder_reconstruction_loss"] += float(reconstruction.total.cpu())
            for key, value in {**causal, **appearance, **motion}.items():
                totals[key] += float(value.cpu())
            for offset, row_index in enumerate(pair):
                verb = corpus.rows[row_index].verb
                verb_totals[verb] += float(
                    runtime.nn.functional.l1_loss(predicted_pm[offset], target_pm[offset]).cpu()
                )
                verb_counts[verb] += 1
                predicted_delta = predicted_pm[offset, 1:] - predicted_pm[offset, :-1]
                target_delta = target_pm[offset, 1:] - target_pm[offset, :-1]
                generated_motion = predicted_delta.abs().mean()
                target_motion = target_delta.abs().mean()
                totals[f"verb_{verb}_generated_temporal_magnitude"] += float(generated_motion.cpu())
                totals[f"verb_{verb}_target_temporal_magnitude"] += float(target_motion.cpu())
    model.train()
    count = len(selection)
    output = {key: value / count for key, value in totals.items()}
    output["action_token_loss_delta"] = (
        output["action_permuted_mse"] - output["latent_endpoint_mse"]
    )
    if output["target_temporal_magnitude"] > 1e-8:
        output["temporal_motion_ratio"] = (
            output["generated_temporal_magnitude"] / output["target_temporal_magnitude"]
        )
    else:
        output["static_target_spurious_temporal_magnitude"] = output["generated_temporal_magnitude"]
    for verb in sorted(verb_totals, key=str.encode):
        output[f"verb_{verb}_premultiplied_rgba_mae"] = verb_totals[verb] / verb_counts[verb]
        generated_key = f"verb_{verb}_generated_temporal_magnitude"
        target_key = f"verb_{verb}_target_temporal_magnitude"
        generated_motion = totals[generated_key] / verb_counts[verb]
        target_motion = totals[target_key] / verb_counts[verb]
        output[generated_key] = generated_motion
        output[target_key] = target_motion
        if target_motion > 1e-8:
            output[f"verb_{verb}_temporal_motion_ratio"] = generated_motion / target_motion
        else:
            output[f"verb_{verb}_static_target_spurious_temporal_magnitude"] = generated_motion
    return output


def _paired_action_metrics(
    runtime: Any,
    *,
    predicted_pm: Any,
    permuted_pm: Any,
    target_pm: Any,
) -> dict[str, Any]:
    """Measure two-action separation and directed action-token substitution."""

    expected = tuple(target_pm.shape)
    if len(expected) != 5 or expected[0] != 2:
        raise ValueError("paired action tensors must have shape [2,T,C,H,W]")
    if tuple(predicted_pm.shape) != expected or tuple(permuted_pm.shape) != expected:
        raise ValueError("paired action tensors must share one shape")
    target_distance = runtime.nn.functional.l1_loss(target_pm[0], target_pm[1])
    generated_distance = runtime.nn.functional.l1_loss(predicted_pm[0], predicted_pm[1])
    separation_ratio = generated_distance / target_distance.clamp_min(1e-8)
    correct_preference = []
    replacement_movement = []
    margins = []
    for offset, replacement in ((0, 1), (1, 0)):
        correct_error = runtime.nn.functional.l1_loss(predicted_pm[offset], target_pm[offset])
        alternate_error = runtime.nn.functional.l1_loss(
            predicted_pm[offset], target_pm[replacement]
        )
        swapped_replacement_error = runtime.nn.functional.l1_loss(
            permuted_pm[offset], target_pm[replacement]
        )
        correct_preference.append((correct_error < alternate_error).to(runtime.float32))
        replacement_movement.append(
            (swapped_replacement_error < alternate_error).to(runtime.float32)
        )
        margins.append(alternate_error - correct_error)
    return {
        "action_separation_ratio": separation_ratio,
        "action_correct_target_preference_rate": runtime.stack(correct_preference).mean(),
        "action_swap_moves_toward_replacement_rate": runtime.stack(replacement_movement).mean(),
        "action_correct_target_margin": runtime.stack(margins).mean(),
        "target_action_distance": target_distance,
        "generated_action_distance": generated_distance,
    }


def _matched_action_contrast_loss(
    runtime: Any,
    *,
    estimated_clean: Any,
    target_clean: Any,
) -> Any:
    """Match the authored action delta for one same-identity two-action batch."""

    expected = tuple(target_clean.shape)
    if len(expected) != 5 or expected[0] != 2:
        raise ValueError("matched action tensors must have shape [2,T,C,H,W]")
    if tuple(estimated_clean.shape) != expected:
        raise ValueError("matched action tensors must share one shape")
    return runtime.nn.functional.mse_loss(
        estimated_clean[0] - estimated_clean[1],
        target_clean[0] - target_clean[1],
    )


def _matched_pixel_action_contrast_loss(
    runtime: Any,
    *,
    predicted_rgba: Any,
    target_rgba: Any,
) -> Any:
    """Force authored pixel deltas and correct action-target preference."""

    expected = tuple(target_rgba.shape)
    if len(expected) != 5 or expected[0] != 2 or expected[2] != 4:
        raise ValueError("matched pixel action tensors must have shape [2,T,4,H,W]")
    if tuple(predicted_rgba.shape) != expected:
        raise ValueError("matched pixel action tensors must share one shape")
    predicted_alpha = predicted_rgba[:, :, 3:4]
    target_alpha = target_rgba[:, :, 3:4]
    predicted_pm = runtime.cat((predicted_rgba[:, :, :3] * predicted_alpha, predicted_alpha), dim=2)
    target_pm = runtime.cat((target_rgba[:, :, :3] * target_alpha, target_alpha), dim=2)
    target_distance = runtime.nn.functional.l1_loss(target_pm[0], target_pm[1])
    delta_error = runtime.nn.functional.l1_loss(
        predicted_pm[0] - predicted_pm[1],
        target_pm[0] - target_pm[1],
    )
    own_error = 0.5 * (
        runtime.nn.functional.l1_loss(predicted_pm[0], target_pm[0])
        + runtime.nn.functional.l1_loss(predicted_pm[1], target_pm[1])
    )
    swapped_error = 0.5 * (
        runtime.nn.functional.l1_loss(predicted_pm[0], target_pm[1])
        + runtime.nn.functional.l1_loss(predicted_pm[1], target_pm[0])
    )
    denominator = target_distance.clamp_min(1e-4)
    preference = runtime.relu(own_error - swapped_error + 0.1 * target_distance)
    return delta_error / denominator + preference / denominator


def _matched_pixel_action_bundle_loss(
    runtime: Any,
    *,
    predicted_rgba: Any,
    target_rgba: Any,
) -> Any:
    """Separate every generated action from all other targets for one identity."""

    expected = tuple(target_rgba.shape)
    if len(expected) != 5 or expected[0] < 3 or expected[2] != 4:
        raise ValueError("matched pixel action bundles must have shape [B>=3,T,4,H,W]")
    if tuple(predicted_rgba.shape) != expected:
        raise ValueError("matched pixel action bundles must share one shape")
    predicted_alpha = predicted_rgba[:, :, 3:4]
    target_alpha = target_rgba[:, :, 3:4]
    predicted_pm = runtime.cat((predicted_rgba[:, :, :3] * predicted_alpha, predicted_alpha), dim=2)
    target_pm = runtime.cat((target_rgba[:, :, :3] * target_alpha, target_alpha), dim=2)
    own_errors = [
        runtime.nn.functional.l1_loss(predicted_pm[index], target_pm[index])
        for index in range(expected[0])
    ]
    delta_losses = []
    preference_losses = []
    for left, right in combinations(range(expected[0]), 2):
        target_distance = runtime.nn.functional.l1_loss(target_pm[left], target_pm[right])
        if float(target_distance.detach().cpu()) <= 1e-4:
            continue
        denominator = target_distance.clamp_min(1e-4)
        delta_losses.append(
            runtime.nn.functional.l1_loss(
                predicted_pm[left] - predicted_pm[right],
                target_pm[left] - target_pm[right],
            )
            / denominator
        )
        preference_losses.extend(
            (
                runtime.relu(
                    own_errors[left]
                    - runtime.nn.functional.l1_loss(predicted_pm[left], target_pm[right])
                    + 0.1 * target_distance
                )
                / denominator,
                runtime.relu(
                    own_errors[right]
                    - runtime.nn.functional.l1_loss(predicted_pm[right], target_pm[left])
                    + 0.1 * target_distance
                )
                / denominator,
            )
        )
    if not delta_losses:
        raise ValueError("matched pixel action bundle has no target-distinct pairs")
    return runtime.stack(delta_losses).mean() + runtime.stack(preference_losses).mean()


def _temporal_motion_anti_collapse_loss(
    runtime: Any,
    *,
    predicted_rgba: Any,
    target_rgba: Any,
) -> Any:
    """Penalize static averaging wherever the authored clip actually moves."""

    expected = tuple(target_rgba.shape)
    if len(expected) != 5 or expected[1] < 2 or expected[2] != 4:
        raise ValueError("temporal motion tensors must have shape [B,T>=2,4,H,W]")
    if tuple(predicted_rgba.shape) != expected:
        raise ValueError("temporal motion tensors must share one shape")
    predicted_alpha = predicted_rgba[:, :, 3:4]
    target_alpha = target_rgba[:, :, 3:4]
    predicted_pm = runtime.cat((predicted_rgba[:, :, :3] * predicted_alpha, predicted_alpha), dim=2)
    target_pm = runtime.cat((target_rgba[:, :, :3] * target_alpha, target_alpha), dim=2)
    predicted_delta = predicted_pm[:, 1:] - predicted_pm[:, :-1]
    target_delta = target_pm[:, 1:] - target_pm[:, :-1]
    target_magnitude = target_delta.abs().mean(dim=(1, 2, 3, 4))
    dynamic = target_magnitude > 1e-4
    if not bool(dynamic.any()):
        return predicted_rgba.sum() * 0

    predicted_delta = predicted_delta[dynamic]
    target_delta = target_delta[dynamic]
    target_magnitude = target_magnitude[dynamic]
    change_strength = target_delta.abs().amax(dim=2, keepdim=True)
    change_weight = 1 + 8 * change_strength
    delta_error = ((predicted_delta - target_delta).abs() * change_weight).sum(
        dim=(1, 2, 3, 4)
    ) / change_weight.expand_as(target_delta).sum(dim=(1, 2, 3, 4))
    normalized_delta_error = delta_error / target_magnitude.clamp_min(1e-4)
    predicted_magnitude = predicted_delta.abs().mean(dim=(1, 2, 3, 4))
    motion_shortfall = runtime.relu(target_magnitude - predicted_magnitude) / target_magnitude
    return (normalized_delta_error + motion_shortfall).mean()


def _target_directed_motion_floor_loss(
    runtime: Any,
    *,
    predicted_rgba: Any,
    target_rgba: Any,
    minimum_progress: float,
) -> Any:
    """Require signed progress along every authored non-static frame transition.

    A static prediction has zero progress, the exact target has progress one, and
    unrelated motion does not satisfy the floor. Unlike an absolute-motion hinge,
    this objective has a useful target-directed gradient even at a perfectly static
    prediction.
    """

    expected = tuple(target_rgba.shape)
    if len(expected) != 5 or expected[1] < 2 or expected[2] != 4:
        raise ValueError("target-directed motion tensors must have shape [B,T>=2,4,H,W]")
    if tuple(predicted_rgba.shape) != expected:
        raise ValueError("target-directed motion tensors must share one shape")
    if not math.isfinite(minimum_progress) or not 0 < minimum_progress <= 1:
        raise ValueError("minimum_progress must be in (0,1]")

    predicted_alpha = predicted_rgba[:, :, 3:4]
    target_alpha = target_rgba[:, :, 3:4]
    predicted_pm = runtime.cat((predicted_rgba[:, :, :3] * predicted_alpha, predicted_alpha), dim=2)
    target_pm = runtime.cat((target_rgba[:, :, :3] * target_alpha, target_alpha), dim=2)
    predicted_delta = predicted_pm[:, 1:] - predicted_pm[:, :-1]
    target_delta = target_pm[:, 1:] - target_pm[:, :-1]
    target_magnitude = target_delta.abs().mean(dim=(2, 3, 4))
    dynamic = target_magnitude > 1e-4
    if not bool(dynamic.any()):
        return predicted_rgba.sum() * 0

    change_strength = target_delta.abs().amax(dim=2, keepdim=True)
    change_weight = 1 + 8 * change_strength
    weighted_target_magnitude = (target_delta.abs() * change_weight).sum(dim=(2, 3, 4))
    directed_progress = (predicted_delta * target_delta.sign() * change_weight).sum(
        dim=(2, 3, 4)
    ) / weighted_target_magnitude.clamp_min(1e-8)
    shortfall = runtime.relu(minimum_progress - directed_progress)
    return shortfall[dynamic].mean()


def _paired_appearance_metrics(
    runtime: Any,
    *,
    predicted_pm: Any,
    target_pm: Any,
) -> dict[str, Any]:
    """Separate sprite fidelity from transparent-canvas performance."""

    expected = tuple(target_pm.shape)
    if len(expected) != 5 or expected[0] != 2 or expected[2] != 4:
        raise ValueError("paired appearance tensors must have shape [2,T,4,H,W]")
    if tuple(predicted_pm.shape) != expected:
        raise ValueError("paired appearance tensors must share one shape")
    predicted_alpha = predicted_pm[:, :, 3:4]
    target_alpha = target_pm[:, :, 3:4]
    predicted_foreground = predicted_alpha >= 0.5
    target_foreground = target_alpha >= 0.5
    intersection = (predicted_foreground & target_foreground).sum()
    union = (predicted_foreground | target_foreground).sum().clamp_min(1)
    predicted_count = predicted_foreground.sum()
    target_count = target_foreground.sum()
    absolute_error = (predicted_pm - target_pm).abs()
    foreground_channels = target_foreground.expand_as(absolute_error)
    background_channels = (~target_foreground).expand_as(absolute_error)
    foreground_error = absolute_error.masked_select(foreground_channels)
    background_error = absolute_error.masked_select(background_channels)
    zero = absolute_error.new_zeros(())
    return {
        "alpha_iou_127": intersection / union,
        "alpha_precision_127": intersection / predicted_count.clamp_min(1),
        "alpha_recall_127": intersection / target_count.clamp_min(1),
        "foreground_premultiplied_rgba_mae": (
            foreground_error.mean() if foreground_error.numel() else zero
        ),
        "background_premultiplied_rgba_mae": (
            background_error.mean() if background_error.numel() else zero
        ),
        "foreground_occupancy_ratio": predicted_count / target_count.clamp_min(1),
    }


def _paired_temporal_motion_metrics(
    runtime: Any,
    *,
    predicted_pm: Any,
    target_pm: Any,
) -> dict[str, Any]:
    """Expose motion collapse separately from frame reconstruction error."""

    expected = tuple(target_pm.shape)
    if len(expected) != 5 or expected[0] != 2 or expected[1] < 2:
        raise ValueError("paired temporal tensors must have shape [2,T>=2,C,H,W]")
    if tuple(predicted_pm.shape) != expected:
        raise ValueError("paired temporal tensors must share one shape")
    predicted_delta = predicted_pm[:, 1:] - predicted_pm[:, :-1]
    target_delta = target_pm[:, 1:] - target_pm[:, :-1]
    generated_magnitude = predicted_delta.abs().mean()
    target_magnitude = target_delta.abs().mean()
    return {
        "generated_temporal_magnitude": generated_magnitude,
        "target_temporal_magnitude": target_magnitude,
    }


def _export_validation_previews(
    runtime: Any,
    corpus: LatentMotionTrainingCorpus,
    selection: tuple[tuple[int, int], ...],
    model: Any,
    decoder: Any,
    *,
    output: Path,
    device: Any,
    dtype: Any,
    autocast: bool,
    mean: Any,
    std: Any,
    seed: int,
    inference_steps: int,
    sampler_algorithm: FlowSampler,
    disk_guard: DiskGuard,
) -> list[dict[str, Any]]:
    rows = []
    generator = runtime.Generator(device=device).manual_seed(seed)
    model.eval()
    with runtime.no_grad():
        for pair_index, pair in enumerate(selection):
            target, reference, _target_rgba, phases, actions = _batch(
                runtime, corpus, pair, device=device, mean=mean, std=std
            )
            noise = runtime.randn(
                (1, *target.shape[1:]), device=device, generator=generator
            ).expand_as(target)
            with runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                residual = _sample_motion_residual(
                    runtime,
                    model,
                    noise=noise,
                    reference=reference,
                    actions=actions,
                    phases=phases,
                    inference_steps=inference_steps,
                    sampler_algorithm=sampler_algorithm,
                )
                counterfactual_residual = _sample_motion_residual(
                    runtime,
                    model,
                    noise=noise,
                    reference=reference,
                    actions=runtime.flip(actions, (0,)),
                    phases=phases,
                    inference_steps=inference_steps,
                    sampler_algorithm=sampler_algorithm,
                )
                latent = (reference.unsqueeze(1) + residual) * std + mean
                counterfactual_latent = (
                    reference.unsqueeze(1) + counterfactual_residual
                ) * std + mean
                decoded = decoder.decode(latent.reshape(-1, 8, 64, 64)).clamp(0, 1)
                counterfactual_decoded = decoder.decode(
                    counterfactual_latent.reshape(-1, 8, 64, 64)
                ).clamp(0, 1)
            arrays = (
                decoded.mul(255)
                .round()
                .to(runtime.uint8)
                .reshape(2, 8, 4, 128, 128)
                .permute(0, 1, 3, 4, 2)
                .cpu()
                .numpy()
            )
            counterfactual_arrays = (
                counterfactual_decoded.mul(255)
                .round()
                .to(runtime.uint8)
                .reshape(2, 8, 4, 128, 128)
                .permute(0, 1, 3, 4, 2)
                .cpu()
                .numpy()
            )
            for offset, row_index in enumerate(pair):
                row = corpus.rows[row_index]
                replacement = corpus.rows[pair[1 - offset]]
                target_array = corpus.target_rgba[row_index]
                generated_array = arrays[offset]
                counterfactual_array = counterfactual_arrays[offset]
                matched_metrics = compare_matched_sequences(
                    [Image.fromarray(frame) for frame in generated_array],
                    [Image.fromarray(frame) for frame in target_array],
                    loop_mode=row.loop_mode,
                    alpha_threshold=127,
                )
                target_preview = export_rgba_clip_preview(
                    target_array,
                    output,
                    artifact_stem=f"{pair_index:02d}-{offset}-{row.identity_id[-8:]}-{row.verb}-target",
                    duration_ms=row.duration_ms,
                    loop_mode=_preview_loop_mode(row.loop_mode),
                    integer_scale=2,
                    preserve_frame_slots=True,
                    disk_guard=disk_guard,
                )
                generated_preview = export_rgba_clip_preview(
                    generated_array,
                    output,
                    artifact_stem=f"{pair_index:02d}-{offset}-{row.identity_id[-8:]}-{row.verb}-generated",
                    duration_ms=row.duration_ms,
                    loop_mode=_preview_loop_mode(row.loop_mode),
                    integer_scale=2,
                    preserve_frame_slots=True,
                    disk_guard=disk_guard,
                )
                counterfactual_preview = export_rgba_clip_preview(
                    counterfactual_array,
                    output,
                    artifact_stem=(
                        f"{pair_index:02d}-{offset}-{row.identity_id[-8:]}-{row.verb}-"
                        f"action-swap-to-{replacement.verb}"
                    ),
                    duration_ms=row.duration_ms,
                    loop_mode=_preview_loop_mode(row.loop_mode),
                    integer_scale=2,
                    preserve_frame_slots=True,
                    disk_guard=disk_guard,
                )
                rows.append(
                    {
                        "counterfactual_action": replacement.verb,
                        "counterfactual_animated_path": _preview_relative_path(
                            counterfactual_preview.animated_png_path, output
                        ),
                        "counterfactual_animated_sha256": (
                            counterfactual_preview.animated_png_sha256
                        ),
                        "counterfactual_sheet_path": _preview_relative_path(
                            counterfactual_preview.contact_sheet_path, output
                        ),
                        "counterfactual_sheet_sha256": (
                            counterfactual_preview.contact_sheet_sha256
                        ),
                        "generated_animated_path": _preview_relative_path(
                            generated_preview.animated_png_path, output
                        ),
                        "generated_animated_sha256": generated_preview.animated_png_sha256,
                        "generated_sheet_path": _preview_relative_path(
                            generated_preview.contact_sheet_path, output
                        ),
                        "generated_sheet_sha256": generated_preview.contact_sheet_sha256,
                        "identity_id": row.identity_id,
                        "matched_metrics": asdict(matched_metrics),
                        "sequence_id": row.sequence_id,
                        "target_array_content_sha256": _array_sha256(target_array),
                        "generated_array_content_sha256": _array_sha256(generated_array),
                        "counterfactual_array_content_sha256": _array_sha256(counterfactual_array),
                        "target_animated_path": _preview_relative_path(
                            target_preview.animated_png_path, output
                        ),
                        "target_animated_sha256": target_preview.animated_png_sha256,
                        "target_sheet_path": _preview_relative_path(
                            target_preview.contact_sheet_path, output
                        ),
                        "target_sheet_sha256": target_preview.contact_sheet_sha256,
                        "verb": row.verb,
                    }
                )
    return rows


def _preview_relative_path(path: Path, output: Path) -> str:
    return f"{output.name}/{path.name}"


def _load_frozen_decoder(runtime: Any, corpus: LatentMotionTrainingCorpus, *, device: Any) -> Any:
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


def _write_training_checkpoint(
    runtime: Any,
    path: Path,
    *,
    corpus: LatentMotionTrainingCorpus,
    config: LatentMotionTrainingConfig,
    step: int,
    model: Any,
    ema: Any,
    optimizer: Any,
    sampler_generator: Any,
    noise_generator: Any,
    device: Any,
    disk_guard: DiskGuard,
) -> None:
    _atomic_torch_save(
        runtime,
        path,
        {
            "artifact_kind": "mugen_reference_latent_motion_resume_checkpoint",
            "config": asdict(config),
            "corpus": corpus.contract,
            "ema_policy": _ema_policy(config),
            "ema_model": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "raw_model": model.state_dict(),
            "rng_state": {
                "cuda": runtime.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "noise": noise_generator.get_state(),
                "sampler": sampler_generator.get_state(),
                "torch_cpu": runtime.get_rng_state(),
            },
            "runtime": _runtime_facts(runtime, device),
            "step": step,
        },
        disk_guard=disk_guard,
    )


def _load_resume(
    runtime: Any,
    path: Path,
    *,
    expected_sha256: str,
    corpus: LatentMotionTrainingCorpus,
    config: LatentMotionTrainingConfig,
) -> dict[str, Any]:
    if _file_sha256(path) != expected_sha256:
        raise LatentMotionTrainingError("resume checkpoint SHA-256 mismatch")
    try:
        value = runtime.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise LatentMotionTrainingError("resume checkpoint failed safe load") from error
    if not isinstance(value, dict) or value.get("artifact_kind") != (
        "mugen_reference_latent_motion_resume_checkpoint"
    ):
        raise LatentMotionTrainingError("resume checkpoint has the wrong artifact kind")
    if value.get("corpus") != corpus.contract:
        raise LatentMotionTrainingError("resume corpus contract differs")
    parent_config = value.get("config")
    if not isinstance(parent_config, dict):
        raise LatentMotionTrainingError("resume config is missing")
    current = asdict(config)
    for key, parent_value in parent_config.items():
        if key != "steps" and current.get(key) != parent_value:
            raise LatentMotionTrainingError(f"resume config differs at {key!r}")
    return value


def _load_warm_start(
    runtime: Any,
    path: Path,
    *,
    expected_sha256: str,
    corpus: LatentMotionTrainingCorpus,
    config: LatentMotionTrainingConfig,
) -> dict[str, Any]:
    """Load only compatible EMA weights; never inherit optimizer or RNG state."""

    if _file_sha256(path) != expected_sha256:
        raise LatentMotionTrainingError("warm-start checkpoint SHA-256 mismatch")
    try:
        value = runtime.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise LatentMotionTrainingError("warm-start checkpoint failed safe load") from error
    artifact_kind = _ema_checkpoint_artifact_kind(value)
    if value.get("corpus") != corpus.contract:
        raise LatentMotionTrainingError("warm-start corpus contract differs")
    if _checkpoint_action_vocabulary(value, artifact_kind) != list(corpus.action_vocabulary):
        raise LatentMotionTrainingError("warm-start action vocabulary differs")
    parent_config = _config_from_dict(value.get("config"))
    if parent_config.model != config.model:
        raise LatentMotionTrainingError("warm-start model architecture differs")
    if not isinstance(value.get("step"), int) or value["step"] <= 0:
        raise LatentMotionTrainingError("warm-start step is invalid")
    if not isinstance(value.get("ema_model"), dict):
        raise LatentMotionTrainingError("warm-start EMA state is missing")
    return value


def _load_warm_start_model_state(
    model: Any,
    state: dict[str, Any],
    *,
    parent_config: LatentMotionTrainingConfig,
    current_config: LatentMotionTrainingConfig,
) -> None:
    """Permit only the audited single-token to expanded-action migration."""

    if parent_config.action_conditioning_mode == current_config.action_conditioning_mode:
        model.load_state_dict(state, strict=True)
        return
    if not (
        parent_config.action_conditioning_mode == "single"
        and current_config.action_conditioning_mode == "expanded"
    ):
        raise LatentMotionTrainingError("unsupported action-conditioning warm start")
    result = model.load_state_dict(state, strict=False)
    expected_missing = {
        "action_frame_mlp.0.bias",
        "action_frame_mlp.0.weight",
        "action_frame_mlp.2.bias",
        "action_frame_mlp.2.weight",
        "action_token_norm.bias",
        "action_token_norm.weight",
        "action_token_projection.bias",
        "action_token_projection.weight",
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise LatentMotionTrainingError("expanded action-conditioning warm-start state differs")


def _load_array(
    root: Path,
    record: dict[str, Any],
    *,
    dtype: Any,
    shape: tuple[int, ...],
    label: str,
    verify_hashes: bool,
    cache: dict[str, np.ndarray] | None,
) -> np.ndarray:
    relative = _required_text(record, "relative_path")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise LatentMotionTrainingError(f"{label} path escapes root")
    cache_key = str(path)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    payload = path.read_bytes()
    if verify_hashes and hashlib.sha256(payload).hexdigest() != record.get("file_sha256"):
        raise LatentMotionTrainingError(f"{label} file hash differs")
    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise LatentMotionTrainingError(f"{label} is unreadable") from error
    if value.dtype != dtype or value.shape != shape or not bool(np.isfinite(value).all()):
        raise LatentMotionTrainingError(f"{label} geometry/content differs")
    value = np.ascontiguousarray(value)
    if verify_hashes and _array_sha256(value) != record.get("array_content_sha256"):
        raise LatentMotionTrainingError(f"{label} array hash differs")
    if cache is not None:
        cache[cache_key] = value
    return value


def _lazy_array_entry(
    root: Path,
    record: dict[str, Any],
    *,
    shape: tuple[int, ...],
    label: str,
    item_index: int | None = None,
    item_array_content_sha256: str | None = None,
) -> _LazyArrayEntry:
    relative = _required_text(record, "relative_path")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise LatentMotionTrainingError(f"{label} path escapes root")
    if not path.is_file():
        raise LatentMotionTrainingError(f"{label} is absent")
    if (item_index is None) != (item_array_content_sha256 is None):
        raise ValueError("lazy item index and content hash must be supplied together")
    if item_index is not None and not 0 <= item_index < shape[0]:
        raise LatentMotionTrainingError(f"{label} item index differs")
    return _LazyArrayEntry(
        path=path,
        file_sha256=_required_hash(record, "file_sha256", label),
        array_content_sha256=_required_hash(record, "array_content_sha256", label),
        source_shape=shape,
        item_index=item_index,
        item_array_content_sha256=item_array_content_sha256,
    )


def _required_hash(record: dict[str, Any], key: str, label: str) -> str:
    value = record.get(key)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LatentMotionTrainingError(f"{label} {key} is not canonical SHA-256")
    return value


def _learning_rate(step: int, config: LatentMotionTrainingConfig) -> float:
    if step <= config.warmup_steps and config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
    return (
        config.minimum_learning_rate
        + (config.learning_rate - config.minimum_learning_rate) * cosine
    )


def _ema_update(runtime: Any, ema: Any, model: Any, decay: float) -> None:
    with runtime.no_grad():
        for target, source in zip(ema.parameters(), model.parameters(), strict=True):
            target.lerp_(source.detach(), 1 - decay)
        for target, source in zip(ema.buffers(), model.buffers(), strict=True):
            target.copy_(source)


def _ema_policy(config: LatentMotionTrainingConfig) -> dict[str, Any]:
    return {
        "decay_after_warmup": config.ema_decay,
        "policy": "copy_raw_through_learning_rate_warmup_then_fixed_decay",
        "warmup_steps": config.warmup_steps,
    }


def _record_verb(record: dict[str, Any]) -> str:
    return _required_text(_required_dict(record, "conditioning"), "verb")


def _config_from_dict(value: Any) -> LatentMotionTrainingConfig:
    if not isinstance(value, dict):
        raise LatentMotionTrainingError("checkpoint config is missing")
    fields = dict(value)
    model = fields.get("model")
    if not isinstance(model, dict):
        raise LatentMotionTrainingError("checkpoint model config is missing")
    try:
        fields["model"] = LatentMotionDiTConfig(**model)
        return LatentMotionTrainingConfig(**fields)
    except (TypeError, ValueError) as error:
        raise LatentMotionTrainingError("checkpoint config is invalid") from error


def _counted_records(artifact: dict[str, Any], label: str) -> list[dict[str, Any]]:
    records = artifact.get("records")
    counts = artifact.get("counts")
    if (
        not isinstance(records, list)
        or not isinstance(counts, dict)
        or counts.get("sequences") != len(records)
        or any(not isinstance(record, dict) for record in records)
    ):
        raise LatentMotionTrainingError(f"{label} record count differs")
    return records


def _required_text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise LatentMotionTrainingError(f"field {key} must be non-empty text")
    return result


def _required_dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise LatentMotionTrainingError(f"field {key} must be an object")
    return result


def _float_tuple(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 8:
        raise LatentMotionTrainingError(f"{label} must contain eight values")
    output = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in output) or (
        "std" in label and any(item <= 0 for item in output)
    ):
        raise LatentMotionTrainingError(f"{label} values are invalid")
    return output


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LatentMotionTrainingError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise LatentMotionTrainingError(f"{label} must contain an object")
    return value


def _preview_loop_mode(value: str) -> str:
    return value if value in {"loop", "one_shot", "ping_pong"} else "one_shot"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_facts(runtime: Any, device: Any) -> dict[str, Any]:
    return {
        "cuda_version": runtime.version.cuda,
        "device": str(device),
        "device_name": runtime.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(runtime.__version__),
    }


def _atomic_bytes(path: Path, payload: bytes, *, disk_guard: DiskGuard) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace artifact: {path}")
    disk_guard.require_capacity(len(payload), label=path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
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
    disk_guard: DiskGuard,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {path}")
    disk_guard.require_capacity(2 * 1024**3, label=path.name)
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
    if torch is None or nn is None:
        raise RuntimeError("latent motion training requires PyTorch") from _TORCH_IMPORT_ERROR
    return torch
