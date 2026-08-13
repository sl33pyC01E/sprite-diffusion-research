"""Verified loading and native-RGBA conversion for materialized sprite clips."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from spritelab.captions import SpriteGenerationRequest
from spritelab.temporal import TemporalSelection, apply_temporal_selection, select_temporal_frames


class TrainingDataError(RuntimeError):
    """Raised when a materialized training artifact violates its manifest."""


MATERIALIZATION_SCHEMA_VERSION = 1
SOURCE_SNAPSHOT_SCHEMA_VERSION = 1
_KNOWN_SPLITS = frozenset({"train", "validation", "test"})
_KNOWN_LOOP_MODES = frozenset({"loop", "one_shot", "ping_pong", "intro_then_loop", "unknown"})


@dataclass(frozen=True, slots=True)
class IntroLoopTailProjection:
    """Exact source-ordinal projection from a prefixed timeline to its loop tail."""

    source_frame_count: int
    prefix_frame_count: int
    loop_source_ordinals: tuple[int, ...]
    loop_source_phases: tuple[float, ...]
    source_total_duration_ms: float
    discarded_prefix_duration_ms: float
    loop_total_duration_ms: float
    method: str = "verified_contiguous_intro_then_loop_tail_v1"


@dataclass(frozen=True, slots=True)
class MaterializedTrainingClip:
    """One hash-verified RGBA clip and its generation condition."""

    sequence_id: str
    identity_id: str
    split: str
    source_id: str
    quality_tier: str
    request: SpriteGenerationRequest
    rgba: np.ndarray
    frame_phases: tuple[float, ...]
    duration_ms: tuple[float, ...]
    source_path: Path
    source_file_sha256: str
    source_array_sha256: str
    source_blob_sha256: tuple[str, ...]
    source_duration_ms: tuple[float, ...]
    materialization_manifest_sha256: str
    source_snapshot_canonical_sha256: str
    source_snapshot_manifest_sha256: str
    source_loop_mode: str
    intro_loop_projection: IntroLoopTailProjection | None
    temporal_selection: TemporalSelection | None
    temporal_duration_method: str | None

    @property
    def model_array(self) -> np.ndarray:
        return rgba_uint8_to_model(self.rgba)


@dataclass(frozen=True, slots=True)
class MaterializedTrainingBatch:
    """Stacked NumPy batch ready for conversion to framework tensors."""

    sequence_ids: tuple[str, ...]
    requests: tuple[SpriteGenerationRequest, ...]
    clean: np.ndarray
    frame_phases: np.ndarray


def load_materialized_training_clips(
    manifest_path: Path | str,
    *,
    sequence_ids: Iterable[str] | None = None,
    split: str | None = "train",
    target_bucket: int | tuple[int, int] | None = None,
    target_frames: int | None = None,
) -> tuple[MaterializedTrainingClip, ...]:
    """Load selected clips only after verifying file, array, and semantic facts."""

    path = Path(manifest_path).resolve()
    try:
        manifest_bytes = path.read_bytes()
        root = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TrainingDataError(f"Cannot read materialization manifest: {path}") from error
    if not isinstance(root, Mapping):
        raise TrainingDataError("materialization manifest root must be an object")
    schema_version = _required_integer(root, "schema_version")
    if schema_version != MATERIALIZATION_SCHEMA_VERSION:
        raise TrainingDataError(
            "unsupported materialization schema version: "
            f"expected {MATERIALIZATION_SCHEMA_VERSION}, got {schema_version}"
        )
    records = root.get("sequences")
    if not isinstance(records, list) or not records:
        raise TrainingDataError("materialization manifest must contain sequences")
    declared_count = _required_integer(root, "sequence_count")
    if declared_count != len(records):
        raise TrainingDataError(
            f"sequence_count mismatch: declared {declared_count}, found {len(records)}"
        )
    snapshot = root.get("source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise TrainingDataError("source_snapshot must be an object")
    source_snapshot_canonical_sha256 = _required_digest(snapshot, "canonical_sha256")
    source_snapshot_manifest_sha256 = _required_digest(snapshot, "manifest_sha256")
    snapshot_schema_version = _required_integer(snapshot, "schema_version")
    if snapshot_schema_version != SOURCE_SNAPSHOT_SCHEMA_VERSION:
        raise TrainingDataError(
            "unsupported source snapshot schema version: "
            f"expected {SOURCE_SNAPSHOT_SCHEMA_VERSION}, got {snapshot_schema_version}"
        )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    wanted = None if sequence_ids is None else _normalize_sequence_ids(sequence_ids)
    if wanted is not None and not wanted:
        raise ValueError("sequence_ids cannot be empty when supplied")
    if split is not None and split not in _KNOWN_SPLITS:
        raise ValueError(f"unsupported split: {split!r}")
    bucket = _normalize_bucket(target_bucket)
    if target_frames is not None:
        _positive_integer("target_frames", target_frames)

    clips: list[MaterializedTrainingClip] = []
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise TrainingDataError("each sequence record must be an object")
        sequence_id = _required_string(raw, "sequence_id")
        if sequence_id in seen:
            raise TrainingDataError(f"duplicate sequence record: {sequence_id!r}")
        seen.add(sequence_id)
        record_split = _required_string(raw, "split")
        if record_split not in _KNOWN_SPLITS:
            raise TrainingDataError(
                f"sequence {sequence_id!r} has unsupported split {record_split!r}"
            )
        record_bucket = _integer_pair(raw.get("target_bucket"), "target_bucket")
        if wanted is not None and sequence_id not in wanted:
            continue
        if split is not None and record_split != split:
            continue
        if bucket is not None and record_bucket != bucket:
            continue
        clips.append(
            _load_clip(
                raw,
                manifest_root=path.parent,
                manifest_sha256=manifest_sha256,
                record_bucket=record_bucket,
                source_snapshot_canonical_sha256=source_snapshot_canonical_sha256,
                source_snapshot_manifest_sha256=source_snapshot_manifest_sha256,
                target_frames=target_frames,
            )
        )
    if wanted is not None:
        missing = wanted.difference(seen)
        if missing:
            raise TrainingDataError(
                f"requested sequence IDs are absent from manifest: {sorted(missing)!r}"
            )
        filtered = wanted.difference(clip.sequence_id for clip in clips)
        if filtered:
            raise TrainingDataError(
                "requested sequence IDs were excluded by split/bucket filters: "
                f"{sorted(filtered)!r}"
            )
    if not clips:
        raise TrainingDataError("no materialized clips match the requested filters")
    return tuple(sorted(clips, key=lambda clip: clip.sequence_id.encode("utf-8")))


def collate_materialized_clips(
    clips: Sequence[MaterializedTrainingClip],
) -> MaterializedTrainingBatch:
    """Stack clips only when their exact model tensor shapes agree."""

    if not clips:
        raise ValueError("at least one clip is required")
    shapes = {clip.model_array.shape for clip in clips}
    if len(shapes) != 1:
        raise ValueError(f"all clips must share one model shape; got {sorted(shapes)!r}")
    clean = np.stack([clip.model_array for clip in clips], axis=0)
    phases = np.asarray([clip.frame_phases for clip in clips], dtype=np.float32)
    return MaterializedTrainingBatch(
        sequence_ids=tuple(clip.sequence_id for clip in clips),
        requests=tuple(clip.request for clip in clips),
        clean=np.ascontiguousarray(clean),
        frame_phases=np.ascontiguousarray(phases),
    )


def rgba_uint8_to_model(rgba: np.ndarray) -> np.ndarray:
    """Convert straight uint8 RGBA ``[T,H,W,4]`` to premultiplied ``[-1,1]``."""

    _validate_rgba_array(rgba)
    unit = rgba.astype(np.float32) / np.float32(255.0)
    alpha = unit[..., 3:4]
    premultiplied = np.concatenate((unit[..., :3] * alpha, alpha), axis=-1)
    channels_first = np.transpose(premultiplied * 2.0 - 1.0, (0, 3, 1, 2))
    return np.ascontiguousarray(channels_first, dtype=np.float32)


def model_to_rgba_uint8(model_clip: np.ndarray) -> np.ndarray:
    """Convert premultiplied model values ``[T,4,H,W]`` to straight uint8 RGBA."""

    if not isinstance(model_clip, np.ndarray):
        raise TypeError("model_clip must be a NumPy array")
    if model_clip.ndim != 4 or model_clip.shape[1] != 4:
        raise ValueError(f"model_clip must have shape [T,4,H,W]; got {model_clip.shape!r}")
    if not np.issubdtype(model_clip.dtype, np.floating):
        raise TypeError("model_clip must use a floating-point dtype")
    if not np.isfinite(model_clip).all():
        raise ValueError("model_clip must contain only finite values")
    unit = np.clip((model_clip.astype(np.float32) + 1.0) * 0.5, 0.0, 1.0)
    channels_last = np.transpose(unit, (0, 2, 3, 1))
    alpha = channels_last[..., 3:4]
    straight_rgb = np.divide(
        channels_last[..., :3],
        alpha,
        out=np.zeros_like(channels_last[..., :3]),
        where=alpha > (0.5 / 255.0),
    )
    straight = np.concatenate((np.clip(straight_rgb, 0.0, 1.0), alpha), axis=-1)
    return np.ascontiguousarray(np.rint(straight * 255.0).astype(np.uint8))


def _load_clip(
    raw: Mapping[str, Any],
    *,
    manifest_root: Path,
    manifest_sha256: str,
    record_bucket: tuple[int, int],
    source_snapshot_canonical_sha256: str,
    source_snapshot_manifest_sha256: str,
    target_frames: int | None,
) -> MaterializedTrainingClip:
    sequence_id = _required_string(raw, "sequence_id")
    output = raw.get("output")
    if not isinstance(output, Mapping):
        raise TrainingDataError(f"sequence {sequence_id!r} has no output record")
    output_format = _required_string(output, "format")
    if output_format != "numpy_npy_v1":
        raise TrainingDataError(
            f"sequence {sequence_id!r} uses unsupported output format {output_format!r}"
        )
    declared_dtype = _required_string(output, "dtype")
    if declared_dtype != "uint8":
        raise TrainingDataError(
            f"sequence {sequence_id!r} declares unsupported dtype {declared_dtype!r}"
        )
    relative_path = _safe_relative_path(_required_string(output, "relative_path"))
    source_path = (manifest_root / Path(*relative_path.parts)).resolve()
    try:
        source_path.relative_to(manifest_root)
    except ValueError as error:
        raise TrainingDataError(
            f"clip path escapes materialization root: {relative_path}"
        ) from error
    if not source_path.is_file():
        raise TrainingDataError(f"materialized clip does not exist: {source_path}")
    expected_size = _required_integer(output, "size_bytes")
    if source_path.stat().st_size != expected_size:
        raise TrainingDataError(
            f"clip size mismatch for {sequence_id!r}: expected {expected_size}, "
            f"got {source_path.stat().st_size}"
        )
    expected_file_sha = _required_digest(output, "file_sha256")
    actual_file_sha = _sha256_file(source_path)
    if actual_file_sha != expected_file_sha:
        raise TrainingDataError(
            f"clip file SHA-256 mismatch for {sequence_id!r}: "
            f"expected {expected_file_sha}, got {actual_file_sha}"
        )
    try:
        rgba = np.load(source_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise TrainingDataError(f"cannot load clip array for {sequence_id!r}") from error
    if not isinstance(rgba, np.ndarray):
        close = getattr(rgba, "close", None)
        if close is not None:
            close()
        raise TrainingDataError(
            f"materialized clip for {sequence_id!r} must contain one NumPy array"
        )
    expected_shape = tuple(_required_integer_sequence(output, "shape", length=4))
    if expected_shape[-1] != 4:
        raise TrainingDataError(
            f"sequence {sequence_id!r} output must have four RGBA channels; "
            f"got shape {expected_shape!r}"
        )
    declared_frame_count = _required_integer(raw, "frame_count")
    if declared_frame_count != expected_shape[0]:
        raise TrainingDataError(
            f"sequence {sequence_id!r} frame_count mismatch: declared "
            f"{declared_frame_count}, output has {expected_shape[0]}"
        )
    output_bucket = (expected_shape[2], expected_shape[1])
    if record_bucket != output_bucket:
        raise TrainingDataError(
            f"sequence {sequence_id!r} target_bucket mismatch: declared "
            f"{record_bucket!r}, output is {output_bucket!r}"
        )
    if rgba.dtype != np.uint8 or tuple(rgba.shape) != expected_shape:
        raise TrainingDataError(
            f"clip array contract mismatch for {sequence_id!r}: "
            f"expected uint8 {expected_shape!r}, got {rgba.dtype} {rgba.shape!r}"
        )
    expected_array_sha = _required_digest(output, "array_content_sha256")
    actual_array_sha = _array_sha256(rgba)
    if actual_array_sha != expected_array_sha:
        raise TrainingDataError(
            f"clip array SHA-256 mismatch for {sequence_id!r}: "
            f"expected {expected_array_sha}, got {actual_array_sha}"
        )
    _validate_rgba_array(rgba)

    timing = raw.get("timing")
    if not isinstance(timing, Mapping):
        raise TrainingDataError(f"sequence {sequence_id!r} has no timing record")
    raw_phases = _optional_float_sequence(
        timing.get("phase"),
        "timing.phase",
        expected_shape[0],
    )
    durations = _positive_float_sequence(
        timing.get("duration_ms"),
        "timing.duration_ms",
        expected_shape[0],
    )
    source_loop_mode = _required_string(raw, "loop_mode")
    if source_loop_mode not in _KNOWN_LOOP_MODES:
        raise TrainingDataError(f"unsupported loop_mode: {source_loop_mode!r}")
    source_durations = durations
    intro_projection: IntroLoopTailProjection | None = None
    if source_loop_mode == "intro_then_loop":
        rgba, durations, phases, intro_projection = _project_intro_loop_tail(
            rgba,
            durations,
            raw_phases,
            sequence_id=sequence_id,
        )
        loop_mode = "loop"
        duration_method: str | None = "intro_then_loop_verified_tail_only_v1"
    else:
        if any(phase is None for phase in raw_phases):
            raise TrainingDataError(
                f"sequence {sequence_id!r} has null phases outside intro_then_loop semantics"
            )
        phases = tuple(phase for phase in raw_phases if phase is not None)
        loop_mode = source_loop_mode
        _validate_frame_phases(phases, loop_mode=loop_mode)
        duration_method = None
    selection: TemporalSelection | None = None
    if target_frames is not None and target_frames != rgba.shape[0]:
        if loop_mode == "unknown":
            raise TrainingDataError(
                f"sequence {sequence_id!r} cannot be temporally resampled because "
                "its loop mode is unknown"
            )
        try:
            selection = select_temporal_frames(
                rgba.shape[0],
                target_frames,
                loop_mode=loop_mode,  # type: ignore[arg-type]
                source_phases=phases,
            )
        except ValueError as error:
            raise TrainingDataError(
                f"sequence {sequence_id!r} has invalid temporal metadata: {error}"
            ) from error
        rgba = np.stack(apply_temporal_selection(tuple(rgba), selection), axis=0)
        durations = _retime_selected_durations(durations, selection)
        phases = selection.target_phases
        duration_method = (
            "intro_then_loop_verified_tail_then_selected_duration_weights_preserve_loop_total_v1"
            if intro_projection is not None
            else "selected_authored_duration_weights_preserve_total_v1"
        )

    caption = raw.get("caption")
    if not isinstance(caption, Mapping):
        raise TrainingDataError(f"sequence {sequence_id!r} has no caption record")
    request = SpriteGenerationRequest(
        description=_required_string(caption, "description"),
        entity_class=_required_string(raw, "entity_class"),
        action=_required_string(raw, "action"),
        view=_required_string(raw, "view"),
        direction=_required_string(raw, "direction"),
        loop_mode=loop_mode,
    )
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TrainingDataError(f"sequence {sequence_id!r} provenance must be an object")
    source_blob_sha256 = _digest_sequence(
        provenance.get("source_blob_sha256"),
        "provenance.source_blob_sha256",
    )
    return MaterializedTrainingClip(
        sequence_id=sequence_id,
        identity_id=_required_string(raw, "identity_id"),
        split=_required_string(raw, "split"),
        source_id=_required_string(provenance, "source_id"),
        quality_tier=_required_string(raw, "quality_tier"),
        request=request,
        rgba=np.ascontiguousarray(rgba),
        frame_phases=phases,
        duration_ms=durations,
        source_path=source_path,
        source_file_sha256=actual_file_sha,
        source_array_sha256=actual_array_sha,
        source_blob_sha256=source_blob_sha256,
        source_duration_ms=source_durations,
        materialization_manifest_sha256=manifest_sha256,
        source_snapshot_canonical_sha256=source_snapshot_canonical_sha256,
        source_snapshot_manifest_sha256=source_snapshot_manifest_sha256,
        source_loop_mode=source_loop_mode,
        intro_loop_projection=intro_projection,
        temporal_selection=selection,
        temporal_duration_method=duration_method,
    )


def _validate_rgba_array(value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError("rgba must be a NumPy array")
    if value.dtype != np.uint8:
        raise TypeError(f"rgba must use uint8; got {value.dtype}")
    if value.ndim != 4 or value.shape[-1] != 4:
        raise ValueError(f"rgba must have shape [T,H,W,4]; got {value.shape!r}")
    if not all(dimension > 0 for dimension in value.shape):
        raise ValueError("rgba dimensions must be positive")


def _normalize_bucket(value: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("target_bucket must contain two positive integers")
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, tuple):
        pair = value
    else:
        raise ValueError("target_bucket must be an integer or a two-integer tuple")
    if len(pair) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in pair
    ):
        raise ValueError("target_bucket must contain two positive integers")
    return tuple(pair)


def _normalize_sequence_ids(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, str | bytes):
        raise ValueError("sequence_ids must be an iterable of non-empty strings")
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise ValueError("sequence_ids must be an iterable of non-empty strings") from error
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("sequence_ids must contain only non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("sequence_ids must not contain duplicates")
    return frozenset(normalized)


def _validate_frame_phases(phases: tuple[float, ...], *, loop_mode: str) -> None:
    if loop_mode not in _KNOWN_LOOP_MODES:
        raise TrainingDataError(f"unsupported loop_mode: {loop_mode!r}")
    if loop_mode == "intro_then_loop":
        raise TrainingDataError(
            "intro_then_loop phases require an explicit verified tail projection"
        )
    previous = -math.inf
    for index, phase in enumerate(phases):
        upper_ok = phase < 1.0 if loop_mode == "loop" else phase <= 1.0
        if phase < 0.0 or not upper_ok:
            interval = "[0, 1)" if loop_mode == "loop" else "[0, 1]"
            raise TrainingDataError(
                f"timing.phase[{index}] must be in {interval} for {loop_mode!r}; got {phase}"
            )
        if phase < previous:
            raise TrainingDataError("timing.phase must be nondecreasing")
        if phase == previous:
            raise TrainingDataError("timing.phase must not contain duplicates")
        previous = phase


def _project_intro_loop_tail(
    rgba: np.ndarray,
    durations: tuple[float, ...],
    phases: tuple[float | None, ...],
    *,
    sequence_id: str,
) -> tuple[
    np.ndarray,
    tuple[float, ...],
    tuple[float, ...],
    IntroLoopTailProjection,
]:
    """Drop only the phase-null prefix and retain the verified contiguous loop tail."""

    first_loop = next((index for index, phase in enumerate(phases) if phase is not None), None)
    if first_loop is None:
        raise TrainingDataError(f"intro_then_loop sequence {sequence_id!r} has no phased loop tail")
    if first_loop == 0:
        raise TrainingDataError(
            f"intro_then_loop sequence {sequence_id!r} has no phase-null intro prefix"
        )
    if any(phase is None for phase in phases[first_loop:]):
        raise TrainingDataError(
            f"intro_then_loop sequence {sequence_id!r} loop tail is not contiguous"
        )
    loop_phases = tuple(phase for phase in phases[first_loop:] if phase is not None)
    _validate_frame_phases(loop_phases, loop_mode="loop")
    if loop_phases[0] != 0.0:
        raise TrainingDataError(
            f"intro_then_loop sequence {sequence_id!r} loop tail must begin at phase 0"
        )
    loop_durations = durations[first_loop:]
    source_total = math.fsum(durations)
    prefix_total = math.fsum(durations[:first_loop])
    loop_total = math.fsum(loop_durations)
    projection = IntroLoopTailProjection(
        source_frame_count=len(phases),
        prefix_frame_count=first_loop,
        loop_source_ordinals=tuple(range(first_loop, len(phases))),
        loop_source_phases=loop_phases,
        source_total_duration_ms=source_total,
        discarded_prefix_duration_ms=prefix_total,
        loop_total_duration_ms=loop_total,
    )
    return (
        np.ascontiguousarray(rgba[first_loop:]),
        loop_durations,
        loop_phases,
        projection,
    )


def _retime_selected_durations(
    source_durations: tuple[float, ...],
    selection: TemporalSelection,
) -> tuple[float, ...]:
    """Preserve total authored playback time after nearest-phase frame selection."""

    selected = [source_durations[index] for index in selection.source_ordinals]
    source_total = math.fsum(source_durations)
    selected_total = math.fsum(selected)
    scale = source_total / selected_total
    result = [value * scale for value in selected]
    result[-1] += source_total - math.fsum(result)
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise TrainingDataError("temporal duration retiming produced an invalid duration")
    return tuple(result)


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise TrainingDataError(f"output relative_path is unsafe: {value!r}")
    return path


def _array_sha256(array: np.ndarray) -> str:
    header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _required_string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise TrainingDataError(f"{key} must be a non-empty string")
    return value


def _required_integer(record: Mapping[str, Any], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrainingDataError(f"{key} must be a non-negative integer")
    return value


def _required_digest(record: Mapping[str, Any], key: str) -> str:
    value = _required_string(record, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise TrainingDataError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _digest_sequence(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise TrainingDataError(f"{name} must contain at least one SHA-256 digest")
    result: list[str] = []
    for index, digest in enumerate(value):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise TrainingDataError(f"{name}[{index}] must be a lowercase SHA-256 digest")
        result.append(digest)
    return tuple(result)


def _required_integer_sequence(
    record: Mapping[str, Any],
    key: str,
    *,
    length: int,
) -> tuple[int, ...]:
    value = record.get(key)
    if not isinstance(value, list | tuple) or len(value) != length:
        raise TrainingDataError(f"{key} must contain {length} integers")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in result):
        raise TrainingDataError(f"{key} must contain {length} positive integers")
    return result


def _integer_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise TrainingDataError(f"{name} must contain two integers")
    result = tuple(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in result):
        raise TrainingDataError(f"{name} must contain two positive integers")
    return result  # type: ignore[return-value]


def _finite_float_sequence(value: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise TrainingDataError(f"{name} must contain {length} numbers")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise TrainingDataError(f"{name} must contain {length} numbers")
    result = tuple(float(item) for item in value)
    if not all(np.isfinite(item) for item in result):
        raise TrainingDataError(f"{name} must contain only finite numbers")
    return result


def _optional_float_sequence(
    value: object,
    name: str,
    length: int,
) -> tuple[float | None, ...]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise TrainingDataError(f"{name} must contain {length} numbers or nulls")
    result: list[float | None] = []
    for item in value:
        if item is None:
            result.append(None)
            continue
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TrainingDataError(f"{name} must contain {length} numbers or nulls")
        converted = float(item)
        if not math.isfinite(converted):
            raise TrainingDataError(f"{name} must contain only finite numbers or nulls")
        result.append(converted)
    return tuple(result)


def _positive_float_sequence(value: object, name: str, length: int) -> tuple[float, ...]:
    result = _finite_float_sequence(value, name, length)
    if any(item <= 0 for item in result):
        raise TrainingDataError(f"{name} must contain only positive numbers")
    return result


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
