"""Reproducible hard-alpha threshold calibration against exactly paired arrays.

Calibration is deliberately file-backed: every source and target is loaded from
the same bytes that are hashed into the resulting artifact.  This prevents a
threshold report from silently describing arrays other than its cited inputs.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

from spritelab.decode import (
    GlobalPaletteDecodeConfig,
    HardAlphaDecodeConfig,
    global_palette_decode_rgba,
    hard_alpha_decode_rgba,
)
from spritelab.evaluation import compare_matched_sequences
from spritelab.storage import DiskGuard

CalibrationEstimateKind = Literal["training_target_estimate", "held_out_validation"]

_ESTIMATE_NOTES: dict[str, str] = {
    "training_target_estimate": (
        "Threshold selection and metrics use targets seen during model training; "
        "this is an optimistic in-sample diagnostic, not held-out validation."
    ),
    "held_out_validation": (
        "Threshold selection and metrics use targets explicitly designated as "
        "held-out validation data."
    ),
}


@dataclass(frozen=True, slots=True)
class CalibrationArrayRef:
    """One named ``.npy`` array in an ordered calibration input list."""

    sample_id: str
    path: Path | str


@dataclass(frozen=True, slots=True)
class HardAlphaCalibrationResult:
    artifact_path: Path
    artifact_sha256: str
    selected_threshold: int
    pair_count: int


@dataclass(frozen=True, slots=True)
class GlobalPaletteCalibrationResult:
    artifact_path: Path
    artifact_sha256: str
    selected_maximum_colors: int
    pair_count: int


@dataclass(frozen=True, slots=True)
class _LoadedArray:
    sample_id: str
    path: Path
    rgba: np.ndarray
    file_sha256: str
    array_sha256: str
    size_bytes: int


def export_hard_alpha_threshold_calibration(
    sources: Sequence[CalibrationArrayRef],
    targets: Sequence[CalibrationArrayRef],
    thresholds: Sequence[int],
    output_path: Path | str,
    *,
    estimate_kind: CalibrationEstimateKind,
    disk_guard: DiskGuard | None = None,
) -> HardAlphaCalibrationResult:
    """Evaluate thresholds and atomically publish one canonical JSON artifact.

    ``sources`` and ``targets`` are parallel ordered lists. Their IDs must be
    unique and identical in the same order; no filename-based or positional
    realignment is attempted. The output is immutable and never overwritten.
    """

    estimate_note = _validate_estimate_kind(estimate_kind)
    normalized_thresholds = _validate_thresholds(thresholds)
    _validate_pairing(sources, targets)

    output = Path(output_path).resolve()
    if output.suffix.casefold() != ".json":
        raise ValueError("output_path must end in .json")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing calibration artifact: {output}")

    loaded_sources = tuple(_load_array(reference, role="source") for reference in sources)
    loaded_targets = tuple(_load_array(reference, role="target") for reference in targets)
    for source, target in zip(loaded_sources, loaded_targets, strict=True):
        if source.rgba.shape != target.rgba.shape:
            raise ValueError(
                f"source and target for sample {source.sample_id!r} must have identical "
                f"[T, H, W, RGBA] shapes; got {source.rgba.shape!r} and {target.rgba.shape!r}"
            )

    threshold_results = tuple(
        _evaluate_threshold(loaded_sources, loaded_targets, threshold)
        for threshold in normalized_thresholds
    )
    selected = min(
        threshold_results,
        key=lambda result: (
            result["aggregate_metrics"]["premultiplied_rgba_mae"],
            -result["aggregate_metrics"]["alpha_iou"],
            result["aggregate_metrics"]["alpha_mae"],
            result["threshold"],
        ),
    )

    artifact = {
        "artifact_kind": "hard_alpha_threshold_calibration",
        "decode_operation": {
            "alpha_at_or_above_threshold": 255,
            "alpha_below_threshold": 0,
            "hidden_rgb": "zero",
            "visible_rgb": "unchanged",
        },
        "estimate": {
            "held_out": estimate_kind == "held_out_validation",
            "interpretation": estimate_note,
            "kind": estimate_kind,
        },
        "inputs": {
            "pair_count": len(loaded_sources),
            "pairing_contract": "unique sample IDs, identical source/target order and shape",
            "pairs": [
                {
                    "sample_id": source.sample_id,
                    "source": _input_record(source),
                    "target": _input_record(target),
                }
                for source, target in zip(loaded_sources, loaded_targets, strict=True)
            ],
        },
        "metrics": {
            "aggregation": "unweighted macro mean across matched sample arrays",
            "alpha_iou_mask": "alpha > 0",
            "units": "normalized [0, 1] except alpha_iou",
        },
        "schema_version": 1,
        "selection": {
            "objective": [
                {"direction": "minimize", "metric": "premultiplied_rgba_mae"},
                {"direction": "maximize", "metric": "alpha_iou"},
                {"direction": "minimize", "metric": "alpha_mae"},
                {"direction": "minimize", "metric": "threshold"},
            ],
            "rationale": (
                "Lexicographic selection: lowest macro premultiplied RGBA MAE, then "
                "highest macro alpha IoU, then lowest macro alpha MAE, then lowest "
                "numeric threshold."
            ),
            "selected_aggregate_metrics": selected["aggregate_metrics"],
            "selected_threshold": selected["threshold"],
        },
        "threshold_results": list(threshold_results),
        "thresholds": list(normalized_thresholds),
    }
    payload = (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_no_clobber(output, payload, disk_guard=disk_guard)
    return HardAlphaCalibrationResult(
        artifact_path=output,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        selected_threshold=selected["threshold"],
        pair_count=len(loaded_sources),
    )


def export_global_palette_size_calibration(
    sources: Sequence[CalibrationArrayRef],
    targets: Sequence[CalibrationArrayRef],
    palette_sizes: Sequence[int],
    output_path: Path | str,
    *,
    alpha_threshold: int,
    estimate_kind: CalibrationEstimateKind,
    disk_guard: DiskGuard | None = None,
) -> GlobalPaletteCalibrationResult:
    """Select a generated-only clip-global palette size on exact paired arrays.

    Alpha thresholding is fixed before the sweep. Each candidate fits one adaptive
    median-cut palette to all visible generated pixels in a clip, with no target
    colors and no dithering. Sources and targets retain the same strict file-backed
    pairing contract as hard-alpha calibration.
    """

    estimate_note = _validate_estimate_kind(estimate_kind)
    HardAlphaDecodeConfig(threshold=alpha_threshold)
    normalized_sizes = _validate_palette_sizes(palette_sizes, alpha_threshold)
    _validate_pairing(sources, targets)

    output = Path(output_path).resolve()
    if output.suffix.casefold() != ".json":
        raise ValueError("output_path must end in .json")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing calibration artifact: {output}")

    loaded_sources = tuple(_load_array(reference, role="source") for reference in sources)
    loaded_targets = tuple(_load_array(reference, role="target") for reference in targets)
    for source, target in zip(loaded_sources, loaded_targets, strict=True):
        if source.rgba.shape != target.rgba.shape:
            raise ValueError(
                f"source and target for sample {source.sample_id!r} must have identical "
                f"[T, H, W, RGBA] shapes; got {source.rgba.shape!r} and {target.rgba.shape!r}"
            )

    palette_results = tuple(
        _evaluate_palette_size(
            loaded_sources,
            loaded_targets,
            alpha_threshold=alpha_threshold,
            maximum_colors=maximum_colors,
        )
        for maximum_colors in normalized_sizes
    )
    selected = min(
        palette_results,
        key=lambda result: (
            result["aggregate_metrics"]["premultiplied_rgba_mae"],
            result["aggregate_metrics"]["composite_black_mae"],
            result["aggregate_metrics"]["temporal_delta_mae"],
            result["maximum_colors"],
        ),
    )

    artifact = {
        "artifact_kind": "clip_global_palette_size_calibration",
        "decode_operation": {
            "alpha_at_or_above_threshold": 255,
            "alpha_below_threshold": 0,
            "dithering": "none",
            "hidden_rgb": "zero",
            "palette_fit_scope": "all visible generated RGB pixels in each clip",
            "palette_method": "Pillow MEDIANCUT",
            "reference_or_target_palette_used": False,
        },
        "estimate": {
            "held_out": estimate_kind == "held_out_validation",
            "interpretation": estimate_note,
            "kind": estimate_kind,
        },
        "inputs": {
            "pair_count": len(loaded_sources),
            "pairing_contract": "unique sample IDs, identical source/target order and shape",
            "pairs": [
                {
                    "sample_id": source.sample_id,
                    "source": _input_record(source),
                    "target": _input_record(target),
                }
                for source, target in zip(loaded_sources, loaded_targets, strict=True)
            ],
        },
        "metrics": {
            "aggregation": "unweighted macro mean across matched sample arrays",
            "alpha_iou_mask": "alpha > 0",
            "units": "normalized [0, 1] except alpha_iou and visible RGB counts",
        },
        "palette_size_results": list(palette_results),
        "palette_sizes": list(normalized_sizes),
        "parameters": {"alpha_threshold": alpha_threshold},
        "runtime": {"pillow_version": pillow_version},
        "schema_version": 1,
        "selection": {
            "objective": [
                {"direction": "minimize", "metric": "premultiplied_rgba_mae"},
                {"direction": "minimize", "metric": "composite_black_mae"},
                {"direction": "minimize", "metric": "temporal_delta_mae"},
                {"direction": "minimize", "metric": "maximum_colors"},
            ],
            "rationale": (
                "Lexicographic selection: lowest macro premultiplied RGBA MAE, then "
                "lowest composite-black MAE, then lowest temporal-delta MAE, then "
                "the smaller palette. Targets are used only for evaluation and never "
                "to fit a palette."
            ),
            "selected_aggregate_metrics": selected["aggregate_metrics"],
            "selected_maximum_colors": selected["maximum_colors"],
        },
    }
    payload = (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_no_clobber(output, payload, disk_guard=disk_guard)
    return GlobalPaletteCalibrationResult(
        artifact_path=output,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        selected_maximum_colors=selected["maximum_colors"],
        pair_count=len(loaded_sources),
    )


def _validate_estimate_kind(estimate_kind: str) -> str:
    try:
        return _ESTIMATE_NOTES[estimate_kind]
    except (KeyError, TypeError) as error:
        allowed = ", ".join(sorted(_ESTIMATE_NOTES))
        raise ValueError(f"estimate_kind must be one of: {allowed}") from error


def _validate_thresholds(thresholds: Sequence[int]) -> tuple[int, ...]:
    if isinstance(thresholds, (str, bytes)) or not isinstance(thresholds, Sequence):
        raise TypeError("thresholds must be an explicit sequence of integers")
    if not thresholds:
        raise ValueError("at least one threshold is required")
    normalized: list[int] = []
    for threshold in thresholds:
        HardAlphaDecodeConfig(threshold=threshold)
        normalized.append(threshold)
    if len(set(normalized)) != len(normalized):
        raise ValueError("thresholds must not contain duplicates")
    return tuple(sorted(normalized))


def _validate_palette_sizes(palette_sizes: Sequence[int], alpha_threshold: int) -> tuple[int, ...]:
    if isinstance(palette_sizes, (str, bytes)) or not isinstance(palette_sizes, Sequence):
        raise TypeError("palette_sizes must be an explicit sequence of integers")
    if not palette_sizes:
        raise ValueError("at least one palette size is required")
    normalized: list[int] = []
    for maximum_colors in palette_sizes:
        GlobalPaletteDecodeConfig(
            alpha_threshold=alpha_threshold,
            maximum_colors=maximum_colors,
        )
        normalized.append(maximum_colors)
    if len(set(normalized)) != len(normalized):
        raise ValueError("palette_sizes must not contain duplicates")
    return tuple(sorted(normalized))


def _validate_pairing(
    sources: Sequence[CalibrationArrayRef], targets: Sequence[CalibrationArrayRef]
) -> None:
    if isinstance(sources, (str, bytes)) or isinstance(targets, (str, bytes)):
        raise TypeError("sources and targets must be sequences of CalibrationArrayRef")
    if not sources:
        raise ValueError("at least one source/target pair is required")
    if len(sources) != len(targets):
        raise ValueError(
            f"sources and targets must have the same length; got {len(sources)} and {len(targets)}"
        )
    source_ids = _validated_ids(sources, role="source")
    target_ids = _validated_ids(targets, role="target")
    if source_ids != target_ids:
        if set(source_ids) == set(target_ids):
            raise ValueError("source and target sample order must match exactly")
        raise ValueError("source and target sample IDs must pair one-to-one")


def _validated_ids(references: Sequence[CalibrationArrayRef], *, role: str) -> tuple[str, ...]:
    identifiers: list[str] = []
    for reference in references:
        if not isinstance(reference, CalibrationArrayRef):
            raise TypeError(f"{role}s must contain only CalibrationArrayRef values")
        if not isinstance(reference.sample_id, str) or not reference.sample_id.strip():
            raise ValueError(f"{role} sample IDs must be non-empty strings")
        identifiers.append(reference.sample_id)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{role} sample IDs must be unique")
    return tuple(identifiers)


def _load_array(reference: CalibrationArrayRef, *, role: str) -> _LoadedArray:
    path = Path(reference.path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} array does not exist: {path}")
    payload = path.read_bytes()
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"{role} array is not a readable NumPy array: {path}") from error
    if not isinstance(loaded, np.ndarray):
        if hasattr(loaded, "close"):
            loaded.close()
        raise ValueError(f"{role} path must contain exactly one NumPy array: {path}")
    _validate_rgba(loaded, role=role, sample_id=reference.sample_id)
    rgba = np.ascontiguousarray(loaded)
    return _LoadedArray(
        sample_id=reference.sample_id,
        path=path,
        rgba=rgba,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        array_sha256=_array_sha256(rgba),
        size_bytes=len(payload),
    )


def _validate_rgba(rgba: np.ndarray, *, role: str, sample_id: str) -> None:
    if rgba.dtype != np.uint8:
        raise TypeError(f"{role} array for sample {sample_id!r} must have dtype uint8")
    if rgba.ndim != 4 or rgba.shape[-1] != 4 or min(rgba.shape) < 1:
        raise ValueError(
            f"{role} array for sample {sample_id!r} must have shape [T, H, W, 4]; "
            f"got {rgba.shape!r}"
        )


def _evaluate_threshold(
    sources: Sequence[_LoadedArray],
    targets: Sequence[_LoadedArray],
    threshold: int,
) -> dict[str, object]:
    per_sample: list[dict[str, object]] = []
    for source, target in zip(sources, targets, strict=True):
        decoded = hard_alpha_decode_rgba(
            source.rgba,
            config=HardAlphaDecodeConfig(threshold=threshold),
        )
        metrics = compare_matched_sequences(
            _to_frames(decoded),
            _to_frames(target.rgba),
            loop_mode="unknown",
            alpha_threshold=0,
        )
        per_sample.append(
            {
                "alpha_iou": metrics.alpha_iou,
                "alpha_mae": metrics.alpha_mae,
                "premultiplied_rgba_mae": metrics.premultiplied_rgba_mae,
                "sample_id": source.sample_id,
            }
        )
    aggregate = {
        metric: _mean([float(row[metric]) for row in per_sample])
        for metric in ("premultiplied_rgba_mae", "alpha_iou", "alpha_mae")
    }
    return {
        "aggregate_metrics": aggregate,
        "per_sample_metrics": per_sample,
        "threshold": threshold,
    }


def _evaluate_palette_size(
    sources: Sequence[_LoadedArray],
    targets: Sequence[_LoadedArray],
    *,
    alpha_threshold: int,
    maximum_colors: int,
) -> dict[str, object]:
    per_sample: list[dict[str, object]] = []
    for source, target in zip(sources, targets, strict=True):
        decoded = global_palette_decode_rgba(
            source.rgba,
            config=GlobalPaletteDecodeConfig(
                alpha_threshold=alpha_threshold,
                maximum_colors=maximum_colors,
            ),
        )
        metrics = compare_matched_sequences(
            _to_frames(decoded),
            _to_frames(target.rgba),
            loop_mode="unknown",
            alpha_threshold=0,
        )
        per_sample.append(
            {
                "alpha_iou": metrics.alpha_iou,
                "composite_black_mae": metrics.composite_black_mae,
                "premultiplied_rgba_mae": metrics.premultiplied_rgba_mae,
                "sample_id": source.sample_id,
                "temporal_delta_mae": metrics.temporal_delta_mae,
                "visible_colors_after": _visible_color_count(decoded),
            }
        )
    aggregate = {
        metric: _mean([float(row[metric]) for row in per_sample])
        for metric in (
            "premultiplied_rgba_mae",
            "alpha_iou",
            "composite_black_mae",
            "temporal_delta_mae",
            "visible_colors_after",
        )
    }
    return {
        "aggregate_metrics": aggregate,
        "maximum_colors": maximum_colors,
        "per_sample_metrics": per_sample,
    }


def _to_frames(rgba: np.ndarray) -> tuple[Image.Image, ...]:
    return tuple(Image.fromarray(frame) for frame in rgba)


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _visible_color_count(rgba: np.ndarray) -> int:
    visible_rgb = rgba[..., :3][rgba[..., 3] > 0]
    return int(len(np.unique(visible_rgb, axis=0))) if visible_rgb.size else 0


def _input_record(array: _LoadedArray) -> dict[str, object]:
    return {
        "array_sha256": array.array_sha256,
        "dtype": array.rgba.dtype.name,
        "file_sha256": array.file_sha256,
        "path": str(array.path),
        "shape": list(array.rgba.shape),
        "size_bytes": array.size_bytes,
    }


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _atomic_write_no_clobber(
    path: Path,
    payload: bytes,
    *,
    disk_guard: DiskGuard | None,
) -> None:
    if disk_guard is not None:
        disk_guard.require_capacity(len(payload) + 65_536, label="calibration artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to replace existing calibration artifact: {path}"
            ) from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
