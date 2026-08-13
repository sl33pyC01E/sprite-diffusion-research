"""Hash-verified, detection-only audits of materialized RGBA sprite pixels."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.storage import DiskGuard
from spritelab.training_data import MaterializedTrainingClip, load_materialized_training_clips

DEFAULT_OPAQUE_RGB_SENTINELS: tuple[tuple[int, int, int], ...] = ((255, 0, 255),)


@dataclass(frozen=True, slots=True)
class PixelQualityDetectionConfig:
    """Exact opaque RGB values to detect without changing or interpreting pixels."""

    opaque_rgb_sentinels: tuple[tuple[int, int, int], ...] = DEFAULT_OPAQUE_RGB_SENTINELS

    def __post_init__(self) -> None:
        colors = self.opaque_rgb_sentinels
        if not isinstance(colors, tuple):
            raise TypeError("opaque_rgb_sentinels must be a tuple of RGB tuples")
        normalized: list[tuple[int, int, int]] = []
        for color in colors:
            if not isinstance(color, tuple) or len(color) != 3:
                raise ValueError("each opaque RGB sentinel must be a three-integer tuple")
            if any(
                isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
                for channel in color
            ):
                raise ValueError("opaque RGB sentinel channels must be integers in [0, 255]")
            normalized.append(color)
        if len(set(normalized)) != len(normalized):
            raise ValueError("opaque RGB sentinels must be unique")


@dataclass(frozen=True, slots=True)
class PixelQualityAuditResult:
    """Identity of one atomically published pixel-quality report."""

    artifact_path: Path
    artifact_sha256: str
    verified_clip_count: int
    selected_clip_count: int


@dataclass(frozen=True, slots=True)
class _VerifiedClipRef:
    sequence_id: str
    split: str
    source_id: str
    path: Path
    file_sha256: str
    array_sha256: str
    shape: tuple[int, int, int, int]
    size_bytes: int


def build_materialized_pixel_quality_audit(
    manifest_path: Path | str,
    *,
    config: PixelQualityDetectionConfig | None = None,
    source_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Audit native stored RGBA arrays after verifying every manifest entry.

    Source filtering changes only which verified clips are scanned and reported.
    Schema, count, file-hash, array-hash, and semantic verification always covers
    the complete materialization manifest.
    """

    detection = config or PixelQualityDetectionConfig()
    if not isinstance(detection, PixelQualityDetectionConfig):
        raise TypeError("config must be a PixelQualityDetectionConfig")
    requested_sources = _normalize_source_filter(source_ids)
    manifest = Path(manifest_path).resolve()
    initial_manifest_bytes, manifest_root = _read_manifest(manifest)

    verified_clips = load_materialized_training_clips(manifest, split=None)
    if manifest.read_bytes() != initial_manifest_bytes:
        raise RuntimeError(f"materialization manifest changed during verification: {manifest}")
    references = tuple(_clip_reference(clip) for clip in verified_clips)
    del verified_clips

    known_sources = frozenset(reference.source_id for reference in references)
    if requested_sources is not None:
        unknown = requested_sources.difference(known_sources)
        if unknown:
            raise ValueError(f"requested source IDs are absent from manifest: {sorted(unknown)!r}")
    selected = tuple(
        reference
        for reference in references
        if requested_sources is None or reference.source_id in requested_sources
    )
    if not selected:
        raise ValueError("no materialized clips match the source filter")

    config_record = _config_record(detection)
    rows = tuple(_scan_clip(reference, detection) for reference in selected)
    if manifest.read_bytes() != initial_manifest_bytes:
        raise RuntimeError(f"materialization manifest changed during pixel scan: {manifest}")

    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_source_split: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source_id = str(row["source_id"])
        split = str(row["split"])
        by_source[source_id].append(row)
        by_split[split].append(row)
        by_source_split[(source_id, split)].append(row)

    return {
        "artifact_kind": "materialized_rgba_pixel_quality_audit",
        "clips": list(rows),
        "detection_config": config_record,
        "interpretation": {
            "opaque_rgb_sentinels": (
                "Exact RGB matches are counted only where stored alpha equals 255. "
                "They are detection signals, not inferred transparency or evidence of a defect."
            ),
            "pixel_mutation": False,
            "transparency_inference": False,
            "visible_canvas": "A stored pixel is visible exactly when alpha is greater than zero.",
        },
        "manifest": {
            "file_sha256": hashlib.sha256(initial_manifest_bytes).hexdigest(),
            "path": str(manifest),
            "schema_version": manifest_root["schema_version"],
            "sequence_count": manifest_root["sequence_count"],
        },
        "schema_version": 1,
        "selection": {
            "requested_source_ids": (
                None if requested_sources is None else sorted(requested_sources, key=str.encode)
            ),
            "selected_clip_count": len(selected),
            "selected_source_ids": sorted(by_source, key=str.encode),
        },
        "source_splits": [
            {
                "source_id": source_id,
                "split": split,
                "statistics": _aggregate(group, detection),
            }
            for (source_id, split), group in sorted(by_source_split.items())
        ],
        "sources": {
            source_id: _aggregate(group, detection)
            for source_id, group in sorted(by_source.items(), key=lambda item: item[0].encode())
        },
        "splits": {
            split: _aggregate(group, detection)
            for split, group in sorted(by_split.items(), key=lambda item: item[0].encode())
        },
        "summary": _aggregate(rows, detection),
        "verification": {
            "array_hash_algorithm": "sha256(dtype.str\\0shape\\0 + C-order array bytes)",
            "file_hash_algorithm": "sha256(file bytes)",
            "native_stored_arrays_scanned": True,
            "scope": "complete_manifest_before_source_filter",
            "verified_clip_count": len(references),
        },
    }


def export_materialized_pixel_quality_audit(
    manifest_path: Path | str,
    output_path: Path | str,
    *,
    config: PixelQualityDetectionConfig | None = None,
    source_ids: Iterable[str] | None = None,
    disk_guard: DiskGuard | None = None,
) -> PixelQualityAuditResult:
    """Build and atomically publish a canonical, no-clobber JSON report."""

    output = Path(output_path).resolve()
    if output.suffix.casefold() != ".json":
        raise ValueError("output_path must end in .json")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing pixel-quality audit: {output}")
    artifact = build_materialized_pixel_quality_audit(
        manifest_path,
        config=config,
        source_ids=source_ids,
    )
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
    return PixelQualityAuditResult(
        artifact_path=output,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        verified_clip_count=int(artifact["verification"]["verified_clip_count"]),
        selected_clip_count=int(artifact["selection"]["selected_clip_count"]),
    )


def _read_manifest(path: Path) -> tuple[bytes, Mapping[str, Any]]:
    try:
        payload = path.read_bytes()
        root = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read materialization manifest: {path}") from error
    if not isinstance(root, Mapping):
        raise ValueError("materialization manifest root must be an object")
    schema = root.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int):
        raise ValueError("materialization schema_version must be an integer")
    records = root.get("sequences")
    if not isinstance(records, list) or not records:
        raise ValueError("materialization manifest must contain sequences")
    declared = root.get("sequence_count")
    if declared != len(records):
        raise ValueError(f"sequence_count mismatch: declared {declared!r}, found {len(records)}")
    return payload, root


def _clip_reference(clip: MaterializedTrainingClip) -> _VerifiedClipRef:
    path = clip.source_path.resolve()
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        close = getattr(array, "close", None)
        if close is not None:
            close()
        raise ValueError(f"materialized clip must contain one NumPy array: {path}")
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 4:
        raise ValueError(f"materialized clip is not [T,H,W,RGBA]: {path}")
    return _VerifiedClipRef(
        sequence_id=clip.sequence_id,
        split=clip.split,
        source_id=clip.source_id,
        path=path,
        file_sha256=clip.source_file_sha256,
        array_sha256=clip.source_array_sha256,
        shape=shape,
        size_bytes=path.stat().st_size,
    )


def _scan_clip(
    reference: _VerifiedClipRef,
    config: PixelQualityDetectionConfig,
) -> dict[str, Any]:
    payload = reference.path.read_bytes()
    actual_file_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_file_sha256 != reference.file_sha256:
        raise RuntimeError(
            f"clip file changed during pixel audit for {reference.sequence_id!r}: "
            f"expected {reference.file_sha256}, got {actual_file_sha256}"
        )
    if len(payload) != reference.size_bytes:
        raise RuntimeError(
            f"clip size changed during pixel audit for {reference.sequence_id!r}: "
            f"expected {reference.size_bytes}, got {len(payload)}"
        )
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load materialized clip: {reference.path}") from error
    if not isinstance(loaded, np.ndarray):
        close = getattr(loaded, "close", None)
        if close is not None:
            close()
        raise ValueError(f"materialized clip must contain one NumPy array: {reference.path}")
    rgba = np.ascontiguousarray(loaded)
    actual_array_sha256 = _array_sha256(rgba)
    if actual_array_sha256 != reference.array_sha256:
        raise RuntimeError(
            f"clip array changed during pixel audit for {reference.sequence_id!r}: "
            f"expected {reference.array_sha256}, got {actual_array_sha256}"
        )
    if rgba.dtype != np.uint8 or tuple(rgba.shape) != reference.shape or rgba.shape[-1] != 4:
        raise ValueError(
            f"clip array contract changed for {reference.sequence_id!r}: "
            f"expected uint8 {reference.shape!r}, got {rgba.dtype} {rgba.shape!r}"
        )

    statistics = _array_statistics(rgba, config)
    return {
        "input": {
            "array_content_sha256": actual_array_sha256,
            "dtype": rgba.dtype.name,
            "file_sha256": actual_file_sha256,
            "path": str(reference.path),
            "shape": list(rgba.shape),
            "size_bytes": len(payload),
        },
        "sequence_id": reference.sequence_id,
        "source_id": reference.source_id,
        "split": reference.split,
        "statistics": statistics,
    }


def _array_statistics(
    rgba: np.ndarray,
    config: PixelQualityDetectionConfig,
) -> dict[str, Any]:
    frame_count, height, width, _ = rgba.shape
    pixel_count = frame_count * height * width
    alpha = rgba[..., 3]
    transparent = alpha == 0
    opaque = alpha == 255
    partial = (~transparent) & (~opaque)
    visible = ~transparent

    border_spatial = np.zeros((height, width), dtype=bool)
    border_spatial[0, :] = True
    border_spatial[-1, :] = True
    border_spatial[:, 0] = True
    border_spatial[:, -1] = True
    corner_spatial = np.zeros((height, width), dtype=bool)
    corner_spatial[0, 0] = True
    corner_spatial[0, -1] = True
    corner_spatial[-1, 0] = True
    corner_spatial[-1, -1] = True

    return {
        "alpha_distribution": {
            "fully_opaque": _count_fraction(np.count_nonzero(opaque), pixel_count),
            "fully_transparent": _count_fraction(np.count_nonzero(transparent), pixel_count),
            "partially_transparent": _count_fraction(np.count_nonzero(partial), pixel_count),
        },
        "border_occupancy": _occupancy(visible, border_spatial),
        "canvas": {
            "frame_count": frame_count,
            "height": height,
            "pixel_count": pixel_count,
            "width": width,
        },
        "clip_flags": {
            "fully_opaque": bool(np.all(opaque)),
            "fully_transparent": bool(np.all(transparent)),
        },
        "corner_occupancy": _occupancy(visible, corner_spatial),
        "frames": {
            "fully_opaque": _count_fraction(
                np.count_nonzero(np.all(opaque, axis=(1, 2))), frame_count
            ),
            "fully_transparent": _count_fraction(
                np.count_nonzero(np.all(transparent, axis=(1, 2))), frame_count
            ),
            "total": frame_count,
            "with_any_partial_alpha": _count_fraction(
                np.count_nonzero(np.any(partial, axis=(1, 2))), frame_count
            ),
            "with_any_visible_pixel": _count_fraction(
                np.count_nonzero(np.any(visible, axis=(1, 2))), frame_count
            ),
        },
        "opaque_rgb_sentinels": {
            _color_key(color): _sentinel_statistics(rgba, opaque, color)
            for color in config.opaque_rgb_sentinels
        },
        "visible_canvas": _count_fraction(np.count_nonzero(visible), pixel_count),
    }


def _occupancy(visible: np.ndarray, spatial_mask: np.ndarray) -> dict[str, Any]:
    frame_count = visible.shape[0]
    slot_count_per_frame = int(np.count_nonzero(spatial_mask))
    selected = visible[:, spatial_mask]
    return {
        "affected_frames": _count_fraction(np.count_nonzero(np.any(selected, axis=1)), frame_count),
        "fully_occupied_frames": _count_fraction(
            np.count_nonzero(np.all(selected, axis=1)), frame_count
        ),
        "visible_pixels": _count_fraction(
            np.count_nonzero(selected), frame_count * slot_count_per_frame
        ),
    }


def _sentinel_statistics(
    rgba: np.ndarray,
    opaque: np.ndarray,
    color: tuple[int, int, int],
) -> dict[str, Any]:
    matches = opaque & np.all(rgba[..., :3] == np.asarray(color, dtype=np.uint8), axis=-1)
    frame_count = rgba.shape[0]
    pixel_count = int(np.prod(rgba.shape[:3]))
    opaque_count = int(np.count_nonzero(opaque))
    match_count = int(np.count_nonzero(matches))
    return {
        "affected_clip": match_count > 0,
        "affected_frames": _count_fraction(
            np.count_nonzero(np.any(matches, axis=(1, 2))), frame_count
        ),
        "opaque_exact_match_pixels": _count_fraction(match_count, pixel_count),
        "opaque_pixel_fraction": _count_fraction(match_count, opaque_count),
        "rgb": list(color),
    }


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    config: PixelQualityDetectionConfig,
) -> dict[str, Any]:
    statistics = [row["statistics"] for row in rows]
    clip_count = len(rows)
    frame_count = sum(int(item["canvas"]["frame_count"]) for item in statistics)
    pixel_count = sum(int(item["canvas"]["pixel_count"]) for item in statistics)
    opaque_pixels = sum(
        int(item["alpha_distribution"]["fully_opaque"]["count"]) for item in statistics
    )
    result = {
        "alpha_distribution": {
            name: _count_fraction(
                sum(int(item["alpha_distribution"][name]["count"]) for item in statistics),
                pixel_count,
            )
            for name in ("fully_opaque", "fully_transparent", "partially_transparent")
        },
        "border_occupancy": _aggregate_occupancy(statistics, "border_occupancy"),
        "clip_count": clip_count,
        "clip_flags": {
            flag: _count_fraction(
                sum(bool(item["clip_flags"][flag]) for item in statistics), clip_count
            )
            for flag in ("fully_opaque", "fully_transparent")
        },
        "corner_occupancy": _aggregate_occupancy(statistics, "corner_occupancy"),
        "frame_count": frame_count,
        "frames": {
            name: _count_fraction(
                sum(int(item["frames"][name]["count"]) for item in statistics), frame_count
            )
            for name in (
                "fully_opaque",
                "fully_transparent",
                "with_any_partial_alpha",
                "with_any_visible_pixel",
            )
        },
        "opaque_rgb_sentinels": {},
        "pixel_count": pixel_count,
        "visible_canvas": _count_fraction(
            sum(int(item["visible_canvas"]["count"]) for item in statistics), pixel_count
        ),
    }
    sentinels: dict[str, Any] = result["opaque_rgb_sentinels"]
    for color in config.opaque_rgb_sentinels:
        key = _color_key(color)
        match_pixels = sum(
            int(item["opaque_rgb_sentinels"][key]["opaque_exact_match_pixels"]["count"])
            for item in statistics
        )
        affected_frames = sum(
            int(item["opaque_rgb_sentinels"][key]["affected_frames"]["count"])
            for item in statistics
        )
        affected_clips = sum(
            bool(item["opaque_rgb_sentinels"][key]["affected_clip"]) for item in statistics
        )
        sentinels[key] = {
            "affected_clips": _count_fraction(affected_clips, clip_count),
            "affected_frames": _count_fraction(affected_frames, frame_count),
            "opaque_exact_match_pixels": _count_fraction(match_pixels, pixel_count),
            "opaque_pixel_fraction": _count_fraction(match_pixels, opaque_pixels),
            "rgb": list(color),
        }
    return result


def _aggregate_occupancy(
    statistics: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    frame_count = sum(int(item["canvas"]["frame_count"]) for item in statistics)
    pixel_slots = sum(int(item[key]["visible_pixels"]["denominator"]) for item in statistics)
    return {
        "affected_frames": _count_fraction(
            sum(int(item[key]["affected_frames"]["count"]) for item in statistics),
            frame_count,
        ),
        "fully_occupied_frames": _count_fraction(
            sum(int(item[key]["fully_occupied_frames"]["count"]) for item in statistics),
            frame_count,
        ),
        "visible_pixels": _count_fraction(
            sum(int(item[key]["visible_pixels"]["count"]) for item in statistics),
            pixel_slots,
        ),
    }


def _config_record(config: PixelQualityDetectionConfig) -> dict[str, Any]:
    core = {
        "alpha_requirement": 255,
        "match_operation": "exact_stored_uint8_rgb_equality",
        "opaque_rgb_sentinels": [
            {"hex": _color_key(color), "rgb": list(color)} for color in config.opaque_rgb_sentinels
        ],
        "purpose": "detection_only",
    }
    payload = json.dumps(core, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {**core, "sha256": hashlib.sha256(payload).hexdigest()}


def _normalize_source_filter(values: Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, str | bytes):
        raise ValueError("source_ids must be an iterable of non-empty strings")
    try:
        normalized = tuple(values)
    except TypeError as error:
        raise ValueError("source_ids must be an iterable of non-empty strings") from error
    if not normalized:
        raise ValueError("source_ids cannot be empty when supplied")
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError("source_ids must contain only non-empty strings")
    return frozenset(normalized)


def _count_fraction(count: int | np.integer[Any], denominator: int) -> dict[str, Any]:
    numerator = int(count)
    total = int(denominator)
    return {
        "count": numerator,
        "denominator": total,
        "fraction": 0.0 if total == 0 else numerator / total,
    }


def _color_key(color: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in color)


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
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing pixel-quality audit: {path}")
    if disk_guard is not None:
        disk_guard.require_capacity(len(payload) + 65_536, label="pixel-quality audit")
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
                f"Refusing to replace existing pixel-quality audit: {path}"
            ) from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
