"""Deterministic materialization of canonical snapshots into RGBA clip tensors.

This module is intentionally a strict boundary between indexed provenance and model
inputs.  It does not guess sheet geometry, synthesize timing, resample time, or
downsample pixels.  Every output frame is selected by the source blob digest and
source frame index recorded in a canonical dataset snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from spritelab.captions import build_sprite_caption
from spritelab.dataset import SequenceSample
from spritelab.media.animation import extract_animation
from spritelab.media.models import Size
from spritelab.normalization import (
    Anchor,
    NormalizedSequence,
    OversizedSpriteError,
    normalize_sprite_sequence,
)
from spritelab.snapshot import SNAPSHOT_SCHEMA_VERSION
from spritelab.storage import DiskGuard

MATERIALIZATION_SCHEMA_VERSION = 1
DEFAULT_BUCKET_SIZES: tuple[int, ...] = (64, 128, 256, 512)
MATERIALIZATION_MANIFEST = "materialization.json"
PIXEL_TRANSFORM_SCHEMA = "spritelab.pixel_transform.v1"
PIXEL_TRANSFORM_OP = "exact_uint8_rgb_to_rgba_zero"
PIXEL_TRANSFORM_EXECUTION_METHOD = "audited_metadata_exact_uint8_rgb_v1"
SOURCE_RECT_COORDINATE_SPACES = frozenset(("source_image", "source_sheet"))


class MaterializationError(RuntimeError):
    """Base error for snapshot or clip materialization failures."""


class SnapshotValidationError(MaterializationError):
    """Raised when a snapshot does not satisfy the canonical input contract."""


class SnapshotHashMismatch(SnapshotValidationError):
    """Raised when the embedded dataset-manifest digest is incorrect."""


class SourceBlobValidationError(MaterializationError):
    """Raised when an indexed source blob cannot be validated or decoded."""


class SourceBlobHashMismatch(SourceBlobValidationError):
    """Raised when source bytes disagree with their indexed SHA-256."""


class FrameReconstructionError(MaterializationError):
    """Raised when exact source-frame reconstruction is not possible."""


class UnsupportedSheetCoordinatesError(FrameReconstructionError):
    """Raised instead of guessing an incomplete or unaudited sheet crop."""


class UnsupportedPixelTransformError(FrameReconstructionError):
    """Raised instead of executing malformed or unaudited pixel operations."""


class NoLosslessBucketError(MaterializationError):
    """Raised when a clip cannot fit any target bucket without downsampling."""


class ExistingOutputError(MaterializationError):
    """Raised before replacing an existing materialization artifact."""


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Paths and canonical content produced by :func:`materialize_snapshot`."""

    output_directory: Path
    manifest_path: Path
    clip_paths: tuple[Path, ...]
    manifest: dict[str, Any]

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.manifest)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _SnapshotSample:
    sample: SequenceSample
    split: str


@dataclass(frozen=True, slots=True)
class _ValidatedSnapshot:
    schema_version: int
    canonical_sha256: str
    manifest_sha256: str
    samples: tuple[_SnapshotSample, ...]


@dataclass(frozen=True, slots=True)
class _BlobSpec:
    sha256: str
    size_bytes: int
    mime_type: str | None
    path: Path


@dataclass(frozen=True, slots=True)
class _ReconstructedFrames:
    images: tuple[Image.Image, ...]
    provenance: tuple[dict[str, Any], ...]
    durations_ms: tuple[float | int | None, ...]
    phases: tuple[float | int | None, ...]


@dataclass(frozen=True, slots=True)
class _SourceImageRect:
    bounds: tuple[int, int, int, int]
    coordinate_space: str


@dataclass(frozen=True, slots=True)
class _PixelTransformSpec:
    schema: str
    op: str
    rgb: tuple[int, int, int]
    evidence: tuple[dict[str, Any], ...]
    transform_sha256: str

    def as_metadata(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "op": self.op,
            "rgb": self.rgb,
            "evidence": self.evidence,
            "transform_sha256": self.transform_sha256,
        }


def materialize_snapshot(
    snapshot_path: Path | str,
    output_directory: Path | str,
    *,
    blob_root: Path | str | None = None,
    bucket_sizes: Sequence[int | Size] = DEFAULT_BUCKET_SIZES,
    anchor: Anchor = "bottom_center",
    padding: int | tuple[int, int, int, int] = 0,
    alpha_threshold: int = 0,
    upscale: bool = True,
    max_integer_scale: int | None = None,
    overwrite: bool = False,
    disk_guard: DiskGuard | None = None,
) -> MaterializationResult:
    """Materialize one canonical snapshot into deterministic ``.npy`` RGBA clips.

    Relative blob ``storage_path`` values are resolved below ``blob_root`` when it is
    supplied, otherwise below the snapshot's parent directory. Existing outputs are
    never replaced unless ``overwrite=True``. Each clip is published atomically as it
    succeeds; if a later clip fails, earlier valid clips remain in place.
    """

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    snapshot_file = Path(snapshot_path).resolve()
    output_root = Path(output_directory).resolve()
    relative_blob_root = snapshot_file.parent if blob_root is None else Path(blob_root).resolve()
    snapshot = _load_snapshot(snapshot_file)
    buckets = _normalize_bucket_sizes(bucket_sizes)
    normalized_padding = _normalize_padding(padding)

    targets = tuple(
        output_root / _clip_relative_path(entry.sample.sequence_id, entry.split)
        for entry in snapshot.samples
    )
    manifest_path = output_root / MATERIALIZATION_MANIFEST
    if not overwrite:
        existing = tuple(path for path in (*targets, manifest_path) if path.exists())
        if existing:
            rendered = ", ".join(str(path) for path in existing[:3])
            suffix = " ..." if len(existing) > 3 else ""
            raise ExistingOutputError(
                f"Refusing to replace {len(existing)} existing output artifact(s): "
                f"{rendered}{suffix}"
            )

    validated_paths: dict[tuple[str, Path], _BlobSpec] = {}
    records: list[dict[str, Any]] = []
    clip_paths: list[Path] = []
    for entry, destination in zip(snapshot.samples, targets, strict=True):
        specs = _blob_specs(
            entry.sample,
            snapshot_file=snapshot_file,
            blob_root=relative_blob_root,
        )
        for spec in specs.values():
            cache_key = spec.sha256, spec.path
            if cache_key not in validated_paths:
                _validate_blob(spec)
                validated_paths[cache_key] = spec

        reconstructed = _reconstruct_frames(entry.sample, specs)
        normalized, bucket = _smallest_lossless_bucket(
            reconstructed.images,
            buckets=buckets,
            anchor=anchor,
            padding=normalized_padding,
            alpha_threshold=alpha_threshold,
            upscale=upscale,
            max_integer_scale=max_integer_scale,
        )
        array = _rgba_array(normalized.frames)
        file_sha256, size_bytes = _atomic_write_npy(
            destination,
            array,
            overwrite=overwrite,
            disk_guard=disk_guard,
        )
        clip_paths.append(destination)
        records.append(
            _materialized_record(
                entry,
                reconstructed=reconstructed,
                normalized=normalized,
                bucket=bucket,
                array=array,
                output_root=output_root,
                destination=destination,
                file_sha256=file_sha256,
                size_bytes=size_bytes,
                blob_specs=specs,
            )
        )

    manifest = {
        "config": {
            "alpha_threshold": alpha_threshold,
            "anchor": anchor,
            "bucket_sizes": buckets,
            "frame_order": "frame_provenance.ordinal",
            "max_integer_scale": max_integer_scale,
            "padding": normalized_padding,
            "source_frame_selector": "source_blob_sha256+source_frame_index",
            "pixel_transform_contract": PIXEL_TRANSFORM_SCHEMA,
            "pixel_transform_execution": PIXEL_TRANSFORM_EXECUTION_METHOD,
            "spatial_resampling": "none_or_nearest_positive_integer",
            "temporal_resampling": "none",
            "upscale": upscale,
        },
        "schema_version": MATERIALIZATION_SCHEMA_VERSION,
        "sequence_count": len(records),
        "sequences": records,
        "source_snapshot": {
            "canonical_sha256": snapshot.canonical_sha256,
            "manifest_sha256": snapshot.manifest_sha256,
            "schema_version": snapshot.schema_version,
        },
    }
    manifest_bytes = (_canonical_json(manifest) + "\n").encode("utf-8")
    _atomic_write_bytes(
        manifest_path,
        manifest_bytes,
        overwrite=overwrite,
        disk_guard=disk_guard,
        label="materialization manifest",
    )
    return MaterializationResult(
        output_directory=output_root,
        manifest_path=manifest_path,
        clip_paths=tuple(clip_paths),
        manifest=manifest,
    )


def _load_snapshot(path: Path) -> _ValidatedSnapshot:
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot does not exist: {path}")
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotValidationError(f"Snapshot is not valid UTF-8 JSON: {path}") from error
    if not isinstance(root, Mapping):
        raise SnapshotValidationError("Snapshot root must be a JSON object")
    schema_version = _required_integer(root, "schema_version", minimum=1)
    if schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotValidationError(
            f"Unsupported snapshot schema {schema_version}; expected {SNAPSHOT_SCHEMA_VERSION}"
        )
    manifest = root.get("manifest")
    if not isinstance(manifest, Mapping):
        raise SnapshotValidationError("Snapshot manifest must be a JSON object")
    manifest_schema_version = _required_integer(manifest, "schema_version", minimum=1)
    if manifest_schema_version != 1:
        raise SnapshotValidationError(
            f"Unsupported dataset manifest schema {manifest_schema_version}; expected 1"
        )
    expected_manifest_sha = _required_digest(root, "manifest_sha256")
    actual_manifest_sha = hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()
    if actual_manifest_sha != expected_manifest_sha:
        raise SnapshotHashMismatch(
            f"Snapshot manifest SHA-256 mismatch: expected {expected_manifest_sha}, "
            f"computed {actual_manifest_sha}"
        )

    raw_samples = manifest.get("samples")
    raw_assignments = manifest.get("assignments")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise SnapshotValidationError("Snapshot manifest must contain at least one sample")
    if not isinstance(raw_assignments, list):
        raise SnapshotValidationError("Snapshot manifest assignments must be a list")

    assignments: dict[str, str] = {}
    for raw in raw_assignments:
        if not isinstance(raw, Mapping):
            raise SnapshotValidationError("Each split assignment must be a JSON object")
        sequence_id = _required_string(raw, "sequence_id")
        split = _required_string(raw, "split")
        if split not in {"train", "validation", "test"}:
            raise SnapshotValidationError(f"Unsupported split {split!r} for {sequence_id!r}")
        if sequence_id in assignments:
            raise SnapshotValidationError(f"Duplicate split assignment for {sequence_id!r}")
        assignments[sequence_id] = split

    samples: list[_SnapshotSample] = []
    seen: set[str] = set()
    for raw in raw_samples:
        if not isinstance(raw, Mapping):
            raise SnapshotValidationError("Each manifest sample must be a JSON object")
        sample = _parse_sample(raw)
        if sample.sequence_id in seen:
            raise SnapshotValidationError(f"Duplicate sample sequence ID {sample.sequence_id!r}")
        seen.add(sample.sequence_id)
        if sample.sequence_id not in assignments:
            raise SnapshotValidationError(f"Sample {sample.sequence_id!r} has no split assignment")
        samples.append(_SnapshotSample(sample=sample, split=assignments[sample.sequence_id]))
    extra_assignments = set(assignments).difference(seen)
    if extra_assignments:
        raise SnapshotValidationError(
            f"Split assignments reference unknown samples: {sorted(extra_assignments)!r}"
        )

    canonical_snapshot_sha = hashlib.sha256(_canonical_json(root).encode("utf-8")).hexdigest()
    return _ValidatedSnapshot(
        schema_version=schema_version,
        canonical_sha256=canonical_snapshot_sha,
        manifest_sha256=actual_manifest_sha,
        samples=tuple(sorted(samples, key=lambda entry: entry.sample.sequence_id.encode("utf-8"))),
    )


def _parse_sample(raw: Mapping[str, Any]) -> SequenceSample:
    source_hashes = _digest_sequence(raw.get("source_blob_sha256"), "source_blob_sha256")
    duplicate_ids = _string_sequence(raw.get("duplicate_group_ids", ()), "duplicate_group_ids")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise SnapshotValidationError("sample metadata must be a JSON object")
    frame_count = _required_integer(raw, "frame_count", minimum=1)
    sample_weight = raw.get("sample_weight", 1.0)
    if not isinstance(sample_weight, int | float) or isinstance(sample_weight, bool):
        raise SnapshotValidationError("sample_weight must be numeric")
    try:
        return SequenceSample(
            sequence_id=_required_string(raw, "sequence_id"),
            identity_id=_required_string(raw, "identity_id"),
            source_id=_required_string(raw, "source_id"),
            source_pack_id=_required_string(raw, "source_pack_id"),
            entity_class=_required_string(raw, "entity_class"),
            action=_required_string(raw, "action"),
            view=_required_string(raw, "view"),
            direction=_required_string(raw, "direction"),
            loop_mode=_required_string(raw, "loop_mode"),
            frame_count=frame_count,
            source_blob_sha256=source_hashes,
            duplicate_group_ids=duplicate_ids,
            quality_tier=_required_string(raw, "quality_tier"),
            sample_weight=float(sample_weight),
            metadata=metadata,
        )
    except ValueError as error:
        raise SnapshotValidationError(f"Invalid sequence sample: {error}") from error


def _blob_specs(
    sample: SequenceSample,
    *,
    snapshot_file: Path,
    blob_root: Path,
) -> dict[str, _BlobSpec]:
    raw_records = sample.metadata.get("blob_records")
    if not isinstance(raw_records, list):
        raise SnapshotValidationError(f"Sample {sample.sequence_id!r} has no blob_records list")
    records: dict[str, _BlobSpec] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise SnapshotValidationError(
                f"Sample {sample.sequence_id!r} has a non-object blob record"
            )
        digest = _required_digest(raw, "sha256")
        if digest in records:
            raise SnapshotValidationError(
                f"Sample {sample.sequence_id!r} repeats blob record {digest}"
            )
        size_bytes = _required_integer(raw, "size_bytes", minimum=0)
        mime_type = raw.get("mime_type")
        if mime_type is not None and not isinstance(mime_type, str):
            raise SnapshotValidationError(f"Blob {digest} has a non-string MIME type")
        storage_path = _required_string(raw, "storage_path")
        records[digest] = _BlobSpec(
            sha256=digest,
            size_bytes=size_bytes,
            mime_type=mime_type,
            path=_resolve_storage_path(
                storage_path,
                snapshot_file=snapshot_file,
                blob_root=blob_root,
            ),
        )
    expected = set(sample.source_blob_sha256)
    actual = set(records)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise SnapshotValidationError(
            f"Sample {sample.sequence_id!r} blob_records disagree with source hashes; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return records


def _resolve_storage_path(
    storage_path: str,
    *,
    snapshot_file: Path,
    blob_root: Path,
) -> Path:
    del snapshot_file  # Kept in the signature to make the resolution context explicit.
    raw = Path(storage_path)
    if raw.is_absolute():
        return raw.resolve()
    root = blob_root.resolve()
    resolved = (root / raw).resolve()
    if not resolved.is_relative_to(root):
        raise SnapshotValidationError(
            f"Relative blob storage path escapes blob_root: {storage_path!r}"
        )
    return resolved


def _validate_blob(spec: _BlobSpec) -> None:
    if not spec.path.is_file():
        raise SourceBlobValidationError(f"Source blob is missing: {spec.path}")
    actual_size = spec.path.stat().st_size
    if actual_size != spec.size_bytes:
        raise SourceBlobHashMismatch(
            f"Blob {spec.sha256} size mismatch at {spec.path}: "
            f"expected {spec.size_bytes}, received {actual_size}"
        )
    actual_hash = _hash_path(spec.path)
    if actual_hash != spec.sha256:
        raise SourceBlobHashMismatch(
            f"Blob SHA-256 mismatch at {spec.path}: expected {spec.sha256}, received {actual_hash}"
        )


def _reconstruct_frames(
    sample: SequenceSample,
    blob_specs: Mapping[str, _BlobSpec],
) -> _ReconstructedFrames:
    raw_frames = sample.metadata.get("frame_provenance")
    if not isinstance(raw_frames, list):
        raise FrameReconstructionError(
            f"Sample {sample.sequence_id!r} has no frame_provenance list"
        )
    ordered: list[Mapping[str, Any]] = []
    for raw in raw_frames:
        if not isinstance(raw, Mapping):
            raise FrameReconstructionError(
                f"Sample {sample.sequence_id!r} has a non-object frame provenance row"
            )
        exact_source_rect = _exact_source_image_rect(sample.sequence_id, raw)
        _reject_other_sheet_coordinates(
            sample.sequence_id,
            raw,
            allow_exact_frame_rect=exact_source_rect is not None,
        )
        ordered.append(raw)
    ordered.sort(key=lambda row: _required_integer(row, "ordinal", minimum=0))
    ordinals = [_required_integer(row, "ordinal", minimum=0) for row in ordered]
    if ordinals != list(range(sample.frame_count)):
        raise FrameReconstructionError(
            f"Sample {sample.sequence_id!r} frame ordinals must be exactly "
            f"0..{sample.frame_count - 1}; received {ordinals!r}"
        )

    carrier_cache: dict[str, Any] = {}
    images: list[Image.Image] = []
    provenance: list[dict[str, Any]] = []
    durations: list[float | int | None] = []
    phases: list[float | int | None] = []
    for raw in ordered:
        ordinal = _required_integer(raw, "ordinal", minimum=0)
        digest = _required_digest(raw, "source_blob_sha256")
        if digest not in blob_specs:
            raise FrameReconstructionError(
                f"Sample {sample.sequence_id!r} frame {ordinal} references undeclared blob {digest}"
            )
        source_index = _required_integer(raw, "source_frame_index", minimum=0)
        exact_source_rect = _exact_source_image_rect(sample.sequence_id, raw)
        duration = _optional_finite_number(raw.get("duration_ms"), "duration_ms", minimum=0)
        phase = _optional_finite_number(raw.get("phase"), "phase")
        if digest not in carrier_cache:
            try:
                inspection = extract_animation(blob_specs[digest].path)
            except (OSError, ValueError) as error:
                raise SourceBlobValidationError(
                    f"Blob {digest} cannot be decoded as a supported animation carrier"
                ) from error
            if inspection.source_sha256 != digest:
                raise SourceBlobHashMismatch(
                    f"Animation decoder read SHA-256 {inspection.source_sha256}, expected {digest}"
                )
            carrier_cache[digest] = inspection
        inspection = carrier_cache[digest]
        by_source_index = {frame.source_index: frame for frame in inspection.frames}
        if exact_source_rect is not None:
            if inspection.frame_count != 1 or len(by_source_index) != 1:
                raise FrameReconstructionError(
                    f"Sample {sample.sequence_id!r} frame {ordinal} supplies a source-sheet "
                    "rectangle for a multi-frame carrier"
                )
            source_frame = next(iter(inspection.frames))
            left, top, right, bottom = exact_source_rect.bounds
            sheet_width, sheet_height = source_frame.image.size
            if right > sheet_width or bottom > sheet_height:
                raise FrameReconstructionError(
                    f"Sample {sample.sequence_id!r} frame {ordinal} rectangle "
                    f"{exact_source_rect.bounds!r} exceeds source sheet "
                    f"{(sheet_width, sheet_height)!r}"
                )
            image = source_frame.image.crop(exact_source_rect.bounds).convert("RGBA")
            reconstruction_method = "audited_source_sheet_rectangle_v1"
        else:
            source_frame = by_source_index.get(source_index)
            if source_frame is None:
                detail = (
                    " Static sprite sheets require an exact audited frame_rect; this "
                    "materializer does not infer one from source_frame_index."
                    if inspection.frame_count == 1 and source_index > 0
                    else ""
                )
                raise FrameReconstructionError(
                    f"Sample {sample.sequence_id!r} frame {ordinal} requests source frame "
                    f"{source_index}, but blob {digest} exposes playback indices "
                    f"{sorted(by_source_index)!r}.{detail}"
                )
            image = source_frame.image.copy()
            reconstruction_method = "animation_playback_frame_index_v1"
        pre_transform_pixel_sha256 = _rgba_pixel_sha256(image)
        transforms = _validated_pixel_transforms(sample.sequence_id, ordinal, raw)
        image, transform_results = _execute_pixel_transforms(image, transforms)
        post_transform_pixel_sha256 = _rgba_pixel_sha256(image)
        images.append(image)
        durations.append(duration)
        phases.append(phase)
        provenance.append(
            {
                "carrier_format": inspection.format,
                "carrier_frame_duration_ms": source_frame.duration_ms,
                "direction": raw.get("direction"),
                "duration_ms": duration,
                "native_size": image.size,
                "ordinal": ordinal,
                "phase": phase,
                "pixel_transform_execution": PIXEL_TRANSFORM_EXECUTION_METHOD,
                "pixel_transform_results": transform_results,
                "pixel_transforms": tuple(transform.as_metadata() for transform in transforms),
                "post_transform_pixel_sha256": post_transform_pixel_sha256,
                "pre_transform_pixel_sha256": pre_transform_pixel_sha256,
                "reconstruction_method": reconstruction_method,
                "source_blob_sha256": digest,
                "source_carrier_size": (
                    source_frame.image.size if exact_source_rect is not None else None
                ),
                "source_frame_index": source_index,
                "source_frame_pixel_sha256": pre_transform_pixel_sha256,
                "source_rect": (
                    exact_source_rect.bounds if exact_source_rect is not None else None
                ),
                "source_rect_coordinate_space": (
                    exact_source_rect.coordinate_space if exact_source_rect is not None else None
                ),
                "source_sheet_size": (
                    source_frame.image.size if exact_source_rect is not None else None
                ),
                "view": raw.get("view"),
            }
        )
    return _ReconstructedFrames(
        images=tuple(images),
        provenance=tuple(provenance),
        durations_ms=tuple(durations),
        phases=tuple(phases),
    )


def _exact_source_image_rect(
    sequence_id: str,
    frame: Mapping[str, Any],
) -> _SourceImageRect | None:
    metadata = frame.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("frame_rect") is None:
        return None
    raw = metadata["frame_rect"]
    if not isinstance(raw, Mapping):
        raise UnsupportedSheetCoordinatesError(
            f"Sample {sequence_id!r} metadata.frame_rect must be an object"
        )
    coordinate_space = raw.get("coordinate_space")
    if (
        not isinstance(coordinate_space, str)
        or coordinate_space not in SOURCE_RECT_COORDINATE_SPACES
    ):
        raise UnsupportedSheetCoordinatesError(
            f"Sample {sequence_id!r} metadata.frame_rect must declare "
            "coordinate_space='source_image' or 'source_sheet'"
        )
    values: dict[str, int] = {}
    for key in ("left", "top", "right", "bottom"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise UnsupportedSheetCoordinatesError(
                f"Sample {sequence_id!r} metadata.frame_rect.{key} must be an integer"
            )
        values[key] = value
    left, top, right, bottom = (values[key] for key in ("left", "top", "right", "bottom"))
    if left < 0 or top < 0 or right <= left or bottom <= top:
        raise UnsupportedSheetCoordinatesError(
            f"Sample {sequence_id!r} metadata.frame_rect has invalid bounds "
            f"{(left, top, right, bottom)!r}"
        )
    for key, expected in (("width", right - left), ("height", bottom - top)):
        value = raw.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value != expected
        ):
            raise UnsupportedSheetCoordinatesError(
                f"Sample {sequence_id!r} metadata.frame_rect.{key} must equal {expected}"
            )
    return _SourceImageRect(
        bounds=(left, top, right, bottom),
        coordinate_space=coordinate_space,
    )


def _validated_pixel_transforms(
    sequence_id: str,
    ordinal: int,
    frame: Mapping[str, Any],
) -> tuple[_PixelTransformSpec, ...]:
    metadata = frame.get("metadata")
    if not isinstance(metadata, Mapping):
        if metadata is None:
            return ()
        raise UnsupportedPixelTransformError(
            f"Sample {sequence_id!r} frame {ordinal} metadata must be an object"
        )
    raw_transforms = metadata.get("pixel_transforms")
    if raw_transforms is None:
        return ()
    if not isinstance(raw_transforms, list):
        raise UnsupportedPixelTransformError(
            f"Sample {sequence_id!r} frame {ordinal} metadata.pixel_transforms must be a list"
        )

    transforms: list[_PixelTransformSpec] = []
    seen_hashes: set[str] = set()
    required_transform_keys = {"schema", "op", "rgb", "evidence", "transform_sha256"}
    required_evidence_keys = {"member_path", "sha256", "line_numbers", "scope", "claim"}
    for transform_index, raw in enumerate(raw_transforms):
        label = (
            f"Sample {sequence_id!r} frame {ordinal} metadata.pixel_transforms[{transform_index}]"
        )
        if not isinstance(raw, Mapping):
            raise UnsupportedPixelTransformError(f"{label} must be an object")
        raw_keys = {str(key) for key in raw}
        if raw_keys != required_transform_keys:
            raise UnsupportedPixelTransformError(
                f"{label} keys must be exactly {sorted(required_transform_keys)!r}; "
                f"received {sorted(raw_keys)!r}"
            )
        schema = raw.get("schema")
        if schema != PIXEL_TRANSFORM_SCHEMA:
            raise UnsupportedPixelTransformError(
                f"{label}.schema must equal {PIXEL_TRANSFORM_SCHEMA!r}"
            )
        op = raw.get("op")
        if op != PIXEL_TRANSFORM_OP:
            raise UnsupportedPixelTransformError(f"{label}.op must equal {PIXEL_TRANSFORM_OP!r}")
        raw_rgb = raw.get("rgb")
        if (
            not isinstance(raw_rgb, list)
            or len(raw_rgb) != 3
            or any(
                isinstance(channel, bool)
                or not isinstance(channel, int)
                or channel < 0
                or channel > 255
                for channel in raw_rgb
            )
        ):
            raise UnsupportedPixelTransformError(
                f"{label}.rgb must be exactly three uint8 integer values"
            )
        rgb = tuple(raw_rgb)

        raw_evidence = raw.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise UnsupportedPixelTransformError(f"{label}.evidence must be a non-empty list")
        evidence: list[dict[str, Any]] = []
        seen_members: set[str] = set()
        for evidence_index, raw_item in enumerate(raw_evidence):
            evidence_label = f"{label}.evidence[{evidence_index}]"
            if not isinstance(raw_item, Mapping):
                raise UnsupportedPixelTransformError(f"{evidence_label} must be an object")
            item_keys = {str(key) for key in raw_item}
            if item_keys != required_evidence_keys:
                raise UnsupportedPixelTransformError(
                    f"{evidence_label} keys must be exactly "
                    f"{sorted(required_evidence_keys)!r}; received {sorted(item_keys)!r}"
                )
            member_path = _pixel_transform_string(raw_item, "member_path", evidence_label)
            parsed_member = PurePosixPath(member_path)
            if (
                parsed_member.is_absolute()
                or not parsed_member.parts
                or any(part in {"", ".", ".."} for part in parsed_member.parts)
                or ":" in parsed_member.parts[0]
            ):
                raise UnsupportedPixelTransformError(
                    f"{evidence_label}.member_path is not a safe archive member path"
                )
            if member_path in seen_members:
                raise UnsupportedPixelTransformError(
                    f"{label}.evidence repeats member_path {member_path!r}"
                )
            seen_members.add(member_path)
            digest = _pixel_transform_digest(raw_item, "sha256", evidence_label)
            raw_lines = raw_item.get("line_numbers")
            if (
                not isinstance(raw_lines, list)
                or not raw_lines
                or any(
                    isinstance(line, bool) or not isinstance(line, int) or line < 1
                    for line in raw_lines
                )
            ):
                raise UnsupportedPixelTransformError(
                    f"{evidence_label}.line_numbers must be non-empty positive integers"
                )
            if raw_lines != sorted(set(raw_lines)):
                raise UnsupportedPixelTransformError(
                    f"{evidence_label}.line_numbers must be sorted and unique"
                )
            evidence.append(
                {
                    "member_path": member_path,
                    "sha256": digest,
                    "line_numbers": tuple(raw_lines),
                    "scope": _pixel_transform_string(raw_item, "scope", evidence_label),
                    "claim": _pixel_transform_string(raw_item, "claim", evidence_label),
                }
            )

        transform_hash = _pixel_transform_digest(raw, "transform_sha256", label)
        hash_payload = {
            "schema": schema,
            "op": op,
            "rgb": rgb,
            "evidence": tuple(evidence),
        }
        computed_hash = hashlib.sha256(_canonical_json(hash_payload).encode("utf-8")).hexdigest()
        if computed_hash != transform_hash:
            raise UnsupportedPixelTransformError(
                f"{label}.transform_sha256 mismatch: expected {transform_hash}, "
                f"computed {computed_hash}"
            )
        if transform_hash in seen_hashes:
            raise UnsupportedPixelTransformError(
                f"{label} repeats transform SHA-256 {transform_hash}"
            )
        seen_hashes.add(transform_hash)
        transforms.append(
            _PixelTransformSpec(
                schema=str(schema),
                op=str(op),
                rgb=(int(rgb[0]), int(rgb[1]), int(rgb[2])),
                evidence=tuple(evidence),
                transform_sha256=transform_hash,
            )
        )
    return tuple(transforms)


def _pixel_transform_string(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedPixelTransformError(f"{label}.{key} must be a non-empty string")
    return value


def _pixel_transform_digest(
    mapping: Mapping[str, Any],
    key: str,
    label: str,
) -> str:
    value = _pixel_transform_string(mapping, key, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise UnsupportedPixelTransformError(f"{label}.{key} is not a lowercase SHA-256 digest")
    return value


def _execute_pixel_transforms(
    image: Image.Image,
    transforms: tuple[_PixelTransformSpec, ...],
) -> tuple[Image.Image, tuple[dict[str, Any], ...]]:
    if not transforms:
        return image.convert("RGBA"), ()
    pixels = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    results: list[dict[str, Any]] = []
    for transform in transforms:
        rgb = np.asarray(transform.rgb, dtype=np.uint8)
        matches = np.all(pixels[..., :3] == rgb, axis=-1)
        matched_pixel_count = int(np.count_nonzero(matches))
        pixels[matches] = np.asarray((0, 0, 0, 0), dtype=np.uint8)
        results.append(
            {
                "matched_pixel_count": matched_pixel_count,
                "transform_sha256": transform.transform_sha256,
            }
        )
    return Image.fromarray(pixels), tuple(results)


def _reject_other_sheet_coordinates(
    sequence_id: str,
    frame: Mapping[str, Any],
    *,
    allow_exact_frame_rect: bool,
) -> None:
    inspected: Mapping[str, Any] = frame
    if allow_exact_frame_rect:
        inspected = dict(frame)
        metadata = frame.get("metadata")
        assert isinstance(metadata, Mapping)
        sanitized_metadata = dict(metadata)
        sanitized_metadata.pop("frame_rect", None)
        inspected["metadata"] = sanitized_metadata  # type: ignore[index]
    coordinate_path = _sheet_coordinate_path(inspected)
    if coordinate_path is not None:
        raise UnsupportedSheetCoordinatesError(
            f"Sample {sequence_id!r} contains unsupported sheet coordinates at "
            f"{coordinate_path}; only a complete audited metadata.frame_rect in "
            "source_image/source_sheet coordinates is executable"
        )


def _sheet_coordinate_path(value: object, path: str = "frame_provenance") -> str | None:
    if isinstance(value, Mapping):
        lowered = {str(key).casefold(): key for key in value}
        explicit = {
            "bbox",
            "bbox_json",
            "cell_bbox",
            "cell_rect",
            "crop",
            "crop_box",
            "frame_rect",
            "sheet_coordinates",
            "sheet_rect",
            "source_bbox",
            "source_rect",
            "uv_rect",
        }
        for key in sorted(lowered):
            nested = value[lowered[key]]
            coordinate_named = (
                key in explicit
                or "uv_rect" in key
                or (
                    key.endswith(("_bbox", "_crop", "_rect"))
                    and not key.startswith(("has_", "is_", "within_"))
                )
                or key
                in {
                    "cell",
                    "cells",
                    "cell_coordinates",
                    "coordinates",
                    "frame_cell",
                    "frame_cells",
                }
            )
            if coordinate_named and nested is not None:
                return f"{path}.{lowered[key]}"
        if (
            {"x", "y", "width", "height"}.issubset(lowered)
            or {"x", "y", "w", "h"}.issubset(lowered)
            or {"left", "top", "right", "bottom"}.issubset(lowered)
        ):
            return path
        for key, nested in value.items():
            result = _sheet_coordinate_path(nested, f"{path}.{key}")
            if result is not None:
                return result
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            result = _sheet_coordinate_path(nested, f"{path}[{index}]")
            if result is not None:
                return result
    return None


def _smallest_lossless_bucket(
    frames: Sequence[Image.Image],
    *,
    buckets: tuple[Size, ...],
    anchor: Anchor,
    padding: tuple[int, int, int, int],
    alpha_threshold: int,
    upscale: bool,
    max_integer_scale: int | None,
) -> tuple[NormalizedSequence, Size]:
    failures: list[str] = []
    for bucket in buckets:
        try:
            normalized = normalize_sprite_sequence(
                frames,
                target_size=bucket,
                padding=padding,
                alpha_threshold=alpha_threshold,
                anchor=anchor,
                upscale=upscale,
                max_integer_scale=max_integer_scale,
            )
        except OversizedSpriteError as error:
            failures.append(str(error))
            continue
        if normalized.transform.integer_scale < 1 or normalized.transform.resampling not in {
            "none",
            "nearest_positive_integer",
        }:
            raise MaterializationError("Normalization violated the lossless spatial contract")
        return normalized, bucket
    raise NoLosslessBucketError(
        f"Clip does not fit any configured bucket {buckets!r} without downsampling; "
        f"last rejection: {failures[-1] if failures else 'no buckets'}"
    )


def _materialized_record(
    entry: _SnapshotSample,
    *,
    reconstructed: _ReconstructedFrames,
    normalized: NormalizedSequence,
    bucket: Size,
    array: np.ndarray,
    output_root: Path,
    destination: Path,
    file_sha256: str,
    size_bytes: int,
    blob_specs: Mapping[str, _BlobSpec],
) -> dict[str, Any]:
    sample = entry.sample
    caption = build_sprite_caption(sample)
    metadata = sample.metadata
    total_duration: float | None = None
    if all(value is not None for value in reconstructed.durations_ms):
        total_duration = float(sum(float(value) for value in reconstructed.durations_ms))
    return {
        "action": sample.action,
        "caption": asdict(caption),
        "direction": sample.direction,
        "entity_class": sample.entity_class,
        "frame_count": sample.frame_count,
        "frame_provenance": reconstructed.provenance,
        "identity_id": sample.identity_id,
        "loop_mode": sample.loop_mode,
        "normalization": {
            "frame_pixel_sha256": normalized.frame_pixel_sha256,
            "transform": asdict(normalized.transform),
            "transform_sha256": normalized.transform.sha256,
        },
        "pixel_transform": _pixel_transform_manifest(reconstructed.provenance),
        "output": {
            "array_content_sha256": _array_sha256(array),
            "dtype": str(array.dtype),
            "file_sha256": file_sha256,
            "format": "numpy_npy_v1",
            "relative_path": _relative_posix(destination, output_root),
            "shape": array.shape,
            "size_bytes": size_bytes,
        },
        "provenance": {
            "archive_occurrences": metadata.get("archive_occurrences", []),
            "item": metadata.get("item"),
            "item_blob_occurrence_ids": metadata.get("item_blob_occurrence_ids", []),
            "retrieval_ids": metadata.get("retrieval_ids", []),
            "rights_observation_ids": metadata.get("rights_observation_ids", []),
            "sequence_source_keys": metadata.get("sequence_source_keys", []),
            "source": metadata.get("source"),
            "source_blob_records": [
                {
                    "mime_type": blob_specs[digest].mime_type,
                    "sha256": digest,
                    "size_bytes": blob_specs[digest].size_bytes,
                }
                for digest in sample.source_blob_sha256
            ],
            "source_blob_sha256": sample.source_blob_sha256,
            "source_id": sample.source_id,
            "source_ids": metadata.get("source_ids", [sample.source_id]),
            "source_pack_id": sample.source_pack_id,
        },
        "quality_tier": sample.quality_tier,
        "sample_weight": sample.sample_weight,
        "sequence_id": sample.sequence_id,
        "split": entry.split,
        "target_bucket": bucket,
        "timing": {
            "duration_ms": reconstructed.durations_ms,
            "phase": reconstructed.phases,
            "temporal_evidence": metadata.get("temporal_evidence"),
            "total_duration_ms": total_duration,
        },
        "view": sample.view,
    }


def _pixel_transform_manifest(
    frame_provenance: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    unique_transforms: dict[str, dict[str, Any]] = {}
    frame_results: list[dict[str, Any]] = []
    total_matches = 0
    transformed_frames = 0
    for frame in frame_provenance:
        transforms = frame["pixel_transforms"]
        results = frame["pixel_transform_results"]
        for transform in transforms:
            unique_transforms[transform["transform_sha256"]] = transform
        frame_matches = sum(int(result["matched_pixel_count"]) for result in results)
        total_matches += frame_matches
        transformed_frames += bool(transforms)
        frame_results.append(
            {
                "matched_pixel_count": frame_matches,
                "ordinal": frame["ordinal"],
                "post_transform_pixel_sha256": frame["post_transform_pixel_sha256"],
                "pre_transform_pixel_sha256": frame["pre_transform_pixel_sha256"],
                "transform_sha256": tuple(
                    transform["transform_sha256"] for transform in transforms
                ),
            }
        )
    return {
        "contract_schema": PIXEL_TRANSFORM_SCHEMA,
        "execution_method": PIXEL_TRANSFORM_EXECUTION_METHOD,
        "frame_results": tuple(frame_results),
        "frames_with_declared_transform": transformed_frames,
        "total_matched_pixel_count": total_matches,
        "transforms": tuple(unique_transforms[key] for key in sorted(unique_transforms)),
    }


def _normalize_bucket_sizes(values: Sequence[int | Size]) -> tuple[Size, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("bucket_sizes must be a sequence of integers or two-integer sizes")
    sizes: set[Size] = set()
    for value in values:
        if isinstance(value, int) and not isinstance(value, bool):
            size = value, value
        elif isinstance(value, tuple | list) and len(value) == 2:
            if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
                raise TypeError("bucket dimensions must be integers")
            size = int(value[0]), int(value[1])
        else:
            raise TypeError("each bucket must be an integer or a two-integer size")
        if size[0] < 1 or size[1] < 1:
            raise ValueError("bucket dimensions must be positive")
        sizes.add(size)
    if not sizes:
        raise ValueError("at least one bucket size is required")
    return tuple(sorted(sizes, key=lambda size: (size[0] * size[1], max(size), size)))


def _normalize_padding(
    value: int | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    result = (value, value, value, value) if isinstance(value, int) else value
    if len(result) != 4:
        raise ValueError("padding must be an integer or four-integer tuple")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in result):
        raise TypeError("padding must contain integers")
    if any(item < 0 for item in result):
        raise ValueError("padding must be non-negative")
    return tuple(result)


def _rgba_array(frames: Sequence[Image.Image]) -> np.ndarray:
    return np.stack(
        [np.asarray(frame.convert("RGBA"), dtype=np.uint8) for frame in frames],
        axis=0,
    )


def _rgba_pixel_sha256(image: Image.Image) -> str:
    rgba = image.convert("RGBA")
    header = f"RGBA\0{rgba.width}x{rgba.height}\0".encode()
    return hashlib.sha256(header + rgba.tobytes()).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _clip_relative_path(sequence_id: str, split: str) -> Path:
    digest = hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()
    return Path("clips") / split / f"{digest}.npy"


def _relative_posix(path: Path, root: Path) -> str:
    return PurePosixPath(path.relative_to(root)).as_posix()


def _atomic_write_npy(
    path: Path,
    array: np.ndarray,
    *,
    overwrite: bool,
    disk_guard: DiskGuard | None,
) -> tuple[str, int]:
    estimated_size = int(array.nbytes) + 65_536

    def writer(handle: Any) -> None:
        np.save(handle, array, allow_pickle=False)

    return _atomic_write(
        path,
        writer,
        overwrite=overwrite,
        disk_guard=disk_guard,
        estimated_size=estimated_size,
        label="materialized clip",
    )


def _atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool,
    disk_guard: DiskGuard | None,
    label: str,
) -> tuple[str, int]:
    return _atomic_write(
        path,
        lambda handle: handle.write(payload),
        overwrite=overwrite,
        disk_guard=disk_guard,
        estimated_size=len(payload),
        label=label,
    )


def _atomic_write(
    path: Path,
    writer: Callable[[Any], Any],
    *,
    overwrite: bool,
    disk_guard: DiskGuard | None,
    estimated_size: int,
    label: str,
) -> tuple[str, int]:
    if disk_guard is not None:
        disk_guard.require_capacity(estimated_size, label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _hash_path(temporary_path)
        size_bytes = temporary_path.stat().st_size
        if overwrite:
            os.replace(temporary_path, path)
        else:
            try:
                os.link(temporary_path, path)
            except FileExistsError as error:
                raise ExistingOutputError(f"Refusing to replace existing output: {path}") from error
            temporary_path.unlink()
        return digest, size_bytes
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise SnapshotValidationError(f"Value is not canonical JSON data: {error}") from error


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SnapshotValidationError(f"{key} must be a non-empty string")
    return value


def _required_integer(mapping: Mapping[str, Any], key: str, *, minimum: int) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SnapshotValidationError(f"{key} must be an integer >= {minimum}")
    return value


def _required_digest(mapping: Mapping[str, Any], key: str) -> str:
    value = _required_string(mapping, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SnapshotValidationError(f"{key} is not a lowercase SHA-256 digest: {value!r}")
    return value


def _digest_sequence(value: object, label: str) -> tuple[str, ...]:
    values = _string_sequence(value, label)
    if not values:
        raise SnapshotValidationError(f"{label} cannot be empty")
    for digest in values:
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise SnapshotValidationError(f"{label} contains invalid SHA-256 {digest!r}")
    if len(values) != len(set(values)):
        raise SnapshotValidationError(f"{label} contains duplicate SHA-256 values")
    return values


def _string_sequence(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise SnapshotValidationError(f"{label} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        raise SnapshotValidationError(f"{label} must contain non-empty strings")
    return tuple(value)


def _optional_finite_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
) -> float | int | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise FrameReconstructionError(f"{label} must be numeric or null")
    if not math.isfinite(value):
        raise FrameReconstructionError(f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise FrameReconstructionError(f"{label} must be >= {minimum}")
    return value
