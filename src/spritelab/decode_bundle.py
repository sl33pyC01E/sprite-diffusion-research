"""Atomic, provenance-bound preview bundles for derived pixel decodes.

The raw generated ``.npy`` clips accepted here remain canonical and are never
modified.  A bundle contains explicitly labelled hard-alpha and clip-global
palette derivatives plus display previews.  All payloads are built in a sibling
staging directory and become visible only when that directory is promoted to a
previously absent destination.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.decode import (
    GlobalPaletteDecodeConfig,
    HardAlphaDecodeConfig,
    global_palette_decode_rgba,
    hard_alpha_decode_rgba,
)
from spritelab.previews import export_rgba_clip_preview
from spritelab.storage import DiskGuard, HashMismatch

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_LOOP_MODES = frozenset({"loop", "one_shot", "ping_pong"})
_HARD_ALPHA_CALIBRATION_KIND = "hard_alpha_threshold_calibration"
_PALETTE_CALIBRATION_KIND = "clip_global_palette_size_calibration"


@dataclass(frozen=True, slots=True)
class DecodeBundleClipRef:
    """One ordered, hash-pinned canonical raw clip and its display timing."""

    sample_id: str
    source_path: Path | str
    source_file_sha256: str
    duration_ms: tuple[float, ...]
    loop_mode: str


@dataclass(frozen=True, slots=True)
class DecodeBundleArtifactRef:
    """One external provenance artifact bound by its exact file digest."""

    artifact_id: str
    path: Path | str
    file_sha256: str


@dataclass(frozen=True, slots=True)
class DecodePreviewBundleResult:
    bundle_path: Path
    index_path: Path
    index_sha256: str
    clip_count: int
    palette_sizes: tuple[int, ...]
    payload_file_count: int


@dataclass(frozen=True, slots=True)
class _LoadedClip:
    sample_id: str
    source_path: Path
    source_file_sha256: str
    source_array_sha256: str
    size_bytes: int
    rgba: np.ndarray
    duration_ms: tuple[float, ...]
    loop_mode: str


@dataclass(frozen=True, slots=True)
class _LoadedArtifact:
    artifact_id: str
    path: Path
    file_sha256: str
    size_bytes: int
    document: dict[str, Any] | None


def export_decode_preview_bundle(
    clips: Sequence[DecodeBundleClipRef],
    output_directory: Path | str,
    *,
    hard_alpha_threshold: int,
    palette_sizes: Sequence[int],
    source_report: DecodeBundleArtifactRef,
    hard_alpha_calibrations: Sequence[DecodeBundleArtifactRef],
    palette_calibrations: Sequence[DecodeBundleArtifactRef],
    integer_scale: int = 4,
    disk_guard: DiskGuard | None = None,
) -> DecodePreviewBundleResult:
    """Publish an immutable derived-decode preview bundle.

    Inputs are validated and hash-pinned before staging begins.  Palette fitting
    uses only each generated source clip through ``global_palette_decode_rgba``;
    this API has no target-array or reference-palette parameter.  The returned
    checksum identifies the canonical index; the index in turn binds every other
    payload file in the bundle.
    """

    settings = HardAlphaDecodeConfig(threshold=hard_alpha_threshold)
    normalized_palette_sizes = _validate_palette_sizes(
        palette_sizes,
        alpha_threshold=settings.threshold,
    )
    if isinstance(integer_scale, bool) or not isinstance(integer_scale, int):
        raise TypeError("integer_scale must be an integer")
    if integer_scale < 1:
        raise ValueError("integer_scale must be positive")

    output = Path(output_directory).resolve()
    if not output.name:
        raise ValueError("output_directory must name a bundle directory")
    if os.path.lexists(output):
        raise FileExistsError(f"Refusing to replace existing decode preview bundle: {output}")

    loaded_clips = _load_clips(clips)
    loaded_source_report = _load_artifact(source_report, role="source report")
    loaded_hard_calibrations = _load_artifacts(
        hard_alpha_calibrations,
        role="hard-alpha calibration",
        required_kind=_HARD_ALPHA_CALIBRATION_KIND,
    )
    loaded_palette_calibrations = _load_artifacts(
        palette_calibrations,
        role="palette calibration",
        required_kind=_PALETTE_CALIBRATION_KIND,
    )
    _validate_provenance_uniqueness(
        loaded_source_report,
        loaded_hard_calibrations,
        loaded_palette_calibrations,
    )
    _validate_calibration_coverage(
        hard_alpha_threshold=settings.threshold,
        palette_sizes=normalized_palette_sizes,
        hard_alpha_calibrations=loaded_hard_calibrations,
        palette_calibrations=loaded_palette_calibrations,
    )

    if disk_guard is not None:
        disk_guard.require_capacity(label="decode preview bundle staging")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".partial",
            dir=output.parent,
        )
    )
    published = False
    try:
        file_roles: dict[Path, str] = {}
        member_records: list[dict[str, object]] = []
        source_report_record = _provenance_record(loaded_source_report)
        hard_calibration_records = [
            _provenance_record(artifact) for artifact in loaded_hard_calibrations
        ]
        palette_calibration_records = [
            _provenance_record(artifact) for artifact in loaded_palette_calibrations
        ]

        for clip in loaded_clips:
            hard_alpha = hard_alpha_decode_rgba(
                clip.rgba,
                config=settings,
            )
            hard_member = _stage_variant(
                staging=staging,
                output=output,
                clip=clip,
                variant_id="hard-alpha",
                variant_directory="hard-alpha",
                decoded=hard_alpha,
                artifact_kind="derived_hard_alpha_pixel_decode_bundle_member",
                operation={
                    "alpha_at_or_above_threshold": 255,
                    "alpha_below_threshold": 0,
                    "hidden_rgb": "zero",
                    "visible_rgb": "unchanged",
                },
                parameters={"alpha_threshold": settings.threshold},
                calibration_records=hard_calibration_records,
                source_report_record=source_report_record,
                integer_scale=integer_scale,
                disk_guard=disk_guard,
                file_roles=file_roles,
            )

            palette_members: list[dict[str, object]] = []
            for maximum_colors in normalized_palette_sizes:
                palette_config = GlobalPaletteDecodeConfig(
                    alpha_threshold=settings.threshold,
                    maximum_colors=maximum_colors,
                )
                palette = global_palette_decode_rgba(
                    clip.rgba,
                    config=palette_config,
                )
                palette_members.append(
                    _stage_variant(
                        staging=staging,
                        output=output,
                        clip=clip,
                        variant_id=f"clip-global-palette-{maximum_colors}",
                        variant_directory=f"palette-{maximum_colors}",
                        decoded=palette,
                        artifact_kind=("derived_clip_global_palette_pixel_decode_bundle_member"),
                        operation={
                            "alpha_at_or_above_threshold": 255,
                            "alpha_below_threshold": 0,
                            "dithering": "none",
                            "hidden_rgb": "zero",
                            "palette_fit_scope": (
                                "all visible generated RGB pixels across the entire clip"
                            ),
                            "palette_method": "Pillow MEDIANCUT",
                            "reference_or_target_palette_used": False,
                        },
                        parameters={
                            "alpha_threshold": settings.threshold,
                            "maximum_colors": maximum_colors,
                        },
                        calibration_records=[
                            *hard_calibration_records,
                            *palette_calibration_records,
                        ],
                        source_report_record=source_report_record,
                        integer_scale=integer_scale,
                        disk_guard=disk_guard,
                        file_roles=file_roles,
                    )
                )
            member_records.append(
                {
                    "hard_alpha": hard_member,
                    "palette_variants": palette_members,
                    "sample_id": clip.sample_id,
                }
            )

        payload_files = _indexed_payload_files(staging, file_roles)
        _reverify_external_inputs(
            loaded_clips,
            (loaded_source_report, *loaded_hard_calibrations, *loaded_palette_calibrations),
        )
        index = {
            "artifact_kind": "derived_pixel_decode_preview_bundle_index",
            "bundle": {
                "atomic_publication": "sibling staging directory promoted by no-clobber rename",
                "index_self_inclusion": False,
                "index_self_inclusion_reason": (
                    "The canonical index checksum is returned externally; a file cannot "
                    "contain its own SHA-256 digest."
                ),
                "path": str(output),
            },
            "claim_scope": (
                "Display-only derivatives of hash-pinned raw generated clips. Raw source "
                "arrays remain canonical and authoritative for model evaluation."
            ),
            "derivative_policy": {
                "canonical_raw_outputs_mutated": False,
                "evaluation_authority": "hash-pinned raw source arrays",
                "palette_fit_input": "generated clip only",
                "reference_or_target_palette_used": False,
                "spatial_resampling_in_decode": False,
            },
            "inputs": {
                "clip_count": len(loaded_clips),
                "clips": [_clip_record(clip) for clip in loaded_clips],
                "ordering": "caller-supplied order preserved",
            },
            "members": member_records,
            "parameters": {
                "hard_alpha_threshold": settings.threshold,
                "integer_scale": integer_scale,
                "palette_sizes": list(normalized_palette_sizes),
            },
            "payload_files": payload_files,
            "payload_summary": {
                "file_count": len(payload_files),
                "total_size_bytes": sum(int(record["size_bytes"]) for record in payload_files),
            },
            "provenance": {
                "calibration_relationship": (
                    "Hash-linked calibration evidence; bundle settings are explicit and "
                    "need not equal every linked artifact's selected candidate."
                ),
                "hard_alpha_calibrations": hard_calibration_records,
                "palette_calibrations": palette_calibration_records,
                "source_report": source_report_record,
            },
            "schema_version": 1,
        }
        index_bytes = _canonical_json_bytes(index)
        staged_index = staging / "bundle-index.json"
        _atomic_write_no_clobber(
            staged_index,
            index_bytes,
            disk_guard=disk_guard,
            label="decode preview bundle checksum index",
        )

        expected_files = {*file_roles, staged_index.resolve()}
        actual_files = {path.resolve() for path in staging.rglob("*") if path.is_file()}
        if actual_files != expected_files:
            missing = sorted(str(path) for path in expected_files - actual_files)
            unexpected = sorted(str(path) for path in actual_files - expected_files)
            raise RuntimeError(
                f"staged bundle file inventory mismatch; missing={missing!r}, "
                f"unexpected={unexpected!r}"
            )

        if os.path.lexists(output):
            raise FileExistsError(f"Refusing to replace existing decode preview bundle: {output}")
        try:
            os.rename(staging, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to replace existing decode preview bundle: {output}"
            ) from error
        published = True
        index_path = output / "bundle-index.json"
        index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        if _sha256_file(index_path) != index_sha256:
            raise RuntimeError("published decode preview bundle index failed SHA-256 verification")
        return DecodePreviewBundleResult(
            bundle_path=output,
            index_path=index_path,
            index_sha256=index_sha256,
            clip_count=len(loaded_clips),
            palette_sizes=normalized_palette_sizes,
            payload_file_count=len(payload_files),
        )
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _load_clips(clips: Sequence[DecodeBundleClipRef]) -> tuple[_LoadedClip, ...]:
    _require_sequence(clips, name="clips")
    if not clips:
        raise ValueError("at least one clip is required")
    loaded: list[_LoadedClip] = []
    identifiers: set[str] = set()
    paths: set[Path] = set()
    for reference in clips:
        if not isinstance(reference, DecodeBundleClipRef):
            raise TypeError("clips must contain only DecodeBundleClipRef values")
        _validate_id(reference.sample_id, "sample_id")
        if reference.sample_id in identifiers:
            raise ValueError(f"duplicate sample_id: {reference.sample_id!r}")
        identifiers.add(reference.sample_id)
        source = Path(reference.source_path).resolve()
        if source.suffix.casefold() != ".npy":
            raise ValueError(f"source clip must end in .npy: {source}")
        if not source.is_file():
            raise FileNotFoundError(f"source clip does not exist: {source}")
        if source in paths:
            raise ValueError(f"source clip paths must be unique: {source}")
        paths.add(source)
        _validate_digest(reference.source_file_sha256, "source_file_sha256")
        source_bytes = source.read_bytes()
        actual_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if actual_sha256 != reference.source_file_sha256:
            raise HashMismatch(
                f"source clip {source} expected SHA-256 {reference.source_file_sha256}, "
                f"received {actual_sha256}"
            )
        try:
            array = np.load(io.BytesIO(source_bytes), allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"source clip is not a readable NumPy array: {source}") from error
        if not isinstance(array, np.ndarray):
            if hasattr(array, "close"):
                array.close()
            raise ValueError(f"source clip must contain exactly one NumPy array: {source}")
        _validate_rgba(array, sample_id=reference.sample_id)
        durations = _validate_durations(reference.duration_ms, frame_count=int(array.shape[0]))
        if reference.loop_mode not in _LOOP_MODES:
            raise ValueError(f"unsupported loop_mode: {reference.loop_mode!r}")
        rgba = np.ascontiguousarray(array)
        loaded.append(
            _LoadedClip(
                sample_id=reference.sample_id,
                source_path=source,
                source_file_sha256=actual_sha256,
                source_array_sha256=_array_sha256(rgba),
                size_bytes=len(source_bytes),
                rgba=rgba,
                duration_ms=durations,
                loop_mode=reference.loop_mode,
            )
        )
    return tuple(loaded)


def _load_artifacts(
    references: Sequence[DecodeBundleArtifactRef],
    *,
    role: str,
    required_kind: str,
) -> tuple[_LoadedArtifact, ...]:
    _require_sequence(references, name=f"{role}s")
    if not references:
        raise ValueError(f"at least one {role} is required")
    loaded = tuple(_load_artifact(reference, role=role) for reference in references)
    identifiers = [artifact.artifact_id for artifact in loaded]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{role} artifact IDs must be unique")
    paths = [artifact.path for artifact in loaded]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{role} paths must be unique")
    for artifact in loaded:
        kind = artifact.document.get("artifact_kind") if artifact.document is not None else None
        if kind != required_kind:
            raise ValueError(
                f"{role} {artifact.path} must have artifact_kind {required_kind!r}; got {kind!r}"
            )
    return loaded


def _load_artifact(reference: DecodeBundleArtifactRef, *, role: str) -> _LoadedArtifact:
    if not isinstance(reference, DecodeBundleArtifactRef):
        raise TypeError(f"{role} must be a DecodeBundleArtifactRef")
    _validate_id(reference.artifact_id, "artifact_id")
    _validate_digest(reference.file_sha256, f"{role} file_sha256")
    path = Path(reference.path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != reference.file_sha256:
        raise HashMismatch(
            f"{role} {path} expected SHA-256 {reference.file_sha256}, received {actual_sha256}"
        )
    document: dict[str, Any] | None = None
    if path.suffix.casefold() == ".json":
        try:
            parsed = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{role} is not valid UTF-8 JSON: {path}") from error
        if not isinstance(parsed, dict):
            raise ValueError(f"{role} JSON must contain an object: {path}")
        document = parsed
    return _LoadedArtifact(
        artifact_id=reference.artifact_id,
        path=path,
        file_sha256=actual_sha256,
        size_bytes=len(payload),
        document=document,
    )


def _validate_provenance_uniqueness(
    source_report: _LoadedArtifact,
    hard_alpha_calibrations: Sequence[_LoadedArtifact],
    palette_calibrations: Sequence[_LoadedArtifact],
) -> None:
    artifacts = (source_report, *hard_alpha_calibrations, *palette_calibrations)
    identifiers = [artifact.artifact_id for artifact in artifacts]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("all source report and calibration artifact IDs must be unique")
    paths = [artifact.path for artifact in artifacts]
    if len(set(paths)) != len(paths):
        raise ValueError("all source report and calibration paths must be unique")


def _validate_calibration_coverage(
    *,
    hard_alpha_threshold: int,
    palette_sizes: tuple[int, ...],
    hard_alpha_calibrations: Sequence[_LoadedArtifact],
    palette_calibrations: Sequence[_LoadedArtifact],
) -> None:
    calibrated_thresholds: set[int] = set()
    for artifact in hard_alpha_calibrations:
        assert artifact.document is not None
        values = artifact.document.get("thresholds")
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"hard-alpha calibration has invalid thresholds: {artifact.path}")
        for value in values:
            HardAlphaDecodeConfig(threshold=value)
        calibrated_thresholds.update(values)
    if hard_alpha_threshold not in calibrated_thresholds:
        raise ValueError(
            f"hard_alpha_threshold {hard_alpha_threshold} is absent from linked calibrations"
        )

    calibrated_palette_sizes: set[int] = set()
    for artifact in palette_calibrations:
        assert artifact.document is not None
        parameters = artifact.document.get("parameters")
        calibration_threshold = (
            parameters.get("alpha_threshold") if isinstance(parameters, dict) else None
        )
        if (
            isinstance(calibration_threshold, bool)
            or not isinstance(calibration_threshold, int)
            or calibration_threshold != hard_alpha_threshold
        ):
            raise ValueError(
                f"palette calibration alpha threshold does not match "
                f"{hard_alpha_threshold}: {artifact.path}"
            )
        values = artifact.document.get("palette_sizes")
        if not isinstance(values, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in values
        ):
            raise ValueError(f"palette calibration has invalid palette_sizes: {artifact.path}")
        for value in values:
            GlobalPaletteDecodeConfig(
                alpha_threshold=hard_alpha_threshold,
                maximum_colors=value,
            )
        calibrated_palette_sizes.update(values)
    missing = sorted(set(palette_sizes) - calibrated_palette_sizes)
    if missing:
        raise ValueError(f"palette sizes absent from linked calibrations: {missing!r}")


def _validate_palette_sizes(
    palette_sizes: Sequence[int],
    *,
    alpha_threshold: int,
) -> tuple[int, ...]:
    _require_sequence(palette_sizes, name="palette_sizes")
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


def _stage_variant(
    *,
    staging: Path,
    output: Path,
    clip: _LoadedClip,
    variant_id: str,
    variant_directory: str,
    decoded: np.ndarray,
    artifact_kind: str,
    operation: dict[str, object],
    parameters: dict[str, object],
    calibration_records: list[dict[str, object]],
    source_report_record: dict[str, object],
    integer_scale: int,
    disk_guard: DiskGuard | None,
    file_roles: dict[Path, str],
) -> dict[str, object]:
    staged_directory = staging / variant_directory
    final_directory = output / variant_directory
    staged_array = staged_directory / f"{clip.sample_id}.npy"
    final_array = final_directory / f"{clip.sample_id}.npy"
    staged_metadata = staged_array.with_suffix(staged_array.suffix + ".decode.json")
    final_metadata = final_array.with_suffix(final_array.suffix + ".decode.json")

    array_bytes = _npy_bytes(decoded)
    array_file_sha256 = hashlib.sha256(array_bytes).hexdigest()
    array_content_sha256 = _array_sha256(decoded)
    _atomic_write_no_clobber(
        staged_array,
        array_bytes,
        disk_guard=disk_guard,
        label=f"{variant_id} decoded array",
    )
    _register_file(file_roles, staged_array, f"{variant_id}_array")

    metadata = {
        "artifact_kind": artifact_kind,
        "bundle_member": {
            "bundle_relative_path": final_array.relative_to(output).as_posix(),
            "sample_id": clip.sample_id,
            "variant_id": variant_id,
        },
        "decoded": {
            "array_sha256": array_content_sha256,
            "dtype": decoded.dtype.name,
            "file_sha256": array_file_sha256,
            "path": str(final_array),
            "shape": list(decoded.shape),
        },
        "derivative_status": {
            "canonical_model_output": False,
            "display_only": True,
            "evaluation_authority": "raw source array",
            "raw_source_mutated": False,
        },
        "operation": operation,
        "parameters": parameters,
        "provenance": {
            "calibrations": calibration_records,
            "source_report": source_report_record,
        },
        "schema_version": 1,
        "source": {
            "array_sha256": clip.source_array_sha256,
            "dtype": clip.rgba.dtype.name,
            "file_sha256": clip.source_file_sha256,
            "path": str(clip.source_path),
            "shape": list(clip.rgba.shape),
            "size_bytes": clip.size_bytes,
        },
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    _atomic_write_no_clobber(
        staged_metadata,
        metadata_bytes,
        disk_guard=disk_guard,
        label=f"{variant_id} decode metadata",
    )
    _register_file(file_roles, staged_metadata, f"{variant_id}_decode_metadata")

    preview = export_rgba_clip_preview(
        decoded,
        staged_directory,
        artifact_stem=clip.sample_id,
        duration_ms=clip.duration_ms,
        loop_mode=clip.loop_mode,
        integer_scale=integer_scale,
        source_sample_path=final_array,
        source_sample_sha256=array_file_sha256,
        source_report_sha256=str(source_report_record["file_sha256"]),
        overwrite=False,
        disk_guard=disk_guard,
    )
    _register_file(file_roles, preview.animated_png_path, f"{variant_id}_animated_png")
    _register_file(file_roles, preview.contact_sheet_path, f"{variant_id}_contact_sheet")
    _register_file(file_roles, preview.metadata_path, f"{variant_id}_preview_metadata")
    return {
        "files": {
            "animated_png": preview.animated_png_path.relative_to(staging).as_posix(),
            "contact_sheet": preview.contact_sheet_path.relative_to(staging).as_posix(),
            "decode_metadata": final_metadata.relative_to(output).as_posix(),
            "decoded_array": final_array.relative_to(output).as_posix(),
            "preview_metadata": preview.metadata_path.relative_to(staging).as_posix(),
        },
        "parameters": parameters,
        "variant_id": variant_id,
    }


def _indexed_payload_files(
    staging: Path,
    file_roles: dict[Path, str],
) -> list[dict[str, object]]:
    actual_files = {path.resolve() for path in staging.rglob("*") if path.is_file()}
    registered_files = set(file_roles)
    if actual_files != registered_files:
        missing = sorted(str(path) for path in registered_files - actual_files)
        unexpected = sorted(str(path) for path in actual_files - registered_files)
        raise RuntimeError(
            f"staged payload file inventory mismatch; missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )
    records: list[dict[str, object]] = []
    for path in sorted(actual_files, key=lambda value: value.relative_to(staging).as_posix()):
        relative_path = path.relative_to(staging).as_posix()
        records.append(
            {
                "file_sha256": _sha256_file(path),
                "media_type": _media_type(path),
                "path": relative_path,
                "role": file_roles[path],
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _clip_record(clip: _LoadedClip) -> dict[str, object]:
    return {
        "array_sha256": clip.source_array_sha256,
        "dtype": clip.rgba.dtype.name,
        "duration_ms": list(clip.duration_ms),
        "file_sha256": clip.source_file_sha256,
        "loop_mode": clip.loop_mode,
        "path": str(clip.source_path),
        "sample_id": clip.sample_id,
        "shape": list(clip.rgba.shape),
        "size_bytes": clip.size_bytes,
    }


def _provenance_record(artifact: _LoadedArtifact) -> dict[str, object]:
    record: dict[str, object] = {
        "artifact_id": artifact.artifact_id,
        "file_sha256": artifact.file_sha256,
        "path": str(artifact.path),
        "size_bytes": artifact.size_bytes,
    }
    if artifact.document is not None:
        record["artifact_kind"] = artifact.document.get("artifact_kind")
        estimate = artifact.document.get("estimate")
        if isinstance(estimate, dict):
            record["estimate_kind"] = estimate.get("kind")
            record["held_out"] = estimate.get("held_out")
    return record


def _reverify_external_inputs(
    clips: Sequence[_LoadedClip],
    artifacts: Sequence[_LoadedArtifact],
) -> None:
    for clip in clips:
        actual = _sha256_file(clip.source_path)
        if actual != clip.source_file_sha256:
            raise HashMismatch(
                f"source clip changed during bundle construction: {clip.source_path}"
            )
    for artifact in artifacts:
        actual = _sha256_file(artifact.path)
        if actual != artifact.file_sha256:
            raise HashMismatch(
                f"provenance artifact changed during bundle construction: {artifact.path}"
            )


def _validate_rgba(rgba: np.ndarray, *, sample_id: str) -> None:
    if rgba.dtype != np.uint8:
        raise TypeError(f"source clip {sample_id!r} must have dtype uint8")
    if rgba.ndim != 4 or rgba.shape[-1] != 4 or min(rgba.shape) < 1:
        raise ValueError(
            f"source clip {sample_id!r} must have shape [T, H, W, 4]; got {rgba.shape!r}"
        )


def _validate_durations(values: tuple[float, ...], *, frame_count: int) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError("duration_ms must be an explicit tuple")
    if len(values) != frame_count:
        raise ValueError("duration_ms length must match the source clip frame count")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError("duration_ms values must be finite and positive")
    return tuple(float(value) for value in values)


def _validate_id(value: str, name: str) -> None:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe 1-128 character identifier")


def _validate_digest(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_sequence(value: object, *, name: str) -> None:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be an explicit sequence")


def _register_file(file_roles: dict[Path, str], path: Path, role: str) -> None:
    resolved = path.resolve()
    if resolved in file_roles:
        raise RuntimeError(f"duplicate staged bundle path: {resolved}")
    file_roles[resolved] = role


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(document: dict[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _media_type(path: Path) -> str:
    if path.suffix.casefold() == ".npy":
        return "application/x-npy"
    if path.suffix.casefold() == ".json":
        return "application/json"
    if path.suffix.casefold() == ".png":
        return "image/png"
    raise ValueError(f"unsupported bundle payload extension: {path}")


def _atomic_write_no_clobber(
    path: Path,
    payload: bytes,
    *,
    disk_guard: DiskGuard | None,
    label: str,
) -> None:
    if disk_guard is not None:
        disk_guard.require_capacity(len(payload) + 65_536, label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
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
            raise FileExistsError(f"Refusing to replace staged bundle artifact: {path}") from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
