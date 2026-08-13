"""Display-preserving, provenance-linked previews for generated RGBA sprite clips."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.storage import DiskGuard

_SAFE_STEM = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_LOOP_MODES = frozenset({"loop", "one_shot", "ping_pong"})


@dataclass(frozen=True, slots=True)
class ClipPreviewResult:
    animated_png_path: Path
    contact_sheet_path: Path
    metadata_path: Path
    animated_png_sha256: str
    contact_sheet_sha256: str
    metadata_sha256: str


def export_npy_clip_preview(
    sample_path: Path | str,
    output_directory: Path | str,
    *,
    artifact_stem: str,
    duration_ms: tuple[float, ...],
    loop_mode: str,
    integer_scale: int = 4,
    source_report_sha256: str | None = None,
    preserve_frame_slots: bool = False,
    overwrite: bool = False,
    disk_guard: DiskGuard | None = None,
) -> ClipPreviewResult:
    """Render a uint8 RGBA sample as APNG plus a nearest-neighbor sheet."""

    source = Path(sample_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"sample array does not exist: {source}")
    source_bytes = source.read_bytes()
    try:
        rgba = np.load(io.BytesIO(source_bytes), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"sample is not a readable NumPy array: {source}") from error
    if not isinstance(rgba, np.ndarray):
        raise ValueError("sample must contain one NumPy array")
    return export_rgba_clip_preview(
        rgba,
        output_directory,
        artifact_stem=artifact_stem,
        duration_ms=duration_ms,
        loop_mode=loop_mode,
        integer_scale=integer_scale,
        source_sample_path=source,
        source_sample_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_report_sha256=source_report_sha256,
        preserve_frame_slots=preserve_frame_slots,
        overwrite=overwrite,
        disk_guard=disk_guard,
    )


def export_rgba_clip_preview(
    rgba: np.ndarray,
    output_directory: Path | str,
    *,
    artifact_stem: str,
    duration_ms: tuple[float, ...],
    loop_mode: str,
    integer_scale: int = 4,
    source_sample_path: Path | None = None,
    source_sample_sha256: str | None = None,
    source_report_sha256: str | None = None,
    preserve_frame_slots: bool = False,
    overwrite: bool = False,
    disk_guard: DiskGuard | None = None,
) -> ClipPreviewResult:
    """Create display-only files with zeroed invisible RGB and no interpolation."""

    _validate_inputs(
        rgba,
        artifact_stem=artifact_stem,
        duration_ms=duration_ms,
        loop_mode=loop_mode,
        integer_scale=integer_scale,
        preserve_frame_slots=preserve_frame_slots,
    )
    if source_sample_sha256 is not None:
        _validate_digest(source_sample_sha256, "source_sample_sha256")
    if source_report_sha256 is not None:
        _validate_digest(source_report_sha256, "source_report_sha256")
    output = Path(output_directory).resolve()
    animated_path = output / f"{artifact_stem}-animated.png"
    sheet_path = output / f"{artifact_stem}-sheet.png"
    metadata_path = output / f"{artifact_stem}-preview.json"
    targets = (animated_path, sheet_path, metadata_path)
    if not overwrite:
        existing = next((path for path in targets if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Refusing to replace existing preview artifact: {existing}")

    display_rgba = rgba.copy()
    display_rgba[..., :3][display_rgba[..., 3] == 0] = 0
    scaled_frames = tuple(
        Image.fromarray(frame).resize(
            (frame.shape[1] * integer_scale, frame.shape[0] * integer_scale),
            resample=Image.Resampling.NEAREST,
        )
        for frame in display_rgba
    )
    integer_durations = tuple(max(1, round(value)) for value in duration_ms)
    animated_buffer = io.BytesIO()
    scaled_frames[0].save(
        animated_buffer,
        format="PNG",
        save_all=True,
        append_images=list(scaled_frames[1:]),
        duration=integer_durations,
        loop=0 if loop_mode in {"loop", "ping_pong"} else 1,
        disposal=(
            tuple(index % 2 for index in range(len(scaled_frames))) if preserve_frame_slots else 0
        ),
        blend=0,
        optimize=False,
    )
    animated_bytes = animated_buffer.getvalue()

    width, height = scaled_frames[0].size
    sheet = Image.new("RGBA", (width * len(scaled_frames), height), (0, 0, 0, 0))
    for index, frame in enumerate(scaled_frames):
        sheet.paste(frame, (index * width, 0))
    sheet_buffer = io.BytesIO()
    sheet.save(sheet_buffer, format="PNG", optimize=False)
    sheet_bytes = sheet_buffer.getvalue()

    animated_sha256 = hashlib.sha256(animated_bytes).hexdigest()
    sheet_sha256 = hashlib.sha256(sheet_bytes).hexdigest()
    metadata = {
        "animated_png": {
            "file_sha256": animated_sha256,
            "path": animated_path.name,
        },
        "array_content_sha256": _array_sha256(rgba),
        "artifact_kind": "display_preserving_nearest_neighbor_sprite_clip_preview",
        "contact_sheet": {
            "file_sha256": sheet_sha256,
            "path": sheet_path.name,
        },
        "duration_ms": list(duration_ms),
        "frame_count": int(rgba.shape[0]),
        "integer_scale": integer_scale,
        "invisible_rgb_policy": "zero_where_alpha_is_zero",
        "loop_mode": loop_mode,
        "native_size": [int(rgba.shape[2]), int(rgba.shape[1])],
        "preview_size": [width, height],
        "preserve_frame_slots": preserve_frame_slots,
        "resampling": "nearest_positive_integer",
        "source_report_sha256": source_report_sha256,
        "source_sample_path": str(source_sample_path) if source_sample_path else None,
        "source_sample_sha256": source_sample_sha256,
    }
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(
        animated_path,
        animated_bytes,
        overwrite=overwrite,
        disk_guard=disk_guard,
        label="animated sprite preview",
    )
    _atomic_write(
        sheet_path,
        sheet_bytes,
        overwrite=overwrite,
        disk_guard=disk_guard,
        label="sprite contact sheet",
    )
    _atomic_write(
        metadata_path,
        metadata_bytes,
        overwrite=overwrite,
        disk_guard=disk_guard,
        label="sprite preview metadata",
    )
    return ClipPreviewResult(
        animated_png_path=animated_path,
        contact_sheet_path=sheet_path,
        metadata_path=metadata_path,
        animated_png_sha256=animated_sha256,
        contact_sheet_sha256=sheet_sha256,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
    )


def _validate_inputs(
    rgba: np.ndarray,
    *,
    artifact_stem: str,
    duration_ms: tuple[float, ...],
    loop_mode: str,
    integer_scale: int,
    preserve_frame_slots: bool,
) -> None:
    if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8:
        raise TypeError("rgba must be a uint8 NumPy array")
    if rgba.ndim != 4 or rgba.shape[-1] != 4 or min(rgba.shape) < 1:
        raise ValueError(f"rgba must have shape [T, H, W, 4]; got {rgba.shape!r}")
    if not isinstance(artifact_stem, str) or _SAFE_STEM.fullmatch(artifact_stem) is None:
        raise ValueError("artifact_stem must be a safe 1-128 character filename stem")
    if loop_mode not in _LOOP_MODES:
        raise ValueError(f"unsupported loop_mode: {loop_mode!r}")
    if isinstance(integer_scale, bool) or not isinstance(integer_scale, int) or integer_scale < 1:
        raise ValueError("integer_scale must be a positive integer")
    if not isinstance(preserve_frame_slots, bool):
        raise TypeError("preserve_frame_slots must be a boolean")
    if len(duration_ms) != rgba.shape[0]:
        raise ValueError("duration_ms length must match the frame count")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
        for value in duration_ms
    ):
        raise ValueError("duration_ms values must be finite and positive")


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _array_sha256(array: np.ndarray) -> str:
    header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool,
    disk_guard: DiskGuard | None,
    label: str,
) -> None:
    if disk_guard is not None:
        disk_guard.require_capacity(len(payload) + 65_536, label=label)
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
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Refusing to replace existing preview artifact: {path}"
                ) from error
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
