"""Explicit pixel-art display decoders for continuous generated RGBA clips."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import __version__ as pillow_version

from spritelab.storage import DiskGuard


@dataclass(frozen=True, slots=True)
class HardAlphaDecodeConfig:
    threshold: int = 128

    def __post_init__(self) -> None:
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, int):
            raise TypeError("threshold must be an integer")
        if not 1 <= self.threshold <= 255:
            raise ValueError("threshold must be between 1 and 255")


@dataclass(frozen=True, slots=True)
class HardAlphaDecodeResult:
    array_path: Path
    metadata_path: Path
    source_file_sha256: str
    decoded_file_sha256: str
    decoded_array_sha256: str
    metadata_sha256: str


@dataclass(frozen=True, slots=True)
class GlobalPaletteDecodeConfig:
    """Hard-alpha plus one adaptive RGB palette shared by the whole clip."""

    alpha_threshold: int = 128
    maximum_colors: int = 32

    def __post_init__(self) -> None:
        HardAlphaDecodeConfig(threshold=self.alpha_threshold)
        if isinstance(self.maximum_colors, bool) or not isinstance(self.maximum_colors, int):
            raise TypeError("maximum_colors must be an integer")
        if not 2 <= self.maximum_colors <= 256:
            raise ValueError("maximum_colors must be between 2 and 256")


@dataclass(frozen=True, slots=True)
class GlobalPaletteDecodeResult:
    array_path: Path
    metadata_path: Path
    source_file_sha256: str
    decoded_file_sha256: str
    decoded_array_sha256: str
    metadata_sha256: str
    visible_colors_before: int
    visible_colors_after: int


def hard_alpha_decode_rgba(
    rgba: np.ndarray,
    *,
    config: HardAlphaDecodeConfig | None = None,
) -> np.ndarray:
    """Threshold alpha exactly and zero RGB below it without altering visible RGB."""

    settings = config or HardAlphaDecodeConfig()
    _validate_rgba(rgba)
    decoded = np.ascontiguousarray(rgba.copy())
    visible = decoded[..., 3] >= settings.threshold
    decoded[..., 3] = np.where(visible, 255, 0).astype(np.uint8)
    decoded[..., :3][~visible] = 0
    return decoded


def global_palette_decode_rgba(
    rgba: np.ndarray,
    *,
    config: GlobalPaletteDecodeConfig | None = None,
) -> np.ndarray:
    """Make alpha crisp and quantize visible RGB with one clip-global palette.

    The adaptive palette is fitted only to the generated clip. All frames share
    it, and dithering is disabled, so this derivative cannot introduce a
    frame-specific palette or dither noise. No target or reference colors are
    accepted by this API.
    """

    settings = config or GlobalPaletteDecodeConfig()
    decoded = hard_alpha_decode_rgba(
        rgba,
        config=HardAlphaDecodeConfig(threshold=settings.alpha_threshold),
    )
    visible = decoded[..., 3] > 0
    visible_rgb = decoded[..., :3][visible]
    if visible_rgb.size == 0:
        return decoded
    unique_colors = np.unique(visible_rgb, axis=0)
    if len(unique_colors) <= settings.maximum_colors:
        return decoded
    strip = Image.fromarray(visible_rgb.reshape(1, -1, 3))
    quantized = strip.quantize(
        colors=settings.maximum_colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    ).convert("RGB")
    decoded[..., :3][visible] = np.asarray(quantized, dtype=np.uint8).reshape(-1, 3)
    return np.ascontiguousarray(decoded)


def export_hard_alpha_decode(
    source_path: Path | str,
    output_path: Path | str,
    *,
    config: HardAlphaDecodeConfig | None = None,
    overwrite: bool = False,
    disk_guard: DiskGuard | None = None,
) -> HardAlphaDecodeResult:
    """Hash-verify one source array and publish a decoded array plus JSON sidecar."""

    settings = config or HardAlphaDecodeConfig()
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    metadata_path = output.with_suffix(output.suffix + ".decode.json")
    if output.suffix.casefold() != ".npy":
        raise ValueError("output_path must end in .npy")
    if not source.is_file():
        raise FileNotFoundError(f"source sample does not exist: {source}")
    if not overwrite:
        existing = next((path for path in (output, metadata_path) if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Refusing to replace existing decode artifact: {existing}")
    source_bytes = source.read_bytes()
    try:
        rgba = np.load(io.BytesIO(source_bytes), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"source sample is not a readable NumPy array: {source}") from error
    if not isinstance(rgba, np.ndarray):
        raise ValueError("source sample must contain one NumPy array")
    decoded = hard_alpha_decode_rgba(rgba, config=settings)
    array_buffer = io.BytesIO()
    np.save(array_buffer, decoded, allow_pickle=False)
    decoded_bytes = array_buffer.getvalue()
    source_file_sha256 = hashlib.sha256(source_bytes).hexdigest()
    decoded_file_sha256 = hashlib.sha256(decoded_bytes).hexdigest()
    decoded_array_sha256 = _array_sha256(decoded)
    metadata = {
        "artifact_kind": "derived_hard_alpha_pixel_decode",
        "decoded": {
            "array_sha256": decoded_array_sha256,
            "file_sha256": decoded_file_sha256,
            "path": str(output),
        },
        "operation": {
            "alpha_at_or_above_threshold": 255,
            "alpha_below_threshold": 0,
            "hidden_rgb": "zero",
            "visible_rgb": "unchanged",
        },
        "source": {
            "array_sha256": _array_sha256(rgba),
            "file_sha256": source_file_sha256,
            "path": str(source),
        },
        "threshold": settings.threshold,
    }
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(
        output,
        decoded_bytes,
        overwrite=overwrite,
        disk_guard=disk_guard,
        label="hard-alpha decoded sample",
    )
    _atomic_write(
        metadata_path,
        metadata_bytes,
        overwrite=overwrite,
        disk_guard=disk_guard,
        label="hard-alpha decode metadata",
    )
    return HardAlphaDecodeResult(
        array_path=output,
        metadata_path=metadata_path,
        source_file_sha256=source_file_sha256,
        decoded_file_sha256=decoded_file_sha256,
        decoded_array_sha256=decoded_array_sha256,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
    )


def export_global_palette_decode(
    source_path: Path | str,
    output_path: Path | str,
    *,
    config: GlobalPaletteDecodeConfig | None = None,
    overwrite: bool = False,
    disk_guard: DiskGuard | None = None,
) -> GlobalPaletteDecodeResult:
    """Publish a no-clobber clip-global palette derivative and JSON sidecar."""

    settings = config or GlobalPaletteDecodeConfig()
    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    metadata_path = output.with_suffix(output.suffix + ".decode.json")
    if output.suffix.casefold() != ".npy":
        raise ValueError("output_path must end in .npy")
    if not source.is_file():
        raise FileNotFoundError(f"source sample does not exist: {source}")
    if not overwrite:
        existing = next((path for path in (output, metadata_path) if path.exists()), None)
        if existing is not None:
            raise FileExistsError(f"Refusing to replace existing decode artifact: {existing}")
    source_bytes = source.read_bytes()
    try:
        rgba = np.load(io.BytesIO(source_bytes), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"source sample is not a readable NumPy array: {source}") from error
    if not isinstance(rgba, np.ndarray):
        raise ValueError("source sample must contain one NumPy array")
    _validate_rgba(rgba)
    hard_alpha = hard_alpha_decode_rgba(
        rgba,
        config=HardAlphaDecodeConfig(threshold=settings.alpha_threshold),
    )
    visible_colors_before = _visible_color_count(hard_alpha)
    decoded = global_palette_decode_rgba(rgba, config=settings)
    visible_colors_after = _visible_color_count(decoded)
    array_buffer = io.BytesIO()
    np.save(array_buffer, decoded, allow_pickle=False)
    decoded_bytes = array_buffer.getvalue()
    source_file_sha256 = hashlib.sha256(source_bytes).hexdigest()
    decoded_file_sha256 = hashlib.sha256(decoded_bytes).hexdigest()
    decoded_array_sha256 = _array_sha256(decoded)
    metadata = {
        "artifact_kind": "derived_clip_global_palette_pixel_decode",
        "decoded": {
            "array_sha256": decoded_array_sha256,
            "file_sha256": decoded_file_sha256,
            "path": str(output),
        },
        "operation": {
            "alpha_at_or_above_threshold": 255,
            "alpha_below_threshold": 0,
            "dithering": "none",
            "hidden_rgb": "zero",
            "palette_fit_scope": "all visible generated RGB pixels across the entire clip",
            "palette_method": "Pillow MEDIANCUT",
            "reference_or_target_palette_used": False,
        },
        "parameters": {
            "alpha_threshold": settings.alpha_threshold,
            "maximum_colors": settings.maximum_colors,
        },
        "runtime": {"pillow_version": pillow_version},
        "source": {
            "array_sha256": _array_sha256(rgba),
            "file_sha256": source_file_sha256,
            "path": str(source),
        },
        "visible_colors_after": visible_colors_after,
        "visible_colors_before": visible_colors_before,
    }
    metadata_bytes = (
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(
        output,
        decoded_bytes,
        overwrite=overwrite,
        disk_guard=disk_guard,
        label="clip-global palette decoded sample",
    )
    _atomic_write(
        metadata_path,
        metadata_bytes,
        overwrite=overwrite,
        disk_guard=disk_guard,
        label="clip-global palette decode metadata",
    )
    return GlobalPaletteDecodeResult(
        array_path=output,
        metadata_path=metadata_path,
        source_file_sha256=source_file_sha256,
        decoded_file_sha256=decoded_file_sha256,
        decoded_array_sha256=decoded_array_sha256,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        visible_colors_before=visible_colors_before,
        visible_colors_after=visible_colors_after,
    )


def _validate_rgba(rgba: np.ndarray) -> None:
    if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8:
        raise TypeError("rgba must be a uint8 NumPy array")
    if rgba.ndim != 4 or rgba.shape[-1] != 4 or min(rgba.shape) < 1:
        raise ValueError(f"rgba must have shape [T, H, W, 4]; got {rgba.shape!r}")


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _visible_color_count(rgba: np.ndarray) -> int:
    visible_rgb = rgba[..., :3][rgba[..., 3] > 0]
    return int(len(np.unique(visible_rgb, axis=0))) if visible_rgb.size else 0


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
                    f"Refusing to replace existing decode artifact: {path}"
                ) from error
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
