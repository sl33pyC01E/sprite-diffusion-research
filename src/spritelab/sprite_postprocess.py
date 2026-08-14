"""Generated-only display decoding for gray-background RGB sprite controls."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.storage import DiskGuard


@dataclass(frozen=True, slots=True)
class SpriteDisplayDecodeConfig:
    """Hard-alpha and palette contract for a generated RGB control."""

    background_rgb_distance: float = 28.0
    minimum_component_pixels: int = 10
    palette_colors: int = 32

    def __post_init__(self) -> None:
        if (
            not isinstance(self.background_rgb_distance, (int, float))
            or isinstance(self.background_rgb_distance, bool)
            or not np.isfinite(self.background_rgb_distance)
            or self.background_rgb_distance < 0
        ):
            raise ValueError("background_rgb_distance must be finite and non-negative")
        if (
            isinstance(self.minimum_component_pixels, bool)
            or not isinstance(self.minimum_component_pixels, int)
            or self.minimum_component_pixels < 1
        ):
            raise ValueError("minimum_component_pixels must be positive")
        if (
            isinstance(self.palette_colors, bool)
            or not isinstance(self.palette_colors, int)
            or not 2 <= self.palette_colors <= 256
        ):
            raise ValueError("palette_colors must be in [2,256]")


def decode_generated_rgb_sprite(
    rgb: np.ndarray, *, config: SpriteDisplayDecodeConfig | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer border-connected background, hard alpha, and a generated-only palette."""

    operation = config or SpriteDisplayDecodeConfig()
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
    ):
        raise ValueError("rgb must be a uint8 [H,W,3] array")
    height, width = rgb.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("rgb geometry must be at least 2x2")
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background_rgb = np.median(border.astype(np.float64), axis=0)
    distance = np.sqrt(((rgb.astype(np.float64) - background_rgb) ** 2).sum(axis=2))
    candidate_background = distance <= operation.background_rgb_distance
    connected_background = _border_connected(candidate_background)
    foreground = _remove_small_components(
        ~connected_background, minimum_pixels=operation.minimum_component_pixels
    )
    alpha = foreground.astype(np.uint8) * 255
    raw_rgba = np.dstack((rgb, alpha))
    quantized = np.array(
        Image.fromarray(raw_rgba, mode="RGBA")
        .quantize(
            colors=operation.palette_colors,
            method=Image.Quantize.FASTOCTREE,
            dither=Image.Dither.NONE,
        )
        .convert("RGBA"),
        dtype=np.uint8,
        copy=True,
    )
    quantized[..., 3] = alpha
    quantized[~foreground, :3] = 0
    output = np.ascontiguousarray(quantized)
    return output, {
        "background_border_median_rgb": [float(value) for value in background_rgb],
        "background_pixel_count": int(connected_background.sum()),
        "foreground_pixel_count": int(foreground.sum()),
        "hard_alpha": True,
        "operation": asdict(operation),
        "palette_fit_scope": "generated_rgb_with_inferred_alpha_only",
        "reference_or_target_pixels_used": False,
    }


def composite_rgba_on_checkerboard(rgba: np.ndarray, *, tile_pixels: int = 8) -> np.ndarray:
    """Render an RGBA sprite on a deterministic neutral checkerboard."""

    if (
        not isinstance(rgba, np.ndarray)
        or rgba.dtype != np.uint8
        or rgba.ndim != 3
        or rgba.shape[2] != 4
    ):
        raise ValueError("rgba must be a uint8 [H,W,4] array")
    if isinstance(tile_pixels, bool) or not isinstance(tile_pixels, int) or tile_pixels < 1:
        raise ValueError("tile_pixels must be positive")
    height, width = rgba.shape[:2]
    yy, xx = np.indices((height, width))
    tile = np.where(((xx // tile_pixels + yy // tile_pixels) % 2)[..., None] == 0, 96, 128)
    background = np.repeat(tile, 3, axis=2).astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    return np.rint(rgba[..., :3] * alpha + background * (1 - alpha)).astype(np.uint8)


def export_inference_sprite_display_bundle(
    inference_report_path: Path | str,
    output_directory: Path | str,
    *,
    expected_inference_report_sha256: str,
    config: SpriteDisplayDecodeConfig | None = None,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Decode every 128px inference sample into transparent display derivatives."""

    report_path = Path(inference_report_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace sprite display bundle: {output}")
    report_bytes = report_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != expected_inference_report_sha256:
        raise ValueError("inference report SHA-256 mismatch")
    report = _object(report_bytes, "inference report")
    if report.get("artifact_kind") != "mugen_sd14_attention_lora_rgb_inference":
        raise ValueError("inference report has the wrong artifact kind")
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("inference report samples are absent")
    operation = config or SpriteDisplayDecodeConfig()
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(128 * 1024**2, label="sprite display decode bundle")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent)
    )
    rows = []
    try:
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError("inference sample is invalid")
            generated = sample.get("downsample_128")
            if not isinstance(generated, dict):
                raise ValueError("inference sample lacks 128px RGB")
            source_path = (report_path.parent / _text(generated, "path")).resolve()
            source_bytes = source_path.read_bytes()
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            if source_sha256 != generated.get("file_sha256"):
                raise ValueError(f"inference sample hash differs at row {index}")
            with Image.open(source_path) as source_image:
                rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
            rgba, metadata = decode_generated_rgb_sprite(rgb, config=operation)
            stem = (
                f"{index:03d}-{hashlib.sha256(_text(sample, 'prompt').encode()).hexdigest()[:10]}"
            )
            transparent_path = staging / f"{stem}-transparent-palette.png"
            preview_path = staging / f"{stem}-checker-preview.png"
            Image.fromarray(rgba, mode="RGBA").save(transparent_path, optimize=False)
            Image.fromarray(composite_rgba_on_checkerboard(rgba), mode="RGB").resize(
                (512, 512), Image.Resampling.NEAREST
            ).save(preview_path, optimize=False)
            rows.append(
                {
                    "decode": metadata,
                    "prompt": sample["prompt"],
                    "source_rgb": {
                        "file_sha256": source_sha256,
                        "path": str(source_path),
                    },
                    "transparent_rgba": {
                        "array_content_sha256": _array_sha256(rgba),
                        "file_sha256": _file_sha256(transparent_path),
                        "path": transparent_path.name,
                    },
                    "checker_preview": {
                        "file_sha256": _file_sha256(preview_path),
                        "path": preview_path.name,
                    },
                }
            )
        manifest = {
            "artifact_kind": "generated_rgb_sprite_display_decode_bundle",
            "claim": (
                "display-only generated derivative; transparency is inferred from border color, "
                "not canonical model alpha"
            ),
            "config": asdict(operation),
            "record_count": len(rows),
            "records": rows,
            "schema_version": 1,
            "source": {
                "inference_report_file_sha256": expected_inference_report_sha256,
                "inference_report_path": str(report_path),
            },
        }
        payload = _canonical_json(manifest)
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output / "manifest.json", hashlib.sha256(payload).hexdigest()


def _border_connected(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    output = np.zeros_like(mask, dtype=np.bool_)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        for y in (0, height - 1):
            if mask[y, x] and not output[y, x]:
                output[y, x] = True
                queue.append((y, x))
    for y in range(height):
        for x in (0, width - 1):
            if mask[y, x] and not output[y, x]:
                output[y, x] = True
                queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_y, next_x = y + dy, x + dx
            if (
                0 <= next_y < height
                and 0 <= next_x < width
                and mask[next_y, next_x]
                and not output[next_y, next_x]
            ):
                output[next_y, next_x] = True
                queue.append((next_y, next_x))
    return output


def _remove_small_components(mask: np.ndarray, *, minimum_pixels: int) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    output = np.zeros_like(mask, dtype=np.bool_)
    for start_y, start_x in zip(*np.nonzero(mask), strict=True):
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        queue = deque([(int(start_y), int(start_x))])
        component = []
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_y, next_x = y + dy, x + dx
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        if len(component) >= minimum_pixels:
            for y, x in component:
                output[y, x] = True
    return output


def _object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key} must be non-empty text")
    return value


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
