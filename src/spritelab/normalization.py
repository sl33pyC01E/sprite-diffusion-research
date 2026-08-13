"""Deterministic, integer-safe normalization for transparent sprite clips."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from PIL import Image

from spritelab.media.alignment import align_frames_to_union, scale_integer_nearest
from spritelab.media.models import BBox, Point, Size

Anchor = Literal["bottom_center", "center", "top_left"]


class OversizedSpriteError(ValueError):
    """Raised when lossless integer-safe normalization cannot fit the target."""


@dataclass(frozen=True, slots=True)
class NormalizationTransform:
    schema_version: int
    target_size: Size
    anchor: Anchor
    source_offsets: tuple[Point, ...]
    source_content_bbox: BBox | None
    aligned_output_bbox: BBox
    aligned_size: Size
    padding: tuple[int, int, int, int]
    alpha_threshold: int
    integer_scale: int
    scaled_size: Size
    destination: Point
    resampling: str

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            asdict(self),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NormalizedSequence:
    frames: tuple[Image.Image, ...]
    transform: NormalizationTransform
    frame_pixel_sha256: tuple[str, ...]


def normalize_sprite_sequence(
    frames: Sequence[Image.Image],
    *,
    target_size: Size = (64, 64),
    offsets: Sequence[Point] | None = None,
    padding: int | tuple[int, int, int, int] = 0,
    alpha_threshold: int = 0,
    anchor: Anchor = "bottom_center",
    upscale: bool = True,
    max_integer_scale: int | None = None,
) -> NormalizedSequence:
    """Crop to a shared world-space union and place it on a fixed RGBA canvas.

    The only possible resampling is positive-integer nearest-neighbor upscale.
    Oversized clips are rejected instead of silently downsampled or cropped.
    Callers can route those clips to a separately documented high-resolution
    bucket or an explicit downsampling policy.
    """

    target_width, target_height = _size(target_size, name="target_size")
    if anchor not in {"bottom_center", "center", "top_left"}:
        raise ValueError(f"Unknown anchor: {anchor!r}")
    if not isinstance(upscale, bool):
        raise TypeError("upscale must be a boolean")
    if max_integer_scale is not None:
        if not isinstance(max_integer_scale, int) or isinstance(max_integer_scale, bool):
            raise TypeError("max_integer_scale must be an integer or None")
        if max_integer_scale < 1:
            raise ValueError("max_integer_scale must be positive")

    aligned = align_frames_to_union(
        frames,
        offsets=offsets,
        padding=padding,
        alpha_threshold=alpha_threshold,
    )
    aligned_width, aligned_height = aligned.size
    if aligned_width > target_width or aligned_height > target_height:
        raise OversizedSpriteError(
            f"aligned sprite {aligned.size!r} does not fit target {target_size!r}; "
            "integer-safe normalization does not downsample"
        )

    scale = 1
    if upscale:
        scale = min(target_width // aligned_width, target_height // aligned_height)
        if max_integer_scale is not None:
            scale = min(scale, max_integer_scale)
        scale = max(scale, 1)
    scaled_size = aligned_width * scale, aligned_height * scale
    destination = _destination(
        target_size=(target_width, target_height),
        content_size=scaled_size,
        anchor=anchor,
    )

    normalized: list[Image.Image] = []
    hashes: list[str] = []
    for frame in aligned.frames:
        scaled = scale_integer_nearest(frame, scale) if scale > 1 else frame.copy()
        canvas = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        canvas.paste(scaled, destination)
        normalized.append(canvas)
        hashes.append(_pixel_sha256(canvas))

    transform = NormalizationTransform(
        schema_version=1,
        target_size=(target_width, target_height),
        anchor=anchor,
        source_offsets=aligned.source_offsets,
        source_content_bbox=aligned.content_bbox,
        aligned_output_bbox=aligned.output_bbox,
        aligned_size=aligned.size,
        padding=aligned.padding,
        alpha_threshold=alpha_threshold,
        integer_scale=scale,
        scaled_size=scaled_size,
        destination=destination,
        resampling="none" if scale == 1 else "nearest_positive_integer",
    )
    return NormalizedSequence(
        frames=tuple(normalized),
        transform=transform,
        frame_pixel_sha256=tuple(hashes),
    )


def _destination(*, target_size: Size, content_size: Size, anchor: Anchor) -> Point:
    target_width, target_height = target_size
    content_width, content_height = content_size
    if anchor == "bottom_center":
        return (target_width - content_width) // 2, target_height - content_height
    if anchor == "center":
        return (target_width - content_width) // 2, (target_height - content_height) // 2
    return 0, 0


def _size(value: Size, *, name: str) -> Size:
    if len(value) != 2 or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise TypeError(f"{name} must contain two integers")
    if value[0] < 1 or value[1] < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _pixel_sha256(image: Image.Image) -> str:
    header = f"RGBA\0{image.width}x{image.height}\0".encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()
