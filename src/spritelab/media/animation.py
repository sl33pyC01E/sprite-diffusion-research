from __future__ import annotations

import hashlib
from io import BytesIO
from numbers import Real

from PIL import Image, UnidentifiedImageError

from ._source import MediaSource, read_source_bytes
from .models import AnimationFrame, AnimationInspection, BBox


class UnsupportedAnimationError(ValueError):
    """Raised when a carrier is not a GIF or PNG/APNG."""


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"expected integer animation metadata, got {value!r}")
    return value


def _optional_duration(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"expected numeric frame duration, got {value!r}")
    duration = float(value)
    if duration < 0:
        raise ValueError(f"frame duration must be non-negative, got {duration}")
    return duration


def _optional_bbox(value: object) -> BBox | None:
    if value is None:
        return None
    if not isinstance(value, tuple | list) or len(value) != 4:
        raise ValueError(f"expected four-value frame extent, got {value!r}")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in value):
        raise ValueError(f"frame extent must contain integers, got {value!r}")
    left, top, right, bottom = value
    return left, top, right, bottom


def extract_animation(source: MediaSource) -> AnimationInspection:
    """Extract GIF, WebP, or PNG/APNG as composited full-canvas RGBA frames.

    Pillow applies GIF disposal and APNG disposal/blend operations while
    seeking. Each frame is copied immediately so subsequent seeks cannot
    mutate an earlier result. Static PNG and single-frame GIF carriers are
    accepted and represented as one frame.
    """

    payload = read_source_bytes(source)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        image_context = Image.open(BytesIO(payload))
    except UnidentifiedImageError as error:
        raise UnsupportedAnimationError("source is not a recognized image") from error

    with image_context as image:
        carrier_format = image.format
        if carrier_format not in {"GIF", "PNG", "WEBP"}:
            raise UnsupportedAnimationError(
                f"animation extraction supports GIF, WebP, and PNG/APNG, not {carrier_format!r}"
            )

        canvas_size = image.size
        source_mode = image.mode
        source_frame_count = getattr(image, "n_frames", 1)
        loop_count = _optional_int(image.info.get("loop"))
        has_default_image = carrier_format == "PNG" and bool(image.info.get("default_image", False))

        default_image: Image.Image | None = None
        first_animation_index = 0
        if has_default_image:
            image.seek(0)
            image.load()
            default_image = image.convert("RGBA").copy()
            first_animation_index = 1

        frames: list[AnimationFrame] = []
        for index in range(first_animation_index, source_frame_count):
            image.seek(index)
            image.load()
            frame_info = image.info
            rgba = image.convert("RGBA")
            if rgba.size != canvas_size:
                raise ValueError(
                    f"decoder returned frame {index} at {rgba.size}, expected {canvas_size}"
                )

            if carrier_format in {"GIF", "WEBP"}:
                disposal = _optional_int(getattr(image, "disposal_method", None))
                source_extent = _optional_bbox(getattr(image, "dispose_extent", None))
                blend = None
            else:
                disposal = _optional_int(frame_info.get("disposal"))
                source_extent = _optional_bbox(frame_info.get("bbox"))
                blend = _optional_int(frame_info.get("blend"))

            frames.append(
                AnimationFrame(
                    source_index=index,
                    image=rgba.copy(),
                    duration_ms=_optional_duration(frame_info.get("duration")),
                    disposal=disposal,
                    blend=blend,
                    source_extent=source_extent,
                )
            )

        format_name = (
            "APNG" if carrier_format == "PNG" and source_frame_count > 1 else carrier_format
        )
        return AnimationInspection(
            source_sha256=digest,
            format=format_name,
            canvas_size=canvas_size,
            source_mode=source_mode,
            source_frame_count=source_frame_count,
            loop_count=loop_count,
            frames=tuple(frames),
            default_image=default_image,
        )
