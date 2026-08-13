from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from .models import AlignedFrames, BBox, Point


def _union(left: BBox | None, right: BBox) -> BBox:
    if left is None:
        return right
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _padding(value: int | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    result = (value, value, value, value) if isinstance(value, int) else value
    if len(result) != 4:
        raise ValueError("padding must be an integer or (left, top, right, bottom)")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in result):
        raise TypeError("padding must contain integers")
    if any(item < 0 for item in result):
        raise ValueError("padding must be non-negative")
    return result


def _content_bbox(image: Image.Image, alpha_threshold: int) -> BBox | None:
    alpha = image.getchannel("A")
    if alpha_threshold:
        alpha = alpha.point(lambda value: 255 if value > alpha_threshold else 0)
    return alpha.getbbox()


def align_frames_to_union(
    frames: Sequence[Image.Image],
    *,
    offsets: Sequence[Point] | None = None,
    padding: int | tuple[int, int, int, int] = 0,
    alpha_threshold: int = 0,
) -> AlignedFrames:
    """Place frames on the padded union of their visible alpha bounds.

    Offsets are positions in a shared world coordinate system. Frames are
    converted to RGBA, translated, cropped, and padded only; no resampling is
    performed. If every frame is transparent, the union of the source canvas
    extents is used so the result remains deterministic and non-empty.
    """

    if not frames:
        raise ValueError("at least one frame is required")
    if not isinstance(alpha_threshold, int) or isinstance(alpha_threshold, bool):
        raise TypeError("alpha_threshold must be an integer")
    if not 0 <= alpha_threshold <= 254:
        raise ValueError("alpha_threshold must be between 0 and 254")

    if offsets is None:
        normalized_offsets = tuple((0, 0) for _ in frames)
    else:
        if len(offsets) != len(frames):
            raise ValueError("offset count must equal frame count")
        normalized_offsets = tuple(offsets)
    for offset in normalized_offsets:
        if len(offset) != 2 or any(
            not isinstance(item, int) or isinstance(item, bool) for item in offset
        ):
            raise TypeError("each offset must contain two integers")

    rgba_frames = tuple(frame.convert("RGBA") for frame in frames)
    for frame in rgba_frames:
        if frame.width < 1 or frame.height < 1:
            raise ValueError("frames must have non-zero dimensions")

    visible_union: BBox | None = None
    canvas_union: BBox | None = None
    for frame, (offset_x, offset_y) in zip(rgba_frames, normalized_offsets, strict=True):
        canvas_union = _union(
            canvas_union,
            (offset_x, offset_y, offset_x + frame.width, offset_y + frame.height),
        )
        local_bbox = _content_bbox(frame, alpha_threshold)
        if local_bbox is not None:
            visible_union = _union(
                visible_union,
                (
                    local_bbox[0] + offset_x,
                    local_bbox[1] + offset_y,
                    local_bbox[2] + offset_x,
                    local_bbox[3] + offset_y,
                ),
            )

    assert canvas_union is not None
    base_bbox = visible_union if visible_union is not None else canvas_union
    pad_left, pad_top, pad_right, pad_bottom = _padding(padding)
    output_bbox = (
        base_bbox[0] - pad_left,
        base_bbox[1] - pad_top,
        base_bbox[2] + pad_right,
        base_bbox[3] + pad_bottom,
    )
    output_size = output_bbox[2] - output_bbox[0], output_bbox[3] - output_bbox[1]

    aligned: list[Image.Image] = []
    for frame, (offset_x, offset_y) in zip(rgba_frames, normalized_offsets, strict=True):
        canvas = Image.new("RGBA", output_size, (0, 0, 0, 0))
        destination = offset_x - output_bbox[0], offset_y - output_bbox[1]
        canvas.paste(frame, destination)
        aligned.append(canvas)

    return AlignedFrames(
        frames=tuple(aligned),
        source_offsets=normalized_offsets,
        content_bbox=visible_union,
        output_bbox=output_bbox,
        padding=(pad_left, pad_top, pad_right, pad_bottom),
    )


def scale_integer_nearest(
    image: Image.Image, scale_x: int, scale_y: int | None = None
) -> Image.Image:
    """Scale by positive integer factors using nearest-neighbor sampling only."""

    if scale_y is None:
        scale_y = scale_x
    if any(
        not isinstance(factor, int) or isinstance(factor, bool) for factor in (scale_x, scale_y)
    ):
        raise TypeError("scale factors must be integers")
    if scale_x < 1 or scale_y < 1:
        raise ValueError("scale factors must be positive")
    return image.resize(
        (image.width * scale_x, image.height * scale_y),
        resample=Image.Resampling.NEAREST,
    )
