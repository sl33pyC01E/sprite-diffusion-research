from __future__ import annotations

import hashlib
from io import BytesIO

import pytest
from PIL import Image

from spritelab.media import UnsupportedAnimationError, extract_animation

TRANSPARENT = (0, 0, 0, 0)
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)


def _dot_frame(x: int, color: tuple[int, int, int, int]) -> Image.Image:
    frame = Image.new("RGBA", (4, 3), TRANSPARENT)
    frame.putpixel((x, 0), color)
    return frame


def _animated_bytes(
    format_name: str,
    *,
    disposal: list[int],
    blend: list[int] | None = None,
) -> bytes:
    frames = [_dot_frame(0, RED), _dot_frame(1, GREEN), _dot_frame(2, BLUE)]
    output = BytesIO()
    kwargs: dict[str, object] = {
        "save_all": True,
        "append_images": frames[1:],
        "duration": [70, 130, 90],
        "loop": 4,
        "disposal": disposal,
    }
    if format_name == "GIF":
        kwargs.update(transparency=0, optimize=True)
    else:
        kwargs["blend"] = blend
    frames[0].save(output, format_name, **kwargs)
    return output.getvalue()


def test_extract_gif_returns_composited_independent_rgba_frames() -> None:
    payload = _animated_bytes("GIF", disposal=[1, 2, 1])

    result = extract_animation(payload)

    assert result.format == "GIF"
    assert result.canvas_size == (4, 3)
    assert result.source_frame_count == result.frame_count == 3
    assert result.loop_count == 4
    assert result.total_duration_ms == 290
    assert [frame.duration_ms for frame in result.frames] == [70, 130, 90]
    assert [frame.disposal for frame in result.frames] == [1, 2, 1]
    assert all(frame.image.mode == "RGBA" for frame in result.frames)
    assert all(frame.image.size == result.canvas_size for frame in result.frames)
    assert result.source_sha256 == hashlib.sha256(payload).hexdigest()

    # GIF frame 1 overlays frame 0; disposal 2 then clears its update region.
    assert result.frames[1].image.getpixel((0, 0)) == RED
    assert result.frames[1].image.getpixel((1, 0)) == GREEN
    assert result.frames[2].image.getpixel((0, 0)) == TRANSPARENT
    assert result.frames[2].image.getpixel((1, 0)) == TRANSPARENT
    assert result.frames[2].image.getpixel((2, 0)) == BLUE

    result.frames[2].image.putpixel((0, 0), BLUE)
    assert result.frames[0].image.getpixel((0, 0)) == RED


def test_extract_apng_composites_blend_and_disposal() -> None:
    payload = _animated_bytes("PNG", disposal=[0, 0, 0], blend=[0, 1, 1])

    result = extract_animation(payload)

    assert result.format == "APNG"
    assert result.loop_count == 4
    assert [frame.duration_ms for frame in result.frames] == [70.0, 130.0, 90.0]
    assert [frame.disposal for frame in result.frames] == [0, 0, 0]
    assert [frame.blend for frame in result.frames] == [0, 1, 1]
    assert result.frames[2].image.getpixel((0, 0)) == RED
    assert result.frames[2].image.getpixel((1, 0)) == GREEN
    assert result.frames[2].image.getpixel((2, 0)) == BLUE


def test_extract_apng_keeps_default_image_out_of_playback_frames() -> None:
    poster = Image.new("RGBA", (3, 2), (255, 0, 255, 255))
    first = Image.new("RGBA", (3, 2), RED)
    second = Image.new("RGBA", (3, 2), BLUE)
    output = BytesIO()
    poster.save(
        output,
        "PNG",
        save_all=True,
        append_images=[first, second],
        default_image=True,
        duration=[40, 90],
        loop=2,
    )

    result = extract_animation(output.getvalue())

    assert result.source_frame_count == 3
    assert result.frame_count == 2
    assert result.default_image is not None
    assert result.default_image.getpixel((0, 0)) == (255, 0, 255, 255)
    assert [frame.source_index for frame in result.frames] == [1, 2]
    assert [frame.duration_ms for frame in result.frames] == [40.0, 90.0]
    assert [frame.image.getpixel((0, 0)) for frame in result.frames] == [RED, BLUE]


def test_extract_static_png_as_one_rgba_frame_and_preserves_stream_position() -> None:
    output = BytesIO()
    Image.new("RGB", (2, 1), (12, 34, 56)).save(output, "PNG")
    stream = BytesIO(output.getvalue())
    stream.seek(7)

    result = extract_animation(stream)

    assert stream.tell() == 7
    assert result.format == "PNG"
    assert result.frame_count == 1
    assert result.loop_count is None
    assert result.frames[0].duration_ms is None
    assert result.frames[0].image.getpixel((0, 0)) == (12, 34, 56, 255)


def test_extract_animated_webp_as_rgba_frames() -> None:
    frames = [Image.new("RGBA", (3, 2), RED), Image.new("RGBA", (3, 2), BLUE)]
    output = BytesIO()
    frames[0].save(
        output,
        "WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=[50, 75],
        loop=0,
        lossless=True,
    )

    result = extract_animation(output.getvalue())

    assert result.format == "WEBP"
    assert result.is_animated
    assert result.frame_count == 2
    assert result.loop_count == 0
    assert [frame.duration_ms for frame in result.frames] == [50, 75]
    assert [frame.image.getpixel((0, 0)) for frame in result.frames] == [RED, BLUE]


def test_extract_animation_rejects_other_image_formats() -> None:
    output = BytesIO()
    Image.new("RGB", (1, 1), "red").save(output, "BMP")

    with pytest.raises(UnsupportedAnimationError, match="supports GIF, WebP"):
        extract_animation(output.getvalue())
