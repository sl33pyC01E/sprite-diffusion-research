from __future__ import annotations

import pytest
from PIL import Image

from spritelab.media import align_frames_to_union, scale_integer_nearest

TRANSPARENT = (0, 0, 0, 0)
RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def test_union_alignment_respects_world_offsets_and_asymmetric_padding() -> None:
    first = Image.new("RGBA", (4, 4), TRANSPARENT)
    first.putpixel((1, 1), RED)
    second = Image.new("RGBA", (3, 3), TRANSPARENT)
    second.putpixel((2, 0), BLUE)

    result = align_frames_to_union(
        [first, second],
        offsets=[(10, 20), (8, 22)],
        padding=(1, 2, 3, 4),
    )

    # Visible world pixels are red=(11,21), blue=(10,22).
    assert result.content_bbox == (10, 21, 12, 23)
    assert result.output_bbox == (9, 19, 15, 27)
    assert result.size == (6, 8)
    assert result.source_offsets == ((10, 20), (8, 22))
    assert all(frame.mode == "RGBA" for frame in result.frames)
    assert result.frames[0].getpixel((2, 2)) == RED
    assert result.frames[1].getpixel((1, 3)) == BLUE
    assert result.frames[0].getpixel((1, 3)) == TRANSPARENT


def test_union_alignment_threshold_does_not_modify_pixel_values() -> None:
    frame = Image.new("RGBA", (3, 1), TRANSPARENT)
    frame.putpixel((0, 0), (10, 20, 30, 4))
    frame.putpixel((2, 0), (40, 50, 60, 200))

    result = align_frames_to_union([frame], alpha_threshold=10)

    assert result.content_bbox == (2, 0, 3, 1)
    assert result.output_bbox == (2, 0, 3, 1)
    assert result.frames[0].getpixel((0, 0)) == (40, 50, 60, 200)


def test_fully_transparent_alignment_falls_back_to_source_canvas_union() -> None:
    first = Image.new("RGBA", (2, 3), TRANSPARENT)
    second = Image.new("RGBA", (4, 1), TRANSPARENT)

    result = align_frames_to_union([first, second], offsets=[(-2, 1), (1, -1)])

    assert result.content_bbox is None
    assert result.output_bbox == (-2, -1, 5, 4)
    assert result.size == (7, 5)
    assert all(frame.getbbox() is None for frame in result.frames)


def test_integer_scale_is_pixel_exact_nearest_neighbor() -> None:
    image = Image.new("RGBA", (2, 1))
    image.putdata([RED, BLUE])

    scaled = scale_integer_nearest(image, 3, 2)

    assert scaled.size == (6, 2)
    assert list(scaled.get_flattened_data()) == [RED, RED, RED, BLUE, BLUE, BLUE] * 2


@pytest.mark.parametrize("scale", [0, -1, 1.5, True])
def test_integer_scale_rejects_invalid_factors(scale: object) -> None:
    image = Image.new("RGBA", (1, 1), RED)
    expected_error = TypeError if isinstance(scale, float | bool) else ValueError
    with pytest.raises(expected_error):
        scale_integer_nearest(image, scale)  # type: ignore[arg-type]
