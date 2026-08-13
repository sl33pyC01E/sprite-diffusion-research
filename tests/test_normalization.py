from __future__ import annotations

import pytest
from PIL import Image

from spritelab.normalization import OversizedSpriteError, normalize_sprite_sequence


def _frame(size: tuple[int, int], bbox: tuple[int, int, int, int]) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    for y in range(bbox[1], bbox[3]):
        for x in range(bbox[0], bbox[2]):
            image.putpixel((x, y), (x * 10 % 256, y * 10 % 256, 50, 255))
    return image


def test_normalization_uses_union_alignment_integer_scale_and_bottom_anchor() -> None:
    first = _frame((8, 8), (2, 3, 4, 5))
    second = _frame((8, 8), (3, 2, 5, 5))

    result = normalize_sprite_sequence(
        (first, second),
        target_size=(12, 12),
        padding=1,
        max_integer_scale=2,
    )

    assert result.transform.source_content_bbox == (2, 2, 5, 5)
    assert result.transform.aligned_output_bbox == (1, 1, 6, 6)
    assert result.transform.aligned_size == (5, 5)
    assert result.transform.integer_scale == 2
    assert result.transform.scaled_size == (10, 10)
    assert result.transform.destination == (1, 2)
    assert result.transform.resampling == "nearest_positive_integer"
    assert all(frame.size == (12, 12) and frame.mode == "RGBA" for frame in result.frames)
    assert len(result.frame_pixel_sha256) == 2
    assert result.frame_pixel_sha256[0] != result.frame_pixel_sha256[1]


def test_world_offsets_preserve_relative_motion_without_resampling() -> None:
    frame = _frame((4, 4), (1, 1, 3, 3))
    result = normalize_sprite_sequence(
        (frame, frame.copy()),
        target_size=(8, 6),
        offsets=((0, 0), (2, 0)),
        anchor="top_left",
        upscale=False,
    )

    assert result.transform.source_content_bbox == (1, 1, 5, 3)
    assert result.transform.integer_scale == 1
    assert result.transform.destination == (0, 0)
    assert result.transform.resampling == "none"
    assert result.frames[0].getbbox() == (0, 0, 2, 2)
    assert result.frames[1].getbbox() == (2, 0, 4, 2)


def test_transparent_sequence_has_deterministic_nonempty_canvas() -> None:
    transparent = Image.new("RGBA", (3, 2), (123, 45, 67, 0))
    result = normalize_sprite_sequence(
        (transparent,),
        target_size=(6, 6),
        anchor="center",
    )

    assert result.transform.source_content_bbox is None
    assert result.transform.aligned_size == (3, 2)
    assert result.transform.integer_scale == 2
    assert result.transform.destination == (0, 1)
    assert result.frames[0].getbbox() is None


def test_oversized_sequence_is_rejected_without_downsampling_or_crop() -> None:
    frame = _frame((16, 16), (0, 0, 16, 16))
    with pytest.raises(OversizedSpriteError, match="does not fit"):
        normalize_sprite_sequence((frame,), target_size=(8, 8))


def test_normalization_transform_hash_is_stable_and_parameter_sensitive() -> None:
    frame = _frame((4, 4), (1, 1, 3, 3))
    first = normalize_sprite_sequence((frame,), target_size=(8, 8), anchor="center")
    repeat = normalize_sprite_sequence((frame,), target_size=(8, 8), anchor="center")
    bottom = normalize_sprite_sequence((frame,), target_size=(8, 8), anchor="bottom_center")

    assert first.transform.canonical_json == repeat.transform.canonical_json
    assert first.transform.sha256 == repeat.transform.sha256
    assert first.frame_pixel_sha256 == repeat.frame_pixel_sha256
    assert first.transform.sha256 != bottom.transform.sha256


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"target_size": (0, 8)}, ValueError),
        ({"target_size": (True, 8)}, TypeError),
        ({"anchor": "feet"}, ValueError),
        ({"upscale": 1}, TypeError),
        ({"max_integer_scale": 0}, ValueError),
        ({"max_integer_scale": True}, TypeError),
    ],
)
def test_normalization_rejects_invalid_contracts(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        normalize_sprite_sequence((_frame((4, 4), (1, 1, 3, 3)),), **kwargs)
