from __future__ import annotations

import hashlib
from io import BytesIO

import pytest
from PIL import Image

from spritelab.media import InvalidPNGError, inspect_png, inspect_sprite_sheet


def _indexed_sheet_bytes() -> bytes:
    image = Image.new("P", (5, 2))
    palette = [255, 0, 0, 0, 255, 0, 0, 0, 255, 20, 20, 20]
    image.putpalette(palette + [0] * (768 - len(palette)))
    image.putdata([0, 1, 3, 2, 3, 1, 0, 3, 2, 3])
    image.info["transparency"] = bytes([255, 128, 0, 255])
    output = BytesIO()
    image.save(output, "PNG", bits=2)
    return output.getvalue()


def test_inspect_png_reports_lossless_indexed_palette_structure() -> None:
    payload = _indexed_sheet_bytes()

    result = inspect_png(payload)

    assert result.source_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.size == (5, 2)
    assert result.mode == "P"
    assert result.bit_depth == 2
    assert result.color_type_code == 3
    assert result.color_type == "indexed"
    assert result.alpha_kind == "palette"
    assert result.has_alpha is True
    assert result.palette_rgba == (
        (255, 0, 0, 255),
        (0, 255, 0, 128),
        (0, 0, 255, 0),
        (20, 20, 20, 255),
    )
    assert result.palette_sha256 is not None
    assert result.chunk_types == ("IHDR", "PLTE", "tRNS", "IDAT", "IEND")
    assert result.is_animated is False
    assert result.display_frame_count == 1


def test_inspect_apng_reads_animation_control_without_decoding_as_sheet() -> None:
    first = Image.new("RGBA", (2, 2), "red")
    second = Image.new("RGBA", (2, 2), "blue")
    output = BytesIO()
    first.save(
        output,
        "PNG",
        save_all=True,
        append_images=[second],
        duration=[50, 75],
        loop=6,
    )
    payload = output.getvalue()

    result = inspect_png(payload)

    assert result.is_animated is True
    assert result.animation_frame_count == 2
    assert result.display_frame_count == 2
    assert result.loop_count == 6
    assert "acTL" in result.chunk_types
    with pytest.raises(ValueError, match="static PNG"):
        inspect_sprite_sheet(payload, cell_size=1)


def test_inspect_sprite_sheet_preserves_palette_indices_and_grid_order() -> None:
    payload = _indexed_sheet_bytes()

    result = inspect_sprite_sheet(
        payload,
        cell_size=(2, 2),
        spacing=(1, 0),
    )

    assert result.grid_size == (2, 1)
    assert result.remainder == (0, 0)
    assert [cell.index for cell in result.cells] == [0, 1]
    assert [cell.bbox for cell in result.cells] == [(0, 0, 2, 2), (3, 0, 5, 2)]
    assert all(cell.image.mode == "P" for cell in result.cells)
    assert list(result.cells[0].image.get_flattened_data()) == [0, 1, 1, 0]
    assert list(result.cells[1].image.get_flattened_data()) == [2, 3, 2, 3]
    assert result.cells[0].is_fully_transparent is False
    assert result.cells[1].is_fully_transparent is False
    assert result.cells[0].pixel_data_sha256 == hashlib.sha256(bytes([0, 1, 1, 0])).hexdigest()


def test_sheet_inspection_rejects_unaccounted_pixels_by_default() -> None:
    output = BytesIO()
    Image.new("RGBA", (5, 4), (0, 0, 0, 0)).save(output, "PNG")

    with pytest.raises(ValueError, match="remainder"):
        inspect_sprite_sheet(output.getvalue(), cell_size=(2, 2))

    result = inspect_sprite_sheet(
        output.getvalue(),
        cell_size=(2, 2),
        require_exact=False,
    )
    assert result.grid_size == (2, 2)
    assert result.remainder == (1, 0)
    assert all(cell.is_fully_transparent for cell in result.cells)


def test_inspect_png_rejects_non_png_and_trailing_bytes() -> None:
    with pytest.raises(InvalidPNGError, match="signature"):
        inspect_png(b"not-png")

    payload = _indexed_sheet_bytes() + b"unexpected"
    with pytest.raises(InvalidPNGError, match="IEND"):
        inspect_png(payload)
