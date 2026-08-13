from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from struct import unpack

from PIL import Image, UnidentifiedImageError

from ._source import MediaSource, read_source_bytes
from .models import PNGInspection, Point, SheetCell, Size, SpriteSheetInspection

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_COLOR_TYPES = {
    0: "grayscale",
    2: "truecolor",
    3: "indexed",
    4: "grayscale-alpha",
    6: "truecolor-alpha",
}


class InvalidPNGError(ValueError):
    """Raised when PNG structure or decoding is invalid."""


@dataclass(frozen=True, slots=True)
class _Chunk:
    kind: bytes
    data: bytes


def _chunks(payload: bytes) -> tuple[_Chunk, ...]:
    if not payload.startswith(_PNG_SIGNATURE):
        raise InvalidPNGError("source does not have a PNG signature")
    chunks: list[_Chunk] = []
    cursor = len(_PNG_SIGNATURE)
    while cursor < len(payload):
        if cursor + 12 > len(payload):
            raise InvalidPNGError("truncated PNG chunk header")
        length = int.from_bytes(payload[cursor : cursor + 4], "big")
        kind = payload[cursor + 4 : cursor + 8]
        data_start = cursor + 8
        data_end = data_start + length
        chunk_end = data_end + 4
        if chunk_end > len(payload):
            raise InvalidPNGError(f"truncated {kind!r} PNG chunk")
        chunks.append(_Chunk(kind, payload[data_start:data_end]))
        cursor = chunk_end
        if kind == b"IEND":
            if cursor != len(payload):
                raise InvalidPNGError("data follows the PNG IEND chunk")
            break
    if not chunks or chunks[-1].kind != b"IEND":
        raise InvalidPNGError("PNG has no IEND chunk")
    return tuple(chunks)


def _only_chunk(chunks: tuple[_Chunk, ...], kind: bytes) -> bytes | None:
    matches = [chunk.data for chunk in chunks if chunk.kind == kind]
    if len(matches) > 1:
        raise InvalidPNGError(f"PNG contains multiple {kind.decode('ascii')} chunks")
    return matches[0] if matches else None


def _palette(
    plte: bytes | None, transparency: bytes | None
) -> tuple[tuple[tuple[int, int, int, int], ...] | None, str | None]:
    if plte is None:
        return None, None
    if not plte or len(plte) % 3:
        raise InvalidPNGError("PNG PLTE length must be a non-zero multiple of three")
    entry_count = len(plte) // 3
    if entry_count > 256:
        raise InvalidPNGError("PNG palette contains more than 256 entries")
    alpha = transparency or b""
    if len(alpha) > entry_count:
        raise InvalidPNGError("PNG tRNS contains more entries than PLTE")
    rgba = tuple(
        (
            plte[index * 3],
            plte[index * 3 + 1],
            plte[index * 3 + 2],
            alpha[index] if index < len(alpha) else 255,
        )
        for index in range(entry_count)
    )
    palette_hash = hashlib.sha256(plte + b"\0" + alpha).hexdigest()
    return rgba, palette_hash


def _inspect_png_bytes(payload: bytes) -> PNGInspection:
    chunks = _chunks(payload)
    ihdr = _only_chunk(chunks, b"IHDR")
    if ihdr is None or len(ihdr) != 13:
        raise InvalidPNGError("PNG must contain one 13-byte IHDR chunk")
    width, height, bit_depth, color_type_code, compression, filtering, interlace = unpack(
        ">IIBBBBB", ihdr
    )
    if not width or not height:
        raise InvalidPNGError("PNG dimensions must be non-zero")
    if compression != 0 or filtering != 0 or interlace not in {0, 1}:
        raise InvalidPNGError("PNG uses unsupported IHDR method values")
    try:
        color_type = _COLOR_TYPES[color_type_code]
    except KeyError as error:
        raise InvalidPNGError(f"unknown PNG color type {color_type_code}") from error

    plte = _only_chunk(chunks, b"PLTE")
    transparency = _only_chunk(chunks, b"tRNS")
    palette_rgba, palette_sha256 = _palette(plte, transparency)
    if color_type_code == 3 and palette_rgba is None:
        raise InvalidPNGError("indexed PNG has no palette")

    if color_type_code in {4, 6}:
        alpha_kind = "channel"
    elif transparency is not None and color_type_code == 3:
        alpha_kind = "palette"
    elif transparency is not None:
        alpha_kind = "color-key"
    else:
        alpha_kind = "none"

    animation_control = _only_chunk(chunks, b"acTL")
    if animation_control is not None:
        if len(animation_control) != 8:
            raise InvalidPNGError("APNG acTL chunk must contain eight bytes")
        animation_frame_count, loop_count = unpack(">II", animation_control)
        if animation_frame_count == 0:
            raise InvalidPNGError("APNG declares zero animation frames")
        is_animated = True
    else:
        animation_frame_count = 0
        loop_count = None
        is_animated = False

    try:
        with Image.open(BytesIO(payload)) as verifier:
            if verifier.format != "PNG":
                raise InvalidPNGError(f"expected PNG decoder, got {verifier.format!r}")
            verifier.verify()
        with Image.open(BytesIO(payload)) as image:
            image.seek(0)
            image.load()
            mode = image.mode
            display_frame_count = getattr(image, "n_frames", 1)
            has_default_image = bool(image.info.get("default_image", False))
            first_frame_pixel_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
    except (OSError, SyntaxError, UnidentifiedImageError) as error:
        raise InvalidPNGError("Pillow could not validate the PNG") from error

    if image.size != (width, height):
        raise InvalidPNGError("IHDR dimensions disagree with decoded dimensions")

    return PNGInspection(
        source_sha256=hashlib.sha256(payload).hexdigest(),
        size=(width, height),
        mode=mode,
        bit_depth=bit_depth,
        color_type_code=color_type_code,
        color_type=color_type,
        interlaced=bool(interlace),
        chunk_types=tuple(chunk.kind.decode("ascii") for chunk in chunks),
        has_alpha=alpha_kind != "none",
        alpha_kind=alpha_kind,
        palette_rgba=palette_rgba,
        palette_sha256=palette_sha256,
        first_frame_pixel_sha256=first_frame_pixel_sha256,
        is_animated=is_animated,
        animation_frame_count=animation_frame_count,
        display_frame_count=display_frame_count,
        loop_count=loop_count,
        has_default_image=has_default_image,
    )


def inspect_png(source: MediaSource) -> PNGInspection:
    """Inspect PNG structure without changing palette indices or source bytes."""

    return _inspect_png_bytes(read_source_bytes(source))


def _pair(value: int | Point, *, name: str, minimum: int) -> Point:
    result = (value, value) if isinstance(value, int) else value
    if len(result) != 2:
        raise ValueError(f"{name} must contain two integers")
    if any(not isinstance(item, int) or isinstance(item, bool) for item in result):
        raise TypeError(f"{name} must contain integers")
    if any(item < minimum for item in result):
        raise ValueError(f"{name} values must be at least {minimum}")
    return result


def _grid_axis(available: int, cell: int, spacing: int) -> int:
    return (available + spacing) // (cell + spacing)


def inspect_sprite_sheet(
    source: MediaSource,
    *,
    cell_size: int | Size,
    origin: int | Point = (0, 0),
    spacing: int | Point = (0, 0),
    grid_size: Size | None = None,
    require_exact: bool = True,
) -> SpriteSheetInspection:
    """Slice a static PNG sheet on an explicit native-pixel grid.

    Cropping preserves the source mode, palette, and pixel indices. The
    function never resizes. With ``require_exact=True`` (the default), pixels
    after the last cell are rejected instead of being silently discarded.
    """

    payload = read_source_bytes(source)
    png = _inspect_png_bytes(payload)
    if png.is_animated:
        raise ValueError("sprite-sheet inspection requires a static PNG")

    cell_width, cell_height = _pair(cell_size, name="cell_size", minimum=1)
    origin_x, origin_y = _pair(origin, name="origin", minimum=0)
    spacing_x, spacing_y = _pair(spacing, name="spacing", minimum=0)
    available_width = png.size[0] - origin_x
    available_height = png.size[1] - origin_y
    if available_width <= 0 or available_height <= 0:
        raise ValueError("origin must leave image pixels available for cells")

    if grid_size is None:
        columns = _grid_axis(available_width, cell_width, spacing_x)
        rows = _grid_axis(available_height, cell_height, spacing_y)
    else:
        columns, rows = _pair(grid_size, name="grid_size", minimum=1)
    if columns < 1 or rows < 1:
        raise ValueError("cell size and spacing do not fit any complete cells")

    used_width = columns * cell_width + (columns - 1) * spacing_x
    used_height = rows * cell_height + (rows - 1) * spacing_y
    remainder = available_width - used_width, available_height - used_height
    if remainder[0] < 0 or remainder[1] < 0:
        raise ValueError("declared sprite grid exceeds the PNG dimensions")
    if require_exact and remainder != (0, 0):
        raise ValueError(
            f"sprite grid leaves an unconsumed right/bottom remainder of {remainder} pixels"
        )

    cells: list[SheetCell] = []
    with Image.open(BytesIO(payload)) as image:
        image.load()
        for row in range(rows):
            top = origin_y + row * (cell_height + spacing_y)
            for column in range(columns):
                left = origin_x + column * (cell_width + spacing_x)
                bbox = left, top, left + cell_width, top + cell_height
                cell = image.crop(bbox)
                alpha = cell.convert("RGBA").getchannel("A")
                cells.append(
                    SheetCell(
                        index=row * columns + column,
                        row=row,
                        column=column,
                        bbox=bbox,
                        image=cell,
                        pixel_data_sha256=hashlib.sha256(cell.tobytes()).hexdigest(),
                        is_fully_transparent=alpha.getbbox() is None,
                    )
                )

    return SpriteSheetInspection(
        png=png,
        cell_size=(cell_width, cell_height),
        origin=(origin_x, origin_y),
        spacing=(spacing_x, spacing_y),
        grid_size=(columns, rows),
        remainder=remainder,
        cells=tuple(cells),
    )
