from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

BBox = tuple[int, int, int, int]
Point = tuple[int, int]
Size = tuple[int, int]
RGBA = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class AnimationFrame:
    """One decoded, full-canvas animation frame.

    ``image`` is always an independent RGBA image. ``source_index`` is the
    frame number in the carrier; APNG default images therefore leave index 0
    out of the animation frame sequence.
    """

    source_index: int
    image: Image.Image
    duration_ms: float | None
    disposal: int | None
    blend: int | None
    source_extent: BBox | None


@dataclass(frozen=True, slots=True)
class AnimationInspection:
    """Decoded animation and carrier-level playback metadata.

    A loop count of zero means infinite looping. ``None`` means that the
    carrier did not declare a loop extension.
    """

    source_sha256: str
    format: str
    canvas_size: Size
    source_mode: str
    source_frame_count: int
    loop_count: int | None
    frames: tuple[AnimationFrame, ...]
    default_image: Image.Image | None = None

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def total_duration_ms(self) -> float:
        return sum(frame.duration_ms or 0.0 for frame in self.frames)

    @property
    def is_animated(self) -> bool:
        return self.format in {"GIF", "APNG", "WEBP"} and self.source_frame_count > 1


@dataclass(frozen=True, slots=True)
class PNGInspection:
    """Lossless structural and first-frame inspection of a PNG carrier."""

    source_sha256: str
    size: Size
    mode: str
    bit_depth: int
    color_type_code: int
    color_type: str
    interlaced: bool
    chunk_types: tuple[str, ...]
    has_alpha: bool
    alpha_kind: str
    palette_rgba: tuple[RGBA, ...] | None
    palette_sha256: str | None
    first_frame_pixel_sha256: str
    is_animated: bool
    animation_frame_count: int
    display_frame_count: int
    loop_count: int | None
    has_default_image: bool


@dataclass(frozen=True, slots=True)
class SheetCell:
    index: int
    row: int
    column: int
    bbox: BBox
    image: Image.Image
    pixel_data_sha256: str
    is_fully_transparent: bool


@dataclass(frozen=True, slots=True)
class SpriteSheetInspection:
    png: PNGInspection
    cell_size: Size
    origin: Point
    spacing: Point
    grid_size: Size
    remainder: Size
    cells: tuple[SheetCell, ...]


@dataclass(frozen=True, slots=True)
class AlignedFrames:
    """RGBA frames positioned in one deterministic union-bbox canvas."""

    frames: tuple[Image.Image, ...]
    source_offsets: tuple[Point, ...]
    content_bbox: BBox | None
    output_bbox: BBox
    padding: tuple[int, int, int, int]

    @property
    def size(self) -> Size:
        left, top, right, bottom = self.output_bbox
        return right - left, bottom - top
