"""Pure image inspection, extraction, and pixel-safe alignment utilities."""

from .alignment import align_frames_to_union, scale_integer_nearest
from .animation import UnsupportedAnimationError, extract_animation
from .models import (
    AlignedFrames,
    AnimationFrame,
    AnimationInspection,
    PNGInspection,
    SheetCell,
    SpriteSheetInspection,
)
from .png import InvalidPNGError, inspect_png, inspect_sprite_sheet

__all__ = [
    "AlignedFrames",
    "AnimationFrame",
    "AnimationInspection",
    "InvalidPNGError",
    "PNGInspection",
    "SheetCell",
    "SpriteSheetInspection",
    "UnsupportedAnimationError",
    "align_frames_to_union",
    "extract_animation",
    "inspect_png",
    "inspect_sprite_sheet",
    "scale_integer_nearest",
]
