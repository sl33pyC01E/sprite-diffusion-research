"""Derived fixed-shape views of authoritative native MUGEN action pixels."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

from spritelab.temporal import TemporalSelection, select_temporal_frames


@dataclass(frozen=True, slots=True)
class WorldViewTransform:
    target_size: int
    padding: int
    scale: float
    anchor_x: int
    anchor_y: int
    world_left: int
    world_top: int
    world_right: int
    world_bottom: int


@dataclass(frozen=True, slots=True)
class PlacedWorldClip:
    rgba: np.ndarray
    clipped_visible_pixels: int


@dataclass(frozen=True, slots=True)
class LeakageIdentity:
    identity_id: str
    source_digest: str
    content_digests: tuple[str, ...]


def assign_leakage_safe_splits(
    identities: tuple[LeakageIdentity, ...],
) -> dict[str, str]:
    """Group exact source/content duplicates before deterministic 90/5/5 splitting."""

    if not identities:
        return {}
    by_id = {row.identity_id: row for row in identities}
    if len(by_id) != len(identities):
        raise ValueError("leakage identities must have unique identity_id values")
    parent = {identity_id: identity_id for identity_id in by_id}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if left_root.encode("utf-8") < right_root.encode("utf-8"):
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    owners: dict[str, str] = {}
    for row in sorted(identities, key=lambda value: value.identity_id.encode("utf-8")):
        if not row.identity_id or not row.source_digest or not row.content_digests:
            raise ValueError("leakage identities require nonempty IDs/source/content digests")
        for token in (
            f"source:{row.source_digest}",
            *(f"content:{x}" for x in row.content_digests),
        ):
            previous = owners.setdefault(token, row.identity_id)
            union(previous, row.identity_id)
    components: dict[str, list[LeakageIdentity]] = {}
    for row in identities:
        components.setdefault(find(row.identity_id), []).append(row)
    output: dict[str, str] = {}
    for rows in components.values():
        tokens = sorted(
            {
                *(f"identity:{row.identity_id}" for row in rows),
                *(f"source:{row.source_digest}" for row in rows),
                *(f"content:{digest}" for row in rows for digest in row.content_digests),
            }
        )
        digest = hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        split = "train" if bucket < 90 else "validation" if bucket < 95 else "test"
        for row in rows:
            output[row.identity_id] = split
    return output


def plan_world_view(
    extents: tuple[tuple[int, int, int, int], ...],
    *,
    target_size: int = 128,
    padding: int = 8,
    maximum_scale: float = 4.0,
) -> WorldViewTransform:
    """Fit world-coordinate extents around MUGEN's player axis without excluding data."""

    if not extents:
        raise ValueError("extents must not be empty")
    if target_size <= 0 or padding < 0 or padding * 2 >= target_size:
        raise ValueError("target_size/padding leave no drawable canvas")
    if not np.isfinite(maximum_scale) or maximum_scale <= 0:
        raise ValueError("maximum_scale must be finite and positive")
    left = min(row[0] for row in extents)
    top = min(row[1] for row in extents)
    right = max(row[2] for row in extents)
    bottom = max(row[3] for row in extents)
    if right <= left or bottom <= top:
        raise ValueError("extents must have positive area")
    anchor_x = target_size // 2
    anchor_y = target_size - padding
    limits = [maximum_scale]
    if left < 0:
        limits.append((anchor_x - padding) / -left)
    if right > 0:
        limits.append((target_size - padding - anchor_x) / right)
    if top < 0:
        limits.append((anchor_y - padding) / -top)
    if bottom > 0:
        limits.append((target_size - anchor_y) / bottom)
    scale = min(limits)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("world extents cannot be placed on the requested canvas")
    return WorldViewTransform(
        target_size=target_size,
        padding=padding,
        scale=float(scale),
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        world_left=left,
        world_top=top,
        world_right=right,
        world_bottom=bottom,
    )


def place_world_clip(
    rgba: np.ndarray,
    *,
    world_left: int,
    world_top: int,
    transform: WorldViewTransform,
) -> PlacedWorldClip:
    """Nearest-scale and place one native [T,H,W,4] action around the player axis."""

    if rgba.ndim != 4 or rgba.shape[-1] != 4 or rgba.dtype != np.uint8:
        raise ValueError("rgba must be uint8 [T,H,W,4]")
    if rgba.shape[0] <= 0 or rgba.shape[1] <= 0 or rgba.shape[2] <= 0:
        raise ValueError("rgba dimensions must be positive")
    scaled = _nearest_scale(rgba, transform.scale)
    output = np.zeros(
        (rgba.shape[0], transform.target_size, transform.target_size, 4), dtype=np.uint8
    )
    destination_left = transform.anchor_x + _round_half_away(world_left * transform.scale)
    destination_top = transform.anchor_y + _round_half_away(world_top * transform.scale)
    source_left = max(0, -destination_left)
    source_top = max(0, -destination_top)
    source_right = min(scaled.shape[2], transform.target_size - destination_left)
    source_bottom = min(scaled.shape[1], transform.target_size - destination_top)
    visible_before = int(np.count_nonzero(scaled[..., 3]))
    if source_right > source_left and source_bottom > source_top:
        target_left = max(0, destination_left)
        target_top = max(0, destination_top)
        output[
            :,
            target_top : target_top + source_bottom - source_top,
            target_left : target_left + source_right - source_left,
        ] = scaled[:, source_top:source_bottom, source_left:source_right]
    visible_after = int(np.count_nonzero(output[..., 3]))
    return PlacedWorldClip(output, visible_before - visible_after)


def select_action_frames(
    durations_ticks: tuple[int, ...],
    *,
    loop_mode: str,
    target_frame_count: int = 8,
) -> TemporalSelection:
    """Create a fixed-frame derivative while retaining zero-tick/terminal authored frames."""

    if not durations_ticks:
        raise ValueError("durations_ticks must not be empty")
    if any(value < -1 for value in durations_ticks):
        raise ValueError("MUGEN durations must be -1 or nonnegative")
    weights = tuple(max(1, value) for value in durations_ticks)
    if loop_mode == "loop":
        total = sum(weights)
        elapsed = 0
        phases = []
        for weight in weights:
            phases.append(elapsed / total)
            elapsed += weight
        normalized: Literal["loop", "one_shot"] = "loop"
    else:
        if len(weights) == 1:
            phases = [0.0]
        else:
            starts = [0]
            for weight in weights[:-1]:
                starts.append(starts[-1] + weight)
            denominator = starts[-1]
            phases = [value / denominator for value in starts]
        normalized = "one_shot"
    return select_temporal_frames(
        len(durations_ticks),
        target_frame_count,
        loop_mode=normalized,
        source_phases=phases,
    )


def _nearest_scale(rgba: np.ndarray, scale: float) -> np.ndarray:
    height, width = rgba.shape[1:3]
    output_width = max(1, _round_half_away(width * scale))
    output_height = max(1, _round_half_away(height * scale))
    source_x = np.minimum((np.arange(output_width) / scale).astype(int), width - 1)
    source_y = np.minimum((np.arange(output_height) / scale).astype(int), height - 1)
    return np.ascontiguousarray(rgba[:, source_y[:, None], source_x[None, :], :])


def _round_half_away(value: float) -> int:
    if value >= 0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))
