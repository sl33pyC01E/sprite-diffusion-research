from __future__ import annotations

import numpy as np

from spritelab.mugen_schema_view import (
    LeakageIdentity,
    assign_leakage_safe_splits,
    place_world_clip,
    plan_world_view,
    select_action_frames,
)


def test_leakage_split_groups_shared_source_and_pixel_content() -> None:
    rows = (
        LeakageIdentity("a", "source-a", ("pixels-a",)),
        LeakageIdentity("b", "source-b", ("pixels-a", "pixels-b")),
        LeakageIdentity("c", "source-b", ("pixels-c",)),
        LeakageIdentity("d", "source-d", ("pixels-d",)),
    )

    splits = assign_leakage_safe_splits(rows)

    assert splits["a"] == splits["b"] == splits["c"]
    assert set(splits) == {"a", "b", "c", "d"}


def test_leakage_split_rejects_duplicate_identity_ids() -> None:
    rows = (
        LeakageIdentity("a", "source-a", ("pixels-a",)),
        LeakageIdentity("a", "source-b", ("pixels-b",)),
    )

    with np.testing.assert_raises_regex(ValueError, "unique identity_id"):
        assign_leakage_safe_splits(rows)


def test_world_view_keeps_mugen_axis_and_nearest_pixels() -> None:
    transform = plan_world_view(((-2, -3, 2, 0),), target_size=16, padding=2, maximum_scale=3)
    source = np.zeros((1, 3, 4, 4), dtype=np.uint8)
    source[0, 0, 0] = (10, 20, 30, 255)

    placed = place_world_clip(source, world_left=-2, world_top=-3, transform=transform)

    assert transform.scale == 3.0
    assert transform.anchor_x == 8
    assert transform.anchor_y == 14
    assert placed.clipped_visible_pixels == 0
    assert np.all(placed.rgba[0, 5:8, 2:5] == (10, 20, 30, 255))


def test_world_view_reports_clipped_visible_attack_pixels() -> None:
    transform = plan_world_view(((-1, -1, 1, 0),), target_size=8, padding=1)
    source = np.full((1, 2, 8, 4), 255, dtype=np.uint8)

    placed = place_world_clip(source, world_left=-4, world_top=-2, transform=transform)

    assert placed.clipped_visible_pixels > 0
    assert np.count_nonzero(placed.rgba[..., 3]) > 0


def test_temporal_view_respects_weighted_loop_timing() -> None:
    selection = select_action_frames((1, 3), loop_mode="loop", target_frame_count=4)

    assert selection.source_phases == (0.0, 0.25)
    assert selection.source_ordinals == (0, 1, 1, 0)


def test_temporal_view_retains_zero_tick_and_terminal_frames_for_one_shot() -> None:
    selection = select_action_frames((0, 4, -1), loop_mode="terminal_hold", target_frame_count=3)

    assert selection.loop_mode == "one_shot"
    assert selection.source_phases == (0.0, 0.2, 1.0)
    assert selection.source_ordinals == (0, 1, 2)


def test_static_action_repeats_only_in_derived_view() -> None:
    selection = select_action_frames((-1,), loop_mode="terminal_hold", target_frame_count=8)

    assert selection.source_frame_count == 1
    assert selection.source_ordinals == (0,) * 8
