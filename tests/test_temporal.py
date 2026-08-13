from __future__ import annotations

import pytest

from spritelab.temporal import apply_temporal_selection, select_temporal_frames


def test_loop_selection_uses_cyclic_phase_and_records_rotation() -> None:
    selection = select_temporal_frames(
        4,
        8,
        loop_mode="loop",
        phase_offset=0.25,
    )

    assert selection.target_phases == (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.0, 0.125)
    assert selection.source_ordinals == (1, 1, 2, 2, 3, 0, 0, 0)
    assert apply_temporal_selection(("a", "b", "c", "d"), selection) == (
        "b",
        "b",
        "c",
        "c",
        "d",
        "a",
        "a",
        "a",
    )
    assert len(selection.sha256) == 64


def test_one_shot_selection_includes_both_authored_endpoints() -> None:
    selection = select_temporal_frames(3, 5, loop_mode="one_shot")

    assert selection.target_phases == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert selection.source_ordinals == (0, 0, 1, 1, 2)
    assert selection.source_ordinals[0] == 0
    assert selection.source_ordinals[-1] == 2


def test_ping_pong_selection_covers_forward_and_backward_cycle() -> None:
    selection = select_temporal_frames(3, 8, loop_mode="ping_pong")

    assert selection.target_phases == (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
    assert selection.source_ordinals == (0, 0, 1, 1, 2, 1, 1, 0)


def test_explicit_source_phases_control_selection_without_interpolation() -> None:
    selection = select_temporal_frames(
        3,
        4,
        loop_mode="one_shot",
        source_phases=(0.0, 0.8, 1.0),
    )

    assert selection.source_ordinals == (0, 0, 1, 2)


def test_static_pose_can_fill_a_temporal_batch_explicitly() -> None:
    selection = select_temporal_frames(1, 4, loop_mode="loop")

    assert selection.source_ordinals == (0, 0, 0, 0)
    assert apply_temporal_selection(("pose",), selection) == ("pose",) * 4


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"source_frame_count": 0, "target_frame_count": 8, "loop_mode": "loop"}, "positive"),
        ({"source_frame_count": 2, "target_frame_count": 8, "loop_mode": "unknown"}, "loop_mode"),
        (
            {
                "source_frame_count": 2,
                "target_frame_count": 8,
                "loop_mode": "one_shot",
                "phase_offset": 0.25,
            },
            "must be zero",
        ),
        (
            {
                "source_frame_count": 2,
                "target_frame_count": 8,
                "loop_mode": "loop",
                "source_phases": (0.0, 1.0),
            },
            r"\[0, 1\)",
        ),
        (
            {
                "source_frame_count": 2,
                "target_frame_count": 8,
                "loop_mode": "one_shot",
                "source_phases": (0.5, 0.25),
            },
            "nondecreasing",
        ),
    ],
)
def test_invalid_temporal_contracts_fail(arguments: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        select_temporal_frames(**arguments)  # type: ignore[arg-type]


def test_selection_rejects_wrong_source_sequence_length() -> None:
    selection = select_temporal_frames(2, 4, loop_mode="loop")

    with pytest.raises(ValueError, match="length"):
        apply_temporal_selection(("only-one",), selection)
