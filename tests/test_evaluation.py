from __future__ import annotations

import math

import pytest
from PIL import Image

from spritelab.evaluation import (
    compare_matched_sequences,
    evaluate_sprite_sequence,
    nearest_sequence_match,
)


def _frame(
    *,
    size: tuple[int, int] = (4, 4),
    pixel: tuple[int, int] | None = None,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
) -> Image.Image:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    if pixel is not None:
        image.putpixel(pixel, color)
    return image


def test_sequence_metrics_ignore_hidden_rgb_and_measure_motion() -> None:
    first = _frame(pixel=(0, 1))
    second = _frame(pixel=(1, 1))
    hidden_rgb = second.copy()
    hidden_rgb.putpixel((3, 3), (12, 34, 56, 0))

    metrics = evaluate_sprite_sequence(
        (first, hidden_rgb, first.copy()),
        loop_mode="loop",
    )

    assert metrics.frame_count == 3
    assert metrics.union_palette_size == 1
    assert metrics.mean_adjacent_palette_jaccard_distance == 0
    assert metrics.exact_unique_frame_ratio == pytest.approx(2 / 3)
    assert metrics.dynamic_transition_fraction == 1
    assert metrics.alpha_centroid_jitter_px > 0
    assert metrics.alpha_bbox_size_jitter_px == 0
    assert metrics.loop_seam_premultiplied_rgba_l1 == 0
    assert metrics.loop_seam_alpha_l1 == 0


def test_alpha_crispness_and_translucency_are_explicit() -> None:
    frame = _frame(pixel=(0, 0), color=(20, 40, 60, 128))
    metrics = evaluate_sprite_sequence((frame,), loop_mode="one_shot")

    assert metrics.visible_canvas_fraction == pytest.approx(1 / 16)
    assert metrics.translucent_visible_fraction == 1
    assert metrics.alpha_crisp_fraction == pytest.approx(15 / 16)
    assert metrics.loop_seam_alpha_l1 is None
    assert metrics.dynamic_transition_fraction == 0


def test_identical_matched_sequences_have_perfect_metrics() -> None:
    frames = (_frame(pixel=(0, 0)), _frame(pixel=(1, 0)))
    metrics = compare_matched_sequences(
        frames, tuple(frame.copy() for frame in frames), loop_mode="loop"
    )

    assert metrics.premultiplied_rgba_mae == 0
    assert metrics.alpha_mae == 0
    assert metrics.alpha_iou == 1
    assert metrics.composite_black_mae == 0
    assert metrics.composite_white_mae == 0
    assert metrics.alpha_centroid_error_px == 0
    assert metrics.alpha_bbox_edge_mae_px == 0
    assert metrics.temporal_delta_mae == 0
    assert metrics.loop_delta_mae == 0
    assert metrics.exact_frame_match_fraction == 1
    assert metrics.alpha_visibility_threshold == 0
    assert metrics.alpha_precision == 1
    assert metrics.alpha_recall == 1
    assert metrics.target_visible_premultiplied_rgba_mae == 0
    assert metrics.target_background_premultiplied_rgba_mae == 0
    assert metrics.predicted_visible_canvas_fraction == pytest.approx(1 / 16)
    assert metrics.target_visible_canvas_fraction == pytest.approx(1 / 16)
    assert metrics.predicted_to_target_visible_canvas_ratio == 1


def test_matched_metrics_detect_shift_and_alpha_error() -> None:
    reference = (_frame(pixel=(0, 0)),)
    prediction = (_frame(pixel=(1, 0), color=(255, 0, 0, 128)),)
    metrics = compare_matched_sequences(prediction, reference, loop_mode="one_shot")

    assert metrics.alpha_iou == 0
    assert metrics.alpha_mae > 0
    assert metrics.alpha_centroid_error_px == 1
    assert metrics.alpha_bbox_edge_mae_px == pytest.approx(0.5)
    assert metrics.temporal_delta_mae == 0
    assert metrics.loop_delta_mae is None
    assert metrics.exact_frame_match_fraction == 0


def test_empty_alpha_iou_is_one_and_missing_subject_uses_canvas_penalty() -> None:
    empty = (_frame(),)
    assert compare_matched_sequences(empty, empty, loop_mode="unknown").alpha_iou == 1

    visible = (_frame(pixel=(1, 1)),)
    metrics = compare_matched_sequences(empty, visible, loop_mode="unknown")
    assert metrics.alpha_centroid_error_px == pytest.approx(math.hypot(4, 4))
    assert metrics.alpha_bbox_edge_mae_px == 4


def test_sparse_target_metrics_define_empty_and_one_sided_masks() -> None:
    empty = (_frame(),)
    both_empty = compare_matched_sequences(empty, empty, loop_mode="unknown")
    assert both_empty.alpha_precision == 1
    assert both_empty.alpha_recall == 1
    assert both_empty.predicted_to_target_visible_canvas_ratio == 1
    assert both_empty.predicted_visible_canvas_fraction == 0
    assert both_empty.target_visible_canvas_fraction == 0
    assert both_empty.target_visible_premultiplied_rgba_mae == 0
    assert both_empty.target_background_premultiplied_rgba_mae == 0

    visible = (_frame(pixel=(0, 0)),)
    prediction_only = compare_matched_sequences(visible, empty, loop_mode="unknown")
    assert prediction_only.alpha_precision == 0
    assert prediction_only.alpha_recall == 1
    assert prediction_only.predicted_to_target_visible_canvas_ratio == 0
    assert prediction_only.predicted_visible_canvas_fraction == pytest.approx(1 / 16)
    assert prediction_only.target_visible_canvas_fraction == 0
    assert prediction_only.target_visible_premultiplied_rgba_mae == 0
    assert prediction_only.target_background_premultiplied_rgba_mae == pytest.approx(1 / 32)

    reference_only = compare_matched_sequences(empty, visible, loop_mode="unknown")
    assert reference_only.alpha_precision == 0
    assert reference_only.alpha_recall == 0
    assert reference_only.predicted_to_target_visible_canvas_ratio == 0
    assert reference_only.predicted_visible_canvas_fraction == 0
    assert reference_only.target_visible_canvas_fraction == pytest.approx(1 / 16)
    assert reference_only.target_visible_premultiplied_rgba_mae == pytest.approx(0.5)
    assert reference_only.target_background_premultiplied_rgba_mae == 0


def test_sparse_target_metrics_use_explicit_threshold_and_absent_region_zero() -> None:
    prediction = (_frame(pixel=(0, 0), color=(255, 0, 0, 128)),)
    reference = (_frame(pixel=(0, 0), color=(255, 0, 0, 127)),)
    metrics = compare_matched_sequences(
        prediction,
        reference,
        loop_mode="unknown",
        alpha_threshold=127,
    )

    assert metrics.alpha_visibility_threshold == 127
    assert metrics.alpha_precision == 0
    assert metrics.alpha_recall == 1
    assert metrics.predicted_to_target_visible_canvas_ratio == 0
    assert metrics.target_visible_premultiplied_rgba_mae == 0
    assert metrics.target_background_premultiplied_rgba_mae > 0

    full_reference = (Image.new("RGBA", (2, 2), (255, 0, 0, 255)),)
    full_prediction = (Image.new("RGBA", (2, 2), (0, 0, 255, 255)),)
    full_metrics = compare_matched_sequences(
        full_prediction,
        full_reference,
        loop_mode="unknown",
    )
    assert full_metrics.target_visible_premultiplied_rgba_mae == pytest.approx(0.5)
    assert full_metrics.target_background_premultiplied_rgba_mae == 0


def test_nearest_sequence_match_is_deterministic_and_detects_exact_copy() -> None:
    query = (_frame(pixel=(0, 0)), _frame(pixel=(1, 0)))
    different = (_frame(pixel=(2, 0)), _frame(pixel=(3, 0)))
    exact = tuple(frame.copy() for frame in query)

    match = nearest_sequence_match(
        query,
        (
            ("far", different),
            ("z-exact", exact),
            ("a-exact", exact),
            ("wrong-shape", (_frame(size=(8, 8), pixel=(0, 0)),)),
        ),
    )

    assert match.candidate_id == "a-exact"
    assert match.distance == 0
    assert match.exact_match is True


def test_evaluation_refuses_implicit_alignment_and_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="identical canvas"):
        evaluate_sprite_sequence((_frame(), _frame(size=(5, 4))), loop_mode="loop")
    with pytest.raises(ValueError, match="identical"):
        compare_matched_sequences((_frame(),), (_frame(size=(8, 8)),), loop_mode="loop")
    with pytest.raises(TypeError, match="integer"):
        evaluate_sprite_sequence((_frame(),), loop_mode="loop", alpha_threshold=True)
    with pytest.raises(ValueError, match="between"):
        evaluate_sprite_sequence((_frame(),), loop_mode="loop", alpha_threshold=255)
    with pytest.raises(TypeError, match="integer"):
        compare_matched_sequences(
            (_frame(),),
            (_frame(),),
            loop_mode="loop",
            alpha_threshold=True,
        )
    with pytest.raises(ValueError, match="no candidate"):
        nearest_sequence_match((_frame(),), (("other", (_frame(size=(2, 2)),)),))
