"""Pixel- and motion-aware metrics for transparent animated sprites.

The functions in this module never resize, quantize, or otherwise regularize
their inputs.  Pair metrics therefore require frame-for-frame, pixel-for-pixel
matched canvases.  This makes a failed comparison an explicit data-contract
error instead of silently measuring interpolation artifacts.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class SpriteSequenceMetrics:
    frame_count: int
    width: int
    height: int
    visible_canvas_fraction: float
    translucent_visible_fraction: float
    alpha_crisp_fraction: float
    union_palette_size: int
    mean_frame_palette_size: float
    mean_adjacent_palette_jaccard_distance: float
    exact_unique_frame_ratio: float
    mean_temporal_premultiplied_rgba_l1: float
    mean_temporal_alpha_l1: float
    dynamic_transition_fraction: float
    alpha_centroid_jitter_px: float
    alpha_bbox_size_jitter_px: float
    loop_seam_premultiplied_rgba_l1: float | None
    loop_seam_alpha_l1: float | None


@dataclass(frozen=True, slots=True)
class MatchedSequenceMetrics:
    frame_count: int
    width: int
    height: int
    premultiplied_rgba_mae: float
    alpha_mae: float
    alpha_iou: float
    composite_black_mae: float
    composite_white_mae: float
    alpha_centroid_error_px: float
    alpha_bbox_edge_mae_px: float
    temporal_delta_mae: float
    loop_delta_mae: float | None
    exact_frame_match_fraction: float
    alpha_visibility_threshold: int = 0
    alpha_precision: float = 1.0
    alpha_recall: float = 1.0
    target_visible_premultiplied_rgba_mae: float = 0.0
    target_background_premultiplied_rgba_mae: float = 0.0
    predicted_visible_canvas_fraction: float = 0.0
    target_visible_canvas_fraction: float = 0.0
    predicted_to_target_visible_canvas_ratio: float = 1.0


@dataclass(frozen=True, slots=True)
class NearestSequenceMatch:
    candidate_id: str
    distance: float
    exact_match: bool


def evaluate_sprite_sequence(
    frames: Sequence[Image.Image],
    *,
    loop_mode: str,
    alpha_threshold: int = 0,
    dynamic_threshold: float = 0.0,
) -> SpriteSequenceMetrics:
    """Measure one sequence without making assumptions about its identity or action.

    Pixel distances are in normalized ``[0, 1]`` units. RGB is premultiplied by
    alpha before temporal comparison so arbitrary RGB values under fully
    transparent pixels cannot masquerade as motion.
    """

    rgba = _rgba_stack(frames)
    _validate_alpha_threshold(alpha_threshold)
    if not math.isfinite(dynamic_threshold) or dynamic_threshold < 0:
        raise ValueError("dynamic_threshold must be finite and non-negative")

    frame_count, height, width, _channels = rgba.shape
    alpha_u8 = rgba[..., 3]
    visible = alpha_u8 > alpha_threshold
    visible_count = int(visible.sum())
    total_pixels = int(visible.size)
    translucent = visible & (alpha_u8 < 255)
    crisp = (alpha_u8 == 0) | (alpha_u8 == 255)

    palettes = tuple(_visible_palette(frame, alpha_threshold) for frame in rgba)
    union_palette = set().union(*palettes)
    palette_distances = tuple(
        _jaccard_distance(left, right) for left, right in zip(palettes, palettes[1:], strict=False)
    )

    premultiplied = _premultiplied_rgba(rgba)
    alpha = alpha_u8.astype(np.float32) / 255.0
    temporal_rgba = _transition_l1(premultiplied)
    temporal_alpha = _transition_l1(alpha)
    hashes = tuple(_frame_digest(frame) for frame in rgba)

    centroids = tuple(_alpha_centroid(frame_alpha) for frame_alpha in alpha_u8)
    bbox_sizes = tuple(_alpha_bbox_size(frame_alpha, alpha_threshold) for frame_alpha in alpha_u8)
    loop = loop_mode == "loop"

    return SpriteSequenceMetrics(
        frame_count=frame_count,
        width=width,
        height=height,
        visible_canvas_fraction=visible_count / total_pixels,
        translucent_visible_fraction=(
            float(translucent.sum()) / visible_count if visible_count else 0.0
        ),
        alpha_crisp_fraction=float(crisp.mean()),
        union_palette_size=len(union_palette),
        mean_frame_palette_size=float(np.mean([len(palette) for palette in palettes])),
        mean_adjacent_palette_jaccard_distance=(
            float(np.mean(palette_distances)) if palette_distances else 0.0
        ),
        exact_unique_frame_ratio=len(set(hashes)) / frame_count,
        mean_temporal_premultiplied_rgba_l1=(
            float(np.mean(temporal_rgba)) if temporal_rgba else 0.0
        ),
        mean_temporal_alpha_l1=(float(np.mean(temporal_alpha)) if temporal_alpha else 0.0),
        dynamic_transition_fraction=(
            sum(distance > dynamic_threshold for distance in temporal_rgba) / len(temporal_rgba)
            if temporal_rgba
            else 0.0
        ),
        alpha_centroid_jitter_px=_point_jitter(centroids),
        alpha_bbox_size_jitter_px=_point_jitter(bbox_sizes),
        loop_seam_premultiplied_rgba_l1=(
            _array_l1(premultiplied[-1], premultiplied[0]) if loop else None
        ),
        loop_seam_alpha_l1=(_array_l1(alpha[-1], alpha[0]) if loop else None),
    )


def compare_matched_sequences(
    prediction: Sequence[Image.Image],
    reference: Sequence[Image.Image],
    *,
    loop_mode: str,
    alpha_threshold: int = 0,
) -> MatchedSequenceMetrics:
    """Compare two already-aligned sequences with identical frame/canvas shapes.

    Alpha visibility is defined as ``alpha > alpha_threshold``. When both masks
    are empty, precision, recall, IoU, and the visible-canvas ratio are 1. When
    only the reference is visible, precision, recall, and the ratio are 0. When
    only the prediction is visible, precision and the ratio are 0 while recall
    is 1 because there are no reference-positive pixels to miss. A target-region
    premultiplied error is 0 when that target region contains no pixels.
    """

    predicted = _rgba_stack(prediction)
    expected = _rgba_stack(reference)
    _validate_alpha_threshold(alpha_threshold)
    if predicted.shape != expected.shape:
        raise ValueError(
            "prediction and reference must have identical [T, H, W, RGBA] shapes; "
            f"got {predicted.shape!r} and {expected.shape!r}"
        )

    frame_count, height, width, _channels = predicted.shape
    predicted_pm = _premultiplied_rgba(predicted)
    expected_pm = _premultiplied_rgba(expected)
    predicted_alpha = predicted[..., 3].astype(np.float32) / 255.0
    expected_alpha = expected[..., 3].astype(np.float32) / 255.0
    predicted_mask = predicted[..., 3] > alpha_threshold
    expected_mask = expected[..., 3] > alpha_threshold
    predicted_visible_count = int(predicted_mask.sum())
    expected_visible_count = int(expected_mask.sum())
    union = int(np.logical_or(predicted_mask, expected_mask).sum())
    intersection = int(np.logical_and(predicted_mask, expected_mask).sum())
    total_pixels = int(predicted_mask.size)
    alpha_precision = (
        intersection / predicted_visible_count
        if predicted_visible_count
        else (1.0 if expected_visible_count == 0 else 0.0)
    )
    alpha_recall = intersection / expected_visible_count if expected_visible_count else 1.0
    predicted_to_target_visible_canvas_ratio = (
        predicted_visible_count / expected_visible_count
        if expected_visible_count
        else (1.0 if predicted_visible_count == 0 else 0.0)
    )

    predicted_black = _composite(predicted, background=0.0)
    expected_black = _composite(expected, background=0.0)
    predicted_white = _composite(predicted, background=1.0)
    expected_white = _composite(expected, background=1.0)

    centroid_errors = [
        _paired_point_error(
            _alpha_centroid(predicted[index, ..., 3]),
            _alpha_centroid(expected[index, ..., 3]),
            missing_error=math.hypot(width, height),
        )
        for index in range(frame_count)
    ]
    bbox_errors = [
        _bbox_edge_error(
            _alpha_bbox(predicted[index, ..., 3], alpha_threshold),
            _alpha_bbox(expected[index, ..., 3], alpha_threshold),
            missing_error=float(max(width, height)),
        )
        for index in range(frame_count)
    ]

    loop = loop_mode == "loop"
    predicted_deltas = _temporal_deltas(predicted_pm, include_loop_edge=loop)
    expected_deltas = _temporal_deltas(expected_pm, include_loop_edge=loop)
    temporal_delta_mae = (
        float(np.mean(np.abs(predicted_deltas - expected_deltas))) if predicted_deltas.size else 0.0
    )
    exact_matches = sum(
        bool(np.array_equal(left, right)) for left, right in zip(predicted, expected, strict=True)
    )

    return MatchedSequenceMetrics(
        frame_count=frame_count,
        width=width,
        height=height,
        premultiplied_rgba_mae=float(np.mean(np.abs(predicted_pm - expected_pm))),
        alpha_mae=float(np.mean(np.abs(predicted_alpha - expected_alpha))),
        alpha_iou=intersection / union if union else 1.0,
        composite_black_mae=float(np.mean(np.abs(predicted_black - expected_black))),
        composite_white_mae=float(np.mean(np.abs(predicted_white - expected_white))),
        alpha_centroid_error_px=float(np.mean(centroid_errors)),
        alpha_bbox_edge_mae_px=float(np.mean(bbox_errors)),
        temporal_delta_mae=temporal_delta_mae,
        loop_delta_mae=(
            float(
                np.mean(
                    np.abs(
                        (predicted_pm[0] - predicted_pm[-1]) - (expected_pm[0] - expected_pm[-1])
                    )
                )
            )
            if loop
            else None
        ),
        exact_frame_match_fraction=exact_matches / frame_count,
        alpha_visibility_threshold=alpha_threshold,
        alpha_precision=alpha_precision,
        alpha_recall=alpha_recall,
        target_visible_premultiplied_rgba_mae=_masked_array_l1(
            predicted_pm, expected_pm, expected_mask
        ),
        target_background_premultiplied_rgba_mae=_masked_array_l1(
            predicted_pm, expected_pm, ~expected_mask
        ),
        predicted_visible_canvas_fraction=predicted_visible_count / total_pixels,
        target_visible_canvas_fraction=expected_visible_count / total_pixels,
        predicted_to_target_visible_canvas_ratio=predicted_to_target_visible_canvas_ratio,
    )


def nearest_sequence_match(
    query: Sequence[Image.Image],
    candidates: Sequence[tuple[str, Sequence[Image.Image]]],
) -> NearestSequenceMatch:
    """Return the closest shape-compatible sequence by premultiplied RGBA L1.

    Shape-incompatible candidates are ignored. This is an exact-canvas
    memorization diagnostic, not a perceptual retrieval metric.
    """

    query_rgba = _rgba_stack(query)
    query_pm = _premultiplied_rgba(query_rgba)
    if not candidates:
        raise ValueError("at least one candidate sequence is required")
    matches: list[NearestSequenceMatch] = []
    seen_ids: set[str] = set()
    for candidate_id, frames in candidates:
        if not candidate_id:
            raise ValueError("candidate IDs cannot be empty")
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        seen_ids.add(candidate_id)
        candidate_rgba = _rgba_stack(frames)
        if candidate_rgba.shape != query_rgba.shape:
            continue
        distance = float(np.mean(np.abs(query_pm - _premultiplied_rgba(candidate_rgba))))
        matches.append(
            NearestSequenceMatch(
                candidate_id=candidate_id,
                distance=distance,
                exact_match=bool(np.array_equal(query_rgba, candidate_rgba)),
            )
        )
    if not matches:
        raise ValueError("no candidate sequence has the query shape")
    return min(matches, key=lambda match: (match.distance, match.candidate_id.encode("utf-8")))


def _rgba_stack(frames: Sequence[Image.Image]) -> np.ndarray:
    if not frames:
        raise ValueError("at least one frame is required")
    size = frames[0].size
    if size[0] < 1 or size[1] < 1:
        raise ValueError("frames must have non-zero dimensions")
    if any(frame.size != size for frame in frames):
        raise ValueError("all frames in a sequence must have identical canvas dimensions")
    return np.stack(
        [np.asarray(frame.convert("RGBA"), dtype=np.uint8) for frame in frames],
        axis=0,
    )


def _validate_alpha_threshold(alpha_threshold: int) -> None:
    if not isinstance(alpha_threshold, int) or isinstance(alpha_threshold, bool):
        raise TypeError("alpha_threshold must be an integer")
    if not 0 <= alpha_threshold <= 254:
        raise ValueError("alpha_threshold must be between 0 and 254")


def _premultiplied_rgba(rgba: np.ndarray) -> np.ndarray:
    normalized = rgba.astype(np.float32) / 255.0
    alpha = normalized[..., 3:4]
    return np.concatenate((normalized[..., :3] * alpha, alpha), axis=-1)


def _composite(rgba: np.ndarray, *, background: float) -> np.ndarray:
    normalized = rgba.astype(np.float32) / 255.0
    alpha = normalized[..., 3:4]
    return normalized[..., :3] * alpha + background * (1.0 - alpha)


def _array_l1(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def _masked_array_l1(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(np.mean(np.abs(left[mask] - right[mask])))


def _transition_l1(values: np.ndarray) -> tuple[float, ...]:
    return tuple(_array_l1(left, right) for left, right in zip(values, values[1:], strict=False))


def _temporal_deltas(values: np.ndarray, *, include_loop_edge: bool) -> np.ndarray:
    if len(values) <= 1:
        return np.empty((0, *values.shape[1:]), dtype=np.float32)
    deltas = values[1:] - values[:-1]
    if include_loop_edge:
        deltas = np.concatenate((deltas, (values[0] - values[-1])[None, ...]), axis=0)
    return deltas


def _visible_palette(frame: np.ndarray, alpha_threshold: int) -> set[bytes]:
    visible = frame[frame[..., 3] > alpha_threshold]
    return {bytes(color) for color in visible}


def _jaccard_distance(left: set[bytes], right: set[bytes]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def _frame_digest(frame: np.ndarray) -> str:
    height, width, channels = frame.shape
    header = f"{width}x{height}x{channels}\0".encode()
    return hashlib.sha256(header + frame.tobytes()).hexdigest()


def _alpha_centroid(alpha: np.ndarray) -> tuple[float, float] | None:
    weights = alpha.astype(np.float64) / 255.0
    total = float(weights.sum())
    if total == 0:
        return None
    y, x = np.indices(weights.shape)
    return float((x * weights).sum() / total), float((y * weights).sum() / total)


def _alpha_bbox(alpha: np.ndarray, threshold: int) -> tuple[int, int, int, int] | None:
    y, x = np.nonzero(alpha > threshold)
    if not len(x):
        return None
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def _alpha_bbox_size(alpha: np.ndarray, threshold: int) -> tuple[float, float] | None:
    bbox = _alpha_bbox(alpha, threshold)
    if bbox is None:
        return None
    return float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])


def _point_jitter(points: Sequence[tuple[float, float] | None]) -> float:
    valid = np.asarray([point for point in points if point is not None], dtype=np.float64)
    if len(valid) <= 1:
        return 0.0
    center = valid.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(np.square(valid - center), axis=1))))


def _paired_point_error(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
    *,
    missing_error: float,
) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return missing_error
    return math.dist(left, right)


def _bbox_edge_error(
    left: tuple[int, int, int, int] | None,
    right: tuple[int, int, int, int] | None,
    *,
    missing_error: float,
) -> float:
    if left is None and right is None:
        return 0.0
    if left is None or right is None:
        return missing_error
    return (
        sum(abs(left_edge - right_edge) for left_edge, right_edge in zip(left, right, strict=True))
        / 4
    )
