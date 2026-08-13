"""Deterministic evaluation sidecar for tiny generated-sprite experiments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from spritelab.evaluation import compare_matched_sequences, evaluate_sprite_sequence
from spritelab.storage import DiskGuard
from spritelab.training_data import load_materialized_training_clips


def evaluate_overfit_experiment(
    experiment_directory: Path | str,
    materialization_manifest: Path | str,
    *,
    alpha_threshold: int = 0,
    output_path: Path | str | None = None,
    overwrite: bool = False,
    disk_guard: DiskGuard | None = None,
) -> Path:
    """Compare every declared generated sample to its exact training target.

    Visibility-based metrics use the explicit ``alpha > alpha_threshold``
    contract. Existing callers retain the original threshold-zero behavior.
    """

    root = Path(experiment_directory).resolve()
    report_path = root / "overfit-report.json"
    try:
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read overfit report: {report_path}") from error
    sample_records = report.get("sample_files")
    sequence_ids = report.get("sequence_ids")
    config = report.get("config")
    if not isinstance(sample_records, list) or not sample_records:
        raise ValueError("overfit report must declare sample_files")
    if not isinstance(sequence_ids, list) or not sequence_ids:
        raise ValueError("overfit report must declare sequence_ids")
    if not isinstance(config, dict):
        raise ValueError("overfit report must declare config")
    target_bucket = _positive_integer(config.get("target_bucket"), "config.target_bucket")
    target_frames = _positive_integer(config.get("target_frames"), "config.target_frames")
    manifest = Path(materialization_manifest).resolve()
    clips = load_materialized_training_clips(
        manifest,
        sequence_ids=tuple(sequence_ids),
        split="train",
        target_bucket=target_bucket,
        target_frames=target_frames,
    )
    by_sequence = {clip.sequence_id: clip for clip in clips}
    if set(by_sequence) != set(sequence_ids):
        raise ValueError("materialization manifest did not yield every reported sequence")
    evaluations: list[dict[str, object]] = []
    generated_by_sequence: dict[str, np.ndarray] = {}
    target_by_sequence: dict[str, np.ndarray] = {}
    for raw in sample_records:
        if not isinstance(raw, dict):
            raise ValueError("sample_files rows must be objects")
        sequence_id = _required_string(raw.get("sequence_id"), "sample.sequence_id")
        relative_path = Path(_required_string(raw.get("path"), "sample.path"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"unsafe sample path: {relative_path}")
        sample_path = (root / relative_path).resolve()
        try:
            sample_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"sample path escapes experiment directory: {relative_path}"
            ) from error
        sample_bytes = sample_path.read_bytes()
        declared_sha = _required_digest(raw.get("file_sha256"), "sample.file_sha256")
        actual_sha = hashlib.sha256(sample_bytes).hexdigest()
        if actual_sha != declared_sha:
            raise ValueError(
                f"sample SHA-256 mismatch for {sequence_id}: "
                f"expected {declared_sha}, got {actual_sha}"
            )
        try:
            generated = np.load(sample_path, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot load generated sample: {sample_path}") from error
        if not isinstance(generated, np.ndarray) or generated.dtype != np.uint8:
            raise ValueError(f"generated sample must be a uint8 array: {sample_path}")
        clip = by_sequence[sequence_id]
        if generated.shape != clip.rgba.shape:
            raise ValueError(
                f"sample/target shape mismatch for {sequence_id}: "
                f"{generated.shape!r} != {clip.rgba.shape!r}"
            )
        generated_frames = _images(generated)
        target_frames_images = _images(clip.rgba)
        generated_by_sequence[sequence_id] = generated
        target_by_sequence[sequence_id] = clip.rgba
        loop_mode = clip.request.loop_mode
        evaluations.append(
            {
                "action": clip.request.action,
                "generated": asdict(
                    evaluate_sprite_sequence(
                        generated_frames,
                        loop_mode=loop_mode,
                        alpha_threshold=alpha_threshold,
                    )
                ),
                "matched_target": asdict(
                    compare_matched_sequences(
                        generated_frames,
                        target_frames_images,
                        loop_mode=loop_mode,
                        alpha_threshold=alpha_threshold,
                    )
                ),
                "sample_file_sha256": actual_sha,
                "sequence_id": sequence_id,
                "source_materialized_array_sha256": clip.source_array_sha256,
                "source_materialized_file_sha256": clip.source_file_sha256,
                "training_target": {
                    "array_sha256": _array_sha256(clip.rgba),
                    "dtype": clip.rgba.dtype.name,
                    "intro_loop_projection": (
                        asdict(clip.intro_loop_projection)
                        if clip.intro_loop_projection is not None
                        else None
                    ),
                    "shape": list(clip.rgba.shape),
                    "temporal_duration_method": clip.temporal_duration_method,
                    "temporal_selection": (
                        asdict(clip.temporal_selection)
                        if clip.temporal_selection is not None
                        else None
                    ),
                },
            }
        )
    evaluations.sort(key=lambda row: str(row["sequence_id"]).encode("utf-8"))
    matched = [row["matched_target"] for row in evaluations]
    generated_metrics = [row["generated"] for row in evaluations]
    causal_pair_separation = _causal_action_pair_separation(
        report.get("matched_endpoint_contrast_plan"),
        generated_by_sequence=generated_by_sequence,
        target_by_sequence=target_by_sequence,
        clips_by_sequence=by_sequence,
    )
    aggregate_fields = (
        "premultiplied_rgba_mae",
        "alpha_mae",
        "alpha_iou",
        "composite_black_mae",
        "composite_white_mae",
        "alpha_centroid_error_px",
        "alpha_bbox_edge_mae_px",
        "temporal_delta_mae",
        "exact_frame_match_fraction",
        "alpha_precision",
        "alpha_recall",
        "target_visible_premultiplied_rgba_mae",
        "target_background_premultiplied_rgba_mae",
        "predicted_visible_canvas_fraction",
        "target_visible_canvas_fraction",
        "predicted_to_target_visible_canvas_ratio",
    )
    payload = {
        "aggregate_mean": {
            field: float(np.mean([float(row[field]) for row in matched]))
            for field in aggregate_fields
        },
        "alpha_visibility_threshold": alpha_threshold,
        "artifact_kind": "tiny_overfit_exact_target_evaluation",
        "schema_version": 2,
        "causal_action_pair_separation": causal_pair_separation,
        "experiment_report_path": str(report_path),
        "experiment_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "generated_alpha_crisp_fraction_mean": float(
            np.mean([float(row["alpha_crisp_fraction"]) for row in generated_metrics])
        ),
        "generated_translucent_visible_fraction_mean": float(
            np.mean([float(row["translucent_visible_fraction"]) for row in generated_metrics])
        ),
        "materialization_manifest_path": str(manifest),
        "materialization_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sample_count": len(evaluations),
        "samples": evaluations,
    }
    output = (
        Path(output_path).resolve()
        if output_path is not None
        else root / "exact-target-evaluation.json"
    )
    output_bytes = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(output, output_bytes, overwrite=overwrite, disk_guard=disk_guard)
    return output


def _causal_action_pair_separation(
    raw_plan: object,
    *,
    generated_by_sequence: dict[str, np.ndarray],
    target_by_sequence: dict[str, np.ndarray],
    clips_by_sequence: dict[str, object],
) -> list[dict[str, object]]:
    if not isinstance(raw_plan, dict):
        return []
    groups = raw_plan.get("groups")
    if not isinstance(groups, list):
        raise ValueError("matched_endpoint_contrast_plan.groups must be a list")
    rows: list[dict[str, object]] = []
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("endpoint contrast groups must be objects")
        group_sha = _required_digest(group.get("key_sha256"), "endpoint_contrast_group.key_sha256")
        sequence_ids = group.get("sequence_ids")
        if not isinstance(sequence_ids, list) or len(sequence_ids) < 2:
            raise ValueError("endpoint contrast group must contain at least two sequence IDs")
        for left_id, right_id in combinations(sequence_ids, 2):
            left_id = _required_string(left_id, "endpoint sequence ID")
            right_id = _required_string(right_id, "endpoint sequence ID")
            try:
                left_generated = generated_by_sequence[left_id]
                right_generated = generated_by_sequence[right_id]
                left_target = target_by_sequence[left_id]
                right_target = target_by_sequence[right_id]
                left_clip = clips_by_sequence[left_id]
                right_clip = clips_by_sequence[right_id]
            except KeyError as error:
                raise ValueError(
                    "endpoint contrast sequence is missing from experiment samples: "
                    f"{error.args[0]}"
                ) from error
            generated_distance = _premultiplied_distance(left_generated, right_generated)
            target_distance = _premultiplied_distance(left_target, right_target)
            rows.append(
                {
                    "contrast_group_sha256": group_sha,
                    "generated_distance": generated_distance,
                    "generated_to_target_distance_ratio": (
                        generated_distance / target_distance if target_distance > 0 else None
                    ),
                    "left_action": left_clip.request.action,
                    "left_sequence_id": left_id,
                    "right_action": right_clip.request.action,
                    "right_sequence_id": right_id,
                    "target_distance": target_distance,
                }
            )
    rows.sort(
        key=lambda row: (
            str(row["contrast_group_sha256"]),
            str(row["left_sequence_id"]).encode("utf-8"),
            str(row["right_sequence_id"]).encode("utf-8"),
        )
    )
    return rows


def _premultiplied_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("action-pair arrays must have identical shapes")
    left_normalized = left.astype(np.float32) / 255.0
    right_normalized = right.astype(np.float32) / 255.0
    left_alpha = left_normalized[..., 3:4]
    right_alpha = right_normalized[..., 3:4]
    left_pm = np.concatenate((left_normalized[..., :3] * left_alpha, left_alpha), axis=-1)
    right_pm = np.concatenate((right_normalized[..., :3] * right_alpha, right_alpha), axis=-1)
    return float(np.mean(np.abs(left_pm - right_pm)))


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _images(rgba: np.ndarray) -> tuple[Image.Image, ...]:
    return tuple(Image.fromarray(frame) for frame in rgba)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _required_digest(value: object, name: str) -> str:
    text = _required_string(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool,
    disk_guard: DiskGuard | None,
) -> None:
    if disk_guard is not None:
        disk_guard.require_capacity(len(payload) + 65_536, label="overfit evaluation")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(f"Refusing to replace existing evaluation: {path}") from error
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
