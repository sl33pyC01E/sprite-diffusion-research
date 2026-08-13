"""Evaluate and preview the exact held-out TMWA validation outputs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.broad_train import prepare_broad_corpus  # noqa: E402
from spritelab.evaluation import compare_matched_sequences  # noqa: E402
from spritelab.previews import export_rgba_clip_preview  # noqa: E402
from spritelab.training_data import load_materialized_training_clips  # noqa: E402

MANIFEST = ROOT / "data/processed/tmwa-model-ready-action-v1/materialization.json"
EXPERIMENT = ROOT / "data/experiments/tmwa-broad-heldout-b128-f8-v2-10000"
OUTPUT = ROOT / "data/inference/tmwa-broad-heldout-b128-f8-v2-10000-evaluation-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _images(rgba: np.ndarray) -> tuple[Image.Image, ...]:
    return tuple(Image.fromarray(frame) for frame in rgba)


def _pm_mae(left: np.ndarray, right: np.ndarray) -> float:
    def premultiply(value: np.ndarray) -> np.ndarray:
        unit = value.astype(np.float32) / 255.0
        alpha = unit[..., 3:4]
        return np.concatenate((unit[..., :3] * alpha, alpha), axis=-1)

    return float(np.mean(np.abs(premultiply(left) - premultiply(right))))


def _aggregate(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "maximum": max(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": min(values),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace held-out evaluation: {OUTPUT}")
    report_path = EXPERIMENT / "broad-training-report.json"
    report_payload = report_path.read_bytes()
    report_sha256 = hashlib.sha256(report_payload).hexdigest()
    report = json.loads(report_payload)
    if report.get("artifact_kind") != "identity_disjoint_minibatch_sprite_training":
        raise ValueError("unexpected broad-training report kind")
    config = report["config"]
    corpus = prepare_broad_corpus(
        MANIFEST,
        target_size=config["target_size"],
        target_frames=config["target_frames"],
    )
    if corpus.corpus_sha256 != report["corpus"]["corpus_sha256"]:
        raise ValueError("evaluation corpus differs from training report")
    source_clips = load_materialized_training_clips(
        MANIFEST, split="validation", target_frames=config["target_frames"]
    )
    source_by_id = {clip.sequence_id: clip for clip in source_clips}
    target_by_id = {row.sequence_id: row for row in corpus.validation}
    sample_records = report["final_validation"]["samples"]
    if {row["sequence_id"] for row in sample_records} != set(target_by_id):
        raise ValueError("validation sample IDs do not exactly match the held-out split")

    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent))
    try:
        previews = stage / "previews"
        metrics: list[dict[str, object]] = []
        generated_by_id: dict[str, np.ndarray] = {}
        for ordinal, sample in enumerate(sample_records):
            sequence_id = sample["sequence_id"]
            path = EXPERIMENT / "validation-samples" / sample["path"]
            if _sha256_file(path) != sample["file_sha256"]:
                raise ValueError(f"sample hash mismatch: {sequence_id}")
            generated = np.load(path, allow_pickle=False)
            target = target_by_id[sequence_id]
            if generated.dtype != np.uint8 or generated.shape != target.rgba.shape:
                raise ValueError(f"sample tensor contract mismatch: {sequence_id}")
            generated_by_id[sequence_id] = generated
            raw = compare_matched_sequences(
                _images(generated),
                _images(target.rgba),
                loop_mode=target.request.loop_mode,
                alpha_threshold=0,
            )
            alpha_127 = compare_matched_sequences(
                _images(generated),
                _images(target.rgba),
                loop_mode=target.request.loop_mode,
                alpha_threshold=127,
            )
            metrics.append(
                {
                    "action": target.action,
                    "alpha_iou_127": alpha_127.alpha_iou,
                    "alpha_precision_127": alpha_127.alpha_precision,
                    "alpha_recall_127": alpha_127.alpha_recall,
                    "direction": target.request.direction,
                    "entity_class": target.request.entity_class,
                    "identity_id": target.identity_id,
                    "premultiplied_rgba_mae": raw.premultiplied_rgba_mae,
                    "sequence_id": sequence_id,
                    "target_background_premultiplied_rgba_mae": (
                        raw.target_background_premultiplied_rgba_mae
                    ),
                    "target_visible_premultiplied_rgba_mae": (
                        raw.target_visible_premultiplied_rgba_mae
                    ),
                    "temporal_delta_mae": raw.temporal_delta_mae,
                    "visible_canvas_ratio": raw.predicted_to_target_visible_canvas_ratio,
                }
            )
            clip = source_by_id[sequence_id]
            stem = (
                f"{ordinal:02d}-{target.identity_id[-8:]}-"
                f"{target.action}-{target.request.direction}"
            )
            for role, rgba in (("target", target.rgba), ("generated", generated)):
                export_rgba_clip_preview(
                    rgba,
                    previews,
                    artifact_stem=f"{stem}-{role}",
                    duration_ms=clip.duration_ms,
                    loop_mode=target.request.loop_mode,
                    integer_scale=2,
                    source_sample_path=path if role == "generated" else None,
                    source_sample_sha256=sample["file_sha256"] if role == "generated" else None,
                    source_report_sha256=report_sha256,
                    preserve_frame_slots=True,
                )

        action_groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
        identity_groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
        for row in corpus.validation:
            action_groups[
                (
                    row.identity_id,
                    row.request.direction,
                    row.request.view,
                    row.request.loop_mode,
                )
            ].append(row.sequence_id)
            identity_groups[
                (
                    row.action,
                    row.request.direction,
                    row.request.view,
                    row.request.loop_mode,
                )
            ].append(row.sequence_id)

        action_preferences = []
        for ids in action_groups.values():
            if len({target_by_id[value].action for value in ids}) < 2:
                continue
            for sequence_id in ids:
                generated = generated_by_id[sequence_id]
                correct = _pm_mae(generated, target_by_id[sequence_id].rgba)
                alternatives = [
                    _pm_mae(generated, target_by_id[other].rgba)
                    for other in ids
                    if target_by_id[other].action != target_by_id[sequence_id].action
                ]
                if alternatives:
                    best_alternative = min(alternatives)
                    action_preferences.append(
                        {
                            "correct_preferred": correct < best_alternative,
                            "correct_target_mae": correct,
                            "identity_id": target_by_id[sequence_id].identity_id,
                            "margin_alternative_minus_correct": best_alternative - correct,
                            "sequence_id": sequence_id,
                        }
                    )

        identity_preferences = []
        for ids in identity_groups.values():
            if len({target_by_id[value].identity_id for value in ids}) < 2:
                continue
            for sequence_id in ids:
                generated = generated_by_id[sequence_id]
                correct = _pm_mae(generated, target_by_id[sequence_id].rgba)
                alternatives = [
                    _pm_mae(generated, target_by_id[other].rgba)
                    for other in ids
                    if target_by_id[other].identity_id != target_by_id[sequence_id].identity_id
                ]
                if alternatives:
                    best_alternative = min(alternatives)
                    identity_preferences.append(
                        {
                            "correct_preferred": correct < best_alternative,
                            "correct_target_mae": correct,
                            "identity_id": target_by_id[sequence_id].identity_id,
                            "margin_alternative_minus_correct": best_alternative - correct,
                            "sequence_id": sequence_id,
                        }
                    )

        artifact = {
            "aggregate": {
                key: _aggregate(metrics, key)
                for key in (
                    "premultiplied_rgba_mae",
                    "target_visible_premultiplied_rgba_mae",
                    "target_background_premultiplied_rgba_mae",
                    "alpha_iou_127",
                    "alpha_precision_127",
                    "alpha_recall_127",
                    "temporal_delta_mae",
                    "visible_canvas_ratio",
                )
            },
            "artifact_kind": "held_out_identity_sprite_generation_evaluation",
            "claim": (
                "Every target identity is excluded from training. This measures interpolation "
                "within the indexed TMWA distribution; it is not open-vocabulary evidence."
            ),
            "action_preference": {
                "correct": sum(row["correct_preferred"] for row in action_preferences),
                "eligible": len(action_preferences),
                "rows": action_preferences,
            },
            "broad_training_report": {
                "file_sha256": report_sha256,
                "path": str(report_path),
            },
            "corpus_sha256": corpus.corpus_sha256,
            "identity_preference": {
                "correct": sum(row["correct_preferred"] for row in identity_preferences),
                "eligible": len(identity_preferences),
                "rows": identity_preferences,
            },
            "metrics": metrics,
            "preview_policy": {
                "all_held_out_sequences": True,
                "display_scale": 2,
                "resampling": "nearest_positive_integer",
                "target_and_generated": True,
            },
            "schema_version": 1,
            "sequence_count": len(metrics),
        }
        payload = (
            json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        report_output = stage / "evaluation.json"
        report_output.write_bytes(payload)
        os.rename(stage, OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        {
            "evaluation": str(OUTPUT / "evaluation.json"),
            "evaluation_sha256": hashlib.sha256(payload).hexdigest(),
            "sequence_count": len(metrics),
        }
    )


if __name__ == "__main__":
    main()
