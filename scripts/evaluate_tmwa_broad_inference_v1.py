"""Evaluate one matched held-out TMWA inference bundle against exact normalized targets."""

from __future__ import annotations

import argparse
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _images(rgba: np.ndarray) -> tuple[Image.Image, ...]:
    return tuple(Image.fromarray(frame) for frame in rgba)


def _pm(rgba: np.ndarray) -> np.ndarray:
    unit = rgba.astype(np.float32) / 255
    alpha = unit[..., 3:4]
    return np.concatenate((unit[..., :3] * alpha, alpha), axis=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inference = args.inference.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace evaluation: {output}")
    inference_report_path = inference / "inference-report.json"
    report_payload = inference_report_path.read_bytes()
    report = json.loads(report_payload)
    model = report["model_config"]
    corpus = prepare_broad_corpus(
        MANIFEST, target_size=model["height"], target_frames=model["num_frames"]
    )
    targets = {row.sequence_id: row for row in corpus.validation}
    source = {
        clip.sequence_id: clip
        for clip in load_materialized_training_clips(
            MANIFEST, split="validation", target_frames=model["num_frames"]
        )
    }
    samples = report["samples"]
    if len(samples) != len(corpus.validation):
        raise ValueError("inference count differs from validation split")
    ordered_ids = tuple(sorted(targets, key=str.encode))
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    rows: list[dict[str, object]] = []
    generated: dict[str, np.ndarray] = {}
    try:
        for index, (sample, sequence_id) in enumerate(zip(samples, ordered_ids, strict=True)):
            request = targets[sequence_id].request
            if sample["request"] != {
                "action": request.action,
                "description": request.description,
                "direction": request.direction,
                "entity_class": request.entity_class,
                "loop_mode": request.loop_mode,
                "view": request.view,
            }:
                raise ValueError(f"request order differs at {index}")
            sample_path = inference / sample["path"]
            if _sha(sample_path) != sample["file_sha256"]:
                raise ValueError(f"sample hash mismatch at {index}")
            prediction = np.load(sample_path, allow_pickle=False)
            target = targets[sequence_id]
            if prediction.shape != target.rgba.shape or prediction.dtype != np.uint8:
                raise ValueError(f"sample tensor mismatch at {index}")
            generated[sequence_id] = prediction
            metric = compare_matched_sequences(
                _images(prediction),
                _images(target.rgba),
                loop_mode=request.loop_mode,
                alpha_threshold=127,
            )
            rows.append(
                {
                    "action": request.action,
                    "alpha_iou_127": metric.alpha_iou,
                    "alpha_precision_127": metric.alpha_precision,
                    "alpha_recall_127": metric.alpha_recall,
                    "direction": request.direction,
                    "entity_class": request.entity_class,
                    "identity_id": target.identity_id,
                    "premultiplied_rgba_mae": metric.premultiplied_rgba_mae,
                    "sequence_id": sequence_id,
                    "target_background_premultiplied_rgba_mae": (
                        metric.target_background_premultiplied_rgba_mae
                    ),
                    "target_visible_premultiplied_rgba_mae": (
                        metric.target_visible_premultiplied_rgba_mae
                    ),
                    "temporal_delta_mae": metric.temporal_delta_mae,
                    "visible_canvas_ratio": metric.predicted_to_target_visible_canvas_ratio,
                }
            )
            stem = f"{index:02d}-{target.identity_id[-8:]}-{request.action}-{request.direction}"
            for role, rgba in (("target", target.rgba), ("generated", prediction)):
                export_rgba_clip_preview(
                    rgba,
                    stage / "previews",
                    artifact_stem=f"{stem}-{role}",
                    duration_ms=source[sequence_id].duration_ms,
                    loop_mode=request.loop_mode,
                    integer_scale=2,
                    source_report_sha256=hashlib.sha256(report_payload).hexdigest(),
                    preserve_frame_slots=True,
                )

        groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
        for target in corpus.validation:
            groups[
                (
                    target.identity_id,
                    target.request.direction,
                    target.request.view,
                    target.request.loop_mode,
                )
            ].append(target.sequence_id)
        preference_rows = []
        for ids in groups.values():
            if len({targets[value].action for value in ids}) < 2:
                continue
            for sequence_id in ids:
                correct = float(
                    np.mean(np.abs(_pm(generated[sequence_id]) - _pm(targets[sequence_id].rgba)))
                )
                alternatives = [
                    float(np.mean(np.abs(_pm(generated[sequence_id]) - _pm(targets[other].rgba))))
                    for other in ids
                    if targets[other].action != targets[sequence_id].action
                ]
                if alternatives:
                    preference_rows.append(
                        {
                            "correct_preferred": correct < min(alternatives),
                            "margin_alternative_minus_correct": min(alternatives) - correct,
                            "sequence_id": sequence_id,
                        }
                    )
        metric_names = (
            "premultiplied_rgba_mae",
            "target_visible_premultiplied_rgba_mae",
            "target_background_premultiplied_rgba_mae",
            "alpha_iou_127",
            "alpha_precision_127",
            "alpha_recall_127",
            "temporal_delta_mae",
            "visible_canvas_ratio",
        )
        artifact = {
            "action_preference": {
                "correct": sum(row["correct_preferred"] for row in preference_rows),
                "eligible": len(preference_rows),
                "rows": preference_rows,
            },
            "aggregate": {
                name: {
                    "mean": float(np.mean([row[name] for row in rows])),
                    "median": float(np.median([row[name] for row in rows])),
                }
                for name in metric_names
            },
            "artifact_kind": "held_out_identity_sprite_generation_evaluation",
            "claim": (
                "All evaluated identities are absent from training; this is indexed-distribution "
                "generalization, not open-vocabulary generation."
            ),
            "corpus_sha256": corpus.corpus_sha256,
            "inference_report": {
                "file_sha256": hashlib.sha256(report_payload).hexdigest(),
                "path": str(inference_report_path),
            },
            "metrics": rows,
            "schema_version": 1,
            "sequence_count": len(rows),
        }
        payload = (
            json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        (stage / "evaluation.json").write_bytes(payload)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        {
            "action_preference": f"{artifact['action_preference']['correct']}/"
            f"{artifact['action_preference']['eligible']}",
            "evaluation_sha256": hashlib.sha256(payload).hexdigest(),
            "pm_mae": artifact["aggregate"]["premultiplied_rgba_mae"]["mean"],
        }
    )


if __name__ == "__main__":
    main()
