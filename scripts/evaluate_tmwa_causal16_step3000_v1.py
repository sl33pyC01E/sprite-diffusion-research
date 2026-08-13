"""Evaluate the exact matched TMWA step-3,000 inference and causal action swaps."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spritelab.evaluation import compare_matched_sequences, evaluate_sprite_sequence  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import load_materialized_training_clips  # noqa: E402

MANIFEST = (
    REPOSITORY / "data/processed/tmwa-model-ready-action-v1/"
    "tmwa-causal-down-idle-walk-16-materialization-v1.json"
)
EXPERIMENT = REPOSITORY / "data/experiments/tmwa-down-idle-walk-causal16-b64-f8-baseline-v1-3000"
SOURCE_REPORT = EXPERIMENT / "overfit-report.json"
OUTPUT = (
    REPOSITORY / "data/index/reports/tmwa-causal16-b64-f8-baseline3000-matched-evaluation-v1.json"
)
EXPECTED_MANIFEST_SHA256 = "095205d49811a0102368d607a5bcb8dd9a0c2b057ebe33ddaeeda893bede45ca"
EXPECTED_SOURCE_REPORT_SHA256 = "0ea6ca38c1d5fff6d8680efa5e65540b3df7e6526b52e5b42ffaabaf929fc415"
INFERENCE_REPORTS = {
    "endpoint_exact": (
        REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-endpoint-exact-v1/"
        "inference-report.json",
        "4b5fc533b24ca99f35b3b08785eb636441087a8236fc217666229e51f537d9ce",
    ),
    "endpoint_swap": (
        REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-endpoint-action-swap-v1/"
        "inference-report.json",
        "de01efd85540337997a108326d7c7e2d7f5b398e2b7b5c811d8110df0b2a6ef3",
    ),
    "euler_exact": (
        REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-euler32-exact-v1/"
        "inference-report.json",
        "6e6c5be6a23869d8f53fea4b72d0ab3af50a871b7b9e7487229d760a37d0a72f",
    ),
    "euler_swap": (
        REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-euler32-action-swap-v1/"
        "inference-report.json",
        "55e4d8c2f4ba256e8877586d0e5e303545663f394e6e52e161d081a1ba9c875b",
    ),
}
AGGREGATE_FIELDS = (
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _images(array: np.ndarray) -> tuple[Image.Image, ...]:
    return tuple(Image.fromarray(frame, mode="RGBA") for frame in array)


def _distance(left: np.ndarray, right: np.ndarray, *, loop_mode: str) -> float:
    return compare_matched_sequences(
        _images(left),
        _images(right),
        loop_mode=loop_mode,
    ).premultiplied_rgba_mae


def _load_inference(
    report_path: Path,
    expected_report_sha256: str,
    sequence_ids: tuple[str, ...],
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    report_bytes = report_path.read_bytes()
    observed = hashlib.sha256(report_bytes).hexdigest()
    if observed != expected_report_sha256:
        raise ValueError(
            f"inference report SHA-256 mismatch: expected {expected_report_sha256}, got {observed}"
        )
    report = json.loads(report_bytes.decode("utf-8"))
    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != len(sequence_ids):
        raise ValueError(f"inference report has an invalid sample count: {report_path}")
    arrays: dict[str, np.ndarray] = {}
    for expected_index, (sequence_id, row) in enumerate(zip(sequence_ids, samples, strict=True)):
        if not isinstance(row, dict) or row.get("index") != expected_index:
            raise ValueError(f"inference sample ordering mismatch: {report_path}")
        relative = Path(str(row.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe inference sample path: {relative}")
        sample_path = (report_path.parent / relative).resolve()
        sample_path.relative_to(report_path.parent.resolve())
        payload = sample_path.read_bytes()
        observed_file_sha = hashlib.sha256(payload).hexdigest()
        if observed_file_sha != row.get("file_sha256"):
            raise ValueError(f"inference sample hash mismatch: {sample_path}")
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        if not isinstance(array, np.ndarray) or array.dtype != np.uint8:
            raise ValueError(f"inference sample must be uint8: {sample_path}")
        arrays[sequence_id] = np.ascontiguousarray(array)
    return report, arrays


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float]:
    return {
        field: float(np.mean([float(row[field]) for row in rows])) for field in AGGREGATE_FIELDS
    }


def _causal_rows(
    groups: list[dict[str, object]],
    *,
    exact: dict[str, np.ndarray],
    swapped: dict[str, np.ndarray],
    clips: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in groups:
        sequence_ids = tuple(str(value) for value in group["sequence_ids"])
        if len(sequence_ids) != 2:
            raise ValueError("causal group must have exactly two sequences")
        idle_id = next(value for value in sequence_ids if clips[value].request.action == "idle")
        walk_id = next(value for value in sequence_ids if clips[value].request.action == "walk")
        idle_target = clips[idle_id].rgba
        walk_target = clips[walk_id].rgba
        idle_exact = exact[idle_id]
        walk_exact = exact[walk_id]
        idle_to_walk = swapped[idle_id]
        walk_to_idle = swapped[walk_id]
        loop_mode = clips[idle_id].request.loop_mode
        target_separation = _distance(idle_target, walk_target, loop_mode=loop_mode)
        generated_separation = _distance(idle_exact, walk_exact, loop_mode=loop_mode)
        idle_to_idle = _distance(idle_exact, idle_target, loop_mode=loop_mode)
        idle_to_walk_target = _distance(idle_exact, walk_target, loop_mode=loop_mode)
        walk_to_walk = _distance(walk_exact, walk_target, loop_mode=loop_mode)
        walk_to_idle_target = _distance(walk_exact, idle_target, loop_mode=loop_mode)
        swapped_idle_to_walk_target = _distance(idle_to_walk, walk_target, loop_mode=loop_mode)
        swapped_walk_to_idle_target = _distance(walk_to_idle, idle_target, loop_mode=loop_mode)
        idle_swap_improvement = idle_to_walk_target - swapped_idle_to_walk_target
        walk_swap_improvement = walk_to_idle_target - swapped_walk_to_idle_target
        rows.append(
            {
                "both_actions_prefer_correct_target": (
                    idle_to_idle < idle_to_walk_target and walk_to_walk < walk_to_idle_target
                ),
                "contrast_group_sha256": group["key_sha256"],
                "description": clips[idle_id].request.description,
                "generated_separation": generated_separation,
                "generated_to_target_separation_ratio": (
                    generated_separation / target_separation if target_separation else 1.0
                ),
                "identity_id": clips[idle_id].identity_id,
                "idle_correct_target_margin": idle_to_walk_target - idle_to_idle,
                "idle_generated_to_idle_target": idle_to_idle,
                "idle_generated_to_walk_target": idle_to_walk_target,
                "idle_prefers_correct_target": idle_to_idle < idle_to_walk_target,
                "idle_sequence_id": idle_id,
                "idle_to_walk_after_distance": swapped_idle_to_walk_target,
                "idle_to_walk_moves_toward_replacement": idle_swap_improvement > 0,
                "idle_to_walk_replacement_improvement": idle_swap_improvement,
                "target_separation": target_separation,
                "walk_correct_target_margin": walk_to_idle_target - walk_to_walk,
                "walk_generated_to_idle_target": walk_to_idle_target,
                "walk_generated_to_walk_target": walk_to_walk,
                "walk_prefers_correct_target": walk_to_walk < walk_to_idle_target,
                "walk_sequence_id": walk_id,
                "walk_to_idle_after_distance": swapped_walk_to_idle_target,
                "walk_to_idle_moves_toward_replacement": walk_swap_improvement > 0,
                "walk_to_idle_replacement_improvement": walk_swap_improvement,
            }
        )
    rows.sort(key=lambda row: str(row["identity_id"]).encode())
    aggregate = {
        "both_actions_correct_target_preference_pair_count": sum(
            bool(row["both_actions_prefer_correct_target"]) for row in rows
        ),
        "idle_correct_target_preference_count": sum(
            bool(row["idle_prefers_correct_target"]) for row in rows
        ),
        "idle_to_walk_moves_toward_replacement_count": sum(
            bool(row["idle_to_walk_moves_toward_replacement"]) for row in rows
        ),
        "mean_generated_separation": float(
            np.mean([float(row["generated_separation"]) for row in rows])
        ),
        "mean_generated_to_target_separation_ratio": float(
            np.mean([float(row["generated_to_target_separation_ratio"]) for row in rows])
        ),
        "mean_idle_to_walk_replacement_improvement": float(
            np.mean([float(row["idle_to_walk_replacement_improvement"]) for row in rows])
        ),
        "mean_target_separation": float(np.mean([float(row["target_separation"]) for row in rows])),
        "mean_walk_to_idle_replacement_improvement": float(
            np.mean([float(row["walk_to_idle_replacement_improvement"]) for row in rows])
        ),
        "walk_correct_target_preference_count": sum(
            bool(row["walk_prefers_correct_target"]) for row in rows
        ),
        "walk_to_idle_moves_toward_replacement_count": sum(
            bool(row["walk_to_idle_moves_toward_replacement"]) for row in rows
        ),
    }
    return rows, aggregate


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace matched evaluation: {OUTPUT}")
    if _sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("materialization manifest SHA-256 mismatch")
    if _sha256_file(SOURCE_REPORT) != EXPECTED_SOURCE_REPORT_SHA256:
        raise ValueError("step-3,000 source report SHA-256 mismatch")
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    sequence_ids = tuple(source_report["sequence_ids"])
    loaded_clips = load_materialized_training_clips(
        MANIFEST,
        sequence_ids=sequence_ids,
        split="train",
        target_bucket=64,
        target_frames=8,
    )
    clips = {clip.sequence_id: clip for clip in loaded_clips}
    if set(clips) != set(sequence_ids):
        raise ValueError("materialization did not load every sequence")

    inference_documents: dict[str, dict[str, object]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    inference_records: dict[str, dict[str, object]] = {}
    noise_sha256: str | None = None
    for label, (path, expected_sha) in INFERENCE_REPORTS.items():
        document, loaded_arrays = _load_inference(path, expected_sha, sequence_ids)
        inference_documents[label] = document
        arrays[label] = loaded_arrays
        observed_noise = document["rng"]["noise_batch_sha256"]
        if noise_sha256 is None:
            noise_sha256 = observed_noise
        elif observed_noise != noise_sha256:
            raise ValueError("inference reports do not share identical noise")
        inference_records[label] = {
            "file_sha256": expected_sha,
            "path": str(path),
            "sampler": document["sampler"],
        }

    samplers: dict[str, object] = {}
    groups = source_report["matched_endpoint_contrast_plan"]["groups"]
    for sampler in ("endpoint", "euler"):
        exact = arrays[f"{sampler}_exact"]
        swapped = arrays[f"{sampler}_swap"]
        per_threshold: dict[str, object] = {}
        for threshold in (0, 127):
            rows: list[dict[str, object]] = []
            generated_rows: list[dict[str, object]] = []
            for sequence_id in sequence_ids:
                clip = clips[sequence_id]
                matched = asdict(
                    compare_matched_sequences(
                        _images(exact[sequence_id]),
                        _images(clip.rgba),
                        loop_mode=clip.request.loop_mode,
                        alpha_threshold=threshold,
                    )
                )
                generated = asdict(
                    evaluate_sprite_sequence(
                        _images(exact[sequence_id]),
                        loop_mode=clip.request.loop_mode,
                        alpha_threshold=threshold,
                    )
                )
                rows.append(matched)
                generated_rows.append(generated)
            per_threshold[str(threshold)] = {
                "aggregate_mean": _aggregate(rows),
                "generated_alpha_crisp_fraction_mean": float(
                    np.mean([float(row["alpha_crisp_fraction"]) for row in generated_rows])
                ),
                "generated_translucent_visible_fraction_mean": float(
                    np.mean([float(row["translucent_visible_fraction"]) for row in generated_rows])
                ),
                "samples": [
                    {
                        "action": clips[sequence_id].request.action,
                        "description": clips[sequence_id].request.description,
                        "matched_target": row,
                        "sequence_id": sequence_id,
                    }
                    for sequence_id, row in zip(sequence_ids, rows, strict=True)
                ],
            }
        causal_rows, causal_aggregate = _causal_rows(
            groups,
            exact=exact,
            swapped=swapped,
            clips=clips,
        )
        samplers[sampler] = {
            "causal_aggregate": causal_aggregate,
            "causal_pairs": causal_rows,
            "matched_metrics_by_alpha_threshold": per_threshold,
        }

    payload = {
        "artifact_kind": "tmwa_causal16_step3000_matched_memorization_and_action_evaluation",
        "checkpoint": {
            "file_sha256": "032b6de0c41c11b5dfa38c4c8e2e4f01ec231863096931d24cf6a676b934bf67",
            "path": str(EXPERIMENT / "checkpoint.pt"),
            "step": 3000,
        },
        "claim_scope": {
            "not_supported": [
                "held-out identity generation",
                "open-vocabulary text generation",
                "production-quality sprite generation",
            ],
            "supported": (
                "exact in-sample matched-noise memorization and action-token sensitivity"
            ),
        },
        "inference_reports": inference_records,
        "matched_noise": {
            "noise_batch_sha256": noise_sha256,
            "seed": 20_260_812,
            "strategy": "shared",
        },
        "materialization_manifest": {
            "file_sha256": EXPECTED_MANIFEST_SHA256,
            "path": str(MANIFEST),
        },
        "samplers": samplers,
        "schema_version": 1,
        "source_report": {
            "file_sha256": EXPECTED_SOURCE_REPORT_SHA256,
            "path": str(SOURCE_REPORT),
        },
    }
    output_bytes = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    guard.require_capacity(len(output_bytes) + 65_536, label="TMWA step-3,000 evaluation")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(output_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if OUTPUT.exists():
            raise FileExistsError(f"Refusing to replace matched evaluation: {OUTPUT}")
        os.rename(temporary, OUTPUT)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": hashlib.sha256(output_bytes).hexdigest(),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
