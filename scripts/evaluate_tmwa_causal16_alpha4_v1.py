"""Evaluate matched TMWA alpha4 step-3,000 inference and causal action swaps."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from evaluate_tmwa_causal16_step3000_v1 import (  # noqa: E402
    _aggregate,
    _causal_rows,
    _images,
    _load_inference,
    _sha256_file,
)

from spritelab.evaluation import compare_matched_sequences, evaluate_sprite_sequence  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import load_materialized_training_clips  # noqa: E402

MANIFEST = (
    REPOSITORY / "data/processed/tmwa-model-ready-action-v1/"
    "tmwa-causal-down-idle-walk-16-materialization-v1.json"
)
EXPERIMENT = REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-3000"
SOURCE_REPORT = EXPERIMENT / "overfit-report.json"
OUTPUT = REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-ew1-3000-matched-v1.json"
EXPECTED_MANIFEST_SHA256 = "095205d49811a0102368d607a5bcb8dd9a0c2b057ebe33ddaeeda893bede45ca"
EXPECTED_CHECKPOINT_SHA256 = "804e4077e0e8d6a20237cd078b97b8069058fd7635390c8d0764ec9eea847826"
EXPECTED_SOURCE_REPORT_SHA256 = "1c471220c2b47cdbd821d8921520db97722abe808f07f02a50f07a499fe035c5"
INFERENCE_REPORTS = {
    "endpoint_exact": (
        REPOSITORY / "data/inference/tmwa-causal16-alpha4-ew1-3000-endpoint-exact-v1/"
        "inference-report.json",
        "ed5299b1f77c435f02961d694ad61ffcd3dd7fa149aa63082f40bc08b642a8b1",
    ),
    "endpoint_swap": (
        REPOSITORY / "data/inference/tmwa-causal16-alpha4-ew1-3000-endpoint-action-swap-v1/"
        "inference-report.json",
        "18dfbf8cb528ec1a79d4199b52702f4f406d2213c8a87b54822451b26d111d01",
    ),
    "euler_exact": (
        REPOSITORY / "data/inference/tmwa-causal16-alpha4-ew1-3000-euler32-exact-v1/"
        "inference-report.json",
        "72f2233f3708bfce9424c12c1c90fbd68748896442eca0f45cf1c225876229fc",
    ),
    "euler_swap": (
        REPOSITORY / "data/inference/tmwa-causal16-alpha4-ew1-3000-euler32-action-swap-v1/"
        "inference-report.json",
        "02dba9050d04bdd24b0bbc3cf87cc1f30f4a19865ff162c13775d8253d07d937",
    ),
}


def run_variant(
    *,
    experiment: Path,
    expected_checkpoint_sha256: str,
    expected_source_report_sha256: str,
    inference_reports: dict[str, tuple[Path, str]],
    output: Path,
    artifact_kind: str,
    step: int,
) -> None:
    source_report_path = experiment / "overfit-report.json"
    if output.exists():
        raise FileExistsError(f"Refusing to replace matched evaluation: {output}")
    if _sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("materialization manifest SHA-256 mismatch")
    if _sha256_file(source_report_path) != expected_source_report_sha256:
        raise ValueError("alpha4 source report SHA-256 mismatch")
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    sequence_ids = tuple(str(value) for value in source_report["sequence_ids"])
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

    arrays: dict[str, dict[str, np.ndarray]] = {}
    inference_records: dict[str, dict[str, object]] = {}
    noise_sha256: str | None = None
    for label, (path, expected_sha) in inference_reports.items():
        document, loaded_arrays = _load_inference(path, expected_sha, sequence_ids)
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
                rows.append(
                    asdict(
                        compare_matched_sequences(
                            _images(exact[sequence_id]),
                            _images(clip.rgba),
                            loop_mode=clip.request.loop_mode,
                            alpha_threshold=threshold,
                        )
                    )
                )
                generated_rows.append(
                    asdict(
                        evaluate_sprite_sequence(
                            _images(exact[sequence_id]),
                            loop_mode=clip.request.loop_mode,
                            alpha_threshold=threshold,
                        )
                    )
                )
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
        "artifact_kind": artifact_kind,
        "checkpoint": {
            "file_sha256": expected_checkpoint_sha256,
            "path": str(experiment / "checkpoint.pt"),
            "step": step,
        },
        "claim_scope": {
            "not_supported": [
                "held-out identity generation",
                "open-vocabulary text generation",
                "production-quality sprite generation",
            ],
            "supported": "exact in-sample alpha-weight and action-token sensitivity ablation",
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
            "file_sha256": expected_source_report_sha256,
            "path": str(source_report_path),
        },
    }
    output_bytes = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    guard.require_capacity(len(output_bytes) + 65_536, label="TMWA alpha4 evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(output_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if output.exists():
            raise FileExistsError(f"Refusing to replace matched evaluation: {output}")
        os.rename(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {"output": str(output), "sha256": hashlib.sha256(output_bytes).hexdigest()},
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def main() -> None:
    run_variant(
        experiment=EXPERIMENT,
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        expected_source_report_sha256=EXPECTED_SOURCE_REPORT_SHA256,
        inference_reports=INFERENCE_REPORTS,
        output=OUTPUT,
        artifact_kind="tmwa_causal16_alpha4_step3000_matched_memorization_evaluation",
        step=3_000,
    )


if __name__ == "__main__":
    main()
