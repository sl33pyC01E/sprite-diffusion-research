"""Calibrate display-only decoding for the TMWA alpha4 step-3,000 endpoint run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from calibrate_tmwa_causal16_step3000_decode_v1 import (  # noqa: E402
    EXPECTED_MANIFEST_SHA256,
    MANIFEST,
    PALETTE_SIZES,
    THRESHOLDS,
    _publish_fixed_targets,
    _sha256,
)

from spritelab.decode_calibration import (  # noqa: E402
    CalibrationArrayRef,
    export_global_palette_size_calibration,
    export_hard_alpha_threshold_calibration,
)
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import load_materialized_training_clips  # noqa: E402

SOURCE_REPORT = (
    REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-3000/overfit-report.json"
)
INFERENCE_REPORT = (
    REPOSITORY
    / "data/inference/tmwa-causal16-alpha4-ew1-3000-endpoint-exact-v1/inference-report.json"
)
ALPHA_OUTPUT = (
    REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-3000-hard-alpha-calibration-v1.json"
)
PALETTE_OUTPUT = (
    REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-3000-palette-calibration-v1.json"
)
EXPECTED_SOURCE_REPORT_SHA256 = "1c471220c2b47cdbd821d8921520db97722abe808f07f02a50f07a499fe035c5"
EXPECTED_INFERENCE_REPORT_SHA256 = (
    "ed5299b1f77c435f02961d694ad61ffcd3dd7fa149aa63082f40bc08b642a8b1"
)


def run_variant(
    *,
    source_report: Path,
    inference_report: Path,
    expected_source_report_sha256: str,
    expected_inference_report_sha256: str,
    alpha_output: Path,
    palette_output: Path,
) -> None:
    for path, expected in (
        (MANIFEST, EXPECTED_MANIFEST_SHA256),
        (source_report, expected_source_report_sha256),
        (inference_report, expected_inference_report_sha256),
    ):
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {observed}")
    source = json.loads(source_report.read_text(encoding="utf-8"))
    inference = json.loads(inference_report.read_text(encoding="utf-8"))
    sequence_ids = tuple(str(value) for value in source["sequence_ids"])
    clips = load_materialized_training_clips(
        MANIFEST,
        sequence_ids=sequence_ids,
        split="train",
        target_bucket=64,
        target_frames=8,
    )
    clips_by_id = {clip.sequence_id: clip for clip in clips}
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    target_paths = _publish_fixed_targets(clips_by_id, sequence_ids, disk_guard=guard)

    samples = inference.get("samples")
    if not isinstance(samples, list) or len(samples) != len(sequence_ids):
        raise ValueError("inference sample count is invalid")
    sources: list[CalibrationArrayRef] = []
    targets: list[CalibrationArrayRef] = []
    for index, (sequence_id, sample) in enumerate(zip(sequence_ids, samples, strict=True)):
        if not isinstance(sample, dict) or sample.get("index") != index:
            raise ValueError("inference sample order is invalid")
        relative = Path(str(sample.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe sample path: {relative}")
        sample_path = (inference_report.parent / relative).resolve()
        sample_path.relative_to(inference_report.parent.resolve())
        if _sha256(sample_path) != sample.get("file_sha256"):
            raise ValueError(f"sample SHA-256 mismatch: {sample_path}")
        sources.append(CalibrationArrayRef(sequence_id, sample_path))
        targets.append(CalibrationArrayRef(sequence_id, target_paths[sequence_id]))

    alpha = export_hard_alpha_threshold_calibration(
        sources,
        targets,
        THRESHOLDS,
        alpha_output,
        estimate_kind="training_target_estimate",
        disk_guard=guard,
    )
    palette = export_global_palette_size_calibration(
        sources,
        targets,
        PALETTE_SIZES,
        palette_output,
        alpha_threshold=alpha.selected_threshold,
        estimate_kind="training_target_estimate",
        disk_guard=guard,
    )
    print(
        json.dumps(
            {
                "alpha": {
                    "selected_threshold": alpha.selected_threshold,
                    "sha256": alpha.artifact_sha256,
                },
                "palette": {
                    "selected_maximum_colors": palette.selected_maximum_colors,
                    "sha256": palette.artifact_sha256,
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def main() -> None:
    run_variant(
        source_report=SOURCE_REPORT,
        inference_report=INFERENCE_REPORT,
        expected_source_report_sha256=EXPECTED_SOURCE_REPORT_SHA256,
        expected_inference_report_sha256=EXPECTED_INFERENCE_REPORT_SHA256,
        alpha_output=ALPHA_OUTPUT,
        palette_output=PALETTE_OUTPUT,
    )


if __name__ == "__main__":
    main()
