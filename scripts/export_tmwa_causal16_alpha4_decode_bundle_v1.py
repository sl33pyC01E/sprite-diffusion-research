"""Export calibrated TMWA alpha4 step-3,000 endpoint display derivatives."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from export_tmwa_causal16_step3000_decode_bundle_v1 import (  # noqa: E402
    EXPECTED_MANIFEST_SHA256,
    MANIFEST,
    _artifact,
    _sample_id,
    _sha256,
)

from spritelab.decode_bundle import DecodeBundleClipRef, export_decode_preview_bundle  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import load_materialized_training_clips  # noqa: E402

SOURCE_REPORT = (
    REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-3000/overfit-report.json"
)
INFERENCE_REPORT = (
    REPOSITORY
    / "data/inference/tmwa-causal16-alpha4-ew1-3000-endpoint-exact-v1/inference-report.json"
)
ALPHA_CALIBRATION = (
    REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-3000-hard-alpha-calibration-v1.json"
)
PALETTE_CALIBRATION = (
    REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-3000-palette-calibration-v1.json"
)
OUTPUT = REPOSITORY / "data/inference/tmwa-causal16-alpha4-ew1-3000-endpoint-decoded-v1"
EXPECTED_SOURCE_REPORT_SHA256 = "1c471220c2b47cdbd821d8921520db97722abe808f07f02a50f07a499fe035c5"
EXPECTED_INFERENCE_REPORT_SHA256 = (
    "ed5299b1f77c435f02961d694ad61ffcd3dd7fa149aa63082f40bc08b642a8b1"
)
EXPECTED_ALPHA_CALIBRATION_SHA256 = (
    "15fe4bed3fd917f1ac09b920487c78869222498bb08521ef31f61cdd9b026ac7"
)
EXPECTED_PALETTE_CALIBRATION_SHA256 = (
    "398515a4933b14357d10698545a7036b79a8de595155262b3d996b1f3d62eee2"
)


def run_variant(
    *,
    source_report: Path,
    inference_report: Path,
    alpha_calibration: Path,
    palette_calibration: Path,
    output: Path,
    expected_source_report_sha256: str,
    expected_inference_report_sha256: str,
    expected_alpha_calibration_sha256: str,
    expected_palette_calibration_sha256: str,
    hard_alpha_threshold: int,
    palette_sizes: tuple[int, ...],
) -> None:
    if _sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("materialization manifest SHA-256 mismatch")
    if _sha256(source_report) != expected_source_report_sha256:
        raise ValueError("source report SHA-256 mismatch")
    if _sha256(inference_report) != expected_inference_report_sha256:
        raise ValueError("inference report SHA-256 mismatch")
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
    samples = inference.get("samples")
    if not isinstance(samples, list) or len(samples) != len(sequence_ids):
        raise ValueError("inference sample count is invalid")
    clip_refs: list[DecodeBundleClipRef] = []
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
        clip = clips_by_id[sequence_id]
        clip_refs.append(
            DecodeBundleClipRef(
                sample_id=_sample_id(clip.request.description, clip.request.action, index),
                source_path=sample_path,
                source_file_sha256=str(sample["file_sha256"]),
                duration_ms=clip.duration_ms,
                loop_mode=clip.request.loop_mode,
            )
        )
    result = export_decode_preview_bundle(
        clip_refs,
        output,
        hard_alpha_threshold=hard_alpha_threshold,
        palette_sizes=palette_sizes,
        source_report=_artifact(
            inference_report,
            "tmwa-alpha4-endpoint-inference",
            expected_inference_report_sha256,
        ),
        hard_alpha_calibrations=(
            _artifact(
                alpha_calibration,
                "tmwa-alpha4-hard-alpha-training-estimate",
                expected_alpha_calibration_sha256,
            ),
        ),
        palette_calibrations=(
            _artifact(
                palette_calibration,
                "tmwa-alpha4-palette-training-estimate",
                expected_palette_calibration_sha256,
            ),
        ),
        integer_scale=4,
        disk_guard=DiskGuard(REPOSITORY, 100 * 1024**3),
    )
    print(
        json.dumps(
            {
                "bundle": str(result.bundle_path),
                "clip_count": result.clip_count,
                "index_sha256": result.index_sha256,
                "palette_sizes": list(result.palette_sizes),
                "payload_file_count": result.payload_file_count,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def main() -> None:
    run_variant(
        source_report=SOURCE_REPORT,
        inference_report=INFERENCE_REPORT,
        alpha_calibration=ALPHA_CALIBRATION,
        palette_calibration=PALETTE_CALIBRATION,
        output=OUTPUT,
        expected_source_report_sha256=EXPECTED_SOURCE_REPORT_SHA256,
        expected_inference_report_sha256=EXPECTED_INFERENCE_REPORT_SHA256,
        expected_alpha_calibration_sha256=EXPECTED_ALPHA_CALIBRATION_SHA256,
        expected_palette_calibration_sha256=EXPECTED_PALETTE_CALIBRATION_SHA256,
        hard_alpha_threshold=144,
        palette_sizes=(16, 24),
    )


if __name__ == "__main__":
    main()
