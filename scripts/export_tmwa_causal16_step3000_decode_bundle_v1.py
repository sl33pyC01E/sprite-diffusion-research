"""Export calibrated display-only decodes for TMWA step-3,000 endpoint samples."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spritelab.decode_bundle import (  # noqa: E402
    DecodeBundleArtifactRef,
    DecodeBundleClipRef,
    export_decode_preview_bundle,
)
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import load_materialized_training_clips  # noqa: E402

MANIFEST = (
    REPOSITORY / "data/processed/tmwa-model-ready-action-v1/"
    "tmwa-causal-down-idle-walk-16-materialization-v1.json"
)
SOURCE_REPORT = (
    REPOSITORY / "data/experiments/tmwa-down-idle-walk-causal16-b64-f8-baseline-v1-3000/"
    "overfit-report.json"
)
INFERENCE_REPORT = (
    REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-endpoint-exact-v1/"
    "inference-report.json"
)
ALPHA_CALIBRATION = (
    REPOSITORY / "data/index/reports/tmwa-causal16-baseline3000-hard-alpha-calibration-v1.json"
)
PALETTE_CALIBRATION = (
    REPOSITORY / "data/index/reports/tmwa-causal16-baseline3000-palette-calibration-v1.json"
)
OUTPUT = REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-endpoint-decoded-v1"

EXPECTED_MANIFEST_SHA256 = "095205d49811a0102368d607a5bcb8dd9a0c2b057ebe33ddaeeda893bede45ca"
EXPECTED_SOURCE_REPORT_SHA256 = "0ea6ca38c1d5fff6d8680efa5e65540b3df7e6526b52e5b42ffaabaf929fc415"
EXPECTED_INFERENCE_REPORT_SHA256 = (
    "4b5fc533b24ca99f35b3b08785eb636441087a8236fc217666229e51f537d9ce"
)
EXPECTED_ALPHA_CALIBRATION_SHA256 = (
    "d601cfaff98b421bd47f55d0af905583e06e28d4771af930e350510412bea48a"
)
EXPECTED_PALETTE_CALIBRATION_SHA256 = (
    "b02388182bc672cb0ea07d92d1f3b64d81f66cf093c002ed0e99c1884b502e0c"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, artifact_id: str, expected_sha256: str) -> DecodeBundleArtifactRef:
    observed = _sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {observed}")
    return DecodeBundleArtifactRef(
        artifact_id=artifact_id,
        path=path,
        file_sha256=expected_sha256,
    )


def _sample_id(description: str, action: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", description.casefold()).strip("-") or "entity"
    return f"{index:02d}-{slug}-{action}"


def main() -> None:
    if _sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("materialization manifest SHA-256 mismatch")
    if _sha256(SOURCE_REPORT) != EXPECTED_SOURCE_REPORT_SHA256:
        raise ValueError("source report SHA-256 mismatch")
    inference = json.loads(INFERENCE_REPORT.read_text(encoding="utf-8"))
    if _sha256(INFERENCE_REPORT) != EXPECTED_INFERENCE_REPORT_SHA256:
        raise ValueError("inference report SHA-256 mismatch")
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
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
        raise ValueError("inference sample count differs from the checkpoint sequence count")

    clip_refs: list[DecodeBundleClipRef] = []
    for index, (sequence_id, sample) in enumerate(zip(sequence_ids, samples, strict=True)):
        if not isinstance(sample, dict) or sample.get("index") != index:
            raise ValueError("inference sample order is invalid")
        relative = Path(str(sample.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe sample path: {relative}")
        sample_path = (INFERENCE_REPORT.parent / relative).resolve()
        sample_path.relative_to(INFERENCE_REPORT.parent.resolve())
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
        OUTPUT,
        hard_alpha_threshold=160,
        palette_sizes=(16, 64),
        source_report=_artifact(
            INFERENCE_REPORT,
            "tmwa-step3000-endpoint-inference",
            EXPECTED_INFERENCE_REPORT_SHA256,
        ),
        hard_alpha_calibrations=(
            _artifact(
                ALPHA_CALIBRATION,
                "tmwa-step3000-hard-alpha-training-estimate",
                EXPECTED_ALPHA_CALIBRATION_SHA256,
            ),
        ),
        palette_calibrations=(
            _artifact(
                PALETTE_CALIBRATION,
                "tmwa-step3000-palette-training-estimate",
                EXPECTED_PALETTE_CALIBRATION_SHA256,
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


if __name__ == "__main__":
    main()
