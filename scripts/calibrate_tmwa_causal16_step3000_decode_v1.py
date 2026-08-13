"""Calibrate display-only alpha and palette decoding for TMWA step 3,000."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spritelab.decode_calibration import (  # noqa: E402
    CalibrationArrayRef,
    export_global_palette_size_calibration,
    export_hard_alpha_threshold_calibration,
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
OUTPUT_DIRECTORY = REPOSITORY / "data/index/reports"
ALPHA_OUTPUT = OUTPUT_DIRECTORY / "tmwa-causal16-baseline3000-hard-alpha-calibration-v1.json"
PALETTE_OUTPUT = OUTPUT_DIRECTORY / "tmwa-causal16-baseline3000-palette-calibration-v1.json"
TARGET_DIRECTORY = (
    REPOSITORY / "data/processed/tmwa-model-ready-action-v1/tmwa-causal16-fixed8-target-arrays-v1"
)

EXPECTED_MANIFEST_SHA256 = "095205d49811a0102368d607a5bcb8dd9a0c2b057ebe33ddaeeda893bede45ca"
EXPECTED_SOURCE_REPORT_SHA256 = "0ea6ca38c1d5fff6d8680efa5e65540b3df7e6526b52e5b42ffaabaf929fc415"
EXPECTED_INFERENCE_REPORT_SHA256 = (
    "4b5fc533b24ca99f35b3b08785eb636441087a8236fc217666229e51f537d9ce"
)
THRESHOLDS = (32, 64, 80, 96, 112, 127, 144, 160, 176, 192, 208, 224)
PALETTE_SIZES = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _publish_fixed_targets(
    clips_by_id: dict[str, object],
    sequence_ids: tuple[str, ...],
    *,
    disk_guard: DiskGuard,
) -> dict[str, Path]:
    if TARGET_DIRECTORY.exists():
        index_path = TARGET_DIRECTORY / "index.json"
        document = json.loads(index_path.read_text(encoding="utf-8"))
        if document.get("materialization_manifest_sha256") != EXPECTED_MANIFEST_SHA256:
            raise ValueError("fixed-target array index cites a different materialization")
        rows = document.get("rows")
        if not isinstance(rows, list) or [row.get("sequence_id") for row in rows] != list(
            sequence_ids
        ):
            raise ValueError("fixed-target array index ordering is invalid")
        resolved: dict[str, Path] = {}
        for row in rows:
            path = (TARGET_DIRECTORY / str(row["path"])).resolve()
            path.relative_to(TARGET_DIRECTORY.resolve())
            if _sha256(path) != row.get("file_sha256"):
                raise ValueError(f"fixed-target array file SHA-256 mismatch: {path}")
            resolved[str(row["sequence_id"])] = path
        return resolved

    disk_guard.require_capacity(16 * 8 * 64 * 64 * 4 + 8 * 1024**2, label="fixed targets")
    TARGET_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{TARGET_DIRECTORY.name}.",
            suffix=".partial",
            dir=TARGET_DIRECTORY.parent,
        )
    )
    try:
        rows: list[dict[str, object]] = []
        for index, sequence_id in enumerate(sequence_ids):
            clip = clips_by_id[sequence_id]
            filename = f"{index:02d}-{sequence_id}.npy"
            path = staging / filename
            with path.open("xb") as handle:
                np.save(handle, clip.rgba, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            rows.append(
                {
                    "array_sha256": _array_sha256(clip.rgba),
                    "file_sha256": _sha256(path),
                    "path": filename,
                    "sequence_id": sequence_id,
                    "shape": list(clip.rgba.shape),
                }
            )
        index = {
            "artifact_kind": "exact_retimed_training_target_array_set",
            "claim_scope": "Exact fixed-eight in-sample targets; not generated model output.",
            "materialization_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "row_count": len(rows),
            "rows": rows,
            "schema_version": 1,
        }
        index_bytes = (
            json.dumps(index, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode("utf-8")
        with (staging / "index.json").open("xb") as handle:
            handle.write(index_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if TARGET_DIRECTORY.exists():
            raise FileExistsError(f"Refusing to replace fixed-target array set: {TARGET_DIRECTORY}")
        os.rename(staging, TARGET_DIRECTORY)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        sequence_id: TARGET_DIRECTORY / f"{index:02d}-{sequence_id}.npy"
        for index, sequence_id in enumerate(sequence_ids)
    }


def main() -> None:
    for path, expected in (
        (MANIFEST, EXPECTED_MANIFEST_SHA256),
        (SOURCE_REPORT, EXPECTED_SOURCE_REPORT_SHA256),
        (INFERENCE_REPORT, EXPECTED_INFERENCE_REPORT_SHA256),
    ):
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {observed}")

    source_document = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    inference_document = json.loads(INFERENCE_REPORT.read_text(encoding="utf-8"))
    sequence_ids = tuple(str(value) for value in source_document["sequence_ids"])
    clips = load_materialized_training_clips(
        MANIFEST,
        sequence_ids=sequence_ids,
        split="train",
        target_bucket=64,
        target_frames=8,
    )
    clips_by_id = {clip.sequence_id: clip for clip in clips}
    if tuple(clips_by_id) != sequence_ids:
        raise ValueError("materialized clip order differs from the checkpoint sequence order")

    samples = inference_document.get("samples")
    if not isinstance(samples, list) or len(samples) != len(sequence_ids):
        raise ValueError("inference report sample count differs from checkpoint sequence count")

    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    fixed_target_paths = _publish_fixed_targets(
        clips_by_id,
        sequence_ids,
        disk_guard=guard,
    )
    sources: list[CalibrationArrayRef] = []
    targets: list[CalibrationArrayRef] = []
    for index, (sequence_id, sample) in enumerate(zip(sequence_ids, samples, strict=True)):
        if not isinstance(sample, dict) or sample.get("index") != index:
            raise ValueError("inference report sample order is invalid")
        relative = Path(str(sample.get("path")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe sample path: {relative}")
        sample_path = (INFERENCE_REPORT.parent / relative).resolve()
        sample_path.relative_to(INFERENCE_REPORT.parent.resolve())
        if _sha256(sample_path) != sample.get("file_sha256"):
            raise ValueError(f"sample file SHA-256 mismatch: {sample_path}")
        sources.append(CalibrationArrayRef(sequence_id, sample_path))
        targets.append(CalibrationArrayRef(sequence_id, fixed_target_paths[sequence_id]))

    alpha = export_hard_alpha_threshold_calibration(
        sources,
        targets,
        THRESHOLDS,
        ALPHA_OUTPUT,
        estimate_kind="training_target_estimate",
        disk_guard=guard,
    )
    palette = export_global_palette_size_calibration(
        sources,
        targets,
        PALETTE_SIZES,
        PALETTE_OUTPUT,
        alpha_threshold=alpha.selected_threshold,
        estimate_kind="training_target_estimate",
        disk_guard=guard,
    )
    print(
        json.dumps(
            {
                "alpha": {
                    "path": str(alpha.artifact_path),
                    "selected_threshold": alpha.selected_threshold,
                    "sha256": alpha.artifact_sha256,
                },
                "palette": {
                    "path": str(palette.artifact_path),
                    "selected_maximum_colors": palette.selected_maximum_colors,
                    "sha256": palette.artifact_sha256,
                },
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
