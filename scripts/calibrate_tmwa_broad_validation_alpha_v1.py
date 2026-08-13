from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.broad_train import prepare_broad_corpus  # noqa: E402
from spritelab.decode_calibration import (  # noqa: E402
    CalibrationArrayRef,
    export_hard_alpha_threshold_calibration,
)
from spritelab.storage import DiskGuard  # noqa: E402

MANIFEST = ROOT / "data/processed/tmwa-model-ready-action-v1/materialization.json"
TARGETS = ROOT / "data/inference/tmwa-broad-validation-target-arrays-v1"
THRESHOLDS = tuple(range(16, 256, 16))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inference = args.inference.resolve()
    report_path = inference / "inference-report.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    corpus = prepare_broad_corpus(MANIFEST, target_size=128, target_frames=8)
    if len(report["samples"]) != len(corpus.validation):
        raise ValueError("inference/validation count mismatch")

    TARGETS.mkdir(parents=True, exist_ok=True)
    sources: list[CalibrationArrayRef] = []
    targets: list[CalibrationArrayRef] = []
    for sample, row in zip(report["samples"], corpus.validation, strict=True):
        if sample["request"] != {
            "action": row.request.action,
            "description": row.request.description,
            "direction": row.request.direction,
            "entity_class": row.request.entity_class,
            "loop_mode": row.request.loop_mode,
            "view": row.request.view,
        }:
            raise ValueError(f"request mismatch for {row.sequence_id}")
        source_path = inference / sample["path"]
        if _sha256(source_path) != sample["file_sha256"]:
            raise ValueError(f"source sample hash mismatch for {row.sequence_id}")
        target_path = TARGETS / f"{row.sequence_id}.npy"
        if target_path.exists():
            existing = np.load(target_path, allow_pickle=False)
            if not np.array_equal(existing, row.rgba):
                raise ValueError(f"existing target differs for {row.sequence_id}")
        else:
            with target_path.open("xb") as handle:
                np.save(handle, row.rgba, allow_pickle=False)
        sources.append(CalibrationArrayRef(row.sequence_id, source_path))
        targets.append(CalibrationArrayRef(row.sequence_id, target_path))

    result = export_hard_alpha_threshold_calibration(
        sources,
        targets,
        THRESHOLDS,
        args.output,
        estimate_kind="held_out_validation",
        disk_guard=DiskGuard(ROOT, 100 * 1024**3),
    )
    print(
        {
            "inference_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "selected_threshold": result.selected_threshold,
            "calibration_sha256": result.artifact_sha256,
        }
    )
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
