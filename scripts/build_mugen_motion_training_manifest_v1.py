"""Publish the conservative broad MUGEN reference-motion training manifest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_motion_training import (  # noqa: E402
    MugenMotionTrainingSelectionConfig,
    export_mugen_motion_training_manifest,
)
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-reference-motion-plan-v2.json"
PIXEL_AUDIT = ROOT / "data/index/reports/mugen-mffa-subject-frame-pixel-gate-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"


def main() -> None:
    path, digest = export_mugen_motion_training_manifest(
        PLAN,
        PIXEL_AUDIT,
        OUTPUT,
        config=MugenMotionTrainingSelectionConfig(one_sequence_per_identity_verb=True),
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    artifact = json.loads(path.read_bytes())
    print(
        json.dumps(
            {"counts": artifact["counts"], "path": str(path), "sha256": digest},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
