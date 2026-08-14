"""Build the CLIP-bounded appearance-only plan for stage-one training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_canonical_still import (  # noqa: E402
    export_mugen_canonical_still_training_plan,
)
from spritelab.storage import DiskGuard  # noqa: E402

SEQUENCE_PLAN = ROOT / "data/processed/mugen-mffa-latent-still-sequence-plan-v1.json"
CAPTIONS = (
    ROOT / "data/processed/mugen-mffa-canonical-still-captions-v3-spark-qwen35-122b/manifest.json"
)
OUTPUT = ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json"


def main() -> None:
    path, sha256 = export_mugen_canonical_still_training_plan(
        SEQUENCE_PLAN,
        CAPTIONS,
        OUTPUT,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"path": str(path), "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
