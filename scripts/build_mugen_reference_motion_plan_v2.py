"""Build the reference-still plus action-to-animation latent training plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_motion_dataset import export_mugen_reference_motion_plan  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402


def main() -> None:
    output, digest = export_mugen_reference_motion_plan(
        ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json",
        ROOT / "data/index/reports/mugen-mffa-action-taxonomy-v1.json",
        ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json",
        ROOT / "data/processed/mugen-mffa-rgba-latents-2x-v1/manifest.json",
        ROOT / "data/processed/mugen-mffa-reference-motion-plan-v2.json",
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"output": str(output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
