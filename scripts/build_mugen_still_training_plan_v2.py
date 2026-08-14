"""Publish the subject-bearing MUGEN latent-still training plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_still_dataset import export_mugen_still_training_plan  # noqa: E402

MATERIALIZATION = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
TAXONOMY = ROOT / "data/index/reports/mugen-mffa-action-taxonomy-v1.json"
CAPTIONS = (
    ROOT / "data/processed/mugen-mffa-canonical-still-captions-v3-spark-qwen35-122b/manifest.json"
)
ELIGIBILITY = ROOT / "data/processed/mugen-mffa-subject-bearing-still-eligibility-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-latent-still-sequence-plan-v2-subject-bearing.json"


def main() -> None:
    path, sha256 = export_mugen_still_training_plan(
        MATERIALIZATION,
        TAXONOMY,
        CAPTIONS,
        OUTPUT,
        frame_eligibility_path=ELIGIBILITY,
    )
    print(json.dumps({"output": str(path), "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
