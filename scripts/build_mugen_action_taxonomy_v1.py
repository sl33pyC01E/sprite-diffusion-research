"""Build the source-number-only MFFA structured action taxonomy."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_taxonomy import build_mugen_action_taxonomy  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402


def main() -> None:
    source = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
    output = ROOT / "data/index/reports/mugen-mffa-action-taxonomy-v1.json"
    digest = build_mugen_action_taxonomy(
        source,
        output,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print({"output": str(output), "sha256": digest})


if __name__ == "__main__":
    main()
