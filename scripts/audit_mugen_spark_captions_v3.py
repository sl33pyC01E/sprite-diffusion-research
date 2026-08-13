"""Publish the full Qwen3.5-122B caption audit and visible review gallery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.caption_audit import build_caption_audit  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

MANIFEST = (
    ROOT / "data/processed/mugen-mffa-canonical-still-captions-v3-spark-qwen35-122b/manifest.json"
)
TOKENIZER = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components/tokenizer"
OUTPUT = ROOT / "data/index/reports/mugen-mffa-spark-caption-v3-audit-v1"


def main() -> None:
    path, sha256 = build_caption_audit(
        MANIFEST,
        TOKENIZER,
        OUTPUT,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"report": str(path), "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
