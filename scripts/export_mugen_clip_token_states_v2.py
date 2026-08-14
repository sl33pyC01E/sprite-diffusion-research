"""Export frozen SD1 CLIP states for the subject-bearing MUGEN plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.text_token_cache import export_clip_text_token_cache  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-latent-still-sequence-plan-v2-subject-bearing.json"
MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
MODEL_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"
OUTPUT = ROOT / "data/processed/mugen-mffa-sd14-clip-token-states-v2-subject-bearing"


def main() -> None:
    path, sha256 = export_clip_text_token_cache(
        PLAN,
        MODEL,
        OUTPUT,
        expected_source_index_sha256=MODEL_INDEX_SHA256,
        batch_size=64,
        device="cuda",
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"manifest": str(path), "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
