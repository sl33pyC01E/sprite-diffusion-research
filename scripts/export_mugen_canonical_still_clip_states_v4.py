"""Export frozen SD1 CLIP states for the canonical appearance-still plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.text_token_cache import export_clip_text_token_cache  # noqa: E402

LEGACY_PLAN = ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json"
DEFAULT_MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
MODEL_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"
LEGACY_OUTPUT = ROOT / "data/processed/mugen-mffa-sd14-clip-token-states-canonical-v4"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=LEGACY_PLAN)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--expected-model-index-sha256", default=MODEL_INDEX_SHA256)
    parser.add_argument("--output", type=Path, default=LEGACY_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    path, sha256 = export_clip_text_token_cache(
        args.plan,
        args.model,
        args.output,
        expected_source_index_sha256=args.expected_model_index_sha256,
        batch_size=args.batch_size,
        device=args.device,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"manifest": str(path), "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
