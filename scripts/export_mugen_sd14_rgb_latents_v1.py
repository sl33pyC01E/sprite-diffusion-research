"""Export the deterministic, noncanonical SD1.4 RGB control cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.sd_control_cache import export_sd14_rgb_latent_cache  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-latent-still-sequence-plan-v1.json"
MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
SOURCE_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"
OUTPUT = ROOT / "data/processed/mugen-mffa-sd14-rgb-vae-latents-v1"


def main() -> None:
    path, sha256 = export_sd14_rgb_latent_cache(
        PLAN,
        MODEL,
        OUTPUT,
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        device="cuda",
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"manifest": str(path), "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
