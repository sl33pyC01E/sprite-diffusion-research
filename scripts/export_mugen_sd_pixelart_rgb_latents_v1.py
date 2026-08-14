"""Export canonical MUGEN RGB latents using the pinned sprite-specific VAE."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.sd_control_cache import export_sd14_rgb_latent_cache  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json"
REVISION = "8229c9b6e928103f0e657cfe6b14d902cb2101d6"
MODEL = ROOT / f"data/models/sd-pixelart-spritesheet-{REVISION}"
SOURCE_INDEX_SHA256 = "5f7eea291d7831ccb4d6bb07b011669f532ebd4371dee35b8743ea90fbc926df"
OUTPUT = ROOT / "data/processed/mugen-mffa-sd-pixelart-rgb-vae-latents-canonical-v1"


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
