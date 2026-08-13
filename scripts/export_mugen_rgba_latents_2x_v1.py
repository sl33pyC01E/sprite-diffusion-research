"""Export the selected 2x MUGEN RGBA latent cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_cache import export_mugen_latent_cache  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

MATERIALIZATION = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
CHECKPOINT = (
    ROOT / "data/experiments/mugen-mffa-rgba-autoencoder-2x-v1-10000/training-step-0010000.pt"
)
CHECKPOINT_SHA256 = "f8993d4b7dcc0b53526f0b4f1fae15a90dc942cc1c2f3c07e74029fc26c6be85"
OUTPUT = ROOT / "data/processed/mugen-mffa-rgba-latents-2x-v1"


def main() -> None:
    path, sha256 = export_mugen_latent_cache(
        MATERIALIZATION,
        CHECKPOINT,
        OUTPUT,
        expected_checkpoint_sha256=CHECKPOINT_SHA256,
        batch_sequences=8,
        device="cuda",
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"manifest": str(path), "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
