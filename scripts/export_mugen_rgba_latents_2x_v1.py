"""Export the selected 2x MUGEN RGBA latent cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_cache import export_mugen_latent_cache  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

LEGACY_MATERIALIZATION = (
    ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
)
LEGACY_CHECKPOINT = (
    ROOT / "data/experiments/mugen-mffa-rgba-autoencoder-2x-v1-10000/training-step-0010000.pt"
)
LEGACY_CHECKPOINT_SHA256 = "f8993d4b7dcc0b53526f0b4f1fae15a90dc942cc1c2f3c07e74029fc26c6be85"
LEGACY_OUTPUT = ROOT / "data/processed/mugen-mffa-rgba-latents-2x-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization", type=Path, default=LEGACY_MATERIALIZATION)
    parser.add_argument("--checkpoint", type=Path, default=LEGACY_CHECKPOINT)
    parser.add_argument("--expected-checkpoint-sha256", default=LEGACY_CHECKPOINT_SHA256)
    parser.add_argument("--output", type=Path, default=LEGACY_OUTPUT)
    parser.add_argument("--batch-sequences", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    path, sha256 = export_mugen_latent_cache(
        args.materialization,
        args.checkpoint,
        args.output,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        batch_sequences=args.batch_sequences,
        device=args.device,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"manifest": str(path), "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
