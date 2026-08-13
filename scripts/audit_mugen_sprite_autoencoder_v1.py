"""Publish a fixed held-out reconstruction audit for the first MUGEN RGBA AE."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.autoencoder_audit import export_autoencoder_reconstruction_audit  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

MANIFEST = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
RUN = ROOT / "data/experiments/mugen-mffa-rgba-autoencoder-v1-20000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--allow-legacy-torch-version", action="store_true")
    args = parser.parse_args()
    if args.step <= 0:
        raise ValueError("--step must be positive")
    checkpoint = RUN / f"training-step-{args.step:07d}.pt"
    output = ROOT / f"data/index/reports/mugen-mffa-rgba-autoencoder-step{args.step}-audit-v1"
    checkpoint_sha256 = _file_sha256(checkpoint)
    result = export_autoencoder_reconstruction_audit(
        MANIFEST,
        checkpoint,
        output,
        expected_checkpoint_sha256=checkpoint_sha256,
        maximum_frames=16,
        integer_scale=2,
        allow_legacy_torch_version=args.allow_legacy_torch_version,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(result)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
