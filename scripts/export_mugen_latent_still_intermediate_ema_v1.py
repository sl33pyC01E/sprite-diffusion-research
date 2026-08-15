"""Export an inference-only EMA checkpoint from an intermediate still training step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_still_checkpoint import (  # noqa: E402
    export_latent_still_intermediate_ema,
)
from spritelab.storage import DiskGuard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-training-sha256", required=True)
    parser.add_argument("--latent-manifest", type=Path, required=True)
    parser.add_argument("--expected-latent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, sha256 = export_latent_still_intermediate_ema(
        args.training_checkpoint,
        args.latent_manifest,
        args.output,
        expected_training_checkpoint_sha256=args.expected_training_sha256,
        expected_latent_manifest_sha256=args.expected_latent_sha256,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"output": str(output), "sha256": sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
