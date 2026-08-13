"""Launch the quality-first scratch MUGEN latent-still DiT."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_still_train import (  # noqa: E402
    LatentStillTrainingConfig,
    run_latent_still_training,
)
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-latent-still-sequence-plan-v1.json"
LATENTS = ROOT / "data/processed/mugen-mffa-rgba-latents-2x-v1/manifest.json"
TEXT = ROOT / "data/processed/mugen-mffa-sd14-clip-token-states-v1/manifest.json"
OUTPUT = ROOT / "data/experiments/mugen-mffa-latent-still-dit-scratch-v1-30000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--expected-resume-sha256")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = run_latent_still_training(
        PLAN,
        LATENTS,
        TEXT,
        args.output,
        config=LatentStillTrainingConfig(),
        resume_checkpoint_path=args.resume_checkpoint,
        expected_resume_sha256=args.expected_resume_sha256,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(
        json.dumps(
            {
                "checkpoint": str(result.training_checkpoint_path),
                "report": str(result.report_path),
                "report_sha256": result.report_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
