"""Run matched-noise endpoint inference for a held-out TMWA split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.inference import CheckpointInferenceConfig, run_checkpoint_inference  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import load_materialized_training_clips  # noqa: E402

MANIFEST = ROOT / "data/processed/tmwa-model-ready-action-v1/materialization.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--sampler-algorithm", choices=("endpoint", "euler", "heun"), default="endpoint"
    )
    parser.add_argument("--sample-steps", type=int, default=1)
    args = parser.parse_args()
    seed = (
        args.seed
        if args.seed is not None
        else {"validation": 20260917, "test": 20260918}[args.split]
    )
    clips = load_materialized_training_clips(
        MANIFEST,
        split=args.split,
        target_frames=8,
    )
    result = run_checkpoint_inference(
        args.checkpoint,
        args.output,
        [clip.request for clip in clips],
        [clip.frame_phases for clip in clips],
        expected_checkpoint_sha256=args.checkpoint_sha256,
        config=CheckpointInferenceConfig(
            seed=seed,
            sample_steps=args.sample_steps,
            sampler_algorithm=args.sampler_algorithm,
            noise_strategy="independent",
            device="cuda",
            deterministic_algorithms=True,
        ),
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(result)


if __name__ == "__main__":
    main()
