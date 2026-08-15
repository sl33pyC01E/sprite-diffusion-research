"""Launch the quality-first scratch MUGEN latent-still DiT."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_still_train import (  # noqa: E402
    LatentStillTrainingConfig,
    load_latent_still_corpus,
    run_latent_still_training,
)
from spritelab.storage import DiskGuard  # noqa: E402

LEGACY_PLAN = ROOT / "data/processed/mugen-mffa-latent-still-sequence-plan-v1.json"
LEGACY_LATENTS = ROOT / "data/processed/mugen-mffa-rgba-latents-2x-v1/manifest.json"
LEGACY_TEXT = ROOT / "data/processed/mugen-mffa-sd14-clip-token-states-v1/manifest.json"
LEGACY_OUTPUT = ROOT / "data/experiments/mugen-mffa-latent-still-dit-scratch-v1-30000"
CORPUS_OUTPUT = ROOT / "data/experiments/mugen-six-action-still-dit-scratch-v1-step50000"
SMOKE_OUTPUT = ROOT / "data/experiments/mugen-latent-still-dit-gpu-smoke-v1-step1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=LEGACY_PLAN)
    parser.add_argument("--latents", type=Path, default=LEGACY_LATENTS)
    parser.add_argument("--text", type=Path, default=LEGACY_TEXT)
    parser.add_argument(
        "--profile", choices=("legacy30000", "corpus50000", "smoke"), default="legacy30000"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--expected-resume-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--recovery-checkpoint-every", type=int)
    parser.add_argument("--recovery-checkpoint-slots", type=int)
    args = parser.parse_args()
    if (args.resume_checkpoint is None) != (args.expected_resume_sha256 is None):
        raise ValueError("resume checkpoint and expected SHA-256 must be provided together")
    if args.profile == "legacy30000":
        config = LatentStillTrainingConfig()
    elif args.profile == "corpus50000":
        config = LatentStillTrainingConfig(steps=50_000, checkpoint_every=10_000)
    else:
        config = LatentStillTrainingConfig(
            batch_size=1,
            gradient_accumulation=1,
            warmup_steps=0,
            steps=1,
            log_every=1,
            validate_every=1,
            checkpoint_every=1,
            validation_rows=1,
        )
    overrides = {
        key: value
        for key, value in (
            ("batch_size", args.batch_size),
            ("gradient_accumulation", args.gradient_accumulation),
            ("checkpoint_every", args.checkpoint_every),
            ("recovery_checkpoint_every", args.recovery_checkpoint_every),
            ("recovery_checkpoint_slots", args.recovery_checkpoint_slots),
        )
        if value is not None
    }
    if overrides:
        config = replace(config, **overrides)
    default_outputs = {
        "corpus50000": CORPUS_OUTPUT,
        "legacy30000": LEGACY_OUTPUT,
        "smoke": SMOKE_OUTPUT,
    }
    output = args.output or default_outputs[args.profile]
    if args.preflight_only:
        corpus = load_latent_still_corpus(
            args.plan, args.latents, args.text, verify_latent_files=True
        )
        print(
            json.dumps(
                {
                    "config": asdict(config),
                    "corpus": corpus.contract,
                    "output": str(output.resolve()),
                    "resume": (
                        {
                            "checkpoint": str(args.resume_checkpoint.resolve()),
                            "sha256": args.expected_resume_sha256,
                        }
                        if args.resume_checkpoint is not None
                        else None
                    ),
                    "train_rows": len(corpus.train_indices),
                    "validation_rows": len(corpus.validation_indices),
                },
                sort_keys=True,
            )
        )
        return
    result = run_latent_still_training(
        args.plan,
        args.latents,
        args.text,
        output,
        config=config,
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
