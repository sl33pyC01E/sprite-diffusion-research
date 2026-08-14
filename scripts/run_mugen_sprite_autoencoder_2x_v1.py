"""Train the high-fidelity 2x RGBA latent-codec ablation on MUGEN frames."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.autoencoder_train import (  # noqa: E402
    SpriteAutoencoderTrainingConfig,
    run_autoencoder_training,
)
from spritelab.broad_train import prepare_broad_corpus  # noqa: E402
from spritelab.models.sprite_autoencoder import SpriteAutoencoderConfig  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

LEGACY_MANIFEST = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
LEGACY_OUTPUT = ROOT / "data/experiments/mugen-mffa-rgba-autoencoder-2x-v1-10000"
CORPUS_OUTPUT = ROOT / "data/experiments/mugen-six-action-rgba-autoencoder-2x-v1-20000"
SMOKE_OUTPUT = ROOT / "data/experiments/mugen-rgba-autoencoder-2x-gpu-smoke-v1-step1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=LEGACY_MANIFEST)
    parser.add_argument(
        "--profile", choices=("legacy10000", "corpus20000", "smoke"), default="legacy10000"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--expected-resume-sha256")
    args = parser.parse_args()
    if (args.resume_checkpoint is None) != (args.expected_resume_sha256 is None):
        raise ValueError("resume checkpoint and expected SHA-256 must be provided together")
    if args.profile == "legacy10000":
        steps = 10_000
        checkpoint_every = 2_500
        validation_frames = 512
    elif args.profile == "corpus20000":
        steps = 20_000
        checkpoint_every = 5_000
        validation_frames = 512
    else:
        steps = 1
        checkpoint_every = 1
        validation_frames = 1
    output = (
        args.output
        or {
            "legacy10000": LEGACY_OUTPUT,
            "corpus20000": CORPUS_OUTPUT,
            "smoke": SMOKE_OUTPUT,
        }[args.profile]
    )
    config = SpriteAutoencoderTrainingConfig(
        architecture=SpriteAutoencoderConfig(
            image_size=128,
            base_channels=64,
            latent_channels=8,
            channel_multipliers=(1, 2),
            residual_blocks=2,
        ),
        batch_size=12,
        checkpoint_every=checkpoint_every,
        gradient_accumulation=1 if args.profile == "smoke" else 2,
        log_every=25,
        seed=20260823,
        steps=steps,
        validate_every=1 if args.profile == "smoke" else 500,
        validation_frames=validation_frames,
        warmup_steps=0 if args.profile == "smoke" else 500,
    )
    corpus = prepare_broad_corpus(
        args.manifest,
        target_size=config.architecture.image_size,
        target_frames=8,
        usage="autoencoder",
    )
    preflight = {
        "architecture": {
            "downsample_factor": config.architecture.downsample_factor,
            "latent_channels": config.architecture.latent_channels,
            "latent_elements": (
                config.architecture.latent_channels * config.architecture.latent_size**2
            ),
            "latent_size": config.architecture.latent_size,
        },
        "config": asdict(config),
        "corpus_sha256": corpus.corpus_sha256,
        "manifest": str(args.manifest.resolve()),
        "output": str(output.resolve()),
        "output_absent": not output.exists(),
        "resume": (
            {
                "checkpoint": str(args.resume_checkpoint.resolve()),
                "sha256": args.expected_resume_sha256,
            }
            if args.resume_checkpoint is not None
            else None
        ),
        "train_frames": sum(row.rgba.shape[0] for row in corpus.train),
        "train_identities": len({row.identity_id for row in corpus.train}),
        "train_sequences": len(corpus.train),
        "validation_frames": sum(row.rgba.shape[0] for row in corpus.validation),
        "validation_identities": len({row.identity_id for row in corpus.validation}),
        "validation_sequences": len(corpus.validation),
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    result = run_autoencoder_training(
        args.manifest,
        output,
        config=config,
        resume_checkpoint_path=args.resume_checkpoint,
        expected_resume_sha256=args.expected_resume_sha256,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(result)


if __name__ == "__main__":
    main()
