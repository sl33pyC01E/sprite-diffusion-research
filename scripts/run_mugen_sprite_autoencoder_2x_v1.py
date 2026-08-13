"""Train the high-fidelity 2x RGBA latent-codec ablation on MUGEN frames."""

from __future__ import annotations

import argparse
import sys
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

MANIFEST = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
OUTPUT = ROOT / "data/experiments/mugen-mffa-rgba-autoencoder-2x-v1-10000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = SpriteAutoencoderTrainingConfig(
        architecture=SpriteAutoencoderConfig(
            image_size=128,
            base_channels=64,
            latent_channels=8,
            channel_multipliers=(1, 2),
            residual_blocks=2,
        ),
        batch_size=12,
        checkpoint_every=2_500,
        gradient_accumulation=2,
        log_every=25,
        seed=20260823,
        steps=10_000,
        validate_every=500,
        validation_frames=512,
        warmup_steps=500,
    )
    corpus = prepare_broad_corpus(
        MANIFEST,
        target_size=config.architecture.image_size,
        target_frames=8,
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
        "corpus_sha256": corpus.corpus_sha256,
        "output_absent": not OUTPUT.exists(),
        "train_frames": sum(row.rgba.shape[0] for row in corpus.train),
        "train_identities": len({row.identity_id for row in corpus.train}),
        "train_sequences": len(corpus.train),
        "validation_frames": sum(row.rgba.shape[0] for row in corpus.validation),
        "validation_identities": len({row.identity_id for row in corpus.validation}),
        "validation_sequences": len(corpus.validation),
    }
    if args.preflight_only:
        print(preflight)
        return
    result = run_autoencoder_training(
        MANIFEST,
        OUTPUT,
        config=config,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(result)


if __name__ == "__main__":
    main()
