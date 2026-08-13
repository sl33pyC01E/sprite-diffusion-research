"""Train the first held-out, scale-stabilized MFFA semantic endpoint model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.broad_train import (  # noqa: E402
    BroadTrainingConfig,
    prepare_broad_corpus,
    run_broad_training,
)
from spritelab.semantic_text import load_semantic_embedding_table  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

MANIFEST = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
SEMANTIC_TABLE = ROOT / "data/processed/semantic-text/mugen-mffa-openai-clip-vit-b32-c7244be-v1"
OUTPUT = ROOT / "data/experiments/mugen-mffa-b128-f8-clip-endpoint-v1-30000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = BroadTrainingConfig(
        batch_size=4,
        checkpoint_every=5_000,
        condition_dim=256,
        depth=8,
        gradient_accumulation=2,
        horizontal_flip_probability=0.5,
        model_dim=256,
        num_heads=8,
        semantic_embedding_table=str(SEMANTIC_TABLE),
        steps=30_000,
        validate_every=1_000,
        warmup_steps=1_000,
        seed=20260821,
    )
    corpus = prepare_broad_corpus(
        MANIFEST,
        target_size=config.target_size,
        target_frames=config.target_frames,
    )
    table = load_semantic_embedding_table(SEMANTIC_TABLE)
    missing = sorted(
        {
            row.request.description
            for row in (*corpus.train, *corpus.validation)
            if row.request.description not in table.descriptions
        }
    )
    preflight = {
        "corpus_sha256": corpus.corpus_sha256,
        "output_absent": not OUTPUT.exists(),
        "semantic_embedding_array_sha256": table.embeddings_array_sha256,
        "semantic_manifest_sha256": table.manifest_sha256,
        "semantic_missing": missing,
        "train": len(corpus.train),
        "validation": len(corpus.validation),
    }
    if args.preflight_only:
        print(preflight)
        return
    if missing:
        raise ValueError(f"semantic table is incomplete: {missing!r}")
    result = run_broad_training(
        MANIFEST,
        OUTPUT,
        config=config,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(result)


if __name__ == "__main__":
    main()
