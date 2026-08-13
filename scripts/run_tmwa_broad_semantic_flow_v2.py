"""Train the held-out TMWA semantic model without endpoint regression."""

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

MANIFEST = ROOT / "data/processed/tmwa-model-ready-action-v1/materialization.json"
SEMANTIC_TABLE = ROOT / "data/processed/semantic-text/tmwa-openai-clip-vit-b32-c7244be-v1"
OUTPUT = ROOT / "data/experiments/tmwa-broad-heldout-b128-f8-clip-flow-v2-10000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = BroadTrainingConfig(
        batch_size=8,
        endpoint_weight=0.0,
        gradient_accumulation=1,
        semantic_embedding_table=str(SEMANTIC_TABLE),
        seed=20260817,
    )
    if args.preflight_only:
        corpus = prepare_broad_corpus(
            MANIFEST,
            target_size=config.target_size,
            target_frames=config.target_frames,
        )
        table = load_semantic_embedding_table(SEMANTIC_TABLE)
        print(
            {
                "corpus_sha256": corpus.corpus_sha256,
                "endpoint_weight": config.endpoint_weight,
                "output_absent": not OUTPUT.exists(),
                "semantic_embedding_array_sha256": table.embeddings_array_sha256,
                "semantic_manifest_sha256": table.manifest_sha256,
                "train": len(corpus.train),
                "validation": len(corpus.validation),
            }
        )
        return
    result = run_broad_training(
        MANIFEST,
        OUTPUT,
        config=config,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(result)


if __name__ == "__main__":
    main()
