from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.broad_train import BroadTrainingConfig, run_broad_training  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

MANIFEST = ROOT / "data/processed/tmwa-model-ready-action-v1/materialization.json"
OUTPUT = ROOT / "data/experiments/tmwa-broad-heldout-b128-f8-v2-10000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = BroadTrainingConfig(batch_size=8, gradient_accumulation=1)
    if args.preflight_only:
        from spritelab.broad_train import prepare_broad_corpus

        corpus = prepare_broad_corpus(
            MANIFEST,
            target_size=config.target_size,
            target_frames=config.target_frames,
        )
        print(
            {
                "corpus_sha256": corpus.corpus_sha256,
                "output_absent": not OUTPUT.exists(),
                "train": len(corpus.train),
                "train_identities": len({row.identity_id for row in corpus.train}),
                "validation": len(corpus.validation),
                "validation_identities": len({row.identity_id for row in corpus.validation}),
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
