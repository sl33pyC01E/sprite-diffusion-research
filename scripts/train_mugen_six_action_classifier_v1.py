"""Train the exact identity-disjoint dense MUGEN six-action recognizer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_action_classifier import (  # noqa: E402
    MugenActionClassifierTrainingConfig,
    train_mugen_action_classifier,
)
from spritelab.storage import DiskGuard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    result = train_mugen_action_classifier(
        args.manifest,
        args.output,
        config=MugenActionClassifierTrainingConfig(epochs=args.epochs),
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(
        json.dumps(
            {
                "checkpoint": str(result.checkpoint_path),
                "checkpoint_sha256": result.checkpoint_sha256,
                "report": str(result.report_path),
                "report_sha256": result.report_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
