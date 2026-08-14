"""Evaluate the broad MUGEN latent-motion EMA on untouched test identities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_motion_train import evaluate_latent_motion_checkpoint  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

LEGACY_MANIFEST = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"


def main(
    *,
    checkpoint: str,
    expected_sha256: str,
    output_name: str,
    manifest: Path = LEGACY_MANIFEST,
) -> None:
    result = evaluate_latent_motion_checkpoint(
        manifest,
        checkpoint,
        ROOT / "data/inference" / output_name,
        expected_checkpoint_sha256=expected_sha256,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(
        json.dumps(
            {"report": str(result.report_path), "sha256": result.report_sha256},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--manifest", type=Path, default=LEGACY_MANIFEST)
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        expected_sha256=args.expected_sha256,
        output_name=args.output_name,
        manifest=args.manifest,
    )
