"""Publish the captioned dense MUGEN latent-still training plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_dense_training_plan import (  # noqa: E402
    export_mugen_dense_still_training_plan,
)
from spritelab.storage import DiskGuard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("captioned_materialization", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    digest = export_mugen_dense_still_training_plan(
        args.captioned_materialization,
        args.output,
        disk_guard=DiskGuard(ROOT, 100 * 1024**3),
    )
    print(json.dumps({"output": str(args.output.resolve()), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
