"""Publish the zero-copy dense MUGEN autoencoder materialization bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_dense_compat import (  # noqa: E402
    export_mugen_dense_autoencoder_materialization,
    export_mugen_dense_captioned_materialization,
)
from spritelab.storage import DiskGuard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dense_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--caption-manifest", type=Path)
    args = parser.parse_args()
    guard = DiskGuard(ROOT, 100 * 1024**3)
    if args.caption_manifest is None:
        digest = export_mugen_dense_autoencoder_materialization(
            args.dense_manifest,
            args.output,
            disk_guard=guard,
        )
    else:
        digest = export_mugen_dense_captioned_materialization(
            args.dense_manifest,
            args.caption_manifest,
            args.output,
            disk_guard=guard,
        )
    print(json.dumps({"output": str(args.output.resolve()), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
