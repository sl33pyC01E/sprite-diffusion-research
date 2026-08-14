"""Publish verified visual-caption inputs for the dense MUGEN corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_dense_caption import export_mugen_dense_caption_inputs  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dense_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    digest = export_mugen_dense_caption_inputs(
        args.dense_manifest,
        args.output_directory,
        disk_guard=DiskGuard(ROOT, 100 * 1024**3),
    )
    print(
        json.dumps(
            {"manifest_sha256": digest, "output": str(args.output_directory.resolve())},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
