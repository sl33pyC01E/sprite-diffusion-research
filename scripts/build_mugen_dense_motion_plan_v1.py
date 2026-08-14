"""Publish dense reference-conditioned MUGEN motion artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_dense_motion_plan import export_mugen_dense_motion_artifacts  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("materialization", type=Path)
    parser.add_argument("latent_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    plan_sha256, training_sha256 = export_mugen_dense_motion_artifacts(
        args.materialization,
        args.latent_manifest,
        args.output_directory,
        disk_guard=DiskGuard(ROOT, 100 * 1024**3),
    )
    print(
        json.dumps(
            {
                "motion_plan_sha256": plan_sha256,
                "output": str(args.output_directory.resolve()),
                "training_manifest_sha256": training_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
