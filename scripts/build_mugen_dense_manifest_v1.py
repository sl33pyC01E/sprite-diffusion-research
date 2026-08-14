"""Build a leakage-safe M.U.G.E.N reference-motion training manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_dense_manifest import export_mugen_dense_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization", action="append", type=Path, required=True)
    parser.add_argument("--quality-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tier", choices=("broad", "dense"), default="dense")
    args = parser.parse_args()
    digest = export_mugen_dense_manifest(
        tuple(args.materialization),
        args.quality_audit,
        args.output,
        tier=args.tier,
    )
    print(json.dumps({"path": str(args.output.resolve()), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
