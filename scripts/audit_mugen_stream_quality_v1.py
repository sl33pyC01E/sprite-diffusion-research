"""Export an exact broad-versus-dense audit for streamed M.U.G.E.N cores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_stream_quality import (  # noqa: E402
    MugenStreamQualityPolicy,
    export_mugen_stream_quality_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-view-scale", type=float, default=0.5)
    parser.add_argument("--minimum-dynamic-slots", type=int, default=3)
    parser.add_argument("--minimum-distinct-slot-arrays", type=int, default=4)
    args = parser.parse_args()
    policy = MugenStreamQualityPolicy(
        minimum_view_scale=args.minimum_view_scale,
        minimum_dynamic_slots=args.minimum_dynamic_slots,
        minimum_distinct_slot_arrays=args.minimum_distinct_slot_arrays,
    )
    digest = export_mugen_stream_quality_audit(
        tuple(args.materialization), args.output, policy=policy
    )
    print(json.dumps({"path": str(args.output.resolve()), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
