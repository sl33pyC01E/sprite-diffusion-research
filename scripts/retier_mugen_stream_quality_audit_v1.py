"""Recompute MUGEN quality admission from an exact prior pixel audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_stream_quality import (  # noqa: E402
    MugenStreamQualityPolicy,
    export_retiered_mugen_stream_quality_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-view-scale", type=float, default=0)
    parser.add_argument("--minimum-dynamic-slots", type=int, default=0)
    parser.add_argument("--minimum-distinct-slot-arrays", type=int, default=1)
    args = parser.parse_args()
    digest = export_retiered_mugen_stream_quality_audit(
        args.source_audit,
        args.output,
        policy=MugenStreamQualityPolicy(
            minimum_view_scale=args.minimum_view_scale,
            minimum_dynamic_slots=args.minimum_dynamic_slots,
            minimum_distinct_slot_arrays=args.minimum_distinct_slot_arrays,
        ),
    )
    print(json.dumps({"output": str(args.output.resolve()), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
