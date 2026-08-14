"""Build the leakage-safe fixed view of Anime All Stars 3 canonical actions."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.build_mugen_schema_core_view_v1 as builder  # noqa: E402


def main() -> int:
    builder.INPUT = (
        ROOT / "data/processed/mugen-anime-all-stars3-schema-core-native-v2/materialization.json"
    )
    builder.OUTPUT = ROOT / "data/processed/mugen-anime-all-stars3-schema-core-b128-f8-v2"
    builder.STAGE = ROOT / "data/processed/.mugen-anime-all-stars3-schema-core-b128-f8-v2.partial"
    return builder.main()


if __name__ == "__main__":
    raise SystemExit(main())
