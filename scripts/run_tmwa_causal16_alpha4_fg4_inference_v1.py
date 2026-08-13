"""Run matched inference for the TMWA alpha4/foreground4 step-3,000 ablation."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from run_tmwa_causal16_alpha4_inference_v1 import run_variant  # noqa: E402


def main() -> None:
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-fg4-ew1-v1-3000"),
        expected_checkpoint_sha256=(
            "60ae0bc06f77a0e752a37b7f2b9f70410250155728a4e54b33fd9602d3039120"
        ),
        expected_source_report_sha256=(
            "e233892ce9fa03ca31bf1c9e9f5093316cebad6c305577ff7deb868eaa2dddd1"
        ),
        output_prefix="tmwa-causal16-alpha4-fg4-ew1-3000",
    )


if __name__ == "__main__":
    main()
