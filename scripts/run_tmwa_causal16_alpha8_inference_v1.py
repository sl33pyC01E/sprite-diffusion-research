"""Run matched inference for the TMWA alpha8 step-3,000 ablation."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from run_tmwa_causal16_alpha4_inference_v1 import run_variant  # noqa: E402


def main() -> None:
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha8-ew1-v1-3000"),
        expected_checkpoint_sha256=(
            "99e1a5f3c6f482bf69c9d987a6bd207e1d243e18b4b5ef0f69329a48489aed53"
        ),
        expected_source_report_sha256=(
            "072beba2766f2cff5404aa884de19d943b254e5763f0273823993a6b85aabc5e"
        ),
        output_prefix="tmwa-causal16-alpha8-ew1-3000",
    )


if __name__ == "__main__":
    main()
