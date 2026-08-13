"""Run matched inference for the continued TMWA alpha4 step-6,000 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from run_tmwa_causal16_alpha4_inference_v1 import run_variant  # noqa: E402


def main() -> None:
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-6000"),
        expected_checkpoint_sha256=(
            "394880c5e067059f01b1f9c2462e75bae66705944e11f04c0a5b058e9689b761"
        ),
        expected_source_report_sha256=(
            "7fef539de909b6612de95c40168c98de02f25f93078433adcd6ec529f26e01de"
        ),
        output_prefix="tmwa-causal16-alpha4-ew1-6000",
    )


if __name__ == "__main__":
    main()
