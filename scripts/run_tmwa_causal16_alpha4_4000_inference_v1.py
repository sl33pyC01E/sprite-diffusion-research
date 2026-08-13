"""Run matched inference for the continued TMWA alpha4 step-4,000 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from run_tmwa_causal16_alpha4_inference_v1 import run_variant  # noqa: E402


def main() -> None:
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-4000"),
        expected_checkpoint_sha256=(
            "6d63446f22444ff36632d6f9869003883ba65c1b2171968c5b657fd77a78624f"
        ),
        expected_source_report_sha256=(
            "5365f324cf49ff1db0d421ce2145b0820d964a262f4b3f21f4874d35ea36e8ea"
        ),
        output_prefix="tmwa-causal16-alpha4-ew1-4000",
    )


if __name__ == "__main__":
    main()
