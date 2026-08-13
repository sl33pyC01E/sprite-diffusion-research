"""Run matched inference for the continued TMWA alpha4 step-5,000 checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from run_tmwa_causal16_alpha4_inference_v1 import run_variant  # noqa: E402


def main() -> None:
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-5000"),
        expected_checkpoint_sha256=(
            "dc2e528372e2a0e17042fa582c6435052297462c7128ec435a86769324bd444a"
        ),
        expected_source_report_sha256=(
            "3635c59396ff4f6e44f0e7623bdfa84e75213bdf56034e4a1a90c098897307d7"
        ),
        output_prefix="tmwa-causal16-alpha4-ew1-5000",
    )


if __name__ == "__main__":
    main()
