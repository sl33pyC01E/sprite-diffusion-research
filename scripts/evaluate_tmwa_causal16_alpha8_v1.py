"""Evaluate the TMWA alpha8 step-3,000 ablation."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from evaluate_tmwa_causal16_alpha4_v1 import run_variant  # noqa: E402


def main() -> None:
    inference_root = REPOSITORY / "data/inference"
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha8-ew1-v1-3000"),
        expected_checkpoint_sha256=(
            "99e1a5f3c6f482bf69c9d987a6bd207e1d243e18b4b5ef0f69329a48489aed53"
        ),
        expected_source_report_sha256=(
            "072beba2766f2cff5404aa884de19d943b254e5763f0273823993a6b85aabc5e"
        ),
        inference_reports={
            "endpoint_exact": (
                inference_root
                / "tmwa-causal16-alpha8-ew1-3000-endpoint-exact-v1/inference-report.json",
                "9dcd7d0bf7a6a04871fd5552f915236e9578e9618eb9480633db37f7b6615fb2",
            ),
            "endpoint_swap": (
                inference_root / "tmwa-causal16-alpha8-ew1-3000-endpoint-action-swap-v1/"
                "inference-report.json",
                "a64c4f212f0aa36cde9c95b6db1d059350cd7dbe0acac4db7cf2d4b80743250c",
            ),
            "euler_exact": (
                inference_root
                / "tmwa-causal16-alpha8-ew1-3000-euler32-exact-v1/inference-report.json",
                "0f27dde2e4eedb887f9323ce303568aeb87be6b1a71c480f381646367a726cf5",
            ),
            "euler_swap": (
                inference_root / "tmwa-causal16-alpha8-ew1-3000-euler32-action-swap-v1/"
                "inference-report.json",
                "8f5e388fe12b6e0421b1862faafb03f4a07057df9a54def4dc07c55bbf2b29cb",
            ),
        },
        output=(REPOSITORY / "data/index/reports/tmwa-causal16-alpha8-ew1-3000-matched-v1.json"),
        artifact_kind="tmwa_causal16_alpha8_step3000_matched_memorization_evaluation",
        step=3_000,
    )


if __name__ == "__main__":
    main()
