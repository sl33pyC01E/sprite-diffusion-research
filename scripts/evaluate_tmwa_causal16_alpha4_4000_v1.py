"""Evaluate the continued TMWA alpha4 step-4,000 matched inference."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from evaluate_tmwa_causal16_alpha4_v1 import run_variant  # noqa: E402


def main() -> None:
    inference_root = REPOSITORY / "data/inference"
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-4000"),
        expected_checkpoint_sha256=(
            "6d63446f22444ff36632d6f9869003883ba65c1b2171968c5b657fd77a78624f"
        ),
        expected_source_report_sha256=(
            "5365f324cf49ff1db0d421ce2145b0820d964a262f4b3f21f4874d35ea36e8ea"
        ),
        inference_reports={
            "endpoint_exact": (
                inference_root
                / "tmwa-causal16-alpha4-ew1-4000-endpoint-exact-v1/inference-report.json",
                "76788c5da1e4da2af0411d337a02b6306a4cb0db1849582bc9d66e5d1823a616",
            ),
            "endpoint_swap": (
                inference_root / "tmwa-causal16-alpha4-ew1-4000-endpoint-action-swap-v1/"
                "inference-report.json",
                "172d51b6d72eabf414582652997bb5cba1b2a1049c093ebf07309ef9fe61449f",
            ),
            "euler_exact": (
                inference_root
                / "tmwa-causal16-alpha4-ew1-4000-euler32-exact-v1/inference-report.json",
                "c09db71fea27214e2d363b9da4431880a38e3f312f9da6f3e3c4ccc36aba025a",
            ),
            "euler_swap": (
                inference_root / "tmwa-causal16-alpha4-ew1-4000-euler32-action-swap-v1/"
                "inference-report.json",
                "02c422f3a3d86e0498d38cd4cd7db805211eb5b7e1a7a0acb48606e5139dd24e",
            ),
        },
        output=(REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-ew1-4000-matched-v1.json"),
        artifact_kind="tmwa_causal16_alpha4_step4000_matched_memorization_evaluation",
        step=4_000,
    )


if __name__ == "__main__":
    main()
