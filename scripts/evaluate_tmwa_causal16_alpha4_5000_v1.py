"""Evaluate the continued TMWA alpha4 step-5,000 matched inference."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from evaluate_tmwa_causal16_alpha4_v1 import run_variant  # noqa: E402


def main() -> None:
    inference_root = REPOSITORY / "data/inference"
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-5000"),
        expected_checkpoint_sha256=(
            "dc2e528372e2a0e17042fa582c6435052297462c7128ec435a86769324bd444a"
        ),
        expected_source_report_sha256=(
            "3635c59396ff4f6e44f0e7623bdfa84e75213bdf56034e4a1a90c098897307d7"
        ),
        inference_reports={
            "endpoint_exact": (
                inference_root
                / "tmwa-causal16-alpha4-ew1-5000-endpoint-exact-v1/inference-report.json",
                "c252fa2394a0b4e9d385c52fe36caa2162ec60fffbed70865eb1d68b43f0dc88",
            ),
            "endpoint_swap": (
                inference_root / "tmwa-causal16-alpha4-ew1-5000-endpoint-action-swap-v1/"
                "inference-report.json",
                "baebe2cf694722f2f3390194c59974ce37606582c0fd4d38520f80bccedd61e8",
            ),
            "euler_exact": (
                inference_root
                / "tmwa-causal16-alpha4-ew1-5000-euler32-exact-v1/inference-report.json",
                "49fba696ae2b9c71fc573ff20d41f4069d6f7f682e1741b1a93c1ed22f48e2de",
            ),
            "euler_swap": (
                inference_root / "tmwa-causal16-alpha4-ew1-5000-euler32-action-swap-v1/"
                "inference-report.json",
                "6f2d048e10bd8547508534b898ae41781de4052ed4b1cfb9081d5a47ef9ddd98",
            ),
        },
        output=(REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-ew1-5000-matched-v1.json"),
        artifact_kind="tmwa_causal16_alpha4_step5000_matched_memorization_evaluation",
        step=5_000,
    )


if __name__ == "__main__":
    main()
