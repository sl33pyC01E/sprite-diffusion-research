"""Evaluate the TMWA alpha4/foreground4 step-3,000 ablation."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from evaluate_tmwa_causal16_alpha4_v1 import run_variant  # noqa: E402


def main() -> None:
    inference_root = REPOSITORY / "data/inference"
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-fg4-ew1-v1-3000"),
        expected_checkpoint_sha256=(
            "60ae0bc06f77a0e752a37b7f2b9f70410250155728a4e54b33fd9602d3039120"
        ),
        expected_source_report_sha256=(
            "e233892ce9fa03ca31bf1c9e9f5093316cebad6c305577ff7deb868eaa2dddd1"
        ),
        inference_reports={
            "endpoint_exact": (
                inference_root / "tmwa-causal16-alpha4-fg4-ew1-3000-endpoint-exact-v1/"
                "inference-report.json",
                "0465ccfb9d34f50bb99c404a9efbca4765ce1d9a2c6e6a0e72de012440847b93",
            ),
            "endpoint_swap": (
                inference_root / "tmwa-causal16-alpha4-fg4-ew1-3000-endpoint-action-swap-v1/"
                "inference-report.json",
                "b651bfb1bf1767c2afc6007031cc9a02f1d7c5013dabc041f5e1f085bd4883ff",
            ),
            "euler_exact": (
                inference_root / "tmwa-causal16-alpha4-fg4-ew1-3000-euler32-exact-v1/"
                "inference-report.json",
                "35611f6c197166f3763eaa5156999dcef3755cb1c02eff3c495ead88f47779f3",
            ),
            "euler_swap": (
                inference_root / "tmwa-causal16-alpha4-fg4-ew1-3000-euler32-action-swap-v1/"
                "inference-report.json",
                "07a53f89886dc621b8894e01c57ca2da7e15552b14cc1d327a8ba462b8645131",
            ),
        },
        output=(
            REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-fg4-ew1-3000-matched-v1.json"
        ),
        artifact_kind="tmwa_causal16_alpha4_fg4_step3000_matched_memorization_evaluation",
        step=3_000,
    )


if __name__ == "__main__":
    main()
