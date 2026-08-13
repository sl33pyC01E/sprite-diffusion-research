"""Evaluate the continued TMWA alpha4 step-6,000 matched inference."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from evaluate_tmwa_causal16_alpha4_v1 import run_variant  # noqa: E402


def main() -> None:
    inference_root = REPOSITORY / "data/inference"
    run_variant(
        experiment=(REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-6000"),
        expected_checkpoint_sha256=(
            "394880c5e067059f01b1f9c2462e75bae66705944e11f04c0a5b058e9689b761"
        ),
        expected_source_report_sha256=(
            "7fef539de909b6612de95c40168c98de02f25f93078433adcd6ec529f26e01de"
        ),
        inference_reports={
            "endpoint_exact": (
                inference_root
                / "tmwa-causal16-alpha4-ew1-6000-endpoint-exact-v1/inference-report.json",
                "7c454da198b50b25ebaff320a3fcf85f2e02cdf009574f5c2cbfa6da033db20d",
            ),
            "endpoint_swap": (
                inference_root / "tmwa-causal16-alpha4-ew1-6000-endpoint-action-swap-v1/"
                "inference-report.json",
                "c3506fe65ddc7d18a0a5999679ba5d228d340a45441596dc9512e7c1ddf0543f",
            ),
            "euler_exact": (
                inference_root
                / "tmwa-causal16-alpha4-ew1-6000-euler32-exact-v1/inference-report.json",
                "5bd9634725c7a6214b2088fcb5aa51a71dbc800eebdc6487f9d5b43362f3b0de",
            ),
            "euler_swap": (
                inference_root / "tmwa-causal16-alpha4-ew1-6000-euler32-action-swap-v1/"
                "inference-report.json",
                "becf9f5ca1eeb09ff763c27b644984d2b2d503f8b938efe778d254bf5d4a4602",
            ),
        },
        output=(REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-ew1-6000-matched-v1.json"),
        artifact_kind="tmwa_causal16_alpha4_step6000_matched_memorization_evaluation",
        step=6_000,
    )


if __name__ == "__main__":
    main()
