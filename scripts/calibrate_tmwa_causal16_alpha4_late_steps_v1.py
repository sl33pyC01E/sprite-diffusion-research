"""Calibrate comparable display decodes for TMWA alpha4 steps 5,000 and 6,000."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from calibrate_tmwa_causal16_alpha4_decode_v1 import run_variant  # noqa: E402


def main() -> None:
    reports = REPOSITORY / "data/index/reports"
    for step, checkpoint_report_sha, inference_report_sha in (
        (
            5_000,
            "3635c59396ff4f6e44f0e7623bdfa84e75213bdf56034e4a1a90c098897307d7",
            "c252fa2394a0b4e9d385c52fe36caa2162ec60fffbed70865eb1d68b43f0dc88",
        ),
        (
            6_000,
            "7fef539de909b6612de95c40168c98de02f25f93078433adcd6ec529f26e01de",
            "7c454da198b50b25ebaff320a3fcf85f2e02cdf009574f5c2cbfa6da033db20d",
        ),
    ):
        run_variant(
            source_report=(
                REPOSITORY / f"data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-{step}/"
                "overfit-report.json"
            ),
            inference_report=(
                REPOSITORY / f"data/inference/tmwa-causal16-alpha4-ew1-{step}-endpoint-exact-v1/"
                "inference-report.json"
            ),
            expected_source_report_sha256=checkpoint_report_sha,
            expected_inference_report_sha256=inference_report_sha,
            alpha_output=reports / f"tmwa-causal16-alpha4-{step}-hard-alpha-calibration-v1.json",
            palette_output=reports / f"tmwa-causal16-alpha4-{step}-palette-calibration-v1.json",
        )


if __name__ == "__main__":
    main()
