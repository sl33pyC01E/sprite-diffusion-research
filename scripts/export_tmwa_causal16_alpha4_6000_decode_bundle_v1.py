"""Export the selected calibrated TMWA alpha4 step-6,000 display bundle."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from export_tmwa_causal16_alpha4_decode_bundle_v1 import run_variant  # noqa: E402


def main() -> None:
    reports = REPOSITORY / "data/index/reports"
    run_variant(
        source_report=(
            REPOSITORY
            / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-6000/overfit-report.json"
        ),
        inference_report=(
            REPOSITORY / "data/inference/tmwa-causal16-alpha4-ew1-6000-endpoint-exact-v1/"
            "inference-report.json"
        ),
        alpha_calibration=(reports / "tmwa-causal16-alpha4-6000-hard-alpha-calibration-v1.json"),
        palette_calibration=(reports / "tmwa-causal16-alpha4-6000-palette-calibration-v1.json"),
        output=(REPOSITORY / "data/inference/tmwa-causal16-alpha4-ew1-6000-endpoint-decoded-v1"),
        expected_source_report_sha256=(
            "7fef539de909b6612de95c40168c98de02f25f93078433adcd6ec529f26e01de"
        ),
        expected_inference_report_sha256=(
            "7c454da198b50b25ebaff320a3fcf85f2e02cdf009574f5c2cbfa6da033db20d"
        ),
        expected_alpha_calibration_sha256=(
            "5304bb27cf986f2c2abca1f2e8474eb9b175cc825e8f93c5aff9c63f0990c2d0"
        ),
        expected_palette_calibration_sha256=(
            "8fda9c5a0d0cba5f49e6c83765cf1ea84d1103daed2ad232b136581dfd67078b"
        ),
        hard_alpha_threshold=144,
        palette_sizes=(24, 64),
    )


if __name__ == "__main__":
    main()
