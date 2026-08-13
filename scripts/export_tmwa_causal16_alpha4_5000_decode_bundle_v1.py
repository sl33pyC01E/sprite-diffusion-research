"""Export the calibrated TMWA alpha4 step-5,000 display bundle."""

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
            / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-5000/overfit-report.json"
        ),
        inference_report=(
            REPOSITORY / "data/inference/tmwa-causal16-alpha4-ew1-5000-endpoint-exact-v1/"
            "inference-report.json"
        ),
        alpha_calibration=(reports / "tmwa-causal16-alpha4-5000-hard-alpha-calibration-v1.json"),
        palette_calibration=(reports / "tmwa-causal16-alpha4-5000-palette-calibration-v1.json"),
        output=(REPOSITORY / "data/inference/tmwa-causal16-alpha4-ew1-5000-endpoint-decoded-v1"),
        expected_source_report_sha256=(
            "3635c59396ff4f6e44f0e7623bdfa84e75213bdf56034e4a1a90c098897307d7"
        ),
        expected_inference_report_sha256=(
            "c252fa2394a0b4e9d385c52fe36caa2162ec60fffbed70865eb1d68b43f0dc88"
        ),
        expected_alpha_calibration_sha256=(
            "9c7d356297ed6d217f17bd76d6cfc473f3cf63aade6c32f6ba7b2fb43b36260c"
        ),
        expected_palette_calibration_sha256=(
            "c6cdd8290cfb1bc81b37549df1c6ccca852a8a98ce23924a693d49f6dee1674d"
        ),
        hard_alpha_threshold=144,
        palette_sizes=(16, 24),
    )


if __name__ == "__main__":
    main()
