"""Build the exact same-noise MUGEN sprite-prior step comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.sprite_postprocess import SpriteDisplayDecodeConfig  # noqa: E402
from spritelab.still_comparison import build_sd_lora_ablation_comparison  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json"
ROWS = (
    (
        "SPRITE PRIOR RAW / 1,000",
        ROOT
        / "data/inference/mugen-mffa-sd-pixelart-lora-canonical-v1-step1000-raw-heldout"
        / "inference-report.json",
        "d1d6e2691121df2569e12bd27fcf3c748b4f0bdeeed2124e640667ff27760b12",
    ),
    (
        "SPRITE PRIOR RAW / 2,500",
        ROOT
        / "data/inference/mugen-mffa-sd-pixelart-lora-canonical-v1-step2500-raw-heldout"
        / "inference-report.json",
        "ac3acd856c58aa9848189a0f61b84c0d8ef0a5f557d21c3be4df8bd0905c206b",
    ),
    (
        "SPRITE PRIOR EMA / 2,500",
        ROOT
        / "data/inference/mugen-mffa-sd-pixelart-lora-canonical-v1-step2500-ema-heldout"
        / "inference-report.json",
        "63307e0e0d2a9ad7075923c37f7802b2cfec78f90499e31efa8cf7299598aa20",
    ),
)
SEQUENCES = (
    "sequence_1f44536f11e1b694307d6fc4398ec235",
    "sequence_52cd84f5a78e3cfa733307422f652db8",
    "sequence_4fffef7cde9adb327e5250567c3e7db5",
    "sequence_1a792bc7806a5a1f910522bc56bc7bd7",
)


def main() -> None:
    report, digest = build_sd_lora_ablation_comparison(
        ROWS,
        PLAN,
        ROOT / "data/inference/mugen-mffa-sd-pixelart-lora-step1000-vs-step2500-comparison",
        target_sequence_ids=SEQUENCES,
        display_decode_config=SpriteDisplayDecodeConfig(),
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"report": str(report), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
