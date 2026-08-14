"""Train a higher-capacity SD1.4 LoRA on subject-bearing MUGEN stills."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.sd_lora_train import (  # noqa: E402
    SDLoraTrainingConfig,
    run_sd14_lora_training,
)
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-latent-still-sequence-plan-v2-subject-bearing.json"
LATENTS = ROOT / "data/processed/mugen-mffa-sd14-rgb-vae-latents-v1/manifest.json"
TEXT = ROOT / "data/processed/mugen-mffa-sd14-clip-token-states-v2-subject-bearing/manifest.json"
MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
SOURCE_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"
OUTPUT = ROOT / "data/experiments/mugen-mffa-sd14-lora-subject-bearing-high-capacity-v3-10000"


def main() -> None:
    result = run_sd14_lora_training(
        PLAN,
        LATENTS,
        TEXT,
        MODEL,
        OUTPUT,
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=SDLoraTrainingConfig(
            rank=32,
            alpha=32,
            target_profile="attention_resnet",
        ),
        stop_after_step=2500,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(
        json.dumps(
            {
                "checkpoint": str(result.checkpoint_path),
                "report": str(result.report_path),
                "report_sha256": result.report_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
