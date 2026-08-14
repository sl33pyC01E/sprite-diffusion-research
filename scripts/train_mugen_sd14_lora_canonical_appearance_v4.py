"""Train the high-capacity appearance-only SD1.4 LoRA stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.sd_lora_train import (  # noqa: E402
    SDLoraTrainingConfig,
    load_sd_lora_corpus,
    run_sd14_lora_training,
)
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json"
LATENTS = ROOT / "data/processed/mugen-mffa-sd14-rgb-vae-latents-v1/manifest.json"
TEXT = ROOT / "data/processed/mugen-mffa-sd14-clip-token-states-canonical-v4/manifest.json"
MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
SOURCE_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"
OUTPUT = ROOT / "data/experiments/mugen-mffa-sd14-lora-canonical-appearance-v4-10000"


def main(*, stop_after_step: int, preflight_only: bool) -> None:
    config = SDLoraTrainingConfig(
        rank=32,
        alpha=32,
        target_profile="attention_resnet",
    )
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    corpus = load_sd_lora_corpus(PLAN, LATENTS, TEXT)
    preflight = {
        "config": {
            "alpha": config.alpha,
            "rank": config.rank,
            "target_profile": config.target_profile,
        },
        "corpus_contract": corpus.contract,
        "disk_writable_budget_bytes": guard.status().writable_budget_bytes,
        "output": str(OUTPUT),
        "stop_after_step": stop_after_step,
    }
    if preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    result = run_sd14_lora_training(
        PLAN,
        LATENTS,
        TEXT,
        MODEL,
        OUTPUT,
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=config,
        stop_after_step=stop_after_step,
        disk_guard=guard,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-after-step", type=int, default=2500)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    main(
        stop_after_step=arguments.stop_after_step,
        preflight_only=arguments.preflight_only,
    )
