"""Train the high-capacity appearance-only SD1.4 LoRA stage."""

from __future__ import annotations

import argparse
import hashlib
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
STEP1000_OUTPUT = ROOT / "data/experiments/mugen-mffa-sd14-lora-canonical-appearance-v4-step1000"
STEP2500_OUTPUT = (
    ROOT / "data/experiments/mugen-mffa-sd14-lora-canonical-appearance-v4-step2500-continuation-v1"
)


def main(*, stage: str, preflight_only: bool) -> None:
    if stage not in {"1000", "2500"}:
        raise ValueError("stage must be 1000 or 2500")
    stop_after_step = int(stage)
    output = STEP1000_OUTPUT if stage == "1000" else STEP2500_OUTPUT
    resume_checkpoint = None if stage == "1000" else STEP1000_OUTPUT / "training-step-0001000.pt"
    resume_sha256 = None if resume_checkpoint is None else _file_sha256(resume_checkpoint)
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
        "output": str(output),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resume_checkpoint_sha256": resume_sha256,
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
        output,
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=config,
        resume_checkpoint_path=resume_checkpoint,
        expected_resume_sha256=resume_sha256,
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("1000", "2500"), default="1000")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    main(
        stage=arguments.stage,
        preflight_only=arguments.preflight_only,
    )
