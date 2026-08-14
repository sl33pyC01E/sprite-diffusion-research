"""Run the canonical broad MUGEN reference-plus-action latent-motion model."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_motion_train import (  # noqa: E402
    LatentMotionTrainingConfig,
    build_matched_action_index,
    load_latent_motion_training_corpus,
    run_latent_motion_training,
)
from spritelab.storage import DiskGuard  # noqa: E402

MANIFEST = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"


def main(*, profile: str, preflight_only: bool) -> None:
    if profile == "smoke":
        config = LatentMotionTrainingConfig(
            gradient_accumulation=1,
            warmup_steps=0,
            steps=1,
            log_every=1,
            validate_every=1,
            checkpoint_every=1,
            validation_pairs=1,
            preview_pairs=1,
        )
        output_name = "mugen-primary-motion-broad-gpu-smoke-v1-step1"
    elif profile == "pilot250":
        config = LatentMotionTrainingConfig(
            warmup_steps=25,
            steps=250,
            validate_every=250,
            checkpoint_every=250,
            validation_pairs=4,
            preview_pairs=2,
        )
        output_name = "mugen-primary-motion-broad-pilot-v1-step250"
    elif profile == "baseline15000":
        config = LatentMotionTrainingConfig()
        output_name = "mugen-primary-motion-broad-v2-step15000"
    else:
        raise ValueError(f"unsupported profile: {profile}")
    corpus = load_latent_motion_training_corpus(MANIFEST, verify_hashes=True)
    matched = build_matched_action_index(corpus.rows, corpus.train_indices)
    output = ROOT / "data/experiments" / output_name
    preflight = {
        "action_vocabulary": list(corpus.action_vocabulary),
        "config": asdict(config),
        "corpus": corpus.contract,
        "matched_train_identities": len(matched),
        "matched_train_rows": sum(len(verbs) for verbs in matched.values()),
        "output": str(output),
    }
    if preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    result = run_latent_motion_training(
        MANIFEST,
        output,
        config=config,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(
        json.dumps(
            {
                "inference_checkpoint": str(result.inference_checkpoint_path),
                "report": str(result.report_path),
                "report_sha256": result.report_sha256,
                "training_checkpoint": str(result.training_checkpoint_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile", choices=("smoke", "pilot250", "baseline15000"), default="smoke"
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    main(profile=args.profile, preflight_only=args.preflight_only)
