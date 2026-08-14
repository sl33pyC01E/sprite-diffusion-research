"""Run the first same-subject reference-conditioned MUGEN motion overfit."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_motion_overfit import (  # noqa: E402
    LatentMotionOverfitConfig,
    load_motion_overfit_corpus,
    run_motion_overfit,
)
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-reference-motion-plan-v2.json"
IDENTITY = "mugen_736300dbce136df7_5d0d3dd2a2377512"
VERBS = ("idle", "walk", "run", "block", "normal_attack", "hurt")


def main(
    *,
    steps: int,
    learning_rate: float,
    base_weight: float,
    endpoint_weight: float,
    pixel_endpoint_weight: float,
    output_name: str | None,
    initial_checkpoint: str | None,
    expected_initial_checkpoint_sha256: str | None,
    preflight_only: bool,
) -> None:
    config = LatentMotionOverfitConfig(
        steps=steps,
        learning_rate=learning_rate,
        base_weight=base_weight,
        endpoint_weight=endpoint_weight,
        pixel_endpoint_weight=pixel_endpoint_weight,
    )
    default_name = f"mugen-reference-motion-736300-six-action-v1-step{steps}"
    output = ROOT / "data/experiments" / (output_name or default_name)
    corpus = load_motion_overfit_corpus(PLAN, identity_id=IDENTITY, verbs=VERBS)
    preflight = {
        "config": asdict(config),
        "identity_id": corpus.identity_id,
        "output": str(output),
        "plan_file_sha256": corpus.plan_file_sha256,
        "sequence_ids": [record["sequence_id"] for record in corpus.records],
        "verbs": list(corpus.verbs),
    }
    if preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    report, digest = run_motion_overfit(
        PLAN,
        output,
        identity_id=IDENTITY,
        verbs=VERBS,
        config=config,
        initial_checkpoint_path=initial_checkpoint,
        expected_initial_checkpoint_sha256=expected_initial_checkpoint_sha256,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"report": str(report), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--base-weight", type=float, default=1.0)
    parser.add_argument("--endpoint-weight", type=float, default=1.0)
    parser.add_argument("--pixel-endpoint-weight", type=float, default=0.0)
    parser.add_argument("--output-name")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--expected-initial-checkpoint-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    main(
        steps=arguments.steps,
        learning_rate=arguments.learning_rate,
        base_weight=arguments.base_weight,
        endpoint_weight=arguments.endpoint_weight,
        pixel_endpoint_weight=arguments.pixel_endpoint_weight,
        output_name=arguments.output_name,
        initial_checkpoint=arguments.initial_checkpoint,
        expected_initial_checkpoint_sha256=arguments.expected_initial_checkpoint_sha256,
        preflight_only=arguments.preflight_only,
    )
