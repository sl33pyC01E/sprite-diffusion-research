"""Run paired endpoint refinements from one exact broad MUGEN motion checkpoint."""

from __future__ import annotations

import argparse
import hashlib
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


def refinement_config(profile: str) -> LatentMotionTrainingConfig:
    """Return paired control/action configs that differ only in contrast weight."""

    profiles = {
        "endpoint-control3000": (3_000, 0.0, 0.0, "pair", 4),
        "endpoint-action3000": (3_000, 1.0, 0.0, "pair", 4),
        "endpoint-pixel-action3000": (3_000, 0.0, 0.5, "pair", 4),
        "endpoint-pixel-action10000": (10_000, 0.0, 0.5, "pair", 4),
        "endpoint-pixel-action-bundle3000": (3_000, 0.0, 0.5, "bundle", 2),
    }
    try:
        steps, action_weight, pixel_action_weight, batch_mode, accumulation = profiles[profile]
    except KeyError as error:
        raise ValueError(f"unsupported refinement profile: {profile}") from error
    return LatentMotionTrainingConfig(
        gradient_accumulation=accumulation,
        learning_rate=5e-5,
        minimum_learning_rate=5e-6,
        warmup_steps=200,
        ema_decay=0.999,
        latent_endpoint_weight=1.0,
        pixel_endpoint_weight=2.0,
        action_contrast_weight=action_weight,
        pixel_action_contrast_weight=pixel_action_weight,
        action_batch_mode=batch_mode,
        time_sampling="endpoint",
        endpoint_sample_probability=0.0,
        inference_steps=1,
        sampler_algorithm="euler",
        steps=steps,
        validate_every=500,
        checkpoint_every=500 if steps == 3_000 else 5_000,
        validation_pairs=16,
        preview_pairs=6,
        seed=20260828,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        choices=(
            "endpoint-control3000",
            "endpoint-action3000",
            "endpoint-pixel-action3000",
            "endpoint-pixel-action10000",
            "endpoint-pixel-action-bundle3000",
        ),
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-parent-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    checkpoint = args.parent_checkpoint.resolve()
    actual_sha256 = file_sha256(checkpoint)
    if actual_sha256 != args.expected_parent_sha256:
        raise ValueError("parent checkpoint SHA-256 differs")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace refinement output: {output}")
    config = refinement_config(args.profile)
    corpus = load_latent_motion_training_corpus(
        args.manifest,
        verify_hashes=True,
        array_loading="lazy",
    )
    matched = build_matched_action_index(corpus.rows, corpus.train_indices)
    preflight = {
        "config": asdict(config),
        "corpus": corpus.contract,
        "matched_train_identities": len(matched),
        "output": str(output),
        "parent": {
            "path": str(checkpoint),
            "sha256": actual_sha256,
        },
        "profile": args.profile,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    result = run_latent_motion_training(
        args.manifest,
        output,
        config=config,
        warm_start_checkpoint_path=checkpoint,
        expected_warm_start_sha256=actual_sha256,
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
    main()
