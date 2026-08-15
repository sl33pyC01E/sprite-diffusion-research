"""Train the fixed-middle MUGEN key-pose stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_keypose_train import (  # noqa: E402
    LatentKeyposeTrainingConfig,
    build_keypose_action_bundles,
    run_latent_keypose_training,
)
from spritelab.latent_motion_train import load_latent_motion_training_corpus  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=(
            "endpoint30000",
            "direct30000",
            "identity-unet-direct30000",
            "identity-unet-flow30000",
        ),
        default="direct30000",
    )
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--expected-resume-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if (args.resume_checkpoint is None) != (args.expected_resume_sha256 is None):
        parser.error("resume checkpoint and expected SHA-256 must be supplied together")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace key-pose output: {output}")
    flow_profile = args.profile == "identity-unet-flow30000"
    config = LatentKeyposeTrainingConfig(
        prediction_mode=(
            "continuous_flow"
            if flow_profile
            else "endpoint_flow"
            if args.profile == "endpoint30000"
            else "direct_residual"
        ),
        model_architecture=(
            "identity_unet"
            if args.profile in {"identity-unet-direct30000", "identity-unet-flow30000"}
            else "dit"
        ),
        checkpoint_every=10_000 if flow_profile else 2_500,
        validate_every=1_000 if flow_profile else 500,
        validation_identities=4 if flow_profile else 16,
    )
    corpus = load_latent_motion_training_corpus(
        args.manifest, verify_hashes=True, array_loading="lazy"
    )
    train_bundles = build_keypose_action_bundles(corpus, corpus.train_indices)
    validation_bundles = build_keypose_action_bundles(corpus, corpus.validation_indices)
    resume = None
    if args.resume_checkpoint is not None:
        checkpoint = args.resume_checkpoint.resolve()
        actual = file_sha256(checkpoint)
        if actual != args.expected_resume_sha256:
            raise ValueError("resume checkpoint SHA-256 differs")
        resume = {"path": str(checkpoint), "sha256": actual}
    preflight = {
        "action_vocabulary": list(corpus.action_vocabulary),
        "config": asdict(config),
        "corpus": corpus.contract,
        "fixed_anchor_contract": {
            "canonical_middle_frame_index": config.keypose_frame_index,
            "inference_frame_index_is_not_predicted": True,
            "reference_condition": "same identity idle reference",
        },
        "output": str(output),
        "resume": resume,
        "train_action_bundles": len(train_bundles),
        "validation_action_bundles": len(validation_bundles),
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    result = run_latent_keypose_training(
        args.manifest,
        output,
        config=config,
        resume_checkpoint_path=args.resume_checkpoint,
        expected_resume_sha256=args.expected_resume_sha256,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(result.checkpoint_path),
                "output": str(result.output_directory),
                "report": str(result.report_path),
                "report_sha256": result.report_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
