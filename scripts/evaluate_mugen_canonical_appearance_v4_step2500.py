"""Compare action-frame and canonical-appearance LoRAs on exact held-out identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.sd_lora_inference import (  # noqa: E402
    SDLoraInferenceConfig,
    run_sd14_lora_inference,
)
from spritelab.sprite_postprocess import (  # noqa: E402
    SpriteDisplayDecodeConfig,
    export_inference_sprite_display_bundle,
)
from spritelab.still_comparison import build_sd_lora_ablation_comparison  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

ACTION_FRAME_CHECKPOINT = (
    ROOT
    / "data/experiments/mugen-mffa-sd14-lora-subject-bearing-v2-10000"
    / "training-step-0002500.pt"
)
CANONICAL_CHECKPOINTS = {
    "1000": ROOT
    / "data/experiments/mugen-mffa-sd14-lora-canonical-appearance-v4-step1000"
    / "training-step-0001000.pt",
    "2500": ROOT
    / "data/experiments/mugen-mffa-sd14-lora-canonical-appearance-v4-step2500-continuation-v1"
    / "training-step-0002500.pt",
}
PLAN = ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json"
MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
SOURCE_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"
IDENTITIES = (
    "mugen_6602b0aa83934ced_5950baeb1fad85d9",
    "mugen_303702787c067e97_27db79521ab654b5",
    "mugen_cf873e2ef7bdeb2e_35c02d5014d6f05d",
    "mugen_effe528cb5b4dde7_c408c72742ec46e1",
)


def main(*, stage: str, preflight_only: bool = False) -> None:
    if stage not in CANONICAL_CHECKPOINTS:
        raise ValueError("stage must be 1000 or 2500")
    canonical_checkpoint = CANONICAL_CHECKPOINTS[stage]
    action_frame_inference = (
        ROOT
        / f"data/inference/mugen-mffa-sd14-lora-action-frame-v2-canonical-step{stage}-control"
    )
    action_frame_display = (
        ROOT
        / f"data/inference/mugen-mffa-sd14-lora-action-frame-v2-canonical-step{stage}-display-v1"
    )
    canonical_inference = (
        ROOT / f"data/inference/mugen-mffa-sd14-lora-canonical-appearance-v4-step{stage}-heldout"
    )
    canonical_display = (
        ROOT / f"data/inference/mugen-mffa-sd14-lora-canonical-appearance-v4-step{stage}-display-v1"
    )
    comparison_output = ROOT / (
        "data/inference/mugen-mffa-sd14-lora-action-frame-v2-vs-"
        f"canonical-appearance-v4-step{stage}"
    )
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    by_identity = {record["identity_id"]: record for record in plan["records"]}
    selected = []
    for identity in IDENTITIES:
        record = by_identity.get(identity)
        if record is None:
            raise RuntimeError(f"fixed held-out identity is absent: {identity}")
        if record["split"] == "train":
            raise RuntimeError(f"fixed identity leaked into training: {identity}")
        selected.append(record)
    prompts = [record["prompt"] for record in selected]
    selection = {
        "canonical_checkpoint": str(canonical_checkpoint),
        "heldout_splits": [record["split"] for record in selected],
        "prompts": prompts,
        "selected_sequences": [record["sequence_id"] for record in selected],
        "stage": stage,
    }
    if preflight_only:
        print(json.dumps(selection, sort_keys=True))
        return
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    config = SDLoraInferenceConfig(seed=20260823)
    action_report, action_sha256 = run_sd14_lora_inference(
        ACTION_FRAME_CHECKPOINT,
        MODEL,
        prompts,
        action_frame_inference,
        expected_checkpoint_sha256=_file_sha256(ACTION_FRAME_CHECKPOINT),
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=config,
        disk_guard=guard,
    )
    canonical_report, canonical_sha256 = run_sd14_lora_inference(
        canonical_checkpoint,
        MODEL,
        prompts,
        canonical_inference,
        expected_checkpoint_sha256=_file_sha256(canonical_checkpoint),
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=config,
        disk_guard=guard,
    )
    display_config = SpriteDisplayDecodeConfig()
    action_display, action_display_sha256 = export_inference_sprite_display_bundle(
        action_report,
        action_frame_display,
        expected_inference_report_sha256=action_sha256,
        config=display_config,
        disk_guard=guard,
    )
    canonical_display, canonical_display_sha256 = export_inference_sprite_display_bundle(
        canonical_report,
        canonical_display,
        expected_inference_report_sha256=canonical_sha256,
        config=display_config,
        disk_guard=guard,
    )
    comparison, comparison_sha256 = build_sd_lora_ablation_comparison(
        [
            ("ACTION-FRAME V2 / 2,500", action_report, action_sha256),
            (
                f"CANONICAL APPEARANCE V4 / {int(stage):,}",
                canonical_report,
                canonical_sha256,
            ),
        ],
        PLAN,
        comparison_output,
        target_sequence_ids=selection["selected_sequences"],
        display_decode_config=display_config,
        disk_guard=guard,
    )
    print(
        json.dumps(
            {
                "action_frame_display_manifest": str(action_display),
                "action_frame_display_manifest_sha256": action_display_sha256,
                "action_frame_inference_report_sha256": action_sha256,
                "canonical_display_manifest": str(canonical_display),
                "canonical_display_manifest_sha256": canonical_display_sha256,
                "canonical_inference_report_sha256": canonical_sha256,
                "comparison_report": str(comparison),
                "comparison_report_sha256": comparison_sha256,
                **selection,
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
    main(stage=arguments.stage, preflight_only=arguments.preflight_only)
