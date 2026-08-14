"""Run a same-prompt/noise dirty-v1 versus subject-bearing-v2 comparison."""

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

DIRTY_CHECKPOINT = (
    ROOT
    / "data/experiments/mugen-mffa-sd14-attention-lora-control-v1-10000"
    / "training-step-0002500.pt"
)
CLEAN_CHECKPOINT = (
    ROOT
    / "data/experiments/mugen-mffa-sd14-lora-subject-bearing-v2-10000"
    / "training-step-0002500.pt"
)
PLAN = ROOT / "data/processed/mugen-mffa-latent-still-sequence-plan-v2-subject-bearing.json"
MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
SOURCE_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"
DIRTY_INFERENCE = ROOT / "data/inference/mugen-mffa-sd14-lora-dirty-v1-step2500-v2-heldout-prompts"
CLEAN_INFERENCE = ROOT / "data/inference/mugen-mffa-sd14-lora-subject-bearing-v2-step2500-heldout"
COMPARISON = ROOT / "data/inference/mugen-mffa-sd14-lora-dirty-v1-vs-subject-bearing-v2-step2500"
DIRTY_DISPLAY = ROOT / "data/inference/mugen-mffa-sd14-lora-dirty-v1-step2500-display-v1"
CLEAN_DISPLAY = ROOT / "data/inference/mugen-mffa-sd14-lora-subject-bearing-v2-step2500-display-v1"
IDENTITIES = (
    "mugen_6602b0aa83934ced_5950baeb1fad85d9",
    "mugen_303702787c067e97_27db79521ab654b5",
    "mugen_cf873e2ef7bdeb2e_35c02d5014d6f05d",
    "mugen_effe528cb5b4dde7_c408c72742ec46e1",
)
VERB_PRIORITY = {
    "idle": 0,
    "walk": 1,
    "block": 2,
    "normal_attack": 3,
    "run": 4,
    "jump": 5,
}


def main(*, preflight_only: bool = False) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    candidates = {identity: [] for identity in IDENTITIES}
    for record in plan["records"]:
        identity = record["identity_id"]
        if identity in candidates:
            if record["split"] == "train":
                raise RuntimeError(f"fixed identity leaked into training: {identity}")
            candidates[identity].append(record)
    selected = []
    for identity in IDENTITIES:
        if not candidates[identity]:
            raise RuntimeError(f"fixed held-out identity is absent: {identity}")
        selected.append(
            min(
                candidates[identity],
                key=lambda record: (
                    VERB_PRIORITY.get(record["conditioning"]["verb"], 99),
                    -len(record["target"]["eligible_frame_indices"]),
                    record["sequence_id"].encode(),
                ),
            )
        )
    prompts = [record["prompt"] for record in selected]
    if len(set(prompts)) != len(prompts):
        raise RuntimeError("fixed held-out prompts are not unique")
    selection = {
        "heldout_splits": [record["split"] for record in selected],
        "prompts": prompts,
        "selected_sequences": [record["sequence_id"] for record in selected],
    }
    if preflight_only:
        print(json.dumps(selection, sort_keys=True))
        return
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    config = SDLoraInferenceConfig(seed=20260822)
    dirty_report, dirty_sha256 = run_sd14_lora_inference(
        DIRTY_CHECKPOINT,
        MODEL,
        prompts,
        DIRTY_INFERENCE,
        expected_checkpoint_sha256=_file_sha256(DIRTY_CHECKPOINT),
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=config,
        disk_guard=guard,
    )
    clean_report, clean_sha256 = run_sd14_lora_inference(
        CLEAN_CHECKPOINT,
        MODEL,
        prompts,
        CLEAN_INFERENCE,
        expected_checkpoint_sha256=_file_sha256(CLEAN_CHECKPOINT),
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=config,
        disk_guard=guard,
    )
    display_config = SpriteDisplayDecodeConfig()
    dirty_display_manifest, dirty_display_sha256 = export_inference_sprite_display_bundle(
        dirty_report,
        DIRTY_DISPLAY,
        expected_inference_report_sha256=dirty_sha256,
        config=display_config,
        disk_guard=guard,
    )
    clean_display_manifest, clean_display_sha256 = export_inference_sprite_display_bundle(
        clean_report,
        CLEAN_DISPLAY,
        expected_inference_report_sha256=clean_sha256,
        config=display_config,
        disk_guard=guard,
    )
    comparison, comparison_sha256 = build_sd_lora_ablation_comparison(
        [
            ("DIRTY V1 / 2,500", dirty_report, dirty_sha256),
            ("SUBJECT-BEARING V2 / 2,500", clean_report, clean_sha256),
        ],
        PLAN,
        COMPARISON,
        target_sequence_ids=[record["sequence_id"] for record in selected],
        display_decode_config=display_config,
        disk_guard=guard,
    )
    print(
        json.dumps(
            {
                "clean_inference_report_sha256": clean_sha256,
                "clean_display_manifest": str(clean_display_manifest),
                "clean_display_manifest_sha256": clean_display_sha256,
                "comparison_report": str(comparison),
                "comparison_report_sha256": comparison_sha256,
                "dirty_display_manifest": str(dirty_display_manifest),
                "dirty_display_manifest_sha256": dirty_display_sha256,
                "dirty_inference_report_sha256": dirty_sha256,
                "heldout_splits": selection["heldout_splits"],
                "selected_sequences": [record["sequence_id"] for record in selected],
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
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    main(preflight_only=arguments.preflight_only)
