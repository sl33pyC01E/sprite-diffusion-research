"""Compare vanilla and sprite-prior MUGEN LoRAs on exact held-out identities."""

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

REVISION = "8229c9b6e928103f0e657cfe6b14d902cb2101d6"
MODEL = ROOT / f"data/models/sd-pixelart-spritesheet-{REVISION}"
SOURCE_INDEX_SHA256 = "5f7eea291d7831ccb4d6bb07b011669f532ebd4371dee35b8743ea90fbc926df"
PLAN = ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json"
CHECKPOINTS = {
    "1000": ROOT
    / "data/experiments/mugen-mffa-sd-pixelart-lora-canonical-v1-step1000"
    / "training-step-0001000.pt",
    "2500": ROOT
    / "data/experiments/mugen-mffa-sd-pixelart-lora-canonical-v1-step2500-continuation-v1"
    / "training-step-0002500.pt",
}
VANILLA_RAW_REPORT = (
    ROOT
    / "data/inference/mugen-mffa-sd14-lora-canonical-appearance-v4-step1000-raw-heldout"
    / "inference-report.json"
)
VANILLA_RAW_REPORT_SHA256 = "386d8091d4ccf18ff8391226782297dde047cd5a75b9aa8a568f7a3ca7663ef5"
IDENTITIES = (
    "mugen_6602b0aa83934ced_5950baeb1fad85d9",
    "mugen_303702787c067e97_27db79521ab654b5",
    "mugen_cf873e2ef7bdeb2e_35c02d5014d6f05d",
    "mugen_effe528cb5b4dde7_c408c72742ec46e1",
)


def main(*, stage: str, preflight_only: bool) -> None:
    if stage not in CHECKPOINTS:
        raise ValueError("stage must be 1000 or 2500")
    checkpoint = CHECKPOINTS[stage]
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    by_identity = {record["identity_id"]: record for record in plan["records"]}
    selected = []
    for identity in IDENTITIES:
        record = by_identity.get(identity)
        if record is None or record["split"] == "train":
            raise RuntimeError(f"held-out identity contract differs: {identity}")
        selected.append(record)
    prompts = [record["prompt"] for record in selected]
    selection = {
        "checkpoint": str(checkpoint),
        "prompts": prompts,
        "selected_sequences": [record["sequence_id"] for record in selected],
        "stage": stage,
    }
    if preflight_only:
        print(json.dumps(selection, sort_keys=True))
        return
    if _file_sha256(VANILLA_RAW_REPORT) != VANILLA_RAW_REPORT_SHA256:
        raise RuntimeError("vanilla raw comparison report differs")
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    ema_output = ROOT / (
        f"data/inference/mugen-mffa-sd-pixelart-lora-canonical-v1-step{stage}-ema-heldout"
    )
    raw_output = ROOT / (
        f"data/inference/mugen-mffa-sd-pixelart-lora-canonical-v1-step{stage}-raw-heldout"
    )
    ema_report, ema_sha256 = run_sd14_lora_inference(
        checkpoint,
        MODEL,
        prompts,
        ema_output,
        expected_checkpoint_sha256=_file_sha256(checkpoint),
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=SDLoraInferenceConfig(seed=20260823, weights_variant="ema"),
        disk_guard=guard,
    )
    raw_report, raw_sha256 = run_sd14_lora_inference(
        checkpoint,
        MODEL,
        prompts,
        raw_output,
        expected_checkpoint_sha256=_file_sha256(checkpoint),
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=SDLoraInferenceConfig(seed=20260823, weights_variant="raw"),
        disk_guard=guard,
    )
    noise_hashes = {
        _noise_sha256(VANILLA_RAW_REPORT),
        _noise_sha256(ema_report),
        _noise_sha256(raw_report),
    }
    if len(noise_hashes) != 1:
        raise RuntimeError("vanilla and sprite-prior inference noise differs")
    display_config = SpriteDisplayDecodeConfig()
    ema_display, ema_display_sha256 = export_inference_sprite_display_bundle(
        ema_report,
        ROOT / f"data/inference/mugen-mffa-sd-pixelart-lora-canonical-v1-step{stage}-ema-display",
        expected_inference_report_sha256=ema_sha256,
        config=display_config,
        disk_guard=guard,
    )
    raw_display, raw_display_sha256 = export_inference_sprite_display_bundle(
        raw_report,
        ROOT / f"data/inference/mugen-mffa-sd-pixelart-lora-canonical-v1-step{stage}-raw-display",
        expected_inference_report_sha256=raw_sha256,
        config=display_config,
        disk_guard=guard,
    )
    comparison, comparison_sha256 = build_sd_lora_ablation_comparison(
        [
            (
                "VANILLA SD RAW / 1,000",
                VANILLA_RAW_REPORT,
                VANILLA_RAW_REPORT_SHA256,
            ),
            (f"SPRITE PRIOR RAW / {int(stage):,}", raw_report, raw_sha256),
            (f"SPRITE PRIOR EMA / {int(stage):,}", ema_report, ema_sha256),
        ],
        PLAN,
        ROOT / f"data/inference/mugen-mffa-vanilla-vs-sprite-prior-lora-step{stage}-comparison",
        target_sequence_ids=selection["selected_sequences"],
        display_decode_config=display_config,
        disk_guard=guard,
    )
    print(
        json.dumps(
            {
                "comparison": str(comparison),
                "comparison_sha256": comparison_sha256,
                "ema_display": str(ema_display),
                "ema_display_sha256": ema_display_sha256,
                "ema_report_sha256": ema_sha256,
                "noise_batch_sha256": next(iter(noise_hashes)),
                "raw_display": str(raw_display),
                "raw_display_sha256": raw_display_sha256,
                "raw_report_sha256": raw_sha256,
                **selection,
            },
            sort_keys=True,
        )
    )


def _noise_sha256(path: Path) -> str:
    report = json.loads(path.read_text(encoding="utf-8"))
    value = report.get("noise_batch_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"inference noise hash is invalid: {path}")
    return value


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
