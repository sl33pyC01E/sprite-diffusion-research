"""Generate the fixed held-out-identity SD1.4-LoRA step-2,500 panel."""

from __future__ import annotations

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
from spritelab.still_comparison import build_sd_lora_target_comparison  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

CHECKPOINT = (
    ROOT
    / "data/experiments/mugen-mffa-sd14-attention-lora-control-v1-10000"
    / "training-step-0002500.pt"
)
PLAN = ROOT / "data/processed/mugen-mffa-latent-still-sequence-plan-v1.json"
MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
SOURCE_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"
INFERENCE = ROOT / "data/inference/mugen-mffa-sd14-lora-step2500-heldout-v1"
COMPARISON = ROOT / "data/inference/mugen-mffa-sd14-lora-step2500-heldout-comparison-v1"
IDENTITIES = (
    "mugen_6602b0aa83934ced_5950baeb1fad85d9",
    "mugen_303702787c067e97_27db79521ab654b5",
    "mugen_cf873e2ef7bdeb2e_35c02d5014d6f05d",
    "mugen_effe528cb5b4dde7_c408c72742ec46e1",
)


def main() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    prompt_by_identity = {}
    for record in plan["records"]:
        identity = record["identity_id"]
        if identity in IDENTITIES:
            prompt_by_identity.setdefault(identity, record["prompt"])
    if set(prompt_by_identity) != set(IDENTITIES):
        raise RuntimeError("fixed validation identities are absent from plan")
    prompts = [prompt_by_identity[identity] for identity in IDENTITIES]
    checkpoint_sha256 = _file_sha256(CHECKPOINT)
    report, report_sha256 = run_sd14_lora_inference(
        CHECKPOINT,
        MODEL,
        prompts,
        INFERENCE,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        config=SDLoraInferenceConfig(),
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    comparison, comparison_sha256 = build_sd_lora_target_comparison(
        report,
        PLAN,
        COMPARISON,
        expected_inference_report_sha256=report_sha256,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint_sha256,
                "comparison_report": str(comparison),
                "comparison_report_sha256": comparison_sha256,
                "inference_report": str(report),
                "inference_report_sha256": report_sha256,
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
    main()
