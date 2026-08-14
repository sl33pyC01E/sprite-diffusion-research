"""Generate a deterministic train/validation/test gallery for the scratch still DiT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_still_inference import (  # noqa: E402
    LatentStillInferenceConfig,
    run_latent_still_inference,
)
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

DEFAULT_TEXT_MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
TEXT_SOURCE_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.per_split <= 0:
        raise ValueError("--per-split must be positive")
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if not isinstance(plan, dict) or plan.get("artifact_kind") != (
        "mugen_latent_still_sequence_training_plan"
    ):
        raise ValueError("still plan has the wrong artifact kind")
    records = plan.get("records")
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ValueError("still plan records are invalid")
    selected = []
    for split in ("train", "validation", "test"):
        candidates = [record for record in records if record.get("split") == split]
        candidates.sort(
            key=lambda record: hashlib.sha256(
                (str(record.get("identity_id")) + "\0" + str(record.get("prompt"))).encode()
            ).digest()
        )
        selected.extend(candidates[: args.per_split])
    if not selected:
        raise ValueError("still plan has no supported split records")
    prompts = [str(record["prompt"]) for record in selected]
    if len(set(prompts)) != len(prompts):
        raise ValueError("selected still prompts are not unique")
    report_path, report_sha256 = run_latent_still_inference(
        args.checkpoint,
        args.codec_checkpoint,
        args.text_model,
        prompts,
        args.output,
        expected_checkpoint_sha256=_file_sha256(args.checkpoint),
        expected_codec_checkpoint_sha256=_file_sha256(args.codec_checkpoint),
        expected_text_source_index_sha256=TEXT_SOURCE_INDEX_SHA256,
        config=LatentStillInferenceConfig(
            seed=args.seed,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
        ),
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    selection = {
        "artifact_kind": "mugen_latent_still_split_inference_selection",
        "inference_report": {
            "file_sha256": report_sha256,
            "path": report_path.name,
        },
        "plan": {
            "file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "path": str(args.plan.resolve()),
        },
        "records": [
            {
                "identity_id": str(record["identity_id"]),
                "prompt": str(record["prompt"]),
                "sample_index": index,
                "split": str(record["split"]),
            }
            for index, record in enumerate(selected)
        ],
        "schema_version": 1,
    }
    selection_path = args.output / "selection.json"
    with selection_path.open("xb") as handle:
        handle.write(canonical_json_bytes(selection))
    print(
        json.dumps(
            {
                "inference_report": str(report_path),
                "inference_report_sha256": report_sha256,
                "selection": str(selection_path),
                "selection_sha256": _file_sha256(selection_path),
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
