"""Generate a deterministic train/validation/test gallery for the scratch still DiT."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

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
    target_previews = [
        _export_target_preview(record, args.output, index=index, plan_root=args.plan.parent)
        for index, record in enumerate(selected)
    ]
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
                "target": target_previews[index],
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


def _export_target_preview(
    record: dict[str, object],
    output: Path,
    *,
    index: int,
    plan_root: Path,
) -> dict[str, object]:
    target = record.get("target")
    if not isinstance(target, dict):
        raise ValueError("still target must be an object")
    relative = target.get("relative_path")
    file_sha256 = target.get("file_sha256")
    array_sha256 = target.get("array_content_sha256")
    eligible = target.get("eligible_frame_indices")
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(file_sha256, str)
        or not isinstance(array_sha256, str)
        or not isinstance(eligible, list)
        or len(eligible) != 1
        or isinstance(eligible[0], bool)
        or not isinstance(eligible[0], int)
        or not 0 <= eligible[0] < 8
    ):
        raise ValueError("still target evidence is invalid")
    source = (plan_root / relative).resolve()
    plan_root_resolved = plan_root.resolve()
    if plan_root_resolved != source and plan_root_resolved not in source.parents:
        raise ValueError("still target escapes the plan root")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != file_sha256:
        raise ValueError("still target file SHA-256 differs")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise ValueError("still target array geometry differs")
    value = np.ascontiguousarray(value)
    if _array_sha256(value) != array_sha256:
        raise ValueError("still target array SHA-256 differs")
    frame_index = eligible[0]
    frame = np.ascontiguousarray(value[frame_index])
    prompt = str(record["prompt"])
    stem = f"{index:03d}-{hashlib.sha256(prompt.encode()).hexdigest()[:10]}-target"
    native = output / f"{stem}-native.png"
    preview = output / f"{stem}-preview.png"
    image = Image.fromarray(frame)
    image.save(native, format="PNG", optimize=False)
    image.resize((512, 512), resample=Image.Resampling.NEAREST).save(
        preview, format="PNG", optimize=False
    )
    return {
        "array_content_sha256": _array_sha256(frame),
        "frame_index": frame_index,
        "native_png": {"file_sha256": _file_sha256(native), "path": native.name},
        "preview_png": {
            "display_only_nearest_neighbor_scale": 4,
            "file_sha256": _file_sha256(preview),
            "path": preview.name,
        },
        "sequence_id": str(record["sequence_id"]),
        "source_array_content_sha256": array_sha256,
        "source_file_sha256": file_sha256,
    }


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


if __name__ == "__main__":
    main()
