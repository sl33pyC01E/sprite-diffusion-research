"""Evaluate a pinned latent sprite-sheet checkpoint on held-out MUGEN descriptions."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.sprite_postprocess import (  # noqa: E402
    composite_rgba_on_checkerboard,
    decode_generated_rgb_sprite,
)
from spritelab.storage import DiskGuard  # noqa: E402

REVISION = "8229c9b6e928103f0e657cfe6b14d902cb2101d6"
MODEL = ROOT / f"data/models/sd-pixelart-spritesheet-{REVISION}"
SOURCE_INDEX = ROOT / "data/index/reports/sd-pixelart-spritesheet-source-index-v1.json"
SOURCE_INDEX_SHA256 = "fd3d6898d01901256215ee04e19142d9d36ec32ae7be1fc0ca09101239233167"
PLAN = ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json"
OUTPUT = ROOT / "data/inference/sd-pixelart-spritesheet-mugen-heldout-v1"
IDENTITIES = (
    "mugen_6602b0aa83934ced_5950baeb1fad85d9",
    "mugen_303702787c067e97_27db79521ab654b5",
    "mugen_cf873e2ef7bdeb2e_35c02d5014d6f05d",
    "mugen_effe528cb5b4dde7_c408c72742ec46e1",
)
SEED = 20260825
NEGATIVE_PROMPT = (
    "photo, realistic, 3d render, blurry, soft edges, noisy, detailed background, text, watermark"
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace baseline evaluation: {OUTPUT}")
    if _file_sha256(SOURCE_INDEX) != SOURCE_INDEX_SHA256:
        raise RuntimeError("sprite-sheet source index differs")
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    materialization = Path(plan["source"]["materialization_path"]).resolve()
    if _file_sha256(materialization) != plan["source"]["materialization_file_sha256"]:
        raise RuntimeError("MUGEN materialization differs")
    by_identity = {record["identity_id"]: record for record in plan["records"]}
    records = []
    for identity in IDENTITIES:
        record = by_identity.get(identity)
        if record is None or record["split"] == "train":
            raise RuntimeError(f"held-out identity contract differs: {identity}")
        records.append(record)
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    guard.require_capacity(512 * 1024**2, label="sprite-sheet baseline evaluation")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{OUTPUT.name}.", suffix=".staging", dir=OUTPUT.parent)
    )
    try:
        from diffusers import StableDiffusionPipeline

        pipeline = StableDiffusionPipeline.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None,
            local_files_only=True,
        ).to("cuda")
        pipeline.set_progress_bar_config(disable=True)
        rows = []
        gallery_rows = []
        for index, record in enumerate(records):
            target = record["target"]
            target_path = (materialization.parent / target["relative_path"]).resolve()
            target_bytes = target_path.read_bytes()
            if hashlib.sha256(target_bytes).hexdigest() != target["file_sha256"]:
                raise RuntimeError("exact MUGEN target bytes differ")
            target_array = np.load(io.BytesIO(target_bytes), allow_pickle=False)
            frame_index = target["eligible_frame_indices"][0]
            target_rgba = np.ascontiguousarray(target_array[frame_index])
            generation_prompt = f"PixelartRSS, {record['prompt']}"
            generator = torch.Generator(device="cuda").manual_seed(SEED)
            generated = (
                pipeline(
                    generation_prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    generator=generator,
                )
                .images[0]
                .convert("RGB")
            )
            raw_rgb = np.asarray(generated, dtype=np.uint8)
            reduced_rgb = np.asarray(
                generated.resize((128, 128), Image.Resampling.BOX), dtype=np.uint8
            )
            decoded_rgba, decode_metadata = decode_generated_rgb_sprite(reduced_rgb)
            stem = f"{index:02d}-{hashlib.sha256(record['identity_id'].encode()).hexdigest()[:10]}"
            target_output = staging / f"{stem}-exact-target.png"
            raw_output = staging / f"{stem}-generated-512.png"
            reduced_output = staging / f"{stem}-generated-128.png"
            decoded_output = staging / f"{stem}-generated-display-rgba.png"
            Image.fromarray(target_rgba).save(target_output, optimize=False)
            Image.fromarray(raw_rgb).save(raw_output, optimize=False)
            Image.fromarray(reduced_rgb).save(reduced_output, optimize=False)
            Image.fromarray(decoded_rgba).save(decoded_output, optimize=False)
            rows.append(
                {
                    "decode": decode_metadata,
                    "generated": {
                        "display_rgba_file_sha256": _file_sha256(decoded_output),
                        "display_rgba_path": decoded_output.name,
                        "raw_512_file_sha256": _file_sha256(raw_output),
                        "raw_512_path": raw_output.name,
                        "reduced_128_file_sha256": _file_sha256(reduced_output),
                        "reduced_128_path": reduced_output.name,
                    },
                    "generation_prompt": generation_prompt,
                    "identity_id": record["identity_id"],
                    "sequence_id": record["sequence_id"],
                    "split": record["split"],
                    "target": {
                        "file_sha256": _file_sha256(target_output),
                        "frame_index": frame_index,
                        "path": target_output.name,
                    },
                }
            )
            gallery_rows.append((target_rgba, reduced_rgb, decoded_rgba, record["prompt"]))
        gallery_path = staging / "exact-target-vs-sprite-checkpoint-gallery.png"
        _gallery(gallery_rows, gallery_path)
        report = {
            "artifact_kind": "external_latent_sprite_checkpoint_mugen_heldout_evaluation",
            "claim": (
                "identity-held-out prompt baseline; no MUGEN fine-tuning; generated display "
                "transparency is inferred and noncanonical"
            ),
            "config": {
                "guidance_scale": 7.5,
                "negative_prompt": NEGATIVE_PROMPT,
                "num_inference_steps": 50,
                "seed_reused_per_prompt_for_matched_initial_noise": SEED,
                "trigger_token": "PixelartRSS",
            },
            "gallery": {"file_sha256": _file_sha256(gallery_path), "path": gallery_path.name},
            "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "rows": rows,
            "schema_version": 1,
            "source_index_file_sha256": SOURCE_INDEX_SHA256,
        }
        payload = _canonical_json(report)
        report_path = staging / "evaluation-report.json"
        with report_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(staging, OUTPUT)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(
        json.dumps(
            {
                "gallery": str(OUTPUT / gallery_path.name),
                "report": str(OUTPUT / report_path.name),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def _gallery(rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]], path: Path) -> None:
    width = 1600
    row_height = 500
    canvas = Image.new("RGB", (width, row_height * len(rows)), (25, 27, 32))
    draw = ImageDraw.Draw(canvas)
    for index, (target, raw, decoded, prompt) in enumerate(rows):
        top = index * row_height
        panels = (
            composite_rgba_on_checkerboard(target),
            raw,
            composite_rgba_on_checkerboard(decoded),
        )
        labels = ("EXACT MUGEN TARGET", "PRETRAINED RAW 128", "DISPLAY DECODE")
        for column, (panel, label) in enumerate(zip(panels, labels, strict=True)):
            left = 32 + column * 520
            image = Image.fromarray(panel).resize((448, 448), Image.Resampling.NEAREST)
            canvas.paste(image, (left, top + 28))
            draw.text((left, top + 8), label, fill=(220, 226, 236))
        draw.text((32, top + 478), _wrap(prompt, 170)[0], fill=(178, 188, 204))
    canvas.save(path, format="PNG", optimize=False)


def _wrap(value: str, maximum: int) -> list[str]:
    output = []
    line = ""
    for word in value.split():
        candidate = f"{line} {word}".strip()
        if line and len(candidate) > maximum:
            output.append(line)
            line = word
        else:
            line = candidate
    if line:
        output.append(line)
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


if __name__ == "__main__":
    main()
