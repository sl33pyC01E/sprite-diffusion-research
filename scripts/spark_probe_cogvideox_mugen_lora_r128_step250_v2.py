"""Evaluate the rank-128 CogVideoX MUGEN LoRA on its exact attack request."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import decord
import numpy as np
import torch
from diffusers import CogVideoXImageToVideoPipeline
from PIL import Image

ROOT = Path("/home/sleepy/sprite-lab-cogvideox")
MODEL = ROOT / "CogVideoX-5b-I2V-a6f0f4858a83"
DATASET = ROOT / "mugen-cogvideox-orange-fighter-i2v-native-caption-v3"
LORA = Path(
    os.environ.get(
        "SPRITELAB_COGVIDEOX_LORA",
        ROOT / "lora-orange-fighter-native-caption-r128-step250-v2",
    )
)
LORA_WEIGHT_NAME = os.environ.get(
    "SPRITELAB_COGVIDEOX_LORA_WEIGHT_NAME", "pytorch_lora_weights.safetensors"
)
TRAINING_STEPS = int(os.environ.get("SPRITELAB_COGVIDEOX_TRAINING_STEPS", "250"))
LORA_SCALE = float(os.environ.get("SPRITELAB_COGVIDEOX_LORA_SCALE", "1.0"))
VERB = os.environ.get("SPRITELAB_COGVIDEOX_VERB", "normal_attack")
TRAIN_LOG = ROOT / "lora-orange-fighter-native-caption-r128-step250-v2.log"
OUTPUT = Path(
    os.environ.get(
        "SPRITELAB_COGVIDEOX_OUTPUT",
        ROOT / "probe-orange-fighter-normal-attack-lora-r128-step250-v2",
    )
)
SOURCE_INDEX_SHA256 = "98fbc592f23269a38d039d16f969844a9da073b56b24567772433d4b02e2f831"
DATASET_MANIFEST_SHA256 = "524a387ef02ce3ef42ac711e80f476d992f28e515edec37196822124821658aa"
SEED = 20260830
NEGATIVE_PROMPT = (
    "photo, realistic, 3d render, blur, soft edges, camera movement, background scene, "
    "multiple characters, sprite sheet, tiled poses, text, watermark"
)


def main() -> None:
    if not 0 < LORA_SCALE <= 2:
        raise ValueError("SPRITELAB_COGVIDEOX_LORA_SCALE must be in (0, 2]")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace CogVideoX LoRA probe: {OUTPUT}")
    source_index_path = MODEL / "source-index.json"
    source_index_bytes = source_index_path.read_bytes()
    if hashlib.sha256(source_index_bytes).hexdigest() != SOURCE_INDEX_SHA256:
        raise RuntimeError("CogVideoX source index differs")
    dataset_path = DATASET / "manifest.json"
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != DATASET_MANIFEST_SHA256:
        raise RuntimeError("CogVideoX MUGEN dataset differs")
    dataset = json.loads(dataset_bytes)
    candidates = [record for record in dataset["records"] if record["verb"] == VERB]
    if len(candidates) != 1:
        raise RuntimeError(f"{VERB} record cardinality differs")
    record = candidates[0]
    target_path = DATASET / record["video"]["path"]
    if file_sha256(target_path) != record["video"]["file_sha256"]:
        raise RuntimeError(f"{VERB} video hash differs")
    target = decode_video(target_path)
    if target.shape != (9, 480, 720, 3):
        raise RuntimeError(f"{VERB} target geometry differs")
    conditioning = Image.fromarray(target[0], mode="RGB")

    weights_path = LORA / LORA_WEIGHT_NAME
    checksum_paths = sorted(LORA.glob("*.sha256"))
    sums_path = LORA / "sha256sums.txt"
    if sums_path.is_file():
        checksum_paths.append(sums_path)
    if not weights_path.is_file() or not checksum_paths or not TRAIN_LOG.is_file():
        raise RuntimeError("completed LoRA training closure is absent")
    weights_sha256 = file_sha256(weights_path)
    sums_text = "\n".join(path.read_text(encoding="utf-8") for path in checksum_paths)
    if weights_sha256 not in sums_text or str(weights_path) not in sums_text:
        raise RuntimeError("LoRA weight is absent from its training checksum closure")

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.load_lora_weights(LORA, weight_name=LORA_WEIGHT_NAME, adapter_name="mugen-action")
    pipe.set_adapters("mugen-action", adapter_weights=LORA_SCALE)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        generated_images = pipe(
            image=conditioning,
            prompt=record["prompt"],
            negative_prompt=NEGATIVE_PROMPT,
            height=480,
            width=720,
            num_frames=9,
            num_inference_steps=50,
            guidance_scale=6,
            use_dynamic_cfg=True,
            generator=generator,
        ).frames[0]
    if len(generated_images) != 9 or any(frame.size != (720, 480) for frame in generated_images):
        raise RuntimeError("CogVideoX LoRA output geometry differs")
    generated = np.stack(
        [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in generated_images]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.stage-", dir=OUTPUT.parent))
    try:
        target_display = make_display_frames(target)
        generated_display = make_display_frames(generated)
        for index, frame in enumerate(generated_images):
            frame.convert("RGB").save(stage / f"frame-{index:02d}-raw-720x480.png", optimize=False)
        target_sheet = make_sheet(target_display)
        generated_sheet = make_sheet(generated_display)
        comparison = Image.new("RGB", (128 * 9, 256))
        comparison.paste(target_sheet, (0, 0))
        comparison.paste(generated_sheet, (0, 128))
        target_sheet_path = stage / "target-display-128-sheet.png"
        generated_sheet_path = stage / "generated-display-128-sheet.png"
        comparison_path = stage / "target-over-generated-display-128.png"
        animation_path = stage / "generated-display-128-animated.png"
        target_sheet.save(target_sheet_path, optimize=False)
        generated_sheet.save(generated_sheet_path, optimize=False)
        comparison.save(comparison_path, optimize=False)
        generated_display[0].save(
            animation_path,
            save_all=True,
            append_images=generated_display[1:],
            duration=[125] * 9,
            loop=0,
            disposal=2,
            blend=0,
            optimize=False,
        )
        report = {
            "artifact_kind": "mugen_cogvideox_i2v_lora_in_sample_probe",
            "claim": "one-identity in-sample action reconstruction; no held-out generalization",
            "dataset": {
                "manifest_sha256": DATASET_MANIFEST_SHA256,
                "prompt": record["prompt"],
                "sequence_id": record["sequence_id"],
                "target_video_file_sha256": record["video"]["file_sha256"],
                "verb": record["verb"],
            },
            "display": {
                "animation_file_sha256": file_sha256(animation_path),
                "animation_path": animation_path.name,
                "comparison_file_sha256": file_sha256(comparison_path),
                "comparison_path": comparison_path.name,
                "generated_sheet_file_sha256": file_sha256(generated_sheet_path),
                "generated_sheet_path": generated_sheet_path.name,
                "target_sheet_file_sha256": file_sha256(target_sheet_path),
                "target_sheet_path": target_sheet_path.name,
                "transform": "center 480x480 crop then 128x128 BOX downsample",
            },
            "generation": {
                "guidance_scale": 6,
                "lora_scale": LORA_SCALE,
                "negative_prompt": NEGATIVE_PROMPT,
                "num_frames": 9,
                "num_inference_steps": 50,
                "seed": SEED,
                "use_dynamic_cfg": True,
            },
            "metrics": {
                "center_crop_rgb_mae": rgb_mae(target[:, :, 120:600], generated[:, :, 120:600]),
                "full_canvas_rgb_mae": rgb_mae(target, generated),
            },
            "model": {
                "model_id": "THUDM/CogVideoX-5b-I2V",
                "resolved_revision": "a6f0f4858a8395e7429d82493864ce92bf73af11",
                "source_index_sha256": SOURCE_INDEX_SHA256,
            },
            "schema_version": 1,
            "training": {
                "lora_file_sha256": weights_sha256,
                "lora_path": str(weights_path),
                "rank": 128,
                "steps": TRAINING_STEPS,
                "training_log_sha256": file_sha256(TRAIN_LOG),
            },
        }
        payload = canonical_json(report)
        (stage / "evaluation-report.json").write_bytes(payload)
        os.rename(stage, OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "comparison_sha256": report["display"]["comparison_file_sha256"],
                "output": str(OUTPUT),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def decode_video(path: Path) -> np.ndarray:
    reader = decord.VideoReader(path.as_posix(), width=720, height=480)
    batch = reader.get_batch(list(range(len(reader))))
    if isinstance(batch, torch.Tensor):
        return batch.cpu().numpy()
    return batch.asnumpy()


def make_display_frames(frames: np.ndarray) -> list[Image.Image]:
    return [
        Image.fromarray(frame, mode="RGB")
        .crop((120, 0, 600, 480))
        .resize((128, 128), Image.Resampling.BOX)
        for frame in frames
    ]


def make_sheet(frames: list[Image.Image]) -> Image.Image:
    sheet = Image.new("RGB", (128 * len(frames), 128))
    for index, frame in enumerate(frames):
        sheet.paste(frame, (index * 128, 0))
    return sheet


def rgb_mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left.astype(np.int16) - right.astype(np.int16)).mean() / 255)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


if __name__ == "__main__":
    main()
