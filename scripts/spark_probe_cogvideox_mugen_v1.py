"""Probe pinned CogVideoX I2V on one exact MUGEN fighter reference."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from diffusers import CogVideoXImageToVideoPipeline
from PIL import Image

MODEL = Path("/home/sleepy/sprite-lab-cogvideox/CogVideoX-5b-I2V-a6f0f4858a83")
REFERENCE = Path("/home/sleepy/sprite-lab-cogvideox/reference-orange-fighter-idle-frame0-480.png")
OUTPUT = Path("/home/sleepy/sprite-lab-cogvideox/probe-orange-fighter-normal-attack-v2")
SOURCE_INDEX_SHA256 = "98fbc592f23269a38d039d16f969844a9da073b56b24567772433d4b02e2f831"
REFERENCE_SHA256 = "54634db9685c680e125b7b1b0a40b1657192db47ab6c2d6c2b7457142b18d0c9"
REFERENCE_SOURCE_VIDEO_SHA256 = "993911108ecbc0b4423167c27894d538823e8c9db9db3a0a925cb3eafe0de1ad"
REFERENCE_IDENTITY_ID = "mugen_13b410983214b11c_cd8d7683410b1695"
REFERENCE_APPEARANCE = (
    "muscular orange-skinned humanoid fighter, short white hair, dark purple sleeveless "
    "top, light yellow pants, dark shoes, purple wristbands, side profile"
)
SEED = 20260830
PROMPT = (
    "pixel art sprite, one isolated muscular orange-skinned humanoid fighter with short "
    "white hair, dark purple sleeveless top, light yellow pants, dark shoes, and purple "
    "wristbands, full body side view, performing a normal light attack, crisp hard pixel "
    "edges, fixed camera, plain neutral background"
)
NEGATIVE_PROMPT = (
    "photo, realistic, 3d render, blur, soft edges, camera movement, background scene, "
    "multiple characters, sprite sheet, tiled poses, text, watermark"
)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace CogVideoX probe: {OUTPUT}")
    source_index_path = MODEL / "source-index.json"
    source_index_bytes = source_index_path.read_bytes()
    source_index_sha256 = hashlib.sha256(source_index_bytes).hexdigest()
    if SOURCE_INDEX_SHA256 is not None and source_index_sha256 != SOURCE_INDEX_SHA256:
        raise RuntimeError("CogVideoX source index differs")
    source_index = json.loads(source_index_bytes)
    for record in source_index["files"]:
        path = MODEL / record["path"]
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"CogVideoX model payload differs: {path}")
    if file_sha256(REFERENCE) != REFERENCE_SHA256:
        raise RuntimeError("MUGEN reference image differs")
    reference = Image.open(REFERENCE).convert("RGB")
    if reference.size != (480, 480):
        raise RuntimeError("MUGEN reference geometry differs")
    conditioning = reference
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        frames = pipe(
            image=conditioning,
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            height=480,
            width=480,
            num_frames=9,
            num_inference_steps=50,
            guidance_scale=6,
            use_dynamic_cfg=True,
            generator=generator,
        ).frames[0]
    if len(frames) != 9 or any(frame.size != (480, 480) for frame in frames):
        raise RuntimeError("CogVideoX output geometry differs")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.stage-", dir=OUTPUT.parent))
    try:
        conditioning.save(stage / "reference-conditioning-480.png", optimize=False)
        display = []
        frame_records = []
        for index, frame in enumerate(frames):
            raw_path = stage / f"frame-{index:02d}-raw-480.png"
            display_path = stage / f"frame-{index:02d}-display-128.png"
            frame.convert("RGB").save(raw_path, optimize=False)
            reduced = frame.convert("RGB").resize((128, 128), Image.Resampling.BOX)
            reduced.save(display_path, optimize=False)
            display.append(reduced)
            frame_records.append(
                {
                    "display_file_sha256": file_sha256(display_path),
                    "display_path": display_path.name,
                    "frame_index": index,
                    "raw_file_sha256": file_sha256(raw_path),
                    "raw_path": raw_path.name,
                }
            )
        animation_path = stage / "generated-display-128-animated.png"
        display[0].save(
            animation_path,
            save_all=True,
            append_images=display[1:],
            duration=[100] * len(display),
            loop=0,
            disposal=2,
            blend=0,
            optimize=False,
        )
        sheet = Image.new("RGB", (128 * len(display), 128))
        for index, frame in enumerate(display):
            sheet.paste(frame, (128 * index, 0))
        sheet_path = stage / "generated-display-128-sheet.png"
        sheet.save(sheet_path, optimize=False)
        report = {
            "artifact_kind": "mugen_cogvideox_i2v_pretrained_probe",
            "claim": "zero-shot pretrained probe; no MUGEN motion fine-tuning",
            "frames": frame_records,
            "generation": {
                "animated_file_sha256": file_sha256(animation_path),
                "animated_path": animation_path.name,
                "guidance_scale": 6,
                "negative_prompt": NEGATIVE_PROMPT,
                "num_frames": 9,
                "num_inference_steps": 50,
                "prompt": PROMPT,
                "seed": SEED,
                "sheet_file_sha256": file_sha256(sheet_path),
                "sheet_path": sheet_path.name,
                "use_dynamic_cfg": True,
            },
            "model": {
                "model_id": source_index["model_id"],
                "resolved_revision": source_index["resolved_revision"],
                "source_index_sha256": source_index_sha256,
            },
            "reference": {
                "appearance_description": REFERENCE_APPEARANCE,
                "conditioning_resize": "none_exact_480x480_rgb_frame",
                "file_sha256": REFERENCE_SHA256,
                "identity_id": REFERENCE_IDENTITY_ID,
                "path": str(REFERENCE),
                "source_video_file_sha256": REFERENCE_SOURCE_VIDEO_SHA256,
                "source_video_frame_index": 0,
            },
            "schema_version": 1,
        }
        report_payload = canonical_json(report)
        (stage / "evaluation-report.json").write_bytes(report_payload)
        os.rename(stage, OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "report_sha256": hashlib.sha256(report_payload).hexdigest(),
                "sheet_sha256": report["generation"]["sheet_file_sha256"],
            },
            sort_keys=True,
        )
    )


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
