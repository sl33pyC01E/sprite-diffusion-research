"""Run a pinned sprite-prior AnimateDiff SparseCtrl reference baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
from PIL import Image

BASE_NAME = "sd-pixelart-spritesheet-8229c9b6e928103f0e657cfe6b14d902cb2101d6"
MOTION_NAME = "animatediff-motion-adapter-v1-5-3-2e8139b1d126"
CONTROL_NAME = "animatediff-sparsectrl-rgb-b8003d681d81"
MOTION_REVISION = "2e8139b1d1269fd8a21deb96ad19455e187692eb"
CONTROL_REVISION = "b8003d681d813c095e459b9141122894daff2d13"
SEED = 20260827
PROMPT = (
    "PixelartRSS, pixel art sprite, transparent background, muscular beige humanoid "
    "fighter with short yellow hair and black shorts, full body side view, performing "
    "a normal light attack, crisp hard pixel edges, consistent character design"
)
NEGATIVE_PROMPT = (
    "photo, realistic, 3d render, blurry, soft edges, noisy, detailed background, "
    "text, watermark, camera movement, scene change, extra character"
)


def main(
    bundle_root: Path,
    reference_path: Path,
    output: Path,
    *,
    loop_anchors: bool,
    control_scale: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to replace output: {output}")
    base = bundle_root / BASE_NAME
    motion = bundle_root / MOTION_NAME
    control = bundle_root / CONTROL_NAME
    motion_index = _verify_source_index(motion, MOTION_REVISION)
    control_index = _verify_source_index(control, CONTROL_REVISION)
    reference_bytes = reference_path.read_bytes()
    reference_sha256 = hashlib.sha256(reference_bytes).hexdigest()
    reference = Image.open(reference_path).convert("RGB")
    if reference.size != (128, 128):
        raise ValueError("reference must be an exact 128x128 RGB composite")
    conditioning = reference.resize((512, 512), Image.Resampling.NEAREST)
    control_indices = [0, 7] if loop_anchors else [0]
    conditioning_frames = [conditioning, conditioning] if loop_anchors else conditioning
    prompt = PROMPT
    negative_prompt = NEGATIVE_PROMPT
    if loop_anchors:
        prompt += ", one isolated single fighter sprite, one pose per video frame"
        negative_prompt += ", sprite sheet, spritesheet, multiple poses, repeated character, tiled"

    from diffusers import (
        AnimateDiffSparseControlNetPipeline,
        DPMSolverMultistepScheduler,
        MotionAdapter,
        SparseControlNetModel,
    )

    dtype = torch.float16
    adapter = MotionAdapter.from_pretrained(
        motion, variant="fp16", torch_dtype=dtype, local_files_only=True
    )
    controlnet = SparseControlNetModel.from_pretrained(
        control, variant="fp16", torch_dtype=dtype, local_files_only=True
    )
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        base / "scheduler",
        beta_schedule="linear",
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
        local_files_only=True,
    )
    pipeline = AnimateDiffSparseControlNetPipeline.from_pretrained(
        base,
        motion_adapter=adapter,
        controlnet=controlnet,
        scheduler=scheduler,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        local_files_only=True,
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        frames = pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_frames=8,
            num_inference_steps=25,
            guidance_scale=7.5,
            conditioning_frames=conditioning_frames,
            controlnet_frame_indices=control_indices,
            controlnet_conditioning_scale=control_scale,
            generator=generator,
        ).frames[0]
    if len(frames) != 8 or any(frame.size != (512, 512) for frame in frames):
        raise RuntimeError("AnimateDiff returned unexpected frame geometry")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    try:
        conditioning.save(stage / "reference-conditioning-512.png", optimize=False)
        reduced = [
            frame.convert("RGB").resize((128, 128), Image.Resampling.BOX) for frame in frames
        ]
        frame_rows = []
        for index, (raw, display) in enumerate(zip(frames, reduced, strict=True)):
            raw_path = stage / f"frame-{index:02d}-raw-512.png"
            display_path = stage / f"frame-{index:02d}-display-128.png"
            raw.save(raw_path, optimize=False)
            display.save(display_path, optimize=False)
            frame_rows.append(
                {
                    "display_128_file_sha256": _file_sha256(display_path),
                    "display_128_path": display_path.name,
                    "frame_index": index,
                    "raw_512_file_sha256": _file_sha256(raw_path),
                    "raw_512_path": raw_path.name,
                }
            )
        animation_path = stage / "generated-display-128-animated.png"
        reduced[0].save(
            animation_path,
            save_all=True,
            append_images=reduced[1:],
            duration=[100] * 8,
            loop=0,
            disposal=2,
            blend=0,
            optimize=False,
        )
        sheet = Image.new("RGB", (128 * 8, 128))
        for index, frame in enumerate(reduced):
            sheet.paste(frame, (index * 128, 0))
        sheet_path = stage / "generated-display-128-sheet.png"
        sheet.save(sheet_path, optimize=False)
        report = {
            "artifact_kind": "mugen_animatediff_sparsectrl_pretrained_baseline",
            "claim": "pretrained reference-conditioned quality probe; no MUGEN motion fine-tuning",
            "conditioning": {
                "composite_background_rgb": [127, 127, 127],
                "controlnet_frame_indices": control_indices,
                "file_sha256": reference_sha256,
                "source_frame": "sequence_7a5bb072f3399e86dbf8fbeac687f53f frame 4",
            },
            "frames": frame_rows,
            "generation": {
                "animated_file_sha256": _file_sha256(animation_path),
                "animated_path": animation_path.name,
                "controlnet_conditioning_scale": control_scale,
                "guidance_scale": 7.5,
                "negative_prompt": negative_prompt,
                "num_frames": 8,
                "num_inference_steps": 25,
                "prompt": prompt,
                "seed": SEED,
                "sheet_file_sha256": _file_sha256(sheet_path),
                "sheet_path": sheet_path.name,
            },
            "models": {
                "base": BASE_NAME,
                "motion_source_index_sha256": hashlib.sha256(motion_index).hexdigest(),
                "sparsectrl_source_index_sha256": hashlib.sha256(control_index).hexdigest(),
            },
            "schema_version": 1,
        }
        payload = (
            json.dumps(report, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        (stage / "evaluation-report.json").write_bytes(payload)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
                "sheet_sha256": report["generation"]["sheet_file_sha256"],
            },
            sort_keys=True,
        )
    )


def _verify_source_index(root: Path, revision: str) -> bytes:
    payload = (root / "source-index.json").read_bytes()
    value = json.loads(payload)
    if value.get("revision") != revision:
        raise RuntimeError(f"model revision differs: {root}")
    for record in value.get("files", []):
        path = root / record["path"]
        if path.stat().st_size != record["bytes"] or _file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"model payload differs: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loop-anchors", action="store_true")
    parser.add_argument("--control-scale", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.control_scale <= 3:
        parser.error("--control-scale must be in (0,3]")
    main(
        args.bundle_root.resolve(),
        args.reference.resolve(),
        args.output.resolve(),
        loop_anchors=args.loop_anchors,
        control_scale=args.control_scale,
    )
