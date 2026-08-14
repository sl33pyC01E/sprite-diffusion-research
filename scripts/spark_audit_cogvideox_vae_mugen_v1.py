"""Audit the frozen CogVideoX VAE ceiling on an exact MUGEN attack clip."""

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
from diffusers import AutoencoderKLCogVideoX
from PIL import Image

ROOT = Path("/home/sleepy/sprite-lab-cogvideox")
MODEL = ROOT / "CogVideoX-5b-I2V-a6f0f4858a83"
DATASET = ROOT / "mugen-cogvideox-orange-fighter-i2v-native-caption-v3"
OUTPUT = ROOT / "codec-audit-orange-fighter-normal-attack-v1"
SOURCE_INDEX_SHA256 = "98fbc592f23269a38d039d16f969844a9da073b56b24567772433d4b02e2f831"
DATASET_MANIFEST_SHA256 = "524a387ef02ce3ef42ac711e80f476d992f28e515edec37196822124821658aa"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace CogVideoX codec audit: {OUTPUT}")
    source_index_path = MODEL / "source-index.json"
    source_index_bytes = source_index_path.read_bytes()
    if hashlib.sha256(source_index_bytes).hexdigest() != SOURCE_INDEX_SHA256:
        raise RuntimeError("CogVideoX source index differs")
    dataset_path = DATASET / "manifest.json"
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != DATASET_MANIFEST_SHA256:
        raise RuntimeError("CogVideoX MUGEN dataset differs")
    dataset = json.loads(dataset_bytes)
    candidates = [record for record in dataset["records"] if record["verb"] == "normal_attack"]
    if len(candidates) != 1:
        raise RuntimeError("normal-attack record cardinality differs")
    record = candidates[0]
    video_path = DATASET / record["video"]["path"]
    if file_sha256(video_path) != record["video"]["file_sha256"]:
        raise RuntimeError("normal-attack video hash differs")
    target = decode_video(video_path)
    if target.shape != (9, 480, 720, 3):
        raise RuntimeError("normal-attack video geometry differs")

    vae = AutoencoderKLCogVideoX.from_pretrained(
        MODEL / "vae", torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    vae.enable_slicing()
    vae.enable_tiling()
    source_tensor = (
        torch.from_numpy(target.copy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device="cuda", dtype=torch.bfloat16)
        / 127.5
        - 1
    )
    with torch.inference_mode():
        latents = vae.encode(source_tensor).latent_dist.mode()
        reconstructed_tensor = vae.decode(latents).sample
    reconstructed = (
        ((reconstructed_tensor.float().clamp(-1, 1) + 1) * 127.5)
        .round()
        .to(torch.uint8)
        .squeeze(0)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )
    if reconstructed.shape != target.shape:
        raise RuntimeError(f"CogVideoX VAE output geometry differs: {reconstructed.shape}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.stage-", dir=OUTPUT.parent))
    try:
        target_display = make_display_frames(target)
        reconstructed_display = make_display_frames(reconstructed)
        target_sheet = make_sheet(target_display)
        reconstruction_sheet = make_sheet(reconstructed_display)
        comparison = Image.new("RGB", (128 * 9, 256))
        comparison.paste(target_sheet, (0, 0))
        comparison.paste(reconstruction_sheet, (0, 128))
        target_path = stage / "target-display-128-sheet.png"
        reconstruction_path = stage / "reconstruction-display-128-sheet.png"
        comparison_path = stage / "target-over-reconstruction-display-128.png"
        target_sheet.save(target_path, optimize=False)
        reconstruction_sheet.save(reconstruction_path, optimize=False)
        comparison.save(comparison_path, optimize=False)
        report = {
            "artifact_kind": "mugen_cogvideox_frozen_vae_codec_audit",
            "claim": "frozen VAE reconstruction ceiling; no denoiser or motion generation",
            "dataset": {
                "manifest_sha256": DATASET_MANIFEST_SHA256,
                "sequence_id": record["sequence_id"],
                "verb": record["verb"],
                "video_file_sha256": record["video"]["file_sha256"],
            },
            "display": {
                "comparison_file_sha256": file_sha256(comparison_path),
                "comparison_path": comparison_path.name,
                "reconstruction_file_sha256": file_sha256(reconstruction_path),
                "reconstruction_path": reconstruction_path.name,
                "target_file_sha256": file_sha256(target_path),
                "target_path": target_path.name,
                "transform": "center 480x480 crop then 128x128 BOX downsample",
            },
            "metrics": {
                "center_crop_rgb_mae": rgb_mae(target[:, :, 120:600], reconstructed[:, :, 120:600]),
                "full_canvas_rgb_mae": rgb_mae(target, reconstructed),
            },
            "model": {
                "model_id": "THUDM/CogVideoX-5b-I2V",
                "resolved_revision": "a6f0f4858a8395e7429d82493864ce92bf73af11",
                "source_index_sha256": SOURCE_INDEX_SHA256,
            },
            "schema_version": 1,
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
