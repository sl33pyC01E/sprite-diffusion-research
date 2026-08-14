"""Pad the exact MUGEN CogVideoX gate dataset to its native 720x480 canvas."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/mugen-cogvideox-orange-fighter-i2v-v1"
OUTPUT = ROOT / "data/processed/mugen-cogvideox-orange-fighter-i2v-native-v2"
BACKGROUND = 127


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace CogVideoX dataset: {OUTPUT}")
    source_manifest_path = SOURCE / "manifest.json"
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    if source_manifest.get("schema_version") != 1:
        raise RuntimeError("source dataset schema differs")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.stage-", dir=OUTPUT.parent))
    try:
        videos_dir = stage / "videos"
        videos_dir.mkdir()
        records = []
        video_lines = []
        prompt_lines = []
        for source_record in source_manifest["records"]:
            relative = Path(source_record["video"]["path"])
            source_video = SOURCE / relative
            if file_sha256(source_video) != source_record["video"]["file_sha256"]:
                raise RuntimeError(f"source video hash differs: {relative}")
            source_frames = decode_rgb(source_video, 480, 480)
            if source_frames.shape != (9, 480, 480, 3):
                raise RuntimeError(f"source video geometry differs: {relative}")
            output_video = videos_dir / relative.name
            encode_padded(source_video, output_video)
            output_frames = decode_rgb(output_video, 720, 480)
            expected = np.full((9, 480, 720, 3), BACKGROUND, dtype=np.uint8)
            expected[:, :, 120:600] = source_frames
            if not np.array_equal(output_frames, expected):
                raise RuntimeError(f"native padded round trip differs: {relative}")

            record = json.loads(json.dumps(source_record))
            record["video"] = {
                "file_sha256": file_sha256(output_video),
                "frames": 9,
                "height": 480,
                "path": f"videos/{relative.name}",
                "width": 720,
            }
            records.append(record)
            video_lines.append(record["video"]["path"])
            prompt_lines.append(record["prompt"])

        (stage / "videos.txt").write_text("\n".join(video_lines) + "\n", encoding="utf-8")
        (stage / "prompts.txt").write_text("\n".join(prompt_lines) + "\n", encoding="utf-8")
        manifest = {
            "artifact_kind": "mugen_cogvideox_i2v_native_overfit_dataset",
            "claim": "one-identity pipeline and action-conditioning gate; not a generalization set",
            "counts": source_manifest["counts"],
            "projection": {
                "alpha_composite_background_rgb": [BACKGROUND] * 3,
                "codec": "libx264rgb_crf0_lossless",
                "frame_order": "reference_then_action_frames",
                "native_geometry": [128, 128],
                "sprite_training_geometry": [480, 480],
                "model_training_geometry": [720, 480],
                "horizontal_padding_pixels_each_side": 120,
                "padding_rgb": [BACKGROUND] * 3,
                "upscale": "nearest_neighbor",
                "interpolation_after_sprite_upscale": "none",
            },
            "records": records,
            "schema_version": 2,
            "source": {
                "dataset_manifest_path": str(source_manifest_path),
                "dataset_manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
            },
        }
        manifest_payload = canonical_json(manifest)
        (stage / "manifest.json").write_bytes(manifest_payload)
        os.rename(stage, OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "bytes": sum((OUTPUT / row["video"]["path"]).stat().st_size for row in records),
                "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "output": str(OUTPUT),
                "sequences": len(records),
            },
            sort_keys=True,
        )
    )


def encode_padded(source: Path, output: Path) -> None:
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vf",
            "pad=720:480:120:0:color=0x7f7f7f",
            "-frames:v",
            "9",
            "-c:v",
            "libx264rgb",
            "-crf",
            "0",
            "-preset",
            "medium",
            "-pix_fmt",
            "rgb24",
            str(output),
        ],
        check=False,
        capture_output=True,
    )
    if process.returncode or not output.is_file():
        raise RuntimeError(
            f"ffmpeg native padding failed: {process.stderr.decode(errors='replace')}"
        )


def decode_rgb(path: Path, width: int, height: int) -> np.ndarray:
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if process.returncode:
        raise RuntimeError(f"ffmpeg decode failed: {process.stderr.decode(errors='replace')}")
    frame_bytes = width * height * 3
    if len(process.stdout) % frame_bytes:
        raise RuntimeError(f"decoded byte count differs: {path}")
    return np.frombuffer(process.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


if __name__ == "__main__":
    main()
