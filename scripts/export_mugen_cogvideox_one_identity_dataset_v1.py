"""Export a hash-bound 9-frame CogVideoX I2V overfit dataset from MUGEN."""

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
IDENTITY = "mugen_13b410983214b11c_cd8d7683410b1695"
CANONICAL = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"
TEXT_PLAN = ROOT / "data/processed/mugen-mffa-sd-primary-motion-text-plan-v1.json"
OUTPUT = ROOT / "data/processed/mugen-cogvideox-orange-fighter-i2v-v1"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace CogVideoX dataset: {OUTPUT}")
    canonical_bytes = CANONICAL.read_bytes()
    text_plan_bytes = TEXT_PLAN.read_bytes()
    canonical = json.loads(canonical_bytes)
    text_plan = json.loads(text_plan_bytes)
    motion_plan_path = Path(canonical["source"]["motion_plan_path"])
    motion_plan_bytes = motion_plan_path.read_bytes()
    if (
        hashlib.sha256(motion_plan_bytes).hexdigest()
        != canonical["source"]["motion_plan_file_sha256"]
    ):
        raise RuntimeError("canonical motion-plan hash differs")
    motion_plan = json.loads(motion_plan_bytes)
    materialization = motion_plan["source"]["materialization"]
    materialization_path = Path(materialization["path"])
    if file_sha256(materialization_path) != materialization["file_sha256"]:
        raise RuntimeError("MUGEN materialization manifest differs")
    materialization_root = materialization_path.parent
    text_by_sequence = {record["sequence_id"]: record for record in text_plan["records"]}
    rows = [
        record
        for record in canonical["records"]
        if record.get("split") == "train" and record.get("identity_id") == IDENTITY
    ]
    rows.sort(key=lambda record: record["conditioning"]["verb"].encode())
    if len(rows) != 10 or len({record["conditioning"]["verb"] for record in rows}) != 10:
        raise RuntimeError("orange-fighter action cardinality differs")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.stage-", dir=OUTPUT.parent))
    try:
        videos_dir = stage / "videos"
        videos_dir.mkdir()
        prompts = []
        videos = []
        records = []
        for index, record in enumerate(rows):
            sequence_id = record["sequence_id"]
            verb = record["conditioning"]["verb"]
            text_record = text_by_sequence.get(sequence_id)
            if text_record is None or text_record.get("identity_id") != IDENTITY:
                raise RuntimeError(f"text-plan join differs for {sequence_id}")
            reference = record["reference"]
            reference_clip = load_rgba(
                materialization_root, _reference_pixels(canonical, reference["sequence_id"])
            )
            target = load_rgba(materialization_root, record["target"]["source_pixels"])
            reference_frame = reference_clip[int(reference["frame_index"])]
            frames = np.concatenate((reference_frame[None], target), axis=0)
            rgb = composite_gray(frames)
            video_name = f"{index:02d}-{verb.replace('_', '-')}-{sequence_id[-8:]}.mp4"
            video_path = videos_dir / video_name
            encode_lossless_video(rgb, video_path)
            prompt = text_record["prompt"]
            prompts.append(prompt)
            videos.append(f"videos/{video_name}")
            records.append(
                {
                    "conditioning_frame": {
                        "frame_index": int(reference["frame_index"]),
                        "sequence_id": reference["sequence_id"],
                    },
                    "frame_contract": "frame_0_reference_then_8_exact_action_frames",
                    "identity_id": IDENTITY,
                    "prompt": prompt,
                    "sequence_id": sequence_id,
                    "split": "train",
                    "target_source_pixels": record["target"]["source_pixels"],
                    "verb": verb,
                    "video": {
                        "file_sha256": file_sha256(video_path),
                        "frames": 9,
                        "height": 480,
                        "path": f"videos/{video_name}",
                        "width": 480,
                    },
                }
            )
        (stage / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
        (stage / "videos.txt").write_text("\n".join(videos) + "\n", encoding="utf-8")
        manifest = {
            "artifact_kind": "mugen_cogvideox_i2v_overfit_dataset",
            "claim": "one-identity pipeline and action-conditioning gate; not a generalization set",
            "counts": {"identities": 1, "sequences": len(records), "verbs": len(records)},
            "projection": {
                "alpha_composite_background_rgb": [127, 127, 127],
                "codec": "libx264rgb_crf0_lossless",
                "frame_order": "reference_then_action_frames",
                "native_geometry": [128, 128],
                "training_geometry": [480, 480],
                "upscale": "nearest_neighbor",
            },
            "records": records,
            "schema_version": 1,
            "source": {
                "canonical_manifest_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
                "canonical_manifest_path": str(CANONICAL),
                "materialization_manifest_sha256": materialization["file_sha256"],
                "materialization_manifest_path": str(materialization_path),
                "motion_plan_sha256": hashlib.sha256(motion_plan_bytes).hexdigest(),
                "motion_plan_path": str(motion_plan_path),
                "text_plan_sha256": hashlib.sha256(text_plan_bytes).hexdigest(),
                "text_plan_path": str(TEXT_PLAN),
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
                "bytes": sum(
                    (OUTPUT / record["video"]["path"]).stat().st_size for record in records
                ),
                "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "output": str(OUTPUT),
                "sequences": len(records),
            },
            sort_keys=True,
        )
    )


def _reference_pixels(canonical: dict[str, object], sequence_id: str) -> dict[str, object]:
    for record in canonical["records"]:
        if record.get("sequence_id") == sequence_id:
            return record["target"]["source_pixels"]
    raise RuntimeError(f"reference sequence is absent: {sequence_id}")


def load_rgba(root: Path, record: dict[str, object]) -> np.ndarray:
    path = (root / record["relative_path"]).resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["file_sha256"]:
        raise RuntimeError(f"MUGEN pixel payload differs: {path}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise RuntimeError(f"MUGEN pixel geometry differs: {path}")
    return value


def composite_gray(rgba: np.ndarray) -> np.ndarray:
    unit = rgba.astype(np.float32) / 255
    alpha = unit[..., 3:4]
    rgb = unit[..., :3] * alpha + (127 / 255) * (1 - alpha)
    return np.rint(rgb * 255).clip(0, 255).astype(np.uint8)


def encode_lossless_video(frames: np.ndarray, output: Path) -> None:
    if frames.dtype != np.uint8 or frames.shape != (9, 128, 128, 3):
        raise RuntimeError("CogVideoX video source geometry differs")
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "128x128",
            "-framerate",
            "8",
            "-i",
            "pipe:0",
            "-vf",
            "scale=480:480:flags=neighbor",
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
        input=frames.tobytes(),
        check=False,
        capture_output=True,
    )
    if process.returncode or not output.is_file():
        raise RuntimeError(
            f"ffmpeg lossless export failed: {process.stderr.decode(errors='replace')}"
        )
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(output),
            "-vf",
            "scale=128:128:flags=neighbor",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    expected_bytes = frames.tobytes()
    if decoded.returncode or decoded.stdout != expected_bytes:
        raise RuntimeError("lossless video round trip differs from exact RGB frames")


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
