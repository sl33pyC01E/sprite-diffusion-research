"""Publish the matched CogVideoX block-versus-attack causal gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/mugen-cogvideox-orange-fighter-i2v-native-caption-v3"
ATTACK = ROOT / "data/inference/mugen-cogvideox-i2v-balanced-r128-step250-orange-attack-v1"
BLOCK = ROOT / "data/inference/mugen-cogvideox-i2v-balanced-r128-step250-orange-block-v1"
OUTPUT = ROOT / "data/index/reports/mugen-cogvideox-balanced-block-attack-causal-v1.json"
DATASET_MANIFEST_SHA256 = "524a387ef02ce3ef42ac711e80f476d992f28e515edec37196822124821658aa"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace causal audit: {OUTPUT}")
    dataset_path = DATASET / "manifest.json"
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != DATASET_MANIFEST_SHA256:
        raise RuntimeError("CogVideoX evaluation dataset differs")
    dataset = json.loads(dataset_bytes)
    records = {record["verb"]: record for record in dataset["records"]}
    target_attack = load_target(records["normal_attack"])
    target_block = load_target(records["block"])
    generated_attack, attack_report = load_generation(ATTACK, "normal_attack")
    generated_block, block_report = load_generation(BLOCK, "block")
    if attack_report["generation"] != block_report["generation"]:
        raise RuntimeError("matched generation contract differs")
    if attack_report["training"] != block_report["training"]:
        raise RuntimeError("matched training contract differs")

    report = {
        "artifact_kind": "mugen_cogvideox_block_attack_matched_causal_audit",
        "claim": "one-identity two-action in-sample capacity gate; no held-out generalization",
        "dataset": {
            "manifest_sha256": DATASET_MANIFEST_SHA256,
            "target_attack_array_sha256": array_sha256(target_attack),
            "target_block_array_sha256": array_sha256(target_block),
        },
        "generation_contract": attack_report["generation"],
        "metrics": {
            "all_9_frames": metrics(target_attack, target_block, generated_attack, generated_block),
            "motion_frames_1_through_8": metrics(
                target_attack[1:],
                target_block[1:],
                generated_attack[1:],
                generated_block[1:],
            ),
        },
        "outputs": {
            "attack": {
                "array_sha256": array_sha256(generated_attack),
                "report_file_sha256": file_sha256(ATTACK / "evaluation-report.json"),
                "report_path": str(ATTACK / "evaluation-report.json"),
            },
            "block": {
                "array_sha256": array_sha256(generated_block),
                "report_file_sha256": file_sha256(BLOCK / "evaluation-report.json"),
                "report_path": str(BLOCK / "evaluation-report.json"),
            },
        },
        "schema_version": 1,
        "training": attack_report["training"],
    }
    payload = canonical_json(report)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent)
    try:
        with os.fdopen(handle, "wb") as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.rename(temporary_name, OUTPUT)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def load_target(record: dict[str, object]) -> np.ndarray:
    path = DATASET / record["video"]["path"]
    if file_sha256(path) != record["video"]["file_sha256"]:
        raise RuntimeError(f"target video differs: {path}")
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
    if process.returncode or len(process.stdout) != 9 * 480 * 720 * 3:
        raise RuntimeError(f"target decode differs: {path}")
    return np.frombuffer(process.stdout, dtype=np.uint8).reshape(9, 480, 720, 3)


def load_generation(directory: Path, expected_verb: str) -> tuple[np.ndarray, dict[str, object]]:
    report_path = directory / "evaluation-report.json"
    report = json.loads(report_path.read_bytes())
    if report["dataset"]["verb"] != expected_verb:
        raise RuntimeError(f"generated verb differs: {directory}")
    frames = []
    for index in range(9):
        frame = np.asarray(
            Image.open(directory / f"frame-{index:02d}-raw-720x480.png").convert("RGB"),
            dtype=np.uint8,
        )
        if frame.shape != (480, 720, 3):
            raise RuntimeError(f"generated frame geometry differs: {directory}")
        frames.append(frame)
    return np.stack(frames), report


def metrics(
    target_attack: np.ndarray,
    target_block: np.ndarray,
    generated_attack: np.ndarray,
    generated_block: np.ndarray,
) -> dict[str, float]:
    target_separation = crop_mae(target_attack, target_block)
    generated_separation = crop_mae(generated_attack, generated_block)
    return {
        "attack_to_attack_rgb_mae": crop_mae(generated_attack, target_attack),
        "attack_to_block_rgb_mae": crop_mae(generated_attack, target_block),
        "block_to_attack_rgb_mae": crop_mae(generated_block, target_attack),
        "block_to_block_rgb_mae": crop_mae(generated_block, target_block),
        "generated_action_separation_rgb_mae": generated_separation,
        "generated_over_target_separation_ratio": generated_separation / target_separation,
        "target_action_separation_rgb_mae": target_separation,
    }


def crop_mae(left: np.ndarray, right: np.ndarray) -> float:
    left_crop = left[:, :, 120:600].astype(np.int16)
    right_crop = right[:, :, 120:600].astype(np.int16)
    return float(np.abs(left_crop - right_crop).mean() / 255)


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes()).hexdigest()


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
