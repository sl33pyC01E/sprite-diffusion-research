"""Export an identity-disjoint six-verb MUGEN CogVideoX benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from spritelab.storage import DiskGuard

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"
TEXT_PLAN = ROOT / "data/processed/mugen-mffa-sd-primary-motion-text-plan-v1.json"
OUTPUT = ROOT / "data/processed/mugen-cogvideox-heldout-six-verb-v1"
VERBS = ("crouch", "idle", "jump", "normal_attack", "turn", "walk")
IDENTITY_LIMITS = {"train": 24, "validation": 8, "test": 8}
BACKGROUND = 127


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace CogVideoX dataset: {OUTPUT}")
    DiskGuard(OUTPUT.parent, min_free_bytes=100 * 1024**3).require_capacity(
        512 * 1024**2, label="MUGEN CogVideoX held-out six-verb export"
    )
    canonical_bytes = CANONICAL.read_bytes()
    text_bytes = TEXT_PLAN.read_bytes()
    canonical = json.loads(canonical_bytes)
    text_plan = json.loads(text_bytes)
    motion_plan_path = Path(canonical["source"]["motion_plan_path"])
    motion_plan_bytes = motion_plan_path.read_bytes()
    motion_plan = json.loads(motion_plan_bytes)
    materialization = motion_plan["source"]["materialization"]
    materialization_path = Path(materialization["path"])
    if file_sha256(materialization_path) != materialization["file_sha256"]:
        raise RuntimeError("MUGEN materialization manifest differs")
    materialization_root = materialization_path.parent

    canonical_by_sequence = {record["sequence_id"]: record for record in canonical["records"]}
    text_by_sequence = {record["sequence_id"]: record for record in text_plan["records"]}
    records_by_identity: defaultdict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    split_by_identity = {}
    for record in canonical["records"]:
        identity_id = record["identity_id"]
        verb = record["conditioning"]["verb"]
        split = record["split"]
        if identity_id in split_by_identity and split_by_identity[identity_id] != split:
            raise RuntimeError(f"identity crosses splits: {identity_id}")
        split_by_identity[identity_id] = split
        if verb in VERBS:
            if verb in records_by_identity[identity_id]:
                raise RuntimeError(f"identity has duplicate canonical verb: {identity_id} {verb}")
            records_by_identity[identity_id][verb] = record

    selected_identities = {}
    required_verbs = set(VERBS)
    for split, limit in IDENTITY_LIMITS.items():
        candidates = [
            identity_id
            for identity_id, by_verb in records_by_identity.items()
            if split_by_identity[identity_id] == split and set(by_verb) == required_verbs
        ]
        candidates.sort(
            key=lambda identity_id: identity_quality_key(records_by_identity[identity_id])
        )
        if len(candidates) < limit:
            raise RuntimeError(f"insufficient six-verb identities for {split}")
        selected_identities[split] = candidates[:limit]
    all_selected = [identity for values in selected_identities.values() for identity in values]
    if len(set(all_selected)) != sum(IDENTITY_LIMITS.values()):
        raise RuntimeError("selected identities overlap across splits")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.stage-", dir=OUTPUT.parent))
    try:
        videos_dir = stage / "videos"
        videos_dir.mkdir()
        output_records = []
        lines_by_split = {split: {"prompts": [], "videos": []} for split in IDENTITY_LIMITS}
        for split in ("train", "validation", "test"):
            for identity_index, identity_id in enumerate(selected_identities[split]):
                for verb in VERBS:
                    record = records_by_identity[identity_id][verb]
                    sequence_id = record["sequence_id"]
                    text_record = text_by_sequence.get(sequence_id)
                    if text_record is None or text_record["identity_id"] != identity_id:
                        raise RuntimeError(f"text-plan join differs: {sequence_id}")
                    prompt = corrected_prompt(text_record["prompt"])
                    reference = record["reference"]
                    reference_record = canonical_by_sequence.get(reference["sequence_id"])
                    if reference_record is None:
                        raise RuntimeError(f"reference sequence is absent: {sequence_id}")
                    reference_clip = load_rgba(
                        materialization_root, reference_record["target"]["source_pixels"]
                    )
                    target_clip = load_rgba(materialization_root, record["target"]["source_pixels"])
                    reference_frame = reference_clip[int(reference["frame_index"])]
                    frames = np.concatenate((reference_frame[None], target_clip), axis=0)
                    rgb = composite_gray(frames)
                    video_name = (
                        f"{split}-{identity_index:02d}-{verb.replace('_', '-')}-"
                        f"{identity_id[-8:]}-{sequence_id[-8:]}.mp4"
                    )
                    video_path = videos_dir / video_name
                    encode_native_lossless_video(rgb, video_path)
                    relative_video = f"videos/{video_name}"
                    lines_by_split[split]["prompts"].append(prompt)
                    lines_by_split[split]["videos"].append(relative_video)
                    output_records.append(
                        {
                            "entity_class": record["entity_class"],
                            "identity_id": identity_id,
                            "prompt": prompt,
                            "quality": record["eligibility"]["representative_quality"],
                            "reference": {
                                "frame_index": int(reference["frame_index"]),
                                "sequence_id": reference["sequence_id"],
                            },
                            "sequence_id": sequence_id,
                            "split": split,
                            "target_source_pixels": record["target"]["source_pixels"],
                            "verb": verb,
                            "video": {
                                "file_sha256": file_sha256(video_path),
                                "frames": 9,
                                "height": 480,
                                "path": relative_video,
                                "width": 720,
                            },
                        }
                    )
        for split, lines in lines_by_split.items():
            (stage / f"{split}_prompts.txt").write_text(
                "\n".join(lines["prompts"]) + "\n", encoding="utf-8"
            )
            (stage / f"{split}_videos.txt").write_text(
                "\n".join(lines["videos"]) + "\n", encoding="utf-8"
            )
        manifest = {
            "artifact_kind": "mugen_cogvideox_identity_disjoint_six_verb_dataset",
            "claim": "balanced multi-identity training plus untouched validation/test identities",
            "counts": {
                "identities": dict(IDENTITY_LIMITS),
                "sequences": {
                    split: IDENTITY_LIMITS[split] * len(VERBS) for split in IDENTITY_LIMITS
                },
                "total_identities": sum(IDENTITY_LIMITS.values()),
                "total_sequences": len(output_records),
                "verbs": len(VERBS),
            },
            "policy": {
                "caption_background_cue": "plain neutral gray background",
                "identity_selection": "best worst-frame canonical quality, then UTF-8 identity ID",
                "split": "inherits immutable canonical identity-disjoint split",
                "training_balance": "every selected identity contributes exactly one clip per verb",
                "verbs": list(VERBS),
            },
            "projection": {
                "alpha_composite_background_rgb": [BACKGROUND] * 3,
                "codec": "libx264rgb_crf0_lossless",
                "frame_order": "reference_then_action_frames",
                "model_training_geometry": [720, 480],
                "native_geometry": [128, 128],
                "padding_rgb": [BACKGROUND] * 3,
                "sprite_training_geometry": [480, 480],
                "transform": "nearest-neighbor 128-to-480 then 120px horizontal padding",
            },
            "records": output_records,
            "schema_version": 1,
            "selected_identities": selected_identities,
            "source": {
                "canonical_manifest_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
                "canonical_manifest_path": str(CANONICAL),
                "materialization_manifest_sha256": materialization["file_sha256"],
                "materialization_manifest_path": str(materialization_path),
                "motion_plan_sha256": hashlib.sha256(motion_plan_bytes).hexdigest(),
                "motion_plan_path": str(motion_plan_path),
                "text_plan_sha256": hashlib.sha256(text_bytes).hexdigest(),
                "text_plan_path": str(TEXT_PLAN),
            },
        }
        payload = canonical_json(manifest)
        (stage / "manifest.json").write_bytes(payload)
        DiskGuard(stage, min_free_bytes=100 * 1024**3).require_capacity(
            16 * 1024**2, label="publish MUGEN CogVideoX held-out six-verb export"
        )
        os.rename(stage, OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "bytes": sum(
                    record["video"]["frames"] and (OUTPUT / record["video"]["path"]).stat().st_size
                    for record in output_records
                ),
                "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                "output": str(OUTPUT),
                "sequences": len(output_records),
            },
            sort_keys=True,
        )
    )


def identity_quality_key(records: dict[str, dict[str, object]]) -> tuple[float | bytes, ...]:
    qualities = [records[verb]["eligibility"]["representative_quality"] for verb in VERBS]
    return (
        -min(float(value["minimum_candidate_palette_coverage"]) for value in qualities),
        -min(float(value["minimum_palette_histogram_intersection"]) for value in qualities),
        -min(float(value["minimum_anchored_overlap"]) for value in qualities),
        -min(float(value["minimum_bbox_iou"]) for value in qualities),
        max(float(value["maximum_occupancy_deviation"]) for value in qualities),
        records[VERBS[0]]["identity_id"].encode(),
    )


def corrected_prompt(prompt: str) -> str:
    if prompt.count("transparent background") != 1:
        raise RuntimeError("source prompt background cue differs")
    return prompt.replace("transparent background", "plain neutral gray background")


def load_rgba(root: Path, record: dict[str, object]) -> np.ndarray:
    path = (root / record["relative_path"]).resolve()
    if file_sha256(path) != record["file_sha256"]:
        raise RuntimeError(f"MUGEN pixel payload differs: {path}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise RuntimeError(f"MUGEN pixel geometry differs: {path}")
    return value


def composite_gray(rgba: np.ndarray) -> np.ndarray:
    unit = rgba.astype(np.float32) / 255
    alpha = unit[..., 3:4]
    rgb = unit[..., :3] * alpha + (BACKGROUND / 255) * (1 - alpha)
    return np.rint(rgb * 255).clip(0, 255).astype(np.uint8)


def encode_native_lossless_video(frames: np.ndarray, output: Path) -> None:
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
            "scale=480:480:flags=neighbor,pad=720:480:120:0:color=0x7f7f7f",
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
            f"ffmpeg native export failed: {process.stderr.decode(errors='replace')}"
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
            "crop=480:480:120:0,scale=128:128:flags=neighbor",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if decoded.returncode or decoded.stdout != frames.tobytes():
        raise RuntimeError("lossless native video round trip differs")


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
