"""Evaluate six matched actions for one held-out MUGEN identity."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from itertools import combinations
from pathlib import Path

import decord
import numpy as np
import torch
from diffusers import CogVideoXImageToVideoPipeline
from PIL import Image

ROOT = Path("/home/sleepy/sprite-lab-cogvideox")
MODEL = ROOT / "CogVideoX-5b-I2V-a6f0f4858a83"
DATASET = ROOT / "mugen-cogvideox-heldout-six-verb-v1"
LORA = ROOT / "lora-mugen-heldout-six-verb-r128-step1000-v1"
TRAIN_LOG = ROOT / "lora-mugen-heldout-six-verb-r128-step1000-v1.log"
OUTPUT = Path(
    os.environ.get(
        "SPRITELAB_COGVIDEOX_OUTPUT",
        ROOT / "evaluation-heldout-six-verb-validation-identity-00-v1",
    )
)
SPLIT = os.environ.get("SPRITELAB_COGVIDEOX_SPLIT", "validation")
IDENTITY_INDEX = int(os.environ.get("SPRITELAB_COGVIDEOX_IDENTITY_INDEX", "0"))
SOURCE_INDEX_SHA256 = "98fbc592f23269a38d039d16f969844a9da073b56b24567772433d4b02e2f831"
DATASET_MANIFEST_SHA256 = "589edaa6a314c381d7bb08684a9ec1dc7807c34aa8b05d24f1d036decd819009"
VERBS = ("crouch", "idle", "jump", "normal_attack", "turn", "walk")
NEGATIVE_PROMPT = (
    "photo, realistic, 3d render, blur, soft edges, camera movement, background scene, "
    "multiple characters, sprite sheet, tiled poses, text, watermark"
)
BATCH_SIZE = 2
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0


def main() -> None:
    if SPLIT not in {"validation", "test"}:
        raise ValueError("held-out evaluation split must be validation or test")
    if IDENTITY_INDEX < 0:
        raise ValueError("identity index must be non-negative")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace held-out evaluation: {OUTPUT}")
    source_index_path = MODEL / "source-index.json"
    if file_sha256(source_index_path) != SOURCE_INDEX_SHA256:
        raise RuntimeError("CogVideoX source index differs")
    dataset_path = DATASET / "manifest.json"
    dataset_bytes = dataset_path.read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest() != DATASET_MANIFEST_SHA256:
        raise RuntimeError("held-out six-verb dataset differs")
    dataset = json.loads(dataset_bytes)
    identities = dataset["selected_identities"][SPLIT]
    if len(identities) <= IDENTITY_INDEX:
        raise ValueError("identity index is outside the selected split")
    identity_id = identities[IDENTITY_INDEX]
    records = [
        record
        for record in dataset["records"]
        if record["split"] == SPLIT and record["identity_id"] == identity_id
    ]
    records.sort(key=lambda record: VERBS.index(record["verb"]))
    if [record["verb"] for record in records] != list(VERBS):
        raise RuntimeError("held-out identity verb closure differs")
    targets = {record["verb"]: decode_video(record) for record in records}
    reference = targets[VERBS[0]][0]
    if any(not np.array_equal(targets[verb][0], reference) for verb in VERBS[1:]):
        raise RuntimeError("held-out identity conditioning frames differ across actions")

    weights_path = LORA / "pytorch_lora_weights.safetensors"
    sums_path = LORA / "sha256sums.txt"
    if not weights_path.is_file() or not sums_path.is_file() or not TRAIN_LOG.is_file():
        raise RuntimeError("completed held-out LoRA closure is absent")
    weights_sha256 = file_sha256(weights_path)
    sums_text = sums_path.read_text(encoding="utf-8")
    if weights_sha256 not in sums_text or str(weights_path) not in sums_text:
        raise RuntimeError("held-out LoRA is absent from its checksum closure")

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.load_lora_weights(LORA, adapter_name="mugen-six-verb")
    pipe.set_adapters("mugen-six-verb", adapter_weights=1.0)
    pipe.set_progress_bar_config(disable=True)
    conditioning = Image.fromarray(reference, mode="RGB")
    generated = {}
    seed = 20260902 + IDENTITY_INDEX + (100 if SPLIT == "test" else 0)
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        generators = [torch.Generator(device="cuda").manual_seed(seed) for _ in batch]
        with torch.inference_mode():
            outputs = pipe(
                image=[conditioning.copy() for _ in batch],
                prompt=[record["prompt"] for record in batch],
                negative_prompt=[NEGATIVE_PROMPT] * len(batch),
                height=480,
                width=720,
                num_frames=9,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                use_dynamic_cfg=True,
                generator=generators,
            ).frames
        if len(outputs) != len(batch):
            raise RuntimeError("CogVideoX evaluation batch cardinality differs")
        for record, frames in zip(batch, outputs, strict=True):
            if len(frames) != 9 or any(frame.size != (720, 480) for frame in frames):
                raise RuntimeError(f"generated geometry differs for {record['verb']}")
            generated[record["verb"]] = np.stack(
                [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in frames]
            )
    if set(generated) != set(VERBS):
        raise RuntimeError("generated action closure differs")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.stage-", dir=OUTPUT.parent))
    try:
        action_reports = {}
        overview = Image.new("RGB", (128 * len(VERBS), 256))
        for verb_index, verb in enumerate(VERBS):
            action_dir = stage / verb
            action_dir.mkdir()
            target_display = make_display_frames(targets[verb])
            generated_display = make_display_frames(generated[verb])
            for frame_index, frame in enumerate(generated[verb]):
                Image.fromarray(frame, mode="RGB").save(
                    action_dir / f"frame-{frame_index:02d}-raw-720x480.png", optimize=False
                )
            target_sheet_path = action_dir / "target-display-128-sheet.png"
            generated_sheet_path = action_dir / "generated-display-128-sheet.png"
            animation_path = action_dir / "generated-display-128-animated.png"
            make_sheet(target_display).save(target_sheet_path, optimize=False)
            make_sheet(generated_display).save(generated_sheet_path, optimize=False)
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
            overview.paste(target_display[4], (verb_index * 128, 0))
            overview.paste(generated_display[4], (verb_index * 128, 128))
            action_reports[verb] = {
                "array_sha256": array_sha256(generated[verb]),
                "center_crop_rgb_mae": crop_mae(generated[verb], targets[verb]),
                "generated_animation_file_sha256": file_sha256(animation_path),
                "generated_animation_path": str(Path(verb) / animation_path.name),
                "generated_sheet_file_sha256": file_sha256(generated_sheet_path),
                "generated_sheet_path": str(Path(verb) / generated_sheet_path.name),
                "target_array_sha256": array_sha256(targets[verb]),
                "target_sheet_file_sha256": file_sha256(target_sheet_path),
                "target_sheet_path": str(Path(verb) / target_sheet_path.name),
            }
        overview_path = stage / "target-over-generated-frame4-overview.png"
        overview.save(overview_path, optimize=False)
        matrix = {
            generated_verb: {
                target_verb: crop_mae(generated[generated_verb], targets[target_verb])
                for target_verb in VERBS
            }
            for generated_verb in VERBS
        }
        pair_metrics = []
        for left, right in combinations(VERBS, 2):
            target_separation = crop_mae(targets[left], targets[right])
            generated_separation = crop_mae(generated[left], generated[right])
            pair_metrics.append(
                {
                    "generated_separation_rgb_mae": generated_separation,
                    "generated_over_target_separation_ratio": (
                        generated_separation / target_separation if target_separation > 0 else None
                    ),
                    "left_verb": left,
                    "right_verb": right,
                    "target_separation_rgb_mae": target_separation,
                }
            )
        own_nearest = sum(
            min(matrix[verb], key=lambda candidate: (matrix[verb][candidate], candidate.encode()))
            == verb
            for verb in VERBS
        )
        finite_ratios = [
            pair["generated_over_target_separation_ratio"]
            for pair in pair_metrics
            if pair["generated_over_target_separation_ratio"] is not None
        ]
        report = {
            "actions": action_reports,
            "aggregate": {
                "mean_own_target_rgb_mae": float(np.mean([matrix[verb][verb] for verb in VERBS])),
                "mean_pair_separation_ratio": float(np.mean(finite_ratios)),
                "own_target_nearest_count": own_nearest,
                "own_target_nearest_total": len(VERBS),
            },
            "artifact_kind": "mugen_cogvideox_heldout_six_verb_evaluation",
            "claim": "identity-disjoint in-domain diagnostic; no open-domain generalization",
            "dataset": {
                "identity_id": identity_id,
                "identity_index": IDENTITY_INDEX,
                "manifest_sha256": DATASET_MANIFEST_SHA256,
                "split": SPLIT,
                "verbs": list(VERBS),
            },
            "generation": {
                "batch_size": BATCH_SIZE,
                "guidance_scale": GUIDANCE_SCALE,
                "negative_prompt": NEGATIVE_PROMPT,
                "num_frames": 9,
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "same_noise_across_actions": True,
                "seed": seed,
                "use_dynamic_cfg": True,
            },
            "model": {
                "model_id": "THUDM/CogVideoX-5b-I2V",
                "resolved_revision": "a6f0f4858a8395e7429d82493864ce92bf73af11",
                "source_index_sha256": SOURCE_INDEX_SHA256,
            },
            "pair_metrics": pair_metrics,
            "schema_version": 1,
            "target_error_matrix": matrix,
            "training": {
                "lora_file_sha256": weights_sha256,
                "lora_path": str(weights_path),
                "rank": 128,
                "steps": 1000,
                "training_dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
                "training_log_sha256": file_sha256(TRAIN_LOG),
            },
            "visualization": {
                "frame_index": 4,
                "overview_file_sha256": file_sha256(overview_path),
                "overview_path": overview_path.name,
                "rows": ["exact target", "generated"],
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
                "output": str(OUTPUT),
                "overview_sha256": report["visualization"]["overview_file_sha256"],
                "report_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def decode_video(record: dict[str, object]) -> np.ndarray:
    path = DATASET / record["video"]["path"]
    if file_sha256(path) != record["video"]["file_sha256"]:
        raise RuntimeError(f"held-out target video differs: {path}")
    reader = decord.VideoReader(path.as_posix(), width=720, height=480)
    batch = reader.get_batch(list(range(len(reader))))
    value = batch.cpu().numpy() if isinstance(batch, torch.Tensor) else batch.asnumpy()
    if value.shape != (9, 480, 720, 3) or value.dtype != np.uint8:
        raise RuntimeError(f"held-out target video geometry differs: {path}")
    return value


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
        sheet.paste(frame, (128 * index, 0))
    return sheet


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
