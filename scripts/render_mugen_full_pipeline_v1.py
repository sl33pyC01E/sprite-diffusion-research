"""Render an honest Stage-1 -> Stage-2 -> Stage-3 MUGEN pipeline gallery."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from spritelab.anchored_motion_train import (  # noqa: E402
    AnchoredMotionTrainingConfig,
    sample_anchored_motion_residual,
)
from spritelab.latent_keypose_train import (  # noqa: E402
    LatentKeyposeTrainingConfig,
    _build_keypose_model,
)
from spritelab.latent_motion_train import (  # noqa: E402
    _load_frozen_decoder,
    load_latent_motion_training_corpus,
)
from spritelab.latent_still_inference import (  # noqa: E402
    LatentStillInferenceConfig,
    run_latent_still_inference,
)
from spritelab.models.anchored_latent_motion_dit import (  # noqa: E402
    AnchoredActionConditionedLatentMotionDiT,
)
from spritelab.models.latent_keypose_unet import LatentKeyposeUNetConfig  # noqa: E402
from spritelab.models.latent_motion_dit import LatentMotionDiTConfig  # noqa: E402
from spritelab.previews import export_rgba_clip_preview  # noqa: E402
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.sprite_postprocess import composite_rgba_on_checkerboard  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_file(path: Path, expected_sha256: str, label: str) -> Path:
    resolved = path.resolve()
    if _file_sha256(resolved) != expected_sha256:
        raise ValueError(f"{label} SHA-256 differs")
    return resolved


def _keypose_config(raw: object) -> LatentKeyposeTrainingConfig:
    if not isinstance(raw, dict):
        raise ValueError("key-pose config must be an object")
    values = dict(raw)
    model = values.get("model")
    if not isinstance(model, dict):
        raise ValueError("key-pose model config must be an object")
    values["model"] = LatentMotionDiTConfig(**model)
    unet = values.get("unet")
    if isinstance(unet, dict):
        values["unet"] = LatentKeyposeUNetConfig(**unet)
    values.setdefault("prediction_mode", "endpoint_flow")
    return LatentKeyposeTrainingConfig(**values)


def _motion_config(raw: object) -> AnchoredMotionTrainingConfig:
    if not isinstance(raw, dict):
        raise ValueError("motion config must be an object")
    values = dict(raw)
    model = values.get("model")
    if not isinstance(model, dict):
        raise ValueError("motion model config must be an object")
    values["model"] = LatentMotionDiTConfig(**model)
    return AnchoredMotionTrainingConfig(**values)


def _load_plan(path: Path, expected_sha256: str, identities: tuple[str, ...]) -> list[dict]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Stage-1 plan SHA-256 differs")
    plan = json.loads(payload)
    if plan.get("artifact_kind") != "mugen_latent_still_sequence_training_plan":
        raise ValueError("Stage-1 plan has the wrong artifact kind")
    by_identity = {record["identity_id"]: record for record in plan.get("records", [])}
    records = []
    for identity in identities:
        record = by_identity.get(identity)
        if not isinstance(record, dict):
            raise ValueError(f"Stage-1 plan lacks identity {identity}")
        if record.get("split") != "train":
            raise ValueError(f"identity {identity} is not in the Stage-1 training split")
        records.append(record)
    return records


def _load_stage1_target(record: dict, plan_root: Path) -> np.ndarray:
    target = record.get("target")
    if not isinstance(target, dict):
        raise ValueError("Stage-1 target evidence is absent")
    source = (plan_root / str(target["relative_path"])).resolve()
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != target.get("file_sha256"):
        raise ValueError("Stage-1 target source hash differs")
    array = np.load(io.BytesIO(payload), allow_pickle=False)
    frame_index = int(target["reference_frame_index"])
    if array.dtype != np.uint8 or array.shape != (8, 128, 128, 4):
        raise ValueError("Stage-1 target geometry differs")
    return np.ascontiguousarray(array[frame_index])


def _load_generated_stills(report_path: Path) -> list[np.ndarray]:
    report = json.loads(report_path.read_bytes())
    output = report_path.parent
    values = []
    for sample in report.get("samples", []):
        evidence = sample.get("array", {})
        path = output / str(evidence.get("path"))
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != evidence.get("file_sha256"):
            raise ValueError("Stage-1 generated array file hash differs")
        value = np.load(io.BytesIO(payload), allow_pickle=False)
        if value.dtype != np.uint8 or value.shape != (128, 128, 4):
            raise ValueError("Stage-1 generated array geometry differs")
        values.append(np.ascontiguousarray(value))
    return values


def _save_pose(value: np.ndarray, path: Path) -> dict[str, object]:
    display = Image.fromarray(composite_rgba_on_checkerboard(value))
    display.resize((512, 512), Image.Resampling.NEAREST).save(path, optimize=False)
    return {"file_sha256": _file_sha256(path), "path": path.name}


def _rgba_uint8(value: torch.Tensor) -> np.ndarray:
    return (
        value.detach()
        .float()
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _relative_json(value: object, root: Path) -> object:
    if isinstance(value, Path):
        return str(value.resolve().relative_to(root.resolve()))
    if isinstance(value, dict):
        return {str(key): _relative_json(item, root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_relative_json(item, root) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-plan", type=Path, required=True)
    parser.add_argument("--expected-stage1-plan-sha256", required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-stage1-checkpoint-sha256", required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-codec-checkpoint-sha256", required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--expected-text-source-index-sha256", required=True)
    parser.add_argument("--motion-manifest", type=Path, required=True)
    parser.add_argument("--keypose-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-keypose-checkpoint-sha256", required=True)
    parser.add_argument("--motion-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-motion-checkpoint-sha256", required=True)
    parser.add_argument("--identity", action="append", required=True)
    parser.add_argument("--verb", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    identities = tuple(dict.fromkeys(args.identity))
    verbs = tuple(dict.fromkeys(args.verb))
    if len(identities) != len(args.identity) or len(verbs) != len(args.verb):
        raise ValueError("identity and verb selections must be unique")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace pipeline output: {output}")
    guard = DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(2 * 1024**3, label="full MUGEN pipeline gallery")

    plan_path = _verified_file(args.stage1_plan, args.expected_stage1_plan_sha256, "Stage-1 plan")
    stage1_checkpoint = _verified_file(
        args.stage1_checkpoint,
        args.expected_stage1_checkpoint_sha256,
        "Stage-1 checkpoint",
    )
    codec_checkpoint = _verified_file(
        args.codec_checkpoint,
        args.expected_codec_checkpoint_sha256,
        "codec checkpoint",
    )
    keypose_checkpoint = _verified_file(
        args.keypose_checkpoint,
        args.expected_keypose_checkpoint_sha256,
        "key-pose checkpoint",
    )
    motion_checkpoint = _verified_file(
        args.motion_checkpoint,
        args.expected_motion_checkpoint_sha256,
        "motion checkpoint",
    )
    records = _load_plan(plan_path, args.expected_stage1_plan_sha256, identities)
    prompts = [str(record["prompt"]) for record in records]

    output.mkdir(parents=True, exist_ok=False)
    stage1_output = output / "stage1"
    stage1_report_path, stage1_report_sha256 = run_latent_still_inference(
        stage1_checkpoint,
        codec_checkpoint,
        args.text_model,
        prompts,
        stage1_output,
        expected_checkpoint_sha256=args.expected_stage1_checkpoint_sha256,
        expected_codec_checkpoint_sha256=args.expected_codec_checkpoint_sha256,
        expected_text_source_index_sha256=args.expected_text_source_index_sha256,
        config=LatentStillInferenceConfig(
            seed=args.seed,
            sample_steps=32,
            guidance_scale=3.5,
        ),
        disk_guard=guard,
    )
    generated_stills = _load_generated_stills(stage1_report_path)
    if len(generated_stills) != len(records):
        raise ValueError("Stage-1 generated count differs")
    gc.collect()
    torch.cuda.empty_cache()

    corpus = load_latent_motion_training_corpus(
        args.motion_manifest.resolve(), verify_hashes=True, array_loading="lazy"
    )
    if any(verb not in corpus.action_vocabulary for verb in verbs):
        raise ValueError("selected verb is absent from the motion vocabulary")
    row_lookup = {(row.identity_id, row.verb): index for index, row in enumerate(corpus.rows)}
    selections = []
    for identity in identities:
        for verb in verbs:
            index = row_lookup.get((identity, verb))
            if index is None:
                raise ValueError(f"motion corpus lacks {identity}/{verb}")
            if corpus.rows[index].split != "train":
                raise ValueError(f"{identity}/{verb} is not in the motion training split")
            selections.append(index)

    device = torch.device("cuda")
    decoder = _load_frozen_decoder(torch, corpus, device=device).eval()
    still_tensor = (
        torch.from_numpy(np.stack(generated_stills))
        .to(device=device, dtype=torch.float32)
        .permute(0, 3, 1, 2)
        .div(255)
    )
    with torch.no_grad():
        reference_latent = decoder.encode(still_tensor)
    mean = torch.tensor(corpus.channel_mean, device=device).view(1, 8, 1, 1)
    std = torch.tensor(corpus.channel_standard_deviation, device=device).view(1, 8, 1, 1)
    references = (reference_latent - mean) / std
    references = references.repeat_interleave(len(verbs), dim=0)
    action_indices = torch.tensor(
        [corpus.rows[index].action_index for index in selections], device=device
    )

    keypose_payload = torch.load(keypose_checkpoint, map_location="cpu", weights_only=True)
    if (
        keypose_payload.get("artifact_kind")
        != "mugen_fixed_middle_latent_keypose_resume_checkpoint"
    ):
        raise ValueError("key-pose checkpoint has the wrong artifact kind")
    if keypose_payload.get("corpus") != corpus.contract:
        raise ValueError("key-pose corpus contract differs")
    keypose_config = _keypose_config(keypose_payload.get("config"))
    keypose_model = _build_keypose_model(keypose_config, len(corpus.action_vocabulary)).to(device)
    keypose_model.load_state_dict(keypose_payload["raw_model"], strict=True)
    keypose_model.eval()
    keypose_input = torch.zeros_like(references).unsqueeze(1)
    keypose_phase = torch.full((len(selections), 1), 0.5, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        keypose_velocity = keypose_model(
            keypose_input,
            references,
            torch.ones((len(selections),), device=device),
            action_indices,
            frame_phase=keypose_phase,
        )[:, 0]
    keypose_residual = -keypose_velocity.float()
    keypose_latent = (references + keypose_residual) * std + mean
    with torch.no_grad():
        keypose_rgba = decoder.decode(keypose_latent)
    keypose_arrays = _rgba_uint8(keypose_rgba)
    del keypose_model, keypose_payload
    gc.collect()
    torch.cuda.empty_cache()

    motion_payload = torch.load(motion_checkpoint, map_location="cpu", weights_only=True)
    if (
        motion_payload.get("artifact_kind")
        != "mugen_start_middle_start_anchored_motion_resume_checkpoint"
    ):
        raise ValueError("motion checkpoint has the wrong artifact kind")
    if motion_payload.get("corpus") != corpus.contract:
        raise ValueError("motion corpus contract differs")
    motion_config = _motion_config(motion_payload.get("config"))
    motion_model = AnchoredActionConditionedLatentMotionDiT(
        motion_config.model,
        len(corpus.action_vocabulary),
        action_token_count=motion_config.action_token_count,
        action_condition_scale=motion_config.action_condition_scale,
    ).to(device)
    motion_model.load_state_dict(motion_payload["ema_model"], strict=True)
    motion_model.eval()
    phases = torch.from_numpy(np.ascontiguousarray(corpus.phases[list(selections)])).to(
        device=device, dtype=torch.float32
    )
    anchor_mask = torch.zeros((len(selections), 8), device=device, dtype=torch.bool)
    anchor_mask[:, (0, 4, 7)] = True
    anchors = torch.zeros((len(selections), 8, 8, 64, 64), device=device, dtype=torch.float32)
    anchors[:, 4] = keypose_residual
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    identity_noise = torch.randn((len(identities), 8, 8, 64, 64), generator=generator)
    noise = identity_noise.repeat_interleave(len(verbs), dim=0).to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        motion_residual = sample_anchored_motion_residual(
            torch,
            motion_model,
            noise=noise,
            reference=references,
            actions=action_indices,
            phases=phases,
            anchor_residuals=anchors,
            anchor_mask=anchor_mask,
        )
    video_latent = (references.unsqueeze(1) + motion_residual.float()) * std.unsqueeze(
        1
    ) + mean.unsqueeze(1)
    with torch.no_grad():
        video_rgba = decoder.decode(video_latent.reshape(-1, 8, 64, 64)).reshape(
            len(selections), 8, 4, 128, 128
        )
    video_arrays = (
        video_rgba.detach()
        .float()
        .clamp(0, 1)
        .permute(0, 1, 3, 4, 2)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    report_rows = []
    for identity_index, (identity, record) in enumerate(zip(identities, records, strict=True)):
        identity_stem = f"identity-{identity_index:02d}-{identity[-8:]}"
        stage1_target = _load_stage1_target(record, plan_path.parent)
        target_path = output / f"{identity_stem}-stage1-target.png"
        stage1_target_preview = _save_pose(stage1_target, target_path)
        sample = json.loads(stage1_report_path.read_bytes())["samples"][identity_index]
        action_rows = []
        for verb_index, verb in enumerate(verbs):
            flat = identity_index * len(verbs) + verb_index
            row_index = selections[flat]
            source_row = corpus.rows[row_index]
            stage2_path = output / f"{identity_stem}-{verb}-stage2-keypose.png"
            stage2_preview = _save_pose(keypose_arrays[flat], stage2_path)
            stage2_target_path = output / f"{identity_stem}-{verb}-stage2-target.png"
            stage2_target = _save_pose(
                np.ascontiguousarray(corpus.target_rgba[row_index][4]), stage2_target_path
            )
            stage3_preview = export_rgba_clip_preview(
                np.ascontiguousarray(video_arrays[flat]),
                output,
                artifact_stem=f"{identity_stem}-{verb}-stage3-animation",
                duration_ms=source_row.duration_ms,
                loop_mode="loop",
                integer_scale=2,
                preserve_frame_slots=True,
                disk_guard=guard,
            )
            stage3_target = export_rgba_clip_preview(
                np.ascontiguousarray(corpus.target_rgba[row_index]),
                output,
                artifact_stem=f"{identity_stem}-{verb}-stage3-target",
                duration_ms=source_row.duration_ms,
                loop_mode="loop",
                integer_scale=2,
                preserve_frame_slots=True,
                disk_guard=guard,
            )
            action_rows.append(
                {
                    "sequence_id": source_row.sequence_id,
                    "verb": verb,
                    "stage2_generated": stage2_preview,
                    "stage2_target": stage2_target,
                    "stage3_generated": _relative_json(asdict(stage3_preview), output),
                    "stage3_target": _relative_json(asdict(stage3_target), output),
                }
            )
        report_rows.append(
            {
                "identity_id": identity,
                "prompt": record["prompt"],
                "split": "train",
                "stage1_generated": sample,
                "stage1_target": stage1_target_preview,
                "actions": action_rows,
            }
        )
    report = {
        "artifact_kind": "mugen_three_stage_chained_in_distribution_gallery",
        "claim": (
            "All identities are in every stage's training split. Stage 2 consumes the "
            "generated Stage-1 sprite; Stage 3 consumes that same sprite and the generated "
            "Stage-2 key pose. Outputs are raw model results with checkerboard display only."
        ),
        "config": {
            "seed": args.seed,
            "verbs": list(verbs),
            "stage1": {"guidance_scale": 3.5, "sample_steps": 32},
            "stage2_state": "raw",
            "stage3_state": "ema",
            "shared_noise_across_verbs_per_identity": True,
        },
        "evidence": {
            "stage1_checkpoint_sha256": args.expected_stage1_checkpoint_sha256,
            "stage1_inference_report_sha256": stage1_report_sha256,
            "stage1_plan_sha256": args.expected_stage1_plan_sha256,
            "codec_checkpoint_sha256": args.expected_codec_checkpoint_sha256,
            "keypose_checkpoint_sha256": args.expected_keypose_checkpoint_sha256,
            "motion_checkpoint_sha256": args.expected_motion_checkpoint_sha256,
        },
        "rows": report_rows,
        "schema_version": 1,
    }
    report_path = output / "report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "output": str(output),
                "report": str(report_path),
                "report_sha256": _file_sha256(report_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
