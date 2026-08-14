"""Evaluate a temporal MUGEN LoRA with shared-noise action comparisons."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_mugen_animatediff_sparsectrl_motion_lora_v1 import (  # noqa: E402
    BASE,
    BASE_SOURCE_INDEX_SHA256,
    CONTROL,
    CONTROL_SOURCE_INDEX_SHA256,
    MOTION,
    MOTION_SOURCE_INDEX_SHA256,
    REFERENCE_CACHE,
    STILL_CHECKPOINT,
    STILL_CHECKPOINT_SHA256,
    TARGET_CACHE,
    TEMPORAL_TARGET_REGEX,
    TEXT_CACHE,
    TEXT_PLAN,
    _file_sha256,
    _load_latent,
    _unique,
    _verify_source_index,
)

from spritelab.sd_lora_train import sd_lora_target_modules  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

CANONICAL = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"


def main() -> None:
    args = _arguments()
    output = args.output.resolve()
    checkpoint_path = args.checkpoint.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace motion-LoRA evaluation: {output}")
    if _file_sha256(checkpoint_path) != args.expected_checkpoint_sha256:
        raise RuntimeError("motion-LoRA checkpoint SHA-256 differs")
    DiskGuard(ROOT, min_free_bytes=100 * 1024**3).require_capacity(
        2 * 1024**3, label="MUGEN AnimateDiff motion-LoRA evaluation"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact_kind") not in {
        "mugen_animatediff_sparsectrl_temporal_adapter_checkpoint",
        "mugen_animatediff_sparsectrl_temporal_lora_checkpoint",
    }:
        raise RuntimeError("motion-LoRA checkpoint kind differs")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("motion-LoRA checkpoint config is absent")
    rows, caches = _load_rows(args.identity)
    if not rows:
        raise RuntimeError("evaluation identity is absent")
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        MotionAdapter,
        PNDMScheduler,
        SparseControlNetModel,
        UNet2DConditionModel,
        UNetMotionModel,
    )
    from peft import LoraConfig, set_peft_model_state_dict

    _verify_source_index(BASE, BASE_SOURCE_INDEX_SHA256)
    _verify_source_index(MOTION, MOTION_SOURCE_INDEX_SHA256)
    _verify_source_index(CONTROL, CONTROL_SOURCE_INDEX_SHA256)
    if _file_sha256(STILL_CHECKPOINT) != STILL_CHECKPOINT_SHA256:
        raise RuntimeError("MUGEN still-LoRA checkpoint differs")
    dtype = torch.float16
    base_unet = UNet2DConditionModel.from_pretrained(
        BASE / "unet", local_files_only=True, torch_dtype=dtype
    )
    still = torch.load(STILL_CHECKPOINT, map_location="cpu", weights_only=True)
    still_config = still["config"]
    base_unet.add_adapter(
        LoraConfig(
            r=int(still_config["rank"]),
            lora_alpha=int(still_config["alpha"]),
            target_modules=list(
                sd_lora_target_modules(still_config.get("target_profile", "attention"))
            ),
        ),
        adapter_name="mugen_still",
    )
    set_peft_model_state_dict(base_unet, still["ema_lora"], adapter_name="mugen_still")
    base_unet.fuse_lora(adapter_names=["mugen_still"])
    base_unet.unload_lora()
    motion_adapter = MotionAdapter.from_pretrained(
        MOTION, variant="fp16", torch_dtype=dtype, local_files_only=True
    )
    unet = UNetMotionModel.from_unet2d(base_unet, motion_adapter).to("cuda").eval()
    del base_unet, motion_adapter, still
    unet.requires_grad_(False)
    target_profile = config.get("target_profile", "temporal_lora")
    if target_profile == "temporal_lora":
        unet.add_adapter(
            LoraConfig(
                r=int(config["rank"]),
                lora_alpha=int(config["alpha"]),
                target_modules=TEMPORAL_TARGET_REGEX,
            ),
            adapter_name="mugen_motion",
        )
    variant_key = f"{args.weights_variant}_trainable_state"
    state = checkpoint.get(variant_key)
    if state is None and target_profile == "temporal_lora":
        state = checkpoint.get(f"{args.weights_variant}_lora")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint lacks {variant_key}")
    if target_profile == "temporal_lora":
        set_peft_model_state_dict(unet, state, adapter_name="mugen_motion")
    elif target_profile == "full_motion":
        named_parameters = dict(unet.named_parameters())
        expected = {name for name in named_parameters if "motion_modules" in name}
        if set(state) != expected:
            raise RuntimeError("full-motion checkpoint parameter names differ")
        with torch.no_grad():
            for name, value in state.items():
                named_parameters[name].copy_(value)
    else:
        raise RuntimeError("checkpoint target profile differs")
    controlnet = (
        SparseControlNetModel.from_pretrained(
            CONTROL, variant="fp16", torch_dtype=dtype, local_files_only=True
        )
        .to("cuda")
        .eval()
    )
    controlnet.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(BASE / "vae", local_files_only=True).to("cuda").eval()
    vae.requires_grad_(False)
    scheduler = DDIMScheduler.from_config(PNDMScheduler.load_config(BASE / "scheduler"))
    scheduler.set_timesteps(args.sample_steps, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    shared_noise = torch.randn((1, 4, 8, 64, 64), device="cuda", dtype=dtype, generator=generator)
    noise_sha256 = _array_sha256(shared_noise.float().cpu().numpy())
    target_root = TARGET_CACHE.parent
    reference_root = REFERENCE_CACHE.parent
    embeddings = caches["embeddings"]
    unconditional = caches["unconditional"]
    prompt_rows = caches["prompt_rows"]
    target_by_sequence = caches["target_by_sequence"]
    reference_by_sequence = caches["reference_by_sequence"]
    materialization_root = Path(caches["materialization_root"])
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    sample_records = []
    generated_arrays = []
    target_arrays = []
    try:
        for index, row in enumerate(rows):
            _load_latent(target_root, target_by_sequence[row["sequence_id"]])
            reference_record = reference_by_sequence[row["reference_sequence_id"]]
            reference_clip = _load_latent(reference_root, reference_record)
            reference = torch.from_numpy(reference_clip[row["reference_frame_index"]].copy()).to(
                "cuda", dtype=dtype
            )
            context_array = np.array(embeddings[prompt_rows[row["prompt"]]], copy=True)
            conditional = torch.from_numpy(context_array).to("cuda", dtype=dtype)[None]
            unconditional_tensor = torch.from_numpy(np.array(unconditional, copy=True)).to(
                "cuda", dtype=dtype
            )[None]
            frame_context = torch.cat((unconditional_tensor, conditional)).repeat_interleave(
                8, dim=0
            )
            control = torch.zeros((2, 4, 8, 64, 64), device="cuda", dtype=dtype)
            control[:, :, 0] = reference
            mask = torch.zeros((2, 1, 8, 64, 64), device="cuda", dtype=dtype)
            mask[:, :, 0] = 1
            latent = shared_noise.clone() * scheduler.init_noise_sigma
            with torch.inference_mode():
                for timestep in scheduler.timesteps:
                    model_input = scheduler.scale_model_input(latent, timestep)
                    doubled = torch.cat((model_input, model_input))
                    with torch.autocast("cuda", dtype=dtype):
                        down, middle = controlnet(
                            doubled,
                            timestep,
                            encoder_hidden_states=frame_context,
                            controlnet_cond=control,
                            conditioning_mask=mask,
                            conditioning_scale=args.control_scale,
                            return_dict=False,
                        )
                        prediction = unet(
                            doubled,
                            timestep,
                            encoder_hidden_states=frame_context,
                            down_block_additional_residuals=down,
                            mid_block_additional_residual=middle,
                        ).sample
                    negative, positive = prediction.chunk(2)
                    guided = negative + args.guidance_scale * (positive - negative)
                    latent = scheduler.step(guided, timestep, latent).prev_sample
                frames = latent[0].permute(1, 0, 2, 3)
                with torch.autocast("cuda", dtype=dtype):
                    decoded = vae.decode(frames / float(vae.config.scaling_factor)).sample
            rgb_512 = (
                decoded.float()
                .add(1)
                .mul(127.5)
                .clamp(0, 255)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
                .transpose(0, 2, 3, 1)
            )
            generated = np.stack(
                [
                    np.asarray(
                        Image.fromarray(frame).resize((128, 128), Image.Resampling.BOX),
                        dtype=np.uint8,
                    )
                    for frame in rgb_512
                ]
            )
            target_rgba = _load_target_pixels(materialization_root, row["source_pixels"])
            target_rgb = _composite_gray(target_rgba)
            generated_arrays.append(generated)
            target_arrays.append(target_rgb)
            stem = f"{index:02d}-{row['verb'].replace('_', '-')}-{row['sequence_id'][-8:]}"
            target_sheet, target_animation = _save_preview(target_rgb, stage, f"{stem}-target")
            generated_sheet, generated_animation = _save_preview(
                generated, stage, f"{stem}-generated"
            )
            sample_records.append(
                {
                    "generated": {
                        "animated_path": generated_animation.name,
                        "animated_sha256": _file_sha256(generated_animation),
                        "sheet_path": generated_sheet.name,
                        "sheet_sha256": _file_sha256(generated_sheet),
                    },
                    "identity_id": row["identity_id"],
                    "prompt": row["prompt"],
                    "sequence_id": row["sequence_id"],
                    "split": row["split"],
                    "target": {
                        "animated_path": target_animation.name,
                        "animated_sha256": _file_sha256(target_animation),
                        "sheet_path": target_sheet.name,
                        "sheet_sha256": _file_sha256(target_sheet),
                    },
                    "verb": row["verb"],
                }
            )
        generated_stack = np.stack(generated_arrays).astype(np.float32) / 255
        target_stack = np.stack(target_arrays).astype(np.float32) / 255
        per_sample_mae = np.abs(generated_stack - target_stack).mean(axis=(1, 2, 3, 4))
        cross_mae = np.abs(generated_stack[:, None] - target_stack[None, :]).mean(axis=(2, 3, 4, 5))
        own_target_ranks = np.argsort(np.argsort(cross_mae, axis=1), axis=1).diagonal() + 1
        generated_pairwise = _mean_pairwise_mae(generated_stack)
        target_pairwise = _mean_pairwise_mae(target_stack)
        report = {
            "artifact_kind": "mugen_animatediff_sparsectrl_temporal_lora_evaluation",
            "claim": (
                "shared-noise exact-request evaluation; in-sample when selection.split=train; "
                "RGB gray-composited quality control only"
            ),
            "checkpoint": {
                "file_sha256": args.expected_checkpoint_sha256,
                "path": str(checkpoint_path),
                "step": checkpoint["step"],
                "weights_variant": args.weights_variant,
            },
            "generation": {
                "controlnet_conditioning_scale": args.control_scale,
                "guidance_scale": args.guidance_scale,
                "noise_batch_sha256": noise_sha256,
                "sample_steps": args.sample_steps,
                "scheduler": "DDIM",
                "seed": args.seed,
            },
            "metrics": {
                "generated_action_pairwise_rgb_mae": generated_pairwise,
                "generated_to_target_action_separation_ratio": (
                    generated_pairwise / target_pairwise if target_pairwise else None
                ),
                "mean_rgb_mae": float(per_sample_mae.mean()),
                "own_target_nearest_count": int((own_target_ranks == 1).sum()),
                "own_target_nearest_denominator": len(rows),
                "own_target_ranks": {
                    row["sequence_id"]: int(rank)
                    for row, rank in zip(rows, own_target_ranks, strict=True)
                },
                "per_sequence_rgb_mae": {
                    row["sequence_id"]: float(value)
                    for row, value in zip(rows, per_sample_mae, strict=True)
                },
                "target_action_pairwise_rgb_mae": target_pairwise,
            },
            "samples": sample_records,
            "schema_version": 1,
            "selection": {
                "identity_id": args.identity,
                "sequences": len(rows),
                "splits": sorted({row["split"] for row in rows}),
            },
        }
        report_payload = _canonical_json(report)
        (stage / "evaluation-report.json").write_bytes(report_payload)
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "mean_rgb_mae": report["metrics"]["mean_rgb_mae"],
                "output": str(output),
                "report_sha256": hashlib.sha256(report_payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights-variant", choices=("ema", "raw"), default="ema")
    parser.add_argument("--sample-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if len(args.expected_checkpoint_sha256) != 64:
        parser.error("--expected-checkpoint-sha256 must be a SHA-256 digest")
    return args


def _load_rows(identity: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    canonical = json.loads(CANONICAL.read_bytes())
    target = json.loads(TARGET_CACHE.read_bytes())
    reference = json.loads(REFERENCE_CACHE.read_bytes())
    text_plan = json.loads(TEXT_PLAN.read_bytes())
    text = json.loads(TEXT_CACHE.read_bytes())
    target_by_sequence = _unique(target["records"], "sequence_id", "target cache")
    reference_by_sequence = _unique(reference["records"], "sequence_id", "reference cache")
    text_by_sequence = _unique(text_plan["records"], "sequence_id", "text plan")
    prompt_rows = {row["prompt"]: row["row_index"] for row in text["rows"]}
    embedding_record = text["arrays"]["embeddings"]
    unconditional_record = text["arrays"]["unconditional_embeddings"]
    embedding_path = TEXT_CACHE.parent / embedding_record["path"]
    unconditional_path = TEXT_CACHE.parent / unconditional_record["path"]
    if _file_sha256(embedding_path) != embedding_record["file_sha256"]:
        raise RuntimeError("text embeddings differ")
    if _file_sha256(unconditional_path) != unconditional_record["file_sha256"]:
        raise RuntimeError("unconditional text embedding differs")
    motion_plan = json.loads(Path(canonical["source"]["motion_plan_path"]).read_bytes())
    materialization_path = Path(motion_plan["source"]["materialization"]["path"])
    rows = []
    for record in canonical["records"]:
        if record.get("identity_id") != identity:
            continue
        sequence_id = record["sequence_id"]
        text_record = text_by_sequence[sequence_id]
        rows.append(
            {
                "identity_id": identity,
                "prompt": text_record["prompt"],
                "reference_frame_index": int(record["reference"]["frame_index"]),
                "reference_sequence_id": record["reference"]["sequence_id"],
                "sequence_id": sequence_id,
                "source_pixels": record["target"]["source_pixels"],
                "split": record["split"],
                "verb": record["conditioning"]["verb"],
            }
        )
    rows.sort(key=lambda row: (str(row["verb"]).encode(), str(row["sequence_id"]).encode()))
    return rows, {
        "embeddings": np.load(embedding_path, mmap_mode="r", allow_pickle=False),
        "materialization_root": materialization_path.parent.resolve(),
        "prompt_rows": prompt_rows,
        "reference_by_sequence": reference_by_sequence,
        "target_by_sequence": target_by_sequence,
        "unconditional": np.load(unconditional_path, mmap_mode="r", allow_pickle=False),
    }


def _load_target_pixels(root: Path, record: dict[str, object]) -> np.ndarray:
    path = (root / str(record["relative_path"])).resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["file_sha256"]:
        raise RuntimeError(f"target pixels differ: {path}")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise RuntimeError(f"target pixel geometry differs: {path}")
    return value


def _composite_gray(rgba: np.ndarray) -> np.ndarray:
    unit = rgba.astype(np.float32) / 255
    alpha = unit[..., 3:4]
    value = unit[..., :3] * alpha + (127 / 255) * (1 - alpha)
    return np.rint(value * 255).clip(0, 255).astype(np.uint8)


def _save_preview(frames: np.ndarray, root: Path, stem: str) -> tuple[Path, Path]:
    images = [Image.fromarray(frame, mode="RGB") for frame in frames]
    sheet = Image.new("RGB", (128 * 8, 128))
    for index, image in enumerate(images):
        sheet.paste(image, (index * 128, 0))
    sheet_path = root / f"{stem}-sheet.png"
    sheet.save(sheet_path, optimize=False)
    animated_path = root / f"{stem}-animated.png"
    images[0].save(
        animated_path,
        save_all=True,
        append_images=images[1:],
        duration=[100] * 8,
        loop=0,
        disposal=2,
        blend=0,
        optimize=False,
    )
    return sheet_path, animated_path


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _mean_pairwise_mae(value: np.ndarray) -> float:
    distances = [
        float(np.abs(value[left] - value[right]).mean())
        for left in range(len(value))
        for right in range(left + 1, len(value))
    ]
    return float(np.mean(distances)) if distances else 0.0


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


if __name__ == "__main__":
    main()
