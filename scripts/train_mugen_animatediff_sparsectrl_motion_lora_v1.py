"""Train a temporal-only LoRA on reference-conditioned canonical MUGEN motion."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.sd_lora_train import sd_lora_target_modules  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

BASE_REVISION = "8229c9b6e928103f0e657cfe6b14d902cb2101d6"
BASE = ROOT / f"data/models/sd-pixelart-spritesheet-{BASE_REVISION}"
BASE_SOURCE_INDEX_SHA256 = "5f7eea291d7831ccb4d6bb07b011669f532ebd4371dee35b8743ea90fbc926df"
MOTION = ROOT / "data/models/animatediff-motion-adapter-v1-5-3-2e8139b1d126"
MOTION_SOURCE_INDEX_SHA256 = "7844d33efad235367e8f8e769dc775d343dc840fc080d010d3adb3cc64006bc0"
CONTROL = ROOT / "data/models/animatediff-sparsectrl-rgb-b8003d681d81"
CONTROL_SOURCE_INDEX_SHA256 = "bbe727be5a232e507bba52a67d36053cc4ea957bdf09d3c22cab95e499d40ec7"
STILL_CHECKPOINT = (
    ROOT
    / "data/experiments/mugen-mffa-sd-pixelart-lora-canonical-v1-step2500-continuation-v1"
    / "training-step-0002500.pt"
)
STILL_CHECKPOINT_SHA256 = "0bc6640361843d3a4b18f67f532f5e0c01c6a00392c0241544fed349b400f68f"
CANONICAL = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"
TARGET_CACHE = (
    ROOT / "data/processed/mugen-mffa-sd-pixelart-rgb-vae-latents-primary-motion-v1/manifest.json"
)
REFERENCE_CACHE = (
    ROOT / "data/processed/mugen-mffa-sd-pixelart-rgb-vae-latents-canonical-v1/manifest.json"
)
TEXT_PLAN = ROOT / "data/processed/mugen-mffa-sd-primary-motion-text-plan-v1.json"
TEXT_CACHE = (
    ROOT / "data/processed/mugen-mffa-sd-pixelart-clip-token-states-primary-motion-v1/manifest.json"
)
TEMPORAL_TARGET_REGEX = r".*motion_modules.*\.(to_q|to_k|to_v|to_out\.0)$"


def main() -> None:
    args = _arguments()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace motion-LoRA output: {output}")
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    required_capacity = 8 * 1024**3 if args.target_profile == "full_motion" else 2 * 1024**3
    guard.require_capacity(required_capacity, label="MUGEN AnimateDiff temporal training")
    inputs = _load_inputs(args.identity)
    rows = inputs["rows"]
    if not rows:
        raise RuntimeError("training selection is empty")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "alpha": args.rank if args.target_profile == "temporal_lora" else None,
        "control_frame_index": 0,
        "controlnet_conditioning_scale": args.control_scale,
        "ema_decay": args.ema_decay,
        "gradient_clip_norm": 1.0,
        "learning_rate": args.learning_rate,
        "min_snr_gamma": args.min_snr_gamma,
        "precision": args.precision,
        "rank": args.rank if args.target_profile == "temporal_lora" else None,
        "seed": args.seed,
        "steps": args.steps,
        "target_modules_regex": (
            TEMPORAL_TARGET_REGEX if args.target_profile == "temporal_lora" else None
        ),
        "target_profile": args.target_profile,
        "trainable_parameter_dtype": (
            "float32" if args.target_profile == "temporal_lora" else args.precision
        ),
        "weight_decay": 0.01,
    }
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    from diffusers import (
        DDPMScheduler,
        MotionAdapter,
        PNDMScheduler,
        SparseControlNetModel,
        UNet2DConditionModel,
        UNetMotionModel,
    )
    from peft import (
        LoraConfig,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )

    _verify_source_index(BASE, BASE_SOURCE_INDEX_SHA256)
    _verify_source_index(MOTION, MOTION_SOURCE_INDEX_SHA256)
    _verify_source_index(CONTROL, CONTROL_SOURCE_INDEX_SHA256)
    if _file_sha256(STILL_CHECKPOINT) != STILL_CHECKPOINT_SHA256:
        raise RuntimeError("MUGEN still-LoRA checkpoint differs")
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
    base_unet = UNet2DConditionModel.from_pretrained(
        BASE / "unet", local_files_only=True, torch_dtype=dtype
    )
    still = torch.load(STILL_CHECKPOINT, map_location="cpu", weights_only=True)
    still_config = still.get("config")
    if not isinstance(still_config, dict):
        raise RuntimeError("MUGEN still-LoRA config is missing")
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
    still_state = still.get("ema_lora")
    if not isinstance(still_state, dict) or not still_state:
        raise RuntimeError("MUGEN still-LoRA EMA is missing")
    set_peft_model_state_dict(base_unet, still_state, adapter_name="mugen_still")
    base_unet.fuse_lora(adapter_names=["mugen_still"])
    base_unet.unload_lora()
    motion_adapter = MotionAdapter.from_pretrained(
        MOTION, variant="fp16", torch_dtype=dtype, local_files_only=True
    )
    unet = UNetMotionModel.from_unet2d(base_unet, motion_adapter).to("cuda")
    del base_unet, motion_adapter, still, still_state
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    adapter_name = None
    if args.target_profile == "temporal_lora":
        adapter_name = "mugen_motion"
        unet.add_adapter(
            LoraConfig(
                r=args.rank,
                lora_alpha=args.rank,
                target_modules=TEMPORAL_TARGET_REGEX,
                init_lora_weights="gaussian",
            ),
            adapter_name=adapter_name,
        )
    else:
        for name, parameter in unet.named_parameters():
            parameter.requires_grad_("motion_modules" in name)
    trainable = [
        (name, parameter) for name, parameter in unet.named_parameters() if parameter.requires_grad
    ]
    if not trainable or any("motion_modules" not in name for name, _ in trainable):
        raise RuntimeError("temporal LoRA attachment escaped motion modules")
    if args.target_profile == "temporal_lora":
        for _, parameter in trainable:
            parameter.data = parameter.data.float()
    controlnet = SparseControlNetModel.from_pretrained(
        CONTROL, variant="fp16", torch_dtype=dtype, local_files_only=True
    ).to("cuda")
    if not controlnet.use_simplified_condition_embedding:
        raise RuntimeError("SparseCtrl checkpoint does not accept SD VAE control latents")
    controlnet.requires_grad_(False)
    controlnet.eval()
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=args.learning_rate,
        weight_decay=0.01,
        fused=True,
    )
    noise_scheduler = DDPMScheduler.from_config(PNDMScheduler.load_config(BASE / "scheduler"))
    noise_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    sampler = random.Random(args.seed + 2)
    ema = _trainable_state(
        unet,
        target_profile=args.target_profile,
        adapter_name=adapter_name,
        get_peft_model_state_dict=get_peft_model_state_dict,
        clone=True,
    )
    embeddings = inputs["embeddings"]
    row_by_prompt = inputs["text_row_by_prompt"]
    reference_by_sequence = inputs["reference_by_sequence"]
    target_by_sequence = inputs["target_by_sequence"]
    reference_root = REFERENCE_CACHE.parent
    target_root = TARGET_CACHE.parent
    history_path = output / "training-history.jsonl"
    unet.train()
    with history_path.open("x", encoding="utf-8", newline="\n") as history:
        for step in range(1, args.steps + 1):
            row = rows[sampler.randrange(len(rows))]
            target = _load_latent(target_root, target_by_sequence[row["sequence_id"]])
            reference_record = reference_by_sequence[row["reference_sequence_id"]]
            reference_clip = _load_latent(reference_root, reference_record)
            reference = reference_clip[row["reference_frame_index"]]
            target_tensor = (
                torch.from_numpy(target).to("cuda", dtype=dtype).permute(1, 0, 2, 3)[None]
            )
            reference_tensor = torch.from_numpy(reference).to("cuda", dtype=dtype)
            context_array = np.array(
                embeddings[row_by_prompt[row["prompt"]]], dtype=np.float16, copy=True
            )
            context = torch.from_numpy(context_array).to("cuda", dtype=dtype)[None]
            frame_context = context.repeat_interleave(8, dim=0)
            timestep = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (1,),
                device="cuda",
                generator=noise_generator,
            )
            noise = torch.randn(
                target_tensor.shape,
                device="cuda",
                dtype=dtype,
                generator=noise_generator,
            )
            noisy = noise_scheduler.add_noise(target_tensor, noise, timestep)
            control = torch.zeros_like(target_tensor)
            control[:, :, 0] = reference_tensor
            mask = torch.zeros((1, 1, 8, 64, 64), device="cuda", dtype=dtype)
            mask[:, :, 0] = 1
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                down, middle = controlnet(
                    noisy,
                    timestep,
                    encoder_hidden_states=frame_context,
                    controlnet_cond=control,
                    conditioning_mask=mask,
                    conditioning_scale=args.control_scale,
                    return_dict=False,
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                prediction = unet(
                    noisy,
                    timestep,
                    encoder_hidden_states=frame_context,
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=middle,
                ).sample
                element_loss = (prediction.float() - noise.float()).square().mean(dim=(1, 2, 3, 4))
                alpha = noise_scheduler.alphas_cumprod.to("cuda")[timestep].float()
                snr = alpha / (1 - alpha).clamp_min(1e-8)
                weight = torch.minimum(snr, torch.full_like(snr, args.min_snr_gamma)) / snr
                loss = (element_loss * weight).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite training loss at step {step}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in trainable], 1.0
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            optimizer.step()
            current = _trainable_state(
                unet,
                target_profile=args.target_profile,
                adapter_name=adapter_name,
                get_peft_model_state_dict=get_peft_model_state_dict,
                clone=False,
            )
            with torch.no_grad():
                for key, value in current.items():
                    ema[key].lerp_(value.detach(), 1 - args.ema_decay)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                record = {
                    "gradient_norm": float(gradient_norm),
                    "identity_id": row["identity_id"],
                    "loss": float(loss.detach()),
                    "raw_mse": float(element_loss.detach()),
                    "sequence_id": row["sequence_id"],
                    "step": step,
                    "timestep": int(timestep.item()),
                    "verb": row["verb"],
                }
                history.write(json.dumps(record, sort_keys=True) + "\n")
                history.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                _save_checkpoint(
                    output / f"training-step-{step:07d}.pt",
                    step=step,
                    config=config,
                    inputs=inputs["contract"],
                    identity=args.identity,
                    raw_state=current,
                    ema_state=ema,
                    optimizer=optimizer,
                    sampler=sampler,
                    noise_generator=noise_generator,
                    torch_module=torch,
                )
    final_checkpoint = output / f"training-step-{args.steps:07d}.pt"
    report = {
        "artifact_kind": "mugen_animatediff_sparsectrl_temporal_adapter_training_report",
        "claim": (
            "pretrained latent reference-to-motion adaptation; quality must be established "
            "by held-out evaluation"
        ),
        "config": config,
        "counts": {
            "identities": len({row["identity_id"] for row in rows}),
            "sequences": len(rows),
            "trainable_parameters": sum(parameter.numel() for _, parameter in trainable),
            "verbs": dict(sorted(Counter(row["verb"] for row in rows).items())),
        },
        "final_checkpoint": {
            "file_sha256": _file_sha256(final_checkpoint),
            "path": final_checkpoint.name,
        },
        "inputs": inputs["contract"],
        "model": {
            "base_source_index_sha256": BASE_SOURCE_INDEX_SHA256,
            "control_source_index_sha256": CONTROL_SOURCE_INDEX_SHA256,
            "motion_source_index_sha256": MOTION_SOURCE_INDEX_SHA256,
            "still_lora_checkpoint_sha256": STILL_CHECKPOINT_SHA256,
        },
        "schema_version": 1,
        "selection": {"identity_id": args.identity, "split": "train"},
    }
    report_payload = _canonical_json(report)
    (output / "training-report.json").write_bytes(report_payload)
    print(
        json.dumps(
            {
                "checkpoint_sha256": report["final_checkpoint"]["file_sha256"],
                "output": str(output),
                "report_sha256": hashlib.sha256(report_payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--identity")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--target-profile", choices=("temporal_lora", "full_motion"), default="temporal_lora"
    )
    parser.add_argument("--precision", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--control-scale", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()
    for name in ("steps", "rank", "log_every", "checkpoint_every"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "min_snr_gamma", "control_scale"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be in [0,1)")
    if args.target_profile == "full_motion" and args.precision != "bfloat16":
        parser.error("full_motion requires --precision bfloat16")
    if args.target_profile == "full_motion" and args.checkpoint_every < args.steps:
        parser.error("full_motion only supports a final checkpoint to protect the disk floor")
    return args


def _load_inputs(identity: str | None) -> dict[str, object]:
    files = [CANONICAL, TARGET_CACHE, REFERENCE_CACHE, TEXT_PLAN, TEXT_CACHE]
    payloads = {path: path.read_bytes() for path in files}
    canonical = json.loads(payloads[CANONICAL])
    target = json.loads(payloads[TARGET_CACHE])
    reference = json.loads(payloads[REFERENCE_CACHE])
    text_plan = json.loads(payloads[TEXT_PLAN])
    text = json.loads(payloads[TEXT_CACHE])
    if len(canonical.get("records", [])) != canonical.get("counts", {}).get("sequences"):
        raise RuntimeError("canonical MUGEN motion manifest differs")
    if target.get("record_count") != len(target.get("records", [])):
        raise RuntimeError("target latent cache count differs")
    if reference.get("record_count") != len(reference.get("records", [])):
        raise RuntimeError("reference latent cache count differs")
    if (
        text.get("source", {}).get("training_plan_file_sha256")
        != hashlib.sha256(payloads[TEXT_PLAN]).hexdigest()
    ):
        raise RuntimeError("text cache is not bound to the exact text plan")
    target_by_sequence = _unique(target["records"], "sequence_id", "target cache")
    reference_by_sequence = _unique(reference["records"], "sequence_id", "reference cache")
    text_plan_by_sequence = _unique(text_plan["records"], "sequence_id", "text plan")
    text_row_by_prompt = {
        row["prompt"]: row["row_index"] for row in text["rows"] if isinstance(row, dict)
    }
    arrays = text.get("arrays", {})
    embedding_record = arrays.get("embeddings", {})
    embedding_path = TEXT_CACHE.parent / embedding_record.get("path", "")
    if _file_sha256(embedding_path) != embedding_record.get("file_sha256"):
        raise RuntimeError("text embedding array differs")
    embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    rows = []
    for record in canonical["records"]:
        if record.get("split") != "train" or (identity and record.get("identity_id") != identity):
            continue
        sequence_id = record["sequence_id"]
        text_record = text_plan_by_sequence.get(sequence_id)
        if text_record is None or text_record.get("identity_id") != record.get("identity_id"):
            raise RuntimeError(f"text-plan join differs for {sequence_id}")
        if sequence_id not in target_by_sequence:
            raise RuntimeError(f"target latent is absent for {sequence_id}")
        reference_data = record.get("reference", {})
        reference_sequence_id = reference_data.get("sequence_id")
        if reference_sequence_id not in reference_by_sequence:
            raise RuntimeError(f"reference latent is absent for {sequence_id}")
        prompt = text_record.get("prompt")
        if prompt not in text_row_by_prompt:
            raise RuntimeError(f"text embedding is absent for {sequence_id}")
        rows.append(
            {
                "identity_id": record["identity_id"],
                "prompt": prompt,
                "reference_frame_index": int(reference_data["frame_index"]),
                "reference_sequence_id": reference_sequence_id,
                "sequence_id": sequence_id,
                "verb": record["conditioning"]["verb"],
            }
        )
    rows.sort(key=lambda row: row["sequence_id"].encode())
    contract = {
        "canonical_manifest_sha256": hashlib.sha256(payloads[CANONICAL]).hexdigest(),
        "reference_cache_manifest_sha256": hashlib.sha256(payloads[REFERENCE_CACHE]).hexdigest(),
        "target_cache_manifest_sha256": hashlib.sha256(payloads[TARGET_CACHE]).hexdigest(),
        "text_cache_manifest_sha256": hashlib.sha256(payloads[TEXT_CACHE]).hexdigest(),
        "text_plan_sha256": hashlib.sha256(payloads[TEXT_PLAN]).hexdigest(),
    }
    return {
        "contract": contract,
        "embeddings": embeddings,
        "reference_by_sequence": reference_by_sequence,
        "rows": rows,
        "target_by_sequence": target_by_sequence,
        "text_row_by_prompt": text_row_by_prompt,
    }


def _load_latent(root: Path, record: dict[str, object]) -> np.ndarray:
    path = (root / str(record["relative_path"])).resolve()
    if root.resolve() not in path.parents:
        raise RuntimeError("latent path escapes cache")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record.get("file_sha256"):
        raise RuntimeError(f"latent file differs: {path}")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.float16 or value.shape != (8, 4, 64, 64):
        raise RuntimeError(f"latent geometry differs: {path}")
    return value


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    config: dict[str, object],
    inputs: dict[str, object],
    identity: str | None,
    raw_state: dict[str, torch.Tensor],
    ema_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    sampler: random.Random,
    noise_generator: torch.Generator,
    torch_module: object,
) -> None:
    payload = {
        "artifact_kind": "mugen_animatediff_sparsectrl_temporal_adapter_checkpoint",
        "config": config,
        "ema_trainable_state": {key: value.detach().cpu() for key, value in ema_state.items()},
        "inputs": inputs,
        "model": {
            "base_source_index_sha256": BASE_SOURCE_INDEX_SHA256,
            "control_source_index_sha256": CONTROL_SOURCE_INDEX_SHA256,
            "motion_source_index_sha256": MOTION_SOURCE_INDEX_SHA256,
            "still_lora_checkpoint_sha256": STILL_CHECKPOINT_SHA256,
        },
        "optimizer": optimizer.state_dict(),
        "raw_trainable_state": {key: value.detach().cpu() for key, value in raw_state.items()},
        "rng": {
            "cuda": torch_module.cuda.get_rng_state(),
            "noise": noise_generator.get_state(),
            "python_sampler": sampler.getstate(),
            "torch_cpu": torch_module.get_rng_state(),
        },
        "schema_version": 1,
        "selection": {"identity_id": identity, "split": "train"},
        "step": step,
    }
    descriptor, stage_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    stage = Path(stage_name)
    try:
        torch.save(payload, stage)
        with stage.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(stage, path)
    finally:
        stage.unlink(missing_ok=True)


def _trainable_state(
    unet: torch.nn.Module,
    *,
    target_profile: str,
    adapter_name: str | None,
    get_peft_model_state_dict: object,
    clone: bool,
) -> dict[str, torch.Tensor]:
    if target_profile == "temporal_lora":
        state = get_peft_model_state_dict(unet, adapter_name=adapter_name)
        return {
            key: value.detach().clone() if clone else value.detach() for key, value in state.items()
        }
    return {
        name: parameter.detach().clone() if clone else parameter.detach()
        for name, parameter in unet.named_parameters()
        if parameter.requires_grad
    }


def _verify_source_index(root: Path, expected_sha256: str) -> None:
    path = root / "source-index.json"
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError(f"model source index differs: {root}")
    value = json.loads(payload)
    for record in value.get("files", []):
        relative = record.get("relative_path", record.get("path"))
        expected = record.get("file_sha256", record.get("sha256"))
        candidate = root / relative
        if _file_sha256(candidate) != expected:
            raise RuntimeError(f"model file differs: {candidate}")


def _unique(records: list[dict[str, object]], key: str, label: str) -> dict[str, dict[str, object]]:
    output = {}
    for record in records:
        value = record.get(key) if isinstance(record, dict) else None
        if not isinstance(value, str) or value in output:
            raise RuntimeError(f"{label} has an invalid {key}")
        output[value] = record
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


if __name__ == "__main__":
    main()
