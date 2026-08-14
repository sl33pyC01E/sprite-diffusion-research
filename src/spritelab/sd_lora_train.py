"""Resumable SD1.4 UNet LoRA quality control for MUGEN sprite stills."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from spritelab.latent_still_train import (
    LatentStillRow,
    build_hierarchical_sampler_index,
    sample_hierarchical_batch,
)
from spritelab.storage import DiskGuard

Precision = Literal["float32", "bfloat16"]
LoraTargetProfile = Literal["attention", "attention_resnet"]


class SDLoraTrainingError(ValueError):
    """Raised when the pretrained control or its immutable inputs differ."""


@dataclass(frozen=True, slots=True)
class SDLoraTrainingConfig:
    """Small, explicit SD1.4 attention-LoRA training contract."""

    rank: int = 16
    alpha: int = 16
    batch_size: int = 1
    gradient_accumulation: int = 4
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    conditioning_dropout_probability: float = 0.1
    steps: int = 10_000
    log_every: int = 25
    validate_every: int = 500
    checkpoint_every: int = 2_500
    validation_rows: int = 16
    seed: int = 20260819
    device: str = "cuda"
    precision: Precision = "bfloat16"
    target_profile: LoraTargetProfile = "attention"

    def __post_init__(self) -> None:
        for name in (
            "rank",
            "alpha",
            "batch_size",
            "gradient_accumulation",
            "steps",
            "log_every",
            "validate_every",
            "checkpoint_every",
            "validation_rows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int):
            raise ValueError("warmup_steps must be an integer")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be in [0,steps)")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("minimum_learning_rate", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate cannot exceed learning_rate")
        if (
            not math.isfinite(self.conditioning_dropout_probability)
            or not 0 <= self.conditioning_dropout_probability <= 1
        ):
            raise ValueError("conditioning_dropout_probability must be in [0,1]")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if self.target_profile not in {"attention", "attention_resnet"}:
            raise ValueError("target_profile must be attention or attention_resnet")


def sd_lora_target_modules(profile: LoraTargetProfile | str) -> tuple[str, ...]:
    """Return the exact PEFT module suffixes for a quality-control profile."""

    attention = ("to_q", "to_k", "to_v", "to_out.0")
    if profile == "attention":
        return attention
    if profile == "attention_resnet":
        return (*attention, "proj_in", "proj_out", "conv1", "conv2", "conv_shortcut")
    raise ValueError("unknown SD LoRA target profile")


@dataclass(frozen=True, slots=True)
class SDLoraCorpus:
    rows: tuple[LatentStillRow, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    context_embeddings: np.ndarray
    unconditional_embedding: np.ndarray
    contract: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SDLoraTrainingResult:
    output_directory: Path
    report_path: Path
    checkpoint_path: Path
    report_sha256: str


def load_sd_lora_corpus(
    plan_path: Path | str,
    rgb_latent_manifest_path: Path | str,
    text_manifest_path: Path | str,
) -> SDLoraCorpus:
    """Hash-verify and join the SD control cache to exact prompts and splits."""

    plan_file = Path(plan_path).resolve()
    latent_file = Path(rgb_latent_manifest_path).resolve()
    text_file = Path(text_manifest_path).resolve()
    plan_bytes = plan_file.read_bytes()
    latent_bytes = latent_file.read_bytes()
    text_bytes = text_file.read_bytes()
    plan = _json_object(plan_bytes, "training plan")
    latent = _json_object(latent_bytes, "RGB latent manifest")
    text = _json_object(text_bytes, "text manifest")
    if latent.get("artifact_kind") != "mugen_sd14_noncanonical_rgb_vae_latent_cache":
        raise SDLoraTrainingError("RGB latent cache has the wrong artifact kind")
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    cache_plan_exact = latent.get("source", {}).get("plan_file_sha256") == plan_sha256
    if text.get("source", {}).get("training_plan_file_sha256") != plan_sha256:
        raise SDLoraTrainingError("text cache was not built from this plan")
    plan_records = _counted_records(
        plan.get("records"), plan.get("counts", {}).get("sequences"), "training plan"
    )
    latent_records = _counted_records(
        latent.get("records"), latent.get("record_count"), "RGB latent manifest"
    )
    text_rows = _counted_records(text.get("rows"), text.get("prompt_count"), "text manifest")
    latent_by_id = _unique(latent_records, "sequence_id", "RGB latent manifest")
    prompt_rows = _unique(text_rows, "prompt", "text manifest")
    plan_sequence_ids = {record.get("sequence_id") for record in plan_records}
    if not plan_sequence_ids.issubset(latent_by_id):
        raise SDLoraTrainingError("RGB latent cache lacks plan sequences")
    arrays = text.get("arrays")
    if not isinstance(arrays, dict):
        raise SDLoraTrainingError("text array records are missing")
    embeddings = _load_array(
        text_file.parent, arrays, "embeddings", np.float16, (len(text_rows), 77, 768)
    )
    unconditional = _load_array(
        text_file.parent,
        arrays,
        "unconditional_embeddings",
        np.float16,
        (77, 768),
    )
    latent_root = latent_file.parent
    rows = []
    for plan_record in sorted(plan_records, key=lambda row: str(row.get("sequence_id")).encode()):
        sequence_id = _text(plan_record, "sequence_id")
        identity_id = _text(plan_record, "identity_id")
        split = _text(plan_record, "split")
        prompt = _text(plan_record, "prompt")
        conditioning = plan_record.get("conditioning")
        if not isinstance(conditioning, dict):
            raise SDLoraTrainingError(f"conditioning is absent for {sequence_id}")
        latent_record = latent_by_id[sequence_id]
        if latent_record.get("identity_id") != identity_id or latent_record.get("split") != split:
            raise SDLoraTrainingError(f"RGB latent identity/split differs for {sequence_id}")
        target = plan_record.get("target")
        source_target = latent_record.get("source_target")
        if not isinstance(target, dict) or not isinstance(source_target, dict):
            raise SDLoraTrainingError(f"RGB source target is absent for {sequence_id}")
        for key in ("array_content_sha256", "file_sha256", "relative_path"):
            if source_target.get(key) != target.get(key):
                raise SDLoraTrainingError(f"RGB source target differs for {sequence_id} at {key}")
        raw_eligible = target.get("eligible_frame_indices", list(range(8)))
        if (
            not isinstance(raw_eligible, list)
            or not raw_eligible
            or any(
                isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < 8
                for frame in raw_eligible
            )
            or raw_eligible != sorted(set(raw_eligible))
        ):
            raise SDLoraTrainingError(f"eligible frames are invalid for {sequence_id}")
        text_record = prompt_rows.get(prompt)
        if text_record is None or not isinstance(text_record.get("row_index"), int):
            raise SDLoraTrainingError(f"text cache lacks prompt for {sequence_id}")
        relative = _text(latent_record, "relative_path")
        path = (latent_root / relative).resolve()
        row = LatentStillRow(
            sequence_id=sequence_id,
            identity_id=identity_id,
            verb=_text(conditioning, "verb"),
            split=split,
            prompt=prompt,
            prompt_row=text_record["row_index"],
            latent_path=path,
            latent_file_sha256=_text(latent_record, "file_sha256"),
            latent_array_sha256=_text(latent_record, "array_content_sha256"),
            eligible_frame_indices=tuple(raw_eligible),
        )
        _load_rgb_latent(row, verify=True)
        rows.append(row)
    train = tuple(index for index, row in enumerate(rows) if row.split == "train")
    validation = tuple(index for index, row in enumerate(rows) if row.split == "validation")
    if not train or not validation:
        raise SDLoraTrainingError("training and validation rows are required")
    if {rows[index].identity_id for index in train}.intersection(
        rows[index].identity_id for index in validation
    ):
        raise SDLoraTrainingError("training and validation identities overlap")
    contract = {
        "plan_file_sha256": plan_sha256,
        "record_count": len(rows),
        "rgb_cache_binding": (
            "exact_plan" if cache_plan_exact else "hash_verified_source_target_subset"
        ),
        "rgb_latent_manifest_file_sha256": hashlib.sha256(latent_bytes).hexdigest(),
        "text_manifest_file_sha256": hashlib.sha256(text_bytes).hexdigest(),
        "train_identities": len({rows[index].identity_id for index in train}),
        "train_rows": len(train),
        "validation_identities": len({rows[index].identity_id for index in validation}),
        "validation_rows": len(validation),
    }
    contract["canonical_sha256"] = hashlib.sha256(_canonical_json(contract)).hexdigest()
    return SDLoraCorpus(
        rows=tuple(rows),
        train_indices=train,
        validation_indices=validation,
        context_embeddings=embeddings,
        unconditional_embedding=unconditional,
        contract=contract,
    )


def run_sd14_lora_training(
    plan_path: Path | str,
    rgb_latent_manifest_path: Path | str,
    text_manifest_path: Path | str,
    model_directory: Path | str,
    output_directory: Path | str,
    *,
    expected_source_index_sha256: str,
    config: SDLoraTrainingConfig | None = None,
    resume_checkpoint_path: Path | str | None = None,
    expected_resume_sha256: str | None = None,
    stop_after_step: int | None = None,
    disk_guard: DiskGuard | None = None,
) -> SDLoraTrainingResult:
    """Fine-tune only SD1.4 attention LoRA weights on the matched RGB control."""

    try:
        import diffusers
        import peft
        import torch
        from diffusers import DDPMScheduler, PNDMScheduler, UNet2DConditionModel
        from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("SD1.4 LoRA training requires Torch, Diffusers, and PEFT") from error
    experiment = config or SDLoraTrainingConfig()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace SD LoRA output: {output}")
    if (resume_checkpoint_path is None) != (expected_resume_sha256 is None):
        raise ValueError("resume checkpoint and expected SHA-256 must be supplied together")
    if stop_after_step is not None and (
        isinstance(stop_after_step, bool)
        or not isinstance(stop_after_step, int)
        or not 0 < stop_after_step <= experiment.steps
    ):
        raise ValueError("stop_after_step must be in (0, config.steps]")
    device = torch.device(experiment.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if experiment.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 SD LoRA training requires CUDA")
    model_root = Path(model_directory).resolve()
    source_index_path = model_root / "source-index.json"
    if _file_sha256(source_index_path) != expected_source_index_sha256:
        raise SDLoraTrainingError("Stable Diffusion source-index SHA-256 mismatch")
    source_index = _json_object(source_index_path.read_bytes(), "Stable Diffusion source index")
    _verify_model_files(model_root, source_index)
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(1024**3, label="SD1.4 LoRA training")
    corpus = load_sd_lora_corpus(plan_path, rgb_latent_manifest_path, text_manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(experiment.seed)
    torch.cuda.manual_seed_all(experiment.seed) if device.type == "cuda" else None
    unet = UNet2DConditionModel.from_pretrained(
        model_root / "unet", local_files_only=True, use_safetensors=True
    ).to(device)
    unet.requires_grad_(False)
    unet.enable_gradient_checkpointing()
    adapter_name = "mugen"
    target_modules = sd_lora_target_modules(experiment.target_profile)
    lora_config = LoraConfig(
        r=experiment.rank,
        lora_alpha=experiment.alpha,
        init_lora_weights="gaussian",
        target_modules=list(target_modules),
    )
    unet.add_adapter(lora_config, adapter_name=adapter_name)
    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    if not trainable:
        raise SDLoraTrainingError("PEFT attached no trainable LoRA parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=experiment.learning_rate,
        weight_decay=experiment.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler_config = PNDMScheduler.load_config(model_root / "scheduler")
    noise_scheduler = DDPMScheduler.from_config(scheduler_config)
    sampler_generator = torch.Generator(device="cpu").manual_seed(experiment.seed + 1)
    noise_generator = torch.Generator(device=device).manual_seed(experiment.seed + 2)
    dropout_generator = torch.Generator(device=device).manual_seed(experiment.seed + 3)
    ema = {
        key: value.detach().clone()
        for key, value in get_peft_model_state_dict(unet, adapter_name=adapter_name).items()
    }
    start_step = 0
    lineage = None
    if resume_checkpoint_path is not None:
        assert expected_resume_sha256 is not None
        parent = _load_resume(
            torch,
            Path(resume_checkpoint_path).resolve(),
            expected_sha256=expected_resume_sha256,
            corpus=corpus,
            config=experiment,
            source_index_sha256=expected_source_index_sha256,
        )
        start_step = int(parent["step"])
        set_peft_model_state_dict(unet, parent["raw_lora"], adapter_name=adapter_name)
        ema = {key: value.to(device) for key, value in parent["ema_lora"].items()}
        optimizer.load_state_dict(parent["optimizer"])
        sampler_generator.set_state(parent["rng_state"]["sampler"])
        noise_generator.set_state(parent["rng_state"]["noise"])
        dropout_generator.set_state(parent["rng_state"]["dropout"])
        torch.set_rng_state(parent["rng_state"]["torch_cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(parent["rng_state"]["cuda"], device=device)
        lineage = {
            "parent_checkpoint_path": str(resume_checkpoint_path),
            "parent_checkpoint_sha256": expected_resume_sha256,
            "parent_step": start_step,
        }
    final_step = experiment.steps if stop_after_step is None else stop_after_step
    if final_step <= start_step:
        raise SDLoraTrainingError("stop_after_step must exceed the resume step")
    index = build_hierarchical_sampler_index(corpus.rows, corpus.train_indices)
    eligible_frames = tuple(row.eligible_frame_indices for row in corpus.rows)
    validation = _validation_selection(corpus, experiment.validation_rows)
    dtype = torch.bfloat16 if experiment.precision == "bfloat16" else torch.float32
    autocast = experiment.precision == "bfloat16"
    history_path = output / "training-history.jsonl"
    unet.train()
    with history_path.open("x", encoding="utf-8", newline="\n") as history:
        for step_index in range(start_step, final_step):
            step = step_index + 1
            learning_rate = _learning_rate(step, experiment)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0.0
            dropped = 0.0
            for _ in range(experiment.gradient_accumulation):
                selection = sample_hierarchical_batch(
                    index,
                    batch_size=experiment.batch_size,
                    generator=sampler_generator,
                    frame_indices_by_row=eligible_frames,
                )
                clean, context = _batch(torch, corpus, selection, device=device)
                noise = torch.randn(
                    clean.shape, device=device, dtype=clean.dtype, generator=noise_generator
                )
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (clean.shape[0],),
                    device=device,
                    generator=noise_generator,
                ).long()
                noisy = noise_scheduler.add_noise(clean, noise, timesteps)
                dropout_rows = (
                    torch.rand((clean.shape[0],), device=device, generator=dropout_generator)
                    < experiment.conditioning_dropout_probability
                )
                unconditional = torch.from_numpy(
                    np.asarray(corpus.unconditional_embedding, dtype=np.float32)
                ).to(device)
                context = torch.where(dropout_rows[:, None, None], unconditional[None], context)
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast):
                    prediction = unet(noisy, timesteps, encoder_hidden_states=context).sample
                    loss = torch.nn.functional.mse_loss(prediction.float(), noise.float())
                    scaled_loss = loss / experiment.gradient_accumulation
                if not bool(torch.isfinite(scaled_loss)):
                    raise RuntimeError(f"non-finite SD LoRA loss at step {step}")
                scaled_loss.backward()
                accumulated += float(loss.detach().cpu())
                dropped += float(dropout_rows.float().mean().cpu())
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, experiment.gradient_clip_norm)
                .detach()
                .cpu()
            )
            optimizer.step()
            current_state = get_peft_model_state_dict(unet, adapter_name=adapter_name)
            with torch.no_grad():
                for key, value in current_state.items():
                    ema[key].lerp_(value.detach(), 1 - 0.999)
            validation_loss = None
            if step == 1 or step % experiment.validate_every == 0 or step == final_step:
                validation_loss = _validate(
                    torch,
                    corpus,
                    validation,
                    unet,
                    noise_scheduler,
                    device=device,
                    dtype=dtype,
                    autocast=autocast,
                    seed=experiment.seed + 20_000,
                )
            if step == 1 or step % experiment.log_every == 0 or step == final_step:
                history.write(
                    json.dumps(
                        {
                            "conditioning_dropout_fraction": dropped
                            / experiment.gradient_accumulation,
                            "gradient_norm_before_clip": gradient_norm,
                            "learning_rate": learning_rate,
                            "loss": accumulated / experiment.gradient_accumulation,
                            "step": step,
                            "validation_epsilon_mse": validation_loss,
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
                history.flush()
                os.fsync(history.fileno())
            if step % experiment.checkpoint_every == 0 or step == final_step:
                _write_checkpoint(
                    torch,
                    output / f"training-step-{step:07d}.pt",
                    corpus=corpus,
                    config=experiment,
                    source_index_sha256=expected_source_index_sha256,
                    step=step,
                    raw_lora=current_state,
                    ema_lora=ema,
                    optimizer=optimizer,
                    sampler_generator=sampler_generator,
                    noise_generator=noise_generator,
                    dropout_generator=dropout_generator,
                    device=device,
                    disk_guard=guard,
                )
    final_checkpoint = output / f"training-step-{final_step:07d}.pt"
    report = {
        "artifact_kind": "mugen_sd14_attention_lora_rgb_quality_control",
        "claim": "pretrained RGB control only; not canonical RGBA output",
        "config": asdict(experiment),
        "corpus": corpus.contract,
        "history": {"file_sha256": _file_sha256(history_path), "path": history_path.name},
        "lineage": lineage,
        "lora": {
            "adapter_name": adapter_name,
            "target_modules": list(target_modules),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        },
        "runtime": {
            "device": str(device),
            "diffusers": diffusers.__version__,
            "peft": peft.__version__,
            "torch": str(torch.__version__),
        },
        "source_index_file_sha256": expected_source_index_sha256,
        "step": final_step,
        "training_extent": {
            "is_partial_cumulative_run": final_step < experiment.steps,
            "planned_cumulative_steps": experiment.steps,
            "published_cumulative_step": final_step,
        },
        "training_checkpoint": {
            "file_sha256": _file_sha256(final_checkpoint),
            "path": final_checkpoint.name,
        },
    }
    report_path = output / "training-report.json"
    payload = _canonical_json(report)
    _atomic_bytes(report_path, payload, disk_guard=guard)
    return SDLoraTrainingResult(
        output_directory=output,
        report_path=report_path,
        checkpoint_path=final_checkpoint,
        report_sha256=hashlib.sha256(payload).hexdigest(),
    )


def _batch(
    runtime: Any,
    corpus: SDLoraCorpus,
    selection: tuple[tuple[int, int], ...],
    *,
    device: Any,
) -> tuple[Any, Any]:
    latents = []
    prompt_rows = []
    for row_index, frame_index in selection:
        row = corpus.rows[row_index]
        latents.append(_load_rgb_latent(row, verify=False)[frame_index])
        prompt_rows.append(row.prompt_row)
    clean = runtime.from_numpy(np.ascontiguousarray(np.stack(latents), dtype=np.float32)).to(device)
    context = runtime.from_numpy(
        np.asarray(corpus.context_embeddings[prompt_rows], dtype=np.float32)
    ).to(device)
    return clean, context


def _validation_selection(corpus: SDLoraCorpus, maximum_rows: int) -> tuple[tuple[int, int], ...]:
    by_identity = {}
    for index in corpus.validation_indices:
        by_identity.setdefault(corpus.rows[index].identity_id, index)
    return tuple(
        (
            by_identity[identity],
            corpus.rows[by_identity[identity]].eligible_frame_indices[0],
        )
        for identity in sorted(by_identity, key=str.encode)[:maximum_rows]
    )


def _validate(
    runtime: Any,
    corpus: SDLoraCorpus,
    selection: tuple[tuple[int, int], ...],
    unet: Any,
    scheduler: Any,
    *,
    device: Any,
    dtype: Any,
    autocast: bool,
    seed: int,
) -> float:
    clean, context = _batch(runtime, corpus, selection, device=device)
    generator = runtime.Generator(device=device).manual_seed(seed)
    noise = runtime.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
    timesteps = runtime.full(
        (clean.shape[0],), scheduler.config.num_train_timesteps // 2, device=device
    ).long()
    noisy = scheduler.add_noise(clean, noise, timesteps)
    unet.eval()
    with (
        runtime.no_grad(),
        runtime.autocast(device_type=device.type, dtype=dtype, enabled=autocast),
    ):
        prediction = unet(noisy, timesteps, encoder_hidden_states=context).sample
    unet.train()
    return float(runtime.nn.functional.mse_loss(prediction.float(), noise.float()).cpu())


def _write_checkpoint(
    runtime: Any,
    path: Path,
    *,
    corpus: SDLoraCorpus,
    config: SDLoraTrainingConfig,
    source_index_sha256: str,
    step: int,
    raw_lora: dict[str, Any],
    ema_lora: dict[str, Any],
    optimizer: Any,
    sampler_generator: Any,
    noise_generator: Any,
    dropout_generator: Any,
    device: Any,
    disk_guard: DiskGuard,
) -> None:
    _atomic_torch_save(
        runtime,
        path,
        {
            "artifact_kind": "mugen_sd14_attention_lora_resume_checkpoint",
            "config": asdict(config),
            "corpus": corpus.contract,
            "ema_lora": {key: value.detach().cpu() for key, value in ema_lora.items()},
            "optimizer": optimizer.state_dict(),
            "raw_lora": {key: value.detach().cpu() for key, value in raw_lora.items()},
            "rng_state": {
                "cuda": runtime.cuda.get_rng_state(device) if device.type == "cuda" else None,
                "dropout": dropout_generator.get_state(),
                "noise": noise_generator.get_state(),
                "sampler": sampler_generator.get_state(),
                "torch_cpu": runtime.get_rng_state(),
            },
            "source_index_file_sha256": source_index_sha256,
            "step": step,
        },
        disk_guard=disk_guard,
    )


def _load_resume(
    runtime: Any,
    path: Path,
    *,
    expected_sha256: str,
    corpus: SDLoraCorpus,
    config: SDLoraTrainingConfig,
    source_index_sha256: str,
) -> dict[str, Any]:
    if _file_sha256(path) != expected_sha256:
        raise SDLoraTrainingError("resume checkpoint SHA-256 mismatch")
    value = runtime.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("artifact_kind") != (
        "mugen_sd14_attention_lora_resume_checkpoint"
    ):
        raise SDLoraTrainingError("resume checkpoint has the wrong artifact kind")
    if value.get("corpus") != corpus.contract:
        raise SDLoraTrainingError("resume corpus differs")
    if value.get("source_index_file_sha256") != source_index_sha256:
        raise SDLoraTrainingError("resume Stable Diffusion source differs")
    parent_config = value.get("config")
    if not isinstance(parent_config, dict):
        raise SDLoraTrainingError("resume config is missing")
    current = asdict(config)
    for key, parent_value in parent_config.items():
        if key != "steps" and current.get(key) != parent_value:
            raise SDLoraTrainingError(f"resume config differs at {key!r}")
    return value


def _load_rgb_latent(row: LatentStillRow, *, verify: bool) -> np.ndarray:
    payload = row.latent_path.read_bytes()
    if verify and hashlib.sha256(payload).hexdigest() != row.latent_file_sha256:
        raise SDLoraTrainingError(f"RGB latent file hash mismatch: {row.sequence_id}")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.float16 or value.shape != (8, 4, 64, 64):
        raise SDLoraTrainingError(f"RGB latent geometry differs: {row.sequence_id}")
    if verify and _array_sha256(value) != row.latent_array_sha256:
        raise SDLoraTrainingError(f"RGB latent array hash mismatch: {row.sequence_id}")
    if not bool(np.isfinite(value).all()):
        raise SDLoraTrainingError(f"RGB latent is non-finite: {row.sequence_id}")
    return value


def _load_array(
    root: Path,
    arrays: dict[str, Any],
    name: str,
    dtype: Any,
    shape: tuple[int, ...],
) -> np.ndarray:
    record = arrays.get(name)
    if not isinstance(record, dict):
        raise SDLoraTrainingError(f"text array record is missing: {name}")
    path = (root / _text(record, "path")).resolve()
    if _file_sha256(path) != record.get("file_sha256"):
        raise SDLoraTrainingError(f"text array file differs: {name}")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if (
        value.dtype != dtype
        or value.shape != shape
        or _array_sha256(value) != record.get("array_content_sha256")
    ):
        raise SDLoraTrainingError(f"text array content differs: {name}")
    return value


def _verify_model_files(root: Path, source_index: dict[str, Any]) -> None:
    files = source_index.get("files")
    if not isinstance(files, list) or not files:
        raise SDLoraTrainingError("Stable Diffusion source index files are invalid")
    for record in files:
        relative = record.get("relative_path") if isinstance(record, dict) else None
        expected = record.get("file_sha256") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise SDLoraTrainingError("Stable Diffusion source file record is invalid")
        path = (root / relative).resolve()
        if root not in path.parents or _file_sha256(path) != expected:
            raise SDLoraTrainingError(f"Stable Diffusion source file differs: {relative}")


def _learning_rate(step: int, config: SDLoraTrainingConfig) -> float:
    if step <= config.warmup_steps and config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    cosine = 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
    return (
        config.minimum_learning_rate
        + (config.learning_rate - config.minimum_learning_rate) * cosine
    )


def _counted_records(value: Any, count: Any, label: str) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or count != len(value)
        or not all(isinstance(row, dict) for row in value)
    ):
        raise SDLoraTrainingError(f"{label} record count differs")
    return value


def _unique(records: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        value = _text(record, key)
        if value in output:
            raise SDLoraTrainingError(f"{label} has duplicate {key}: {value}")
        output[value] = record
    return output


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise SDLoraTrainingError(f"field {key} must be non-empty text")
    return result


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SDLoraTrainingError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise SDLoraTrainingError(f"{label} must be an object")
    return value


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes, *, disk_guard: DiskGuard) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace artifact: {path}")
    disk_guard.require_capacity(len(payload), label=path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_torch_save(
    runtime: Any,
    path: Path,
    payload: dict[str, Any],
    *,
    disk_guard: DiskGuard,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace checkpoint: {path}")
    disk_guard.require_capacity(1024**3, label=path.name)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        runtime.save(payload, temporary)
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
