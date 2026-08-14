"""Hash-bound text-to-RGB inference for the SD1.4 MUGEN LoRA control."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from spritelab.sd_lora_train import sd_lora_target_modules
from spritelab.storage import DiskGuard


class SDLoraInferenceError(ValueError):
    """Raised when base weights, LoRA evidence, prompt, or output differs."""


@dataclass(frozen=True, slots=True)
class SDLoraInferenceConfig:
    seed: int = 20260821
    sample_steps: int = 50
    guidance_scale: float = 5.0
    device: str = "cuda"
    precision: str = "bfloat16"
    weights_variant: Literal["ema", "raw"] = "ema"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(self.sample_steps, bool)
            or not isinstance(self.sample_steps, int)
            or self.sample_steps <= 0
        ):
            raise ValueError("sample_steps must be a positive integer")
        if not np.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and non-negative")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("precision must be float32 or bfloat16")
        if self.weights_variant not in {"ema", "raw"}:
            raise ValueError("weights_variant must be ema or raw")


def run_sd14_lora_inference(
    checkpoint_path: Path | str,
    model_directory: Path | str,
    prompts: list[str],
    output_directory: Path | str,
    *,
    expected_checkpoint_sha256: str,
    expected_source_index_sha256: str,
    config: SDLoraInferenceConfig | None = None,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Generate full 512px RGB controls plus honest 128px display derivatives."""

    try:
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler, PNDMScheduler, UNet2DConditionModel
        from peft import LoraConfig, set_peft_model_state_dict
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "SD LoRA inference requires Torch, Diffusers, PEFT, Transformers"
        ) from error
    experiment = config or SDLoraInferenceConfig()
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain unique non-empty strings")
    if len(set(prompts)) != len(prompts):
        raise ValueError("prompts must be unique")
    checkpoint_file = Path(checkpoint_path).resolve()
    model_root = Path(model_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace SD LoRA inference: {output}")
    if _file_sha256(checkpoint_file) != expected_checkpoint_sha256:
        raise SDLoraInferenceError("LoRA checkpoint SHA-256 mismatch")
    if _file_sha256(model_root / "source-index.json") != expected_source_index_sha256:
        raise SDLoraInferenceError("Stable Diffusion source-index SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("artifact_kind") != (
        "mugen_sd14_attention_lora_resume_checkpoint"
    ):
        raise SDLoraInferenceError("LoRA checkpoint has the wrong artifact kind")
    if checkpoint.get("source_index_file_sha256") != expected_source_index_sha256:
        raise SDLoraInferenceError("LoRA checkpoint was trained from another base model")
    training_config = checkpoint.get("config")
    if not isinstance(training_config, dict):
        raise SDLoraInferenceError("LoRA training config is absent")
    device = torch.device(experiment.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but unavailable")
    if experiment.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 inference requires CUDA")
    unet = (
        UNet2DConditionModel.from_pretrained(
            model_root / "unet", local_files_only=True, use_safetensors=True
        )
        .to(device)
        .eval()
    )
    unet.requires_grad_(False)
    adapter_name = "mugen"
    unet.add_adapter(
        LoraConfig(
            r=int(training_config["rank"]),
            lora_alpha=int(training_config["alpha"]),
            target_modules=list(
                sd_lora_target_modules(training_config.get("target_profile", "attention"))
            ),
        ),
        adapter_name=adapter_name,
    )
    state_key = f"{experiment.weights_variant}_lora"
    lora_state = checkpoint.get(state_key)
    if not isinstance(lora_state, dict) or not lora_state:
        raise SDLoraInferenceError(f"checkpoint {state_key} is absent")
    set_peft_model_state_dict(unet, lora_state, adapter_name=adapter_name)
    tokenizer = CLIPTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    overlong = [
        (prompt, len(tokenizer.encode(prompt, add_special_tokens=True)))
        for prompt in prompts
        if len(tokenizer.encode(prompt, add_special_tokens=True)) > 77
    ]
    if overlong:
        raise SDLoraInferenceError(f"prompt exceeds 77 CLIP tokens: {overlong[0]!r}")
    text_encoder = (
        CLIPTextModel.from_pretrained(
            model_root / "text_encoder", local_files_only=True, use_safetensors=True
        )
        .to(device)
        .eval()
    )
    tokens = tokenizer(
        ["", *prompts],
        padding="max_length",
        truncation=False,
        max_length=77,
        return_tensors="pt",
    )
    with torch.no_grad():
        hidden = text_encoder(
            tokens.input_ids.to(device), attention_mask=tokens.attention_mask.to(device)
        ).last_hidden_state
    unconditional = hidden[0:1].expand(len(prompts), -1, -1)
    conditioned = hidden[1:]
    del text_encoder
    scheduler = DDIMScheduler.from_config(PNDMScheduler.load_config(model_root / "scheduler"))
    scheduler.set_timesteps(experiment.sample_steps, device=device)
    generator = torch.Generator(device=device).manual_seed(experiment.seed)
    latent = (
        torch.randn((len(prompts), 4, 64, 64), device=device, generator=generator)
        * scheduler.init_noise_sigma
    )
    noise_sha256 = _array_sha256(np.ascontiguousarray(latent.float().cpu().numpy()))
    dtype = torch.bfloat16 if experiment.precision == "bfloat16" else torch.float32
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            model_input = scheduler.scale_model_input(latent, timestep)
            combined_latent = torch.cat((model_input, model_input))
            combined_context = torch.cat((unconditional, conditioned))
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=experiment.precision == "bfloat16",
            ):
                prediction = unet(
                    combined_latent,
                    timestep,
                    encoder_hidden_states=combined_context,
                ).sample
            unconditioned_prediction, conditioned_prediction = prediction.chunk(2)
            guided = unconditioned_prediction + experiment.guidance_scale * (
                conditioned_prediction - unconditioned_prediction
            )
            latent = scheduler.step(guided, timestep, latent).prev_sample
    vae = (
        AutoencoderKL.from_pretrained(
            model_root / "vae", local_files_only=True, use_safetensors=True
        )
        .to(device)
        .eval()
    )
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=experiment.precision == "bfloat16",
        ),
    ):
        decoded = vae.decode(latent / float(vae.config.scaling_factor)).sample
    rgb = decoded.float().add(1).mul(127.5).clamp(0, 255).round().to(torch.uint8)
    arrays = rgb.cpu().numpy().transpose(0, 2, 3, 1)
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(256 * 1024**2, label="SD LoRA inference")
    output.mkdir(parents=True, exist_ok=False)
    samples = []
    for index, (prompt, value) in enumerate(zip(prompts, arrays, strict=True)):
        stem = f"{index:03d}-{hashlib.sha256(prompt.encode()).hexdigest()[:10]}"
        raw_path = output / f"{stem}-512.png"
        reduced_path = output / f"{stem}-128.png"
        preview_path = output / f"{stem}-preview.png"
        image = Image.fromarray(value, mode="RGB")
        image.save(raw_path, format="PNG", optimize=False)
        reduced = image.resize((128, 128), resample=Image.Resampling.BOX)
        reduced.save(reduced_path, format="PNG", optimize=False)
        reduced.resize((512, 512), resample=Image.Resampling.NEAREST).save(
            preview_path, format="PNG", optimize=False
        )
        samples.append(
            {
                "downsample_128": {
                    "file_sha256": _file_sha256(reduced_path),
                    "method": "4x_box_display_derivative",
                    "path": reduced_path.name,
                },
                "full_512": {
                    "file_sha256": _file_sha256(raw_path),
                    "path": raw_path.name,
                },
                "nearest_preview": {
                    "file_sha256": _file_sha256(preview_path),
                    "path": preview_path.name,
                },
                "prompt": prompt,
            }
        )
    report = {
        "artifact_kind": "mugen_sd14_attention_lora_rgb_inference",
        "checkpoint": {
            "file_sha256": expected_checkpoint_sha256,
            "path": str(checkpoint_file),
            "step": checkpoint.get("step"),
        },
        "claim": "noncanonical RGB pretrained quality control; no alpha output",
        "config": asdict(experiment),
        "noise_batch_sha256": noise_sha256,
        "samples": samples,
        "source_index_file_sha256": expected_source_index_sha256,
    }
    report_path = output / "inference-report.json"
    payload = _canonical_json(report)
    _atomic_bytes(report_path, payload)
    return report_path, hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace artifact: {path}")
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
