"""Hash-bound text-to-RGBA inference for the scratch MUGEN latent still DiT."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.models.latent_still_dit import LatentStillDiT, LatentStillDiTConfig
from spritelab.models.sprite_autoencoder import SpriteAutoencoderConfig, SpriteRGBAAutoencoder
from spritelab.storage import DiskGuard


class LatentStillInferenceError(ValueError):
    """Raised when inference evidence, prompt geometry, or output differs."""


@dataclass(frozen=True, slots=True)
class LatentStillInferenceConfig:
    seed: int = 20260820
    sample_steps: int = 32
    guidance_scale: float = 3.5
    device: str = "cuda"
    precision: str = "bfloat16"

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


def heun_sample_latent_still(
    model: Any,
    noise: Any,
    context: Any,
    context_mask: Any,
    *,
    steps: int,
    guidance_scale: float,
    autocast_context: Any,
) -> Any:
    """Integrate noise t=1 to data t=0 with classifier-free Heun updates."""

    import torch

    if noise.ndim != 4:
        raise ValueError("noise must have shape [B,C,H,W]")
    if context.ndim != 3 or context.shape[0] != noise.shape[0]:
        raise ValueError("context must have matching [B,L,D] geometry")
    if context_mask.shape != context.shape[:2]:
        raise ValueError("context_mask must match context [B,L]")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    state = noise
    times = torch.linspace(1, 0, steps + 1, device=noise.device, dtype=noise.dtype)

    def velocity(value: Any, timestep: Any) -> Any:
        time_batch = timestep.expand(value.shape[0])
        with autocast_context():
            conditioned = model(value, time_batch, context, context_mask=context_mask)
            unconditional = model(value, time_batch)
        return unconditional + guidance_scale * (conditioned - unconditional)

    with torch.no_grad():
        for index in range(steps):
            current = times[index]
            following = times[index + 1]
            delta = following - current
            first = velocity(state, current)
            proposal = state + delta * first
            if index == steps - 1:
                state = proposal
            else:
                second = velocity(proposal, following)
                state = state + delta * (first + second) * 0.5
    return state


def run_latent_still_inference(
    checkpoint_path: Path | str,
    codec_checkpoint_path: Path | str,
    text_model_directory: Path | str,
    prompts: list[str],
    output_directory: Path | str,
    *,
    expected_checkpoint_sha256: str,
    expected_codec_checkpoint_sha256: str,
    expected_text_source_index_sha256: str,
    config: LatentStillInferenceConfig | None = None,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Generate raw native RGBA arrays and nearest-neighbor previews."""

    try:
        import torch
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("latent still inference requires Torch and Transformers") from error
    experiment = config or LatentStillInferenceConfig()
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain non-empty strings")
    if len(set(prompts)) != len(prompts):
        raise ValueError("prompts must be unique")
    checkpoint_file = Path(checkpoint_path).resolve()
    codec_file = Path(codec_checkpoint_path).resolve()
    model_root = Path(text_model_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace inference output: {output}")
    for expected, actual, label in (
        (expected_checkpoint_sha256, _file_sha256(checkpoint_file), "DiT checkpoint"),
        (
            expected_codec_checkpoint_sha256,
            _file_sha256(codec_file),
            "codec checkpoint",
        ),
        (
            expected_text_source_index_sha256,
            _file_sha256(model_root / "source-index.json"),
            "text source index",
        ),
    ):
        if expected != actual:
            raise LatentStillInferenceError(f"{label} SHA-256 mismatch")
    device = torch.device(experiment.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if experiment.precision == "bfloat16" and device.type != "cuda":
        raise ValueError("bfloat16 inference requires CUDA")
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("artifact_kind") != (
        "mugen_latent_still_dit_ema_inference_checkpoint"
    ):
        raise LatentStillInferenceError("DiT checkpoint has the wrong artifact kind")
    config_record = checkpoint.get("config")
    normalization = checkpoint.get("normalization")
    if not isinstance(config_record, dict) or not isinstance(normalization, dict):
        raise LatentStillInferenceError("DiT config/normalization is absent")
    model_record = config_record.get("model")
    if not isinstance(model_record, dict):
        raise LatentStillInferenceError("DiT model config is absent")
    model = LatentStillDiT(LatentStillDiTConfig(**model_record)).to(device).eval()
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    codec_checkpoint = torch.load(codec_file, map_location="cpu", weights_only=True)
    codec_config_record = codec_checkpoint.get("config", {}).get("architecture")
    if not isinstance(codec_config_record, dict) or not isinstance(
        codec_checkpoint.get("ema"), dict
    ):
        raise LatentStillInferenceError("codec checkpoint config/EMA is absent")
    codec_values = dict(codec_config_record)
    if isinstance(codec_values.get("channel_multipliers"), list):
        codec_values["channel_multipliers"] = tuple(codec_values["channel_multipliers"])
    codec = SpriteRGBAAutoencoder(SpriteAutoencoderConfig(**codec_values)).to(device).eval()
    codec.load_state_dict(codec_checkpoint["ema"], strict=True)
    tokenizer = CLIPTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    overlong = [
        (prompt, len(tokenizer.encode(prompt, add_special_tokens=True)))
        for prompt in prompts
        if len(tokenizer.encode(prompt, add_special_tokens=True)) > 77
    ]
    if overlong:
        raise LatentStillInferenceError(f"prompt exceeds 77 CLIP tokens: {overlong[0]!r}")
    text_encoder = (
        CLIPTextModel.from_pretrained(
            model_root / "text_encoder", local_files_only=True, use_safetensors=True
        )
        .to(device)
        .eval()
    )
    tokens = tokenizer(
        prompts,
        padding="max_length",
        truncation=False,
        max_length=77,
        return_tensors="pt",
    )
    with torch.no_grad():
        context = text_encoder(
            tokens.input_ids.to(device),
            attention_mask=tokens.attention_mask.to(device),
        ).last_hidden_state.float()
    context_mask = tokens.attention_mask.to(device=device, dtype=torch.bool)
    del text_encoder
    generator = torch.Generator(device=device).manual_seed(experiment.seed)
    noise = torch.randn((len(prompts), 8, 64, 64), device=device, generator=generator)
    noise_sha256 = _tensor_sha256(noise)
    dtype = torch.bfloat16 if experiment.precision == "bfloat16" else torch.float32

    def autocast_context() -> Any:
        return torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=experiment.precision == "bfloat16",
        )

    normalized = heun_sample_latent_still(
        model,
        noise,
        context,
        context_mask,
        steps=experiment.sample_steps,
        guidance_scale=experiment.guidance_scale,
        autocast_context=autocast_context,
    )
    mean = torch.tensor(normalization.get("channel_mean"), device=device, dtype=torch.float32)[
        None, :, None, None
    ]
    std = torch.tensor(
        normalization.get("channel_standard_deviation"),
        device=device,
        dtype=torch.float32,
    )[None, :, None, None]
    if mean.shape != std.shape or mean.shape != (1, 8, 1, 1):
        raise LatentStillInferenceError("latent normalization geometry differs")
    latent = normalized.float() * std + mean
    with torch.no_grad():
        decoded = codec.decode(latent).clamp(0, 1).mul(255).round().to(torch.uint8)
    rgba = decoded.cpu().numpy().transpose(0, 2, 3, 1)
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(256 * 1024**2, label="latent still inference")
    output.mkdir(parents=True, exist_ok=False)
    sample_records = []
    for index, (prompt, value) in enumerate(zip(prompts, rgba, strict=True)):
        stem = f"{index:03d}-{hashlib.sha256(prompt.encode()).hexdigest()[:10]}"
        array_path = output / f"{stem}.npy"
        array_payload = _npy_bytes(np.ascontiguousarray(value))
        array_path.write_bytes(array_payload)
        native_path = output / f"{stem}-native.png"
        preview_path = output / f"{stem}-preview.png"
        image = Image.fromarray(value, mode="RGBA")
        image.save(native_path, format="PNG", optimize=False)
        image.resize((512, 512), resample=Image.Resampling.NEAREST).save(
            preview_path, format="PNG", optimize=False
        )
        sample_records.append(
            {
                "array": {
                    "array_content_sha256": _array_sha256(value),
                    "file_sha256": hashlib.sha256(array_payload).hexdigest(),
                    "path": array_path.name,
                    "shape": list(value.shape),
                },
                "native_png": {
                    "file_sha256": _file_sha256(native_path),
                    "path": native_path.name,
                },
                "preview_png": {
                    "display_only_nearest_neighbor_scale": 4,
                    "file_sha256": _file_sha256(preview_path),
                    "path": preview_path.name,
                },
                "prompt": prompt,
                "prompt_utf8_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    report = {
        "artifact_kind": "mugen_latent_still_dit_inference",
        "checkpoint": {
            "file_sha256": expected_checkpoint_sha256,
            "path": str(checkpoint_file),
            "step": checkpoint.get("step"),
        },
        "codec_checkpoint": {
            "file_sha256": expected_codec_checkpoint_sha256,
            "path": str(codec_file),
        },
        "config": asdict(experiment),
        "noise_batch_sha256": noise_sha256,
        "samples": sample_records,
        "text_encoder_source_index_sha256": expected_text_source_index_sha256,
        "transparency": "native learned RGBA decoded by the frozen 2x codec",
    }
    report_path = output / "inference-report.json"
    report_payload = _canonical_json(report)
    _atomic_bytes(report_path, report_payload)
    return report_path, hashlib.sha256(report_payload).hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _tensor_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value.detach().float().cpu().numpy())
    return _array_sha256(array)


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
