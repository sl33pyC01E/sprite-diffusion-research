"""Safe, hash-bound export of intermediate latent-still EMA checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from spritelab.storage import DiskGuard


class LatentStillCheckpointError(ValueError):
    """Raised when intermediate checkpoint evidence is incomplete or inconsistent."""


def export_latent_still_intermediate_ema(
    training_checkpoint_path: Path | str,
    latent_manifest_path: Path | str,
    output_path: Path | str,
    *,
    expected_training_checkpoint_sha256: str,
    expected_latent_manifest_sha256: str,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Project one resumable training checkpoint into the inference-only schema."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("intermediate checkpoint export requires PyTorch") from error
    training_path = Path(training_checkpoint_path).resolve()
    latent_path = Path(latent_manifest_path).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace inference checkpoint: {output}")
    _require_digest(expected_training_checkpoint_sha256, "training checkpoint")
    _require_digest(expected_latent_manifest_sha256, "latent manifest")
    if _file_sha256(training_path) != expected_training_checkpoint_sha256:
        raise LatentStillCheckpointError("training checkpoint SHA-256 differs")
    latent_bytes = latent_path.read_bytes()
    if hashlib.sha256(latent_bytes).hexdigest() != expected_latent_manifest_sha256:
        raise LatentStillCheckpointError("latent manifest SHA-256 differs")
    try:
        checkpoint = torch.load(training_path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise LatentStillCheckpointError("training checkpoint failed safe load") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("artifact_kind") != (
        "mugen_latent_still_dit_resume_checkpoint"
    ):
        raise LatentStillCheckpointError("training checkpoint has the wrong artifact kind")
    corpus = checkpoint.get("corpus")
    config = checkpoint.get("config")
    ema_model = checkpoint.get("ema_model")
    step = checkpoint.get("step")
    if not isinstance(corpus, dict) or corpus.get("latent_manifest_file_sha256") != (
        expected_latent_manifest_sha256
    ):
        raise LatentStillCheckpointError("training checkpoint latent corpus differs")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise LatentStillCheckpointError("training checkpoint model config is missing")
    if not isinstance(ema_model, dict) or not ema_model:
        raise LatentStillCheckpointError("training checkpoint EMA weights are missing")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise LatentStillCheckpointError("training checkpoint step is invalid")
    try:
        latent_manifest = json.loads(latent_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LatentStillCheckpointError("latent manifest is invalid JSON") from error
    if not isinstance(latent_manifest, dict) or latent_manifest.get("artifact_kind") != (
        "mugen_frozen_rgba_autoencoder_latent_cache"
    ):
        raise LatentStillCheckpointError("latent manifest has the wrong artifact kind")
    normalization = latent_manifest.get("normalization")
    if not isinstance(normalization, dict):
        raise LatentStillCheckpointError("latent normalization is missing")
    mean = _normalization_values(normalization, "channel_mean")
    standard_deviation = _normalization_values(
        normalization, "channel_standard_deviation", positive=True
    )
    artifact = {
        "artifact_kind": "mugen_latent_still_dit_ema_inference_checkpoint",
        "config": config,
        "corpus": corpus,
        "ema_model": ema_model,
        "ema_policy": checkpoint.get("ema_policy"),
        "lineage": {
            "source_training_checkpoint_path": str(training_path),
            "source_training_checkpoint_sha256": expected_training_checkpoint_sha256,
        },
        "normalization": {
            "channel_mean": mean,
            "channel_standard_deviation": standard_deviation,
        },
        "step": step,
    }
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(1024**3, label="intermediate latent-still EMA checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace temporary checkpoint: {temporary}")
    try:
        with temporary.open("xb") as handle:
            torch.save(artifact, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return output, _file_sha256(output)


def _normalization_values(
    normalization: dict[str, Any], key: str, *, positive: bool = False
) -> list[float]:
    value = normalization.get(key)
    if not isinstance(value, list) or len(value) != 8:
        raise LatentStillCheckpointError(f"{key} must contain eight values")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) or (positive and item <= 0) for item in result):
        raise LatentStillCheckpointError(f"{key} contains invalid values")
    return result


def _require_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} SHA-256 must be lowercase hexadecimal")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
