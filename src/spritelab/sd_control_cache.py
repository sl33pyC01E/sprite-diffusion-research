"""Frozen Stable Diffusion 1.4 RGB-control latent cache for MUGEN stills."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.storage import DiskGuard


class SDControlCacheError(ValueError):
    """Raised when RGB projection, source evidence, or VAE output differs."""


def composite_rgba_on_background(
    rgba: np.ndarray, *, background_rgb: tuple[int, int, int] = (127, 127, 127)
) -> np.ndarray:
    """Composite uint8 RGBA exactly in float32 for the noncanonical RGB control."""

    if rgba.dtype != np.uint8 or rgba.ndim != 4 or rgba.shape[-1] != 4:
        raise ValueError("rgba must be uint8 [T,H,W,4]")
    if (
        len(background_rgb) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) for item in background_rgb)
        or any(not 0 <= item <= 255 for item in background_rgb)
    ):
        raise ValueError("background_rgb must contain three uint8 integers")
    unit = rgba.astype(np.float32) / 255
    alpha = unit[..., 3:4]
    background = np.asarray(background_rgb, dtype=np.float32) / 255
    return np.ascontiguousarray(unit[..., :3] * alpha + background * (1 - alpha))


def export_sd14_rgb_latent_cache(
    plan_path: Path | str,
    model_directory: Path | str,
    output_directory: Path | str,
    *,
    expected_source_index_sha256: str,
    device: str = "cuda",
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Project all exact RGBA sequences into deterministic SD1.4 VAE modes."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    _validate_digest(expected_source_index_sha256, "expected_source_index_sha256")
    try:
        import torch
        from diffusers import AutoencoderKL
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("SD control latent export requires Torch and Diffusers") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA SD control latent export requested but unavailable")
    plan_file = Path(plan_path).resolve()
    model_root = Path(model_directory).resolve()
    output = Path(output_directory).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace finalized SD control cache: {manifest_path}")
    source_index_path = model_root / "source-index.json"
    source_index_bytes = source_index_path.read_bytes()
    if hashlib.sha256(source_index_bytes).hexdigest() != expected_source_index_sha256:
        raise SDControlCacheError("Stable Diffusion source-index SHA-256 mismatch")
    source_index = _json_object(source_index_bytes, "Stable Diffusion source index")
    _verify_model_files(model_root, source_index)
    plan_bytes = plan_file.read_bytes()
    plan = _json_object(plan_bytes, "training plan")
    records = plan.get("records")
    expected_count = (
        plan.get("counts", {}).get("sequences") if isinstance(plan.get("counts"), dict) else None
    )
    if not isinstance(records, list) or expected_count != len(records):
        raise SDControlCacheError("training plan record count differs")
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(
        len(records) * 8 * 4 * 64 * 64 * 2 + 512 * 1024**2, label="SD1.4 RGB latent cache"
    )
    output.mkdir(parents=True, exist_ok=True)
    latent_dir = output / "latents"
    latent_dir.mkdir(exist_ok=True)
    journal_path = output / "records.jsonl"
    completed = _load_journal(journal_path, output)
    model = (
        AutoencoderKL.from_pretrained(
            model_root / "vae", local_files_only=True, use_safetensors=True
        )
        .to(device)
        .eval()
    )
    model.requires_grad_(False)
    scaling_factor = float(model.config.scaling_factor)
    if not math_isclose(scaling_factor, 0.18215):
        raise SDControlCacheError(f"unexpected SD1.4 VAE scaling factor: {scaling_factor}")
    by_id = {_required_text(row, "sequence_id"): row for row in records}
    if len(by_id) != len(records) or set(completed) - set(by_id):
        raise SDControlCacheError("training plan or journal sequence IDs are invalid")
    pending = [by_id[key] for key in sorted(set(by_id) - set(completed), key=str.encode)]
    with journal_path.open("a", encoding="utf-8", newline="\n") as journal:
        for plan_record in pending:
            sequence_id = _required_text(plan_record, "sequence_id")
            target = plan_record.get("target")
            if not isinstance(target, dict):
                raise SDControlCacheError(f"target is missing for {sequence_id}")
            relative = _required_text(target, "relative_path")
            target_path = (plan_file.parent / relative).resolve()
            if plan_file.parent not in target_path.parents:
                raise SDControlCacheError(f"target path escapes plan root for {sequence_id}")
            payload = target_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != target.get("file_sha256"):
                raise SDControlCacheError(f"target file hash mismatch for {sequence_id}")
            rgba = np.load(io.BytesIO(payload), allow_pickle=False)
            if rgba.dtype != np.uint8 or rgba.shape != (8, 128, 128, 4):
                raise SDControlCacheError(f"target geometry differs for {sequence_id}")
            if _array_sha256(rgba) != target.get("array_content_sha256"):
                raise SDControlCacheError(f"target array hash mismatch for {sequence_id}")
            rgb = composite_rgba_on_background(rgba)
            tensor = torch.from_numpy(rgb.transpose(0, 3, 1, 2)).to(device)
            tensor = torch.nn.functional.interpolate(tensor, size=(512, 512), mode="nearest")
            tensor = tensor.mul(2).sub(1)
            with torch.no_grad():
                encoded = model.encode(tensor).latent_dist.mode().mul(scaling_factor)
            value = np.ascontiguousarray(encoded.float().cpu().numpy(), dtype=np.float16)
            if value.shape != (8, 4, 64, 64) or not bool(np.isfinite(value).all()):
                raise SDControlCacheError(f"VAE output is invalid for {sequence_id}")
            output_payload = _npy_bytes(value)
            output_relative = f"latents/{sequence_id}.npy"
            output_path = output / output_relative
            if output_path.exists():
                if output_path.read_bytes() != output_payload:
                    raise SDControlCacheError(f"orphan output differs for {sequence_id}")
            else:
                with output_path.open("xb") as handle:
                    handle.write(output_payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            record = {
                "array_content_sha256": _array_sha256(value),
                "file_sha256": hashlib.sha256(output_payload).hexdigest(),
                "identity_id": plan_record.get("identity_id"),
                "relative_path": output_relative,
                "sequence_id": sequence_id,
                "shape": list(value.shape),
                "split": plan_record.get("split"),
                "source_target": {
                    "array_content_sha256": target["array_content_sha256"],
                    "file_sha256": target["file_sha256"],
                    "relative_path": relative,
                },
            }
            journal.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            )
            journal.flush()
            os.fsync(journal.fileno())
            completed[sequence_id] = record
            print(
                json.dumps(
                    {"completed": len(completed), "remaining": len(records) - len(completed)},
                    sort_keys=True,
                ),
                flush=True,
            )
    ordered = [completed[key] for key in sorted(completed, key=str.encode)]
    if len(ordered) != len(records):
        raise SDControlCacheError("SD control cache journal is incomplete")
    manifest = {
        "artifact_kind": "mugen_sd14_noncanonical_rgb_vae_latent_cache",
        "claim": "RGB quality control only; gray compositing discards canonical alpha",
        "projection": {
            "alpha_composite_background_rgb": [127, 127, 127],
            "input_geometry": "128x128_RGBA_uint8",
            "resize": "512x512_nearest_neighbor_before_VAE",
        },
        "record_count": len(ordered),
        "records": ordered,
        "schema_version": 1,
        "source": {
            "model_id": source_index.get("model_id"),
            "model_revision": source_index.get("resolved_revision"),
            "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "plan_path": str(plan_file),
            "source_index_file_sha256": expected_source_index_sha256,
        },
        "storage": {
            "dtype": "float16",
            "latent_layout": "[T,C,H,W]",
            "scaling_factor": scaling_factor,
            "vae_posterior": "mode_deterministic",
        },
    }
    manifest_payload = _canonical_json(manifest)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_payload)
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_path, hashlib.sha256(manifest_payload).hexdigest()


def _load_journal(path: Path, root: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    output = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise SDControlCacheError(f"journal line {line_number} is invalid") from error
        sequence_id = record.get("sequence_id") if isinstance(record, dict) else None
        if not isinstance(sequence_id, str) or sequence_id in output:
            raise SDControlCacheError(f"journal line {line_number} has invalid sequence")
        relative = _required_text(record, "relative_path")
        payload = (root / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != record.get("file_sha256"):
            raise SDControlCacheError(f"journal latent hash mismatch for {sequence_id}")
        output[sequence_id] = record
    return output


def _verify_model_files(root: Path, source_index: dict[str, Any]) -> None:
    files = source_index.get("files")
    if not isinstance(files, list) or not files:
        raise SDControlCacheError("Stable Diffusion source file index is invalid")
    for record in files:
        relative = record.get("relative_path") if isinstance(record, dict) else None
        expected = record.get("file_sha256") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise SDControlCacheError("Stable Diffusion source file record is invalid")
        path = (root / relative).resolve()
        if root not in path.parents or _file_sha256(path) != expected:
            raise SDControlCacheError(f"Stable Diffusion source file differs: {relative}")


def _required_text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise SDControlCacheError(f"field {key} must be non-empty text")
    return result


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SDControlCacheError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise SDControlCacheError(f"{label} must contain an object")
    return value


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


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


def _validate_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-8 * max(abs(left), abs(right), 1.0)
