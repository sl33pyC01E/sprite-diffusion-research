"""Resumable exact latent-cache export for the selected sprite codec."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.models.sprite_autoencoder import SpriteAutoencoderConfig, SpriteRGBAAutoencoder
from spritelab.storage import DiskGuard

try:
    import torch
except ImportError as exc:  # pragma: no cover - optional dependency boundary
    torch = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class LatentCacheError(ValueError):
    """Raised when source, checkpoint, journal, or latent evidence disagrees."""


def export_mugen_latent_cache(
    materialization_path: Path | str,
    checkpoint_path: Path | str,
    output_directory: Path | str,
    *,
    expected_checkpoint_sha256: str,
    batch_sequences: int = 8,
    device: str = "cuda",
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Encode every sequence with the frozen EMA codec into float16 latents."""

    runtime = _require_torch()
    if batch_sequences <= 0:
        raise ValueError("batch_sequences must be positive")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not runtime.cuda.is_available():
        raise RuntimeError("CUDA latent export requested but unavailable")
    _validate_digest(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    materialization_file = Path(materialization_path).resolve()
    checkpoint_file = Path(checkpoint_path).resolve()
    output = Path(output_directory).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace finalized latent cache: {manifest_path}")
    checkpoint_sha256 = _file_sha256(checkpoint_file)
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise LatentCacheError("autoencoder checkpoint SHA-256 mismatch")
    checkpoint = runtime.load(checkpoint_file, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise LatentCacheError("autoencoder checkpoint must contain a dictionary")
    if checkpoint.get("artifact_kind") != "sprite_rgba_autoencoder_resume_checkpoint":
        raise LatentCacheError("autoencoder checkpoint has the wrong artifact kind")
    checkpoint_step = checkpoint.get("step")
    if (
        checkpoint.get("schema_version") != 1
        or isinstance(checkpoint_step, bool)
        or not isinstance(checkpoint_step, int)
        or checkpoint_step <= 0
    ):
        raise LatentCacheError("autoencoder checkpoint version/step is unsupported")
    config_record = checkpoint.get("config")
    ema = checkpoint.get("ema")
    if not isinstance(config_record, dict) or not isinstance(ema, dict):
        raise LatentCacheError("autoencoder checkpoint lacks config or EMA weights")
    architecture_record = config_record.get("architecture")
    if not isinstance(architecture_record, dict):
        raise LatentCacheError("autoencoder architecture record is missing")
    architecture_values = dict(architecture_record)
    if isinstance(architecture_values.get("channel_multipliers"), list):
        architecture_values["channel_multipliers"] = tuple(
            architecture_values["channel_multipliers"]
        )
    architecture = SpriteAutoencoderConfig(**architecture_values)
    if (
        architecture.image_size != 128
        or architecture.latent_size != 64
        or architecture.latent_channels != 8
        or architecture.downsample_factor != 2
    ):
        raise LatentCacheError("checkpoint is not the selected 128-to-64x64x8 codec")
    materialization_bytes = materialization_file.read_bytes()
    materialization = _json_object(materialization_bytes, "materialization")
    sequences = materialization.get("sequences")
    if (
        not isinstance(sequences, list)
        or materialization.get("sequence_count") != len(sequences)
        or not all(isinstance(row, dict) for row in sequences)
    ):
        raise LatentCacheError("materialization sequence count is invalid")
    sequence_by_id = _unique_sequences(sequences)
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(
        len(sequences) * 8 * 8 * 64 * 64 * 2 + 512 * 1024**2,
        label="MUGEN float16 latent cache",
    )
    output.mkdir(parents=True, exist_ok=True)
    latent_dir = output / "latents"
    latent_dir.mkdir(exist_ok=True)
    journal_path = output / "records.jsonl"
    completed = _load_journal(journal_path, output)
    unknown = set(completed) - set(sequence_by_id)
    if unknown:
        raise LatentCacheError(f"latent journal contains unknown sequences: {sorted(unknown)!r}")
    model = SpriteRGBAAutoencoder(architecture).to(device).eval()
    model.load_state_dict(ema, strict=True)
    pending = [
        sequence_by_id[key] for key in sorted(set(sequence_by_id) - set(completed), key=str.encode)
    ]
    with journal_path.open("a", encoding="utf-8", newline="\n") as journal:
        for start in range(0, len(pending), batch_sequences):
            batch_rows = pending[start : start + batch_sequences]
            arrays = []
            source_records = []
            for sequence in batch_rows:
                array, source_record = _load_source_array(materialization_file.parent, sequence)
                arrays.append(array)
                source_records.append(source_record)
            stacked = np.concatenate(arrays, axis=0)
            tensor = runtime.from_numpy(
                np.ascontiguousarray(stacked.transpose(0, 3, 1, 2), dtype=np.float32) / 255
            ).to(device)
            with runtime.no_grad():
                encoded = model.encode(tensor).float().cpu().numpy()
            encoded = encoded.reshape(
                len(batch_rows),
                8,
                architecture.latent_channels,
                architecture.latent_size,
                architecture.latent_size,
            )
            for sequence, source_record, latent in zip(
                batch_rows, source_records, encoded, strict=True
            ):
                sequence_id = sequence["sequence_id"]
                latent = np.ascontiguousarray(latent, dtype=np.float16)
                payload = _npy_bytes(latent)
                relative_path = f"latents/{sequence_id}.npy"
                latent_path = output / relative_path
                if latent_path.exists():
                    existing_payload = latent_path.read_bytes()
                    if existing_payload != payload:
                        raise LatentCacheError(
                            f"unjournaled latent file differs from recomputation: {relative_path}"
                        )
                else:
                    with latent_path.open("xb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                record = {
                    "action": sequence.get("action"),
                    "array_content_sha256": _array_sha256(latent),
                    "dtype": "float16",
                    "entity_class": sequence.get("entity_class"),
                    "file_sha256": hashlib.sha256(payload).hexdigest(),
                    "identity_id": sequence.get("identity_id"),
                    "relative_path": relative_path,
                    "sequence_id": sequence_id,
                    "shape": list(latent.shape),
                    "source": source_record,
                    "split": sequence.get("split"),
                    "structured_verb_pending_taxonomy_join": True,
                }
                journal.write(_canonical_json(record).decode())
                journal.flush()
                os.fsync(journal.fileno())
                completed[sequence_id] = record
            print(
                json.dumps(
                    {"completed": len(completed), "remaining": len(sequences) - len(completed)},
                    sort_keys=True,
                ),
                flush=True,
            )
    if set(completed) != set(sequence_by_id):
        raise LatentCacheError("latent cache journal is incomplete")
    ordered = [completed[key] for key in sorted(completed, key=str.encode)]
    statistics = latent_channel_statistics(
        output,
        [record for record in ordered if record["split"] == "train"],
    )
    quantization_audit = _quantized_reconstruction_audit(
        runtime,
        model,
        materialization_file.parent,
        sequence_by_id,
        completed,
        output,
        device=device,
    )
    manifest = {
        "artifact_kind": "mugen_frozen_rgba_autoencoder_latent_cache",
        "codec": {
            "architecture": architecture_record,
            "checkpoint_file_sha256": checkpoint_sha256,
            "checkpoint_path": str(checkpoint_file),
            "checkpoint_step": checkpoint["step"],
            "state": "EMA",
        },
        "normalization": {
            "application": "per_channel_(value-mean)/std",
            "estimated_from_split": "train",
            **statistics,
        },
        "quantized_storage_audit": quantization_audit,
        "record_count": len(ordered),
        "records": ordered,
        "schema_version": 1,
        "source": {
            "materialization_file_sha256": hashlib.sha256(materialization_bytes).hexdigest(),
            "materialization_path": str(materialization_file),
        },
        "storage": {
            "dtype": "float16",
            "format": "numpy_npy_v1_allow_pickle_false",
            "latent_layout": "[T,C,H,W]",
            "source_encoder_compute_dtype": "float32",
        },
    }
    payload = _canonical_json(manifest)
    with manifest_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_path, hashlib.sha256(payload).hexdigest()


def latent_channel_statistics(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute deterministic float64 channel moments from verified latent files."""

    if not records:
        raise ValueError("latent statistics require at least one record")
    total = None
    total_square = None
    count = 0
    for record in records:
        latent = _load_latent_record(root, record)
        values = latent.astype(np.float64)
        row_sum = values.sum(axis=(0, 2, 3))
        row_square = np.square(values).sum(axis=(0, 2, 3))
        total = row_sum if total is None else total + row_sum
        total_square = row_square if total_square is None else total_square + row_square
        count += latent.shape[0] * latent.shape[2] * latent.shape[3]
    if total is None or total_square is None or count <= 0:
        raise AssertionError("unreachable empty latent statistics")
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 0)
    standard_deviation = np.sqrt(variance)
    if not bool(np.isfinite(standard_deviation).all()) or bool((standard_deviation <= 0).any()):
        raise LatentCacheError("latent channel standard deviations must be finite and positive")
    return {
        "channel_mean": mean.tolist(),
        "channel_standard_deviation": standard_deviation.tolist(),
        "scalar_count_per_channel": count,
    }


def _quantized_reconstruction_audit(
    runtime: Any,
    model: Any,
    materialization_root: Path,
    sequence_by_id: dict[str, dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    output: Path,
    *,
    device: str,
) -> dict[str, Any]:
    candidates: dict[str, str] = {}
    for sequence_id in sorted(sequence_by_id, key=str.encode):
        sequence = sequence_by_id[sequence_id]
        if sequence.get("split") == "validation":
            identity_id = sequence.get("identity_id")
            if isinstance(identity_id, str):
                candidates.setdefault(identity_id, sequence_id)
    selected = list(candidates.items())[:16]
    if not selected:
        raise LatentCacheError("latent audit has no validation identities")
    targets = []
    reconstructions = []
    samples = []
    for identity_id, sequence_id in selected:
        source, _ = _load_source_array(materialization_root, sequence_by_id[sequence_id])
        latent = _load_latent_record(output, completed[sequence_id])
        latent_tensor = runtime.from_numpy(latent[0:1].astype(np.float32)).to(device)
        with runtime.no_grad():
            reconstructed = model.decode(latent_tensor).clamp(0, 1).mul(255).round()
        reconstructed = reconstructed.to(runtime.uint8).cpu().numpy().transpose(0, 2, 3, 1)[0]
        target = np.ascontiguousarray(source[0])
        targets.append(target)
        reconstructions.append(np.ascontiguousarray(reconstructed))
        samples.append({"identity_id": identity_id, "sequence_id": sequence_id})
    target_array = np.stack(targets)
    reconstruction_array = np.stack(reconstructions)
    return {
        "aggregate_metrics": _rgba_metrics(target_array, reconstruction_array),
        "sample_count": len(samples),
        "samples": samples,
        "selection": "first_UTF8_sequence_per_validation_identity_then_first_16_frame0",
    }


def _rgba_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    target_unit = target.astype(np.float64) / 255
    predicted_unit = predicted.astype(np.float64) / 255
    target_alpha = target_unit[..., 3:4]
    predicted_alpha = predicted_unit[..., 3:4]
    target_pm = np.concatenate((target_unit[..., :3] * target_alpha, target_alpha), axis=-1)
    predicted_pm = np.concatenate(
        (predicted_unit[..., :3] * predicted_alpha, predicted_alpha), axis=-1
    )
    visible = target_alpha[..., 0] > 0
    visible_rgb = (
        float(np.abs(target_unit[..., :3][visible] - predicted_unit[..., :3][visible]).mean())
        if bool(visible.any())
        else 0.0
    )
    intersection = np.logical_and(target_alpha[..., 0] >= 0.5, predicted_alpha[..., 0] >= 0.5)
    union = np.logical_or(target_alpha[..., 0] >= 0.5, predicted_alpha[..., 0] >= 0.5)
    return {
        "alpha_iou_127": float(intersection.sum() / max(int(union.sum()), 1)),
        "alpha_mae": float(np.abs(target_alpha - predicted_alpha).mean()),
        "premultiplied_rgba_mae": float(np.abs(target_pm - predicted_pm).mean()),
        "visible_rgb_mae": visible_rgb,
    }


def _load_source_array(root: Path, sequence: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    sequence_id = sequence.get("sequence_id")
    output = sequence.get("output")
    if not isinstance(sequence_id, str) or not isinstance(output, dict):
        raise LatentCacheError("materialization sequence/output is invalid")
    relative = output.get("relative_path")
    expected_file = output.get("file_sha256")
    expected_array = output.get("array_content_sha256")
    if not all(isinstance(value, str) for value in (relative, expected_file, expected_array)):
        raise LatentCacheError(f"materialization output evidence is invalid for {sequence_id}")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise LatentCacheError(f"materialization path escapes root for {sequence_id}")
    payload = path.read_bytes()
    actual_file = hashlib.sha256(payload).hexdigest()
    if actual_file != expected_file:
        raise LatentCacheError(f"materialization file hash mismatch for {sequence_id}")
    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise LatentCacheError(f"materialization array is unreadable for {sequence_id}") from error
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.shape != (8, 128, 128, 4)
    ):
        raise LatentCacheError(f"materialization array geometry is invalid for {sequence_id}")
    actual_array = _array_sha256(value)
    if actual_array != expected_array:
        raise LatentCacheError(f"materialization array hash mismatch for {sequence_id}")
    return np.ascontiguousarray(value), {
        "array_content_sha256": actual_array,
        "file_sha256": actual_file,
        "relative_path": relative,
    }


def _load_journal(path: Path, root: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise LatentCacheError(f"latent journal has invalid line {line_number}") from error
            sequence_id = record.get("sequence_id") if isinstance(record, dict) else None
            if not isinstance(sequence_id, str) or sequence_id in output:
                raise LatentCacheError(f"latent journal has invalid sequence at line {line_number}")
            _load_latent_record(root, record)
            output[sequence_id] = record
    return output


def _load_latent_record(root: Path, record: dict[str, Any]) -> np.ndarray:
    relative = record.get("relative_path")
    if not isinstance(relative, str):
        raise LatentCacheError("latent record path is invalid")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise LatentCacheError("latent record escapes cache root")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record.get("file_sha256"):
        raise LatentCacheError(f"latent file hash mismatch: {relative}")
    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise LatentCacheError(f"latent file is unreadable: {relative}") from error
    if value.dtype != np.float16 or value.shape != (8, 8, 64, 64):
        raise LatentCacheError(f"latent array geometry is invalid: {relative}")
    if _array_sha256(value) != record.get("array_content_sha256"):
        raise LatentCacheError(f"latent array hash mismatch: {relative}")
    if not bool(np.isfinite(value).all()):
        raise LatentCacheError(f"latent array contains non-finite values: {relative}")
    return np.ascontiguousarray(value)


def _unique_sequences(sequences: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for sequence in sequences:
        sequence_id = sequence.get("sequence_id")
        if not isinstance(sequence_id, str) or sequence_id in output:
            raise LatentCacheError("materialization sequence IDs must be unique strings")
        output[sequence_id] = sequence
    return output


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LatentCacheError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise LatentCacheError(f"{label} must contain an object")
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


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError("latent cache export requires PyTorch") from _TORCH_IMPORT_ERROR
    return torch
