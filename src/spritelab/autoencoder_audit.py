"""Hash-bound held-out reconstruction audits for sprite autoencoders."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from spritelab.broad_train import PreparedBroadRow, prepare_broad_corpus
from spritelab.models.sprite_autoencoder import SpriteAutoencoderConfig, SpriteRGBAAutoencoder
from spritelab.storage import DiskGuard

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency boundary
    torch = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class AutoencoderAuditContractError(ValueError):
    """Raised when checkpoint, corpus, or output contracts do not match."""


@dataclass(frozen=True, slots=True)
class AutoencoderAuditResult:
    output_directory: Path
    report_path: Path
    gallery_path: Path
    target_array_path: Path
    reconstruction_array_path: Path
    report_sha256: str
    gallery_sha256: str


def select_identity_diverse_frames(
    rows: tuple[PreparedBroadRow, ...], *, maximum_frames: int
) -> tuple[tuple[int, int], ...]:
    """Select one stable frame per held-out identity before any second sample."""

    if isinstance(maximum_frames, bool) or not isinstance(maximum_frames, int):
        raise TypeError("maximum_frames must be an integer")
    if maximum_frames <= 0:
        raise ValueError("maximum_frames must be positive")
    candidates: dict[str, list[tuple[bytes, int, int]]] = {}
    for row_index, row in enumerate(rows):
        bucket = candidates.setdefault(row.identity_id, [])
        for frame_index in range(row.rgba.shape[0]):
            digest = hashlib.sha256(
                f"{row.identity_id}\0{row.action}\0{row.sequence_id}\0{frame_index}".encode()
            ).digest()
            bucket.append((digest, row_index, frame_index))
    for bucket in candidates.values():
        bucket.sort(key=lambda value: value[0])
    output: list[tuple[bytes, int, int]] = []
    depth = 0
    while len(output) < maximum_frames:
        round_values = [
            bucket[depth]
            for _, bucket in sorted(candidates.items(), key=lambda item: item[0].encode())
            if depth < len(bucket)
        ]
        if not round_values:
            break
        output.extend(sorted(round_values, key=lambda value: value[0]))
        depth += 1
    return tuple((row, frame) for _, row, frame in output[:maximum_frames])


def export_autoencoder_reconstruction_audit(
    materialization_manifest: Path | str,
    checkpoint_path: Path | str,
    output_directory: Path | str,
    *,
    expected_checkpoint_sha256: str,
    maximum_frames: int = 16,
    maximum_gallery_frames: int = 16,
    integer_scale: int = 2,
    allow_legacy_torch_version: bool = False,
    disk_guard: DiskGuard | None = None,
) -> AutoencoderAuditResult:
    """Reconstruct fixed held-out frames and publish exact arrays plus a display gallery."""

    runtime = _require_torch()
    source = Path(checkpoint_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace reconstruction audit output: {output}")
    if not source.is_file():
        raise FileNotFoundError(f"autoencoder checkpoint does not exist: {source}")
    _validate_digest(expected_checkpoint_sha256, "expected_checkpoint_sha256")
    actual_checkpoint_sha256 = _file_sha256(source)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise AutoencoderAuditContractError(
            "checkpoint SHA-256 mismatch: "
            f"expected {expected_checkpoint_sha256}, got {actual_checkpoint_sha256}"
        )
    checkpoint, load_contract = _safe_load_checkpoint(
        runtime, source, allow_legacy_torch_version=allow_legacy_torch_version
    )
    if checkpoint.get("artifact_kind") != "sprite_rgba_autoencoder_resume_checkpoint":
        raise AutoencoderAuditContractError("checkpoint has the wrong artifact kind")
    if checkpoint.get("schema_version") != 1:
        raise AutoencoderAuditContractError("checkpoint schema version is unsupported")
    step = checkpoint.get("step")
    config_record = checkpoint.get("config")
    corpus_record = checkpoint.get("corpus")
    ema_state = checkpoint.get("ema")
    if not isinstance(step, int) or step <= 0:
        raise AutoencoderAuditContractError("checkpoint step must be positive")
    if not isinstance(config_record, dict) or not isinstance(corpus_record, dict):
        raise AutoencoderAuditContractError("checkpoint config/corpus records are invalid")
    if not isinstance(ema_state, dict):
        raise AutoencoderAuditContractError("checkpoint EMA state is invalid")
    architecture_record = config_record.get("architecture")
    if not isinstance(architecture_record, dict):
        raise AutoencoderAuditContractError("checkpoint architecture record is invalid")
    architecture_values = dict(architecture_record)
    multipliers = architecture_values.get("channel_multipliers")
    if isinstance(multipliers, list):
        architecture_values["channel_multipliers"] = tuple(multipliers)
    architecture = SpriteAutoencoderConfig(**architecture_values)
    corpus = prepare_broad_corpus(
        materialization_manifest,
        target_size=architecture.image_size,
        target_frames=8,
        usage="autoencoder",
    )
    expected_corpus = {
        "corpus_sha256": corpus.corpus_sha256,
        "materialization_manifest_sha256": corpus.materialization_manifest_sha256,
        "source_snapshot_canonical_sha256": corpus.source_snapshot_canonical_sha256,
        "source_snapshot_manifest_sha256": corpus.source_snapshot_manifest_sha256,
        "train_rows": len(corpus.train),
        "validation_rows": len(corpus.validation),
    }
    if corpus_record != expected_corpus:
        raise AutoencoderAuditContractError("checkpoint corpus contract does not match manifest")
    if isinstance(maximum_gallery_frames, bool) or not isinstance(maximum_gallery_frames, int):
        raise TypeError("maximum_gallery_frames must be an integer")
    if maximum_gallery_frames <= 0:
        raise ValueError("maximum_gallery_frames must be positive")
    selection = select_identity_diverse_frames(corpus.validation, maximum_frames=maximum_frames)
    if not selection:
        raise AutoencoderAuditContractError("held-out reconstruction selection is empty")
    target = np.stack([corpus.validation[row].rgba[frame] for row, frame in selection], axis=0)
    target_tensor = runtime.from_numpy(
        np.ascontiguousarray(target.transpose(0, 3, 1, 2), dtype=np.float32) / 255
    )
    model = SpriteRGBAAutoencoder(architecture).cpu().eval()
    model.load_state_dict(ema_state, strict=True)
    with runtime.no_grad():
        reconstruction_tensor = runtime.sigmoid(model(target_tensor)).float()
    reconstruction = (
        reconstruction_tensor.clamp(0, 1)
        .mul(255)
        .round()
        .to(runtime.uint8)
        .numpy()
        .transpose(0, 2, 3, 1)
    )
    reconstruction = np.ascontiguousarray(reconstruction)
    samples = []
    for index, (row_index, frame_index) in enumerate(selection):
        row = corpus.validation[row_index]
        samples.append(
            {
                "action": row.action,
                "description": str(row.request.description),
                "frame_index": frame_index,
                "identity_id": row.identity_id,
                "metrics": _metrics(target[index], reconstruction[index]),
                "sequence_id": row.sequence_id,
            }
        )
    aggregate = _metrics(target, reconstruction)
    target_bytes = _npy_bytes(target)
    reconstruction_bytes = _npy_bytes(reconstruction)
    gallery_count = min(len(samples), maximum_gallery_frames)
    gallery_bytes = _gallery_bytes(
        target[:gallery_count],
        reconstruction[:gallery_count],
        samples[:gallery_count],
        integer_scale=integer_scale,
    )
    target_sha256 = hashlib.sha256(target_bytes).hexdigest()
    reconstruction_sha256 = hashlib.sha256(reconstruction_bytes).hexdigest()
    gallery_sha256 = hashlib.sha256(gallery_bytes).hexdigest()
    report = {
        "aggregate_metrics": aggregate,
        "artifact_kind": "sprite_rgba_autoencoder_held_out_reconstruction_audit",
        "checkpoint": {
            "artifact_kind": checkpoint["artifact_kind"],
            "file_sha256": actual_checkpoint_sha256,
            "load_contract": load_contract,
            "step": step,
        },
        "corpus": expected_corpus,
        "display_gallery": {
            "file_sha256": gallery_sha256,
            "integer_scale": integer_scale,
            "layout": "two_samples_per_row_each_target_then_reconstruction",
            "path": "held-out-reconstruction-gallery.png",
            "resampling": "nearest_positive_integer",
            "sample_count": gallery_count,
        },
        "reconstruction_array": {
            "array_content_sha256": _array_sha256(reconstruction),
            "file_sha256": reconstruction_sha256,
            "path": "held-out-reconstructions.npy",
        },
        "samples": samples,
        "schema_version": 1,
        "selection": "sha256_ranked_one_frame_per_identity_before_second_frame_v1",
        "target_array": {
            "array_content_sha256": _array_sha256(target),
            "file_sha256": target_sha256,
            "path": "held-out-targets.npy",
        },
    }
    report_bytes = _canonical_json(report)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    if disk_guard is not None:
        disk_guard.require_capacity(
            len(target_bytes) + len(reconstruction_bytes) + len(gallery_bytes) + len(report_bytes),
            label="autoencoder reconstruction audit",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent)
    )
    _exclusive_write(staging / "held-out-targets.npy", target_bytes)
    _exclusive_write(staging / "held-out-reconstructions.npy", reconstruction_bytes)
    _exclusive_write(staging / "held-out-reconstruction-gallery.png", gallery_bytes)
    _exclusive_write(staging / "audit-report.json", report_bytes)
    try:
        os.rename(staging, output)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing to replace reconstruction audit output: {output}"
        ) from error
    return AutoencoderAuditResult(
        output,
        output / "audit-report.json",
        output / "held-out-reconstruction-gallery.png",
        output / "held-out-targets.npy",
        output / "held-out-reconstructions.npy",
        report_sha256,
        gallery_sha256,
    )


def _safe_load_checkpoint(
    runtime: Any, path: Path, *, allow_legacy_torch_version: bool
) -> tuple[dict[str, Any], str]:
    try:
        payload = runtime.load(path, map_location="cpu", weights_only=True)
        contract = "torch.load(weights_only=True,map_location='cpu')"
    except Exception as first_error:
        if not allow_legacy_torch_version:
            raise AutoencoderAuditContractError(
                "checkpoint failed stock weights_only=True loading"
            ) from first_error
        try:
            from torch.torch_version import TorchVersion

            with runtime.serialization.safe_globals([TorchVersion]):
                payload = runtime.load(path, map_location="cpu", weights_only=True)
        except Exception as second_error:
            raise AutoencoderAuditContractError(
                "checkpoint failed legacy TorchVersion-only safe loading"
            ) from second_error
        contract = (
            "torch.load(weights_only=True,map_location='cpu',"
            "safe_globals=[torch.torch_version.TorchVersion])"
        )
    if not isinstance(payload, dict):
        raise AutoencoderAuditContractError("checkpoint payload must be a dictionary")
    return payload, contract


def _metrics(target: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    target_unit = target.astype(np.float32) / 255
    reconstruction_unit = reconstruction.astype(np.float32) / 255
    target_alpha = target_unit[..., 3:4]
    reconstruction_alpha = reconstruction_unit[..., 3:4]
    target_pm = np.concatenate((target_unit[..., :3] * target_alpha, target_alpha), axis=-1)
    reconstruction_pm = np.concatenate(
        (reconstruction_unit[..., :3] * reconstruction_alpha, reconstruction_alpha), axis=-1
    )
    target_mask = target[..., 3] >= 127
    reconstruction_mask = reconstruction[..., 3] >= 127
    union = np.logical_or(target_mask, reconstruction_mask).sum()
    visible_denominator = float(target_alpha.sum() * 3)
    visible = np.abs(reconstruction_unit[..., :3] - target_unit[..., :3]) * target_alpha
    return {
        "alpha_iou_127": float(
            np.logical_and(target_mask, reconstruction_mask).sum() / union if union else 1.0
        ),
        "alpha_mae": float(np.abs(reconstruction_alpha - target_alpha).mean()),
        "premultiplied_rgba_mae": float(np.abs(reconstruction_pm - target_pm).mean()),
        "visible_rgb_mae": float(visible.sum() / visible_denominator)
        if visible_denominator
        else 0.0,
    }


def _gallery_bytes(
    target: np.ndarray,
    reconstruction: np.ndarray,
    samples: list[dict[str, Any]],
    *,
    integer_scale: int,
) -> bytes:
    if isinstance(integer_scale, bool) or not isinstance(integer_scale, int):
        raise TypeError("integer_scale must be an integer")
    if integer_scale <= 0:
        raise ValueError("integer_scale must be positive")
    height, width = target.shape[1:3]
    cell_width = width * integer_scale
    cell_height = height * integer_scale
    label_height = 24
    samples_per_row = 2
    rows = (len(samples) + samples_per_row - 1) // samples_per_row
    image = Image.new(
        "RGBA",
        (cell_width * 4, rows * (cell_height + label_height)),
        (18, 20, 26, 255),
    )
    draw = ImageDraw.Draw(image)
    for index, sample in enumerate(samples):
        row = index // samples_per_row
        group = index % samples_per_row
        left = group * cell_width * 2
        top = row * (cell_height + label_height)
        for offset, frame in enumerate((target[index], reconstruction[index])):
            display = frame.copy()
            display[..., :3][display[..., 3] == 0] = 0
            sprite = Image.fromarray(display).resize(
                (cell_width, cell_height), Image.Resampling.NEAREST
            )
            image.alpha_composite(sprite, (left + offset * cell_width, top))
        label = f"{sample['action']}  target | reconstruction"
        draw.text((left + 4, top + cell_height + 4), label, fill=(235, 238, 245, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return buffer.getvalue()


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _exclusive_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "autoencoder reconstruction audit requires PyTorch"
        ) from _TORCH_IMPORT_ERROR
    return torch
