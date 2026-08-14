"""Verified identity-reference inputs for dense M.U.G.E.N visual captioning."""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from spritelab.mugen_stills import compose_caption_input
from spritelab.spark_caption import caption_prompt_sha256
from spritelab.storage import DiskGuard


@dataclass(frozen=True, slots=True)
class MugenDenseCaptionReference:
    variant_id: str
    identity_id: str
    identity_label_provenance_only: str
    split: str
    frame_index: int
    rgba: np.ndarray
    source_array_file_sha256: str
    source_array_content_sha256: str
    reference_frame_array_content_sha256: str


def load_mugen_dense_caption_references(
    dense_manifest_path: Path | str,
) -> tuple[MugenDenseCaptionReference, ...]:
    """Hash-verify each selected temporal-medoid reference frame."""

    path = Path(dense_manifest_path).resolve()
    dense = _object(json.loads(path.read_bytes()), "dense manifest")
    if dense.get("artifact_kind") != "mugen_dense_reference_motion_training_manifest":
        raise ValueError("dense manifest has the wrong artifact kind")
    records = dense.get("records")
    sources = dense.get("source_materializations")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("dense records are invalid")
    if not isinstance(sources, list) or any(not isinstance(row, dict) for row in sources):
        raise ValueError("dense source materializations are invalid")
    roots = [Path(_text(row, "root")).resolve() for row in sources]
    output = []
    seen: set[str] = set()
    for record in records:
        variant_id = _text(record, "variant_id")
        if variant_id in seen:
            raise ValueError(f"dense manifest duplicates variant: {variant_id}")
        seen.add(variant_id)
        source_index = record.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"source_index is invalid: {variant_id}")
        if not 0 <= source_index < len(roots):
            raise ValueError(f"source_index is out of range: {variant_id}")
        reference = _object(record.get("reference"), "reference")
        if reference.get("selection_method") != "premultiplied_rgba_temporal_medoid_v1":
            raise ValueError(f"reference selection method differs: {variant_id}")
        array = _object(reference.get("array"), "reference array")
        relative = _safe_relative(_text(array, "relative_path"))
        root = roots[source_index]
        source_path = root.joinpath(*relative.parts).resolve()
        if root != source_path and root not in source_path.parents:
            raise ValueError(f"reference array escapes source root: {variant_id}")
        payload = source_path.read_bytes()
        file_sha256 = hashlib.sha256(payload).hexdigest()
        if file_sha256 != _digest(array, "file_sha256"):
            raise ValueError(f"reference array file hash differs: {variant_id}")
        value = np.load(source_path, allow_pickle=False)
        if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
            raise ValueError(f"reference array geometry differs: {variant_id}")
        if _array_sha256(value) != _digest(array, "array_content_sha256"):
            raise ValueError(f"reference array content hash differs: {variant_id}")
        frame_index = reference.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < 8
        ):
            raise ValueError(f"reference frame index differs: {variant_id}")
        rgba = np.ascontiguousarray(value[frame_index])
        frame_sha256 = _array_sha256(rgba)
        if frame_sha256 != _digest(reference, "frame_array_content_sha256"):
            raise ValueError(f"reference frame hash differs: {variant_id}")
        identity = _object(record.get("identity"), "identity")
        output.append(
            MugenDenseCaptionReference(
                variant_id=variant_id,
                identity_id=_text(record, "identity_id"),
                identity_label_provenance_only=_text(identity, "label"),
                split=_text(record, "split"),
                frame_index=frame_index,
                rgba=rgba,
                source_array_file_sha256=file_sha256,
                source_array_content_sha256=array["array_content_sha256"],
                reference_frame_array_content_sha256=frame_sha256,
            )
        )
    return tuple(sorted(output, key=lambda row: row.variant_id.encode("utf-8")))


def export_mugen_dense_caption_inputs(
    dense_manifest_path: Path | str,
    output_directory: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> str:
    """Publish 512px nearest-neighbor RGB inputs atomically and no-clobber."""

    dense_path = Path(dense_manifest_path).resolve()
    dense_bytes = dense_path.read_bytes()
    references = load_mugen_dense_caption_references(dense_path)
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace dense caption inputs: {output}")
    guard = disk_guard or DiskGuard(Path(output.anchor), 100 * 1024**3)
    guard.require_capacity(256 * 1024**2, label="dense MUGEN caption inputs")
    stage = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if stage.exists():
        raise FileExistsError(f"Refusing to replace caption-input stage: {stage}")
    input_dir = stage / "caption-inputs"
    input_dir.mkdir(parents=True)
    try:
        records = []
        for reference in references:
            payload = _caption_input_png(reference.rgba)
            digest = hashlib.sha256(payload).hexdigest()
            relative = f"caption-inputs/{reference.variant_id}-{digest[:12]}.png"
            destination = stage.joinpath(*PurePosixPath(relative).parts)
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            records.append(
                {
                    "caption_input": {
                        "composite_background_rgb": [127, 127, 127],
                        "file_sha256": digest,
                        "relative_path": relative,
                        "resize": "512x512_nearest_neighbor",
                        "size_bytes": len(payload),
                    },
                    "frame_index": reference.frame_index,
                    "identity_id": reference.identity_id,
                    "identity_label_provenance_only": reference.identity_label_provenance_only,
                    "reference_frame_array_content_sha256": (
                        reference.reference_frame_array_content_sha256
                    ),
                    "source_array_content_sha256": reference.source_array_content_sha256,
                    "source_array_file_sha256": reference.source_array_file_sha256,
                    "split": reference.split,
                    "variant_id": reference.variant_id,
                }
            )
        manifest = {
            "artifact_kind": "mugen_dense_literal_visual_caption_input_dataset",
            "caption_contract": {
                "franchise_and_identity_label_hidden_from_model": True,
                "prompt_sha256": caption_prompt_sha256(),
                "response": "strict literal structured visual JSON",
            },
            "record_count": len(records),
            "records": records,
            "schema_version": 1,
            "source": {
                "dense_manifest_file_sha256": hashlib.sha256(dense_bytes).hexdigest(),
                "dense_manifest_path": str(dense_path),
            },
        }
        manifest_payload = _canonical(manifest)
        with (stage / "manifest.json").open("xb") as handle:
            handle.write(manifest_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(stage, output)
    except BaseException:
        if stage.exists():
            import shutil

            shutil.rmtree(stage)
        raise
    return hashlib.sha256(manifest_payload).hexdigest()


def _caption_input_png(rgba: np.ndarray) -> bytes:
    composite = compose_caption_input(rgba)
    image = Image.fromarray(composite).resize((512, 512), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be non-empty text")
    return result


def _digest(value: dict[str, Any], key: str) -> str:
    result = _text(value, key)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{key} must be a lowercase SHA-256")
    return result


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
