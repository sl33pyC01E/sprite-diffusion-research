"""Zero-copy training-data bridge for audited dense M.U.G.E.N clips."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from spritelab.storage import DiskGuard


def build_mugen_dense_autoencoder_materialization(
    dense_manifest_path: Path | str,
    *,
    manifest_root: Path | str,
) -> dict[str, Any]:
    """Project dense clips into the verified loader schema without copying arrays.

    Descriptions are deliberately provenance labels, not visual captions.  The
    resulting artifact is explicitly restricted to autoencoder reconstruction.
    """

    dense_path = Path(dense_manifest_path).resolve()
    root = Path(manifest_root).resolve()
    dense_bytes = dense_path.read_bytes()
    dense = _object(json.loads(dense_bytes), "dense manifest")
    if dense.get("artifact_kind") != "mugen_dense_reference_motion_training_manifest":
        raise ValueError("dense manifest has the wrong artifact kind")
    if dense.get("schema_version") != 1:
        raise ValueError("dense manifest has an unsupported schema version")
    records = dense.get("records")
    sources = dense.get("source_materializations")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("dense records are invalid")
    if not isinstance(sources, list) or any(not isinstance(row, dict) for row in sources):
        raise ValueError("dense source materializations are invalid")
    source_roots = [_source_root(row, root) for row in sources]

    sequences = []
    seen: set[str] = set()
    for record in records:
        variant_id = _text(record, "variant_id")
        identity_id = _text(record, "identity_id")
        split = _text(record, "split")
        source_index = record.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"source_index is invalid: {variant_id}")
        if not 0 <= source_index < len(source_roots):
            raise ValueError(f"source_index is out of range: {variant_id}")
        source_root = source_roots[source_index]
        identity = _object(record.get("identity"), "identity")
        label = _text(identity, "label")
        sff_sha256 = _digest(record, "sff_sha256")
        actions = record.get("actions")
        if not isinstance(actions, list) or any(not isinstance(row, dict) for row in actions):
            raise ValueError(f"actions are invalid: {variant_id}")
        for action in actions:
            sequence_id = _text(action, "record_id")
            if sequence_id in seen:
                raise ValueError(f"duplicate sequence ID: {sequence_id}")
            seen.add(sequence_id)
            array = _object(action.get("array"), "action array")
            source_relative = _safe_relative(_text(array, "relative_path"))
            source_path = source_root.joinpath(*source_relative.parts).resolve()
            if source_root != source_path and source_root not in source_path.parents:
                raise ValueError(f"array escapes source root: {sequence_id}")
            try:
                relative = source_path.relative_to(root).as_posix()
            except ValueError as error:
                raise ValueError("all source roots must remain below manifest_root") from error
            payload = source_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != _digest(array, "file_sha256"):
                raise ValueError(f"array file hash differs: {sequence_id}")
            value = np.load(source_path, allow_pickle=False)
            if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
                raise ValueError(f"array geometry differs: {sequence_id}")
            if _array_sha256(value) != _digest(array, "array_content_sha256"):
                raise ValueError(f"array content hash differs: {sequence_id}")
            temporal = _object(action.get("temporal_selection"), "temporal selection")
            phases = temporal.get("target_phases")
            loop_mode = _text(action, "loop_mode")
            _validate_phases(phases, loop_mode=loop_mode, sequence_id=sequence_id)
            sequences.append(
                {
                    "action": _text(action, "slot"),
                    "caption": {
                        "description": label,
                        "description_basis": "def_identity_label_autoencoder_only",
                    },
                    "direction": "unknown",
                    "entity_class": "unknown",
                    "frame_count": 8,
                    "identity_id": identity_id,
                    "loop_mode": loop_mode,
                    "model_eligibility": {
                        "autoencoder_reconstruction": True,
                        "conditional_generation": False,
                        "conditional_generation_blocker": "structured_visual_caption_pending",
                    },
                    "output": {
                        "array_content_sha256": array["array_content_sha256"],
                        "dtype": "uint8",
                        "file_sha256": array["file_sha256"],
                        "format": "numpy_npy_v1",
                        "relative_path": relative,
                        "shape": [8, 128, 128, 4],
                        "size_bytes": len(payload),
                    },
                    "provenance": {
                        "source_blob_sha256": [sff_sha256],
                        "source_id": "mugen_manual_rar_core_v2",
                        "variant_id": variant_id,
                    },
                    "quality_tier": _text(
                        _object(dense.get("quality_audit"), "quality audit"),
                        "selected_tier",
                    ),
                    "sequence_id": sequence_id,
                    "split": split,
                    "target_bucket": [128, 128],
                    "timing": {
                        "duration_ms": [125.0] * 8,
                        "duration_method": "normalized_eight_phase_training_clock_v1",
                        "phase": phases,
                    },
                    "view": "side",
                }
            )
    sequences.sort(key=lambda row: row["sequence_id"].encode("utf-8"))
    dense_sha256 = hashlib.sha256(dense_bytes).hexdigest()
    quality = _object(dense.get("quality_audit"), "quality audit")
    return {
        "artifact_kind": "mugen_dense_autoencoder_only_materialization_bridge",
        "model_eligibility": {
            "autoencoder_reconstruction": True,
            "conditional_generation": False,
            "reason": "visual captions have not yet been joined",
        },
        "schema_version": 1,
        "sequence_count": len(sequences),
        "sequences": sequences,
        "source": {
            "dense_manifest_file_sha256": dense_sha256,
            "dense_manifest_path": str(dense_path),
        },
        "source_snapshot": {
            "canonical_sha256": _digest(quality, "file_sha256"),
            "manifest_sha256": dense_sha256,
            "schema_version": 1,
        },
    }


def export_mugen_dense_autoencoder_materialization(
    dense_manifest_path: Path | str,
    output_path: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> str:
    """Publish the zero-copy bridge canonically and without replacement."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace dense bridge: {output}")
    artifact = build_mugen_dense_autoencoder_materialization(
        dense_manifest_path,
        manifest_root=output.parent,
    )
    payload = _canonical(artifact)
    (disk_guard or DiskGuard(Path(output.anchor), 100 * 1024**3)).require_capacity(
        len(payload), label="MUGEN dense autoencoder bridge"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace temporary dense bridge: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return hashlib.sha256(payload).hexdigest()


def build_mugen_dense_captioned_materialization(
    dense_manifest_path: Path | str,
    caption_manifest_path: Path | str,
    *,
    manifest_root: Path | str,
) -> dict[str, Any]:
    """Join literal visual captions and enable conditional-generation loading."""

    dense_path = Path(dense_manifest_path).resolve()
    dense_value = _object(json.loads(dense_path.read_bytes()), "dense manifest")
    dense_rows = dense_value.get("records")
    if not isinstance(dense_rows, list) or any(not isinstance(row, dict) for row in dense_rows):
        raise ValueError("dense records are invalid")
    reference_by_variant = {
        _text(row, "variant_id"): _digest(
            _object(row.get("reference"), "dense reference"),
            "frame_array_content_sha256",
        )
        for row in dense_rows
    }
    artifact = build_mugen_dense_autoencoder_materialization(
        dense_manifest_path,
        manifest_root=manifest_root,
    )
    caption_path = Path(caption_manifest_path).resolve()
    caption_bytes = caption_path.read_bytes()
    caption = _object(json.loads(caption_bytes), "caption manifest")
    if caption.get("artifact_kind") != "mugen_dense_literal_visual_caption_dataset":
        raise ValueError("caption manifest has the wrong artifact kind")
    caption_rows = caption.get("records")
    if (
        not isinstance(caption_rows, list)
        or caption.get("record_count") != len(caption_rows)
        or any(not isinstance(row, dict) for row in caption_rows)
    ):
        raise ValueError("caption record count differs")
    caption_by_variant = {}
    for row in caption_rows:
        variant_id = _text(row, "variant_id")
        if variant_id in caption_by_variant:
            raise ValueError(f"caption manifest duplicates variant: {variant_id}")
        structured = _object(row.get("structured_caption"), "structured caption")
        _text(structured, "subject_type")
        _text(row, "training_appearance_prompt")
        frame_index = row.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < 8
        ):
            raise ValueError(f"caption reference frame index differs: {variant_id}")
        expected_reference = reference_by_variant.get(variant_id)
        if (
            expected_reference is not None
            and _digest(row, "reference_frame_array_content_sha256") != expected_reference
        ):
            raise ValueError(f"caption reference frame differs: {variant_id}")
        caption_by_variant[variant_id] = row
    variants = {sequence["provenance"]["variant_id"] for sequence in artifact["sequences"]}
    missing_captions = variants - set(caption_by_variant)
    if missing_captions:
        raise ValueError(
            f"caption manifest omits dense materialization variants: {len(missing_captions)}"
        )
    unused_captions = set(caption_by_variant) - variants
    for sequence in artifact["sequences"]:
        row = caption_by_variant[sequence["provenance"]["variant_id"]]
        if (
            row.get("identity_id") != sequence["identity_id"]
            or row.get("split") != sequence["split"]
        ):
            raise ValueError(f"caption identity/split differs: {sequence['sequence_id']}")
        sequence["caption"] = {
            "description": row["training_appearance_prompt"],
            "description_basis": "spark_literal_visual_structured_caption_v1",
            "reference_frame_array_content_sha256": _digest(
                row, "reference_frame_array_content_sha256"
            ),
            "reference_frame_index": int(row["frame_index"]),
            "request_body_sha256": _digest(row, "request_body_sha256"),
        }
        sequence["entity_class"] = _text(
            _object(row.get("structured_caption"), "structured caption"),
            "subject_type",
        )
        sequence["model_eligibility"] = {
            "autoencoder_reconstruction": True,
            "conditional_generation": True,
        }
    caption_sha256 = hashlib.sha256(caption_bytes).hexdigest()
    artifact["artifact_kind"] = "mugen_dense_captioned_materialization_bridge"
    artifact["model_eligibility"] = {
        "autoencoder_reconstruction": True,
        "conditional_generation": True,
        "reason": "literal visual caption closure complete",
    }
    artifact["source"]["caption_manifest_file_sha256"] = caption_sha256
    artifact["source"]["caption_manifest_path"] = str(caption_path)
    artifact["source"]["caption_manifest_scope"] = {
        "joined_variants": len(variants),
        "policy": "exact_closure_or_verified_superset",
        "unused_caption_variants": len(unused_captions),
    }
    artifact["source_snapshot"]["canonical_sha256"] = hashlib.sha256(
        (artifact["source_snapshot"]["canonical_sha256"] + "\0" + caption_sha256).encode()
    ).hexdigest()
    return artifact


def export_mugen_dense_captioned_materialization(
    dense_manifest_path: Path | str,
    caption_manifest_path: Path | str,
    output_path: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> str:
    """Publish the caption-complete zero-copy bridge without replacement."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace captioned dense bridge: {output}")
    artifact = build_mugen_dense_captioned_materialization(
        dense_manifest_path,
        caption_manifest_path,
        manifest_root=output.parent,
    )
    payload = _canonical(artifact)
    (disk_guard or DiskGuard(Path(output.anchor), 100 * 1024**3)).require_capacity(
        len(payload), label="captioned MUGEN dense bridge"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace temporary captioned bridge: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return hashlib.sha256(payload).hexdigest()


def _source_root(row: dict[str, Any], manifest_root: Path) -> Path:
    root = Path(_text(row, "root")).resolve()
    if manifest_root != root and manifest_root not in root.parents:
        raise ValueError("source materialization root is outside manifest_root")
    return root


def _validate_phases(value: Any, *, loop_mode: str, sequence_id: str) -> None:
    if loop_mode not in {"loop", "one_shot", "ping_pong"}:
        raise ValueError(f"unsupported loop mode: {sequence_id}: {loop_mode}")
    if (
        not isinstance(value, list)
        or len(value) != 8
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"target phases differ: {sequence_id}")
    phases = [float(item) for item in value]
    upper = 1.0 if loop_mode == "one_shot" else 1.0 - 1e-12
    if (
        phases != sorted(phases)
        or phases[0] != 0.0
        or any(item < 0.0 or item > upper for item in phases)
    ):
        raise ValueError(f"target phases differ: {sequence_id}")
    if loop_mode == "one_shot" and phases[-1] != 1.0:
        raise ValueError(f"one-shot target phases do not reach one: {sequence_id}")


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
