"""Immutable reference-conditioned MUGEN motion training plan."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard


def build_mugen_reference_motion_plan(
    materialization_path: Path | str,
    taxonomy_path: Path | str,
    canonical_still_plan_path: Path | str,
    latent_manifest_path: Path | str,
) -> dict[str, Any]:
    """Join exact appearance references, action evidence, and target clip latents."""

    paths = {
        "materialization": Path(materialization_path).resolve(),
        "taxonomy": Path(taxonomy_path).resolve(),
        "canonical_still_plan": Path(canonical_still_plan_path).resolve(),
        "latent_manifest": Path(latent_manifest_path).resolve(),
    }
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    documents = {name: _object(payload, name) for name, payload in payloads.items()}
    materialization = documents["materialization"]
    taxonomy = documents["taxonomy"]
    still_plan = documents["canonical_still_plan"]
    latent_manifest = documents["latent_manifest"]

    if materialization.get("schema_version") != 1:
        raise ValueError("materialization schema is unsupported")
    if taxonomy.get("artifact_kind") != "mugen_materialized_structured_action_taxonomy":
        raise ValueError("taxonomy has the wrong artifact kind")
    if still_plan.get("artifact_kind") != "mugen_canonical_appearance_still_training_plan":
        raise ValueError("canonical still plan has the wrong artifact kind")
    if latent_manifest.get("artifact_kind") != "mugen_frozen_rgba_autoencoder_latent_cache":
        raise ValueError("latent manifest has the wrong artifact kind")

    sequences = _counted(materialization, "sequences", "sequence_count", "materialization")
    taxonomy_records = _counted(taxonomy, "records", "sequence_count", "taxonomy")
    still_records = _nested_counted(
        still_plan, "records", ("counts", "identities"), "canonical still plan"
    )
    latent_records = _counted(latent_manifest, "records", "record_count", "latent manifest")
    sequence_by_id = _unique(sequences, "sequence_id", "materialization")
    taxonomy_by_id = _unique(taxonomy_records, "sequence_id", "taxonomy")
    latent_by_id = _unique(latent_records, "sequence_id", "latent manifest")
    still_by_identity = _unique(still_records, "identity_id", "canonical still plan")
    sequence_ids = set(sequence_by_id)
    if set(taxonomy_by_id) != sequence_ids or set(latent_by_id) != sequence_ids:
        raise ValueError("sequence closure differs across materialization, taxonomy, and latents")
    identities = {_text(record, "identity_id") for record in sequences}
    if set(still_by_identity) != identities:
        raise ValueError("canonical still identity closure differs from materialization")

    reference_by_identity: dict[str, dict[str, Any]] = {}
    for identity_id, still in still_by_identity.items():
        reference_sequence_id = _text(still, "sequence_id")
        reference_sequence = sequence_by_id.get(reference_sequence_id)
        reference_latent = latent_by_id.get(reference_sequence_id)
        if reference_sequence is None or reference_latent is None:
            raise ValueError(f"canonical reference sequence is absent: {reference_sequence_id}")
        target = _dict(still, "target")
        indices = target.get("eligible_frame_indices")
        if not isinstance(indices, list) or len(indices) != 1 or not isinstance(indices[0], int):
            raise ValueError(f"canonical reference frame is invalid: {identity_id}")
        frame_index = indices[0]
        if not 0 <= frame_index < 8:
            raise ValueError(f"canonical reference frame is out of range: {identity_id}")
        output = _dict(reference_sequence, "output")
        if target.get("file_sha256") != output.get("file_sha256") or target.get(
            "array_content_sha256"
        ) != output.get("array_content_sha256"):
            raise ValueError(
                f"canonical reference pixels differ from materialization: {identity_id}"
            )
        _validate_latent(reference_latent, reference_sequence, reference_sequence_id)
        latent_file = _resolve_under(
            paths["latent_manifest"].parent, _text(reference_latent, "relative_path")
        )
        latent_bytes = latent_file.read_bytes()
        if hashlib.sha256(latent_bytes).hexdigest() != reference_latent.get("file_sha256"):
            raise ValueError(f"canonical reference latent file differs: {identity_id}")
        array = np.load(io.BytesIO(latent_bytes), allow_pickle=False)
        if array.dtype != np.float16 or array.shape != (8, 8, 64, 64):
            raise ValueError(f"canonical reference latent tensor differs: {identity_id}")
        if _array_sha256(array) != reference_latent.get("array_content_sha256"):
            raise ValueError(f"canonical reference latent array differs: {identity_id}")
        reference_frame = np.ascontiguousarray(array[frame_index])
        reference_by_identity[identity_id] = {
            "appearance_prompt": _text(still, "prompt"),
            "frame_index": frame_index,
            "identity_reference_array_sha256": _dict(still, "caption_reference")[
                "identity_reference_array_sha256"
            ],
            "latent": {
                "array_content_sha256": reference_latent["array_content_sha256"],
                "dtype": "float16",
                "file_sha256": reference_latent["file_sha256"],
                "frame_array_content_sha256": _array_sha256(reference_frame),
                "frame_shape": [8, 64, 64],
                "relative_path": reference_latent["relative_path"],
                "shape": [8, 8, 64, 64],
            },
            "sample_id": _text(still, "sample_id"),
            "sequence_id": reference_sequence_id,
        }

    records: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    entity_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    for sequence_id in sorted(sequence_by_id, key=lambda value: value.encode()):
        sequence = sequence_by_id[sequence_id]
        action = taxonomy_by_id[sequence_id]
        latent = latent_by_id[sequence_id]
        identity_id = _text(sequence, "identity_id")
        split = _text(sequence, "split")
        if action.get("identity_id") != identity_id or action.get("split") != split:
            raise ValueError(f"taxonomy identity/split differs: {sequence_id}")
        _validate_latent(latent, sequence, sequence_id)
        reference = reference_by_identity[identity_id]
        reference_target_relation = (
            "same_sequence" if reference["sequence_id"] == sequence_id else "cross_sequence"
        )
        verb = _text(action, "verb")
        entity_class = _text(sequence, "entity_class")
        timing = _dict(sequence, "timing")
        phases = timing.get("phase")
        durations = timing.get("duration_ms")
        if (
            not isinstance(phases, list)
            or len(phases) != 8
            or not all(isinstance(value, (int, float)) for value in phases)
            or not isinstance(durations, list)
            or len(durations) != 8
            or not all(isinstance(value, (int, float)) and value > 0 for value in durations)
        ):
            raise ValueError(f"target timing is invalid: {sequence_id}")
        records.append(
            {
                "conditioning": {
                    "attack_form": action.get("attack_form"),
                    "attack_strength": action.get("attack_strength"),
                    "attack_tier": action.get("attack_tier"),
                    "direction": action.get("direction"),
                    "legacy_action": action.get("legacy_action"),
                    "phase": action.get("phase"),
                    "stance": action.get("stance"),
                    "verb": verb,
                },
                "entity_class": entity_class,
                "identity_id": identity_id,
                "reference": reference,
                "reference_target_relation": reference_target_relation,
                "sample_id": "motion_"
                + hashlib.sha256(
                    f"mugen_reference_motion_v2\0{identity_id}\0{sequence_id}".encode()
                ).hexdigest()[:32],
                "sequence_id": sequence_id,
                "split": split,
                "target": {
                    "duration_ms": [float(value) for value in durations],
                    "latent": {
                        "array_content_sha256": latent["array_content_sha256"],
                        "dtype": "float16",
                        "file_sha256": latent["file_sha256"],
                        "relative_path": latent["relative_path"],
                        "shape": [8, 8, 64, 64],
                    },
                    "loop_mode": sequence.get("loop_mode"),
                    "phase": [float(value) for value in phases],
                    "source_pixels": {
                        "array_content_sha256": _dict(sequence, "output")["array_content_sha256"],
                        "file_sha256": _dict(sequence, "output")["file_sha256"],
                        "relative_path": _dict(sequence, "output")["relative_path"],
                        "shape": [8, 128, 128, 4],
                    },
                },
            }
        )
        action_counts[verb] += 1
        split_counts[split] += 1
        entity_counts[entity_class] += 1
        relation_counts[reference_target_relation] += 1

    return {
        "artifact_kind": "mugen_reference_conditioned_latent_motion_plan",
        "counts": {
            "actions": _sorted_counter(action_counts),
            "entity_classes": _sorted_counter(entity_counts),
            "identities": len(reference_by_identity),
            "reference_target_relation": _sorted_counter(relation_counts),
            "sequences": len(records),
            "splits": _sorted_counter(split_counts),
        },
        "records": records,
        "schema_version": 2,
        "source": {
            name: {"file_sha256": hashlib.sha256(payloads[name]).hexdigest(), "path": str(path)}
            for name, path in paths.items()
        },
        "training_contract": {
            "identity_split_disjoint": True,
            "reference": "one_exact_vlm_selected_canonical_still_latent_per_identity",
            "same_sequence_reference_is_standard_image_to_video_conditioning": True,
            "stage": "canonical_still_plus_structured_action_to_latent_animation",
            "target": "ordered_eight_frame_rgba_codec_latents",
            "target_prediction": "motion_or_residual_latents_not_direct_pixels",
        },
    }


def export_mugen_reference_motion_plan(
    materialization_path: Path | str,
    taxonomy_path: Path | str,
    canonical_still_plan_path: Path | str,
    latent_manifest_path: Path | str,
    output_path: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Publish the reference-motion plan atomically with no-clobber semantics."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace MUGEN reference-motion plan: {output}")
    plan = build_mugen_reference_motion_plan(
        materialization_path, taxonomy_path, canonical_still_plan_path, latent_manifest_path
    )
    payload = canonical_json_bytes(plan)
    (disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)).require_capacity(
        len(payload) + 16 * 1024**2, label="MUGEN reference-motion plan"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return output, hashlib.sha256(payload).hexdigest()


def _validate_latent(latent: dict[str, Any], sequence: dict[str, Any], sequence_id: str) -> None:
    output = _dict(sequence, "output")
    source = _dict(latent, "source")
    if latent.get("shape") != [8, 8, 64, 64] or latent.get("dtype") != "float16":
        raise ValueError(f"latent geometry is invalid: {sequence_id}")
    if latent.get("identity_id") != sequence.get("identity_id") or latent.get(
        "split"
    ) != sequence.get("split"):
        raise ValueError(f"latent identity/split differs: {sequence_id}")
    if source.get("file_sha256") != output.get("file_sha256") or source.get(
        "array_content_sha256"
    ) != output.get("array_content_sha256"):
        raise ValueError(f"latent source pixels differ: {sequence_id}")


def _counted(value: dict[str, Any], records_key: str, count_key: str, label: str) -> list[dict]:
    records = value.get(records_key)
    if not isinstance(records, list) or value.get(count_key) != len(records):
        raise ValueError(f"{label} record count differs")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{label} records must be objects")
    return records


def _nested_counted(
    value: dict[str, Any], records_key: str, count_path: tuple[str, str], label: str
) -> list[dict]:
    counts = value.get(count_path[0])
    records = value.get(records_key)
    if (
        not isinstance(counts, dict)
        or not isinstance(records, list)
        or counts.get(count_path[1]) != len(records)
        or not all(isinstance(record, dict) for record in records)
    ):
        raise ValueError(f"{label} record count differs")
    return records


def _unique(records: list[dict], key: str, label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for record in records:
        identifier = _text(record, key)
        if identifier in result:
            raise ValueError(f"{label} duplicates {key}: {identifier}")
        result[identifier] = record
    return result


def _object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be non-empty text")
    return result


def _dict(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be an object")
    return result


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"path escapes artifact root: {relative}")
    return path


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: item[0].encode()))
