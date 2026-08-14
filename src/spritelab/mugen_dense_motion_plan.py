"""Reference-conditioned latent-motion plans for dense M.U.G.E.N actions."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.storage import DiskGuard


def build_mugen_dense_motion_plan(
    materialization_path_value: Path | str,
    latent_manifest_path: Path | str,
) -> dict[str, Any]:
    """Join exact source clips, codec latents, idle references, and actions."""

    materialization_path = Path(materialization_path_value).resolve()
    latent_path = Path(latent_manifest_path).resolve()
    materialization_bytes = materialization_path.read_bytes()
    latent_bytes = latent_path.read_bytes()
    materialization = _object(json.loads(materialization_bytes), "materialization")
    latent = _object(json.loads(latent_bytes), "latent manifest")
    materialization_kind = materialization.get("artifact_kind")
    if materialization_kind not in {
        "mugen_dense_captioned_materialization_bridge",
        "mugen_dense_reference_motion_training_manifest",
    }:
        raise ValueError("materialization has the wrong artifact kind")
    if latent.get("artifact_kind") != "mugen_frozen_rgba_autoencoder_latent_cache":
        raise ValueError("latent manifest has the wrong artifact kind")
    latent_source = _object(latent.get("source"), "latent source")
    materialization_sha256 = hashlib.sha256(materialization_bytes).hexdigest()
    latent_source_materialization_path = Path(
        _text(latent_source, "materialization_path")
    ).resolve()
    latent_source_materialization_bytes = latent_source_materialization_path.read_bytes()
    latent_source_materialization_sha256 = hashlib.sha256(
        latent_source_materialization_bytes
    ).hexdigest()
    if latent_source.get("materialization_file_sha256") != latent_source_materialization_sha256:
        raise ValueError("latent cache source materialization hash differs")
    if materialization_kind == "mugen_dense_reference_motion_training_manifest":
        sequences = _raw_dense_sequences(
            materialization,
            relative_root=latent_source_materialization_path.parent,
        )
    else:
        sequences = _counted(materialization, "sequences", "sequence_count", "materialization")
    latent_rows = _counted(latent, "records", "record_count", "latent manifest")
    latent_by_sequence = _unique(latent_rows, "sequence_id", "latent manifest")
    sequence_by_id = _unique(sequences, "sequence_id", "materialization")
    missing_latents = set(sequence_by_id) - set(latent_by_sequence)
    if missing_latents:
        raise ValueError(f"latent cache omits materialization sequences: {len(missing_latents)}")
    unused_latents = set(latent_by_sequence) - set(sequence_by_id)
    by_identity: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        by_identity[_text(sequence, "identity_id")].append(sequence)
    reference_by_identity = {}
    for identity_id, rows in by_identity.items():
        idle = [row for row in rows if row.get("action") == "idle"]
        if len(idle) != 1:
            raise ValueError(f"identity must have exactly one idle sequence: {identity_id}")
        reference_by_identity[identity_id] = _reference_record(
            idle[0],
            latent_by_sequence[_text(idle[0], "sequence_id")],
            latent_root=latent_path.parent,
        )

    records = []
    action_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for sequence in sorted(sequences, key=lambda row: _text(row, "sequence_id").encode()):
        sequence_id = _text(sequence, "sequence_id")
        identity_id = _text(sequence, "identity_id")
        action = _text(sequence, "action")
        latent_row = latent_by_sequence[sequence_id]
        _validate_latent_source(latent_row, sequence, sequence_id)
        timing = _object(sequence.get("timing"), "timing")
        phases = _float_list(timing.get("phase"), "timing.phase", 8)
        durations = _float_list(timing.get("duration_ms"), "timing.duration_ms", 8)
        output = _object(sequence.get("output"), "output")
        reference = reference_by_identity[identity_id]
        records.append(
            {
                "conditioning": {
                    "direction": _text(sequence, "direction"),
                    "verb": action,
                    "view": _text(sequence, "view"),
                },
                "entity_class": _text(sequence, "entity_class"),
                "identity_id": identity_id,
                "reference": reference,
                "reference_target_relation": (
                    "same_sequence_idle_reference"
                    if action == "idle"
                    else "same_identity_idle_reference"
                ),
                "sample_id": "motion_"
                + hashlib.sha256(f"mugen_dense_motion_v1\0{sequence_id}".encode()).hexdigest()[:32],
                "sequence_id": sequence_id,
                "split": _text(sequence, "split"),
                "target": {
                    "duration_ms": durations,
                    "latent": _latent_array_record(latent_row),
                    "loop_mode": _text(sequence, "loop_mode"),
                    "phase": phases,
                    "source_pixels": {
                        "array_content_sha256": _digest(output, "array_content_sha256"),
                        "file_sha256": _digest(output, "file_sha256"),
                        "relative_path": _text(output, "relative_path"),
                        "shape": [8, 128, 128, 4],
                    },
                },
            }
        )
        action_counts[action] += 1
        split_counts[_text(sequence, "split")] += 1
    return {
        "artifact_kind": "mugen_reference_conditioned_latent_motion_plan",
        "counts": {
            "actions": dict(sorted(action_counts.items())),
            "identities": len(by_identity),
            "sequences": len(records),
            "splits": dict(sorted(split_counts.items())),
        },
        "records": records,
        "schema_version": 3,
        "source": {
            "latent_manifest": {
                "file_sha256": hashlib.sha256(latent_bytes).hexdigest(),
                "path": str(latent_path),
                "scope": {
                    "joined_sequences": len(sequence_by_id),
                    "policy": "exact_closure_or_verified_superset",
                    "source_materialization_file_sha256": (latent_source_materialization_sha256),
                    "source_materialization_path": str(latent_source_materialization_path),
                    "unused_latent_sequences": len(unused_latents),
                },
            },
            "materialization": {
                "artifact_kind": materialization_kind,
                "file_sha256": materialization_sha256,
                "path": str(materialization_path),
            },
        },
        "training_contract": {
            "action_vocabulary": sorted(action_counts),
            "identity_component_split_disjoint": True,
            "reference": "exact_dense_selected_idle_temporal_medoid_latent",
            "stage": "canonical_still_plus_action_token_to_latent_animation",
            "target": "ordered_eight_frame_rgba_codec_motion_residual",
        },
    }


def build_mugen_dense_motion_training_manifest(
    motion_plan_path: Path | str,
) -> dict[str, Any]:
    """Admit every one-per-identity/action dense record to the motion trainer."""

    path = Path(motion_plan_path).resolve()
    payload = path.read_bytes()
    plan = _object(json.loads(payload), "motion plan")
    if plan.get("artifact_kind") != "mugen_reference_conditioned_latent_motion_plan":
        raise ValueError("motion plan has the wrong artifact kind")
    records = plan.get("records")
    counts = _object(plan.get("counts"), "motion counts")
    if (
        not isinstance(records, list)
        or counts.get("sequences") != len(records)
        or any(not isinstance(row, dict) for row in records)
    ):
        raise ValueError("motion plan sequence count differs")
    identity_actions = set()
    for row in records:
        key = (
            _text(row, "identity_id"),
            _text(_object(row.get("conditioning"), "conditioning"), "verb"),
        )
        if key in identity_actions:
            raise ValueError(f"motion plan duplicates identity/action: {key}")
        identity_actions.add(key)
    split_identities: defaultdict[str, set[str]] = defaultdict(set)
    for row in records:
        split_identities[_text(row, "split")].add(_text(row, "identity_id"))
    return {
        "artifact_kind": "mugen_reference_conditioned_primary_motion_training_manifest",
        "config": {
            "one_sequence_per_identity_verb": True,
            "required_pixel_gate_status": "dense_stream_quality_all_pass",
            "verbs": sorted(counts.get("actions", {})),
        },
        "counts": {
            "actions": counts.get("actions"),
            "identities": counts.get("identities"),
            "sequences": len(records),
            "split_identities": {
                split: len(values) for split, values in sorted(split_identities.items())
            },
            "splits": counts.get("splits"),
        },
        "policy": {
            "action_balance": "hierarchical identity_then_action sampling",
            "admission": "dense streamed-v2 six-action quality tier",
            "target_cardinality": "one canonical sequence per identity and action",
        },
        "records": records,
        "schema_version": 2,
        "source": {
            "motion_plan_file_sha256": hashlib.sha256(payload).hexdigest(),
            "motion_plan_path": str(path),
        },
    }


def export_mugen_dense_motion_artifacts(
    materialization_path: Path | str,
    latent_manifest_path: Path | str,
    output_directory: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> tuple[str, str]:
    """Publish plan and training manifest together via one atomic directory."""

    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace dense motion artifacts: {output}")
    guard = disk_guard or DiskGuard(Path(output.anchor), 100 * 1024**3)
    guard.require_capacity(256 * 1024**2, label="dense MUGEN motion plans")
    stage = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if stage.exists():
        raise FileExistsError(f"Refusing to replace dense motion stage: {stage}")
    stage.mkdir(parents=True)
    final_plan = output / "motion-plan.json"
    try:
        plan = build_mugen_dense_motion_plan(materialization_path, latent_manifest_path)
        plan_payload = _canonical(plan)
        stage_plan = stage / "motion-plan.json"
        _write(stage_plan, plan_payload)
        # The training manifest must bind the final immutable pathname, while
        # its bytes bind the already-written plan payload.
        training = build_mugen_dense_motion_training_manifest(stage_plan)
        training["source"]["motion_plan_path"] = str(final_plan)
        training_payload = _canonical(training)
        _write(stage / "training-manifest.json", training_payload)
        os.rename(stage, output)
    except BaseException:
        if stage.exists():
            for child in stage.iterdir():
                child.unlink()
            stage.rmdir()
        raise
    return hashlib.sha256(plan_payload).hexdigest(), hashlib.sha256(training_payload).hexdigest()


def _raw_dense_sequences(
    materialization: dict[str, Any], *, relative_root: Path
) -> list[dict[str, Any]]:
    """Project the audited dense schema without requiring appearance captions.

    Motion conditioning consumes only the exact idle reference, action token,
    and phase.  Caption closure belongs to the independent text-to-still stage.
    """

    if materialization.get("schema_version") != 1:
        raise ValueError("dense materialization schema version is unsupported")
    records = materialization.get("records")
    sources = materialization.get("source_materializations")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("dense materialization records are invalid")
    if not isinstance(sources, list) or any(not isinstance(row, dict) for row in sources):
        raise ValueError("dense materialization sources are invalid")
    roots = [Path(_text(row, "root")).resolve() for row in sources]
    sequences: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        identity_id = _text(record, "identity_id")
        source_index = record.get("source_index")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise ValueError(f"dense source index is invalid: {identity_id}")
        if not 0 <= source_index < len(roots):
            raise ValueError(f"dense source index is out of range: {identity_id}")
        reference = _object(record.get("reference"), "dense reference")
        reference_frame_index = reference.get("frame_index")
        if (
            isinstance(reference_frame_index, bool)
            or not isinstance(reference_frame_index, int)
            or not 0 <= reference_frame_index < 8
        ):
            raise ValueError(f"dense reference frame index is invalid: {identity_id}")
        reference_sha256 = _digest(reference, "frame_array_content_sha256")
        actions = record.get("actions")
        if not isinstance(actions, list) or any(not isinstance(row, dict) for row in actions):
            raise ValueError(f"dense actions are invalid: {identity_id}")
        for action in actions:
            sequence_id = _text(action, "record_id")
            if sequence_id in seen:
                raise ValueError(f"dense materialization duplicates sequence: {sequence_id}")
            seen.add(sequence_id)
            array = _object(action.get("array"), "dense action array")
            source_path = roots[source_index].joinpath(_text(array, "relative_path")).resolve()
            if (
                roots[source_index] != source_path
                and roots[source_index] not in source_path.parents
            ):
                raise ValueError(f"dense action array escapes source root: {sequence_id}")
            try:
                relative_path = source_path.relative_to(relative_root).as_posix()
            except ValueError as error:
                raise ValueError(
                    f"dense action array is outside latent materialization root: {sequence_id}"
                ) from error
            temporal = _object(action.get("temporal_selection"), "temporal selection")
            phases = _float_list(temporal.get("target_phases"), "target phases", 8)
            loop_mode = _text(action, "loop_mode")
            if loop_mode == "loop":
                if phases[0] != 0.0 or any(not 0 <= value < 1 for value in phases):
                    raise ValueError(f"loop phases are invalid: {sequence_id}")
            elif loop_mode == "one_shot":
                if phases[0] != 0.0 or phases[-1] != 1.0:
                    raise ValueError(f"one-shot phases are invalid: {sequence_id}")
            else:
                raise ValueError(f"loop mode is invalid: {sequence_id}")
            sequences.append(
                {
                    "action": _text(action, "slot"),
                    "caption": {
                        "reference_frame_array_content_sha256": reference_sha256,
                        "reference_frame_index": reference_frame_index,
                    },
                    "direction": "unknown",
                    "entity_class": "unknown",
                    "identity_id": identity_id,
                    "loop_mode": loop_mode,
                    "output": {
                        "array_content_sha256": _digest(array, "array_content_sha256"),
                        "file_sha256": _digest(array, "file_sha256"),
                        "relative_path": relative_path,
                        "shape": [8, 128, 128, 4],
                    },
                    "sequence_id": sequence_id,
                    "split": _text(record, "split"),
                    "timing": {
                        "duration_ms": [125.0] * 8,
                        "phase": phases,
                    },
                    "view": "side",
                }
            )
    return sequences


def _reference_record(
    sequence: dict[str, Any], latent: dict[str, Any], *, latent_root: Path
) -> dict[str, Any]:
    sequence_id = _text(sequence, "sequence_id")
    _validate_latent_source(latent, sequence, sequence_id)
    relative = _text(latent, "relative_path")
    path = (latent_root / relative).resolve()
    if latent_root not in path.parents:
        raise ValueError(f"latent path escapes root: {sequence_id}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != _digest(latent, "file_sha256"):
        raise ValueError(f"latent file hash differs: {sequence_id}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.float16 or value.shape != (8, 8, 64, 64):
        raise ValueError(f"latent geometry differs: {sequence_id}")
    if _array_sha256(value) != _digest(latent, "array_content_sha256"):
        raise ValueError(f"latent content hash differs: {sequence_id}")
    caption = _object(sequence.get("caption"), "caption")
    frame_index = caption.get("reference_frame_index")
    if (
        isinstance(frame_index, bool)
        or not isinstance(frame_index, int)
        or not 0 <= frame_index < 8
    ):
        raise ValueError(f"reference frame index differs: {sequence_id}")
    frame = np.ascontiguousarray(value[frame_index])
    return {
        "frame_index": frame_index,
        "latent": {
            **_latent_array_record(latent),
            "frame_array_content_sha256": _array_sha256(frame),
        },
        "sequence_id": sequence_id,
        "source_pixel_frame_array_content_sha256": _digest(
            caption, "reference_frame_array_content_sha256"
        ),
    }


def _validate_latent_source(
    latent: dict[str, Any], sequence: dict[str, Any], sequence_id: str
) -> None:
    if latent.get("identity_id") != sequence.get("identity_id") or latent.get(
        "split"
    ) != sequence.get("split"):
        raise ValueError(f"latent identity/split differs: {sequence_id}")
    source = _object(latent.get("source"), "latent source")
    output = _object(sequence.get("output"), "sequence output")
    for key in ("array_content_sha256", "file_sha256", "relative_path"):
        if source.get(key) != output.get(key):
            raise ValueError(f"latent source differs: {sequence_id}: {key}")


def _latent_array_record(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("dtype") != "float16" or value.get("shape") != [8, 8, 64, 64]:
        raise ValueError(f"latent record geometry differs: {value.get('sequence_id')}")
    return {
        "array_content_sha256": _digest(value, "array_content_sha256"),
        "dtype": "float16",
        "file_sha256": _digest(value, "file_sha256"),
        "relative_path": _text(value, "relative_path"),
        "shape": [8, 8, 64, 64],
    }


def _counted(value: dict[str, Any], key: str, count_key: str, label: str) -> list[dict[str, Any]]:
    rows = value.get(key)
    if (
        not isinstance(rows, list)
        or value.get(count_key) != len(rows)
        or any(not isinstance(row, dict) for row in rows)
    ):
        raise ValueError(f"{label} count differs")
    return rows


def _unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        identifier = _text(row, key)
        if identifier in output:
            raise ValueError(f"{label} duplicates {key}: {identifier}")
        output[identifier] = row
    return output


def _float_list(value: Any, label: str, count: int) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"{label} is invalid")
    output = [float(item) for item in value]
    if any(not np.isfinite(item) for item in output):
        raise ValueError(f"{label} contains non-finite values")
    return output


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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
