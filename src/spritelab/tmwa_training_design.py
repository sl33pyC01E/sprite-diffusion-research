"""Exact, provenance-bound training design for the TMWA action corpus.

The design deliberately operates only on an existing canonical snapshot and
materialization.  It verifies every materialized array through the training-data
loader, projects exact fixed-frame targets without interpolation, inventories
duplicates and causal action contrasts, and emits a no-clobber subset manifest.
It never reads or mutates the live index.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.storage import DiskGuard
from spritelab.training_data import MaterializedTrainingClip, load_materialized_training_clips


@dataclass(frozen=True, slots=True)
class TMWATrainingDesignDocuments:
    """Canonical in-memory documents produced by the exact design audit."""

    subset_manifest: dict[str, Any]
    design_audit: dict[str, Any]
    selected_sequence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TMWATrainingDesignResult:
    """Identity of one atomically published subset and design audit."""

    subset_manifest_path: Path
    subset_manifest_sha256: str
    subset_manifest_canonical_sha256: str
    design_audit_path: Path
    design_audit_sha256: str
    selected_sequence_count: int


def build_tmwa_training_design(
    source_snapshot_path: Path | str,
    materialization_path: Path | str,
    training_audit_path: Path | str,
    pixel_audit_path: Path | str,
    *,
    subset_manifest_path: Path | str,
    target_frames: int = 8,
    target_bucket: int = 64,
    direction: str = "down",
    actions: tuple[str, str] = ("idle", "walk"),
) -> TMWATrainingDesignDocuments:
    """Audit all clips and select every matching target-distinct action pair.

    A contrast condition holds identity, description, entity class, view,
    direction, loop mode, and the fixed-frame phase vector constant.  Within one
    condition, each action must map to one unambiguous model target, and retained
    actions must have distinct target hashes.
    """

    _positive_integer("target_frames", target_frames)
    _positive_integer("target_bucket", target_bucket)
    if not isinstance(direction, str) or not direction:
        raise ValueError("direction must be a non-empty string")
    if (
        not isinstance(actions, tuple)
        or len(actions) != 2
        or any(not isinstance(action, str) or not action for action in actions)
        or len(set(actions)) != 2
    ):
        raise ValueError("actions must contain two distinct non-empty strings")
    ordered_actions = tuple(sorted(actions, key=str.encode))

    snapshot_path = Path(source_snapshot_path).resolve()
    manifest_path = Path(materialization_path).resolve()
    readiness_path = Path(training_audit_path).resolve()
    pixels_path = Path(pixel_audit_path).resolve()
    subset_path = Path(subset_manifest_path).resolve()
    if subset_path.parent != manifest_path.parent:
        raise ValueError(
            "subset_manifest_path must share the source materialization directory "
            "so clip relative paths retain their exact meaning"
        )
    if subset_path.suffix.casefold() != ".json":
        raise ValueError("subset_manifest_path must end in .json")

    snapshot_bytes, snapshot = _read_object(snapshot_path, "source snapshot")
    manifest_bytes, manifest = _read_object(manifest_path, "materialization")
    readiness_bytes, readiness = _read_object(readiness_path, "training-readiness audit")
    pixels_bytes, pixels = _read_object(pixels_path, "pixel-quality audit")
    records = manifest.get("sequences")
    if not isinstance(records, list) or not records:
        raise ValueError("materialization must contain sequence records")
    if manifest.get("sequence_count") != len(records):
        raise ValueError("materialization sequence_count does not match its records")
    raw_by_sequence = _records_by_sequence(records)

    snapshot_canonical_sha256 = _canonical_sha256(snapshot)
    manifest_canonical_sha256 = _canonical_sha256(manifest)
    manifest_file_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    snapshot_file_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    readiness_file_sha256 = hashlib.sha256(readiness_bytes).hexdigest()
    pixels_file_sha256 = hashlib.sha256(pixels_bytes).hexdigest()
    _verify_upstream_contracts(
        snapshot,
        manifest,
        readiness,
        pixels,
        snapshot_canonical_sha256=snapshot_canonical_sha256,
        manifest_file_sha256=manifest_file_sha256,
        sequence_count=len(records),
        target_frames=target_frames,
    )

    native = load_materialized_training_clips(manifest_path, split=None)
    fixed = load_materialized_training_clips(
        manifest_path,
        split=None,
        target_frames=target_frames,
    )
    native_by_sequence = {clip.sequence_id: clip for clip in native}
    fixed_by_sequence = {clip.sequence_id: clip for clip in fixed}
    expected_ids = set(raw_by_sequence)
    if set(native_by_sequence) != expected_ids or set(fixed_by_sequence) != expected_ids:
        raise ValueError("verified native/fixed clips do not match materialization records")

    rows = tuple(
        _audit_row(
            native_by_sequence[sequence_id],
            fixed_by_sequence[sequence_id],
            raw_by_sequence[sequence_id],
        )
        for sequence_id in sorted(expected_ids, key=str.encode)
    )
    row_by_sequence = {str(row["sequence_id"]): row for row in rows}
    contrast_inventory = _contrast_inventory(fixed, row_by_sequence)
    selected_groups = tuple(
        group
        for group in contrast_inventory["groups"]
        if group["partition"] == f"train:{target_bucket}x{target_bucket}"
        and group["direction"] == direction
        and tuple(group["actions"]) == ordered_actions
    )
    if not selected_groups:
        raise ValueError("no target-distinct groups match the requested bounded subset")
    selected_identity_ids = [str(group["identity_id"]) for group in selected_groups]
    if len(selected_identity_ids) != len(set(selected_identity_ids)):
        raise ValueError("bounded selection has more than one group for an identity")
    selected_sequence_ids = tuple(
        sorted(
            (
                str(sequence_id)
                for group in selected_groups
                for sequence_id in group["sequence_ids"]
            ),
            key=str.encode,
        )
    )
    if len(selected_sequence_ids) != 2 * len(selected_groups):
        raise ValueError("each selected causal group must contribute exactly two rows")

    subset = copy.deepcopy(dict(manifest))
    subset_records = [copy.deepcopy(raw_by_sequence[value]) for value in selected_sequence_ids]
    subset["sequence_count"] = len(subset_records)
    subset["sequences"] = subset_records
    selected_rows = tuple(row_by_sequence[value] for value in selected_sequence_ids)
    selection_group_digest = _canonical_sha256(list(selected_groups))
    selection = {
        "action_counts": _counts(str(row["action"]) for row in selected_rows),
        "artifact_kind": "target_distinct_multi_identity_endpoint_training_subset",
        "causal_claim_scope": (
            "in-sample multi-identity idle-versus-walk memorization diagnostic at one "
            "fixed authored direction; not open-vocabulary or held-out generalization"
        ),
        "condition_contract": contrast_inventory["contract"],
        "direction": direction,
        "entity_class_counts": _counts(str(row["entity_class"]) for row in selected_rows),
        "fixed_target_frames": target_frames,
        "identity_count": len(selected_identity_ids),
        "selected_group_count": len(selected_groups),
        "selected_groups_sha256": selection_group_digest,
        "selected_sequence_ids_sha256": _canonical_sha256(list(selected_sequence_ids)),
        "source_materialization_canonical_sha256": manifest_canonical_sha256,
        "source_materialization_file_sha256": manifest_file_sha256,
        "source_materialization_path": str(manifest_path),
        "source_pixel_audit_file_sha256": pixels_file_sha256,
        "source_snapshot_canonical_sha256": snapshot_canonical_sha256,
        "source_snapshot_file_sha256": snapshot_file_sha256,
        "source_training_audit_file_sha256": readiness_file_sha256,
        "source_sequence_count": len(records),
        "split_counts": _counts(str(row["split"]) for row in selected_rows),
        "target_bucket": [target_bucket, target_bucket],
    }
    subset["selection"] = selection
    subset_bytes = _canonical_json_bytes(subset)
    subset_file_sha256 = hashlib.sha256(subset_bytes).hexdigest()
    subset_canonical_sha256 = _canonical_sha256(subset)

    duplicates = {
        "fixed_model_target_sha256": _duplicate_components(rows, "fixed_model_target_sha256"),
        "fixed_rgba_target_sha256": _duplicate_components(rows, "fixed_rgba_target_sha256"),
        "native_array_sha256": _duplicate_components(rows, "native_array_sha256"),
        "native_file_sha256": _duplicate_components(rows, "native_file_sha256"),
        "source_blob_sha256": _source_blob_components(rows),
    }
    design = {
        "artifact_kind": "tmwa_exact_fixed_frame_training_design_audit",
        "contrast_inventory": contrast_inventory,
        "coverage": {
            "action_counts": _counts(str(row["action"]) for row in rows),
            "bucket_counts": _counts(str(row["bucket"]) for row in rows),
            "direction_counts": _counts(str(row["direction"]) for row in rows),
            "entity_class_counts": _counts(str(row["entity_class"]) for row in rows),
            "identity_count": len({str(row["identity_id"]) for row in rows}),
            "loop_mode_counts": _counts(str(row["loop_mode"]) for row in rows),
            "sequence_count": len(rows),
            "split_counts": _counts(str(row["split"]) for row in rows),
        },
        "duplicate_components": duplicates,
        "fixed_target_contract": {
            "fixed_model_target_sha256": (
                "sha256(dtype.str\\0shape\\0 + C-order bytes) over the premultiplied "
                "float32 [-1,1] model tensor"
            ),
            "fixed_rgba_target_sha256": (
                "sha256(dtype.str\\0shape\\0 + C-order bytes) over the exact uint8 "
                "RGBA fixed-frame projection"
            ),
            "projection": "nearest authored phase, no temporal interpolation",
            "target_separation": (
                "pairwise metrics bind both exact target hashes; premultiplied RGBA MAE is "
                "measured in unit [0,1] space, and changed-pixel fraction counts a pixel "
                "when any stored uint8 RGBA channel differs"
            ),
            "target_frames": target_frames,
        },
        "input_artifacts": {
            "materialization": {
                "canonical_sha256": manifest_canonical_sha256,
                "file_sha256": manifest_file_sha256,
                "path": str(manifest_path),
            },
            "pixel_quality_audit": {
                "file_sha256": pixels_file_sha256,
                "path": str(pixels_path),
                "schema_version": pixels.get("schema_version"),
            },
            "source_snapshot": {
                "canonical_sha256": snapshot_canonical_sha256,
                "file_sha256": snapshot_file_sha256,
                "manifest_sha256": snapshot.get("manifest_sha256"),
                "path": str(snapshot_path),
                "schema_version": snapshot.get("schema_version"),
            },
            "training_readiness_audit": {
                "file_sha256": readiness_file_sha256,
                "path": str(readiness_path),
                "schema_version": readiness.get("schema_version"),
            },
        },
        "provenance_inventory": _provenance_inventory(rows),
        "rows": list(rows),
        "schema_version": 1,
        "selection": {
            **selection,
            "groups": list(selected_groups),
            "identity_ids": sorted(selected_identity_ids, key=str.encode),
            "sequence_ids": list(selected_sequence_ids),
            "subset_manifest_canonical_sha256": subset_canonical_sha256,
            "subset_manifest_file_sha256": subset_file_sha256,
            "subset_manifest_path": str(subset_path),
        },
        "verification": {
            "all_materialized_files_and_arrays_hash_verified": True,
            "fixed_clip_count": len(fixed),
            "live_database_read": False,
            "native_clip_count": len(native),
            "pixel_mutation": False,
            "source_manifest_changed": False,
        },
    }
    return TMWATrainingDesignDocuments(
        subset_manifest=subset,
        design_audit=design,
        selected_sequence_ids=selected_sequence_ids,
    )


def export_tmwa_training_design(
    source_snapshot_path: Path | str,
    materialization_path: Path | str,
    training_audit_path: Path | str,
    pixel_audit_path: Path | str,
    *,
    subset_manifest_path: Path | str,
    design_audit_path: Path | str,
    target_frames: int = 8,
    target_bucket: int = 64,
    direction: str = "down",
    actions: tuple[str, str] = ("idle", "walk"),
    disk_guard: DiskGuard | None = None,
) -> TMWATrainingDesignResult:
    """Build and atomically publish a no-clobber subset and design audit."""

    subset_path = Path(subset_manifest_path).resolve()
    design_path = Path(design_audit_path).resolve()
    if design_path.suffix.casefold() != ".json":
        raise ValueError("design_audit_path must end in .json")
    existing = next((path for path in (subset_path, design_path) if path.exists()), None)
    if existing is not None:
        raise FileExistsError(f"Refusing to replace existing TMWA design artifact: {existing}")
    documents = build_tmwa_training_design(
        source_snapshot_path,
        materialization_path,
        training_audit_path,
        pixel_audit_path,
        subset_manifest_path=subset_path,
        target_frames=target_frames,
        target_bucket=target_bucket,
        direction=direction,
        actions=actions,
    )
    subset_bytes = _canonical_json_bytes(documents.subset_manifest)
    design_bytes = _canonical_json_bytes(documents.design_audit)
    _atomic_write_no_clobber(subset_path, subset_bytes, disk_guard=disk_guard)
    _atomic_write_no_clobber(design_path, design_bytes, disk_guard=disk_guard)
    return TMWATrainingDesignResult(
        subset_manifest_path=subset_path,
        subset_manifest_sha256=hashlib.sha256(subset_bytes).hexdigest(),
        subset_manifest_canonical_sha256=_canonical_sha256(documents.subset_manifest),
        design_audit_path=design_path,
        design_audit_sha256=hashlib.sha256(design_bytes).hexdigest(),
        selected_sequence_count=len(documents.selected_sequence_ids),
    )


def _verify_upstream_contracts(
    snapshot: Mapping[str, Any],
    manifest: Mapping[str, Any],
    readiness: Mapping[str, Any],
    pixels: Mapping[str, Any],
    *,
    snapshot_canonical_sha256: str,
    manifest_file_sha256: str,
    sequence_count: int,
    target_frames: int,
) -> None:
    source_snapshot = manifest.get("source_snapshot")
    if not isinstance(source_snapshot, Mapping):
        raise ValueError("materialization source_snapshot must be an object")
    if source_snapshot.get("canonical_sha256") != snapshot_canonical_sha256:
        raise ValueError("materialization is not bound to the supplied snapshot")
    if source_snapshot.get("manifest_sha256") != snapshot.get("manifest_sha256"):
        raise ValueError("materialization and snapshot dataset-manifest hashes differ")
    readiness_manifest = readiness.get("manifest")
    if not isinstance(readiness_manifest, Mapping):
        raise ValueError("training-readiness audit lacks a manifest binding")
    if readiness.get("schema_version") != 2:
        raise ValueError("training-readiness audit must use schema version 2")
    if readiness_manifest.get("file_sha256") != manifest_file_sha256:
        raise ValueError("training-readiness audit is not bound to this materialization")
    if readiness_manifest.get("sequence_count") != sequence_count:
        raise ValueError("training-readiness sequence count differs from materialization")
    projection = readiness.get("fixed_frame_projection")
    if not isinstance(projection, Mapping) or projection.get("target_frames") != target_frames:
        raise ValueError("training-readiness audit uses a different fixed-frame projection")
    pixels_manifest = pixels.get("manifest")
    verification = pixels.get("verification")
    if not isinstance(pixels_manifest, Mapping) or not isinstance(verification, Mapping):
        raise ValueError("pixel-quality audit lacks verification bindings")
    if pixels_manifest.get("file_sha256") != manifest_file_sha256:
        raise ValueError("pixel-quality audit is not bound to this materialization")
    if verification.get("verified_clip_count") != sequence_count:
        raise ValueError("pixel-quality audit did not verify every materialized clip")


def _audit_row(
    native: MaterializedTrainingClip,
    fixed: MaterializedTrainingClip,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    if native.sequence_id != fixed.sequence_id:
        raise ValueError("native and fixed clip sequence IDs differ")
    output = raw.get("output")
    provenance = raw.get("provenance")
    bucket = raw.get("target_bucket")
    if not isinstance(output, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError(f"sequence {native.sequence_id!r} lacks output/provenance")
    if (
        not isinstance(bucket, list)
        or len(bucket) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bucket)
    ):
        raise ValueError(f"sequence {native.sequence_id!r} has an invalid target bucket")
    archive_occurrences = provenance.get("archive_occurrences")
    if not isinstance(archive_occurrences, list):
        raise ValueError(f"sequence {native.sequence_id!r} lacks archive occurrences")
    source_keys = provenance.get("sequence_source_keys")
    if not isinstance(source_keys, list) or not source_keys:
        raise ValueError(f"sequence {native.sequence_id!r} lacks sequence source keys")
    return {
        "action": fixed.request.action,
        "archive_occurrences": copy.deepcopy(archive_occurrences),
        "bucket": f"{bucket[0]}x{bucket[1]}",
        "description": fixed.request.description,
        "direction": fixed.request.direction,
        "entity_class": fixed.request.entity_class,
        "fixed_duration_ms": list(fixed.duration_ms),
        "fixed_frame_phases": list(fixed.frame_phases),
        "fixed_model_target_sha256": _array_sha256(fixed.model_array),
        "fixed_rgba_target_sha256": _array_sha256(fixed.rgba),
        "fixed_shape": list(fixed.rgba.shape),
        "identity_id": fixed.identity_id,
        "item_blob_occurrence_ids": copy.deepcopy(provenance.get("item_blob_occurrence_ids", [])),
        "loop_mode": fixed.request.loop_mode,
        "native_array_sha256": str(output.get("array_content_sha256")),
        "native_file_sha256": str(output.get("file_sha256")),
        "native_relative_path": str(output.get("relative_path")),
        "native_shape": list(native.rgba.shape),
        "provenance_sha256": _canonical_sha256(provenance),
        "retrieval_ids": copy.deepcopy(provenance.get("retrieval_ids", [])),
        "rights_observation_ids": copy.deepcopy(provenance.get("rights_observation_ids", [])),
        "sequence_id": fixed.sequence_id,
        "sequence_source_keys": copy.deepcopy(source_keys),
        "source_blob_sha256": list(fixed.source_blob_sha256),
        "source_id": fixed.source_id,
        "source_pack_id": provenance.get("source_pack_id"),
        "split": fixed.split,
        "view": fixed.request.view,
    }


def _contrast_inventory(
    clips: Sequence[MaterializedTrainingClip],
    row_by_sequence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    partitions: dict[str, list[MaterializedTrainingClip]] = defaultdict(list)
    for clip in clips:
        row = row_by_sequence[clip.sequence_id]
        partitions[f"{clip.split}:{row['bucket']}"].append(clip)
    summaries: dict[str, Any] = {}
    groups: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for partition, selected in sorted(partitions.items(), key=lambda item: item[0].encode()):
        candidates: dict[tuple[Any, ...], list[MaterializedTrainingClip]] = defaultdict(list)
        for clip in selected:
            request = clip.request
            candidates[
                (
                    clip.identity_id,
                    request.description,
                    request.entity_class,
                    request.view,
                    request.direction,
                    request.loop_mode,
                    tuple(float(value) for value in clip.frame_phases),
                )
            ].append(clip)
        partition_groups: list[dict[str, Any]] = []
        partition_exclusions: list[dict[str, Any]] = []
        nominal_multi_action_count = 0
        for key, candidate_rows in sorted(
            candidates.items(), key=lambda item: _canonical_json_bytes(_condition_payload(item[0]))
        ):
            by_action: dict[str, list[MaterializedTrainingClip]] = defaultdict(list)
            for clip in candidate_rows:
                by_action[clip.request.action].append(clip)
            if len(by_action) < 2:
                continue
            nominal_multi_action_count += 1
            representatives: list[tuple[str, MaterializedTrainingClip, str]] = []
            for action, action_rows in sorted(
                by_action.items(), key=lambda item: item[0].encode("utf-8")
            ):
                ordered = sorted(action_rows, key=lambda clip: clip.sequence_id.encode("utf-8"))
                digests = {
                    str(row_by_sequence[clip.sequence_id]["fixed_model_target_sha256"])
                    for clip in ordered
                }
                if len(digests) > 1:
                    partition_exclusions.extend(
                        _exclusion(
                            clip,
                            row_by_sequence,
                            "conflicting_targets_for_identical_action_and_non_action_conditions",
                        )
                        for clip in ordered
                    )
                    continue
                digest = next(iter(digests))
                representatives.append((action, ordered[0], digest))
                partition_exclusions.extend(
                    _exclusion(
                        clip,
                        row_by_sequence,
                        "byte_identical_duplicate_target_uses_one_representative",
                    )
                    for clip in ordered[1:]
                )
            by_target: dict[str, list[tuple[str, MaterializedTrainingClip, str]]] = defaultdict(
                list
            )
            for representative in representatives:
                by_target[representative[2]].append(representative)
            distinct_representatives: list[tuple[str, MaterializedTrainingClip, str]] = []
            for _target_sha256, target_rows in sorted(by_target.items()):
                ordered_target = sorted(target_rows, key=lambda row: row[0].encode("utf-8"))
                distinct_representatives.append(ordered_target[0])
                partition_exclusions.extend(
                    _exclusion(
                        clip,
                        row_by_sequence,
                        "byte_identical_cross_action_target_uses_one_representative",
                    )
                    for _action, clip, _digest in ordered_target[1:]
                )
            distinct_representatives.sort(key=lambda row: row[0].encode("utf-8"))
            if len(distinct_representatives) < 2:
                partition_exclusions.extend(
                    _exclusion(
                        clip,
                        row_by_sequence,
                        "no_target_distinct_multi_action_contrast_after_conflict_and_alias_filter",
                    )
                    for _action, clip, _digest in distinct_representatives
                )
                continue
            condition = _condition_payload(key)
            group = {
                **condition,
                "actions": [row[0] for row in distinct_representatives],
                "fixed_model_target_sha256": [row[2] for row in distinct_representatives],
                "fixed_rgba_target_sha256": [
                    row_by_sequence[row[1].sequence_id]["fixed_rgba_target_sha256"]
                    for row in distinct_representatives
                ],
                "key_sha256": _canonical_sha256(condition),
                "partition": partition,
                "sequence_ids": [row[1].sequence_id for row in distinct_representatives],
                "target_separation": _pairwise_target_separation(
                    distinct_representatives,
                    row_by_sequence,
                ),
            }
            partition_groups.append(group)
        partition_groups.sort(key=lambda row: str(row["key_sha256"]))
        partition_exclusions.sort(
            key=lambda row: (str(row["sequence_id"]).encode("utf-8"), str(row["reason"]))
        )
        reasons = Counter(str(row["reason"]) for row in partition_exclusions)
        summaries[partition] = {
            "condition_count": len(candidates),
            "contrast_group_count": len(partition_groups),
            "endpoint_exclusion_count": len(partition_exclusions),
            "exclusion_reason_counts": dict(sorted(reasons.items())),
            "nominal_multi_action_condition_count": nominal_multi_action_count,
            "selected_representative_count": sum(
                len(group["sequence_ids"]) for group in partition_groups
            ),
            "sequence_count": len(selected),
        }
        groups.extend(partition_groups)
        exclusions.extend(partition_exclusions)
    groups.sort(key=lambda row: (str(row["partition"]).encode(), str(row["key_sha256"])))
    exclusions.sort(key=lambda row: (str(row["sequence_id"]).encode("utf-8"), str(row["reason"])))
    return {
        "contract": (
            "identity, description, entity class, view, direction, loop mode, and exact "
            "fixed-frame phases match; only action may vary; each action must have one "
            "unambiguous target and selected actions must have distinct exact model targets"
        ),
        "exclusions": exclusions,
        "groups": groups,
        "partitions": summaries,
    }


def _condition_payload(key: tuple[Any, ...]) -> dict[str, Any]:
    return {
        "description": key[1],
        "direction": key[4],
        "entity_class": key[2],
        "fixed_frame_phases": list(key[6]),
        "identity_id": key[0],
        "loop_mode": key[5],
        "view": key[3],
    }


def _pairwise_target_separation(
    representatives: Sequence[tuple[str, MaterializedTrainingClip, str]],
    row_by_sequence: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for left_index, (left_action, left_clip, left_model_hash) in enumerate(representatives):
        for right_action, right_clip, right_model_hash in representatives[left_index + 1 :]:
            left_model = left_clip.model_array
            right_model = right_clip.model_array
            left_rgba = left_clip.rgba
            right_rgba = right_clip.rgba
            if left_model.shape != right_model.shape or left_rgba.shape != right_rgba.shape:
                raise ValueError("causal contrast targets must share one exact tensor shape")
            pairs.append(
                {
                    "actions": [left_action, right_action],
                    "changed_rgba_pixel_fraction": float(
                        np.any(left_rgba != right_rgba, axis=-1).mean()
                    ),
                    "fixed_model_target_sha256": [left_model_hash, right_model_hash],
                    "fixed_rgba_target_sha256": [
                        row_by_sequence[left_clip.sequence_id]["fixed_rgba_target_sha256"],
                        row_by_sequence[right_clip.sequence_id]["fixed_rgba_target_sha256"],
                    ],
                    "premultiplied_rgba_unit_mae": float(
                        np.abs(left_model - right_model).mean(dtype=np.float64) / 2.0
                    ),
                    "sequence_ids": [left_clip.sequence_id, right_clip.sequence_id],
                }
            )
    return pairs


def _exclusion(
    clip: MaterializedTrainingClip,
    row_by_sequence: Mapping[str, Mapping[str, Any]],
    reason: str,
) -> dict[str, Any]:
    row = row_by_sequence[clip.sequence_id]
    return {
        "action": clip.request.action,
        "fixed_model_target_sha256": row["fixed_model_target_sha256"],
        "fixed_rgba_target_sha256": row["fixed_rgba_target_sha256"],
        "identity_id": clip.identity_id,
        "reason": reason,
        "sequence_id": clip.sequence_id,
    }


def _duplicate_components(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    by_digest: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_digest[str(row[key])].append(row)
    components = [
        _component(digest, members)
        for digest, members in sorted(by_digest.items())
        if len(members) > 1
    ]
    return {
        "component_count": len(components),
        "components": components,
        "duplicate_row_count": sum(len(row["sequence_ids"]) for row in components),
        "excess_row_count": sum(len(row["sequence_ids"]) - 1 for row in components),
        "unique_value_count": len(by_digest),
    }


def _source_blob_components(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_digest: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for digest in row["source_blob_sha256"]:
            by_digest[str(digest)].append(row)
    components = [
        _component(digest, members)
        for digest, members in sorted(by_digest.items())
        if len(members) > 1
    ]
    return {
        "component_count": len(components),
        "components": components,
        "excess_row_count": sum(len(row["sequence_ids"]) - 1 for row in components),
        "unique_value_count": len(by_digest),
    }


def _component(digest: str, members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(members, key=lambda row: str(row["sequence_id"]).encode("utf-8"))
    return {
        "action_counts": _counts(str(row["action"]) for row in ordered),
        "digest": digest,
        "direction_counts": _counts(str(row["direction"]) for row in ordered),
        "identity_ids": sorted({str(row["identity_id"]) for row in ordered}, key=str.encode),
        "loop_mode_counts": _counts(str(row["loop_mode"]) for row in ordered),
        "sequence_ids": [str(row["sequence_id"]) for row in ordered],
        "splits": sorted({str(row["split"]) for row in ordered}, key=str.encode),
    }


def _provenance_inventory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    role_counts: Counter[str] = Counter()
    role_paths: dict[str, set[str]] = defaultdict(set)
    retrieval_ids: set[str] = set()
    rights_ids: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    for row in rows:
        retrieval_ids.update(str(value) for value in row["retrieval_ids"])
        rights_ids.update(str(value) for value in row["rights_observation_ids"])
        for key in row["sequence_source_keys"]:
            if isinstance(key, Mapping):
                source_keys.add((str(key.get("source_id")), str(key.get("external_sequence_key"))))
        for occurrence in row["archive_occurrences"]:
            if not isinstance(occurrence, Mapping):
                raise ValueError("archive occurrence rows must be objects")
            role = str(occurrence.get("occurrence_role"))
            path = str(occurrence.get("member_path"))
            role_counts[role] += 1
            role_paths[role].add(path)
    return {
        "archive_occurrence_reference_counts": dict(sorted(role_counts.items())),
        "archive_occurrence_unique_member_counts": {
            role: len(paths) for role, paths in sorted(role_paths.items())
        },
        "retrieval_id_count": len(retrieval_ids),
        "rights_observation_id_count": len(rights_ids),
        "sequence_source_key_count": len(source_keys),
        "source_counts": _counts(str(row["source_id"]) for row in rows),
        "source_pack_counts": _counts(str(row["source_pack_id"]) for row in rows),
    }


def _records_by_sequence(records: Sequence[object]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("materialization sequence records must be objects")
        sequence_id = raw.get("sequence_id")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("materialization sequence IDs must be non-empty strings")
        if sequence_id in result:
            raise ValueError(f"duplicate materialization sequence ID: {sequence_id!r}")
        result[sequence_id] = raw
    return result


def _read_object(path: Path, label: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        payload = path.read_bytes()
        decoded = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(decoded, Mapping):
        raise ValueError(f"{label} root must be an object")
    return payload, decoded


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items(), key=lambda item: item[0].encode("utf-8")))


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value, newline=False)).hexdigest()


def _canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + suffix
    ).encode("utf-8")


def _positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _atomic_write_no_clobber(
    path: Path,
    payload: bytes,
    *,
    disk_guard: DiskGuard | None,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing TMWA design artifact: {path}")
    if disk_guard is not None:
        disk_guard.require_capacity(len(payload) + 65_536, label="TMWA training design")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to replace existing TMWA design artifact: {path}"
            ) from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
