"""Hash-verified materialization coverage and split-leakage audit."""

from __future__ import annotations

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
class TrainingAuditResult:
    artifact_path: Path
    artifact_sha256: str
    sequence_count: int
    fixed_frame_count: int


def build_materialization_training_audit(
    manifest_path: Path | str,
    *,
    target_frames: int = 8,
) -> dict[str, Any]:
    """Return an exact coverage audit after verifying every declared clip and hash."""

    if isinstance(target_frames, bool) or not isinstance(target_frames, int) or target_frames < 1:
        raise ValueError("target_frames must be a positive integer")
    manifest = Path(manifest_path).resolve()
    manifest_bytes = manifest.read_bytes()
    try:
        raw = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot decode materialization manifest: {manifest}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("materialization manifest root must be an object")
    records = raw.get("sequences")
    if not isinstance(records, list) or not records:
        raise ValueError("materialization manifest must contain sequences")
    declared = raw.get("sequence_count")
    if declared != len(records):
        raise ValueError(f"sequence_count mismatch: declared {declared!r}, found {len(records)}")

    native = load_materialized_training_clips(manifest, split=None)
    fixed = load_materialized_training_clips(
        manifest,
        split=None,
        target_frames=target_frames,
    )
    if len(native) != len(records) or len(fixed) != len(records):
        raise ValueError("verified clip count does not match the manifest")
    by_sequence = {clip.sequence_id: clip for clip in fixed}
    raw_by_sequence = _records_by_sequence(records)
    if set(by_sequence) != set(raw_by_sequence):
        raise ValueError("loaded sequence IDs do not match manifest sequence IDs")

    rows = tuple(
        _audit_row(by_sequence[sequence_id], raw_by_sequence[sequence_id])
        for sequence_id in sorted(by_sequence, key=str.encode)
    )
    split_names = tuple(sorted({row["split"] for row in rows}, key=str.encode))
    split_summary = {
        split: _coverage_summary(tuple(row for row in rows if row["split"] == split))
        for split in split_names
    }
    leakage = _split_overlap_audit(rows)
    endpoint = _endpoint_contrast_audit(fixed, raw_by_sequence)
    snapshot = raw.get("source_snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("source_snapshot must be an object")
    return {
        "artifact_kind": "materialized_training_readiness_audit",
        "coverage": _coverage_summary(rows),
        "endpoint_action_contrasts": endpoint,
        "fixed_frame_projection": {
            "projected_sequence_count": len(fixed),
            "target_frames": target_frames,
        },
        "manifest": {
            "file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "path": str(manifest),
            "schema_version": raw.get("schema_version"),
            "sequence_count": len(records),
        },
        "schema_version": 2,
        "source_snapshot": dict(snapshot),
        "split_leakage_audit": leakage,
        "splits": split_summary,
    }


def export_materialization_training_audit(
    manifest_path: Path | str,
    output_path: Path | str,
    *,
    target_frames: int = 8,
    disk_guard: DiskGuard | None = None,
) -> TrainingAuditResult:
    """Build and atomically publish a canonical no-clobber JSON audit."""

    artifact = build_materialization_training_audit(
        manifest_path,
        target_frames=target_frames,
    )
    output = Path(output_path).resolve()
    if output.suffix.casefold() != ".json":
        raise ValueError("output_path must end in .json")
    payload = (
        json.dumps(
            artifact,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_write_no_clobber(output, payload, disk_guard=disk_guard)
    return TrainingAuditResult(
        artifact_path=output,
        artifact_sha256=hashlib.sha256(payload).hexdigest(),
        sequence_count=int(artifact["manifest"]["sequence_count"]),
        fixed_frame_count=int(artifact["fixed_frame_projection"]["projected_sequence_count"]),
    )


def _records_by_sequence(records: Sequence[object]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("sequence records must be objects")
        sequence_id = raw.get("sequence_id")
        if not isinstance(sequence_id, str) or not sequence_id:
            raise ValueError("sequence_id must be a non-empty string")
        if sequence_id in result:
            raise ValueError(f"duplicate sequence_id: {sequence_id!r}")
        result[sequence_id] = raw
    return result


def _audit_row(
    clip: MaterializedTrainingClip,
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = raw.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"sequence {clip.sequence_id!r} lacks provenance")
    pack = provenance.get("source_pack_id")
    if pack is not None and (not isinstance(pack, str) or not pack):
        raise ValueError(f"invalid source_pack_id for {clip.sequence_id!r}")
    bucket = raw.get("target_bucket")
    if (
        not isinstance(bucket, list)
        or len(bucket) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bucket)
    ):
        raise ValueError(f"invalid target_bucket for {clip.sequence_id!r}")
    return {
        "action": clip.request.action,
        "array_sha256": clip.source_array_sha256,
        "bucket": f"{bucket[0]}x{bucket[1]}",
        "description": clip.request.description,
        "direction": clip.request.direction,
        "entity_class": clip.request.entity_class,
        "fixed_target_array_sha256": _array_sha256(clip.rgba),
        "identity_id": clip.identity_id,
        "loop_mode": clip.request.loop_mode,
        "sequence_id": clip.sequence_id,
        "source_blob_sha256": tuple(sorted(clip.source_blob_sha256)),
        "source_id": clip.source_id,
        "source_loop_mode": clip.source_loop_mode,
        "source_pack_id": pack,
        "split": clip.split,
        "view": clip.request.view,
    }


def _coverage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        identities[str(row["identity_id"])].add(str(row["action"]))
    return {
        "action_counts": _counts(str(row["action"]) for row in rows),
        "bucket_counts": _counts(str(row["bucket"]) for row in rows),
        "direction_counts": _counts(str(row["direction"]) for row in rows),
        "entity_action_counts": _counts(f"{row['entity_class']}:{row['action']}" for row in rows),
        "entity_class_counts": _counts(str(row["entity_class"]) for row in rows),
        "identity_count": len(identities),
        "loop_mode_counts": _counts(str(row["loop_mode"]) for row in rows),
        "multi_action_identity_count": sum(len(actions) >= 2 for actions in identities.values()),
        "sequence_count": len(rows),
        "source_counts": _counts(str(row["source_id"]) for row in rows),
        "source_loop_mode_counts": _counts(str(row["source_loop_mode"]) for row in rows),
        "source_pack_counts": _counts(
            str(row["source_pack_id"]) for row in rows if row["source_pack_id"] is not None
        ),
        "view_counts": _counts(str(row["view"]) for row in rows),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items(), key=lambda item: item[0].encode("utf-8")))


def _split_overlap_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "array_sha256": _overlaps(rows, "array_sha256"),
        "fixed_target_array_sha256": _overlaps(rows, "fixed_target_array_sha256"),
        "identity_id": _overlaps(rows, "identity_id"),
        "source_blob_sha256": _multi_value_overlaps(rows, "source_blob_sha256"),
        "source_pack_id": _overlaps(rows, "source_pack_id", omit_none=True),
        "interpretation": {
            "array_sha256": "exact materialized arrays appearing in more than one split",
            "fixed_target_array_sha256": (
                "exact fixed-frame training tensors appearing in more than one split"
            ),
            "identity_id": "subject identities appearing in more than one split",
            "source_blob_sha256": "source image/carrier blobs appearing in more than one split",
            "source_pack_id": (
                "source packs appearing in more than one split; this is a style/domain "
                "overlap even when identity and exact art remain disjoint"
            ),
        },
    }


def _overlaps(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    *,
    omit_none: bool = False,
) -> list[dict[str, Any]]:
    by_value: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row[key]
        if value is None and omit_none:
            continue
        by_value[str(value)].add(str(row["split"]))
    return [
        {"value": value, "splits": sorted(splits, key=str.encode)}
        for value, splits in sorted(by_value.items(), key=lambda item: item[0].encode("utf-8"))
        if len(splits) > 1
    ]


def _multi_value_overlaps(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    by_value: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for value in row[key]:
            by_value[str(value)].add(str(row["split"]))
    return [
        {"value": value, "splits": sorted(splits, key=str.encode)}
        for value, splits in sorted(by_value.items(), key=lambda item: item[0].encode("utf-8"))
        if len(splits) > 1
    ]


def _endpoint_contrast_audit(
    clips: Sequence[MaterializedTrainingClip],
    raw_by_sequence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_bucket_split: dict[tuple[str, str], list[MaterializedTrainingClip]] = defaultdict(list)
    for clip in clips:
        raw = raw_by_sequence[clip.sequence_id]
        bucket = raw["target_bucket"]
        by_bucket_split[(f"{bucket[0]}x{bucket[1]}", clip.split)].append(clip)
    summaries: dict[str, Any] = {}
    for (bucket, split), selected in sorted(by_bucket_split.items()):
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
        group_count = 0
        selected_count = 0
        duplicate_count = 0
        conflict_count = 0
        cross_action_alias_count = 0
        no_target_distinct_count = 0
        for rows in candidates.values():
            by_action: dict[str, list[MaterializedTrainingClip]] = defaultdict(list)
            for clip in rows:
                by_action[clip.request.action].append(clip)
            if len(by_action) < 2:
                continue
            representatives: list[tuple[str, str]] = []
            for action, action_rows in sorted(
                by_action.items(), key=lambda item: item[0].encode("utf-8")
            ):
                hashes = {_array_sha256(clip.rgba) for clip in action_rows}
                if len(hashes) > 1:
                    conflict_count += len(action_rows)
                    continue
                duplicate_count += max(0, len(action_rows) - 1)
                representatives.append((action, next(iter(hashes))))
            by_target: dict[str, list[str]] = defaultdict(list)
            for action, target_sha256 in representatives:
                by_target[target_sha256].append(action)
            target_distinct_count = len(by_target)
            cross_action_alias_count += sum(
                max(0, len(actions) - 1) for actions in by_target.values()
            )
            if target_distinct_count >= 2:
                group_count += 1
                selected_count += target_distinct_count
            else:
                no_target_distinct_count += target_distinct_count
        summaries[f"{split}:{bucket}"] = {
            "conflicting_same_action_rows": conflict_count,
            "contrast_group_count": group_count,
            "cross_action_alias_rows_omitted": cross_action_alias_count,
            "duplicate_target_rows_omitted": duplicate_count,
            "endpoint_excluded_row_count": (
                conflict_count
                + cross_action_alias_count
                + duplicate_count
                + no_target_distinct_count
            ),
            "no_target_distinct_rows_omitted": no_target_distinct_count,
            "selected_representative_count": selected_count,
            "sequence_count": len(selected),
        }
    return {
        "contract": (
            "groups match identity, description, entity, view, direction, loop mode, "
            "and exact fixed-frame phases; only action may vary, and selected actions "
            "must have distinct fixed-frame target hashes"
        ),
        "partitions": summaries,
    }


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _atomic_write_no_clobber(
    path: Path,
    payload: bytes,
    *,
    disk_guard: DiskGuard | None,
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace existing training audit: {path}")
    if disk_guard is not None:
        disk_guard.require_capacity(len(payload) + 65_536, label="training audit")
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
            raise FileExistsError(f"Refusing to replace existing training audit: {path}") from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
