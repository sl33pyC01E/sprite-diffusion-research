from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from spritelab.dataset import (
    CoverageReport,
    DatasetManifest,
    SequenceSample,
    SplitPolicy,
    build_dataset_manifest,
    coverage_report,
)
from spritelab.temporal import select_temporal_frames

SNAPSHOT_SCHEMA_VERSION = 1
TEMPORAL_KNOWN_CONTRACT = (
    "frame_count>1; every ordinal has a positive recorded duration in one frame table; "
    "no source annotation explicitly says timing or state order is unknown"
)

TemporalMode = Literal["known", "model_ready", "pose_only", "all"]


@dataclass(frozen=True, slots=True)
class SnapshotFilters:
    """Selection rules applied before deterministic leakage-aware splitting.

    The conservative default excludes single poses and unordered pose projections. Set
    ``temporal_mode='all'`` to build a spatial-pose dataset instead.
    """

    minimum_frame_count: int = 2
    actions: tuple[str, ...] = ()
    temporal_mode: TemporalMode = "known"
    include_source_ids: tuple[str, ...] = ()
    exclude_source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.minimum_frame_count < 1:
            raise ValueError("minimum_frame_count must be positive")
        if self.temporal_mode not in {"known", "model_ready", "pose_only", "all"}:
            raise ValueError(f"Unknown temporal mode: {self.temporal_mode!r}")
        actions = tuple(
            sorted(
                {action.strip().casefold() for action in self.actions if action.strip()},
                key=lambda value: value.encode("utf-8"),
            )
        )
        object.__setattr__(self, "actions", actions)
        include_source_ids = _normalize_source_ids(
            self.include_source_ids,
            field_name="include_source_ids",
        )
        exclude_source_ids = _normalize_source_ids(
            self.exclude_source_ids,
            field_name="exclude_source_ids",
        )
        overlap = set(include_source_ids).intersection(exclude_source_ids)
        if overlap:
            rendered = ", ".join(sorted(overlap, key=lambda value: value.encode("utf-8")))
            raise ValueError(f"Source IDs cannot be both included and excluded: {rendered}")
        object.__setattr__(self, "include_source_ids", include_source_ids)
        object.__setattr__(self, "exclude_source_ids", exclude_source_ids)


def _normalize_source_ids(values: Iterable[str], *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise ValueError(f"{field_name} must be an iterable of source IDs, not a string")
    try:
        candidates = tuple(values)
    except TypeError as error:
        raise ValueError(f"{field_name} must be an iterable of source IDs") from error
    if any(not isinstance(value, str) for value in candidates):
        raise ValueError(f"{field_name} must contain only strings")
    return tuple(
        sorted(
            {value.strip() for value in candidates if value.strip()},
            key=lambda value: value.encode("utf-8"),
        )
    )


def _filter_record(filters: SnapshotFilters) -> dict[str, Any]:
    """Serialize new optional filters without changing legacy default snapshot bytes."""

    record = asdict(filters)
    if not filters.include_source_ids:
        record.pop("include_source_ids")
    if not filters.exclude_source_ids:
        record.pop("exclude_source_ids")
    return record


@dataclass(frozen=True, slots=True)
class SnapshotArtifact:
    """A canonical, timestamp-free export derived from one read transaction."""

    schema_version: int
    filters: SnapshotFilters
    temporal_known_contract: str
    index_schema_versions: tuple[int, ...]
    manifest: DatasetManifest
    coverage: CoverageReport
    timing_counts: dict[str, int]

    @property
    def canonical_json(self) -> str:
        payload = {
            "coverage": asdict(self.coverage),
            "filters": _filter_record(self.filters),
            "index_schema_versions": self.index_schema_versions,
            "manifest": asdict(self.manifest),
            "manifest_sha256": self.manifest.sha256,
            "schema_version": self.schema_version,
            "temporal_known_contract": self.temporal_known_contract,
            "timing_counts": self.timing_counts,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@contextmanager
def _read_connection(database_path: Path | str) -> Iterator[sqlite3.Connection]:
    path = Path(database_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Index database does not exist: {path}")
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("BEGIN")
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def _decode_json(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {"_invalid_json": True, "_raw": value}


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _candidate_rows(
    connection: sqlite3.Connection,
    filters: SnapshotFilters,
) -> list[sqlite3.Row]:
    action_expression = (
        "LOWER(COALESCE(NULLIF(m.normalized_action, ''), NULLIF(s.action, ''), 'unknown'))"
    )
    clauses = ["s.frame_count >= ?"]
    parameters: list[Any] = [filters.minimum_frame_count]
    if filters.actions:
        placeholders = ",".join("?" for _ in filters.actions)
        clauses.append(f"{action_expression} IN ({placeholders})")
        parameters.extend(filters.actions)
    for source_ids, excluded in (
        (filters.include_source_ids, False),
        (filters.exclude_source_ids, True),
    ):
        if not source_ids:
            continue
        placeholders = ",".join("?" for _ in source_ids)
        source_association = f"""(
            (i.source_id IS NOT NULL AND i.source_id IN ({placeholders}))
            OR EXISTS (
                SELECT 1 FROM sequence_source_keys AS source_key
                WHERE source_key.sequence_id=s.id
                  AND source_key.source_id IN ({placeholders})
            )
            OR EXISTS (
                SELECT 1
                FROM sequence_subjects AS subject
                JOIN entities AS entity ON entity.id=subject.entity_id
                WHERE subject.sequence_id=s.id
                  AND entity.source_id IN ({placeholders})
            )
        )"""
        clauses.append(f"NOT {source_association}" if excluded else source_association)
        parameters.extend(source_ids)
        parameters.extend(source_ids)
        parameters.extend(source_ids)
    return connection.execute(
        f"""
        SELECT
            s.*,
            {action_expression} AS selected_action,
            i.source_id AS item_source_id,
            i.external_id AS item_external_id,
            i.canonical_url AS item_canonical_url,
            i.metadata_json AS item_metadata_json,
            m.vocabulary_version,
            m.source_action,
            m.normalized_action,
            m.action_family,
            m.view AS motion_view,
            m.direction AS motion_direction,
            m.loopable,
            m.cycle_frames,
            m.phase_zero_frame,
            m.confidence AS motion_confidence,
            m.annotation_method,
            m.conditioning_json
        FROM sequences AS s
        LEFT JOIN items AS i ON i.id=s.item_id
        LEFT JOIN motion_annotations AS m ON m.sequence_id=s.id
        WHERE {" AND ".join(clauses)}
        ORDER BY s.id COLLATE BINARY
        """,
        parameters,
    ).fetchall()


def _chunked(values: Sequence[str], size: int = 400) -> Iterator[tuple[str, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def _related_rows(
    connection: sqlite3.Connection,
    *,
    table_expression: str,
    sequence_column: str,
    sequence_ids: Sequence[str],
    order_by: str,
) -> list[sqlite3.Row]:
    rows: list[sqlite3.Row] = []
    for chunk in _chunked(sequence_ids):
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            connection.execute(
                f"""
                SELECT * FROM {table_expression}
                WHERE {sequence_column} IN ({placeholders})
                ORDER BY {order_by}
                """,
                chunk,
            ).fetchall()
        )
    return rows


def _item_related_rows(
    connection: sqlite3.Connection,
    *,
    table: str,
    item_ids: Sequence[str],
    order_by: str,
) -> list[sqlite3.Row]:
    rows: list[sqlite3.Row] = []
    for chunk in _chunked(item_ids):
        placeholders = ",".join("?" for _ in chunk)
        rows.extend(
            connection.execute(
                f"SELECT * FROM {table} WHERE item_id IN ({placeholders}) ORDER BY {order_by}",
                chunk,
            ).fetchall()
        )
    return rows


def _group_rows(rows: Iterable[sqlite3.Row], key: str) -> dict[str, list[sqlite3.Row]]:
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


class _BlobUnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root), key=lambda value: value.encode("utf-8"))
        self.parent[high] = low


@dataclass(frozen=True, slots=True)
class _DuplicateIndex:
    blob_group: Mapping[str, str]
    group_edge_ids: Mapping[str, tuple[str, ...]]


def _duplicate_index(connection: sqlite3.Connection) -> _DuplicateIndex:
    if not _table_exists(connection, "duplicate_edges"):
        return _DuplicateIndex({}, {})
    edges = connection.execute(
        """
        SELECT id, left_blob_sha256, right_blob_sha256
        FROM duplicate_edges
        ORDER BY id COLLATE BINARY
        """
    ).fetchall()
    union = _BlobUnionFind()
    for edge in edges:
        union.union(str(edge["left_blob_sha256"]), str(edge["right_blob_sha256"]))

    members: dict[str, list[str]] = defaultdict(list)
    for blob in union.parent:
        members[union.find(blob)].append(blob)
    root_group: dict[str, str] = {}
    for root, blobs in members.items():
        ordered = sorted(blobs, key=lambda value: value.encode("utf-8"))
        digest = hashlib.sha256("\0".join(ordered).encode("utf-8")).hexdigest()
        root_group[root] = f"duplicate-component:{digest}"
    blob_group = {blob: root_group[union.find(blob)] for blob in union.parent}

    edge_ids: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        group_id = blob_group[str(edge["left_blob_sha256"])]
        edge_ids[group_id].append(str(edge["id"]))
    return _DuplicateIndex(
        blob_group=blob_group,
        group_edge_ids={
            group: tuple(sorted(ids, key=lambda value: value.encode("utf-8")))
            for group, ids in edge_ids.items()
        },
    )


def _positive_complete_timing(rows: Sequence[sqlite3.Row], frame_count: int) -> bool:
    if len(rows) != frame_count:
        return False
    if [int(row["ordinal"]) for row in rows] != list(range(frame_count)):
        return False
    return all(row["duration_ms"] is not None and float(row["duration_ms"]) > 0 for row in rows)


def _explicit_negative_timing_claims(
    sequence_metadata: Any,
    conditioning: Any,
) -> tuple[str, ...]:
    claims: list[str] = []
    fields = ("timing_known", "exact_engine_timing", "state_occurrence_order_preserved")
    for prefix, value in (("sequence_metadata", sequence_metadata), ("conditioning", conditioning)):
        if not isinstance(value, Mapping):
            continue
        for field_name in fields:
            if value.get(field_name) is False:
                claims.append(f"{prefix}.{field_name}=false")
    return tuple(sorted(claims, key=lambda value: value.encode("utf-8")))


def _frame_record(row: sqlite3.Row, *, legacy: bool) -> dict[str, Any]:
    if legacy:
        return {
            "bbox": _decode_json(row["bbox_json"]),
            "blob_sha256": row["blob_sha256"],
            "duration_ms": row["duration_ms"],
            "metadata": _decode_json(row["metadata_json"]),
            "ordinal": row["ordinal"],
        }
    return {
        "direction": row["direction"],
        "duration_ms": row["duration_ms"],
        "metadata": _decode_json(row["metadata_json"]),
        "ordinal": row["ordinal"],
        "phase": row["phase"],
        "source_blob_sha256": row["source_blob_sha256"],
        "source_frame_index": row["source_frame_index"],
        "view": row["view"],
    }


def _temporal_evidence(
    *,
    frame_count: int,
    sequence_rows: Sequence[sqlite3.Row],
    legacy_rows: Sequence[sqlite3.Row],
    sequence_metadata: Any,
    conditioning: Any,
) -> dict[str, Any]:
    sequence_complete = _positive_complete_timing(sequence_rows, frame_count)
    legacy_complete = _positive_complete_timing(legacy_rows, frame_count)
    negatives = _explicit_negative_timing_claims(sequence_metadata, conditioning)
    duration_source: str | None = None
    if sequence_complete:
        duration_source = "sequence_frames.duration_ms"
    elif legacy_complete:
        duration_source = "frames.duration_ms"
    known = frame_count > 1 and duration_source is not None and not negatives
    return {
        "contract": TEMPORAL_KNOWN_CONTRACT,
        "duration_source": duration_source,
        "explicit_negative_claims": negatives,
        "known": known,
        "legacy_frame_row_count": len(legacy_rows),
        "sequence_frame_row_count": len(sequence_rows),
    }


def _fixed_phase_model_ready(
    *,
    frame_count: int,
    loop_mode: str,
    sequence_rows: Sequence[sqlite3.Row],
    temporal_evidence: Mapping[str, Any],
) -> bool:
    """Return whether the current fixed-phase loader can consume a sequence exactly."""

    if temporal_evidence.get("known") is not True:
        return False
    if temporal_evidence.get("duration_source") != "sequence_frames.duration_ms":
        return False
    if len(sequence_rows) != frame_count:
        return False
    phases = tuple(row["phase"] for row in sequence_rows)
    if loop_mode == "intro_then_loop":
        first_loop = next((index for index, phase in enumerate(phases) if phase is not None), None)
        if first_loop is None or first_loop == 0:
            return False
        if any(phase is None for phase in phases[first_loop:]):
            return False
        loop_phases = tuple(float(phase) for phase in phases[first_loop:])
        if not loop_phases or loop_phases[0] != 0.0:
            return False
        try:
            select_temporal_frames(
                len(loop_phases),
                len(loop_phases),
                loop_mode="loop",
                source_phases=loop_phases,
            )
        except (TypeError, ValueError):
            return False
        return True
    if loop_mode not in {"loop", "one_shot", "ping_pong"}:
        return False
    if any(phase is None for phase in phases):
        return False
    try:
        select_temporal_frames(
            frame_count,
            frame_count,
            loop_mode=loop_mode,  # type: ignore[arg-type]
            source_phases=tuple(float(phase) for phase in phases),
        )
    except (TypeError, ValueError):
        return False
    return True


def _source_blob_records(
    connection: sqlite3.Connection,
    digests: Sequence[str],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for chunk in _chunked(digests):
        placeholders = ",".join("?" for _ in chunk)
        for row in connection.execute(
            f"""
            SELECT sha256, size_bytes, mime_type, storage_path
            FROM blobs WHERE sha256 IN ({placeholders})
            ORDER BY sha256 COLLATE BINARY
            """,
            chunk,
        ):
            records[str(row["sha256"])] = {
                "mime_type": row["mime_type"],
                "sha256": row["sha256"],
                "size_bytes": row["size_bytes"],
                "storage_path": row["storage_path"],
            }
    return records


def _select_primary_subject(subjects: Sequence[sqlite3.Row]) -> sqlite3.Row | None:
    if not subjects:
        return None
    return min(
        subjects,
        key=lambda row: (
            str(row["role"]) != "primary",
            str(row["role"]).encode("utf-8"),
            str(row["entity_id"]).encode("utf-8"),
        ),
    )


def _resolved_sequence_source_ids(
    base: sqlite3.Row,
    source_keys: Sequence[sqlite3.Row],
    subjects: Sequence[sqlite3.Row],
) -> set[str]:
    source_ids = {str(row["source_id"]) for row in source_keys}
    source_ids.update(str(row["source_id"]) for row in subjects)
    if base["item_source_id"] is not None:
        source_ids.add(str(base["item_source_id"]))
    return source_ids


def _source_filter_matches(source_ids: set[str], filters: SnapshotFilters) -> bool:
    if filters.include_source_ids and source_ids.isdisjoint(filters.include_source_ids):
        return False
    return source_ids.isdisjoint(filters.exclude_source_ids)


def _load_samples(
    connection: sqlite3.Connection,
    filters: SnapshotFilters,
) -> tuple[SequenceSample, ...]:
    base_rows = _candidate_rows(connection, filters)
    if not base_rows:
        return ()
    sequence_ids = tuple(str(row["id"]) for row in base_rows)
    item_ids = tuple(
        sorted(
            {str(row["item_id"]) for row in base_rows if row["item_id"] is not None},
            key=lambda value: value.encode("utf-8"),
        )
    )

    source_keys = _group_rows(
        _related_rows(
            connection,
            table_expression="sequence_source_keys",
            sequence_column="sequence_id",
            sequence_ids=sequence_ids,
            order_by="source_id COLLATE BINARY, external_sequence_key COLLATE BINARY",
        ),
        "sequence_id",
    )
    subjects = _group_rows(
        _related_rows(
            connection,
            table_expression=("sequence_subjects AS ss JOIN entities AS e ON e.id=ss.entity_id"),
            sequence_column="ss.sequence_id",
            sequence_ids=sequence_ids,
            order_by="ss.role COLLATE BINARY, ss.entity_id COLLATE BINARY",
        ),
        "sequence_id",
    )
    sequence_frames = _group_rows(
        _related_rows(
            connection,
            table_expression="sequence_frames",
            sequence_column="sequence_id",
            sequence_ids=sequence_ids,
            order_by="sequence_id COLLATE BINARY, ordinal",
        ),
        "sequence_id",
    )
    legacy_frames = _group_rows(
        _related_rows(
            connection,
            table_expression="frames",
            sequence_column="sequence_id",
            sequence_ids=sequence_ids,
            order_by="sequence_id COLLATE BINARY, ordinal",
        ),
        "sequence_id",
    )
    occurrences = _group_rows(
        _related_rows(
            connection,
            table_expression=(
                "sequence_occurrences AS so JOIN archive_members AS am "
                "ON am.archive_blob_sha256=so.archive_blob_sha256 "
                "AND am.ordinal=so.archive_member_ordinal"
            ),
            sequence_column="so.sequence_id",
            sequence_ids=sequence_ids,
            order_by=(
                "so.archive_blob_sha256 COLLATE BINARY, so.archive_member_ordinal, "
                "so.occurrence_role COLLATE BINARY"
            ),
        ),
        "sequence_id",
    )
    retrievals = _group_rows(
        _item_related_rows(
            connection,
            table="retrievals",
            item_ids=item_ids,
            order_by="id COLLATE BINARY",
        ),
        "item_id",
    )
    rights = _group_rows(
        _item_related_rows(
            connection,
            table="rights_observations",
            item_ids=item_ids,
            order_by="id COLLATE BINARY",
        ),
        "item_id",
    )
    item_blobs = _group_rows(
        _item_related_rows(
            connection,
            table="item_blobs",
            item_ids=item_ids,
            order_by="id COLLATE BINARY",
        ),
        "item_id",
    )
    sources = {
        str(row["id"]): row
        for row in connection.execute("SELECT * FROM sources ORDER BY id COLLATE BINARY")
    }
    duplicate_index = _duplicate_index(connection)

    all_source_digests: set[str] = set()
    for base in base_rows:
        sequence_id = str(base["id"])
        if base["source_blob_sha256"] is not None:
            all_source_digests.add(str(base["source_blob_sha256"]))
        all_source_digests.update(
            str(frame["source_blob_sha256"]) for frame in sequence_frames.get(sequence_id, ())
        )
        all_source_digests.update(
            str(frame["blob_sha256"]) for frame in legacy_frames.get(sequence_id, ())
        )
    blob_records = _source_blob_records(
        connection,
        tuple(sorted(all_source_digests, key=lambda value: value.encode("utf-8"))),
    )

    samples: list[SequenceSample] = []
    for base in base_rows:
        sequence_id = str(base["id"])
        sequence_metadata = _decode_json(base["metadata_json"])
        conditioning = _decode_json(base["conditioning_json"])
        current_sequence_frames = sequence_frames.get(sequence_id, [])
        current_legacy_frames = legacy_frames.get(sequence_id, [])
        temporal = _temporal_evidence(
            frame_count=int(base["frame_count"]),
            sequence_rows=current_sequence_frames,
            legacy_rows=current_legacy_frames,
            sequence_metadata=sequence_metadata,
            conditioning=conditioning,
        )
        if filters.temporal_mode == "known" and not temporal["known"]:
            continue
        if filters.temporal_mode == "model_ready" and not _fixed_phase_model_ready(
            frame_count=int(base["frame_count"]),
            loop_mode=str(base["loop_mode"] or "unknown"),
            sequence_rows=current_sequence_frames,
            temporal_evidence=temporal,
        ):
            continue
        if filters.temporal_mode == "pose_only" and temporal["known"]:
            continue

        current_subjects = subjects.get(sequence_id, [])
        primary = _select_primary_subject(current_subjects)
        entity_ids = tuple(
            sorted(
                {str(row["entity_id"]) for row in current_subjects},
                key=lambda value: value.encode("utf-8"),
            )
        )
        identity_id = str(primary["entity_id"]) if primary else f"unassigned:{sequence_id}"
        entity_class = str(primary["entity_class"]) if primary else "unknown"

        current_source_keys = source_keys.get(sequence_id, [])
        source_ids = _resolved_sequence_source_ids(base, current_source_keys, current_subjects)
        if not _source_filter_matches(source_ids, filters):
            continue
        source_id = (
            str(base["item_source_id"])
            if base["item_source_id"] is not None
            else min(source_ids, key=lambda value: value.encode("utf-8"))
            if source_ids
            else "unresolved"
        )

        current_occurrences = occurrences.get(sequence_id, [])
        if base["item_id"] is not None:
            source_pack_id = str(base["item_id"])
        elif current_occurrences:
            source_pack_id = f"archive:{current_occurrences[0]['archive_blob_sha256']}"
        else:
            source_pack_id = f"source:{source_id}"

        source_digests: set[str] = set()
        if base["source_blob_sha256"] is not None:
            source_digests.add(str(base["source_blob_sha256"]))
        source_digests.update(str(frame["source_blob_sha256"]) for frame in current_sequence_frames)
        source_digests.update(str(frame["blob_sha256"]) for frame in current_legacy_frames)
        ordered_digests = tuple(sorted(source_digests, key=lambda value: value.encode("utf-8")))

        duplicate_groups = {
            duplicate_index.blob_group[digest]
            for digest in ordered_digests
            if digest in duplicate_index.blob_group
        }
        leakage_group_ids = {f"entity:{entity_id}" for entity_id in entity_ids}
        leakage_group_ids.update(duplicate_groups)
        duplicate_edge_ids = sorted(
            {
                edge_id
                for group_id in duplicate_groups
                for edge_id in duplicate_index.group_edge_ids[group_id]
            },
            key=lambda value: value.encode("utf-8"),
        )

        item_id = None if base["item_id"] is None else str(base["item_id"])
        source = sources.get(source_id)
        metadata = {
            "archive_occurrences": [
                {
                    "archive_blob_sha256": row["archive_blob_sha256"],
                    "archive_member_ordinal": row["archive_member_ordinal"],
                    "member_path": row["normalized_path"],
                    "occurrence_role": row["occurrence_role"],
                }
                for row in current_occurrences
            ],
            "blob_records": [blob_records[digest] for digest in ordered_digests],
            "duplicate_edge_ids": duplicate_edge_ids,
            "frame_provenance": [
                _frame_record(row, legacy=False) for row in current_sequence_frames
            ],
            "item": (
                None
                if item_id is None
                else {
                    "canonical_url": base["item_canonical_url"],
                    "external_id": base["item_external_id"],
                    "id": item_id,
                    "metadata": _decode_json(base["item_metadata_json"]),
                }
            ),
            "item_blob_occurrence_ids": [
                str(row["id"]) for row in item_blobs.get(item_id or "", [])
            ],
            "legacy_frame_provenance": [
                _frame_record(row, legacy=True) for row in current_legacy_frames
            ],
            "motion_annotation": {
                "action_family": base["action_family"],
                "annotation_method": base["annotation_method"],
                "conditioning": conditioning,
                "cycle_frames": base["cycle_frames"],
                "normalized_action": base["normalized_action"],
                "source_action": base["source_action"],
                "vocabulary_version": base["vocabulary_version"],
            },
            "retrieval_ids": [str(row["id"]) for row in retrievals.get(item_id or "", [])],
            "rights_observation_ids": [str(row["id"]) for row in rights.get(item_id or "", [])],
            "sequence_action": base["action"],
            "sequence_extraction_method": base["extraction_method"],
            "sequence_metadata": sequence_metadata,
            "sequence_source_keys": [
                {
                    "external_sequence_key": row["external_sequence_key"],
                    "source_id": row["source_id"],
                }
                for row in current_source_keys
            ],
            "source": (
                {"id": source_id}
                if source is None
                else {
                    "adapter_version": source["adapter_version"],
                    "id": source["id"],
                    "kind": source["kind"],
                    "name": source["name"],
                    "root_url": source["root_url"],
                }
            ),
            "source_ids": tuple(sorted(source_ids, key=lambda value: value.encode("utf-8"))),
            "subjects": [
                {
                    "entity_class": row["entity_class"],
                    "entity_id": row["entity_id"],
                    "entity_subclass": row["entity_subclass"],
                    "external_identity_key": row["external_identity_key"],
                    "role": row["role"],
                    "species_or_type": row["species_or_type"],
                }
                for row in current_subjects
            ],
            "temporal_evidence": temporal,
        }
        first_frame = current_sequence_frames[0] if current_sequence_frames else None
        samples.append(
            SequenceSample(
                sequence_id=sequence_id,
                identity_id=identity_id,
                source_id=source_id,
                source_pack_id=source_pack_id,
                entity_class=entity_class,
                action=str(base["selected_action"]),
                view=str(base["motion_view"] or (first_frame and first_frame["view"]) or "unknown"),
                direction=str(
                    base["motion_direction"]
                    or base["direction"]
                    or (first_frame and first_frame["direction"])
                    or "unknown"
                ),
                loop_mode=str(base["loop_mode"] or "unknown"),
                frame_count=int(base["frame_count"]),
                source_blob_sha256=ordered_digests,
                duplicate_group_ids=tuple(
                    sorted(leakage_group_ids, key=lambda value: value.encode("utf-8"))
                ),
                quality_tier=str(base["quality_tier"]),
                metadata=metadata,
            )
        )
    return tuple(sorted(samples, key=lambda sample: sample.sequence_id.encode("utf-8")))


def load_sequence_samples(
    database_path: Path | str,
    filters: SnapshotFilters | None = None,
) -> tuple[SequenceSample, ...]:
    """Read eligible sequence samples without mutating or locking the provenance index."""

    selected_filters = filters or SnapshotFilters()
    with _read_connection(database_path) as connection:
        return _load_samples(connection, selected_filters)


def build_snapshot_from_index(
    database_path: Path | str,
    *,
    policy: SplitPolicy,
    filters: SnapshotFilters | None = None,
) -> SnapshotArtifact:
    """Build a deterministic split artifact from one consistent read transaction."""

    selected_filters = filters or SnapshotFilters()
    with _read_connection(database_path) as connection:
        samples = _load_samples(connection, selected_filters)
        versions = tuple(
            int(row["version"])
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        )
    manifest = build_dataset_manifest(samples, policy)
    known_count = sum(bool(sample.metadata["temporal_evidence"]["known"]) for sample in samples)
    return SnapshotArtifact(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        filters=selected_filters,
        temporal_known_contract=TEMPORAL_KNOWN_CONTRACT,
        index_schema_versions=versions,
        manifest=manifest,
        coverage=coverage_report(manifest),
        timing_counts={"known": known_count, "pose_only": len(samples) - known_count},
    )


def write_snapshot(snapshot: SnapshotArtifact, output_path: Path | str) -> Path:
    """Atomically write the canonical UTF-8 representation of an artifact."""

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(snapshot.canonical_json)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return path


def export_snapshot(
    database_path: Path | str,
    output_path: Path | str,
    *,
    policy: SplitPolicy,
    filters: SnapshotFilters | None = None,
) -> SnapshotArtifact:
    """Build and atomically export a deterministic dataset snapshot."""

    snapshot = build_snapshot_from_index(database_path, policy=policy, filters=filters)
    write_snapshot(snapshot, output_path)
    return snapshot
