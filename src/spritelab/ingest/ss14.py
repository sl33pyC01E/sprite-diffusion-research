"""Deterministic projection of audited Space Station 14 RSI states.

The projection is intentionally narrower than the archive audit.  It admits
only exact-capacity, rights-allowed, complete-entity candidates whose action
cue is already canonical in the configured taxonomy.  Every other audited
state remains in a deterministic exclusion ledger; nothing is silently
relabelled or repaired.

Each admitted RSI state becomes one sequence per source direction.  Frames
follow the pinned RobustToolbox folded timeline and retain both the engine
interval and the original source-cell delay, native source-sheet rectangle,
direction, state-local cell index, rights scope, lineage, and deduplication
evidence.  RSI does not encode loop policy, so the projection never infers it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spritelab.adapters.ss14 import (
    EXPECTED_ROBUST_TOOLBOX_ARCHIVE_SHA256,
    EXPECTED_SS14_ARCHIVE_SHA256,
    ROBUST_TOOLBOX_COMMIT,
    SS14_ARCHIVE_URL,
    SS14_COMMIT,
    SS14_COMMIT_URL,
    SS14_REPOSITORY_URL,
    EvidenceDocument,
    RightsAudit,
    RsiPack,
    RsiState,
    Ss14ArchiveAudit,
    UpstreamReference,
    audit_known_ss14_archive,
)
from spritelab.archive import ArchiveLimits, inspect_zip
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

SOURCE_ID = "space_station_14"
PROJECTION_VERSION = "ss14_exact_rsi_direction_projection_v1"
PREPARATION_VERSION = "ss14_existing_cas_preparation_v1"
SS14_ITEM_EXTERNAL_ID = "space-wizards/space-station-14"
SS14_SELECTED_IMAGE_ROLE = "ss14_projected_state_image"
SS14_MEDIA_INSPECTOR_VERSION = "media-v1"
EXPECTED_PINNED_STATE_COUNT = 189
EXPECTED_PINNED_SEQUENCE_COUNT = 246
EXPECTED_PINNED_FRAME_OCCURRENCE_COUNT = 297
EXPECTED_PINNED_ENTITY_COUNT = 139
EXPECTED_PINNED_EXCLUSION_COUNT = 1_791
EXPECTED_PINNED_REQUIRED_MEMBER_COUNT = 287
EXPECTED_PINNED_ARCHIVE_SIZE_BYTES = 234_732_657
EXPECTED_PINNED_ARCHIVE_INVENTORY_SHA256 = (
    "39ab37b8ed29cef313b89d6488946fa3727d9b04c38fa95a9391e18d8a700c59"
)
EXPECTED_PINNED_PREPARATION_MANIFEST_SHA256 = (
    "9151dbcdbc7927680306d34647d28dc4801be17f497f55cc996baa2d3f7f97a1"
)
EXPECTED_PINNED_ARCHIVE_MEMBER_COUNT = 49_472
EXPECTED_PINNED_ARCHIVE_FILE_COUNT = 43_004
EXPECTED_PINNED_ARCHIVE_DIRECTORY_COUNT = 6_468
EXPECTED_PINNED_ARCHIVE_COMPRESSED_BYTES = 218_134_591
EXPECTED_PINNED_ARCHIVE_UNCOMPRESSED_BYTES = 340_248_827
EXPECTED_PINNED_ALL_PNG_MEMBER_COUNT = 25_832
EXPECTED_PINNED_ALL_PNG_COMPRESSED_BYTES = 40_779_680
EXPECTED_PINNED_ALL_PNG_UNCOMPRESSED_BYTES = 61_014_321
EXPECTED_PINNED_REQUIRED_MEMBER_COMPRESSED_BYTES = 414_157
EXPECTED_PINNED_REQUIRED_MEMBER_UNCOMPRESSED_BYTES = 813_186
EXPECTED_PINNED_SELECTED_IMAGE_COMPRESSED_BYTES = 356_198
EXPECTED_PINNED_SELECTED_IMAGE_UNCOMPRESSED_BYTES = 536_519
EXPECTED_PINNED_UNIQUE_SELECTED_IMAGE_BYTES = 530_803


@dataclass(frozen=True, slots=True)
class Ss14ProjectionRights:
    """Verbatim, pack-scoped rights and upstream lineage."""

    license_expression: str
    copyright: str
    rights_status: str
    metadata_evidence: EvidenceDocument
    upstream_references: tuple[UpstreamReference, ...]

    @property
    def lineage_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    reference.lineage_key
                    for reference in self.upstream_references
                    if reference.lineage_key is not None
                }
            )
        )

    @property
    def upstream_asset_deduplication_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    reference.asset_deduplication_key
                    for reference in self.upstream_references
                    if reference.asset_deduplication_key is not None
                }
            )
        )


@dataclass(frozen=True, slots=True)
class Ss14ProjectionEntity:
    """Stable pack-local appearance identity shared by action states."""

    entity_external_key: str
    display_name: str
    entity_cue: str
    rsi_path: str
    category: str
    entity_class: str
    entity_class_basis: str
    rights: Ss14ProjectionRights


@dataclass(frozen=True, slots=True)
class Ss14ProjectionFrame:
    """One exact folded-engine occurrence of a native RSI source cell."""

    ordinal: int
    source_cell_index: int
    source_direction_frame_index: int
    source_delay_seconds: float
    engine_delay_seconds: float
    duration_milliseconds: float
    direction_index: int
    source_direction: str
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True, slots=True)
class Ss14ProjectionRecord:
    """One admitted state/direction timeline."""

    sequence_source_key: str
    entity: Ss14ProjectionEntity
    rsi_path: str
    state_name: str
    entity_cue: str
    source_action_cue: str
    action_cue_basis: str
    state_role: str
    state_role_basis: str
    direction_index: int
    source_direction: str
    direction_count: int
    delays_declared: bool
    source_delays_seconds: tuple[tuple[float, ...], ...]
    engine_delays_seconds: tuple[float, ...]
    engine_source_cell_indices: tuple[int, ...]
    loop_semantics: str
    frame_width: int
    frame_height: int
    source_sheet_logical_path: str
    source_sheet_member_path: str
    source_sheet_sha256: str
    source_sheet_width: int
    source_sheet_height: int
    source_sheet_format: str
    source_sheet_mode: str
    grid_columns: int
    grid_rows: int
    grid_capacity: int
    expected_source_cell_count: int
    load_srgb: bool
    meta_atlas: bool
    rsic: bool
    rights: Ss14ProjectionRights
    image_payload_deduplication_key: str
    frames: tuple[Ss14ProjectionFrame, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def is_animated(self) -> bool:
        return self.frame_count > 1

    @property
    def total_duration_milliseconds(self) -> float:
        return sum(frame.duration_milliseconds for frame in self.frames)


@dataclass(frozen=True, slots=True)
class Ss14ProjectionExclusion:
    """One audited state quarantined from the DB projection."""

    state_source_key: str
    rsi_path: str
    state_name: str
    entity_cue: str
    source_action_cue: str | None
    action_cue_basis: str
    state_role: str
    state_role_basis: str
    direction_count: int
    engine_timeline_occurrence_count: int
    expected_source_cell_count: int
    decoded_source_cell_count: int
    source_sheet_member_path: str | None
    source_sheet_sha256: str | None
    unused_cell_count: int | None
    rights: Ss14ProjectionRights
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Ss14ProjectionPlan:
    """Pure write-free plan and complete state-level quarantine ledger."""

    archive_sha256: str
    repository_commit: str
    archive_root: str
    source_audit_record_sha256: str
    taxonomy_version: str
    taxonomy_action_values: tuple[str, ...]
    records: tuple[Ss14ProjectionRecord, ...]
    exclusions: tuple[Ss14ProjectionExclusion, ...]
    repository_rights: RightsAudit
    classification_evidence: tuple[EvidenceDocument, ...]
    robust_toolbox_commit: str
    robust_toolbox_archive_sha256: str
    engine_evidence_urls: tuple[str, ...]

    @property
    def projected_state_count(self) -> int:
        return len({(record.rsi_path, record.state_name) for record in self.records})

    @property
    def projected_entity_count(self) -> int:
        return len({record.entity.entity_external_key for record in self.records})

    @property
    def projected_sequence_count(self) -> int:
        return len(self.records)

    @property
    def projected_frame_occurrence_count(self) -> int:
        return sum(record.frame_count for record in self.records)

    @property
    def projected_animated_sequence_count(self) -> int:
        return sum(record.is_animated for record in self.records)

    @property
    def projected_static_sequence_count(self) -> int:
        return sum(not record.is_animated for record in self.records)

    @property
    def excluded_state_count(self) -> int:
        return len(self.exclusions)

    @property
    def projected_occurrence_link_count(self) -> int:
        fixed = 3 + len(self.classification_evidence)
        if self.repository_rights.rsi_schema is not None:
            fixed += 1
        return self.projected_sequence_count * fixed

    @property
    def required_source_image_hashes(self) -> tuple[tuple[str, str], ...]:
        values: dict[str, str] = {}
        for record in self.records:
            previous = values.setdefault(
                record.source_sheet_member_path,
                record.source_sheet_sha256,
            )
            if previous != record.source_sheet_sha256:
                raise ValueError(
                    "One SS14 state image member has multiple audited hashes: "
                    f"{record.source_sheet_member_path!r}"
                )
        return tuple(sorted(values.items()))

    @property
    def required_member_paths(self) -> tuple[str, ...]:
        paths = {path for path, _ in self.required_source_image_hashes}
        paths.update(record.rights.metadata_evidence.member_path for record in self.records)
        paths.add(self.repository_rights.root_license.member_path)
        if self.repository_rights.rsi_schema is not None:
            paths.add(self.repository_rights.rsi_schema.member_path)
        paths.update(document.member_path for document in self.classification_evidence)
        return tuple(sorted(paths))

    @property
    def required_evidence_hashes(self) -> tuple[tuple[str, str], ...]:
        values = dict(self.required_source_image_hashes)
        for record in self.records:
            document = record.rights.metadata_evidence
            previous = values.setdefault(document.member_path, document.sha256)
            if previous != document.sha256:
                raise ValueError(f"Conflicting SS14 metadata evidence: {document.member_path!r}")
        documents = [self.repository_rights.root_license, *self.classification_evidence]
        if self.repository_rights.rsi_schema is not None:
            documents.append(self.repository_rights.rsi_schema)
        for document in documents:
            previous = values.setdefault(document.member_path, document.sha256)
            if previous != document.sha256:
                raise ValueError(f"Conflicting SS14 evidence hash: {document.member_path!r}")
        return tuple(sorted(values.items()))

    @property
    def duplicate_image_payload_group_count(self) -> int:
        counts: dict[str, int] = {}
        for _, digest in self.required_source_image_hashes:
            counts[digest] = counts.get(digest, 0) + 1
        return sum(count > 1 for count in counts.values())

    @property
    def duplicate_image_payload_excess(self) -> int:
        counts: dict[str, int] = {}
        for _, digest in self.required_source_image_hashes:
            counts[digest] = counts.get(digest, 0) + 1
        return sum(count - 1 for count in counts.values() if count > 1)

    @property
    def projection_manifest_sha256(self) -> str:
        payload = {
            "projection_version": PROJECTION_VERSION,
            "archive_sha256": self.archive_sha256,
            "repository_commit": self.repository_commit,
            "archive_root": self.archive_root,
            "source_audit_record_sha256": self.source_audit_record_sha256,
            "taxonomy_version": self.taxonomy_version,
            "taxonomy_action_values": list(self.taxonomy_action_values),
            "records": [asdict(record) for record in self.records],
            "exclusions": [asdict(exclusion) for exclusion in self.exclusions],
            "repository_rights": asdict(self.repository_rights),
            "classification_evidence": [
                asdict(document) for document in self.classification_evidence
            ],
            "robust_toolbox_commit": self.robust_toolbox_commit,
            "robust_toolbox_archive_sha256": self.robust_toolbox_archive_sha256,
            "engine_evidence_urls": list(self.engine_evidence_urls),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Ss14PreparationMember:
    """One archive member required by the projection evidence closure."""

    ordinal: int
    member_path: str
    expected_sha256: str
    size_bytes: int
    compressed_bytes: int
    crc32: int
    compression_method: int
    extraction_required: bool


@dataclass(frozen=True, slots=True)
class Ss14PreparationPlan:
    """Pure plan for adopting an existing SS14 CAS archive into an index."""

    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    repository_commit: str
    repository_url: str
    commit_url: str
    archive_url: str
    source_item_external_id: str
    projection_manifest_sha256: str
    archive_inventory_sha256: str
    archive_member_count: int
    archive_file_count: int
    archive_directory_count: int
    archive_symlink_count: int
    archive_total_compressed_bytes: int
    archive_total_uncompressed_bytes: int
    all_png_member_count: int
    all_png_compressed_bytes: int
    all_png_uncompressed_bytes: int
    required_members: tuple[Ss14PreparationMember, ...]

    def __post_init__(self) -> None:
        member_paths = [member.member_path for member in self.required_members]
        ordinals = [member.ordinal for member in self.required_members]
        if len(member_paths) != len(set(member_paths)):
            raise ValueError("SS14 preparation member paths are not unique")
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("SS14 preparation member ordinals are not unique")
        if any(
            member.size_bytes < 0 or member.compressed_bytes < 0 for member in self.required_members
        ):
            raise ValueError("SS14 preparation member sizes must be non-negative")

    @property
    def required_member_count(self) -> int:
        return len(self.required_members)

    @property
    def required_member_compressed_bytes(self) -> int:
        return sum(member.compressed_bytes for member in self.required_members)

    @property
    def required_member_uncompressed_bytes(self) -> int:
        return sum(member.size_bytes for member in self.required_members)

    @property
    def selected_image_members(self) -> tuple[Ss14PreparationMember, ...]:
        return tuple(member for member in self.required_members if member.extraction_required)

    @property
    def selected_image_member_paths(self) -> tuple[str, ...]:
        return tuple(member.member_path for member in self.selected_image_members)

    @property
    def selected_image_member_count(self) -> int:
        return len(self.selected_image_members)

    @property
    def selected_image_compressed_bytes(self) -> int:
        return sum(member.compressed_bytes for member in self.selected_image_members)

    @property
    def selected_image_uncompressed_bytes(self) -> int:
        return sum(member.size_bytes for member in self.selected_image_members)

    @property
    def unique_selected_image_hashes(self) -> tuple[str, ...]:
        return tuple(sorted({member.expected_sha256 for member in self.selected_image_members}))

    @property
    def unique_selected_image_count(self) -> int:
        return len(self.unique_selected_image_hashes)

    @property
    def unique_selected_image_bytes(self) -> int:
        sizes: dict[str, int] = {}
        for member in self.selected_image_members:
            previous = sizes.setdefault(member.expected_sha256, member.size_bytes)
            if previous != member.size_bytes:
                raise ValueError(
                    f"Equal SS14 image hashes have conflicting sizes: {member.expected_sha256}"
                )
        return sum(sizes.values())

    @property
    def preparation_manifest_sha256(self) -> str:
        payload = {
            "preparation_version": PREPARATION_VERSION,
            "archive_sha256": self.archive_sha256,
            "archive_size_bytes": self.archive_size_bytes,
            "repository_commit": self.repository_commit,
            "repository_url": self.repository_url,
            "commit_url": self.commit_url,
            "archive_url": self.archive_url,
            "source_id": SOURCE_ID,
            "source_item_external_id": self.source_item_external_id,
            "projection_manifest_sha256": self.projection_manifest_sha256,
            "archive_inventory_sha256": self.archive_inventory_sha256,
            "archive_member_count": self.archive_member_count,
            "archive_file_count": self.archive_file_count,
            "archive_directory_count": self.archive_directory_count,
            "archive_symlink_count": self.archive_symlink_count,
            "archive_total_compressed_bytes": self.archive_total_compressed_bytes,
            "archive_total_uncompressed_bytes": self.archive_total_uncompressed_bytes,
            "all_png_member_count": self.all_png_member_count,
            "all_png_compressed_bytes": self.all_png_compressed_bytes,
            "all_png_uncompressed_bytes": self.all_png_uncompressed_bytes,
            "selected_image_role": SS14_SELECTED_IMAGE_ROLE,
            "media_inspector_version": SS14_MEDIA_INSPECTOR_VERSION,
            "required_members": [asdict(member) for member in self.required_members],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Ss14ProjectionReadiness:
    """Query-only status for every prerequisite of a projection plan."""

    database_path: str
    archive_sha256: str
    projection_manifest_sha256: str
    source_registered: bool
    archive_inventory_present: bool
    source_item_count: int
    required_member_count: int
    present_member_count: int
    required_source_image_count: int
    present_source_image_blob_count: int
    missing_member_paths: tuple[str, ...]
    missing_source_image_blobs: tuple[str, ...]
    source_image_hash_mismatches: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.source_registered
            and self.archive_inventory_present
            and self.source_item_count > 0
            and not self.missing_member_paths
            and not self.missing_source_image_blobs
            and not self.source_image_hash_mismatches
        )


@dataclass(frozen=True, slots=True)
class Ss14PreparationReadiness:
    """Query-only adoption, extraction, and media-QA readiness."""

    database_path: str
    archive_sha256: str
    preparation_manifest_sha256: str
    source_registered: bool
    archive_blob_registered: bool
    archive_blob_facts_match: bool
    source_item_count: int
    pinned_source_item_count: int
    source_archive_link_count: int
    archive_retrieval_count: int
    archive_inventory_present: bool
    archive_inventory_exact: bool
    expected_archive_member_count: int
    indexed_archive_member_count: int
    required_member_count: int
    present_required_member_count: int
    selected_image_member_count: int
    present_selected_image_blob_count: int
    unique_selected_image_count: int
    present_unique_selected_image_blob_count: int
    present_unique_selected_image_file_count: int
    selected_image_role_count: int
    media_observation_count: int
    media_inspected_member_count: int
    media_invalid_member_count: int
    media_terminal_unique_image_count: int
    archive_blob_fact_mismatches: tuple[str, ...]
    archive_inventory_mismatches: tuple[str, ...]
    missing_required_member_paths: tuple[str, ...]
    required_member_fact_mismatches: tuple[str, ...]
    missing_selected_image_blobs: tuple[str, ...]
    selected_image_hash_mismatches: tuple[str, ...]
    selected_image_blob_fact_mismatches: tuple[str, ...]
    missing_selected_image_files: tuple[str, ...]
    selected_image_file_hash_mismatches: tuple[str, ...]
    media_invalid_members: tuple[str, ...]

    @property
    def projection_prerequisites_ready(self) -> bool:
        return (
            self.source_registered
            and self.archive_blob_registered
            and self.archive_blob_facts_match
            and self.pinned_source_item_count > 0
            and self.source_archive_link_count > 0
            and self.archive_inventory_present
            and self.archive_inventory_exact
            and self.indexed_archive_member_count == self.expected_archive_member_count
            and self.present_required_member_count == self.required_member_count
            and not self.required_member_fact_mismatches
            and self.present_selected_image_blob_count == self.selected_image_member_count
            and self.present_unique_selected_image_blob_count == self.unique_selected_image_count
            and self.present_unique_selected_image_file_count == self.unique_selected_image_count
            and not self.missing_selected_image_blobs
            and not self.selected_image_hash_mismatches
            and not self.selected_image_blob_fact_mismatches
            and not self.missing_selected_image_files
            and not self.selected_image_file_hash_mismatches
        )

    @property
    def extraction_complete(self) -> bool:
        return (
            self.projection_prerequisites_ready
            and self.selected_image_role_count == self.selected_image_member_count
        )

    @property
    def media_inspection_complete(self) -> bool:
        return (
            self.media_terminal_unique_image_count == self.unique_selected_image_count
            and self.media_inspected_member_count + self.media_invalid_member_count
            == self.selected_image_member_count
        )

    @property
    def all_media_valid(self) -> bool:
        return (
            self.media_observation_count == self.unique_selected_image_count
            and self.media_inspected_member_count == self.selected_image_member_count
            and self.media_invalid_member_count == 0
        )

    @property
    def ready(self) -> bool:
        return self.extraction_complete and self.media_inspection_complete


@dataclass(frozen=True, slots=True)
class Ss14ProjectionResult:
    """Effects of one idempotent projection call."""

    archive_sha256: str
    projection_manifest_sha256: str
    projected_states: int
    projected_entities: int
    projected_sequences: int
    projected_frame_occurrences: int
    projected_animated_sequences: int
    projected_static_sequences: int
    created_sequences: int
    reused_sequences: int
    occurrence_links: int
    excluded_states: int
    rights_observations_added: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _stable_json_key(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _projection_rights(pack: RsiPack) -> Ss14ProjectionRights:
    return Ss14ProjectionRights(
        license_expression=pack.license_expression,
        copyright=pack.copyright,
        rights_status=pack.rights_status,
        metadata_evidence=pack.metadata_evidence,
        upstream_references=pack.upstream_references,
    )


def _entity_external_key(audit: Ss14ArchiveAudit, pack: RsiPack, state: RsiState) -> str:
    return _stable_json_key(
        "ss14_entity",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "rsi_path": pack.logical_path,
            "entity_cue": state.entity_cue,
        },
    )


def _state_source_key(audit: Ss14ArchiveAudit, pack: RsiPack, state: RsiState) -> str:
    return _stable_json_key(
        "ss14_state",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "rsi_path": pack.logical_path,
            "state_name": state.name,
        },
    )


def _sequence_source_key(
    audit: Ss14ArchiveAudit,
    pack: RsiPack,
    state: RsiState,
    direction_index: int,
) -> str:
    return _stable_json_key(
        "ss14_sequence",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "rsi_path": pack.logical_path,
            "state_name": state.name,
            "direction_index": direction_index,
            "source_direction": state.direction_names[direction_index],
        },
    )


def _exclusion_reasons(
    pack: RsiPack,
    state: RsiState,
    taxonomy: Taxonomy,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if state.role != "complete_entity_candidate":
        reasons.append(f"state_role:{state.role}")
    reasons.extend(
        f"quarantine:{reason}" for reason in (*pack.quarantine_reasons, *state.quarantine_reasons)
    )
    if state.image is None:
        reasons.append("image:missing_or_invalid")
    elif state.image.unused_cell_count:
        reasons.append("image:surplus_capacity")
    if state.normalized_action is None:
        reasons.append("action:unmapped")
    elif state.normalized_action not in taxonomy.action_to_family:
        reasons.append(f"action:noncanonical:{state.normalized_action}")
    if len(pack.entity_class_candidates) != 1:
        reasons.append("entity_class:ambiguous")
    else:
        normalized = taxonomy.normalize_entity_class(pack.entity_class_candidates[0])
        if normalized.value == "unknown":
            reasons.append("entity_class:ambiguous")
    return tuple(dict.fromkeys(reasons))


def _projection_entity(
    audit: Ss14ArchiveAudit,
    pack: RsiPack,
    state: RsiState,
) -> Ss14ProjectionEntity:
    if len(pack.entity_class_candidates) != 1:
        raise ValueError(f"SS14 projected entity has ambiguous class: {pack.logical_path}")
    return Ss14ProjectionEntity(
        entity_external_key=_entity_external_key(audit, pack, state),
        display_name=state.entity_cue,
        entity_cue=state.entity_cue,
        rsi_path=pack.logical_path,
        category=pack.category,
        entity_class=pack.entity_class_candidates[0],
        entity_class_basis=pack.entity_class_basis,
        rights=_projection_rights(pack),
    )


def _projection_record(
    audit: Ss14ArchiveAudit,
    pack: RsiPack,
    state: RsiState,
    direction_index: int,
) -> Ss14ProjectionRecord:
    if state.image is None or state.normalized_action is None:
        raise ValueError("Unsafe SS14 state reached projection record construction")
    source_cells = {frame.source_cell_index: frame for frame in state.source_frames}
    engine_indices = state.engine_source_cell_indices[direction_index]
    if len(engine_indices) != len(state.engine_delays_seconds):
        raise ValueError("SS14 folded timeline indices do not match engine delays")
    frames: list[Ss14ProjectionFrame] = []
    for ordinal, (source_cell_index, engine_delay) in enumerate(
        zip(engine_indices, state.engine_delays_seconds, strict=True)
    ):
        source = source_cells.get(source_cell_index)
        if source is None or source.direction_index != direction_index:
            raise ValueError(
                "SS14 folded timeline refers to an absent or cross-direction source cell: "
                f"{pack.logical_path}:{state.name}:{direction_index}:{source_cell_index}"
            )
        frames.append(
            Ss14ProjectionFrame(
                ordinal=ordinal,
                source_cell_index=source.source_cell_index,
                source_direction_frame_index=source.frame_index_in_direction,
                source_delay_seconds=source.delay_seconds,
                engine_delay_seconds=engine_delay,
                duration_milliseconds=engine_delay * 1000.0,
                direction_index=direction_index,
                source_direction=source.direction,
                left=source.left,
                top=source.top,
                right=source.right,
                bottom=source.bottom,
            )
        )
    return Ss14ProjectionRecord(
        sequence_source_key=_sequence_source_key(audit, pack, state, direction_index),
        entity=_projection_entity(audit, pack, state),
        rsi_path=pack.logical_path,
        state_name=state.name,
        entity_cue=state.entity_cue,
        source_action_cue=state.normalized_action,
        action_cue_basis=state.normalized_action_basis,
        state_role=state.role,
        state_role_basis=state.role_basis,
        direction_index=direction_index,
        source_direction=state.direction_names[direction_index],
        direction_count=state.direction_count,
        delays_declared=state.delays_declared,
        source_delays_seconds=state.source_delays_seconds,
        engine_delays_seconds=state.engine_delays_seconds,
        engine_source_cell_indices=engine_indices,
        loop_semantics=state.loop_semantics,
        frame_width=pack.frame_width,
        frame_height=pack.frame_height,
        source_sheet_logical_path=state.image.logical_path,
        source_sheet_member_path=state.image.member_path,
        source_sheet_sha256=state.image.sha256,
        source_sheet_width=state.image.width,
        source_sheet_height=state.image.height,
        source_sheet_format=state.image.detected_format,
        source_sheet_mode=state.image.image_mode,
        grid_columns=state.image.grid_columns,
        grid_rows=state.image.grid_rows,
        grid_capacity=state.image.grid_capacity,
        expected_source_cell_count=state.expected_source_cell_count,
        load_srgb=pack.load_srgb,
        meta_atlas=pack.meta_atlas,
        rsic=pack.rsic,
        rights=_projection_rights(pack),
        image_payload_deduplication_key=f"sha256:{state.image.sha256}",
        frames=tuple(frames),
    )


def _projection_exclusion(
    audit: Ss14ArchiveAudit,
    pack: RsiPack,
    state: RsiState,
    reasons: tuple[str, ...],
) -> Ss14ProjectionExclusion:
    image = state.image
    return Ss14ProjectionExclusion(
        state_source_key=_state_source_key(audit, pack, state),
        rsi_path=pack.logical_path,
        state_name=state.name,
        entity_cue=state.entity_cue,
        source_action_cue=state.normalized_action,
        action_cue_basis=state.normalized_action_basis,
        state_role=state.role,
        state_role_basis=state.role_basis,
        direction_count=state.direction_count,
        engine_timeline_occurrence_count=(state.direction_count * len(state.engine_delays_seconds)),
        expected_source_cell_count=state.expected_source_cell_count,
        decoded_source_cell_count=len(state.source_frames),
        source_sheet_member_path=image.member_path if image is not None else None,
        source_sheet_sha256=image.sha256 if image is not None else None,
        unused_cell_count=image.unused_cell_count if image is not None else None,
        rights=_projection_rights(pack),
        reasons=reasons,
    )


def plan_ss14_projection(
    audit: Ss14ArchiveAudit,
    taxonomy: Taxonomy,
) -> Ss14ProjectionPlan:
    """Build a deterministic plan and partition every audited RSI state."""

    records: list[Ss14ProjectionRecord] = []
    exclusions: list[Ss14ProjectionExclusion] = []
    selected_states = 0
    for pack in audit.packs:
        for state in pack.states:
            reasons = _exclusion_reasons(pack, state, taxonomy)
            if reasons:
                exclusions.append(_projection_exclusion(audit, pack, state, reasons))
                continue
            selected_states += 1
            records.extend(
                _projection_record(audit, pack, state, direction_index)
                for direction_index in range(state.direction_count)
            )
    records.sort(key=lambda record: record.sequence_source_key)
    exclusions.sort(key=lambda exclusion: exclusion.state_source_key)
    record_keys = [record.sequence_source_key for record in records]
    exclusion_keys = [exclusion.state_source_key for exclusion in exclusions]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("SS14 projection sequence source keys are not unique")
    if len(exclusion_keys) != len(set(exclusion_keys)):
        raise ValueError("SS14 projection exclusion source keys are not unique")
    if selected_states + len(exclusions) != audit.counts.states:
        raise AssertionError("SS14 projection does not partition every audited state")
    if any(record.source_action_cue not in taxonomy.action_to_family for record in records):
        raise AssertionError("SS14 projection admitted a noncanonical action cue")
    return Ss14ProjectionPlan(
        archive_sha256=audit.archive_sha256,
        repository_commit=audit.commit,
        archive_root=audit.archive_root,
        source_audit_record_sha256=audit.audit_record_sha256,
        taxonomy_version=taxonomy.version,
        taxonomy_action_values=tuple(sorted(taxonomy.action_to_family)),
        records=tuple(records),
        exclusions=tuple(exclusions),
        repository_rights=audit.rights,
        classification_evidence=audit.classification_evidence,
        robust_toolbox_commit=audit.robust_toolbox_commit,
        robust_toolbox_archive_sha256=EXPECTED_ROBUST_TOOLBOX_ARCHIVE_SHA256,
        engine_evidence_urls=audit.engine_evidence_urls,
    )


def plan_known_ss14_projection(
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> Ss14ProjectionPlan:
    """Audit the exact pinned CAS snapshot and enforce regression counts."""

    plan = plan_ss14_projection(audit_known_ss14_archive(Path(archive_path)), taxonomy)
    expected = (
        EXPECTED_PINNED_STATE_COUNT,
        EXPECTED_PINNED_SEQUENCE_COUNT,
        EXPECTED_PINNED_FRAME_OCCURRENCE_COUNT,
        EXPECTED_PINNED_ENTITY_COUNT,
        EXPECTED_PINNED_EXCLUSION_COUNT,
        EXPECTED_PINNED_REQUIRED_MEMBER_COUNT,
    )
    actual = (
        plan.projected_state_count,
        plan.projected_sequence_count,
        plan.projected_frame_occurrence_count,
        plan.projected_entity_count,
        plan.excluded_state_count,
        len(plan.required_member_paths),
    )
    if actual != expected:
        raise ValueError(f"Pinned SS14 projection count drift: expected {expected}, got {actual}")
    return plan


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_ss14_preparation(
    archive_path: str | Path,
    projection: Ss14ProjectionPlan,
    *,
    limits: ArchiveLimits | None = None,
) -> Ss14PreparationPlan:
    """Build a write-free exact-member preparation plan for one projection."""

    resolved_archive = Path(archive_path).resolve()
    if not resolved_archive.is_file():
        raise FileNotFoundError(f"SS14 archive is not present: {resolved_archive}")
    actual_archive_sha256 = _sha256_path(resolved_archive)
    if actual_archive_sha256 != projection.archive_sha256:
        raise ValueError(
            "SS14 preparation archive hash mismatch: "
            f"projection {projection.archive_sha256}, file {actual_archive_sha256}"
        )

    manifest = inspect_zip(resolved_archive, limits=limits or ArchiveLimits())
    members_by_path = {
        path: member
        for member in manifest.members
        for path in {member.original_name, member.normalized_name}
    }
    expected_hashes = dict(projection.required_evidence_hashes)
    selected_hashes = dict(projection.required_source_image_hashes)
    missing = tuple(sorted(set(expected_hashes).difference(members_by_path)))
    if missing:
        raise ValueError(
            "SS14 preparation evidence is absent from the archive: " + ", ".join(missing[:10])
        )

    required_members: list[Ss14PreparationMember] = []
    for member_path, expected_sha256 in sorted(expected_hashes.items()):
        member = members_by_path[member_path]
        if not member.is_regular_file:
            raise ValueError(f"SS14 preparation evidence is not a regular file: {member_path}")
        extraction_required = member_path in selected_hashes
        if extraction_required and member.extension != ".png":
            raise ValueError(f"SS14 projected state image is not named as PNG: {member_path}")
        required_members.append(
            Ss14PreparationMember(
                ordinal=member.archive_index,
                member_path=member.normalized_name,
                expected_sha256=expected_sha256,
                size_bytes=member.uncompressed_bytes,
                compressed_bytes=member.compressed_bytes,
                crc32=member.crc32,
                compression_method=member.compression_method,
                extraction_required=extraction_required,
            )
        )

    png_members = tuple(
        member
        for member in manifest.members
        if member.is_regular_file and member.extension == ".png"
    )
    return Ss14PreparationPlan(
        archive_path=str(resolved_archive),
        archive_sha256=projection.archive_sha256,
        archive_size_bytes=resolved_archive.stat().st_size,
        repository_commit=projection.repository_commit,
        repository_url=SS14_REPOSITORY_URL,
        commit_url=SS14_COMMIT_URL,
        archive_url=SS14_ARCHIVE_URL,
        source_item_external_id=SS14_ITEM_EXTERNAL_ID,
        projection_manifest_sha256=projection.projection_manifest_sha256,
        archive_inventory_sha256=manifest.inventory_sha256,
        archive_member_count=len(manifest.members),
        archive_file_count=manifest.regular_file_count,
        archive_directory_count=manifest.directory_count,
        archive_symlink_count=manifest.symlink_count,
        archive_total_compressed_bytes=manifest.total_compressed_bytes,
        archive_total_uncompressed_bytes=manifest.total_uncompressed_bytes,
        all_png_member_count=len(png_members),
        all_png_compressed_bytes=sum(member.compressed_bytes for member in png_members),
        all_png_uncompressed_bytes=sum(member.uncompressed_bytes for member in png_members),
        required_members=tuple(required_members),
    )


def plan_known_ss14_preparation(
    archive_path: str | Path,
    taxonomy: Taxonomy,
    *,
    limits: ArchiveLimits | None = None,
) -> Ss14PreparationPlan:
    """Plan adoption of the exact pinned archive and enforce inventory facts."""

    projection = plan_known_ss14_projection(archive_path, taxonomy)
    plan = plan_ss14_preparation(archive_path, projection, limits=limits)
    expected = (
        EXPECTED_PINNED_ARCHIVE_SIZE_BYTES,
        EXPECTED_PINNED_ARCHIVE_INVENTORY_SHA256,
        EXPECTED_PINNED_ARCHIVE_MEMBER_COUNT,
        EXPECTED_PINNED_ARCHIVE_FILE_COUNT,
        EXPECTED_PINNED_ARCHIVE_DIRECTORY_COUNT,
        0,
        EXPECTED_PINNED_ARCHIVE_COMPRESSED_BYTES,
        EXPECTED_PINNED_ARCHIVE_UNCOMPRESSED_BYTES,
        EXPECTED_PINNED_ALL_PNG_MEMBER_COUNT,
        EXPECTED_PINNED_ALL_PNG_COMPRESSED_BYTES,
        EXPECTED_PINNED_ALL_PNG_UNCOMPRESSED_BYTES,
        EXPECTED_PINNED_REQUIRED_MEMBER_COUNT,
        EXPECTED_PINNED_REQUIRED_MEMBER_COMPRESSED_BYTES,
        EXPECTED_PINNED_REQUIRED_MEMBER_UNCOMPRESSED_BYTES,
        189,
        EXPECTED_PINNED_SELECTED_IMAGE_COMPRESSED_BYTES,
        EXPECTED_PINNED_SELECTED_IMAGE_UNCOMPRESSED_BYTES,
        187,
        EXPECTED_PINNED_UNIQUE_SELECTED_IMAGE_BYTES,
    )
    actual = (
        plan.archive_size_bytes,
        plan.archive_inventory_sha256,
        plan.archive_member_count,
        plan.archive_file_count,
        plan.archive_directory_count,
        plan.archive_symlink_count,
        plan.archive_total_compressed_bytes,
        plan.archive_total_uncompressed_bytes,
        plan.all_png_member_count,
        plan.all_png_compressed_bytes,
        plan.all_png_uncompressed_bytes,
        plan.required_member_count,
        plan.required_member_compressed_bytes,
        plan.required_member_uncompressed_bytes,
        plan.selected_image_member_count,
        plan.selected_image_compressed_bytes,
        plan.selected_image_uncompressed_bytes,
        plan.unique_selected_image_count,
        plan.unique_selected_image_bytes,
    )
    if actual != expected:
        raise ValueError(f"Pinned SS14 preparation count drift: expected {expected}, got {actual}")
    if plan.preparation_manifest_sha256 != EXPECTED_PINNED_PREPARATION_MANIFEST_SHA256:
        raise ValueError(
            "Pinned SS14 preparation manifest drift: expected "
            f"{EXPECTED_PINNED_PREPARATION_MANIFEST_SHA256}, got "
            f"{plan.preparation_manifest_sha256}"
        )
    return plan


@contextmanager
def _readonly_connection(database_path: str | Path) -> Iterator[sqlite3.Connection]:
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    try:
        yield connection
    finally:
        connection.close()


def check_ss14_projection_readiness(
    database_path: str | Path,
    plan: Ss14ProjectionPlan,
) -> Ss14ProjectionReadiness:
    """Inspect live or temporary prerequisites without any SQLite write."""

    required_paths = plan.required_member_paths
    expected_images = dict(plan.required_source_image_hashes)
    with _readonly_connection(database_path) as connection:
        source_registered = (
            connection.execute("SELECT 1 FROM sources WHERE id=? LIMIT 1", (SOURCE_ID,)).fetchone()
            is not None
        )
        archive_inventory_present = (
            connection.execute(
                "SELECT 1 FROM archive_inventories WHERE archive_blob_sha256=? LIMIT 1",
                (plan.archive_sha256,),
            ).fetchone()
            is not None
        )
        source_item_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM items i
                JOIN item_blobs ib ON ib.item_id=i.id
                WHERE i.source_id=? AND ib.blob_sha256=?
                """,
                (SOURCE_ID, plan.archive_sha256),
            ).fetchone()[0]
        )
        rows = connection.execute(
            """
            SELECT am.normalized_path, am.member_path, am.extracted_blob_sha256,
                   b.sha256 AS registered_blob_sha256
            FROM archive_members am
            LEFT JOIN blobs b ON b.sha256=am.extracted_blob_sha256
            WHERE am.archive_blob_sha256=?
            """,
            (plan.archive_sha256,),
        ).fetchall()
    members: dict[str, sqlite3.Row] = {}
    for row in rows:
        members[str(row["normalized_path"])] = row
        members[str(row["member_path"])] = row
    missing_paths = tuple(path for path in required_paths if path not in members)
    missing_blobs: list[str] = []
    mismatches: list[str] = []
    present_image_blobs = 0
    for member_path, expected_hash in sorted(expected_images.items()):
        row = members.get(member_path)
        if row is None:
            continue
        actual_hash = row["extracted_blob_sha256"]
        registered_hash = row["registered_blob_sha256"]
        if actual_hash is None or registered_hash is None:
            missing_blobs.append(member_path)
        elif str(actual_hash) != expected_hash:
            mismatches.append(f"{member_path}: expected {expected_hash}, indexed {actual_hash}")
        else:
            present_image_blobs += 1
    return Ss14ProjectionReadiness(
        database_path=str(Path(database_path).resolve()),
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=plan.projection_manifest_sha256,
        source_registered=source_registered,
        archive_inventory_present=archive_inventory_present,
        source_item_count=source_item_count,
        required_member_count=len(required_paths),
        present_member_count=len(required_paths) - len(missing_paths),
        required_source_image_count=len(expected_images),
        present_source_image_blob_count=present_image_blobs,
        missing_member_paths=missing_paths,
        missing_source_image_blobs=tuple(missing_blobs),
        source_image_hash_mismatches=tuple(mismatches),
    )


def _json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def check_ss14_preparation_readiness(
    database_path: str | Path,
    plan: Ss14PreparationPlan,
) -> Ss14PreparationReadiness:
    """Inspect an SS14 archive adoption without mutating SQLite or the CAS."""

    expected_by_path = {member.member_path: member for member in plan.required_members}
    selected_members = plan.selected_image_members
    expected_unique_hashes = plan.unique_selected_image_hashes
    archive_blob_mismatches: list[str] = []
    inventory_mismatches: list[str] = []

    with _readonly_connection(database_path) as connection:
        source_registered = (
            connection.execute("SELECT 1 FROM sources WHERE id=? LIMIT 1", (SOURCE_ID,)).fetchone()
            is not None
        )
        archive_blob = connection.execute(
            "SELECT size_bytes, storage_path FROM blobs WHERE sha256=?",
            (plan.archive_sha256,),
        ).fetchone()
        if archive_blob is not None:
            if int(archive_blob["size_bytes"]) != plan.archive_size_bytes:
                archive_blob_mismatches.append(
                    "size_bytes: expected "
                    f"{plan.archive_size_bytes}, indexed {archive_blob['size_bytes']}"
                )
            indexed_archive_path = Path(str(archive_blob["storage_path"])).resolve()
            if indexed_archive_path != Path(plan.archive_path).resolve():
                archive_blob_mismatches.append(
                    f"storage_path: expected {plan.archive_path}, indexed {indexed_archive_path}"
                )
            if not indexed_archive_path.is_file():
                archive_blob_mismatches.append(
                    f"storage_path is not a file: {indexed_archive_path}"
                )
            elif indexed_archive_path.stat().st_size != plan.archive_size_bytes:
                archive_blob_mismatches.append(
                    "physical size: expected "
                    f"{plan.archive_size_bytes}, observed {indexed_archive_path.stat().st_size}"
                )
            elif _sha256_path(indexed_archive_path) != plan.archive_sha256:
                archive_blob_mismatches.append(
                    f"physical SHA-256 does not match {plan.archive_sha256}: {indexed_archive_path}"
                )

        item_rows = connection.execute(
            """
            SELECT id, canonical_url, metadata_json FROM items
            WHERE source_id=? AND external_id=?
            """,
            (SOURCE_ID, plan.source_item_external_id),
        ).fetchall()
        pinned_item_ids = {
            str(row["id"])
            for row in item_rows
            if str(row["canonical_url"]) == plan.repository_url
            and _json_object(str(row["metadata_json"])).get("commit_sha") == plan.repository_commit
        }
        source_archive_link_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM item_blobs ib
                JOIN items i ON i.id=ib.item_id
                WHERE i.source_id=? AND i.external_id=?
                  AND ib.blob_sha256=? AND ib.role='source_archive'
                  AND ib.original_url=? AND ib.original_filename=?
                """,
                (
                    SOURCE_ID,
                    plan.source_item_external_id,
                    plan.archive_sha256,
                    plan.archive_url,
                    f"space-station-14-{plan.repository_commit}.zip",
                ),
            ).fetchone()[0]
        )
        archive_retrieval_count = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM retrievals r
                JOIN items i ON i.id=r.item_id
                WHERE i.source_id=? AND i.external_id=? AND r.blob_sha256=?
                """,
                (SOURCE_ID, plan.source_item_external_id, plan.archive_sha256),
            ).fetchone()[0]
        )

        inventory = connection.execute(
            "SELECT * FROM archive_inventories WHERE archive_blob_sha256=?",
            (plan.archive_sha256,),
        ).fetchone()
        inventory_expected = {
            "archive_format": "zip",
            "member_count": plan.archive_member_count,
            "file_count": plan.archive_file_count,
            "total_uncompressed_bytes": plan.archive_total_uncompressed_bytes,
            "total_compressed_bytes": plan.archive_total_compressed_bytes,
            "inventory_sha256": plan.archive_inventory_sha256,
        }
        if inventory is not None:
            for field, expected in inventory_expected.items():
                actual = inventory[field]
                if actual != expected:
                    inventory_mismatches.append(f"{field}: expected {expected}, indexed {actual}")
        indexed_archive_member_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM archive_members WHERE archive_blob_sha256=?",
                (plan.archive_sha256,),
            ).fetchone()[0]
        )

        required_rows: list[sqlite3.Row] = []
        required_paths = tuple(expected_by_path)
        for offset in range(0, len(required_paths), 250):
            batch = required_paths[offset : offset + 250]
            placeholders = ",".join("?" for _ in batch)
            required_rows.extend(
                connection.execute(
                    f"""
                    SELECT am.*, b.sha256 AS registered_blob_sha256,
                           b.size_bytes AS registered_blob_size_bytes,
                           b.storage_path AS registered_blob_storage_path
                    FROM archive_members am
                    LEFT JOIN blobs b ON b.sha256=am.extracted_blob_sha256
                    WHERE am.archive_blob_sha256=?
                      AND am.normalized_path IN ({placeholders})
                    """,
                    (plan.archive_sha256, *batch),
                ).fetchall()
            )
        rows_by_path = {str(row["normalized_path"]): row for row in required_rows}

        if expected_unique_hashes:
            placeholders = ",".join("?" for _ in expected_unique_hashes)
            media_hashes = {
                str(row[0])
                for row in connection.execute(
                    f"""
                    SELECT blob_sha256 FROM media_observations
                    WHERE inspector_version=? AND blob_sha256 IN ({placeholders})
                    """,
                    (SS14_MEDIA_INSPECTOR_VERSION, *expected_unique_hashes),
                )
            }
        else:
            media_hashes = set()

    missing_required_paths = tuple(
        member_path for member_path in expected_by_path if member_path not in rows_by_path
    )
    member_fact_mismatches: list[str] = []
    for member_path, expected in expected_by_path.items():
        row = rows_by_path.get(member_path)
        if row is None:
            continue
        expected_facts = {
            "ordinal": expected.ordinal,
            "member_kind": "file",
            "size_bytes": expected.size_bytes,
            "compressed_bytes": expected.compressed_bytes,
            "crc32": expected.crc32,
            "compression_method": expected.compression_method,
        }
        for field, expected_value in expected_facts.items():
            actual_value = row[field]
            if actual_value != expected_value:
                member_fact_mismatches.append(
                    f"{member_path} {field}: expected {expected_value}, indexed {actual_value}"
                )

    missing_image_blobs: list[str] = []
    image_hash_mismatches: list[str] = []
    image_blob_fact_mismatches: list[str] = []
    missing_image_files: list[str] = []
    image_file_hash_mismatches: list[str] = []
    present_image_occurrences = 0
    selected_role_count = 0
    inspected_member_count = 0
    invalid_member_count = 0
    invalid_hashes: set[str] = set()
    invalid_members: list[str] = []
    registered_hashes: set[str] = set()
    valid_file_hashes: set[str] = set()
    checked_files: dict[str, tuple[bool, str | None]] = {}

    for member in selected_members:
        row = rows_by_path.get(member.member_path)
        if row is None:
            missing_image_blobs.append(member.member_path)
            continue
        if row["selected_role"] == SS14_SELECTED_IMAGE_ROLE:
            selected_role_count += 1
        if row["inspection_status"] == "media_inspected":
            inspected_member_count += 1
        elif row["inspection_status"] == "media_invalid":
            invalid_member_count += 1
            invalid_hashes.add(member.expected_sha256)
            invalid_members.append(
                f"{member.member_path}: {row['error'] or 'media inspector rejected payload'}"
            )
        actual_hash = row["extracted_blob_sha256"]
        registered_hash = row["registered_blob_sha256"]
        if actual_hash is None or registered_hash is None:
            missing_image_blobs.append(member.member_path)
            continue
        if str(actual_hash) != member.expected_sha256:
            image_hash_mismatches.append(
                f"{member.member_path}: expected {member.expected_sha256}, indexed {actual_hash}"
            )
            continue
        if int(row["registered_blob_size_bytes"]) != member.size_bytes:
            image_blob_fact_mismatches.append(
                f"{member.member_path}: expected blob size {member.size_bytes}, "
                f"indexed {row['registered_blob_size_bytes']}"
            )
            continue

        present_image_occurrences += 1
        registered_hashes.add(member.expected_sha256)
        physical = checked_files.get(member.expected_sha256)
        if physical is None:
            storage_path = Path(str(row["registered_blob_storage_path"])).resolve()
            if not storage_path.is_file():
                physical = (False, f"{member.expected_sha256}: {storage_path}")
            elif storage_path.stat().st_size != member.size_bytes:
                physical = (
                    False,
                    f"{member.expected_sha256}: expected {member.size_bytes} bytes, "
                    f"observed {storage_path.stat().st_size} at {storage_path}",
                )
            else:
                actual_file_hash = _sha256_path(storage_path)
                physical = (
                    actual_file_hash == member.expected_sha256,
                    (
                        None
                        if actual_file_hash == member.expected_sha256
                        else f"{member.expected_sha256}: observed {actual_file_hash} at "
                        f"{storage_path}"
                    ),
                )
            checked_files[member.expected_sha256] = physical
        if physical[0]:
            valid_file_hashes.add(member.expected_sha256)
        elif physical[1] is not None:
            if "observed" in physical[1] and " at " in physical[1]:
                image_file_hash_mismatches.append(physical[1])
            else:
                missing_image_files.append(physical[1])

    return Ss14PreparationReadiness(
        database_path=str(Path(database_path).resolve()),
        archive_sha256=plan.archive_sha256,
        preparation_manifest_sha256=plan.preparation_manifest_sha256,
        source_registered=source_registered,
        archive_blob_registered=archive_blob is not None,
        archive_blob_facts_match=archive_blob is not None and not archive_blob_mismatches,
        source_item_count=len(item_rows),
        pinned_source_item_count=len(pinned_item_ids),
        source_archive_link_count=source_archive_link_count,
        archive_retrieval_count=archive_retrieval_count,
        archive_inventory_present=inventory is not None,
        archive_inventory_exact=inventory is not None and not inventory_mismatches,
        expected_archive_member_count=plan.archive_member_count,
        indexed_archive_member_count=indexed_archive_member_count,
        required_member_count=plan.required_member_count,
        present_required_member_count=plan.required_member_count - len(missing_required_paths),
        selected_image_member_count=plan.selected_image_member_count,
        present_selected_image_blob_count=present_image_occurrences,
        unique_selected_image_count=plan.unique_selected_image_count,
        present_unique_selected_image_blob_count=len(registered_hashes),
        present_unique_selected_image_file_count=len(valid_file_hashes),
        selected_image_role_count=selected_role_count,
        media_observation_count=len(media_hashes),
        media_inspected_member_count=inspected_member_count,
        media_invalid_member_count=invalid_member_count,
        media_terminal_unique_image_count=len(media_hashes.union(invalid_hashes)),
        archive_blob_fact_mismatches=tuple(archive_blob_mismatches),
        archive_inventory_mismatches=tuple(inventory_mismatches),
        missing_required_member_paths=missing_required_paths,
        required_member_fact_mismatches=tuple(member_fact_mismatches),
        missing_selected_image_blobs=tuple(missing_image_blobs),
        selected_image_hash_mismatches=tuple(image_hash_mismatches),
        selected_image_blob_fact_mismatches=tuple(image_blob_fact_mismatches),
        missing_selected_image_files=tuple(sorted(set(missing_image_files))),
        selected_image_file_hash_mismatches=tuple(sorted(set(image_file_hash_mismatches))),
        media_invalid_members=tuple(invalid_members),
    )


def _archive_members(database: IndexDB, archive_sha256: str) -> dict[str, sqlite3.Row]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT am.ordinal, am.member_path, am.normalized_path,
                   am.extracted_blob_sha256,
                   b.sha256 AS registered_blob_sha256
            FROM archive_members am
            LEFT JOIN blobs b ON b.sha256=am.extracted_blob_sha256
            WHERE am.archive_blob_sha256=? ORDER BY am.ordinal
            """,
            (archive_sha256,),
        ).fetchall()
    members: dict[str, sqlite3.Row] = {}
    for row in rows:
        members[str(row["normalized_path"])] = row
        members[str(row["member_path"])] = row
    return members


def _item_id(database: IndexDB, archive_sha256: str) -> str:
    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT i.id FROM items i
            JOIN item_blobs ib ON ib.item_id=i.id
            WHERE i.source_id=? AND ib.blob_sha256=?
            ORDER BY i.id LIMIT 1
            """,
            (SOURCE_ID, archive_sha256),
        ).fetchone()
    if row is None:
        raise ValueError(
            f"SS14 archive has no indexed source item for {SOURCE_ID!r}: {archive_sha256}"
        )
    return str(row[0])


def _preflight(
    database: IndexDB,
    plan: Ss14ProjectionPlan,
) -> tuple[str, dict[str, sqlite3.Row]]:
    with database.connect() as connection:
        source = connection.execute(
            "SELECT 1 FROM sources WHERE id=?",
            (SOURCE_ID,),
        ).fetchone()
        inventory = connection.execute(
            "SELECT 1 FROM archive_inventories WHERE archive_blob_sha256=?",
            (plan.archive_sha256,),
        ).fetchone()
    if source is None:
        raise ValueError(f"SS14 source registry row is missing: {SOURCE_ID}")
    if inventory is None:
        raise ValueError(f"SS14 archive inventory is missing: {plan.archive_sha256}")
    item_id = _item_id(database, plan.archive_sha256)
    members = _archive_members(database, plan.archive_sha256)
    missing = [path for path in plan.required_member_paths if path not in members]
    if missing:
        raise ValueError("SS14 projection evidence members are missing: " + ", ".join(missing[:10]))
    for member_path, expected_hash in plan.required_source_image_hashes:
        member = members[member_path]
        actual_hash = member["extracted_blob_sha256"]
        if actual_hash is None:
            raise ValueError(f"SS14 source image is not extracted into CAS: {member_path}")
        if str(actual_hash) != expected_hash:
            raise ValueError(
                "SS14 source image CAS hash mismatch for "
                f"{member_path}: expected {expected_hash}, indexed {actual_hash}"
            )
        if member["registered_blob_sha256"] is None:
            raise ValueError(f"SS14 source image CAS blob is not registered: {member_path}")
    return item_id, members


def _rights_metadata(rights: Ss14ProjectionRights) -> dict[str, Any]:
    return {
        "scope": "per_rsi_pack",
        "license_expression": rights.license_expression,
        "copyright": rights.copyright,
        "rights_status": rights.rights_status,
        "metadata_evidence": asdict(rights.metadata_evidence),
        "upstream_references": [asdict(reference) for reference in rights.upstream_references],
        "lineage_keys": list(rights.lineage_keys),
        "upstream_asset_deduplication_keys": list(rights.upstream_asset_deduplication_keys),
        "rights_observation_added": False,
    }


def _sequence_metadata(
    plan: Ss14ProjectionPlan,
    record: Ss14ProjectionRecord,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "source_audit_record_sha256": plan.source_audit_record_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "rsi_path": record.rsi_path,
        "state_name": record.state_name,
        "entity_cue": record.entity_cue,
        "source_action_cue": record.source_action_cue,
        "action_cue_basis": record.action_cue_basis,
        "state_role": record.state_role,
        "state_role_basis": record.state_role_basis,
        "direction_index": record.direction_index,
        "source_direction": record.source_direction,
        "direction_count": record.direction_count,
        "delays_declared": record.delays_declared,
        "source_delays_seconds": [list(row) for row in record.source_delays_seconds],
        "engine_delays_seconds": list(record.engine_delays_seconds),
        "engine_source_cell_indices": list(record.engine_source_cell_indices),
        "duration_ms_per_occurrence": [frame.duration_milliseconds for frame in record.frames],
        "total_duration_ms": record.total_duration_milliseconds,
        "loop_semantics": record.loop_semantics,
        "loop_mode": "unknown",
        "loop_policy_inferred": False,
        "source_sheet_logical_path": record.source_sheet_logical_path,
        "source_sheet_member_path": record.source_sheet_member_path,
        "source_sheet_sha256": record.source_sheet_sha256,
        "source_sheet_dimensions": [record.source_sheet_width, record.source_sheet_height],
        "source_sheet_format": record.source_sheet_format,
        "source_sheet_mode": record.source_sheet_mode,
        "frame_size": [record.frame_width, record.frame_height],
        "grid": {
            "columns": record.grid_columns,
            "rows": record.grid_rows,
            "capacity": record.grid_capacity,
            "expected_source_cell_count": record.expected_source_cell_count,
            "unused_cell_count": 0,
        },
        "load_srgb": record.load_srgb,
        "meta_atlas": record.meta_atlas,
        "rsic": record.rsic,
        "exact_engine_timing": True,
        "state_occurrence_order_preserved": True,
        "native_source_rectangles_preserved": True,
        "clipping_or_repair_applied": False,
        "image_payload_deduplication_key": record.image_payload_deduplication_key,
        "rights_scope": _rights_metadata(record.rights),
        "engine_evidence": {
            "robust_toolbox_commit": plan.robust_toolbox_commit,
            "robust_toolbox_archive_sha256": plan.robust_toolbox_archive_sha256,
            "urls": list(plan.engine_evidence_urls),
        },
    }


def _occurrence_specs(
    plan: Ss14ProjectionPlan,
    record: Ss14ProjectionRecord,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    common = {
        "projection_version": PROJECTION_VERSION,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "rsi_path": record.rsi_path,
        "state_name": record.state_name,
        "direction_index": record.direction_index,
        "source_direction": record.source_direction,
    }
    occurrences: list[tuple[str, str, dict[str, Any]]] = [
        (
            record.source_sheet_member_path,
            "ss14_rsi_state_image",
            {
                **common,
                "source_sheet_sha256": record.source_sheet_sha256,
                "image_payload_deduplication_key": record.image_payload_deduplication_key,
                "rights_scope": _rights_metadata(record.rights),
            },
        ),
        (
            record.rights.metadata_evidence.member_path,
            "ss14_rsi_metadata_and_per_pack_rights",
            {
                **common,
                "metadata_evidence_sha256": record.rights.metadata_evidence.sha256,
                "rights_scope": _rights_metadata(record.rights),
            },
        ),
        (
            plan.repository_rights.root_license.member_path,
            "ss14_repository_license_scope_evidence",
            {
                **common,
                "repository_license_expression": (
                    plan.repository_rights.repository_license_expression
                ),
                "repository_license_scope": plan.repository_rights.repository_license_scope,
                "root_license_sha256": plan.repository_rights.root_license.sha256,
                "per_pack_rights_override_repository_default": True,
            },
        ),
    ]
    if plan.repository_rights.rsi_schema is not None:
        occurrences.append(
            (
                plan.repository_rights.rsi_schema.member_path,
                "ss14_rsi_schema_evidence",
                {
                    **common,
                    "schema_sha256": plan.repository_rights.rsi_schema.sha256,
                },
            )
        )
    occurrences.extend(
        (
            evidence.member_path,
            "ss14_complete_entity_role_evidence",
            {
                **common,
                "evidence_sha256": evidence.sha256,
                "evidence_purpose": evidence.purpose,
                "state_role": record.state_role,
                "state_role_basis": record.state_role_basis,
            },
        )
        for evidence in plan.classification_evidence
    )
    return tuple(occurrences)


def _frame_metadata(
    plan: Ss14ProjectionPlan,
    record: Ss14ProjectionRecord,
    frame: Ss14ProjectionFrame,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "rsi_path": record.rsi_path,
        "state_name": record.state_name,
        "source_action_cue": record.source_action_cue,
        "action_cue_basis": record.action_cue_basis,
        "source_sheet_member_path": record.source_sheet_member_path,
        "source_sheet_sha256": record.source_sheet_sha256,
        "source_cell_index": frame.source_cell_index,
        "source_direction_frame_index": frame.source_direction_frame_index,
        "direction_index": frame.direction_index,
        "source_direction": frame.source_direction,
        "source_delay_seconds": frame.source_delay_seconds,
        "engine_delay_seconds": frame.engine_delay_seconds,
        "duration_milliseconds": frame.duration_milliseconds,
        "frame_rect": {
            "left": frame.left,
            "top": frame.top,
            "right": frame.right,
            "bottom": frame.bottom,
            "width": frame.right - frame.left,
            "height": frame.bottom - frame.top,
            "column": frame.left // record.frame_width,
            "row": frame.top // record.frame_height,
            "coordinate_space": "source_sheet",
        },
        "exact_engine_timing": True,
        "native_source_rectangle": True,
        "clipping_or_repair_applied": False,
        "image_payload_deduplication_key": record.image_payload_deduplication_key,
        "rights_scope": _rights_metadata(record.rights),
    }


def project_ss14_audit(
    database: IndexDB,
    plan: Ss14ProjectionPlan,
    taxonomy: Taxonomy,
) -> Ss14ProjectionResult:
    """Idempotently project a precomputed safe plan into a prepared index."""

    if taxonomy.version != plan.taxonomy_version:
        raise ValueError(
            "SS14 projection taxonomy version mismatch: "
            f"plan {plan.taxonomy_version!r}, runtime {taxonomy.version!r}"
        )
    if tuple(sorted(taxonomy.action_to_family)) != plan.taxonomy_action_values:
        raise ValueError("SS14 projection taxonomy action vocabulary has changed")
    database.initialize()
    item_id, members = _preflight(database, plan)
    manifest_sha256 = plan.projection_manifest_sha256
    entity_ids: dict[str, str] = {}
    created_sequences = 0
    reused_sequences = 0
    occurrence_links = 0

    for record in plan.records:
        entity = record.entity
        entity_id = entity_ids.get(entity.entity_external_key)
        if entity_id is None:
            normalized_entity = taxonomy.normalize_entity_class(entity.entity_class)
            if normalized_entity.value == "unknown":
                raise ValueError(
                    f"SS14 projected entity class became ambiguous: {entity.entity_class}"
                )
            entity_id = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=entity.entity_external_key,
                representative_item_id=item_id,
                display_name=entity.display_name,
                entity_class=normalized_entity.value,
                entity_subclass=entity.category,
                species_or_type=entity.entity_cue,
                taxonomy_version=taxonomy.version,
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "projection_manifest_sha256": manifest_sha256,
                    "source_audit_record_sha256": plan.source_audit_record_sha256,
                    "archive_sha256": plan.archive_sha256,
                    "repository_commit": plan.repository_commit,
                    "rsi_path": entity.rsi_path,
                    "entity_cue": entity.entity_cue,
                    "category": entity.category,
                    "adapter_entity_class": entity.entity_class,
                    "normalized_entity_class": normalized_entity.value,
                    "entity_class_basis": entity.entity_class_basis,
                    "classification_method": normalized_entity.method,
                    "classification_confidence": normalized_entity.confidence,
                    "rights_scope": _rights_metadata(entity.rights),
                },
            )
            entity_ids[entity.entity_external_key] = entity_id

        motion = taxonomy.motion_condition(
            action=record.source_action_cue,
            direction=record.source_direction,
            view=None,
        )
        if motion.normalized_action != record.source_action_cue:
            raise ValueError(
                "SS14 projection action was not taxonomy-canonical at write time: "
                f"{record.source_action_cue!r}"
            )
        sequence_id = database.find_sequence_by_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
        )
        sequence_arguments = {
            "source_blob_sha256": record.source_sheet_sha256,
            "extraction_method": PROJECTION_VERSION,
            "extraction_confidence": 1.0,
            "width": record.frame_width,
            "height": record.frame_height,
            "frame_count": record.frame_count,
            "loop_mode": "unknown",
            "action": motion.normalized_action,
            "direction": motion.direction,
            "quality_tier": "F0_lossless_rsi_cell_exact_engine_timing",
            "metadata": _sequence_metadata(plan, record, manifest_sha256),
        }
        if sequence_id is None:
            sequence_id = database.create_sequence(item_id=item_id, **sequence_arguments)
            created_sequences += 1
        else:
            database.update_sequence_facts(sequence_id=sequence_id, **sequence_arguments)
            reused_sequences += 1
        database.register_sequence_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
            sequence_id=sequence_id,
        )
        database.link_sequence_subject(
            sequence_id=sequence_id,
            entity_id=entity_id,
            role="primary",
            metadata={
                "rsi_path": record.rsi_path,
                "state_name": record.state_name,
                "entity_cue": record.entity_cue,
                "state_role": record.state_role,
                "state_role_basis": record.state_role_basis,
                "whole_entity_projection_candidate": True,
                "runtime_composite_completeness_verified": False,
                "component_layers_composited": False,
                "rights_scope": _rights_metadata(record.rights),
            },
        )
        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version=taxonomy.version,
            source_action=record.state_name,
            normalized_action=motion.normalized_action,
            action_family=motion.action_family,
            annotation_method=PROJECTION_VERSION,
            view=motion.view,
            direction=motion.direction,
            loopable=None,
            cycle_frames=None,
            phase_zero_frame=None,
            confidence=motion.confidence,
            conditioning={
                "source_state_name": record.state_name,
                "source_action_cue": record.source_action_cue,
                "action_cue_basis": record.action_cue_basis,
                "taxonomy_normalization_method": motion.method,
                "direction_index": record.direction_index,
                "source_direction": record.source_direction,
                "timing_known": True,
                "exact_engine_timing": True,
                "source_delays_seconds": [list(row) for row in record.source_delays_seconds],
                "engine_delays_seconds": list(record.engine_delays_seconds),
                "engine_source_cell_indices": list(record.engine_source_cell_indices),
                "duration_ms_per_occurrence": [
                    frame.duration_milliseconds for frame in record.frames
                ],
                "loop_semantics": record.loop_semantics,
                "loop_policy_inferred": False,
                "state_occurrence_order_preserved": True,
                "rights_scope": _rights_metadata(record.rights),
            },
        )

        occurrence_specs = _occurrence_specs(plan, record)
        for member_path, role, metadata in occurrence_specs:
            database.link_sequence_occurrence(
                sequence_id=sequence_id,
                archive_blob_sha256=plan.archive_sha256,
                archive_member_ordinal=int(members[member_path]["ordinal"]),
                occurrence_role=role,
                metadata=metadata,
            )
            occurrence_links += 1

        for frame in record.frames:
            database.add_sequence_frame(
                sequence_id=sequence_id,
                ordinal=frame.ordinal,
                source_blob_sha256=record.source_sheet_sha256,
                source_frame_index=frame.source_cell_index,
                duration_ms=frame.duration_milliseconds,
                phase=None,
                direction=motion.direction,
                view=motion.view,
                metadata=_frame_metadata(plan, record, frame),
            )

    return Ss14ProjectionResult(
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=manifest_sha256,
        projected_states=plan.projected_state_count,
        projected_entities=plan.projected_entity_count,
        projected_sequences=plan.projected_sequence_count,
        projected_frame_occurrences=plan.projected_frame_occurrence_count,
        projected_animated_sequences=plan.projected_animated_sequence_count,
        projected_static_sequences=plan.projected_static_sequence_count,
        created_sequences=created_sequences,
        reused_sequences=reused_sequences,
        occurrence_links=occurrence_links,
        excluded_states=plan.excluded_state_count,
    )


def ingest_known_ss14_sequences(
    database: IndexDB,
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> Ss14ProjectionResult:
    """Audit and project only the exact pinned SS14 snapshot."""

    plan = plan_known_ss14_projection(archive_path, taxonomy)
    if plan.archive_sha256 != EXPECTED_SS14_ARCHIVE_SHA256 or plan.repository_commit != SS14_COMMIT:
        raise ValueError("Refusing SS14 projection for an unexpected archive or commit")
    if plan.robust_toolbox_commit != ROBUST_TOOLBOX_COMMIT:
        raise ValueError("Refusing SS14 projection for an unexpected RobustToolbox contract")
    return project_ss14_audit(database, plan, taxonomy)


__all__ = [
    "EXPECTED_PINNED_ARCHIVE_INVENTORY_SHA256",
    "EXPECTED_PINNED_ARCHIVE_MEMBER_COUNT",
    "EXPECTED_PINNED_ARCHIVE_SIZE_BYTES",
    "EXPECTED_PINNED_PREPARATION_MANIFEST_SHA256",
    "EXPECTED_PINNED_ENTITY_COUNT",
    "EXPECTED_PINNED_EXCLUSION_COUNT",
    "EXPECTED_PINNED_FRAME_OCCURRENCE_COUNT",
    "EXPECTED_PINNED_REQUIRED_MEMBER_COUNT",
    "EXPECTED_PINNED_SEQUENCE_COUNT",
    "EXPECTED_PINNED_STATE_COUNT",
    "PREPARATION_VERSION",
    "PROJECTION_VERSION",
    "SOURCE_ID",
    "SS14_ITEM_EXTERNAL_ID",
    "SS14_MEDIA_INSPECTOR_VERSION",
    "SS14_SELECTED_IMAGE_ROLE",
    "Ss14PreparationMember",
    "Ss14PreparationPlan",
    "Ss14PreparationReadiness",
    "Ss14ProjectionEntity",
    "Ss14ProjectionExclusion",
    "Ss14ProjectionFrame",
    "Ss14ProjectionPlan",
    "Ss14ProjectionReadiness",
    "Ss14ProjectionRecord",
    "Ss14ProjectionResult",
    "Ss14ProjectionRights",
    "check_ss14_preparation_readiness",
    "check_ss14_projection_readiness",
    "ingest_known_ss14_sequences",
    "plan_known_ss14_preparation",
    "plan_known_ss14_projection",
    "plan_ss14_preparation",
    "plan_ss14_projection",
    "project_ss14_audit",
]
