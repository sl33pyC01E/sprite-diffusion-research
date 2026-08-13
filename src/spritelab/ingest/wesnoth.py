"""Plan and project audited Battle for Wesnoth primary animation tracks.

Planning is pure and readiness uses a query-only SQLite connection.  Projection
admits exactly the adapter's ``safe_primary_source_sequence`` subset: literal,
ordered, exactly timed primary PNG frames with no WML branch, unexpanded macro,
image path function, separate image modifier, missing member, or timing repair.

Wesnoth can render auxiliary projectiles, haloes, offsets, layers, and runtime
image transformations around a unit body.  Those declarations remain attached
as evidence, but this projection never composites them into the primary pixels.
Excluded declarations are retained verbatim in the deterministic plan instead
of being repaired or approximately rendered.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spritelab.adapters.wesnoth import (
    EXPECTED_WESNOTH_ARCHIVE_SHA256,
    WESNOTH_COMMIT,
    AnimationRecord,
    EntityRecord,
    FrameDeclaration,
    RightsAudit,
    SourceLocation,
    WesnothArchiveAudit,
    WmlAttribute,
    audit_known_wesnoth_archive,
    primary_declarations_for,
)
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

SOURCE_ID = "wesnoth"
PROJECTION_VERSION = "wesnoth_literal_primary_wml_projection_v1"
RIGHTS_SCOPE_CAVEAT = (
    "Repository and art-collection statements are not per-asset license or attribution claims."
)
EXPECTED_PINNED_SEQUENCE_COUNT = 604
EXPECTED_PINNED_FRAME_OCCURRENCE_COUNT = 2_526
EXPECTED_PINNED_ENTITY_COUNT = 248


@dataclass(frozen=True)
class WesnothProjectionEntity:
    """Exact source identity for one projected unit declaration."""

    entity_external_key: str
    unit_id: str
    name_literal: str | None
    race_literal: str | None
    entity_class: str
    entity_class_basis: str
    config_path: str
    member_path: str
    location: SourceLocation
    raw_attributes: tuple[WmlAttribute, ...]
    base_unit_ids: tuple[str, ...]
    base_image_literal: str | None
    profile_literal: str | None
    macro_invocations: tuple[str, ...]
    unresolved_inheritance: bool
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True)
class WesnothProjectionFrame:
    """One exact standalone source-image occurrence in temporal order."""

    ordinal: int
    declaration_ordinal: int
    ordinal_in_expression: int
    frame_tag: str
    image_attribute: str
    source_expression: str
    logical_image_path: str
    source_member_path: str
    source_blob_sha256: str
    source_image_width: int
    source_image_height: int
    source_image_resolution_basis: str
    duration_milliseconds: int
    declaration_location: SourceLocation
    raw_attributes: tuple[WmlAttribute, ...]
    context_attributes: tuple[WmlAttribute, ...]
    branch_path: tuple[str, ...]
    directions: tuple[str, ...]
    start_time_literal: str | None
    duration_literal: str | None
    begin_literal: str | None
    end_literal: str | None
    layer_literal: str | None
    offset_literal: str | None
    x_literal: str | None
    y_literal: str | None
    directional_x_literal: str | None
    directional_y_literal: str | None
    auto_hflip_literal: str | None
    effective_auto_hflip: bool
    auto_vflip_literal: str | None
    effective_auto_vflip: bool
    primary_literal: str | None
    inline_modifiers: None
    separate_image_mod: None
    exact_timing: bool
    lossless_source_pixels: bool
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True)
class WesnothProjectionRecord:
    """One admitted literal primary-unit animation."""

    sequence_source_key: str
    entity: WesnothProjectionEntity
    source_tag: str
    variant_path: tuple[str, ...]
    normalized_action: str | None
    normalized_action_basis: str
    location: SourceLocation
    raw_attributes: tuple[WmlAttribute, ...]
    apply_to_literal: str | None
    attack_name_filters: tuple[str, ...]
    attack_range_filters: tuple[str, ...]
    directions: tuple[str, ...]
    start_time_literal: str | None
    cycles_literal: str | None
    effective_cycles: bool
    loop_mode: str
    loop_basis: str
    macro_invocations: tuple[str, ...]
    frames: tuple[WesnothProjectionFrame, ...]
    auxiliary_frame_declarations: tuple[FrameDeclaration, ...]
    primary_timeline_exact: bool
    safe_primary_source_sequence: bool

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def sequence_width(self) -> int:
        return max(frame.source_image_width for frame in self.frames)

    @property
    def sequence_height(self) -> int:
        return max(frame.source_image_height for frame in self.frames)

    @property
    def total_duration_milliseconds(self) -> int:
        return sum(frame.duration_milliseconds for frame in self.frames)

    @property
    def source_dimensions_consistent(self) -> bool:
        return (
            len({(frame.source_image_width, frame.source_image_height) for frame in self.frames})
            == 1
        )

    @property
    def source_direction_groups(self) -> tuple[tuple[str, ...], ...]:
        groups: list[tuple[str, ...]] = []
        for frame in self.frames:
            if frame.directions and frame.directions not in groups:
                groups.append(frame.directions)
        if not groups and self.directions:
            groups.append(self.directions)
        return tuple(groups)

    @property
    def direction_hint(self) -> str | None:
        groups = self.source_direction_groups
        if len(groups) == 1 and len(groups[0]) == 1:
            return groups[0][0]
        return None


@dataclass(frozen=True)
class WesnothProjectionExclusion:
    """One quarantined WML animation, retained without repair."""

    sequence_source_key: str
    entity_external_key: str
    unit_id: str
    config_path: str
    member_path: str
    entity_line_number: int
    animation: AnimationRecord
    reasons: tuple[str, ...]
    transformed_primary_frame_count: int
    unresolved_primary_frame_count: int
    unsafe_primary_frame_count: int
    auxiliary_frame_declaration_count: int

    @property
    def primary_frame_count(self) -> int:
        return self.animation.primary_frame_count


@dataclass(frozen=True)
class WesnothProjectionPlan:
    """Pure, deterministic projection plan derived from one immutable audit."""

    archive_sha256: str
    repository_commit: str
    archive_root: str
    source_audit_record_sha256: str
    records: tuple[WesnothProjectionRecord, ...]
    exclusions: tuple[WesnothProjectionExclusion, ...]
    rights: RightsAudit
    engine_evidence_paths: tuple[str, ...]

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
    def projected_loop_count(self) -> int:
        return sum(record.loop_mode == "loop" for record in self.records)

    @property
    def projected_one_shot_count(self) -> int:
        return sum(record.loop_mode == "one_shot" for record in self.records)

    @property
    def projected_normalized_action_count(self) -> int:
        return sum(record.normalized_action is not None for record in self.records)

    @property
    def projected_unknown_action_count(self) -> int:
        return sum(record.normalized_action is None for record in self.records)

    @property
    def projected_exact_single_direction_count(self) -> int:
        return sum(record.direction_hint is not None for record in self.records)

    @property
    def projected_auxiliary_declaration_count(self) -> int:
        return sum(len(record.auxiliary_frame_declarations) for record in self.records)

    @property
    def projected_occurrence_link_count(self) -> int:
        fixed_evidence_count = 1 + len(self.engine_evidence_paths) + len(self.rights.evidence)
        return sum(
            fixed_evidence_count + len({frame.source_member_path for frame in record.frames})
            for record in self.records
        )

    @property
    def excluded_candidate_sequence_count(self) -> int:
        return len(self.exclusions)

    @property
    def excluded_candidate_frame_occurrence_count(self) -> int:
        return sum(exclusion.primary_frame_count for exclusion in self.exclusions)

    @property
    def excluded_transformed_primary_frame_count(self) -> int:
        return sum(exclusion.transformed_primary_frame_count for exclusion in self.exclusions)

    @property
    def excluded_macro_animation_count(self) -> int:
        return sum(bool(exclusion.animation.macro_invocations) for exclusion in self.exclusions)

    @property
    def excluded_conditional_animation_count(self) -> int:
        return sum(
            any(
                declaration.branch_path
                for declaration in primary_declarations_for(exclusion.animation)
            )
            for exclusion in self.exclusions
        )

    @property
    def required_source_image_hashes(self) -> tuple[tuple[str, str], ...]:
        values: dict[str, str] = {}
        for record in self.records:
            for frame in record.frames:
                previous = values.setdefault(frame.source_member_path, frame.source_blob_sha256)
                if previous != frame.source_blob_sha256:
                    raise ValueError(
                        "One Wesnoth image member has multiple audited hashes: "
                        f"{frame.source_member_path!r}"
                    )
        return tuple(sorted(values.items()))

    @property
    def required_member_paths(self) -> tuple[str, ...]:
        paths = {record.entity.member_path for record in self.records}
        paths.update(path for path, _ in self.required_source_image_hashes)
        paths.update(self.engine_evidence_paths)
        paths.update(evidence.member_path for evidence in self.rights.evidence)
        return tuple(sorted(paths))

    @property
    def projection_manifest_sha256(self) -> str:
        payload = {
            "projection_version": PROJECTION_VERSION,
            "archive_sha256": self.archive_sha256,
            "repository_commit": self.repository_commit,
            "archive_root": self.archive_root,
            "source_audit_record_sha256": self.source_audit_record_sha256,
            "records": [asdict(record) for record in self.records],
            "exclusions": [asdict(exclusion) for exclusion in self.exclusions],
            "rights": asdict(self.rights),
            "engine_evidence_paths": list(self.engine_evidence_paths),
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class WesnothProjectionReadiness:
    """Query-only report for all projection prerequisites."""

    database_path: str
    archive_sha256: str
    projection_manifest_sha256: str
    archive_blob_present: bool
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
            self.archive_blob_present
            and self.source_item_count > 0
            and not self.missing_member_paths
            and not self.missing_source_image_blobs
            and not self.source_image_hash_mismatches
        )


@dataclass(frozen=True)
class WesnothProjectionResult:
    """Effects of one idempotent projection call."""

    archive_sha256: str
    projection_manifest_sha256: str
    projected_entities: int
    projected_sequences: int
    projected_frame_occurrences: int
    projected_loops: int
    projected_one_shots: int
    projected_normalized_actions: int
    projected_unknown_actions: int
    created_sequences: int
    reused_sequences: int
    occurrence_links: int
    excluded_candidate_sequences: int
    excluded_candidate_frame_occurrences: int
    excluded_transformed_primary_frames: int
    rights_observations_added: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _stable_json_key(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}:" + json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _entity_external_key(audit: WesnothArchiveAudit, entity: EntityRecord) -> str:
    return _stable_json_key(
        "wesnoth-entity-v1",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "config_path": entity.config_path,
            "member_path": entity.member_path,
            "declaration_line": entity.location.line_number,
            "unit_id": entity.unit_id,
        },
    )


def _sequence_source_key(
    audit: WesnothArchiveAudit,
    entity: EntityRecord,
    animation: AnimationRecord,
) -> str:
    return _stable_json_key(
        "wesnoth-sequence-v1",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "config_path": entity.config_path,
            "entity_declaration_line": entity.location.line_number,
            "unit_id": entity.unit_id,
            "variant_path": animation.variant_path,
            "animation_line": animation.location.line_number,
            "source_tag": animation.source_tag,
        },
    )


def _projection_entity(
    audit: WesnothArchiveAudit,
    entity: EntityRecord,
) -> WesnothProjectionEntity:
    return WesnothProjectionEntity(
        entity_external_key=_entity_external_key(audit, entity),
        unit_id=entity.unit_id,
        name_literal=entity.name_literal,
        race_literal=entity.race_literal,
        entity_class=entity.entity_class,
        entity_class_basis=entity.entity_class_basis,
        config_path=entity.config_path,
        member_path=entity.member_path,
        location=entity.location,
        raw_attributes=entity.raw_attributes,
        base_unit_ids=entity.base_unit_ids,
        base_image_literal=entity.base_image_literal,
        profile_literal=entity.profile_literal,
        macro_invocations=entity.macro_invocations,
        unresolved_inheritance=entity.unresolved_inheritance,
        quarantine_reasons=entity.quarantine_reasons,
    )


def _projection_frame(
    declaration: FrameDeclaration,
    *,
    declaration_ordinal: int,
    frame_ordinal: int,
    expression_ordinal: int,
) -> WesnothProjectionFrame:
    frame = declaration.frames[expression_ordinal]
    resolution = frame.resolution
    if (
        frame.quarantine_reasons
        or not frame.lossless_source_pixels
        or not frame.exact_timing
        or resolution.selected_member_path is None
        or resolution.sha256 is None
        or resolution.width is None
        or resolution.height is None
        or frame.duration_milliseconds is None
        or frame.logical_path is None
        or frame.inline_modifiers is not None
        or frame.separate_image_mod is not None
    ):
        raise ValueError(
            "Adapter marked an unsafe Wesnoth frame as a safe projection candidate: "
            f"{declaration.location.config_path}:{declaration.location.line_number}"
        )
    return WesnothProjectionFrame(
        ordinal=frame_ordinal,
        declaration_ordinal=declaration_ordinal,
        ordinal_in_expression=frame.ordinal_in_expression,
        frame_tag=declaration.frame_tag,
        image_attribute=declaration.image_attribute,
        source_expression=frame.source_expression,
        logical_image_path=frame.logical_path,
        source_member_path=resolution.selected_member_path,
        source_blob_sha256=resolution.sha256,
        source_image_width=resolution.width,
        source_image_height=resolution.height,
        source_image_resolution_basis=resolution.resolution_basis,
        duration_milliseconds=frame.duration_milliseconds,
        declaration_location=declaration.location,
        raw_attributes=declaration.raw_attributes,
        context_attributes=declaration.context_attributes,
        branch_path=declaration.branch_path,
        directions=declaration.directions,
        start_time_literal=declaration.start_time_literal,
        duration_literal=declaration.duration_literal,
        begin_literal=declaration.begin_literal,
        end_literal=declaration.end_literal,
        layer_literal=declaration.layer_literal,
        offset_literal=declaration.offset_literal,
        x_literal=declaration.x_literal,
        y_literal=declaration.y_literal,
        directional_x_literal=declaration.directional_x_literal,
        directional_y_literal=declaration.directional_y_literal,
        auto_hflip_literal=declaration.auto_hflip_literal,
        effective_auto_hflip=declaration.effective_auto_hflip,
        auto_vflip_literal=declaration.auto_vflip_literal,
        effective_auto_vflip=declaration.effective_auto_vflip,
        primary_literal=declaration.primary_literal,
        inline_modifiers=None,
        separate_image_mod=None,
        exact_timing=True,
        lossless_source_pixels=True,
        quarantine_reasons=(),
    )


def _projection_record(
    audit: WesnothArchiveAudit,
    entity: EntityRecord,
    animation: AnimationRecord,
) -> WesnothProjectionRecord:
    if not animation.safe_primary_source_sequence or not animation.primary_timeline_exact:
        raise ValueError("Unsafe Wesnoth animation passed to projection-record construction")
    frames: list[WesnothProjectionFrame] = []
    for declaration_ordinal, declaration in enumerate(primary_declarations_for(animation)):
        if declaration.quarantine_reasons or not declaration.declaration_exact:
            raise ValueError("Unsafe Wesnoth declaration passed to projection-record construction")
        for expression_ordinal in range(len(declaration.frames)):
            frames.append(
                _projection_frame(
                    declaration,
                    declaration_ordinal=declaration_ordinal,
                    frame_ordinal=len(frames),
                    expression_ordinal=expression_ordinal,
                )
            )
    if not frames:
        raise ValueError("Safe Wesnoth projection record has no frames")
    auxiliary = tuple(
        declaration
        for declaration in animation.frame_declarations
        if declaration.render_role != "primary_unit"
    )
    return WesnothProjectionRecord(
        sequence_source_key=_sequence_source_key(audit, entity, animation),
        entity=_projection_entity(audit, entity),
        source_tag=animation.source_tag,
        variant_path=animation.variant_path,
        normalized_action=animation.normalized_action,
        normalized_action_basis=animation.normalized_action_basis,
        location=animation.location,
        raw_attributes=animation.raw_attributes,
        apply_to_literal=animation.apply_to_literal,
        attack_name_filters=animation.attack_name_filters,
        attack_range_filters=animation.attack_range_filters,
        directions=animation.directions,
        start_time_literal=animation.start_time_literal,
        cycles_literal=animation.cycles_literal,
        effective_cycles=animation.effective_cycles,
        loop_mode=animation.loop_mode,
        loop_basis=animation.loop_basis,
        macro_invocations=animation.macro_invocations,
        frames=tuple(frames),
        auxiliary_frame_declarations=auxiliary,
        primary_timeline_exact=animation.primary_timeline_exact,
        safe_primary_source_sequence=animation.safe_primary_source_sequence,
    )


def _projection_exclusion(
    audit: WesnothArchiveAudit,
    entity: EntityRecord,
    animation: AnimationRecord,
) -> WesnothProjectionExclusion:
    primary = primary_declarations_for(animation)
    primary_frames = tuple(frame for declaration in primary for frame in declaration.frames)
    return WesnothProjectionExclusion(
        sequence_source_key=_sequence_source_key(audit, entity, animation),
        entity_external_key=_entity_external_key(audit, entity),
        unit_id=entity.unit_id,
        config_path=entity.config_path,
        member_path=entity.member_path,
        entity_line_number=entity.location.line_number,
        animation=animation,
        reasons=animation.quarantine_reasons,
        transformed_primary_frame_count=sum(
            bool(frame.inline_modifiers or frame.separate_image_mod) for frame in primary_frames
        ),
        unresolved_primary_frame_count=sum(
            frame.resolution.selected_member_path is None for frame in primary_frames
        ),
        unsafe_primary_frame_count=sum(bool(frame.quarantine_reasons) for frame in primary_frames),
        auxiliary_frame_declaration_count=sum(
            declaration.render_role != "primary_unit"
            for declaration in animation.frame_declarations
        ),
    )


def plan_wesnoth_projection(audit: WesnothArchiveAudit) -> WesnothProjectionPlan:
    """Build a deterministic write-free plan and a complete exclusion ledger."""

    records: list[WesnothProjectionRecord] = []
    exclusions: list[WesnothProjectionExclusion] = []
    for entity in audit.entities:
        for animation in entity.animations:
            if animation.safe_primary_source_sequence:
                records.append(_projection_record(audit, entity, animation))
            else:
                exclusions.append(_projection_exclusion(audit, entity, animation))
    records.sort(key=lambda record: record.sequence_source_key)
    exclusions.sort(key=lambda exclusion: exclusion.sequence_source_key)
    keys = [record.sequence_source_key for record in records]
    keys.extend(exclusion.sequence_source_key for exclusion in exclusions)
    if len(keys) != len(set(keys)):
        raise ValueError("Wesnoth projection source keys are not unique")
    if len(records) + len(exclusions) != audit.counts.animation_records:
        raise AssertionError("Wesnoth projection does not partition every audited animation")
    if sum(record.frame_count for record in records) != audit.counts.safe_primary_frame_occurrences:
        raise AssertionError("Wesnoth safe frame count does not reconcile with the source audit")
    return WesnothProjectionPlan(
        archive_sha256=audit.archive_sha256,
        repository_commit=audit.commit,
        archive_root=audit.archive_root,
        source_audit_record_sha256=audit.audit_record_sha256,
        records=tuple(records),
        exclusions=tuple(exclusions),
        rights=audit.rights,
        engine_evidence_paths=audit.engine_evidence_paths,
    )


def plan_known_wesnoth_projection(archive_path: str | Path) -> WesnothProjectionPlan:
    """Audit the exact pinned CAS archive and build a write-free plan."""

    plan = plan_wesnoth_projection(audit_known_wesnoth_archive(Path(archive_path)))
    expected = (
        EXPECTED_PINNED_SEQUENCE_COUNT,
        EXPECTED_PINNED_FRAME_OCCURRENCE_COUNT,
        EXPECTED_PINNED_ENTITY_COUNT,
    )
    actual = (
        plan.projected_sequence_count,
        plan.projected_frame_occurrence_count,
        plan.projected_entity_count,
    )
    if actual != expected:
        raise ValueError(
            f"Pinned Wesnoth projection count drift: expected {expected}, got {actual}"
        )
    return plan


def _readonly_connection(database_path: str | Path) -> sqlite3.Connection:
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def check_wesnoth_projection_readiness(
    database_path: str | Path,
    plan: WesnothProjectionPlan,
) -> WesnothProjectionReadiness:
    """Inspect exact prerequisites without creating a journal or modifying SQLite."""

    required_paths = plan.required_member_paths
    expected_images = dict(plan.required_source_image_hashes)
    with _readonly_connection(database_path) as connection:
        archive_blob_present = (
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
    hash_mismatches: list[str] = []
    for member_path, expected_hash in sorted(expected_images.items()):
        row = members.get(member_path)
        if row is None:
            continue
        actual_hash = row["extracted_blob_sha256"]
        registered_hash = row["registered_blob_sha256"]
        if actual_hash is None or registered_hash is None:
            missing_blobs.append(member_path)
        elif str(actual_hash) != expected_hash:
            hash_mismatches.append(
                f"{member_path}: expected {expected_hash}, indexed {actual_hash}"
            )
    return WesnothProjectionReadiness(
        database_path=str(Path(database_path).resolve()),
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=plan.projection_manifest_sha256,
        archive_blob_present=archive_blob_present,
        source_item_count=source_item_count,
        required_member_count=len(required_paths),
        present_member_count=len(required_paths) - len(missing_paths),
        required_source_image_count=len(expected_images),
        present_source_image_blob_count=(
            len(expected_images) - len(missing_blobs) - len(hash_mismatches)
        ),
        missing_member_paths=missing_paths,
        missing_source_image_blobs=tuple(missing_blobs),
        source_image_hash_mismatches=tuple(hash_mismatches),
    )


def _archive_members(database: IndexDB, archive_sha256: str) -> dict[str, sqlite3.Row]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT am.ordinal, am.member_path, am.normalized_path,
                   am.extracted_blob_sha256,
                   b.sha256 AS registered_blob_sha256
            FROM archive_members AS am
            LEFT JOIN blobs AS b ON b.sha256=am.extracted_blob_sha256
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
            f"Wesnoth archive has no indexed source item for {SOURCE_ID!r}: {archive_sha256}"
        )
    return str(row[0])


def _preflight(
    database: IndexDB,
    plan: WesnothProjectionPlan,
) -> tuple[str, dict[str, sqlite3.Row]]:
    with database.connect() as connection:
        inventory = connection.execute(
            "SELECT 1 FROM archive_inventories WHERE archive_blob_sha256=?",
            (plan.archive_sha256,),
        ).fetchone()
    if inventory is None:
        raise ValueError(f"Wesnoth archive inventory is missing: {plan.archive_sha256}")
    item_id = _item_id(database, plan.archive_sha256)
    members = _archive_members(database, plan.archive_sha256)
    missing = [path for path in plan.required_member_paths if path not in members]
    if missing:
        raise ValueError(
            "Wesnoth projection evidence members are missing: " + ", ".join(missing[:10])
        )
    for member_path, expected_hash in plan.required_source_image_hashes:
        member = members[member_path]
        actual_hash = member["extracted_blob_sha256"]
        if actual_hash is None:
            raise ValueError(f"Wesnoth source image is not extracted into CAS: {member_path}")
        if str(actual_hash) != expected_hash:
            raise ValueError(
                "Wesnoth source image CAS hash mismatch for "
                f"{member_path}: expected {expected_hash}, indexed {actual_hash}"
            )
        if member["registered_blob_sha256"] is None:
            raise ValueError(f"Wesnoth source image CAS blob is not registered: {member_path}")
    return item_id, members


def _rights_scope_metadata(plan: WesnothProjectionPlan) -> dict[str, Any]:
    return {
        "scope": "repository_and_art_collection_only_not_asset_level",
        "caveat": RIGHTS_SCOPE_CAVEAT,
        "repository_license_expression": plan.rights.repository_license_expression,
        "art_scope_statement": plan.rights.art_scope_statement,
        "projection_status": plan.rights.projection_status,
        "copyrights_csv_rows": plan.rights.copyrights_csv_rows,
        "copyrights_csv_image_rows": plan.rights.copyrights_csv_image_rows,
        "evidence": [asdict(evidence) for evidence in plan.rights.evidence],
        "per_asset_manifest_present": False,
        "asset_license_expression": None,
        "asset_creator": None,
        "rights_observation_added": False,
    }


def _frame_phase(ordinal: int, frame_count: int, loop_mode: str) -> float:
    if frame_count <= 1:
        return 0.0
    if loop_mode == "loop":
        return ordinal / frame_count
    return ordinal / (frame_count - 1)


def _attributes(values: tuple[WmlAttribute, ...]) -> list[dict[str, Any]]:
    return [asdict(value) for value in values]


def _sequence_metadata(
    plan: WesnothProjectionPlan,
    record: WesnothProjectionRecord,
    manifest_sha256: str,
) -> dict[str, Any]:
    rights_scope = _rights_scope_metadata(plan)
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "source_audit_record_sha256": plan.source_audit_record_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "unit_id": record.entity.unit_id,
        "source_class": record.entity.unit_id,
        "name_literal": record.entity.name_literal,
        "race_literal": record.entity.race_literal,
        "adapter_entity_class": record.entity.entity_class,
        "entity_class_basis": record.entity.entity_class_basis,
        "config_path": record.entity.config_path,
        "config_member_path": record.entity.member_path,
        "entity_location": asdict(record.entity.location),
        "entity_raw_attributes": _attributes(record.entity.raw_attributes),
        "base_unit_ids": list(record.entity.base_unit_ids),
        "base_image_literal": record.entity.base_image_literal,
        "profile_literal": record.entity.profile_literal,
        "entity_macro_invocations": list(record.entity.macro_invocations),
        "unresolved_inheritance": record.entity.unresolved_inheritance,
        "entity_quarantine_reasons": list(record.entity.quarantine_reasons),
        "source_tag": record.source_tag,
        "variant_path": list(record.variant_path),
        "adapter_normalized_action": record.normalized_action,
        "normalized_action_basis": record.normalized_action_basis,
        "animation_location": asdict(record.location),
        "animation_raw_attributes": _attributes(record.raw_attributes),
        "apply_to_literal": record.apply_to_literal,
        "attack_name_filters": list(record.attack_name_filters),
        "attack_range_filters": list(record.attack_range_filters),
        "animation_directions": list(record.directions),
        "source_direction_groups": [list(group) for group in record.source_direction_groups],
        "single_direction_hint": record.direction_hint,
        "start_time_literal": record.start_time_literal,
        "cycles_literal": record.cycles_literal,
        "effective_cycles": record.effective_cycles,
        "loop_mode": record.loop_mode,
        "loop_basis": record.loop_basis,
        "animation_macro_invocations": list(record.macro_invocations),
        "duration_ms_per_occurrence": [frame.duration_milliseconds for frame in record.frames],
        "total_duration_ms": record.total_duration_milliseconds,
        "source_image_member_order": [frame.source_member_path for frame in record.frames],
        "source_image_hash_order": [frame.source_blob_sha256 for frame in record.frames],
        "source_image_dimensions": [
            [frame.source_image_width, frame.source_image_height] for frame in record.frames
        ],
        "source_dimensions_consistent": record.source_dimensions_consistent,
        "sequence_canvas_dimensions_are_max_source_dimensions": True,
        "primary_timeline_exact": True,
        "safe_primary_source_sequence": True,
        "primary_track_only": True,
        "auxiliary_tracks_composited": False,
        "runtime_composite_complete": not record.auxiliary_frame_declarations,
        "auxiliary_frame_declarations": [
            asdict(declaration) for declaration in record.auxiliary_frame_declarations
        ],
        "exact_engine_timing": True,
        "state_occurrence_order_preserved": True,
        "standalone_source_images": True,
        "geometry_coordinate_space": "source_sheet",
        "geometry_is_full_source_image": True,
        "clipping_or_repair_applied": False,
        "image_path_functions_applied": False,
        "rights_scope": rights_scope,
    }


def _occurrence_specs(
    plan: WesnothProjectionPlan,
    record: WesnothProjectionRecord,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    common = {
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "projection_version": PROJECTION_VERSION,
        "unit_id": record.entity.unit_id,
        "animation_line_number": record.location.line_number,
    }
    specs: list[tuple[str, str, dict[str, Any]]] = [
        (
            record.entity.member_path,
            "wesnoth_unit_animation_definition",
            {
                **common,
                "config_path": record.entity.config_path,
                "entity_line_number": record.entity.location.line_number,
                "source_tag": record.source_tag,
                "variant_path": list(record.variant_path),
            },
        )
    ]
    image_occurrences: dict[str, list[WesnothProjectionFrame]] = {}
    for frame in record.frames:
        image_occurrences.setdefault(frame.source_member_path, []).append(frame)
    for member_path, frames in sorted(image_occurrences.items()):
        exemplar = frames[0]
        specs.append(
            (
                member_path,
                "wesnoth_source_frame_image",
                {
                    **common,
                    "source_blob_sha256": exemplar.source_blob_sha256,
                    "source_image_dimensions": [
                        exemplar.source_image_width,
                        exemplar.source_image_height,
                    ],
                    "logical_image_path": exemplar.logical_image_path,
                    "sequence_ordinals": [frame.ordinal for frame in frames],
                    "occurrence_count": len(frames),
                    "per_asset_license_expression": None,
                    "per_asset_creator": None,
                },
            )
        )
    for path in plan.engine_evidence_paths:
        specs.append(
            (
                path,
                "wesnoth_engine_animation_semantics",
                {**common, "rights_scope_caveat": RIGHTS_SCOPE_CAVEAT},
            )
        )
    for evidence in plan.rights.evidence:
        specs.append(
            (
                evidence.member_path,
                "wesnoth_collection_rights_evidence",
                {
                    **common,
                    "evidence_sha256": evidence.sha256,
                    "evidence_scope": evidence.evidence_scope,
                    "finding": evidence.finding,
                    "rights_scope_caveat": RIGHTS_SCOPE_CAVEAT,
                },
            )
        )
    return tuple(specs)


def _frame_metadata(
    plan: WesnothProjectionPlan,
    record: WesnothProjectionRecord,
    frame: WesnothProjectionFrame,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "source_member_path": frame.source_member_path,
        "source_blob_sha256": frame.source_blob_sha256,
        "logical_image_path": frame.logical_image_path,
        "source_image_resolution_basis": frame.source_image_resolution_basis,
        "source_image_dimensions": [frame.source_image_width, frame.source_image_height],
        "source_frame_index": 0,
        "sequence_ordinal": frame.ordinal,
        "declaration_ordinal": frame.declaration_ordinal,
        "ordinal_in_expression": frame.ordinal_in_expression,
        "frame_tag": frame.frame_tag,
        "image_attribute": frame.image_attribute,
        "source_expression": frame.source_expression,
        "declaration_location": asdict(frame.declaration_location),
        "raw_attributes": _attributes(frame.raw_attributes),
        "context_attributes": _attributes(frame.context_attributes),
        "branch_path": list(frame.branch_path),
        "source_direction_group": list(frame.directions),
        "start_time_literal": frame.start_time_literal,
        "duration_literal": frame.duration_literal,
        "begin_literal": frame.begin_literal,
        "end_literal": frame.end_literal,
        "duration_ms": frame.duration_milliseconds,
        "layer_literal": frame.layer_literal,
        "offset_literal": frame.offset_literal,
        "x_literal": frame.x_literal,
        "y_literal": frame.y_literal,
        "directional_x_literal": frame.directional_x_literal,
        "directional_y_literal": frame.directional_y_literal,
        "auto_hflip_literal": frame.auto_hflip_literal,
        "effective_auto_hflip": frame.effective_auto_hflip,
        "auto_vflip_literal": frame.auto_vflip_literal,
        "effective_auto_vflip": frame.effective_auto_vflip,
        "primary_literal": frame.primary_literal,
        "inline_modifiers": frame.inline_modifiers,
        "separate_image_mod": frame.separate_image_mod,
        "exact_timing": frame.exact_timing,
        "lossless_source_pixels": frame.lossless_source_pixels,
        "quarantine_reasons": list(frame.quarantine_reasons),
        "frame_rect": {
            "left": 0,
            "top": 0,
            "right": frame.source_image_width,
            "bottom": frame.source_image_height,
            "width": frame.source_image_width,
            "height": frame.source_image_height,
            "coordinate_space": "source_sheet",
        },
        "geometry_is_full_source_image": True,
        "clipping_or_repair_applied": False,
        "image_path_functions_applied": False,
        "auxiliary_tracks_composited": False,
        "runtime_composite_complete": not record.auxiliary_frame_declarations,
        "rights_scope": _rights_scope_metadata(plan),
    }


def project_wesnoth_audit(
    database: IndexDB,
    plan: WesnothProjectionPlan,
    taxonomy: Taxonomy,
) -> WesnothProjectionResult:
    """Idempotently project the precomputed exact primary-track subset."""

    database.initialize()
    item_id, members = _preflight(database, plan)
    manifest_sha256 = plan.projection_manifest_sha256
    rights_scope = _rights_scope_metadata(plan)
    entity_ids: dict[str, str] = {}
    created_sequences = 0
    reused_sequences = 0
    occurrence_links = 0

    for record in plan.records:
        entity = record.entity
        entity_id = entity_ids.get(entity.entity_external_key)
        if entity_id is None:
            entity_class = taxonomy.normalize_entity_class(entity.entity_class)
            entity_id = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=entity.entity_external_key,
                representative_item_id=item_id,
                display_name=entity.unit_id,
                entity_class=entity_class.value,
                entity_subclass=entity.entity_class,
                species_or_type=entity.race_literal,
                taxonomy_version=taxonomy.version,
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "projection_manifest_sha256": manifest_sha256,
                    "source_audit_record_sha256": plan.source_audit_record_sha256,
                    "archive_sha256": plan.archive_sha256,
                    "repository_commit": plan.repository_commit,
                    "unit_id": entity.unit_id,
                    "name_literal": entity.name_literal,
                    "race_literal": entity.race_literal,
                    "adapter_entity_class": entity.entity_class,
                    "normalized_entity_class": entity_class.value,
                    "entity_class_basis": entity.entity_class_basis,
                    "config_path": entity.config_path,
                    "member_path": entity.member_path,
                    "location": asdict(entity.location),
                    "raw_attributes": _attributes(entity.raw_attributes),
                    "base_unit_ids": list(entity.base_unit_ids),
                    "base_image_literal": entity.base_image_literal,
                    "profile_literal": entity.profile_literal,
                    "macro_invocations": list(entity.macro_invocations),
                    "unresolved_inheritance": entity.unresolved_inheritance,
                    "quarantine_reasons": list(entity.quarantine_reasons),
                    "classification_method": entity_class.method,
                    "classification_confidence": entity_class.confidence,
                    "rights_scope": rights_scope,
                },
            )
            entity_ids[entity.entity_external_key] = entity_id

        motion = taxonomy.motion_condition(
            action=record.normalized_action,
            direction=record.direction_hint,
            view=None,
        )
        sequence_id = database.find_sequence_by_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
        )
        sequence_arguments = {
            "source_blob_sha256": plan.archive_sha256,
            "extraction_method": PROJECTION_VERSION,
            "extraction_confidence": 1.0,
            "width": record.sequence_width,
            "height": record.sequence_height,
            "frame_count": record.frame_count,
            "loop_mode": record.loop_mode,
            "action": motion.normalized_action,
            "direction": motion.direction,
            "quality_tier": "F0_lossless_primary_png_exact_wml_timing",
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
                "unit_id": entity.unit_id,
                "config_path": entity.config_path,
                "entity_line_number": entity.location.line_number,
                "variant_path": list(record.variant_path),
                "primary_track_only": True,
                "auxiliary_tracks_composited": False,
                "rights_scope": rights_scope,
            },
        )
        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version=taxonomy.version,
            source_action=record.source_tag,
            normalized_action=motion.normalized_action,
            action_family=motion.action_family,
            annotation_method=PROJECTION_VERSION,
            view=motion.view,
            direction=motion.direction,
            loopable=record.effective_cycles,
            cycle_frames=record.frame_count if record.effective_cycles else None,
            phase_zero_frame=0,
            confidence=(
                motion.confidence
                if record.normalized_action is not None and motion.normalized_action != "unknown"
                else 0.0
            ),
            conditioning={
                "source_tag": record.source_tag,
                "apply_to_literal": record.apply_to_literal,
                "adapter_normalized_action": record.normalized_action,
                "normalized_action_basis": record.normalized_action_basis,
                "taxonomy_normalization_method": motion.method,
                "attack_name_filters": list(record.attack_name_filters),
                "attack_range_filters": list(record.attack_range_filters),
                "source_direction_groups": [
                    list(group) for group in record.source_direction_groups
                ],
                "single_direction_hint": record.direction_hint,
                "runtime_auto_hflip": [frame.effective_auto_hflip for frame in record.frames],
                "runtime_auto_vflip": [frame.effective_auto_vflip for frame in record.frames],
                "loop_basis": record.loop_basis,
                "timing_known": True,
                "exact_engine_timing": True,
                "duration_ms_per_occurrence": [
                    frame.duration_milliseconds for frame in record.frames
                ],
                "state_occurrence_order_preserved": True,
                "primary_track_only": True,
                "auxiliary_tracks_composited": False,
                "rights_scope": rights_scope,
            },
        )

        for member_path, role, metadata in _occurrence_specs(plan, record):
            database.link_sequence_occurrence(
                sequence_id=sequence_id,
                archive_blob_sha256=plan.archive_sha256,
                archive_member_ordinal=int(members[member_path]["ordinal"]),
                occurrence_role=role,
                metadata=metadata,
            )
            occurrence_links += 1

        for frame in record.frames:
            frame_direction = None
            if len(frame.directions) == 1:
                frame_direction = frame.directions[0]
            normalized_frame_direction = taxonomy.normalize_direction(frame_direction).value
            database.add_sequence_frame(
                sequence_id=sequence_id,
                ordinal=frame.ordinal,
                source_blob_sha256=frame.source_blob_sha256,
                source_frame_index=0,
                duration_ms=float(frame.duration_milliseconds),
                phase=_frame_phase(frame.ordinal, record.frame_count, record.loop_mode),
                direction=normalized_frame_direction,
                view="unknown",
                metadata=_frame_metadata(plan, record, frame),
            )

    return WesnothProjectionResult(
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=manifest_sha256,
        projected_entities=plan.projected_entity_count,
        projected_sequences=plan.projected_sequence_count,
        projected_frame_occurrences=plan.projected_frame_occurrence_count,
        projected_loops=plan.projected_loop_count,
        projected_one_shots=plan.projected_one_shot_count,
        projected_normalized_actions=plan.projected_normalized_action_count,
        projected_unknown_actions=plan.projected_unknown_action_count,
        created_sequences=created_sequences,
        reused_sequences=reused_sequences,
        occurrence_links=occurrence_links,
        excluded_candidate_sequences=plan.excluded_candidate_sequence_count,
        excluded_candidate_frame_occurrences=plan.excluded_candidate_frame_occurrence_count,
        excluded_transformed_primary_frames=plan.excluded_transformed_primary_frame_count,
    )


def ingest_known_wesnoth_sequences(
    database: IndexDB,
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> WesnothProjectionResult:
    """Audit and project only the exact pinned Wesnoth snapshot."""

    plan = plan_known_wesnoth_projection(archive_path)
    if (
        plan.archive_sha256 != EXPECTED_WESNOTH_ARCHIVE_SHA256
        or plan.repository_commit != WESNOTH_COMMIT
    ):
        raise ValueError("Refusing Wesnoth projection for an unexpected archive or commit")
    return project_wesnoth_audit(database, plan, taxonomy)


__all__ = [
    "EXPECTED_PINNED_ENTITY_COUNT",
    "EXPECTED_PINNED_FRAME_OCCURRENCE_COUNT",
    "EXPECTED_PINNED_SEQUENCE_COUNT",
    "PROJECTION_VERSION",
    "RIGHTS_SCOPE_CAVEAT",
    "SOURCE_ID",
    "WesnothProjectionEntity",
    "WesnothProjectionExclusion",
    "WesnothProjectionFrame",
    "WesnothProjectionPlan",
    "WesnothProjectionReadiness",
    "WesnothProjectionRecord",
    "WesnothProjectionResult",
    "check_wesnoth_projection_readiness",
    "ingest_known_wesnoth_sequences",
    "plan_known_wesnoth_projection",
    "plan_wesnoth_projection",
    "project_wesnoth_audit",
]
