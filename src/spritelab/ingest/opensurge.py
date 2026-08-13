"""Project audited Open Surge ``.spr`` timelines into the provenance database.

The adapter is the source of truth for parsing.  This module applies a stricter
materialization gate: every emitted frame occurrence must have exact, absolute
source-sheet geometry, the source PNG must exist in the audited archive, and the
PNG must have an exact row in Open Surge's per-asset copyright manifest.  Source
declarations are never clipped or repaired here.

Planning and readiness checks are write-free.  Database projection is idempotent
for the immutable, pinned archive: stable source keys select existing sequences,
and all core projection tables are updated through their upsert APIs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from spritelab.adapters.opensurge import (
    EXPECTED_OPEN_SURGE_ARCHIVE_SHA256,
    OPEN_SURGE_COMMIT,
    AnimationDefinition,
    AssetCredit,
    EvidenceDocument,
    FrameOccurrence,
    OpenSurgeAudit,
    Point,
    RawProperty,
    Rectangle,
    SourceSheetAudit,
    SpriteDefinition,
    audit_known_open_surge_archive,
)
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

SOURCE_ID = "open_surge"
PROJECTION_VERSION = "opensurge_spr_projection_v2"
RIGHTS_SCOPE = "asset_level_exact_copyright_manifest_row"
PIXEL_TRANSFORM_SCHEMA = "spritelab.pixel_transform.v1"
PIXEL_TRANSFORM_OP = "exact_uint8_rgb_to_rgba_zero"

_COLOR_ENGINE_PATH = "src/core/color.c"
_SHADER_ENGINE_PATH = "src/core/shader.c"

DeclarationKind = Literal["animation", "transition"]


@dataclass(frozen=True)
class OpenSurgePixelTransformEvidence:
    """One audited engine member supporting the projected pixel operation."""

    member_path: str
    sha256: str
    line_numbers: tuple[int, ...]
    scope: str
    claim: str


@dataclass(frozen=True)
class OpenSurgePixelTransform:
    """Strict, serializable color-key operation established by pinned engine code."""

    schema: str
    op: str
    rgb: tuple[int, int, int]
    evidence: tuple[OpenSurgePixelTransformEvidence, ...]

    @property
    def transform_sha256(self) -> str:
        encoded = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["transform_sha256"] = self.transform_sha256
        return payload


@dataclass(frozen=True)
class OpenSurgeProjectionRecord:
    """One timeline whose exact source occurrences are safe to materialize."""

    sequence_source_key: str
    entity_external_key: str
    sprite_identity: str
    entity_class: str
    entity_class_candidates: tuple[str, ...]
    subject_role: str
    morphology_tags: tuple[str, ...]
    parent_subject: str | None
    classification_basis: str
    source_file: str
    source_sheet_member_path: str
    source_sheet_sha256: str
    source_sheet_width: int
    source_sheet_height: int
    source_sheet_mode: str
    source_sheet_format: str | None
    source_sheet_has_transparency: bool
    source_rect: Rectangle
    source_rect_within_image: bool
    frame_size: Point
    sprite_hot_spot: Point
    sprite_action_spot: Point
    asset_credit: AssetCredit
    source_header_authors: tuple[str, ...]
    source_header_licenses: tuple[str, ...]
    source_comments: tuple[str, ...]
    sprite_unknown_properties: tuple[RawProperty, ...]
    sprite_evidence_member_path: str
    relative_script_path: str
    sprite_line_number: int
    declaration_kind: DeclarationKind
    animation_id: int | None
    transition_from: int | Literal["any"] | None
    transition_to: int | Literal["any"] | None
    transition_ordinal: int | None
    timeline_line_number: int
    source_label: str | None
    source_label_basis: str
    normalized_action: str | None
    normalized_action_basis: str
    direction_hint: str | None
    source_variant_hint: str | None
    repeat: bool
    repeat_was_explicit: bool
    effective_repeat: bool
    repeat_from: int
    repeat_from_was_explicit: bool
    effective_repeat_from: int
    fps: float
    fps_source_token: str
    fps_was_explicit: bool
    data: tuple[int, ...]
    intro_data: tuple[int, ...]
    loop_data: tuple[int, ...]
    timeline_hot_spot: Point
    hot_spot_overridden: bool
    timeline_action_spot: Point
    action_spot_overridden: bool
    programmatic_animation_name: str | None
    frame_occurrences: tuple[FrameOccurrence, ...]
    comments: tuple[str, ...]
    timeline_unknown_properties: tuple[RawProperty, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frame_occurrences)

    @property
    def duration_ms(self) -> float:
        return 1000.0 / self.fps

    @property
    def total_duration_ms(self) -> float:
        return self.frame_count * self.duration_ms

    @property
    def loop_mode(self) -> str:
        if not self.effective_repeat:
            return "one_shot"
        if self.effective_repeat_from:
            return "intro_then_loop"
        return "loop"

    @property
    def direction(self) -> str:
        return self.direction_hint or "unknown"

    @property
    def view(self) -> str:
        return "unknown"


@dataclass(frozen=True)
class OpenSurgeProjectionExclusion:
    """A source timeline omitted without changing its audited declarations."""

    sequence_source_key: str
    sprite_identity: str
    source_file: str
    sprite_evidence_member_path: str
    declaration_kind: DeclarationKind
    animation_id: int | None
    transition_from: int | Literal["any"] | None
    transition_to: int | Literal["any"] | None
    transition_ordinal: int | None
    timeline_line_number: int
    frame_count: int
    reasons: tuple[str, ...]
    unsafe_occurrences: tuple[FrameOccurrence, ...]


@dataclass(frozen=True)
class OpenSurgeProjectionPlan:
    """Pure deterministic projection plan; constructing it performs no writes."""

    archive_sha256: str
    repository_commit: str | None
    pixel_transform: OpenSurgePixelTransform
    records: tuple[OpenSurgeProjectionRecord, ...]
    exclusions: tuple[OpenSurgeProjectionExclusion, ...]

    @property
    def projected_entity_count(self) -> int:
        return len({record.entity_external_key for record in self.records})

    @property
    def projected_sequence_count(self) -> int:
        return len(self.records)

    @property
    def projected_regular_animation_count(self) -> int:
        return sum(record.declaration_kind == "animation" for record in self.records)

    @property
    def projected_transition_count(self) -> int:
        return sum(record.declaration_kind == "transition" for record in self.records)

    @property
    def projected_frame_occurrence_count(self) -> int:
        return sum(record.frame_count for record in self.records)

    @property
    def projected_normalized_action_count(self) -> int:
        return sum(record.normalized_action is not None for record in self.records)

    @property
    def projected_unknown_action_count(self) -> int:
        return sum(record.normalized_action is None for record in self.records)

    @property
    def projected_explicit_direction_count(self) -> int:
        return sum(record.direction_hint is not None for record in self.records)

    @property
    def projected_loop_count(self) -> int:
        return sum(record.loop_mode == "loop" for record in self.records)

    @property
    def projected_intro_then_loop_count(self) -> int:
        return sum(record.loop_mode == "intro_then_loop" for record in self.records)

    @property
    def projected_one_shot_count(self) -> int:
        return sum(record.loop_mode == "one_shot" for record in self.records)

    @property
    def projected_oversized_source_rect_count(self) -> int:
        """Safe timelines whose unused declared rectangle extends past the PNG."""

        return sum(not record.source_rect_within_image for record in self.records)

    @property
    def excluded_candidate_sequence_count(self) -> int:
        return len(self.exclusions)

    @property
    def excluded_candidate_frame_occurrence_count(self) -> int:
        return sum(exclusion.frame_count for exclusion in self.exclusions)

    @property
    def excluded_unsafe_occurrence_count(self) -> int:
        return sum(len(exclusion.unsafe_occurrences) for exclusion in self.exclusions)

    @property
    def required_member_paths(self) -> tuple[str, ...]:
        paths: set[str] = set()
        for record in self.records:
            paths.add(record.source_sheet_member_path)
            paths.add(record.sprite_evidence_member_path)
            paths.add(record.asset_credit.evidence_member_path)
        paths.update(evidence.member_path for evidence in self.pixel_transform.evidence)
        return tuple(sorted(paths))

    @property
    def projection_manifest_sha256(self) -> str:
        payload = {
            "projection_version": PROJECTION_VERSION,
            "archive_sha256": self.archive_sha256,
            "repository_commit": self.repository_commit,
            "pixel_transform": asdict(self.pixel_transform),
            "records": [asdict(record) for record in self.records],
            "exclusions": [asdict(exclusion) for exclusion in self.exclusions],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OpenSurgeProjectionReadiness:
    """Read-only report of whether all DB evidence prerequisites are present."""

    database_path: str
    archive_sha256: str
    projection_manifest_sha256: str
    archive_blob_present: bool
    source_item_count: int
    required_member_count: int
    present_member_count: int
    missing_member_paths: tuple[str, ...]
    missing_sheet_blobs: tuple[str, ...]
    sheet_hash_mismatches: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.archive_blob_present
            and self.source_item_count > 0
            and not self.missing_member_paths
            and not self.missing_sheet_blobs
            and not self.sheet_hash_mismatches
        )


@dataclass(frozen=True)
class OpenSurgeProjectionResult:
    """Core projection effects for one idempotent run."""

    archive_sha256: str
    projection_manifest_sha256: str
    projected_entities: int
    projected_sequences: int
    projected_regular_animations: int
    projected_transitions: int
    projected_frame_occurrences: int
    projected_normalized_actions: int
    projected_unknown_actions: int
    created_sequences: int
    reused_sequences: int
    occurrence_links: int
    excluded_candidate_sequences: int
    excluded_candidate_frame_occurrences: int
    excluded_unsafe_occurrences: int
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


def _entity_external_key(audit: OpenSurgeAudit, sprite: SpriteDefinition) -> str:
    return _stable_json_key(
        "opensurge-entity-v1",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.repository_commit,
            "relative_script_path": sprite.relative_script_path,
            "source_file": sprite.source_file,
            "sprite_identity": sprite.identity,
        },
    )


def _sequence_source_key(
    audit: OpenSurgeAudit,
    sprite: SpriteDefinition,
    timeline: AnimationDefinition,
) -> str:
    return _stable_json_key(
        "opensurge-sequence-v1",
        {
            "animation_id": timeline.animation_id,
            "archive_sha256": audit.archive_sha256,
            "declaration_kind": timeline.declaration_kind,
            "relative_script_path": sprite.relative_script_path,
            "repository_commit": audit.repository_commit,
            "sprite_identity": sprite.identity,
            "transition_from": timeline.transition_from,
            "transition_ordinal": timeline.transition_ordinal,
            "transition_to": timeline.transition_to,
        },
    )


def _timeline_exclusion_reasons(
    sprite: SpriteDefinition,
    timeline: AnimationDefinition,
    sheet: SourceSheetAudit | None,
) -> tuple[tuple[str, ...], tuple[FrameOccurrence, ...]]:
    reasons: list[str] = []
    unsafe_occurrences: list[FrameOccurrence] = []

    if sprite.source_file_exists is not True or sheet is None:
        reasons.append("source_sheet_is_missing")
    if sprite.asset_credit is None or sheet is None or sheet.asset_credit is None:
        reasons.append("source_sheet_has_no_exact_asset_credit")
    elif sprite.asset_credit != sheet.asset_credit:
        reasons.append("sprite_and_sheet_asset_credit_rows_do_not_match")
    if not sprite.source_rect_grid_compatible:
        reasons.append("source_rect_is_not_an_integral_frame_grid")
    if not sprite.referenced_frames_within_declared_grid:
        reasons.append("referenced_frame_is_outside_declared_source_grid")
    if timeline.fps <= 0:
        reasons.append("fps_is_not_positive")
    occurrence_order = tuple(item.source_frame_index for item in timeline.frame_occurrences)
    if (
        not timeline.data
        or len(timeline.data) != len(timeline.frame_occurrences)
        or occurrence_order != timeline.data
    ):
        reasons.append("frame_order_and_occurrence_geometry_do_not_match")
    if timeline.effective_repeat and not timeline.loop_data:
        reasons.append("repeating_timeline_has_no_loop_tail")

    for occurrence in timeline.frame_occurrences:
        unsafe = (
            not occurrence.within_declared_source_rect
            or occurrence.within_source_image is not True
            or occurrence.right - occurrence.left != sprite.frame_size.x
            or occurrence.bottom - occurrence.top != sprite.frame_size.y
        )
        if sheet is not None:
            unsafe = unsafe or (
                occurrence.left < 0
                or occurrence.top < 0
                or occurrence.right > sheet.width
                or occurrence.bottom > sheet.height
            )
        if unsafe:
            unsafe_occurrences.append(occurrence)

    if any(not item.within_declared_source_rect for item in unsafe_occurrences):
        reasons.append("frame_occurrence_is_outside_declared_source_rect")
    if any(item.within_source_image is not True for item in unsafe_occurrences):
        reasons.append("frame_occurrence_is_outside_source_image")
    if unsafe_occurrences and not any(reason.startswith("frame_occurrence_") for reason in reasons):
        reasons.append("frame_occurrence_geometry_is_inconsistent")

    return tuple(dict.fromkeys(reasons)), tuple(unsafe_occurrences)


def _projection_record(
    audit: OpenSurgeAudit,
    sprite: SpriteDefinition,
    timeline: AnimationDefinition,
    sheet: SourceSheetAudit,
) -> OpenSurgeProjectionRecord:
    credit = sprite.asset_credit
    if credit is None:  # guarded by the planner; keeps the type invariant explicit
        raise ValueError(f"Missing asset credit for admitted sprite {sprite.identity!r}")
    return OpenSurgeProjectionRecord(
        sequence_source_key=_sequence_source_key(audit, sprite, timeline),
        entity_external_key=_entity_external_key(audit, sprite),
        sprite_identity=sprite.identity,
        entity_class=sprite.entity.primary_entity_class,
        entity_class_candidates=sprite.entity.entity_class_candidates,
        subject_role=sprite.entity.subject_role,
        morphology_tags=sprite.entity.morphology_tags,
        parent_subject=sprite.entity.parent_subject,
        classification_basis=sprite.entity.classification_basis,
        source_file=sprite.source_file,
        source_sheet_member_path=sheet.member_path,
        source_sheet_sha256=sheet.sha256,
        source_sheet_width=sheet.width,
        source_sheet_height=sheet.height,
        source_sheet_mode=sheet.image_mode,
        source_sheet_format=sheet.image_format,
        source_sheet_has_transparency=sheet.has_transparency,
        source_rect=sprite.source_rect,
        source_rect_within_image=bool(sprite.source_rect_within_image),
        frame_size=sprite.frame_size,
        sprite_hot_spot=sprite.hot_spot,
        sprite_action_spot=sprite.action_spot,
        asset_credit=credit,
        source_header_authors=sprite.source_header_authors,
        source_header_licenses=sprite.source_header_licenses,
        source_comments=sprite.source_comments,
        sprite_unknown_properties=sprite.unknown_properties,
        sprite_evidence_member_path=sprite.evidence_member_path,
        relative_script_path=sprite.relative_script_path,
        sprite_line_number=sprite.line_number,
        declaration_kind=timeline.declaration_kind,
        animation_id=timeline.animation_id,
        transition_from=timeline.transition_from,
        transition_to=timeline.transition_to,
        transition_ordinal=timeline.transition_ordinal,
        timeline_line_number=timeline.line_number,
        source_label=timeline.source_label,
        source_label_basis=timeline.source_label_basis,
        normalized_action=timeline.normalized_action,
        normalized_action_basis=timeline.normalized_action_basis,
        direction_hint=timeline.direction_hint,
        source_variant_hint=timeline.source_variant_hint,
        repeat=timeline.repeat,
        repeat_was_explicit=timeline.repeat_was_explicit,
        effective_repeat=timeline.effective_repeat,
        repeat_from=timeline.repeat_from,
        repeat_from_was_explicit=timeline.repeat_from_was_explicit,
        effective_repeat_from=timeline.effective_repeat_from,
        fps=timeline.fps,
        fps_source_token=timeline.fps_source_token,
        fps_was_explicit=timeline.fps_was_explicit,
        data=timeline.data,
        intro_data=timeline.intro_data,
        loop_data=timeline.loop_data,
        timeline_hot_spot=timeline.hot_spot,
        hot_spot_overridden=timeline.hot_spot_overridden,
        timeline_action_spot=timeline.action_spot,
        action_spot_overridden=timeline.action_spot_overridden,
        programmatic_animation_name=timeline.programmatic_animation_name,
        frame_occurrences=timeline.frame_occurrences,
        comments=timeline.comments,
        timeline_unknown_properties=timeline.unknown_properties,
    )


def _pixel_transform_evidence(
    document: EvidenceDocument,
    *,
    claim: str,
) -> OpenSurgePixelTransformEvidence:
    if not document.relevant_line_numbers:
        raise ValueError(
            f"Open Surge evidence {document.relative_path!r} has no audited relevant lines"
        )
    return OpenSurgePixelTransformEvidence(
        member_path=document.member_path,
        sha256=document.sha256,
        line_numbers=document.relevant_line_numbers,
        scope=document.scope,
        claim=claim,
    )


def _open_surge_pixel_transform(audit: OpenSurgeAudit) -> OpenSurgePixelTransform:
    documents = {document.relative_path: document for document in audit.evidence_documents}
    missing = tuple(
        path for path in (_COLOR_ENGINE_PATH, _SHADER_ENGINE_PATH) if path not in documents
    )
    if missing:
        raise ValueError(f"Open Surge audit is missing pixel-semantics evidence: {missing!r}")
    return OpenSurgePixelTransform(
        schema=PIXEL_TRANSFORM_SCHEMA,
        op=PIXEL_TRANSFORM_OP,
        rgb=(255, 0, 255),
        evidence=(
            _pixel_transform_evidence(
                documents[_COLOR_ENGINE_PATH],
                claim="alpha_zero_or_exact_uint8_rgb_is_transparent",
            ),
            _pixel_transform_evidence(
                documents[_SHADER_ENGINE_PATH],
                claim="exact_rgb_match_zeroes_sampled_rgba",
            ),
        ),
    )


def plan_open_surge_projection(audit: OpenSurgeAudit) -> OpenSurgeProjectionPlan:
    """Build a deterministic, write-free plan from an Open Surge audit.

    A source rectangle that extends past the image does not by itself disqualify
    a timeline: Open Surge includes several such declarations whose referenced
    cells are fully in-image.  Those records retain the declaration caveat.  A
    non-integral grid or any unsafe referenced occurrence is excluded wholesale.
    """

    sheets = {sheet.relative_path: sheet for sheet in audit.source_sheets}
    pixel_transform = _open_surge_pixel_transform(audit)
    records: list[OpenSurgeProjectionRecord] = []
    exclusions: list[OpenSurgeProjectionExclusion] = []
    for sprite in audit.sprites:
        sheet = sheets.get(sprite.source_file)
        timelines = (*sprite.animations, *sprite.transitions)
        for timeline in timelines:
            sequence_key = _sequence_source_key(audit, sprite, timeline)
            reasons, unsafe_occurrences = _timeline_exclusion_reasons(sprite, timeline, sheet)
            if reasons:
                exclusions.append(
                    OpenSurgeProjectionExclusion(
                        sequence_source_key=sequence_key,
                        sprite_identity=sprite.identity,
                        source_file=sprite.source_file,
                        sprite_evidence_member_path=sprite.evidence_member_path,
                        declaration_kind=timeline.declaration_kind,
                        animation_id=timeline.animation_id,
                        transition_from=timeline.transition_from,
                        transition_to=timeline.transition_to,
                        transition_ordinal=timeline.transition_ordinal,
                        timeline_line_number=timeline.line_number,
                        frame_count=timeline.frame_count,
                        reasons=reasons,
                        unsafe_occurrences=unsafe_occurrences,
                    )
                )
                continue
            if sheet is None:  # narrowed by the exclusion gate
                raise AssertionError("admitted Open Surge timeline has no source sheet")
            records.append(_projection_record(audit, sprite, timeline, sheet))

    records.sort(key=lambda record: record.sequence_source_key)
    exclusions.sort(key=lambda exclusion: exclusion.sequence_source_key)
    return OpenSurgeProjectionPlan(
        archive_sha256=audit.archive_sha256,
        repository_commit=audit.repository_commit,
        pixel_transform=pixel_transform,
        records=tuple(records),
        exclusions=tuple(exclusions),
    )


def plan_known_open_surge_projection(
    archive_path: str | Path,
) -> OpenSurgeProjectionPlan:
    """Audit the exact pinned CAS archive and build a write-free plan."""

    return plan_open_surge_projection(audit_known_open_surge_archive(archive_path))


def _readonly_connection(database_path: str | Path) -> sqlite3.Connection:
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def check_open_surge_projection_readiness(
    database_path: str | Path,
    plan: OpenSurgeProjectionPlan,
) -> OpenSurgeProjectionReadiness:
    """Inspect exact prerequisites through a query-only SQLite connection."""

    required_paths = plan.required_member_paths
    expected_sheet_hashes = {
        record.source_sheet_member_path: record.source_sheet_sha256 for record in plan.records
    }
    with _readonly_connection(database_path) as connection:
        archive_blob_present = (
            connection.execute(
                """
                SELECT 1 FROM archive_inventories
                WHERE archive_blob_sha256 = ? LIMIT 1
                """,
                (plan.archive_sha256,),
            ).fetchone()
            is not None
        )
        source_item_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM items i
                JOIN item_blobs ib ON ib.item_id = i.id
                WHERE i.source_id = ? AND ib.blob_sha256 = ?
                """,
                (SOURCE_ID, plan.archive_sha256),
            ).fetchone()[0]
        )
        rows = connection.execute(
            """
            SELECT am.normalized_path, am.member_path, am.extracted_blob_sha256,
                   b.sha256 AS registered_blob_sha256
            FROM archive_members am
            LEFT JOIN blobs b ON b.sha256 = am.extracted_blob_sha256
            WHERE am.archive_blob_sha256 = ?
            """,
            (plan.archive_sha256,),
        ).fetchall()

    members: dict[str, sqlite3.Row] = {}
    for row in rows:
        members[str(row["normalized_path"])] = row
        members[str(row["member_path"])] = row
    missing_paths = tuple(path for path in required_paths if path not in members)
    missing_sheet_blobs: list[str] = []
    hash_mismatches: list[str] = []
    for member_path, expected_hash in sorted(expected_sheet_hashes.items()):
        row = members.get(member_path)
        if row is None:
            continue
        actual_hash = row["extracted_blob_sha256"]
        registered_hash = row["registered_blob_sha256"]
        if actual_hash is None or registered_hash is None:
            missing_sheet_blobs.append(member_path)
        elif str(actual_hash) != expected_hash:
            hash_mismatches.append(
                f"{member_path}: expected {expected_hash}, indexed {actual_hash}"
            )
    return OpenSurgeProjectionReadiness(
        database_path=str(Path(database_path).resolve()),
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=plan.projection_manifest_sha256,
        archive_blob_present=archive_blob_present,
        source_item_count=source_item_count,
        required_member_count=len(required_paths),
        present_member_count=len(required_paths) - len(missing_paths),
        missing_member_paths=missing_paths,
        missing_sheet_blobs=tuple(missing_sheet_blobs),
        sheet_hash_mismatches=tuple(hash_mismatches),
    )


def _archive_members(database: IndexDB, archive_sha256: str) -> dict[str, sqlite3.Row]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT ordinal, member_path, normalized_path, extracted_blob_sha256
            FROM archive_members
            WHERE archive_blob_sha256 = ?
            ORDER BY ordinal
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
            SELECT i.id
            FROM items i
            JOIN item_blobs ib ON ib.item_id = i.id
            WHERE i.source_id = ? AND ib.blob_sha256 = ?
            ORDER BY i.id
            LIMIT 1
            """,
            (SOURCE_ID, archive_sha256),
        ).fetchone()
    if row is None:
        raise ValueError(
            "The Open Surge archive has no indexed source item for "
            f"source_id={SOURCE_ID!r}: {archive_sha256}"
        )
    return str(row[0])


def _preflight(
    database: IndexDB,
    plan: OpenSurgeProjectionPlan,
) -> tuple[str, dict[str, sqlite3.Row]]:
    with database.connect() as connection:
        inventory = connection.execute(
            "SELECT 1 FROM archive_inventories WHERE archive_blob_sha256=?",
            (plan.archive_sha256,),
        ).fetchone()
    if inventory is None:
        raise ValueError(f"Open Surge archive inventory is missing: {plan.archive_sha256}")
    item_id = _item_id(database, plan.archive_sha256)
    members = _archive_members(database, plan.archive_sha256)
    missing = [path for path in plan.required_member_paths if path not in members]
    if missing:
        raise ValueError(
            "Projection evidence members are missing from archive_members: "
            + ", ".join(missing[:10])
        )
    for record in plan.records:
        indexed_hash = members[record.source_sheet_member_path]["extracted_blob_sha256"]
        if indexed_hash is None:
            raise ValueError(
                "Source sprite sheet has not been extracted into CAS: "
                f"{record.source_sheet_member_path}"
            )
        if str(indexed_hash) != record.source_sheet_sha256:
            raise ValueError(
                "Source sprite sheet CAS hash does not match audited ZIP bytes for "
                f"{record.source_sheet_member_path}: expected {record.source_sheet_sha256}, "
                f"indexed {indexed_hash}"
            )
    return item_id, members


def _credit_metadata(credit: AssetCredit) -> dict[str, Any]:
    return {
        "scope": RIGHTS_SCOPE,
        "asset_type": credit.asset_type,
        "file_path": credit.file_path,
        "license_expression": credit.license_expression,
        "author": credit.author,
        "website": credit.website,
        "notes": credit.notes,
        "evidence_member_path": credit.evidence_member_path,
        "evidence_line_number": credit.line_number,
    }


def _sequence_metadata(
    plan: OpenSurgeProjectionPlan,
    record: OpenSurgeProjectionRecord,
    projection_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": projection_manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "sprite_identity": record.sprite_identity,
        "relative_script_path": record.relative_script_path,
        "sprite_evidence_member_path": record.sprite_evidence_member_path,
        "sprite_line_number": record.sprite_line_number,
        "source_file": record.source_file,
        "source_sheet_member_path": record.source_sheet_member_path,
        "source_sheet_sha256": record.source_sheet_sha256,
        "source_sheet_dimensions": [
            record.source_sheet_width,
            record.source_sheet_height,
        ],
        "source_sheet_mode": record.source_sheet_mode,
        "source_sheet_format": record.source_sheet_format,
        "source_sheet_has_transparency": record.source_sheet_has_transparency,
        "pixel_transforms": [plan.pixel_transform.as_metadata()],
        "source_rect": asdict(record.source_rect),
        "source_rect_within_image": record.source_rect_within_image,
        "oversized_source_rect_caveat": not record.source_rect_within_image,
        "frame_size": asdict(record.frame_size),
        "sprite_hot_spot": asdict(record.sprite_hot_spot),
        "sprite_action_spot": asdict(record.sprite_action_spot),
        "declaration_kind": record.declaration_kind,
        "animation_id": record.animation_id,
        "transition_from": record.transition_from,
        "transition_to": record.transition_to,
        "transition_ordinal": record.transition_ordinal,
        "timeline_line_number": record.timeline_line_number,
        "source_label": record.source_label,
        "source_label_basis": record.source_label_basis,
        "adapter_normalized_action": record.normalized_action,
        "normalized_action_basis": record.normalized_action_basis,
        "direction_hint": record.direction_hint,
        "direction_evidence": (
            "explicit_source_comment" if record.direction_hint is not None else "absent"
        ),
        "view_evidence": "absent",
        "source_variant_hint": record.source_variant_hint,
        "repeat": record.repeat,
        "repeat_was_explicit": record.repeat_was_explicit,
        "effective_repeat": record.effective_repeat,
        "repeat_from": record.repeat_from,
        "repeat_from_was_explicit": record.repeat_from_was_explicit,
        "effective_repeat_from": record.effective_repeat_from,
        "intro_frame_count": len(record.intro_data),
        "loop_tail_frame_count": len(record.loop_data),
        "fps": record.fps,
        "fps_source_token": record.fps_source_token,
        "fps_was_explicit": record.fps_was_explicit,
        "duration_ms_per_occurrence": record.duration_ms,
        "total_duration_ms": record.total_duration_ms,
        "data_frame_index_order": list(record.data),
        "intro_data": list(record.intro_data),
        "loop_data": list(record.loop_data),
        "timeline_hot_spot": asdict(record.timeline_hot_spot),
        "hot_spot_overridden": record.hot_spot_overridden,
        "timeline_action_spot": asdict(record.timeline_action_spot),
        "action_spot_overridden": record.action_spot_overridden,
        "programmatic_animation_name": record.programmatic_animation_name,
        "comments": list(record.comments),
        "source_header_authors": list(record.source_header_authors),
        "source_header_licenses": list(record.source_header_licenses),
        "source_comments": list(record.source_comments),
        "sprite_unknown_properties": [
            asdict(property_) for property_ in record.sprite_unknown_properties
        ],
        "timeline_unknown_properties": [
            asdict(property_) for property_ in record.timeline_unknown_properties
        ],
        "state_occurrence_order_preserved": True,
        "repeated_occurrences_preserved": True,
        "exact_engine_timing": True,
        "geometry_coordinate_space": "source_sheet",
        "geometry_absolute": True,
        "clipping_or_repair_applied": False,
        "asset_credit": _credit_metadata(record.asset_credit),
    }


def _occurrence_specs(
    plan: OpenSurgeProjectionPlan,
    record: OpenSurgeProjectionRecord,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    common = {
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "projection_version": PROJECTION_VERSION,
        "sprite_identity": record.sprite_identity,
    }
    occurrences: list[tuple[str, str, dict[str, Any]]] = [
        (
            record.source_sheet_member_path,
            "opensurge_source_sprite_sheet",
            {
                **common,
                "source_file": record.source_file,
                "source_sheet_sha256": record.source_sheet_sha256,
                "sheet_dimensions": [
                    record.source_sheet_width,
                    record.source_sheet_height,
                ],
                "asset_credit": _credit_metadata(record.asset_credit),
            },
        ),
        (
            record.sprite_evidence_member_path,
            "opensurge_sprite_definition",
            {
                **common,
                "sprite_line_number": record.sprite_line_number,
                "timeline_line_number": record.timeline_line_number,
                "declaration_kind": record.declaration_kind,
                "animation_id": record.animation_id,
                "transition_from": record.transition_from,
                "transition_to": record.transition_to,
                "transition_ordinal": record.transition_ordinal,
                "source_header_authors": list(record.source_header_authors),
                "source_header_licenses": list(record.source_header_licenses),
            },
        ),
        (
            record.asset_credit.evidence_member_path,
            "opensurge_asset_credit_manifest",
            {
                **common,
                "asset_credit": _credit_metadata(record.asset_credit),
            },
        ),
    ]
    for evidence in plan.pixel_transform.evidence:
        role = (
            "opensurge_engine_color_key_predicate"
            if evidence.member_path.endswith(f"/{_COLOR_ENGINE_PATH}")
            else "opensurge_engine_color_key_shader"
        )
        occurrences.append(
            (
                evidence.member_path,
                role,
                {
                    **common,
                    "pixel_transform_sha256": plan.pixel_transform.transform_sha256,
                    "pixel_transform_schema": plan.pixel_transform.schema,
                    "pixel_transform_op": plan.pixel_transform.op,
                    "pixel_transform_rgb": list(plan.pixel_transform.rgb),
                    "evidence_sha256": evidence.sha256,
                    "evidence_line_numbers": list(evidence.line_numbers),
                    "evidence_scope": evidence.scope,
                    "evidence_claim": evidence.claim,
                },
            )
        )
    return tuple(occurrences)


def _frame_phase(record: OpenSurgeProjectionRecord, ordinal: int) -> float | None:
    if record.frame_count <= 1:
        return 0.0
    if not record.effective_repeat:
        return ordinal / (record.frame_count - 1)
    if ordinal < record.effective_repeat_from:
        return None
    loop_frame_count = record.frame_count - record.effective_repeat_from
    if loop_frame_count <= 1:
        return 0.0
    return (ordinal - record.effective_repeat_from) / loop_frame_count


def project_open_surge_audit(
    database: IndexDB,
    plan: OpenSurgeProjectionPlan,
    taxonomy: Taxonomy,
) -> OpenSurgeProjectionResult:
    """Idempotently project a precomputed safe plan into core DB tables.

    The source item, archive inventory/members, and source-sheet CAS blobs must
    already exist.  No ``rights_observations`` rows are added because that API is
    append-only; the exact per-image credit row is instead stored on every entity,
    sequence, relevant occurrence edge, and frame.
    """

    database.initialize()
    item_id, members = _preflight(database, plan)
    projection_manifest_sha256 = plan.projection_manifest_sha256
    entity_ids: dict[str, str] = {}
    created_sequences = 0
    reused_sequences = 0
    occurrence_links = 0

    for record in plan.records:
        entity_id = entity_ids.get(record.entity_external_key)
        if entity_id is None:
            normalized_entity = taxonomy.normalize_entity_class(record.entity_class)
            entity_id = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=record.entity_external_key,
                representative_item_id=item_id,
                display_name=record.sprite_identity,
                entity_class=normalized_entity.value,
                entity_subclass=record.subject_role,
                species_or_type=None,
                taxonomy_version=taxonomy.version,
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "projection_manifest_sha256": projection_manifest_sha256,
                    "archive_sha256": plan.archive_sha256,
                    "repository_commit": plan.repository_commit,
                    "sprite_identity": record.sprite_identity,
                    "relative_script_path": record.relative_script_path,
                    "sprite_evidence_member_path": record.sprite_evidence_member_path,
                    "sprite_line_number": record.sprite_line_number,
                    "source_file": record.source_file,
                    "source_sheet_member_path": record.source_sheet_member_path,
                    "source_sheet_sha256": record.source_sheet_sha256,
                    "pixel_transforms": [plan.pixel_transform.as_metadata()],
                    "entity_class_candidates": list(record.entity_class_candidates),
                    "subject_role": record.subject_role,
                    "morphology_tags": list(record.morphology_tags),
                    "parent_subject": record.parent_subject,
                    "classification_basis": record.classification_basis,
                    "source_header_authors": list(record.source_header_authors),
                    "source_header_licenses": list(record.source_header_licenses),
                    "source_comments": list(record.source_comments),
                    "asset_credit": _credit_metadata(record.asset_credit),
                },
            )
            entity_ids[record.entity_external_key] = entity_id

        direction = record.direction
        motion = taxonomy.motion_condition(
            action=record.normalized_action,
            direction=direction,
            view=record.view,
        )
        metadata = _sequence_metadata(plan, record, projection_manifest_sha256)
        sequence_id = database.find_sequence_by_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
        )
        sequence_arguments = {
            "source_blob_sha256": record.source_sheet_sha256,
            "extraction_method": PROJECTION_VERSION,
            "extraction_confidence": 1.0,
            "width": record.frame_size.x,
            "height": record.frame_size.y,
            "frame_count": record.frame_count,
            "loop_mode": record.loop_mode,
            "action": motion.normalized_action,
            "direction": motion.direction,
            "quality_tier": "F0_lossless_source_sheet_exact_spr_timing",
            "metadata": metadata,
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
                "source_subject_role": record.subject_role,
                "parent_subject": record.parent_subject,
                "classification_basis": record.classification_basis,
                "asset_credit": _credit_metadata(record.asset_credit),
            },
        )
        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version=taxonomy.version,
            source_action=record.source_label,
            normalized_action=motion.normalized_action,
            action_family=motion.action_family,
            annotation_method=PROJECTION_VERSION,
            view=motion.view,
            direction=motion.direction,
            loopable=record.effective_repeat,
            cycle_frames=len(record.loop_data) if record.effective_repeat else None,
            phase_zero_frame=(record.effective_repeat_from if record.effective_repeat else 0),
            confidence=motion.confidence if record.normalized_action is not None else 0.0,
            conditioning={
                "source_label": record.source_label,
                "source_label_basis": record.source_label_basis,
                "adapter_normalized_action": record.normalized_action,
                "normalized_action_basis": record.normalized_action_basis,
                "comment_derived_action_only": True,
                "direction_evidence": (
                    "explicit_source_comment" if record.direction_hint is not None else "absent"
                ),
                "view_evidence": "absent",
                "fps": record.fps,
                "fps_source_token": record.fps_source_token,
                "timing_known": True,
                "exact_engine_timing": True,
                "repeat": record.repeat,
                "effective_repeat": record.effective_repeat,
                "effective_repeat_from": record.effective_repeat_from,
                "intro_frame_count": len(record.intro_data),
                "loop_tail_frame_count": len(record.loop_data),
                "state_occurrence_order_preserved": True,
                "asset_credit": _credit_metadata(record.asset_credit),
            },
        )

        for member_path, role, occurrence_metadata in _occurrence_specs(plan, record):
            database.link_sequence_occurrence(
                sequence_id=sequence_id,
                archive_blob_sha256=plan.archive_sha256,
                archive_member_ordinal=int(members[member_path]["ordinal"]),
                occurrence_role=role,
                metadata=occurrence_metadata,
            )
            occurrence_links += 1

        for ordinal, occurrence in enumerate(record.frame_occurrences):
            loop_tail_ordinal = (
                ordinal - record.effective_repeat_from if occurrence.in_loop_tail else None
            )
            database.add_sequence_frame(
                sequence_id=sequence_id,
                ordinal=ordinal,
                source_blob_sha256=record.source_sheet_sha256,
                source_frame_index=occurrence.source_frame_index,
                duration_ms=record.duration_ms,
                phase=_frame_phase(record, ordinal),
                direction=motion.direction,
                view=motion.view,
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "source_file": record.source_file,
                    "source_sheet_member_path": record.source_sheet_member_path,
                    "source_sheet_sha256": record.source_sheet_sha256,
                    "data_occurrence_index": occurrence.occurrence_index,
                    "source_frame_index": occurrence.source_frame_index,
                    "frame_rect": {
                        "left": occurrence.left,
                        "top": occurrence.top,
                        "right": occurrence.right,
                        "bottom": occurrence.bottom,
                        "width": occurrence.right - occurrence.left,
                        "height": occurrence.bottom - occurrence.top,
                        "column": occurrence.column,
                        "row": occurrence.row,
                        "coordinate_space": "source_sheet",
                    },
                    "within_declared_source_rect": occurrence.within_declared_source_rect,
                    "within_source_image": occurrence.within_source_image,
                    "fps": record.fps,
                    "fps_source_token": record.fps_source_token,
                    "duration_ms": record.duration_ms,
                    "in_intro_prefix": (
                        record.effective_repeat and ordinal < record.effective_repeat_from
                    ),
                    "in_loop_tail": occurrence.in_loop_tail,
                    "loop_tail_ordinal": loop_tail_ordinal,
                    "loop_tail_frame_count": len(record.loop_data),
                    "effective_repeat_from": record.effective_repeat_from,
                    "timeline_hot_spot": asdict(record.timeline_hot_spot),
                    "timeline_action_spot": asdict(record.timeline_action_spot),
                    "pixel_transforms": [plan.pixel_transform.as_metadata()],
                    "clipping_or_repair_applied": False,
                    "asset_credit": _credit_metadata(record.asset_credit),
                },
            )

    return OpenSurgeProjectionResult(
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=projection_manifest_sha256,
        projected_entities=plan.projected_entity_count,
        projected_sequences=plan.projected_sequence_count,
        projected_regular_animations=plan.projected_regular_animation_count,
        projected_transitions=plan.projected_transition_count,
        projected_frame_occurrences=plan.projected_frame_occurrence_count,
        projected_normalized_actions=plan.projected_normalized_action_count,
        projected_unknown_actions=plan.projected_unknown_action_count,
        created_sequences=created_sequences,
        reused_sequences=reused_sequences,
        occurrence_links=occurrence_links,
        excluded_candidate_sequences=plan.excluded_candidate_sequence_count,
        excluded_candidate_frame_occurrences=(plan.excluded_candidate_frame_occurrence_count),
        excluded_unsafe_occurrences=plan.excluded_unsafe_occurrence_count,
    )


def ingest_known_open_surge_sequences(
    database: IndexDB,
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> OpenSurgeProjectionResult:
    """Audit and project only the archive with the pinned Open Surge digest."""

    plan = plan_known_open_surge_projection(archive_path)
    if (
        plan.archive_sha256 != EXPECTED_OPEN_SURGE_ARCHIVE_SHA256
        or plan.repository_commit != OPEN_SURGE_COMMIT
    ):
        raise ValueError(
            "Refusing Open Surge projection for an unexpected archive or repository commit"
        )
    return project_open_surge_audit(database, plan, taxonomy)


__all__ = [
    "PROJECTION_VERSION",
    "RIGHTS_SCOPE",
    "SOURCE_ID",
    "PIXEL_TRANSFORM_OP",
    "PIXEL_TRANSFORM_SCHEMA",
    "OpenSurgePixelTransform",
    "OpenSurgePixelTransformEvidence",
    "OpenSurgeProjectionExclusion",
    "OpenSurgeProjectionPlan",
    "OpenSurgeProjectionReadiness",
    "OpenSurgeProjectionRecord",
    "OpenSurgeProjectionResult",
    "check_open_surge_projection_readiness",
    "ingest_known_open_surge_sequences",
    "plan_known_open_surge_projection",
    "plan_open_surge_projection",
    "project_open_surge_audit",
]
