"""Pure-plan, conservative projection for the pinned TMWA client-data audit.

Planning is write-free.  Readiness uses SQLite ``mode=ro`` and
``PRAGMA query_only``.  Projection writes only core provenance tables after a
strict preflight and intentionally does not open an ``IndexDB.transaction()``;
callers may wrap a batch in their own transaction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from spritelab.adapters.tmwa import (
    EXPECTED_INVENTORY_SHA256,
    EXPECTED_TMWA_ARCHIVE_SHA256,
    MANAPLUS_ENGINE_COMMIT,
    RIGHTS_SCOPE_CAVEAT,
    SOURCE_ID,
    TMWA_CLIENT_DATA_COMMIT,
    ArchiveMemberEvidence,
    EffectiveTrack,
    EngineSourceEvidence,
    EvidenceDocument,
    ImageRightsAssessment,
    ResolvedFrame,
    RightsClaim,
    SemanticBinding,
    SemanticCorpusAudit,
    SourceImageAudit,
    TimelineCommand,
    TmwaArchiveAudit,
    TmwaAuditCounts,
    XmlCommentClaim,
    audit_known_tmwa_archive,
)
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

PROJECTION_VERSION = "tmwa_exact_provenance_v3"
QUALITY_TIER = "provenance_safe_exact_cells"
EXPECTED_TMWA_PROJECTION_MANIFEST_SHA256 = (
    "b0962aafa56673c294b6a81a1748430097d8170f5813d4d2f7daf36bc3dfbe6d"
)


@dataclass(frozen=True)
class TmwaProjectionEntityRelation:
    external_entity_key: str
    corpus: str
    entity_external_id: str
    display_name: str | None
    entity_class: str
    entity_subclass: str
    classification_basis: str
    quadruped_cue: bool | None
    entity_location_member_path: str
    entity_location_logical_path: str
    entity_location_line_number: int
    layer_ordinal: int
    layer_count: int
    layer_role: str
    sprite_literal: str
    palette_expression: str | None
    attributes: tuple[tuple[str, str], ...]
    binding_member_path: str
    binding_logical_path: str
    binding_line_number: int


@dataclass(frozen=True)
class TmwaProjectionRecord:
    sequence_source_key: str
    resource_entity_external_key: str
    definition_logical_path: str
    definition_member_path: str
    definition_family: str
    source_documents: tuple[str, ...]
    variant_index: int
    source_action: str
    adapter_normalized_action: str | None
    normalized_action_basis: str
    action_ordinal: int
    action_location_member_path: str
    action_location_logical_path: str
    action_location_line_number: int
    direction_literal: str
    adapter_normalized_direction: str | None
    animation_ordinal: int
    animation_location_member_path: str
    animation_location_logical_path: str
    animation_location_line_number: int
    commands: tuple[TimelineCommand, ...]
    loop_mode: str
    frames: tuple[ResolvedFrame, ...]
    source_image: SourceImageAudit
    entity_relations: tuple[TmwaProjectionEntityRelation, ...]
    rights_assessment: ImageRightsAssessment

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def width(self) -> int:
        return max(frame.rectangle.width for frame in self.frames)

    @property
    def height(self) -> int:
        return max(frame.rectangle.height for frame in self.frames)


@dataclass(frozen=True)
class TmwaProjectionExclusion:
    sequence_source_key: str
    track: EffectiveTrack
    candidate_relations: tuple[TmwaProjectionEntityRelation, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TmwaProjectionPlan:
    archive_sha256: str
    repository_commit: str
    engine_semantics_commit: str
    fix_dead_animation: bool
    fix_dead_animation_basis: str
    archive_inventory_sha256: str
    counts: TmwaAuditCounts
    archive_members: tuple[ArchiveMemberEvidence, ...]
    semantic_corpora: tuple[SemanticCorpusAudit, ...]
    semantic_bindings: tuple[SemanticBinding, ...]
    xml_comments: tuple[XmlCommentClaim, ...]
    rights_documents: tuple[EvidenceDocument, ...]
    rights_claims: tuple[RightsClaim, ...]
    engine_evidence: tuple[EngineSourceEvidence, ...]
    records: tuple[TmwaProjectionRecord, ...]
    exclusions: tuple[TmwaProjectionExclusion, ...]
    rights_scope_caveat: str = RIGHTS_SCOPE_CAVEAT

    @property
    def projected_sequence_count(self) -> int:
        return len(self.records)

    @property
    def projected_frame_count(self) -> int:
        return sum(item.frame_count for item in self.records)

    @property
    def projected_definition_count(self) -> int:
        return len({item.definition_logical_path for item in self.records})

    @property
    def projected_source_image_count(self) -> int:
        return len({item.source_image.logical_path for item in self.records})

    @property
    def projected_semantic_entity_count(self) -> int:
        return len(
            {
                relation.external_entity_key
                for record in self.records
                for relation in record.entity_relations
            }
        )

    @property
    def required_members(self) -> tuple[ArchiveMemberEvidence, ...]:
        return tuple(item for item in self.archive_members if item.content_sha256 is not None)

    @property
    def required_member_paths(self) -> tuple[str, ...]:
        return tuple(item.member_path for item in self.required_members)

    @property
    def required_source_images(self) -> tuple[SourceImageAudit, ...]:
        by_path = {item.source_image.member_path: item.source_image for item in self.records}
        return tuple(by_path[path] for path in sorted(by_path))

    def canonical_payload(self) -> dict[str, Any]:
        """Return the complete deterministic plan payload, excluding local paths."""

        return {
            "projection_version": PROJECTION_VERSION,
            "archive_sha256": self.archive_sha256,
            "repository_commit": self.repository_commit,
            "engine_semantics_commit": self.engine_semantics_commit,
            "fix_dead_animation": self.fix_dead_animation,
            "fix_dead_animation_basis": self.fix_dead_animation_basis,
            "archive_inventory_sha256": self.archive_inventory_sha256,
            "counts": asdict(self.counts),
            "archive_members": [asdict(item) for item in self.archive_members],
            "semantic_corpora": [asdict(item) for item in self.semantic_corpora],
            "semantic_bindings": [asdict(item) for item in self.semantic_bindings],
            "xml_comments": [asdict(item) for item in self.xml_comments],
            "rights_documents": [asdict(item) for item in self.rights_documents],
            "rights_claims": [asdict(item) for item in self.rights_claims],
            "rights_scope_caveat": self.rights_scope_caveat,
            "engine_evidence": [asdict(item) for item in self.engine_evidence],
            "records": [asdict(item) for item in self.records],
            "exclusions": [asdict(item) for item in self.exclusions],
        }

    @cached_property
    def projection_manifest_sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TmwaProjectionReadiness:
    database_path: str
    archive_sha256: str
    projection_manifest_sha256: str
    query_only_enabled: bool
    archive_blob_present: bool
    archive_inventory_present: bool
    archive_inventory_exact: bool
    source_item_count: int
    planned_archive_member_count: int
    present_archive_member_count: int
    required_member_count: int
    present_member_count: int
    extracted_member_blob_count: int
    missing_archive_member_paths: tuple[str, ...]
    missing_member_paths: tuple[str, ...]
    member_metadata_mismatches: tuple[str, ...]
    member_hash_mismatches: tuple[str, ...]
    unregistered_member_blobs: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.query_only_enabled
            and self.archive_blob_present
            and self.archive_inventory_present
            and self.archive_inventory_exact
            and self.source_item_count == 1
            and self.present_archive_member_count == self.planned_archive_member_count
            and self.present_member_count == self.required_member_count
            and self.extracted_member_blob_count == self.required_member_count
            and not self.missing_archive_member_paths
            and not self.missing_member_paths
            and not self.member_metadata_mismatches
            and not self.member_hash_mismatches
            and not self.unregistered_member_blobs
        )


@dataclass(frozen=True)
class TmwaProjectionResult:
    archive_sha256: str
    projection_manifest_sha256: str
    projected_resource_entities: int
    projected_semantic_entities: int
    projected_definitions: int
    projected_sequences: int
    projected_frames: int
    created_sequences: int
    reused_sequences: int
    occurrence_links: int
    excluded_tracks: int
    rights_observations_added: int = 0

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


def _stable_json_key(prefix: str, payload: dict[str, Any]) -> str:
    return (
        prefix
        + ":"
        + json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _identity(audit: TmwaArchiveAudit) -> dict[str, str]:
    return {
        "archive_sha256": audit.archive_sha256,
        "repository_commit": audit.repository_commit,
        "engine_semantics_commit": audit.engine_semantics_commit,
    }


def _sequence_source_key(audit: TmwaArchiveAudit, track: EffectiveTrack) -> str:
    return _stable_json_key(
        "tmwa-sequence-v1",
        {
            **_identity(audit),
            "definition_logical_path": track.definition_logical_path,
            "variant_index": track.variant_index,
            "source_action": track.source_action,
            "action_member_path": track.action_location.member_path,
            "action_line_number": track.action_location.line_number,
            "direction_literal": track.direction_literal,
            "animation_ordinal": track.animation_ordinal,
        },
    )


def _resource_entity_key(audit: TmwaArchiveAudit, definition_path: str) -> str:
    return _stable_json_key(
        "tmwa-animation-resource-v1",
        {**_identity(audit), "definition_logical_path": definition_path},
    )


def _resource_identity_label(definition_path: str) -> str:
    filename = definition_path.rsplit("/", 1)[-1]
    return filename.removesuffix(".xml")


def _semantic_entity_key(audit: TmwaArchiveAudit, binding: SemanticBinding) -> str:
    return _stable_json_key(
        "tmwa-semantic-entity-v1",
        {
            **_identity(audit),
            "corpus": binding.corpus,
            "entity_external_id": binding.entity_external_id,
            "entity_document": binding.entity_location.logical_path,
        },
    )


def _relation(
    audit: TmwaArchiveAudit,
    binding: SemanticBinding,
) -> TmwaProjectionEntityRelation:
    return TmwaProjectionEntityRelation(
        external_entity_key=_semantic_entity_key(audit, binding),
        corpus=binding.corpus,
        entity_external_id=binding.entity_external_id,
        display_name=binding.entity_name,
        entity_class=binding.classification.entity_class,
        entity_subclass=binding.classification.entity_subclass,
        classification_basis=binding.classification.basis,
        quadruped_cue=binding.classification.quadruped_cue,
        entity_location_member_path=binding.entity_location.member_path,
        entity_location_logical_path=binding.entity_location.logical_path,
        entity_location_line_number=binding.entity_location.line_number,
        layer_ordinal=binding.layer_ordinal,
        layer_count=binding.layer_count,
        layer_role=binding.layer_role,
        sprite_literal=binding.sprite_literal,
        palette_expression=binding.palette_expression,
        attributes=binding.attributes,
        binding_member_path=binding.location.member_path,
        binding_logical_path=binding.location.logical_path,
        binding_line_number=binding.location.line_number,
    )


def _safe_bindings(bindings: tuple[SemanticBinding, ...]) -> tuple[SemanticBinding, ...]:
    return tuple(
        binding
        for binding in bindings
        if binding.corpus == "monsters"
        and binding.layer_role == "complete_single_layer_entity"
        and binding.layer_count == 1
        and binding.layer_ordinal == 0
        and binding.palette_expression is None
        and not binding.attributes
        and binding.definition_resolved
    )


def _eligibility_reasons(
    track: EffectiveTrack,
    safe_bindings: tuple[SemanticBinding, ...],
    rights: ImageRightsAssessment | None,
) -> tuple[str, ...]:
    reasons = set(track.issues)
    if not safe_bindings:
        reasons.add("no_safe_complete_single_layer_monster_binding")
    if track.variant_index != 0:
        reasons.add("only_runtime_default_variant_is_considered")
    if track.normalized_direction is None:
        reasons.add("direction_literal_not_in_reviewed_shared_mapping")
    if track.control_flow_present:
        reasons.add("runtime_control_flow_track")
    if track.loop_mode not in {"loop", "hold", "one_shot_return_to_stand"}:
        reasons.add("loop_or_end_semantics_unresolved")
    if track.declared_frame_count != len(track.frames):
        reasons.add("declared_and_resolved_frame_counts_differ")
    if not track.frames:
        reasons.add("no_resolved_frames")
        return tuple(sorted(reasons))
    image_paths = {frame.source_image.logical_path for frame in track.frames}
    image_hashes = {frame.source_image.sha256 for frame in track.frames}
    if len(image_paths) != 1 or len(image_hashes) != 1:
        reasons.add("track_uses_multiple_source_sheets")
    if any(frame.palette_expression is not None for frame in track.frames):
        reasons.add("imageset_palette_transform_unresolved")
    if len(track.frames) > 1 and any(frame.duration_ms <= 0 for frame in track.frames):
        reasons.add("multi_frame_track_has_nonpositive_duration")
    if any(frame.duration_ms < 0 for frame in track.frames):
        reasons.add("negative_duration")
    if rights is None:
        reasons.add("source_image_rights_assessment_absent")
    elif rights.status != "documented_path_claim":
        reasons.add(f"source_image_rights_{rights.status}")
    return tuple(sorted(reasons))


def plan_tmwa_projection(audit: TmwaArchiveAudit) -> TmwaProjectionPlan:
    """Build a deterministic write-free plan from a completed audit."""

    bindings_by_path: defaultdict[str, list[SemanticBinding]] = defaultdict(list)
    for binding in audit.semantic_bindings:
        bindings_by_path[binding.definition_logical_path].append(binding)
    rights_by_path = audit.rights_by_image_path
    records: list[TmwaProjectionRecord] = []
    exclusions: list[TmwaProjectionExclusion] = []
    for track in audit.effective_tracks:
        bindings = tuple(bindings_by_path.get(track.definition_logical_path, ()))
        safe = _safe_bindings(bindings)
        relations = tuple(
            sorted(
                (_relation(audit, binding) for binding in safe),
                key=lambda item: item.external_entity_key,
            )
        )
        image = track.frames[0].source_image if track.frames else None
        rights = rights_by_path.get(image.logical_path) if image is not None else None
        reasons = _eligibility_reasons(track, safe, rights)
        sequence_key = _sequence_source_key(audit, track)
        if reasons:
            exclusions.append(
                TmwaProjectionExclusion(
                    sequence_source_key=sequence_key,
                    track=track,
                    candidate_relations=relations,
                    reasons=reasons,
                )
            )
            continue
        if image is None or rights is None:
            raise AssertionError("eligible TMWA track lost its image or rights assessment")
        records.append(
            TmwaProjectionRecord(
                sequence_source_key=sequence_key,
                resource_entity_external_key=_resource_entity_key(
                    audit, track.definition_logical_path
                ),
                definition_logical_path=track.definition_logical_path,
                definition_member_path=track.definition_member_path,
                definition_family=track.definition_family,
                source_documents=track.source_documents,
                variant_index=track.variant_index,
                source_action=track.source_action,
                adapter_normalized_action=track.normalized_action,
                normalized_action_basis=track.normalized_action_basis,
                action_ordinal=track.action_ordinal,
                action_location_member_path=track.action_location.member_path,
                action_location_logical_path=track.action_location.logical_path,
                action_location_line_number=track.action_location.line_number,
                direction_literal=track.direction_literal,
                adapter_normalized_direction=track.normalized_direction,
                animation_ordinal=track.animation_ordinal,
                animation_location_member_path=track.animation_location.member_path,
                animation_location_logical_path=track.animation_location.logical_path,
                animation_location_line_number=track.animation_location.line_number,
                commands=track.commands,
                loop_mode=track.loop_mode,
                frames=track.frames,
                source_image=image,
                entity_relations=relations,
                rights_assessment=rights,
            )
        )
    records.sort(key=lambda item: item.sequence_source_key)
    exclusions.sort(key=lambda item: item.sequence_source_key)
    return TmwaProjectionPlan(
        archive_sha256=audit.archive_sha256,
        repository_commit=audit.repository_commit,
        engine_semantics_commit=audit.engine_semantics_commit,
        fix_dead_animation=audit.fix_dead_animation,
        fix_dead_animation_basis=audit.fix_dead_animation_basis,
        archive_inventory_sha256=audit.inventory_sha256,
        counts=audit.counts,
        archive_members=audit.members,
        semantic_corpora=audit.semantic_corpora,
        semantic_bindings=audit.semantic_bindings,
        xml_comments=audit.xml_comments,
        rights_documents=audit.rights.documents,
        rights_claims=audit.rights.claims,
        engine_evidence=audit.engine_evidence,
        records=tuple(records),
        exclusions=tuple(exclusions),
    )


def plan_known_tmwa_projection(archive_path: str | Path) -> TmwaProjectionPlan:
    """Audit and plan only the exact acquired TMWA CAS object."""

    plan = plan_tmwa_projection(audit_known_tmwa_archive(archive_path))
    if (
        plan.archive_sha256 != EXPECTED_TMWA_ARCHIVE_SHA256
        or plan.repository_commit != TMWA_CLIENT_DATA_COMMIT
        or plan.engine_semantics_commit != MANAPLUS_ENGINE_COMMIT
        or plan.archive_inventory_sha256 != EXPECTED_INVENTORY_SHA256
        or plan.projection_manifest_sha256 != EXPECTED_TMWA_PROJECTION_MANIFEST_SHA256
    ):
        raise ValueError("Refusing a TMWA plan outside the complete immutable pin")
    return plan


def _readonly_connection(database_path: str | Path) -> sqlite3.Connection:
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def check_tmwa_projection_readiness(
    database_path: str | Path,
    plan: TmwaProjectionPlan,
) -> TmwaProjectionReadiness:
    """Check every pinned prerequisite through a query-only SQLite handle."""

    expected_members = {item.member_path: item for item in plan.required_members}
    with _readonly_connection(database_path) as connection:
        query_only = int(connection.execute("PRAGMA query_only").fetchone()[0]) == 1
        archive_blob_present = (
            connection.execute(
                "SELECT 1 FROM blobs WHERE sha256=? LIMIT 1", (plan.archive_sha256,)
            ).fetchone()
            is not None
        )
        inventory = connection.execute(
            """
            SELECT archive_format, member_count, file_count,
                   total_uncompressed_bytes, total_compressed_bytes, inventory_sha256
            FROM archive_inventories WHERE archive_blob_sha256=?
            """,
            (plan.archive_sha256,),
        ).fetchone()
        source_item_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM items i JOIN item_blobs ib ON ib.item_id=i.id
                WHERE i.source_id=? AND ib.blob_sha256=?
                """,
                (SOURCE_ID, plan.archive_sha256),
            ).fetchone()[0]
        )
        rows = connection.execute(
            """
            SELECT am.ordinal, am.member_path, am.normalized_path, am.member_kind,
                   am.size_bytes, am.compressed_bytes, am.crc32,
                   am.compression_method, am.modified_at, am.extracted_blob_sha256,
                   b.sha256 AS registered_blob_sha256
            FROM archive_members am
            LEFT JOIN blobs b ON b.sha256=am.extracted_blob_sha256
            WHERE am.archive_blob_sha256=?
            """,
            (plan.archive_sha256,),
        ).fetchall()
    inventory_present = inventory is not None
    inventory_exact = bool(
        inventory is not None
        and str(inventory["archive_format"]).casefold() == "zip"
        and int(inventory["member_count"]) == plan.counts.zip_member_count
        and int(inventory["file_count"]) == plan.counts.regular_file_member_count
        and int(inventory["total_uncompressed_bytes"]) == plan.counts.expanded_member_bytes
        and int(inventory["total_compressed_bytes"]) == plan.counts.compressed_member_bytes
        and str(inventory["inventory_sha256"]) == plan.archive_inventory_sha256
    )
    actual: dict[str, sqlite3.Row] = {}
    for row in rows:
        actual[str(row["member_path"])] = row
        actual[str(row["normalized_path"])] = row
    missing_archive: list[str] = []
    metadata_mismatches: list[str] = []
    for expected in plan.archive_members:
        row = actual.get(expected.member_path)
        if row is None:
            missing_archive.append(expected.member_path)
            continue
        expected_modified_at = (
            f"{expected.modified_at[0]:04d}-{expected.modified_at[1]:02d}-"
            f"{expected.modified_at[2]:02d}T{expected.modified_at[3]:02d}:"
            f"{expected.modified_at[4]:02d}:{expected.modified_at[5]:02d}"
        )
        observed = (
            int(row["ordinal"]),
            str(row["member_path"]),
            str(row["normalized_path"]),
            str(row["member_kind"]),
            int(row["size_bytes"]),
            int(row["compressed_bytes"]),
            None if row["crc32"] is None else int(row["crc32"]),
            None if row["compression_method"] is None else int(row["compression_method"]),
            str(row["modified_at"]),
        )
        wanted = (
            expected.ordinal,
            expected.member_path,
            expected.normalized_path,
            expected.member_kind,
            expected.size_bytes,
            expected.compressed_bytes,
            expected.crc32,
            expected.compression_method,
            expected_modified_at,
        )
        if observed != wanted:
            metadata_mismatches.append(expected.member_path)
    missing: list[str] = []
    mismatches: list[str] = []
    unregistered: list[str] = []
    present = extracted = 0
    for path, expected in sorted(expected_members.items()):
        row = actual.get(path)
        if row is None:
            missing.append(path)
            continue
        present += 1
        observed = row["extracted_blob_sha256"]
        if observed is not None:
            extracted += 1
        if str(observed) != expected.content_sha256:
            mismatches.append(path)
        elif row["registered_blob_sha256"] is None:
            unregistered.append(path)
    return TmwaProjectionReadiness(
        database_path=str(Path(database_path).resolve()),
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=plan.projection_manifest_sha256,
        query_only_enabled=query_only,
        archive_blob_present=archive_blob_present,
        archive_inventory_present=inventory_present,
        archive_inventory_exact=inventory_exact,
        source_item_count=source_item_count,
        planned_archive_member_count=len(plan.archive_members),
        present_archive_member_count=len(rows),
        required_member_count=len(expected_members),
        present_member_count=present,
        extracted_member_blob_count=extracted,
        missing_archive_member_paths=tuple(missing_archive),
        missing_member_paths=tuple(missing),
        member_metadata_mismatches=tuple(metadata_mismatches),
        member_hash_mismatches=tuple(mismatches),
        unregistered_member_blobs=tuple(unregistered),
    )


def _preflight(
    database: IndexDB,
    plan: TmwaProjectionPlan,
) -> tuple[str, dict[str, sqlite3.Row]]:
    readiness = check_tmwa_projection_readiness(database.path, plan)
    if not readiness.ready:
        problems: list[str] = []
        if not readiness.query_only_enabled:
            problems.append("query-only readiness handle was not enabled")
        if not readiness.archive_blob_present:
            problems.append("archive blob is not registered")
        if not readiness.archive_inventory_present:
            problems.append("archive inventory is missing")
        elif not readiness.archive_inventory_exact:
            problems.append("archive inventory does not match every pinned fact")
        if readiness.source_item_count != 1:
            problems.append(
                f"expected one {SOURCE_ID!r} source item, found {readiness.source_item_count}"
            )
        if readiness.missing_archive_member_paths:
            problems.append(
                "archive inventory members absent: "
                + ", ".join(readiness.missing_archive_member_paths[:5])
            )
        if readiness.missing_member_paths:
            problems.append(
                "required members absent: " + ", ".join(readiness.missing_member_paths[:5])
            )
        if readiness.member_metadata_mismatches:
            problems.append(
                "archive member metadata differs: "
                + ", ".join(readiness.member_metadata_mismatches[:5])
            )
        if readiness.member_hash_mismatches:
            problems.append(
                "extracted member hashes differ: " + ", ".join(readiness.member_hash_mismatches[:5])
            )
        if readiness.unregistered_member_blobs:
            problems.append(
                "extracted member blobs unregistered: "
                + ", ".join(readiness.unregistered_member_blobs[:5])
            )
        raise ValueError("TMWA projection prerequisites are not ready: " + "; ".join(problems))
    with database.connect() as connection:
        source_row = connection.execute(
            """
            SELECT i.id
            FROM items i JOIN item_blobs ib ON ib.item_id=i.id
            WHERE i.source_id=? AND ib.blob_sha256=?
            ORDER BY i.id
            """,
            (SOURCE_ID, plan.archive_sha256),
        ).fetchone()
        rows = connection.execute(
            """
            SELECT ordinal, member_path, normalized_path, extracted_blob_sha256
            FROM archive_members WHERE archive_blob_sha256=?
            """,
            (plan.archive_sha256,),
        ).fetchall()
    if source_row is None:
        raise AssertionError("ready TMWA projection has no source item")
    members: dict[str, sqlite3.Row] = {}
    for row in rows:
        members[str(row["member_path"])] = row
        members[str(row["normalized_path"])] = row
    return str(source_row["id"]), members


def _resource_metadata(
    plan: TmwaProjectionPlan,
    record: TmwaProjectionRecord,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "engine_semantics_commit": plan.engine_semantics_commit,
        "fix_dead_animation": plan.fix_dead_animation,
        "fix_dead_animation_basis": plan.fix_dead_animation_basis,
        "identity_kind": "physical_sprite_definition_resource",
        "definition_logical_path": record.definition_logical_path,
        "definition_member_path": record.definition_member_path,
        "definition_family": record.definition_family,
        "identity_label": _resource_identity_label(record.definition_logical_path),
        "source_documents": list(record.source_documents),
        "semantic_identity_claim": False,
    }


def _semantic_entity_metadata(
    plan: TmwaProjectionPlan,
    relations: tuple[TmwaProjectionEntityRelation, ...],
    manifest_sha256: str,
) -> dict[str, Any]:
    first = relations[0]
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "identity_kind": "source_semantic_entity",
        "corpus": first.corpus,
        "entity_external_id": first.entity_external_id,
        "classification_basis": first.classification_basis,
        "quadruped_cue": first.quadruped_cue,
        "bindings": [asdict(item) for item in relations],
    }


def _sequence_metadata(
    plan: TmwaProjectionPlan,
    record: TmwaProjectionRecord,
    manifest_sha256: str,
) -> dict[str, Any]:
    scoped_comment_members = set(record.source_documents)
    scoped_comment_members.update(
        relation.binding_member_path for relation in record.entity_relations
    )
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "archive_inventory_sha256": plan.archive_inventory_sha256,
        "repository_commit": plan.repository_commit,
        "engine_semantics_commit": plan.engine_semantics_commit,
        "fix_dead_animation": plan.fix_dead_animation,
        "fix_dead_animation_basis": plan.fix_dead_animation_basis,
        "sequence_source_key": record.sequence_source_key,
        "definition_logical_path": record.definition_logical_path,
        "definition_member_path": record.definition_member_path,
        "identity_label": _resource_identity_label(record.definition_logical_path),
        "source_documents": list(record.source_documents),
        "variant_index": record.variant_index,
        "source_action": record.source_action,
        "adapter_normalized_action": record.adapter_normalized_action,
        "normalized_action_basis": record.normalized_action_basis,
        "action_ordinal": record.action_ordinal,
        "action_location": {
            "member_path": record.action_location_member_path,
            "logical_path": record.action_location_logical_path,
            "line_number": record.action_location_line_number,
        },
        "direction_literal": record.direction_literal,
        "adapter_normalized_direction": record.adapter_normalized_direction,
        "animation_ordinal": record.animation_ordinal,
        "animation_location": {
            "member_path": record.animation_location_member_path,
            "logical_path": record.animation_location_logical_path,
            "line_number": record.animation_location_line_number,
        },
        "timeline_commands": [asdict(item) for item in record.commands],
        "loop_mode": record.loop_mode,
        "source_image": asdict(record.source_image),
        "geometry": {
            "coordinate_space": "source_png",
            "cell_rectangles_preserved_per_frame": True,
            "crops_materialized": False,
            "compositing_performed": False,
            "recoloring_performed": False,
        },
        "entity_bindings": [asdict(item) for item in record.entity_relations],
        "scoped_xml_comments": [
            asdict(item)
            for item in plan.xml_comments
            if item.location.member_path in scoped_comment_members
        ],
        "rights": {
            "assessment": asdict(record.rights_assessment),
            "scope_caveat": RIGHTS_SCOPE_CAVEAT,
            "rights_observation_added": False,
        },
    }


def _frame_phase(record: TmwaProjectionRecord, ordinal: int) -> float:
    if record.loop_mode == "loop":
        return ordinal / record.frame_count
    if record.frame_count == 1:
        return 0.0
    return ordinal / (record.frame_count - 1)


def _normalized_loop_mode(record: TmwaProjectionRecord) -> str:
    """Map exact engine behavior to the shared training loop vocabulary."""

    if record.loop_mode in {"loop", "hold"}:
        return "loop"
    if record.loop_mode == "one_shot_return_to_stand":
        return "one_shot"
    raise AssertionError(f"projected TMWA record has unsupported loop mode {record.loop_mode!r}")


def _occurrence_specs(
    plan: TmwaProjectionPlan,
    record: TmwaProjectionRecord,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    specs[(record.source_image.member_path, "tmwa_source_png")] = {
        "logical_path": record.source_image.logical_path,
        "sha256": record.source_image.sha256,
        "dimensions": [record.source_image.width, record.source_image.height],
        "mode": record.source_image.mode,
    }
    for member_path in record.source_documents:
        specs[(member_path, "tmwa_sprite_definition_evidence")] = {
            "definition_logical_path": record.definition_logical_path,
            "runtime_include_closure_member": True,
        }
    for relation in record.entity_relations:
        key = (relation.binding_member_path, "tmwa_semantic_binding_evidence")
        metadata = specs.setdefault(key, {"bindings": []})
        metadata["bindings"].append(asdict(relation))
    for evidence in plan.rights_documents:
        specs[(evidence.member_path, "tmwa_rights_evidence")] = {
            "logical_path": evidence.logical_path,
            "sha256": evidence.sha256,
            "scope": evidence.scope,
            "scope_caveat": RIGHTS_SCOPE_CAVEAT,
        }
    return tuple(
        (member_path, role, metadata) for (member_path, role), metadata in sorted(specs.items())
    )


def project_tmwa_audit(
    database: IndexDB,
    plan: TmwaProjectionPlan,
    taxonomy: Taxonomy,
) -> TmwaProjectionResult:
    """Idempotently project the already-planned safe subset into core tables.

    The database schema and all archive rows must already exist.  This function
    performs strict preflight before its first write and never opens an internal
    transaction.  No crops, dyes, composites, or rights observations are made.
    """

    item_id, members = _preflight(database, plan)
    manifest_sha256 = plan.projection_manifest_sha256
    created_sequences = reused_sequences = occurrence_links = 0
    entity_ids: dict[str, str] = {}
    relation_groups: defaultdict[str, dict[TmwaProjectionEntityRelation, None]] = defaultdict(dict)
    for record in plan.records:
        for relation in record.entity_relations:
            relation_groups[relation.external_entity_key][relation] = None

    for record in plan.records:
        resource_id = entity_ids.get(record.resource_entity_external_key)
        if resource_id is None:
            resource_class = taxonomy.normalize_entity_class(
                record.entity_relations[0].entity_class
            ).value
            resource_id = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=record.resource_entity_external_key,
                representative_item_id=item_id,
                display_name=_resource_identity_label(record.definition_logical_path),
                entity_class=resource_class,
                entity_subclass=f"tmwa_sprite_{record.definition_family}",
                species_or_type=record.entity_relations[0].entity_subclass,
                taxonomy_version=taxonomy.version,
                metadata=_resource_metadata(plan, record, manifest_sha256),
            )
            entity_ids[record.resource_entity_external_key] = resource_id
        for relation in record.entity_relations:
            if relation.external_entity_key in entity_ids:
                continue
            group = tuple(relation_groups[relation.external_entity_key])
            normalized_class = taxonomy.normalize_entity_class(relation.entity_class)
            entity_ids[relation.external_entity_key] = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=relation.external_entity_key,
                representative_item_id=item_id,
                display_name=relation.display_name or relation.entity_external_id,
                entity_class=normalized_class.value,
                entity_subclass=relation.entity_subclass,
                species_or_type=relation.entity_subclass,
                taxonomy_version=taxonomy.version,
                metadata=_semantic_entity_metadata(plan, group, manifest_sha256),
            )

        motion = taxonomy.motion_condition(
            action=record.adapter_normalized_action,
            direction=record.adapter_normalized_direction,
            view=None,
        )
        normalized_loop_mode = _normalized_loop_mode(record)
        sequence_arguments = {
            "source_blob_sha256": record.source_image.sha256,
            "extraction_method": PROJECTION_VERSION,
            "extraction_confidence": 1.0,
            "width": record.width,
            "height": record.height,
            "frame_count": record.frame_count,
            "loop_mode": normalized_loop_mode,
            "action": motion.normalized_action,
            "direction": motion.direction,
            "quality_tier": QUALITY_TIER,
            "metadata": _sequence_metadata(plan, record, manifest_sha256),
        }
        sequence_id = database.find_sequence_by_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
        )
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
            entity_id=resource_id,
            role="primary",
            metadata={
                "identity_kind": "physical_sprite_definition_resource",
                "semantic_identity_claim": False,
                "definition_logical_path": record.definition_logical_path,
            },
        )
        for relation in record.entity_relations:
            database.link_sequence_subject(
                sequence_id=sequence_id,
                entity_id=entity_ids[relation.external_entity_key],
                role="source_entity_binding",
                metadata=asdict(relation),
            )
        loopable = normalized_loop_mode == "loop"
        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version=taxonomy.version,
            source_action=record.source_action,
            normalized_action=motion.normalized_action,
            action_family=motion.action_family,
            annotation_method=PROJECTION_VERSION,
            view=motion.view,
            direction=motion.direction,
            loopable=loopable,
            cycle_frames=record.frame_count if record.loop_mode == "loop" else None,
            phase_zero_frame=0 if loopable else None,
            confidence=motion.confidence,
            conditioning={
                "source_action_literal": record.source_action,
                "adapter_normalized_action": record.adapter_normalized_action,
                "normalized_action_basis": record.normalized_action_basis,
                "source_direction_literal": record.direction_literal,
                "adapter_normalized_direction": record.adapter_normalized_direction,
                "view_evidence": "absent",
                "loop_mode_from_pinned_engine_semantics": record.loop_mode,
                "normalized_loop_mode": normalized_loop_mode,
                "variant_index": record.variant_index,
            },
        )
        for member_path, role, metadata in _occurrence_specs(plan, record):
            member = members.get(member_path)
            if member is None:
                raise AssertionError(f"preflight lost required TMWA member {member_path}")
            database.link_sequence_occurrence(
                sequence_id=sequence_id,
                archive_blob_sha256=plan.archive_sha256,
                archive_member_ordinal=int(member["ordinal"]),
                occurrence_role=role,
                metadata=metadata,
            )
            occurrence_links += 1
        for frame in record.frames:
            database.add_sequence_frame(
                sequence_id=sequence_id,
                ordinal=frame.ordinal,
                source_blob_sha256=frame.source_image.sha256,
                source_frame_index=frame.source_frame_index,
                duration_ms=frame.duration_ms,
                phase=_frame_phase(record, frame.ordinal),
                direction=motion.direction,
                view=motion.view,
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "source_frame_index_semantics": "manaplus_imageset_row_major_cell_index",
                    "unshifted_frame_index": frame.unshifted_frame_index,
                    "variant_index": frame.variant_index,
                    "declaring_variant_count": frame.declaring_variant_count,
                    "declaring_variant_offset": frame.declaring_variant_offset,
                    "declared_duration_ms": frame.declared_duration_ms,
                    "duration_ms": frame.duration_ms,
                    "duration_adjustment_basis": frame.duration_adjustment_basis,
                    "frame_rect": {
                        "left": frame.rectangle.x,
                        "top": frame.rectangle.y,
                        "right": frame.rectangle.x + frame.rectangle.width,
                        "bottom": frame.rectangle.y + frame.rectangle.height,
                        "width": frame.rectangle.width,
                        "height": frame.rectangle.height,
                        "coordinate_space": "source_image",
                    },
                    "source_cell_xywh": [
                        frame.rectangle.x,
                        frame.rectangle.y,
                        frame.rectangle.width,
                        frame.rectangle.height,
                    ],
                    "source_coordinate_space_literal": "source_png",
                    "xml_offset": [frame.xml_offset_x, frame.xml_offset_y],
                    "engine_offset": [frame.engine_offset_x, frame.engine_offset_y],
                    "imageset_name": frame.imageset_name,
                    "imageset_source_literal": frame.imageset_source_literal,
                    "palette_expression": frame.palette_expression,
                    "source_image": asdict(frame.source_image),
                    "source_location": asdict(frame.location),
                    "loop_mode": record.loop_mode,
                    "normalized_loop_mode": normalized_loop_mode,
                    "pixel_transform_applied": False,
                    "composite_applied": False,
                    "crop_materialized": False,
                    "rights": {
                        "assessment": asdict(record.rights_assessment),
                        "scope_caveat": RIGHTS_SCOPE_CAVEAT,
                    },
                },
            )

    return TmwaProjectionResult(
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=manifest_sha256,
        projected_resource_entities=plan.projected_definition_count,
        projected_semantic_entities=plan.projected_semantic_entity_count,
        projected_definitions=plan.projected_definition_count,
        projected_sequences=plan.projected_sequence_count,
        projected_frames=plan.projected_frame_count,
        created_sequences=created_sequences,
        reused_sequences=reused_sequences,
        occurrence_links=occurrence_links,
        excluded_tracks=len(plan.exclusions),
    )


def ingest_known_tmwa_sequences(
    database: IndexDB,
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> TmwaProjectionResult:
    """Plan and project only the exact immutable TMWA archive."""

    return project_tmwa_audit(database, plan_known_tmwa_projection(archive_path), taxonomy)


__all__ = [
    "EXPECTED_TMWA_PROJECTION_MANIFEST_SHA256",
    "PROJECTION_VERSION",
    "QUALITY_TIER",
    "TmwaProjectionEntityRelation",
    "TmwaProjectionExclusion",
    "TmwaProjectionPlan",
    "TmwaProjectionReadiness",
    "TmwaProjectionRecord",
    "TmwaProjectionResult",
    "check_tmwa_projection_readiness",
    "ingest_known_tmwa_sequences",
    "plan_known_tmwa_projection",
    "plan_tmwa_projection",
    "project_tmwa_audit",
]
