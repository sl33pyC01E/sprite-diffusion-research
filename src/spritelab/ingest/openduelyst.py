"""Plan and project audited OpenDuelyst RSX/plist animations.

The adapter is the source of truth for declarations and runtime semantics.  This
module adds a conservative materialization gate and maps safe records to the
existing provenance schema.  Planning and readiness are read-only.  Projection
is idempotent for a stable source key and never creates individual frame blobs:
the encoded atlas remains the source blob while exact TexturePacker geometry is
stored on each ``sequence_frames`` row.

Repository-level CC0 evidence is retained with its exact scope.  It is never
converted into an asset-level license or creator assertion, and this module does
not add append-only ``rights_observations`` rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from spritelab.adapters.openduelyst import (
    EXPECTED_OPENDUELYST_ARCHIVE_SHA256,
    OPENDUELYST_COMMIT,
    AnimationSequence,
    AtlasSheet,
    EntityAnimationField,
    EntityMapping,
    EvidenceDocument,
    OpenDuelystAudit,
    RawExpression,
    ResourceAnimationDeclaration,
    SequenceFrame,
    Size,
    SourceCodeEvidence,
    audit_known_openduelyst_archive,
    runtime_frame_keys,
)
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

SOURCE_ID = "openduelyst"
PROJECTION_VERSION = "openduelyst_rsx_texturepacker_projection_v1"
RIGHTS_SCOPE_CAVEAT = (
    "The archived root files make a repository/project-level CC0-1.0 claim. "
    "The audit found no per-asset creator, license, provenance, or chain-of-title "
    "manifest. Vendor licenses apply only to their named subtrees. Do not promote "
    "the repository claim into independently verified asset-level rights metadata."
)

LoopMode = Literal["loop", "one_shot", "role_dependent", "unknown"]


@dataclass(frozen=True)
class PhysicalFrameAliasEvidence:
    """Other RSX aliases selecting the same physical plist frame key."""

    frame_key: str
    resource_aliases: tuple[str, ...]


@dataclass(frozen=True)
class OpenDuelystProjectionSubject:
    """One source card/entity mapping linked many-to-many to a sequence."""

    entity_external_key: str
    mapping_index: int
    roles_for_sequence: tuple[str, ...]
    identifier_expression: str | None
    identifier_token: str | None
    card_id: int | None
    card_kind: str | None
    card_name_expression: str | None
    localization_key: str | None
    display_name: str | None
    faction_expression: str | None
    race_expression: str | None
    animation_fields: tuple[EntityAnimationField, ...]
    evidence_relative_path: str
    evidence_member_path: str
    line_number: int

    @property
    def effective_display_name(self) -> str:
        return (
            self.display_name
            or self.identifier_token
            or self.identifier_expression
            or f"source mapping at {self.evidence_relative_path}:{self.line_number}"
        )


@dataclass(frozen=True)
class OpenDuelystProjectionRecord:
    """One nonempty RSX alias with exact, internally consistent atlas evidence."""

    sequence_source_key: str
    physical_entity_external_key: str
    resource_alias: str
    runtime_name: str
    category: str
    descriptor_line_number: int
    descriptor_member_path: str
    descriptor_raw_fields: tuple[RawExpression, ...]
    frame_prefix: str
    declared_frame_delay_seconds: float
    declared_frame_delay_expression: str
    runtime_delay_multiplier: float
    effective_frame_delay_seconds: float
    plist_path: str
    plist_member_path: str
    descriptor_image_path: str
    atlas_image_path: str
    atlas_image_member_path: str
    atlas_image_sha256: str
    atlas_image_width: int
    atlas_image_height: int
    atlas_image_mode: str
    atlas_image_format: str
    atlas_metadata_format: int
    atlas_metadata_size: Size | None
    atlas_metadata_matches_image_size: bool | None
    atlas_duplicate_frame_keys: tuple[str, ...]
    frames: tuple[SequenceFrame, ...]
    reconstructed_width: int
    reconstructed_height: int
    source_roles: tuple[str, ...]
    normalized_action: str | None
    normalized_action_basis: str
    loop_mode: LoopMode
    loop_basis: str
    direction_semantics: str
    ambiguity_reasons: tuple[str, ...]
    subjects: tuple[OpenDuelystProjectionSubject, ...]
    physical_resource_aliases: tuple[str, ...]
    runtime_name_aliases: tuple[str, ...]
    exact_timeline_aliases: tuple[str, ...]
    shared_physical_frames: tuple[PhysicalFrameAliasEvidence, ...]
    byte_identical_image_paths: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_ms(self) -> float:
        return self.effective_frame_delay_seconds * 1000.0

    @property
    def total_duration_ms(self) -> float:
        return self.frame_count * self.duration_ms

    @property
    def semantic_loopable(self) -> bool | None:
        if self.loop_mode == "loop":
            return True
        if self.loop_mode == "one_shot":
            return False
        return None


@dataclass(frozen=True)
class OpenDuelystProjectionExclusion:
    """A source alias quarantined without repairing or dropping its evidence."""

    sequence_source_key: str
    resource_alias: str
    runtime_name: str
    frame_prefix: str
    plist_path: str
    descriptor_image_path: str
    atlas_image_path: str | None
    descriptor_member_path: str
    descriptor_line_number: int
    frame_count: int
    reasons: tuple[str, ...]
    unsafe_frame_keys: tuple[str, ...]


@dataclass(frozen=True)
class OpenDuelystProjectionPlan:
    """Pure deterministic projection plan; constructing it performs no writes."""

    archive_sha256: str
    repository_commit: str
    root_prefix: str
    records: tuple[OpenDuelystProjectionRecord, ...]
    exclusions: tuple[OpenDuelystProjectionExclusion, ...]
    rights_evidence: tuple[EvidenceDocument, ...]
    source_code_evidence: tuple[SourceCodeEvidence, ...]
    card_lookup_member_path: str
    localization_member_path: str

    @property
    def projected_sequence_count(self) -> int:
        return len(self.records)

    @property
    def projected_frame_occurrence_count(self) -> int:
        return sum(record.frame_count for record in self.records)

    @property
    def projected_physical_entity_count(self) -> int:
        return len({record.physical_entity_external_key for record in self.records})

    @property
    def projected_mapped_entity_count(self) -> int:
        return len(
            {subject.entity_external_key for record in self.records for subject in record.subjects}
        )

    @property
    def projected_entity_count(self) -> int:
        keys = {record.physical_entity_external_key for record in self.records}
        keys.update(
            subject.entity_external_key for record in self.records for subject in record.subjects
        )
        return len(keys)

    @property
    def projected_subject_link_count(self) -> int:
        return self.projected_sequence_count + sum(len(record.subjects) for record in self.records)

    @property
    def projected_exact_action_count(self) -> int:
        return sum(record.normalized_action is not None for record in self.records)

    @property
    def projected_ambiguous_or_unmapped_action_count(self) -> int:
        return sum(record.normalized_action is None for record in self.records)

    @property
    def projected_loop_count(self) -> int:
        return sum(record.loop_mode == "loop" for record in self.records)

    @property
    def projected_one_shot_count(self) -> int:
        return sum(record.loop_mode == "one_shot" for record in self.records)

    @property
    def projected_role_dependent_loop_count(self) -> int:
        return sum(record.loop_mode == "role_dependent" for record in self.records)

    @property
    def projected_unknown_loop_count(self) -> int:
        return sum(record.loop_mode == "unknown" for record in self.records)

    @property
    def excluded_candidate_sequence_count(self) -> int:
        return len(self.exclusions)

    @property
    def excluded_candidate_frame_occurrence_count(self) -> int:
        return sum(exclusion.frame_count for exclusion in self.exclusions)

    @property
    def required_member_paths(self) -> tuple[str, ...]:
        paths = {
            self.card_lookup_member_path,
            self.localization_member_path,
            *(evidence.member_path for evidence in self.rights_evidence),
            *(evidence.member_path for evidence in self.source_code_evidence),
        }
        for record in self.records:
            paths.update(
                {
                    record.descriptor_member_path,
                    record.plist_member_path,
                    record.atlas_image_member_path,
                }
            )
            paths.update(subject.evidence_member_path for subject in record.subjects)
        return tuple(sorted(paths))

    @property
    def projection_manifest_sha256(self) -> str:
        payload = {
            "projection_version": PROJECTION_VERSION,
            "archive_sha256": self.archive_sha256,
            "repository_commit": self.repository_commit,
            "root_prefix": self.root_prefix,
            "records": [asdict(record) for record in self.records],
            "exclusions": [asdict(exclusion) for exclusion in self.exclusions],
            "rights_evidence": [asdict(evidence) for evidence in self.rights_evidence],
            "source_code_evidence": [asdict(evidence) for evidence in self.source_code_evidence],
            "card_lookup_member_path": self.card_lookup_member_path,
            "localization_member_path": self.localization_member_path,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OpenDuelystProjectionReadiness:
    """Query-only report of indexed evidence and source-atlas prerequisites."""

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
class OpenDuelystProjectionResult:
    """Core-row effects of one idempotent projection run."""

    archive_sha256: str
    projection_manifest_sha256: str
    projected_entities: int
    projected_physical_entities: int
    projected_mapped_entities: int
    projected_sequences: int
    projected_frame_occurrences: int
    projected_subject_links: int
    created_sequences: int
    reused_sequences: int
    occurrence_links: int
    excluded_candidate_sequences: int
    excluded_candidate_frame_occurrences: int
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


def _sequence_source_key(audit: OpenDuelystAudit, resource_alias: str) -> str:
    return _stable_json_key(
        "openduelyst-sequence-v1",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.repository_commit,
            "resource_alias": resource_alias,
        },
    )


def _physical_entity_external_key(
    audit: OpenDuelystAudit,
    sequence: AnimationSequence,
) -> str:
    return _stable_json_key(
        "openduelyst-physical-atlas-entity-v1",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.repository_commit,
            "image_path": sequence.image_path,
            "plist_path": sequence.plist_path,
        },
    )


def _mapped_entity_external_key(
    audit: OpenDuelystAudit,
    mapping: EntityMapping,
) -> str:
    return _stable_json_key(
        "openduelyst-source-entity-v1",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.repository_commit,
            "evidence_relative_path": mapping.evidence_relative_path,
            "line_number": mapping.line_number,
            "identifier_expression": mapping.identifier_expression,
            "card_id": mapping.card_id,
        },
    )


def _subject(
    audit: OpenDuelystAudit,
    sequence: AnimationSequence,
    mapping_index: int,
) -> OpenDuelystProjectionSubject:
    mapping = audit.entity_mappings[mapping_index]
    roles = tuple(
        reference.role
        for reference in mapping.animation_references
        if reference.resource_alias == sequence.resource_alias
    )
    return OpenDuelystProjectionSubject(
        entity_external_key=_mapped_entity_external_key(audit, mapping),
        mapping_index=mapping_index,
        roles_for_sequence=roles,
        identifier_expression=mapping.identifier_expression,
        identifier_token=mapping.identifier_token,
        card_id=mapping.card_id,
        card_kind=mapping.card_kind,
        card_name_expression=mapping.card_name_expression,
        localization_key=mapping.localization_key,
        display_name=mapping.display_name,
        faction_expression=mapping.faction_expression,
        race_expression=mapping.race_expression,
        animation_fields=mapping.animation_fields,
        evidence_relative_path=mapping.evidence_relative_path,
        evidence_member_path=mapping.evidence_member_path,
        line_number=mapping.line_number,
    )


def _sequence_exclusion_reasons(
    sequence: AnimationSequence,
    declaration: ResourceAnimationDeclaration | None,
    atlas: AtlasSheet | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    reasons: list[str] = []
    unsafe_frame_keys: list[str] = []

    if declaration is None:
        reasons.append("rsx_declaration_is_missing")
    else:
        declaration_facts = (
            declaration.alias,
            declaration.name,
            declaration.frame_prefix,
            declaration.plist_path,
            declaration.image_path,
            declaration.frame_delay,
            declaration.frame_delay_expression,
        )
        sequence_facts = (
            sequence.resource_alias,
            sequence.runtime_name,
            sequence.frame_prefix,
            sequence.plist_path,
            sequence.image_path,
            sequence.declared_frame_delay_seconds,
            sequence.declared_frame_delay_expression,
        )
        if declaration_facts != sequence_facts:
            reasons.append("resolved_sequence_disagrees_with_rsx_declaration")

    if not sequence.frames:
        reasons.append("frame_prefix_matches_no_plist_key")
    if atlas is None:
        reasons.append("declared_plist_has_no_texture_atlas")
        return tuple(dict.fromkeys(reasons)), ()
    if atlas.metadata_format != 2:
        reasons.append("texturepacker_metadata_format_is_not_2")
    if atlas.duplicate_frame_keys:
        reasons.append("atlas_has_duplicate_frame_keys")
    if atlas.image_relative_path is None or atlas.image_member_path is None:
        reasons.append("atlas_image_member_is_missing")
    elif sequence.image_path != atlas.image_relative_path:
        reasons.append("descriptor_image_path_differs_from_plist_texture_path")
    if atlas.image_sha256 is None:
        reasons.append("atlas_image_sha256_is_missing")
    if (
        atlas.image_width is None
        or atlas.image_height is None
        or atlas.image_width <= 0
        or atlas.image_height <= 0
    ):
        reasons.append("atlas_image_dimensions_are_missing_or_nonpositive")
    if atlas.image_mode is None or atlas.image_format is None:
        reasons.append("atlas_image_mode_or_format_is_missing")

    if not math.isfinite(sequence.declared_frame_delay_seconds) or (
        sequence.declared_frame_delay_seconds <= 0
    ):
        reasons.append("declared_frame_delay_is_not_positive_finite")
    expected_effective_delay = (
        sequence.declared_frame_delay_seconds * sequence.runtime_delay_multiplier
    )
    if (
        not math.isfinite(sequence.runtime_delay_multiplier)
        or sequence.runtime_delay_multiplier <= 0
        or not math.isclose(
            sequence.effective_frame_delay_seconds,
            expected_effective_delay,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        reasons.append("effective_delay_disagrees_with_runtime_multiplier")
    expected_total = len(sequence.frames) * sequence.effective_frame_delay_seconds
    if not math.isclose(
        sequence.total_duration_seconds,
        expected_total,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        reasons.append("total_duration_disagrees_with_frame_count_and_delay")

    atlas_frames = {frame.key: frame for frame in atlas.frames}
    expected_keys = runtime_frame_keys(tuple(atlas_frames), sequence.frame_prefix)
    actual_keys = tuple(frame.key for frame in sequence.frames)
    if actual_keys != expected_keys:
        reasons.append("runtime_frame_order_disagrees_with_plist_and_prefix")

    source_sizes = {
        (frame.source_size.width, frame.source_size.height) for frame in sequence.frames
    }
    if sequence.frames and len(source_sizes) != 1:
        reasons.append("sequence_frames_do_not_share_one_reconstructed_canvas_size")

    for ordinal, frame in enumerate(sequence.frames):
        frame_unsafe = False
        if frame.occurrence_index != ordinal:
            reasons.append("runtime_occurrence_indices_are_not_contiguous")
            frame_unsafe = True
        atlas_frame = atlas_frames.get(frame.key)
        if atlas_frame is None:
            reasons.append("resolved_frame_key_is_absent_from_atlas")
            frame_unsafe = True
        elif (
            frame.atlas_declaration_index != atlas_frame.declaration_index
            or frame.frame != atlas_frame.frame
            or frame.offset != atlas_frame.offset
            or frame.rotated != atlas_frame.rotated
            or frame.source_color_rect != atlas_frame.source_color_rect
            or frame.source_size != atlas_frame.source_size
        ):
            reasons.append("resolved_frame_geometry_disagrees_with_atlas_record")
            frame_unsafe = True

        packed = frame.frame
        source_rect = frame.source_color_rect
        source_size = frame.source_size
        if packed.width <= 0 or packed.height <= 0:
            reasons.append("packed_frame_dimensions_are_nonpositive")
            frame_unsafe = True
        if source_size.width <= 0 or source_size.height <= 0:
            reasons.append("source_canvas_dimensions_are_nonpositive")
            frame_unsafe = True
        if (
            source_rect.x < 0
            or source_rect.y < 0
            or source_rect.right > source_size.width
            or source_rect.bottom > source_size.height
        ):
            reasons.append("source_color_rect_is_outside_source_canvas")
            frame_unsafe = True
        if (packed.width, packed.height) != (
            source_rect.width,
            source_rect.height,
        ):
            reasons.append("packed_rect_size_differs_from_source_color_rect")
            frame_unsafe = True
        if frame.within_image_bounds is not True:
            reasons.append("packed_frame_is_outside_encoded_atlas_image")
            frame_unsafe = True
        if frame_unsafe:
            unsafe_frame_keys.append(frame.key)

    return tuple(dict.fromkeys(reasons)), tuple(dict.fromkeys(unsafe_frame_keys))


def _repository_relative_member(root_prefix: str, member_path: str) -> str:
    prefix = f"{root_prefix}/"
    if not member_path.startswith(prefix):
        raise ValueError(f"Archive member is outside audited root: {member_path!r}")
    return member_path[len(prefix) :]


def _record(
    audit: OpenDuelystAudit,
    sequence: AnimationSequence,
    declaration: ResourceAnimationDeclaration,
    atlas: AtlasSheet,
    *,
    physical_resource_aliases: tuple[str, ...],
    runtime_name_aliases: tuple[str, ...],
    exact_timeline_aliases: tuple[str, ...],
    shared_physical_frames: tuple[PhysicalFrameAliasEvidence, ...],
    byte_identical_image_paths: tuple[str, ...],
) -> OpenDuelystProjectionRecord:
    if (
        atlas.image_relative_path is None
        or atlas.image_member_path is None
        or atlas.image_sha256 is None
        or atlas.image_width is None
        or atlas.image_height is None
        or atlas.image_mode is None
        or atlas.image_format is None
        or atlas.metadata_format is None
    ):
        raise AssertionError("admitted OpenDuelyst sequence has incomplete atlas evidence")
    source_sizes = {
        (frame.source_size.width, frame.source_size.height) for frame in sequence.frames
    }
    if len(source_sizes) != 1:
        raise AssertionError("admitted OpenDuelyst sequence has varying source canvas sizes")
    reconstructed_width, reconstructed_height = next(iter(source_sizes))
    subjects = tuple(
        _subject(audit, sequence, mapping_index)
        for mapping_index in sequence.entity_mapping_indices
    )
    return OpenDuelystProjectionRecord(
        sequence_source_key=_sequence_source_key(audit, sequence.resource_alias),
        physical_entity_external_key=_physical_entity_external_key(audit, sequence),
        resource_alias=sequence.resource_alias,
        runtime_name=sequence.runtime_name,
        category=sequence.category,
        descriptor_line_number=declaration.line_number,
        descriptor_member_path=declaration.evidence_member_path,
        descriptor_raw_fields=declaration.raw_fields,
        frame_prefix=sequence.frame_prefix,
        declared_frame_delay_seconds=sequence.declared_frame_delay_seconds,
        declared_frame_delay_expression=sequence.declared_frame_delay_expression,
        runtime_delay_multiplier=sequence.runtime_delay_multiplier,
        effective_frame_delay_seconds=sequence.effective_frame_delay_seconds,
        plist_path=sequence.plist_path,
        plist_member_path=atlas.member_path,
        descriptor_image_path=sequence.image_path,
        atlas_image_path=atlas.image_relative_path,
        atlas_image_member_path=atlas.image_member_path,
        atlas_image_sha256=atlas.image_sha256,
        atlas_image_width=atlas.image_width,
        atlas_image_height=atlas.image_height,
        atlas_image_mode=atlas.image_mode,
        atlas_image_format=atlas.image_format,
        atlas_metadata_format=atlas.metadata_format,
        atlas_metadata_size=atlas.metadata_size,
        atlas_metadata_matches_image_size=atlas.metadata_matches_image_size,
        atlas_duplicate_frame_keys=atlas.duplicate_frame_keys,
        frames=sequence.frames,
        reconstructed_width=reconstructed_width,
        reconstructed_height=reconstructed_height,
        source_roles=sequence.source_roles,
        normalized_action=sequence.normalized_action,
        normalized_action_basis=sequence.normalized_action_basis,
        loop_mode=sequence.loop_mode,
        loop_basis=sequence.loop_basis,
        direction_semantics=sequence.direction_semantics,
        ambiguity_reasons=sequence.ambiguity_reasons,
        subjects=subjects,
        physical_resource_aliases=physical_resource_aliases,
        runtime_name_aliases=runtime_name_aliases,
        exact_timeline_aliases=exact_timeline_aliases,
        shared_physical_frames=shared_physical_frames,
        byte_identical_image_paths=byte_identical_image_paths,
    )


def plan_openduelyst_projection(audit: OpenDuelystAudit) -> OpenDuelystProjectionPlan:
    """Build a deterministic, write-free plan from an OpenDuelyst audit.

    A nonempty alias is admitted only when the RSX declaration, plist-selected
    order, encoded atlas image, and reconstructable source canvas agree exactly.
    Empty prefixes and contradictory geometry/evidence are quarantined whole;
    no prefix, rectangle, image path, or duration is repaired.
    """

    declarations = {declaration.alias: declaration for declaration in audit.declarations}
    atlases = {atlas.relative_path: atlas for atlas in audit.atlases}

    runtime_name_groups: defaultdict[str, list[str]] = defaultdict(list)
    physical_resource_groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    timeline_groups: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    physical_frame_groups: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for sequence in audit.sequences:
        runtime_name_groups[sequence.runtime_name].append(sequence.resource_alias)
        physical_resource_groups[(sequence.plist_path, sequence.image_path)].append(
            sequence.resource_alias
        )
        if sequence.frames:
            timeline_groups[
                (
                    sequence.plist_path,
                    tuple(frame.key for frame in sequence.frames),
                    sequence.declared_frame_delay_expression,
                )
            ].append(sequence.resource_alias)
        for frame in sequence.frames:
            physical_frame_groups[(sequence.plist_path, frame.key)].append(sequence.resource_alias)

    png_alias_paths: dict[str, tuple[str, ...]] = {}
    for group in audit.duplicate_groups:
        if group.kind != "byte_identical_png":
            continue
        for path in group.keys:
            png_alias_paths[path] = group.keys

    records: list[OpenDuelystProjectionRecord] = []
    exclusions: list[OpenDuelystProjectionExclusion] = []
    for sequence in audit.sequences:
        declaration = declarations.get(sequence.resource_alias)
        atlas = atlases.get(sequence.plist_path)
        reasons, unsafe_frame_keys = _sequence_exclusion_reasons(
            sequence,
            declaration,
            atlas,
        )
        if reasons:
            exclusions.append(
                OpenDuelystProjectionExclusion(
                    sequence_source_key=_sequence_source_key(audit, sequence.resource_alias),
                    resource_alias=sequence.resource_alias,
                    runtime_name=sequence.runtime_name,
                    frame_prefix=sequence.frame_prefix,
                    plist_path=sequence.plist_path,
                    descriptor_image_path=sequence.image_path,
                    atlas_image_path=(atlas.image_relative_path if atlas else None),
                    descriptor_member_path=sequence.evidence_member_path,
                    descriptor_line_number=sequence.line_number,
                    frame_count=sequence.frame_count,
                    reasons=reasons,
                    unsafe_frame_keys=unsafe_frame_keys,
                )
            )
            continue
        if declaration is None or atlas is None:
            raise AssertionError("admitted OpenDuelyst sequence lacks declaration or atlas")
        frame_aliases = tuple(
            PhysicalFrameAliasEvidence(
                frame_key=frame.key,
                resource_aliases=tuple(
                    dict.fromkeys(physical_frame_groups[(sequence.plist_path, frame.key)])
                ),
            )
            for frame in sequence.frames
            if len(set(physical_frame_groups[(sequence.plist_path, frame.key)])) > 1
        )
        timeline_signature = (
            sequence.plist_path,
            tuple(frame.key for frame in sequence.frames),
            sequence.declared_frame_delay_expression,
        )
        image_repository_path = _repository_relative_member(
            audit.root_prefix,
            atlas.image_member_path or "",
        )
        records.append(
            _record(
                audit,
                sequence,
                declaration,
                atlas,
                physical_resource_aliases=tuple(
                    dict.fromkeys(
                        physical_resource_groups[(sequence.plist_path, sequence.image_path)]
                    )
                ),
                runtime_name_aliases=tuple(
                    dict.fromkeys(runtime_name_groups[sequence.runtime_name])
                ),
                exact_timeline_aliases=(
                    tuple(dict.fromkeys(timeline_groups[timeline_signature]))
                    if len(set(timeline_groups[timeline_signature])) > 1
                    else ()
                ),
                shared_physical_frames=frame_aliases,
                byte_identical_image_paths=png_alias_paths.get(image_repository_path, ()),
            )
        )

    records.sort(key=lambda record: record.sequence_source_key)
    exclusions.sort(key=lambda exclusion: exclusion.sequence_source_key)
    rights_evidence = tuple(
        sorted(audit.evidence_documents, key=lambda evidence: evidence.member_path)
    )
    source_code_evidence = tuple(
        sorted(audit.source_code_evidence, key=lambda evidence: evidence.member_path)
    )
    return OpenDuelystProjectionPlan(
        archive_sha256=audit.archive_sha256,
        repository_commit=audit.repository_commit,
        root_prefix=audit.root_prefix,
        records=tuple(records),
        exclusions=tuple(exclusions),
        rights_evidence=rights_evidence,
        source_code_evidence=source_code_evidence,
        card_lookup_member_path=(f"{audit.root_prefix}/app/sdk/cards/cardsLookup.coffee"),
        localization_member_path=(f"{audit.root_prefix}/app/localization/locales/en/cards.json"),
    )


def plan_known_openduelyst_projection(
    archive_path: str | Path,
) -> OpenDuelystProjectionPlan:
    """Audit the exact pinned CAS archive and return a write-free plan."""

    return plan_openduelyst_projection(audit_known_openduelyst_archive(archive_path))


def _readonly_connection(database_path: str | Path) -> sqlite3.Connection:
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def check_openduelyst_projection_readiness(
    database_path: str | Path,
    plan: OpenDuelystProjectionPlan,
) -> OpenDuelystProjectionReadiness:
    """Inspect exact prerequisites through a query-only SQLite connection."""

    required_paths = plan.required_member_paths
    expected_image_hashes: dict[str, str] = {}
    for record in plan.records:
        previous = expected_image_hashes.setdefault(
            record.atlas_image_member_path,
            record.atlas_image_sha256,
        )
        if previous != record.atlas_image_sha256:
            raise ValueError(
                "Projection plan assigns multiple hashes to atlas image member "
                f"{record.atlas_image_member_path!r}"
            )
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
    missing_image_blobs: list[str] = []
    hash_mismatches: list[str] = []
    for member_path, expected_hash in sorted(expected_image_hashes.items()):
        row = members.get(member_path)
        if row is None:
            continue
        actual_hash = row["extracted_blob_sha256"]
        registered_hash = row["registered_blob_sha256"]
        if actual_hash is None or registered_hash is None:
            missing_image_blobs.append(member_path)
        elif str(actual_hash) != expected_hash:
            hash_mismatches.append(
                f"{member_path}: expected {expected_hash}, indexed {actual_hash}"
            )
    return OpenDuelystProjectionReadiness(
        database_path=str(Path(database_path).resolve()),
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=plan.projection_manifest_sha256,
        archive_blob_present=archive_blob_present,
        source_item_count=source_item_count,
        required_member_count=len(required_paths),
        present_member_count=len(required_paths) - len(missing_paths),
        required_source_image_count=len(expected_image_hashes),
        present_source_image_blob_count=(
            len(expected_image_hashes) - len(missing_image_blobs) - len(hash_mismatches)
        ),
        missing_member_paths=missing_paths,
        missing_source_image_blobs=tuple(missing_image_blobs),
        source_image_hash_mismatches=tuple(hash_mismatches),
    )


def _archive_members(
    database: IndexDB,
    archive_sha256: str,
) -> dict[str, sqlite3.Row]:
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
            "The OpenDuelyst archive has no indexed source item for "
            f"source_id={SOURCE_ID!r}: {archive_sha256}"
        )
    return str(row[0])


def _preflight(
    database: IndexDB,
    plan: OpenDuelystProjectionPlan,
) -> tuple[str, dict[str, sqlite3.Row]]:
    with database.connect() as connection:
        inventory = connection.execute(
            "SELECT 1 FROM archive_inventories WHERE archive_blob_sha256=?",
            (plan.archive_sha256,),
        ).fetchone()
    if inventory is None:
        raise ValueError(f"OpenDuelyst archive inventory is missing: {plan.archive_sha256}")
    item_id = _item_id(database, plan.archive_sha256)
    members = _archive_members(database, plan.archive_sha256)
    missing = [path for path in plan.required_member_paths if path not in members]
    if missing:
        raise ValueError(
            "Projection evidence members are missing from archive_members: "
            + ", ".join(missing[:10])
        )
    expected_images = {
        record.atlas_image_member_path: record.atlas_image_sha256 for record in plan.records
    }
    for member_path, expected_hash in expected_images.items():
        indexed_hash = members[member_path]["extracted_blob_sha256"]
        if indexed_hash is None:
            raise ValueError(f"Source atlas image has not been extracted into CAS: {member_path}")
        if str(indexed_hash) != expected_hash:
            raise ValueError(
                "Source atlas image CAS hash does not match audited ZIP bytes for "
                f"{member_path}: expected {expected_hash}, indexed {indexed_hash}"
            )
    return item_id, members


def _rights_scope_metadata(plan: OpenDuelystProjectionPlan) -> dict[str, Any]:
    repository_claims = [
        evidence for evidence in plan.rights_evidence if evidence.scope == "repository_project"
    ]
    subtree_evidence = [
        evidence for evidence in plan.rights_evidence if evidence.scope != "repository_project"
    ]
    return {
        "scope": "repository_project_claim_only_not_asset_level",
        "caveat": RIGHTS_SCOPE_CAVEAT,
        "repository_claim_identifiers": sorted(
            {
                identifier
                for evidence in repository_claims
                for identifier in evidence.detected_license_identifiers
            }
        ),
        "repository_claim_evidence": [
            {
                "member_path": evidence.member_path,
                "sha256": evidence.sha256,
                "detected_license_identifiers": list(evidence.detected_license_identifiers),
                "scope": evidence.scope,
            }
            for evidence in repository_claims
        ],
        "non_asset_subtree_evidence": [
            {
                "member_path": evidence.member_path,
                "sha256": evidence.sha256,
                "scope": evidence.scope,
                "notes": evidence.notes,
            }
            for evidence in subtree_evidence
        ],
        "per_asset_manifest_present": False,
        "asset_license_expression": None,
        "asset_creator": None,
        "rights_observation_added": False,
    }


def _frame_phase(
    ordinal: int,
    frame_count: int,
    loop_mode: LoopMode,
) -> float | None:
    if loop_mode in {"role_dependent", "unknown"}:
        return None
    if frame_count <= 1:
        return 0.0
    if loop_mode == "loop":
        return ordinal / frame_count
    return ordinal / (frame_count - 1)


def _sequence_metadata(
    plan: OpenDuelystProjectionPlan,
    record: OpenDuelystProjectionRecord,
    projection_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": projection_manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "resource_alias": record.resource_alias,
        "runtime_name": record.runtime_name,
        "runtime_name_aliases": list(record.runtime_name_aliases),
        "runtime_name_collision": len(record.runtime_name_aliases) > 1,
        "category": record.category,
        "descriptor_member_path": record.descriptor_member_path,
        "descriptor_line_number": record.descriptor_line_number,
        "descriptor_raw_fields": [asdict(field) for field in record.descriptor_raw_fields],
        "frame_prefix": record.frame_prefix,
        "plist_path": record.plist_path,
        "plist_member_path": record.plist_member_path,
        "descriptor_image_path": record.descriptor_image_path,
        "atlas_image_path": record.atlas_image_path,
        "atlas_image_member_path": record.atlas_image_member_path,
        "atlas_image_sha256": record.atlas_image_sha256,
        "atlas_image_dimensions": [
            record.atlas_image_width,
            record.atlas_image_height,
        ],
        "atlas_image_mode": record.atlas_image_mode,
        "atlas_image_format": record.atlas_image_format,
        "atlas_metadata_format": record.atlas_metadata_format,
        "atlas_metadata_size": (
            asdict(record.atlas_metadata_size) if record.atlas_metadata_size is not None else None
        ),
        "atlas_metadata_matches_image_size": (record.atlas_metadata_matches_image_size),
        "atlas_duplicate_frame_keys": list(record.atlas_duplicate_frame_keys),
        "reconstructed_canvas_size": [
            record.reconstructed_width,
            record.reconstructed_height,
        ],
        "runtime_frame_key_order": [frame.key for frame in record.frames],
        "atlas_declaration_index_order": [frame.atlas_declaration_index for frame in record.frames],
        "declared_frame_delay_seconds": record.declared_frame_delay_seconds,
        "declared_frame_delay_expression": record.declared_frame_delay_expression,
        "runtime_delay_multiplier": record.runtime_delay_multiplier,
        "effective_frame_delay_seconds": record.effective_frame_delay_seconds,
        "duration_ms_per_occurrence": record.duration_ms,
        "total_duration_ms": record.total_duration_ms,
        "source_roles": list(record.source_roles),
        "adapter_normalized_action": record.normalized_action,
        "normalized_action_basis": record.normalized_action_basis,
        "loop_mode": record.loop_mode,
        "loop_basis": record.loop_basis,
        "direction": None,
        "direction_semantics": record.direction_semantics,
        "ambiguity_reasons": list(record.ambiguity_reasons),
        "entity_mappings": [asdict(subject) for subject in record.subjects],
        "physical_entity_external_key": record.physical_entity_external_key,
        "physical_resource_aliases": list(record.physical_resource_aliases),
        "exact_timeline_aliases": list(record.exact_timeline_aliases),
        "shared_physical_frames": [asdict(evidence) for evidence in record.shared_physical_frames],
        "byte_identical_image_paths": list(record.byte_identical_image_paths),
        "runtime_order_preserved": True,
        "exact_engine_timing": True,
        "geometry_coordinate_space": "encoded_atlas",
        "texturepacker_reconstruction_metadata_preserved": True,
        "individual_frame_pixels_materialized": False,
        "clipping_rotation_or_trim_repair_applied": False,
        "schema_limitations": {
            "sequence_width_height": (
                "Core schema has one width/height pair; admission therefore requires "
                "one exact sourceSize across the sequence."
            ),
            "frame_bbox": (
                "sequence_frames has no bbox columns; exact packed frame/sourceColorRect/"
                "sourceSize/offset/rotated facts are retained in frame metadata."
            ),
            "source_action": (
                "motion_annotations has one source_action scalar; all exact source roles "
                "remain in conditioning metadata and ambiguous roles project as unknown."
            ),
        },
        "source_code_evidence": [asdict(evidence) for evidence in plan.source_code_evidence],
        "rights_scope": _rights_scope_metadata(plan),
    }


def _occurrence_specs(
    plan: OpenDuelystProjectionPlan,
    record: OpenDuelystProjectionRecord,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    common = {
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "projection_version": PROJECTION_VERSION,
        "resource_alias": record.resource_alias,
    }
    specs: list[tuple[str, str, dict[str, Any]]] = [
        (
            record.atlas_image_member_path,
            "openduelyst_source_atlas_image",
            {
                **common,
                "image_path": record.atlas_image_path,
                "sha256": record.atlas_image_sha256,
                "dimensions": [
                    record.atlas_image_width,
                    record.atlas_image_height,
                ],
                "byte_identical_image_paths": list(record.byte_identical_image_paths),
            },
        ),
        (
            record.plist_member_path,
            "openduelyst_texturepacker_plist",
            {
                **common,
                "plist_path": record.plist_path,
                "frame_prefix": record.frame_prefix,
                "metadata_format": record.atlas_metadata_format,
            },
        ),
        (
            record.descriptor_member_path,
            "openduelyst_rsx_animation_declaration",
            {
                **common,
                "line_number": record.descriptor_line_number,
                "runtime_name": record.runtime_name,
                "raw_fields": [asdict(field) for field in record.descriptor_raw_fields],
            },
        ),
    ]
    for evidence in plan.source_code_evidence:
        specs.append(
            (
                evidence.member_path,
                "openduelyst_runtime_semantics_evidence",
                {
                    **common,
                    "sha256": evidence.sha256,
                    "line_numbers": list(evidence.line_numbers),
                    "establishes": evidence.establishes,
                },
            )
        )

    subjects_by_member: defaultdict[str, list[OpenDuelystProjectionSubject]] = defaultdict(list)
    for subject in record.subjects:
        subjects_by_member[subject.evidence_member_path].append(subject)
    for member_path, subjects in sorted(subjects_by_member.items()):
        specs.append(
            (
                member_path,
                "openduelyst_entity_animation_mapping",
                {
                    **common,
                    "mappings": [asdict(subject) for subject in subjects],
                },
            )
        )
    if record.subjects:
        specs.append(
            (
                plan.card_lookup_member_path,
                "openduelyst_card_identifier_mapping",
                {
                    **common,
                    "identifiers": [
                        {
                            "identifier_token": subject.identifier_token,
                            "card_id": subject.card_id,
                        }
                        for subject in record.subjects
                    ],
                },
            )
        )
    if any(subject.localization_key for subject in record.subjects):
        specs.append(
            (
                plan.localization_member_path,
                "openduelyst_english_localization_evidence",
                {
                    **common,
                    "localized_names": [
                        {
                            "localization_key": subject.localization_key,
                            "display_name": subject.display_name,
                        }
                        for subject in record.subjects
                        if subject.localization_key is not None
                    ],
                },
            )
        )
    for evidence in plan.rights_evidence:
        if evidence.scope != "repository_project":
            continue
        specs.append(
            (
                evidence.member_path,
                "openduelyst_repository_rights_claim_evidence",
                {
                    **common,
                    "sha256": evidence.sha256,
                    "detected_license_identifiers": list(evidence.detected_license_identifiers),
                    "scope": evidence.scope,
                    "rights_scope_caveat": RIGHTS_SCOPE_CAVEAT,
                    "asset_level_claim": False,
                },
            )
        )
    return tuple(specs)


def _physical_entity_metadata(
    plan: OpenDuelystProjectionPlan,
    record: OpenDuelystProjectionRecord,
    projection_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": projection_manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "identity_kind": "physical_atlas_resource",
        "semantic_identity_claim": False,
        "plist_path": record.plist_path,
        "plist_member_path": record.plist_member_path,
        "image_path": record.atlas_image_path,
        "image_member_path": record.atlas_image_member_path,
        "image_sha256": record.atlas_image_sha256,
        "resource_aliases": list(record.physical_resource_aliases),
        "byte_identical_image_paths": list(record.byte_identical_image_paths),
        "rights_scope": _rights_scope_metadata(plan),
    }


def _mapped_entity_metadata(
    plan: OpenDuelystProjectionPlan,
    subject: OpenDuelystProjectionSubject,
    projection_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": projection_manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "identity_kind": "source_card_entity_mapping",
        "identifier_expression": subject.identifier_expression,
        "identifier_token": subject.identifier_token,
        "card_id": subject.card_id,
        "card_kind": subject.card_kind,
        "card_name_expression": subject.card_name_expression,
        "localization_key": subject.localization_key,
        "display_name": subject.display_name,
        "faction_expression": subject.faction_expression,
        "race_expression": subject.race_expression,
        "animation_fields": [asdict(field) for field in subject.animation_fields],
        "evidence_relative_path": subject.evidence_relative_path,
        "evidence_member_path": subject.evidence_member_path,
        "line_number": subject.line_number,
        "classification_basis": (
            "No humanoid/animal/monster class is inferred from card names or art; "
            "raw card kind and race expressions are retained."
        ),
        "rights_scope": _rights_scope_metadata(plan),
    }


def project_openduelyst_audit(
    database: IndexDB,
    plan: OpenDuelystProjectionPlan,
    taxonomy: Taxonomy,
) -> OpenDuelystProjectionResult:
    """Idempotently project a precomputed safe plan into core DB tables.

    The source item, archive inventory/members, and source-atlas CAS blobs must
    already exist.  No frame pixels are extracted and no rights observation is
    added.  Repository-level rights caveats and exact source geometry are stored
    in row metadata instead.
    """

    database.initialize()
    item_id, members = _preflight(database, plan)
    projection_manifest_sha256 = plan.projection_manifest_sha256
    entity_ids: dict[str, str] = {}
    created_sequences = 0
    reused_sequences = 0
    occurrence_links = 0
    rights_scope = _rights_scope_metadata(plan)

    for record in plan.records:
        physical_entity_id = entity_ids.get(record.physical_entity_external_key)
        if physical_entity_id is None:
            physical_class = taxonomy.normalize_entity_class(None)
            physical_entity_id = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=record.physical_entity_external_key,
                representative_item_id=item_id,
                display_name=f"physical atlas {record.plist_path}",
                entity_class=physical_class.value,
                entity_subclass="physical_atlas_resource",
                species_or_type=None,
                taxonomy_version=taxonomy.version,
                metadata=_physical_entity_metadata(
                    plan,
                    record,
                    projection_manifest_sha256,
                ),
            )
            entity_ids[record.physical_entity_external_key] = physical_entity_id

        subject_entity_ids: list[tuple[OpenDuelystProjectionSubject, str]] = []
        for subject in record.subjects:
            entity_id = entity_ids.get(subject.entity_external_key)
            if entity_id is None:
                entity_class = taxonomy.normalize_entity_class(None)
                entity_id = database.upsert_entity(
                    source_id=SOURCE_ID,
                    external_identity_key=subject.entity_external_key,
                    representative_item_id=item_id,
                    display_name=subject.effective_display_name,
                    entity_class=entity_class.value,
                    entity_subclass="source_card_entity_mapping",
                    species_or_type=None,
                    taxonomy_version=taxonomy.version,
                    metadata=_mapped_entity_metadata(
                        plan,
                        subject,
                        projection_manifest_sha256,
                    ),
                )
                entity_ids[subject.entity_external_key] = entity_id
            subject_entity_ids.append((subject, entity_id))

        motion = taxonomy.motion_condition(
            action=record.normalized_action,
            direction=None,
            view=None,
        )
        sequence_id = database.find_sequence_by_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
        )
        sequence_arguments = {
            "source_blob_sha256": record.atlas_image_sha256,
            "extraction_method": PROJECTION_VERSION,
            "extraction_confidence": 1.0,
            "width": record.reconstructed_width,
            "height": record.reconstructed_height,
            "frame_count": record.frame_count,
            "loop_mode": record.loop_mode,
            "action": motion.normalized_action,
            "direction": motion.direction,
            "quality_tier": "F0_source_atlas_exact_geometry_and_runtime_timing",
            "metadata": _sequence_metadata(
                plan,
                record,
                projection_manifest_sha256,
            ),
        }
        if sequence_id is None:
            sequence_id = database.create_sequence(
                item_id=item_id,
                **sequence_arguments,
            )
            created_sequences += 1
        else:
            database.update_sequence_facts(
                sequence_id=sequence_id,
                **sequence_arguments,
            )
            reused_sequences += 1
        database.register_sequence_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
            sequence_id=sequence_id,
        )

        database.link_sequence_subject(
            sequence_id=sequence_id,
            entity_id=physical_entity_id,
            role="primary",
            metadata={
                "identity_kind": "physical_atlas_resource",
                "semantic_identity_claim": False,
                "plist_path": record.plist_path,
                "image_path": record.atlas_image_path,
                "rights_scope": rights_scope,
            },
        )
        for subject, entity_id in subject_entity_ids:
            database.link_sequence_subject(
                sequence_id=sequence_id,
                entity_id=entity_id,
                role="source_entity_mapping",
                metadata={
                    "mapping_index": subject.mapping_index,
                    "roles_for_sequence": list(subject.roles_for_sequence),
                    "identifier_expression": subject.identifier_expression,
                    "card_id": subject.card_id,
                    "evidence_member_path": subject.evidence_member_path,
                    "line_number": subject.line_number,
                    "many_to_many_mapping_preserved": True,
                    "rights_scope": rights_scope,
                },
            )

        source_action = record.source_roles[0] if len(record.source_roles) == 1 else None
        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version=taxonomy.version,
            source_action=source_action,
            normalized_action=motion.normalized_action,
            action_family=motion.action_family,
            annotation_method=PROJECTION_VERSION,
            view=motion.view,
            direction=motion.direction,
            loopable=record.semantic_loopable,
            cycle_frames=(record.frame_count if record.loop_mode == "loop" else None),
            phase_zero_frame=(0 if record.loop_mode in {"loop", "one_shot"} else None),
            confidence=(
                motion.confidence
                if record.normalized_action is not None and motion.normalized_action != "unknown"
                else 0.0
            ),
            conditioning={
                "source_roles": list(record.source_roles),
                "adapter_normalized_action": record.normalized_action,
                "normalized_action_basis": record.normalized_action_basis,
                "taxonomy_method": motion.method,
                "taxonomy_confidence": motion.confidence,
                "loop_mode": record.loop_mode,
                "loop_basis": record.loop_basis,
                "loopable_is_role_dependent": (record.loop_mode == "role_dependent"),
                "direction_semantics": record.direction_semantics,
                "runtime_frame_key_order": [frame.key for frame in record.frames],
                "declared_frame_delay_expression": (record.declared_frame_delay_expression),
                "declared_frame_delay_seconds": (record.declared_frame_delay_seconds),
                "runtime_delay_multiplier": record.runtime_delay_multiplier,
                "effective_frame_delay_seconds": (record.effective_frame_delay_seconds),
                "exact_engine_timing": True,
                "runtime_order_preserved": True,
                "geometry_coordinate_space": "encoded_atlas",
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

        shared_frame_aliases = {
            evidence.frame_key: evidence.resource_aliases
            for evidence in record.shared_physical_frames
        }
        for ordinal, frame in enumerate(record.frames):
            database.add_sequence_frame(
                sequence_id=sequence_id,
                ordinal=ordinal,
                source_blob_sha256=record.atlas_image_sha256,
                source_frame_index=frame.atlas_declaration_index,
                duration_ms=record.duration_ms,
                phase=_frame_phase(
                    ordinal,
                    record.frame_count,
                    record.loop_mode,
                ),
                direction=motion.direction,
                view=motion.view,
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "resource_alias": record.resource_alias,
                    "runtime_name": record.runtime_name,
                    "runtime_occurrence_index": frame.occurrence_index,
                    "atlas_declaration_index": frame.atlas_declaration_index,
                    "physical_frame_key": [record.plist_path, frame.key],
                    "frame_key": frame.key,
                    "final_numeric_token": frame.final_numeric_token,
                    "packed_frame_rect": asdict(frame.frame),
                    "rotated": frame.rotated,
                    "source_color_rect": asdict(frame.source_color_rect),
                    "source_size": asdict(frame.source_size),
                    "offset": asdict(frame.offset),
                    "is_trimmed": frame.is_trimmed,
                    "within_encoded_image_bounds": (frame.within_image_bounds),
                    "packed_rect_coordinate_space": "encoded_atlas",
                    "reconstruction_requires_rotation": frame.rotated,
                    "reconstruction_requires_trim_canvas": (
                        frame.is_trimmed or frame.offset.x != 0 or frame.offset.y != 0
                    ),
                    "texturepacker_reconstruction_metadata_complete": True,
                    "frame_pixels_materialized": False,
                    "duration_ms": record.duration_ms,
                    "declared_frame_delay_expression": (record.declared_frame_delay_expression),
                    "runtime_delay_multiplier": (record.runtime_delay_multiplier),
                    "physical_frame_aliases": list(shared_frame_aliases.get(frame.key, ())),
                    "rights_scope": rights_scope,
                },
            )

    return OpenDuelystProjectionResult(
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=projection_manifest_sha256,
        projected_entities=plan.projected_entity_count,
        projected_physical_entities=plan.projected_physical_entity_count,
        projected_mapped_entities=plan.projected_mapped_entity_count,
        projected_sequences=plan.projected_sequence_count,
        projected_frame_occurrences=plan.projected_frame_occurrence_count,
        projected_subject_links=plan.projected_subject_link_count,
        created_sequences=created_sequences,
        reused_sequences=reused_sequences,
        occurrence_links=occurrence_links,
        excluded_candidate_sequences=plan.excluded_candidate_sequence_count,
        excluded_candidate_frame_occurrences=(plan.excluded_candidate_frame_occurrence_count),
    )


def ingest_known_openduelyst_sequences(
    database: IndexDB,
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> OpenDuelystProjectionResult:
    """Audit and project only the exact pinned OpenDuelyst CAS archive."""

    plan = plan_known_openduelyst_projection(archive_path)
    if (
        plan.archive_sha256 != EXPECTED_OPENDUELYST_ARCHIVE_SHA256
        or plan.repository_commit != OPENDUELYST_COMMIT
    ):
        raise ValueError("Refusing OpenDuelyst projection for an unexpected archive or commit")
    return project_openduelyst_audit(database, plan, taxonomy)


__all__ = [
    "PROJECTION_VERSION",
    "RIGHTS_SCOPE_CAVEAT",
    "SOURCE_ID",
    "OpenDuelystProjectionExclusion",
    "OpenDuelystProjectionPlan",
    "OpenDuelystProjectionReadiness",
    "OpenDuelystProjectionRecord",
    "OpenDuelystProjectionResult",
    "OpenDuelystProjectionSubject",
    "PhysicalFrameAliasEvidence",
    "check_openduelyst_projection_readiness",
    "ingest_known_openduelyst_sequences",
    "plan_known_openduelyst_projection",
    "plan_openduelyst_projection",
    "project_openduelyst_audit",
]
