"""Conservative DB projection for audited Flare animation timelines.

The adapter remains the source of truth for parsing.  This module admits only
complete effective action/direction tracks whose ordered slots resolve to exact,
in-bounds PNG rectangles.  Planning and readiness checks are write-free; the
projector uses stable source keys and core-table upserts without creating crops
or append-only rights observations.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spritelab.adapters.flare import (
    EXPECTED_FLARE_ARCHIVE_SHA256,
    FLARE_ACTIVE_MODS,
    FLARE_DEFAULT_ENGINE_FPS,
    FLARE_ENGINE_COMMIT,
    FLARE_GAME_COMMIT,
    AnimationUsage,
    EntityBinding,
    EvidenceDocument,
    FlareArchiveAudit,
    HeroLayerOrder,
    ImageBinding,
    IncludeDirective,
    Point,
    Rectangle,
    SourceImageAudit,
    SourceLocation,
    TickSchedule,
    audit_known_flare_archive,
)
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

SOURCE_ID = "flare_empyrean"
PROJECTION_VERSION = "flare_empyrean_animation_projection_v1"
QUALITY_TIER = "F0_source_png_exact_flare_geometry_default60_timing"
RIGHTS_SCOPE_CAVEAT = (
    "Repository- and mod-scoped license/credit evidence is retained at its "
    "declared scope. It is not upgraded to per-PNG authorship, attribution, "
    "chain-of-title, or an independently verified asset license."
)
OPTIONAL_MODS_EXCLUDED = ("minicore", "alpha_demo", "minicore_alpha", "devlab")


@dataclass(frozen=True)
class FlareProjectionFrame:
    """One exact effective frame slot in action order."""

    ordinal: int
    source_frame_index: int
    effective_direction: int
    effective_direction_token: str
    effective_direction_name: str
    authored_direction: int
    authored_direction_name: str
    explicit: bool
    fallback_from_direction: int | None
    rectangle: Rectangle
    offset: Point
    image_id: str
    image_logical_path: str
    image_member_path: str
    image_sha256: str
    image_width: int
    image_height: int
    image_mode: str
    image_format: str | None
    image_has_transparency: bool
    source_location: SourceLocation
    default_60hz_tick_count: int
    default_60hz_duration_ms: float


@dataclass(frozen=True)
class FlareProjectionEntityRelation:
    """A source entity definition that explicitly selects this animation."""

    external_entity_key: str
    entity_kind: str
    definition_path: str
    member_path: str
    source_mod: str
    display_name: str | None
    humanoid: bool | None
    categories: tuple[str, ...]
    animation_location: SourceLocation
    is_template: bool
    projected_entity_class: str


@dataclass(frozen=True)
class FlareProjectionUsageRelation:
    """One exact power/effect/item/entity usage of this visual definition."""

    external_entity_key: str
    usage_kind: str
    owner_id: str | None
    owner_name: str | None
    location: SourceLocation
    projected_entity_class: str


@dataclass(frozen=True)
class FlareAttachmentCandidate:
    """An exact candidate edge, never a resolved avatar/body selection."""

    external_entity_key: str
    item_id: str
    item_name: str | None
    layer_slot: str
    gfx_id: str
    candidate_animation_paths: tuple[str, ...]
    candidate_animation_path: str
    candidate_body_variant: str | None
    body_variant_choice_resolved: bool
    direction: int
    direction_token: str
    direction_name: str
    hero_layers_back_to_front: tuple[str, ...]
    layer_index_back_to_front: int | None
    layer_position_resolved: bool
    location: SourceLocation


@dataclass(frozen=True)
class FlareAttachmentQuarantine:
    """An item attachment whose runtime composition choice remains unresolved."""

    item_id: str
    item_name: str | None
    layer_slot: str
    gfx_id: str
    candidate_animation_paths: tuple[str, ...]
    missing_layer_directions: tuple[str, ...]
    location: SourceLocation
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FlareProjectionRecord:
    """One complete effective action/direction track safe for projection."""

    sequence_source_key: str
    resource_entity_external_key: str
    definition_logical_path: str
    definition_member_path: str
    definition_source_mod: str
    definition_sha256: str
    definition_size_bytes: int
    source_documents: tuple[str, ...]
    includes: tuple[IncludeDirective, ...]
    image_bindings: tuple[ImageBinding, ...]
    render_size: Point | None
    render_offset: Point
    blend_mode: str
    alpha_mod: int
    color_mod: tuple[int, int, int]
    entity_family: str
    identity: str
    body_variant: str | None
    attachment_id: str | None
    action_ordinal: int
    source_action: str
    adapter_normalized_action: str | None
    normalized_action_basis: str
    declared_frame_count: int
    duration_literal: str
    duration_milliseconds: int
    nominal_fps: float
    animation_type: str
    loop_mode: str
    position: int | None
    active_frames: tuple[int, ...] | str | None
    active_sub_frame: str | None
    action_declaration_location: SourceLocation
    layout_mode: str
    effective_raw_frame_record_count: int
    default_tick_schedule: TickSchedule
    direction: int
    direction_token: str
    direction_name: str
    frames: tuple[FlareProjectionFrame, ...]
    source_images: tuple[SourceImageAudit, ...]
    envelope_origin: Point
    envelope_width: int
    envelope_height: int
    entity_relations: tuple[FlareProjectionEntityRelation, ...]
    usage_relations: tuple[FlareProjectionUsageRelation, ...]
    attachment_candidates: tuple[FlareAttachmentCandidate, ...]
    hero_layer_order: HeroLayerOrder | None
    evidence_source_mods: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def source_blob_sha256(self) -> str | None:
        hashes = {image.sha256 for image in self.source_images}
        return next(iter(hashes)) if len(hashes) == 1 else None

    @property
    def explicit_frame_count(self) -> int:
        return sum(frame.explicit for frame in self.frames)

    @property
    def fallback_frame_count(self) -> int:
        return sum(frame.fallback_from_direction is not None for frame in self.frames)


@dataclass(frozen=True)
class FlareProjectionExclusion:
    """An action/direction track omitted without repairing source declarations."""

    sequence_source_key: str
    definition_logical_path: str
    definition_member_path: str | None
    definition_source_mod: str | None
    action_ordinal: int
    source_action: str
    action_declaration_location: SourceLocation
    direction: int
    direction_token: str
    direction_name: str
    declared_frame_count: int
    unresolved_slot_count: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FlareUsageExclusion:
    """A runtime usage whose animation path is absent from the active mod stack."""

    usage: AnimationUsage
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FlareProjectionPlan:
    """Pure deterministic projection plan; constructing it performs no writes."""

    archive_sha256: str
    repository_commit: str | None
    engine_semantics_commit: str
    active_mods: tuple[str, ...]
    archive_member_count: int
    archive_regular_file_count: int
    archive_expanded_member_bytes: int
    physical_action_declaration_count: int
    physical_explicit_frame_record_count: int
    effective_action_count: int
    effective_direction_track_count: int
    effective_frame_slot_count: int
    geometry_missing_action_count: int
    records: tuple[FlareProjectionRecord, ...]
    exclusions: tuple[FlareProjectionExclusion, ...]
    usage_exclusions: tuple[FlareUsageExclusion, ...]
    attachment_quarantines: tuple[FlareAttachmentQuarantine, ...]
    hero_layers: tuple[HeroLayerOrder, ...]
    rights_evidence: tuple[EvidenceDocument, ...]
    optional_mods_excluded: tuple[str, ...] = OPTIONAL_MODS_EXCLUDED

    @property
    def projected_definition_count(self) -> int:
        return len({record.definition_logical_path for record in self.records})

    @property
    def projected_action_count(self) -> int:
        return len(
            {(record.definition_logical_path, record.action_ordinal) for record in self.records}
        )

    @property
    def projected_sequence_count(self) -> int:
        return len(self.records)

    @property
    def projected_frame_count(self) -> int:
        return sum(record.frame_count for record in self.records)

    @property
    def projected_explicit_frame_count(self) -> int:
        return sum(record.explicit_frame_count for record in self.records)

    @property
    def projected_fallback_frame_count(self) -> int:
        return sum(record.fallback_frame_count for record in self.records)

    @property
    def projected_source_image_count(self) -> int:
        return len(
            {image.logical_path for record in self.records for image in record.source_images}
        )

    @property
    def projected_entity_binding_count(self) -> int:
        return len(
            {
                relation.definition_path
                for record in self.records
                for relation in record.entity_relations
            }
        )

    @property
    def projected_usage_count(self) -> int:
        return len(
            {
                (
                    relation.usage_kind,
                    relation.owner_id,
                    relation.location.member_path,
                    relation.location.line_number,
                )
                for record in self.records
                for relation in record.usage_relations
            }
        )

    @property
    def attachment_candidate_edge_count(self) -> int:
        return len(
            {
                (candidate.item_id, candidate.candidate_animation_path)
                for record in self.records
                for candidate in record.attachment_candidates
            }
        )

    @property
    def unresolved_attachment_layer_count(self) -> int:
        return sum(
            "layer_slot_absent_from_hero_orders" in item.reasons
            for item in self.attachment_quarantines
        )

    @property
    def excluded_direction_track_count(self) -> int:
        return len(self.exclusions)

    @property
    def excluded_unresolved_slot_count(self) -> int:
        return sum(item.unresolved_slot_count for item in self.exclusions)

    @property
    def required_source_images(self) -> tuple[SourceImageAudit, ...]:
        images: dict[str, SourceImageAudit] = {}
        for record in self.records:
            for image in record.source_images:
                previous = images.get(image.member_path)
                if previous is not None and previous.sha256 != image.sha256:
                    raise ValueError(
                        "One Flare image member has contradictory audited hashes: "
                        f"{image.member_path}"
                    )
                images[image.member_path] = image
        return tuple(images[path] for path in sorted(images))

    @property
    def required_member_paths(self) -> tuple[str, ...]:
        paths: set[str] = set()
        for image in self.required_source_images:
            paths.add(image.member_path)
        for record in self.records:
            paths.update(record.source_documents)
            paths.update(relation.member_path for relation in record.entity_relations)
            paths.update(
                relation.location.member_path
                for relation in record.usage_relations
                if relation.location.member_path is not None
            )
            paths.update(
                candidate.location.member_path
                for candidate in record.attachment_candidates
                if candidate.location.member_path is not None
            )
            if record.hero_layer_order and record.hero_layer_order.location.member_path:
                paths.add(record.hero_layer_order.location.member_path)
        for exclusion in self.usage_exclusions:
            if exclusion.usage.location.member_path:
                paths.add(exclusion.usage.location.member_path)
        paths.update(
            item.definition_member_path
            for item in self.exclusions
            if item.definition_member_path is not None
        )
        for quarantine in self.attachment_quarantines:
            if quarantine.location.member_path:
                paths.add(quarantine.location.member_path)
        paths.update(item.member_path for item in self.rights_evidence)
        return tuple(sorted(paths))

    @property
    def projected_subject_entity_count(self) -> int:
        keys = {record.resource_entity_external_key for record in self.records}
        keys.update(
            relation.external_entity_key
            for record in self.records
            for relation in record.entity_relations
        )
        keys.update(
            relation.external_entity_key
            for record in self.records
            for relation in record.usage_relations
        )
        keys.update(
            candidate.external_entity_key
            for record in self.records
            for candidate in record.attachment_candidates
        )
        return len(keys)

    @property
    def projection_manifest_sha256(self) -> str:
        payload = {
            "projection_version": PROJECTION_VERSION,
            "archive_sha256": self.archive_sha256,
            "repository_commit": self.repository_commit,
            "engine_semantics_commit": self.engine_semantics_commit,
            "active_mods": self.active_mods,
            "archive_counts": {
                "member_count": self.archive_member_count,
                "regular_file_count": self.archive_regular_file_count,
                "expanded_member_bytes": self.archive_expanded_member_bytes,
            },
            "physical_action_declaration_count": self.physical_action_declaration_count,
            "physical_explicit_frame_record_count": self.physical_explicit_frame_record_count,
            "effective_action_count": self.effective_action_count,
            "effective_direction_track_count": self.effective_direction_track_count,
            "effective_frame_slot_count": self.effective_frame_slot_count,
            "geometry_missing_action_count": self.geometry_missing_action_count,
            "records": [asdict(record) for record in self.records],
            "exclusions": [asdict(item) for item in self.exclusions],
            "usage_exclusions": [asdict(item) for item in self.usage_exclusions],
            "attachment_quarantines": [asdict(item) for item in self.attachment_quarantines],
            "hero_layers": [asdict(item) for item in self.hero_layers],
            "rights_evidence": [asdict(item) for item in self.rights_evidence],
            "optional_mods_excluded": self.optional_mods_excluded,
            "rights_scope_caveat": RIGHTS_SCOPE_CAVEAT,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class FlareProjectionReadiness:
    """Read-only report of exact indexed prerequisites."""

    database_path: str
    archive_sha256: str
    projection_manifest_sha256: str
    archive_blob_present: bool
    archive_inventory_present: bool
    archive_inventory_matches_audit: bool
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
            and self.archive_inventory_present
            and self.archive_inventory_matches_audit
            and self.source_item_count == 1
            and not self.missing_member_paths
            and not self.missing_source_image_blobs
            and not self.source_image_hash_mismatches
        )


@dataclass(frozen=True)
class FlareProjectionResult:
    """Core projection effects for one idempotent run."""

    archive_sha256: str
    projection_manifest_sha256: str
    projected_entities: int
    projected_definitions: int
    projected_actions: int
    projected_sequences: int
    projected_frames: int
    projected_explicit_frames: int
    projected_fallback_frames: int
    created_sequences: int
    reused_sequences: int
    occurrence_links: int
    excluded_direction_tracks: int
    excluded_unresolved_slots: int
    excluded_usages: int
    quarantined_attachments: int
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


def _identity_payload(audit: FlareArchiveAudit) -> dict[str, Any]:
    return {
        "archive_sha256": audit.archive_sha256,
        "repository_commit": audit.repository_commit,
        "engine_semantics_commit": audit.engine_semantics_commit,
        "active_mods": list(audit.active_mods),
    }


def _sequence_source_key(
    audit: FlareArchiveAudit,
    definition_path: str,
    action_ordinal: int,
    source_action: str,
    direction: int,
) -> str:
    return _stable_json_key(
        "flare-sequence-v1",
        {
            **_identity_payload(audit),
            "definition_logical_path": definition_path,
            "action_ordinal": action_ordinal,
            "source_action": source_action,
            "direction": direction,
        },
    )


def _resource_entity_key(audit: FlareArchiveAudit, definition_path: str) -> str:
    return _stable_json_key(
        "flare-animation-resource-v1",
        {**_identity_payload(audit), "definition_logical_path": definition_path},
    )


def _source_entity_key(audit: FlareArchiveAudit, definition_path: str) -> str:
    return _stable_json_key(
        "flare-source-entity-v1",
        {**_identity_payload(audit), "definition_path": definition_path},
    )


def _usage_entity_key(audit: FlareArchiveAudit, usage: AnimationUsage) -> str:
    if usage.usage_kind in {"enemy", "npc"} and usage.owner_id is not None:
        return _source_entity_key(audit, usage.owner_id)
    return _stable_json_key(
        "flare-animation-usage-v1",
        {
            **_identity_payload(audit),
            "usage_kind": usage.usage_kind,
            "owner_id": usage.owner_id,
        },
    )


def _attachment_entity_key(audit: FlareArchiveAudit, item_id: str) -> str:
    return _stable_json_key(
        "flare-attachment-item-v1",
        {**_identity_payload(audit), "item_id": item_id},
    )


def _entity_relations(
    audit: FlareArchiveAudit,
    definition_path: str,
    bindings: tuple[EntityBinding, ...],
) -> tuple[FlareProjectionEntityRelation, ...]:
    relations: list[FlareProjectionEntityRelation] = []
    for binding in bindings:
        for index, path in enumerate(binding.animation_paths):
            if path != definition_path:
                continue
            location = binding.animation_locations[min(index, len(binding.animation_locations) - 1)]
            relations.append(
                FlareProjectionEntityRelation(
                    external_entity_key=_source_entity_key(audit, binding.definition_path),
                    entity_kind=binding.entity_kind,
                    definition_path=binding.definition_path,
                    member_path=binding.member_path,
                    source_mod=binding.source_mod,
                    display_name=binding.display_name,
                    humanoid=binding.humanoid,
                    categories=binding.categories,
                    animation_location=location,
                    is_template=binding.is_template,
                    projected_entity_class=("humanoid" if binding.humanoid is True else "unknown"),
                )
            )
    return tuple(
        sorted(
            relations,
            key=lambda item: (
                item.definition_path,
                item.animation_location.line_number,
            ),
        )
    )


def _usage_class(usage_kind: str) -> str:
    if usage_kind == "item_loot":
        return "object"
    if usage_kind == "effect":
        return "effect"
    return "unknown"


def _usage_relations(
    audit: FlareArchiveAudit,
    definition_path: str,
    usages: tuple[AnimationUsage, ...],
) -> tuple[FlareProjectionUsageRelation, ...]:
    return tuple(
        FlareProjectionUsageRelation(
            external_entity_key=_usage_entity_key(audit, usage),
            usage_kind=usage.usage_kind,
            owner_id=usage.owner_id,
            owner_name=usage.owner_name,
            location=usage.location,
            projected_entity_class=_usage_class(usage.usage_kind),
        )
        for usage in sorted(
            (item for item in usages if item.animation_path == definition_path),
            key=lambda item: (
                item.usage_kind,
                item.owner_id or "",
                item.location.member_path or "",
                item.location.line_number,
            ),
        )
    )


def _attachment_candidates(
    audit: FlareArchiveAudit,
    definition_path: str,
    body_variant: str | None,
    direction: int,
    direction_token: str,
    direction_name: str,
    hero_layers: dict[int, HeroLayerOrder],
) -> tuple[FlareAttachmentCandidate, ...]:
    order = hero_layers.get(direction)
    candidates: list[FlareAttachmentCandidate] = []
    for attachment in audit.attachments:
        if definition_path not in attachment.candidate_animation_paths:
            continue
        layers = order.layers_back_to_front if order else ()
        layer_index = (
            layers.index(attachment.layer_slot) if attachment.layer_slot in layers else None
        )
        candidates.append(
            FlareAttachmentCandidate(
                external_entity_key=_attachment_entity_key(audit, attachment.item_id),
                item_id=attachment.item_id,
                item_name=attachment.item_name,
                layer_slot=attachment.layer_slot,
                gfx_id=attachment.gfx_id,
                candidate_animation_paths=attachment.candidate_animation_paths,
                candidate_animation_path=definition_path,
                candidate_body_variant=body_variant,
                body_variant_choice_resolved=(len(attachment.candidate_animation_paths) == 1),
                direction=direction,
                direction_token=direction_token,
                direction_name=direction_name,
                hero_layers_back_to_front=layers,
                layer_index_back_to_front=layer_index,
                layer_position_resolved=layer_index is not None,
                location=attachment.location,
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.item_id))


def _attachment_quarantines(audit: FlareArchiveAudit) -> tuple[FlareAttachmentQuarantine, ...]:
    quarantines: list[FlareAttachmentQuarantine] = []
    for attachment in audit.attachments:
        missing_directions = tuple(
            item.direction_name
            for item in audit.hero_layers
            if attachment.layer_slot not in item.layers_back_to_front
        )
        reasons: list[str] = []
        if len(attachment.candidate_animation_paths) != 1:
            reasons.append("body_variant_selection_is_ambiguous")
        if not attachment.candidate_animation_paths:
            reasons.append("no_archived_attachment_candidate")
        if missing_directions:
            reasons.append("layer_slot_absent_from_hero_orders")
        if reasons:
            quarantines.append(
                FlareAttachmentQuarantine(
                    item_id=attachment.item_id,
                    item_name=attachment.item_name,
                    layer_slot=attachment.layer_slot,
                    gfx_id=attachment.gfx_id,
                    candidate_animation_paths=attachment.candidate_animation_paths,
                    missing_layer_directions=missing_directions,
                    location=attachment.location,
                    reasons=tuple(reasons),
                )
            )
    return tuple(sorted(quarantines, key=lambda item: item.item_id))


def _evidence_source_mods(
    definition_source_mod: str,
    source_documents: tuple[str, ...],
    images: tuple[SourceImageAudit, ...],
    active_mods: tuple[str, ...],
) -> tuple[str, ...]:
    mods = {definition_source_mod, *(image.source_mod for image in images)}
    for member_path in source_documents:
        for mod in active_mods:
            if f"/mods/{mod}/" in member_path:
                mods.add(mod)
    return tuple(mod for mod in active_mods if mod in mods)


def plan_flare_projection(audit: FlareArchiveAudit) -> FlareProjectionPlan:
    """Build a deterministic, write-free plan from a Flare archive audit."""

    images = {item.logical_path: item for item in audit.source_images}
    hero_layers = {item.direction: item for item in audit.hero_layers}
    definitions = {item.logical_path for item in audit.definitions}
    records: list[FlareProjectionRecord] = []
    exclusions: list[FlareProjectionExclusion] = []

    for definition in audit.definitions:
        for action_ordinal, action in enumerate(definition.actions):
            for track in action.direction_tracks:
                source_key = _sequence_source_key(
                    audit,
                    definition.logical_path,
                    action_ordinal,
                    action.source_action,
                    track.direction,
                )
                reasons: list[str] = []
                if definition.source_mod not in audit.active_mods:
                    reasons.append("definition_source_mod_is_not_active")
                if not action.has_exact_geometry:
                    reasons.append("action_missing_exact_geometry")
                if not track.complete:
                    reasons.append("direction_track_has_unresolved_slots")
                unresolved = sum(slot.frame is None for slot in track.frames)
                selected_images: dict[str, SourceImageAudit] = {}
                for slot in track.frames:
                    frame = slot.frame
                    if frame is None:
                        continue
                    if frame.image_path is None:
                        reasons.append("frame_has_no_source_image")
                        continue
                    image = images.get(frame.image_path)
                    if image is None:
                        reasons.append("frame_source_image_is_missing")
                        continue
                    selected_images[image.logical_path] = image
                    rectangle = frame.rectangle
                    if (
                        rectangle.x < 0
                        or rectangle.y < 0
                        or rectangle.width <= 0
                        or rectangle.height <= 0
                        or rectangle.right > image.width
                        or rectangle.bottom > image.height
                        or frame.within_image_bounds is not True
                    ):
                        reasons.append("frame_rectangle_is_out_of_bounds")
                reasons = list(dict.fromkeys(reasons))
                if reasons:
                    exclusions.append(
                        FlareProjectionExclusion(
                            sequence_source_key=source_key,
                            definition_logical_path=definition.logical_path,
                            definition_member_path=definition.member_path,
                            definition_source_mod=definition.source_mod,
                            action_ordinal=action_ordinal,
                            source_action=action.source_action,
                            action_declaration_location=action.section_location,
                            direction=track.direction,
                            direction_token=track.direction_token,
                            direction_name=track.direction_name,
                            declared_frame_count=action.declared_frame_count,
                            unresolved_slot_count=unresolved,
                            reasons=tuple(reasons),
                        )
                    )
                    continue

                if (
                    definition.member_path is None
                    or definition.source_mod is None
                    or definition.sha256 is None
                    or definition.size_bytes is None
                ):
                    raise ValueError(
                        "Admitted Flare archive definition lacks physical evidence: "
                        f"{definition.logical_path}"
                    )
                source_images = tuple(selected_images[path] for path in sorted(selected_images))
                projected_frames: list[FlareProjectionFrame] = []
                for ordinal, slot in enumerate(track.frames):
                    frame = slot.frame
                    if frame is None or frame.image_path is None:
                        raise AssertionError("admitted Flare track has an unresolved frame")
                    image = images[frame.image_path]
                    tick_count = action.default_tick_schedule.per_frame_tick_counts[slot.index]
                    projected_frames.append(
                        FlareProjectionFrame(
                            ordinal=ordinal,
                            source_frame_index=slot.index,
                            effective_direction=track.direction,
                            effective_direction_token=track.direction_token,
                            effective_direction_name=track.direction_name,
                            authored_direction=frame.direction,
                            authored_direction_name=frame.direction_name,
                            explicit=slot.explicit,
                            fallback_from_direction=slot.fallback_from_direction,
                            rectangle=frame.rectangle,
                            offset=frame.offset,
                            image_id=frame.image_id,
                            image_logical_path=image.logical_path,
                            image_member_path=image.member_path,
                            image_sha256=image.sha256,
                            image_width=image.width,
                            image_height=image.height,
                            image_mode=image.image_mode,
                            image_format=image.image_format,
                            image_has_transparency=image.has_transparency,
                            source_location=frame.location,
                            default_60hz_tick_count=tick_count,
                            default_60hz_duration_ms=(tick_count * 1000 / FLARE_DEFAULT_ENGINE_FPS),
                        )
                    )
                left = min(item.offset.x for item in projected_frames)
                top = min(item.offset.y for item in projected_frames)
                right = max(item.offset.x + item.rectangle.width for item in projected_frames)
                bottom = max(item.offset.y + item.rectangle.height for item in projected_frames)
                entity_relations = _entity_relations(audit, definition.logical_path, audit.entities)
                usage_relations = _usage_relations(audit, definition.logical_path, audit.usages)
                attachment_candidates = _attachment_candidates(
                    audit,
                    definition.logical_path,
                    definition.body_variant,
                    track.direction,
                    track.direction_token,
                    track.direction_name,
                    hero_layers,
                )
                records.append(
                    FlareProjectionRecord(
                        sequence_source_key=source_key,
                        resource_entity_external_key=_resource_entity_key(
                            audit, definition.logical_path
                        ),
                        definition_logical_path=definition.logical_path,
                        definition_member_path=definition.member_path,
                        definition_source_mod=definition.source_mod,
                        definition_sha256=definition.sha256,
                        definition_size_bytes=definition.size_bytes,
                        source_documents=definition.source_documents,
                        includes=definition.includes,
                        image_bindings=definition.image_bindings,
                        render_size=definition.render_size,
                        render_offset=definition.render_offset,
                        blend_mode=definition.blend_mode,
                        alpha_mod=definition.alpha_mod,
                        color_mod=definition.color_mod,
                        entity_family=definition.entity_family,
                        identity=definition.identity,
                        body_variant=definition.body_variant,
                        attachment_id=definition.attachment_id,
                        action_ordinal=action_ordinal,
                        source_action=action.source_action,
                        adapter_normalized_action=action.normalized_action,
                        normalized_action_basis=action.normalized_action_basis,
                        declared_frame_count=action.declared_frame_count,
                        duration_literal=action.duration_literal,
                        duration_milliseconds=action.duration_milliseconds,
                        nominal_fps=action.nominal_fps,
                        animation_type=action.animation_type,
                        loop_mode=action.loop_mode,
                        position=action.position,
                        active_frames=action.active_frames,
                        active_sub_frame=action.active_sub_frame,
                        action_declaration_location=action.section_location,
                        layout_mode=action.layout_mode,
                        effective_raw_frame_record_count=len(action.raw_frames),
                        default_tick_schedule=action.default_tick_schedule,
                        direction=track.direction,
                        direction_token=track.direction_token,
                        direction_name=track.direction_name,
                        frames=tuple(projected_frames),
                        source_images=source_images,
                        envelope_origin=Point(left, top),
                        envelope_width=right - left,
                        envelope_height=bottom - top,
                        entity_relations=entity_relations,
                        usage_relations=usage_relations,
                        attachment_candidates=attachment_candidates,
                        hero_layer_order=(
                            hero_layers.get(track.direction)
                            if definition.entity_family == "avatar_attachment"
                            else None
                        ),
                        evidence_source_mods=_evidence_source_mods(
                            definition.source_mod,
                            definition.source_documents,
                            source_images,
                            audit.active_mods,
                        ),
                    )
                )

    records.sort(key=lambda item: item.sequence_source_key)
    exclusions.sort(key=lambda item: item.sequence_source_key)
    usage_exclusions = tuple(
        FlareUsageExclusion(
            usage=usage,
            reasons=("animation_path_absent_from_active_mod_stack",),
        )
        for usage in sorted(
            (item for item in audit.usages if item.animation_path not in definitions),
            key=lambda item: (
                item.animation_path,
                item.usage_kind,
                item.owner_id or "",
            ),
        )
    )
    counts = audit.counts
    return FlareProjectionPlan(
        archive_sha256=audit.archive_sha256,
        repository_commit=audit.repository_commit,
        engine_semantics_commit=audit.engine_semantics_commit,
        active_mods=audit.active_mods,
        archive_member_count=counts.zip_member_count,
        archive_regular_file_count=counts.regular_file_member_count,
        archive_expanded_member_bytes=counts.expanded_member_bytes,
        physical_action_declaration_count=counts.physical_action_declaration_count,
        physical_explicit_frame_record_count=counts.physical_explicit_frame_record_count,
        effective_action_count=counts.action_count,
        effective_direction_track_count=counts.direction_track_count,
        effective_frame_slot_count=counts.effective_frame_slot_count,
        geometry_missing_action_count=counts.geometry_missing_action_count,
        records=tuple(records),
        exclusions=tuple(exclusions),
        usage_exclusions=usage_exclusions,
        attachment_quarantines=_attachment_quarantines(audit),
        hero_layers=audit.hero_layers,
        rights_evidence=audit.evidence_documents,
    )


def plan_known_flare_projection(archive_path: str | Path) -> FlareProjectionPlan:
    """Audit the exact pinned CAS archive and build a write-free plan."""

    plan = plan_flare_projection(audit_known_flare_archive(archive_path))
    if (
        plan.archive_sha256 != EXPECTED_FLARE_ARCHIVE_SHA256
        or plan.repository_commit != FLARE_GAME_COMMIT
        or plan.engine_semantics_commit != FLARE_ENGINE_COMMIT
        or plan.active_mods != FLARE_ACTIVE_MODS
    ):
        raise ValueError("Refusing a Flare plan whose complete snapshot pin does not match")
    return plan


def _readonly_connection(database_path: str | Path) -> sqlite3.Connection:
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def check_flare_projection_readiness(
    database_path: str | Path,
    plan: FlareProjectionPlan,
) -> FlareProjectionReadiness:
    """Check exact prerequisites through a query-only SQLite connection."""

    required_paths = plan.required_member_paths
    required_images = {item.member_path: item for item in plan.required_source_images}
    with _readonly_connection(database_path) as connection:
        archive_blob_present = (
            connection.execute(
                "SELECT 1 FROM blobs WHERE sha256=? LIMIT 1",
                (plan.archive_sha256,),
            ).fetchone()
            is not None
        )
        inventory = connection.execute(
            """
            SELECT member_count, file_count, total_uncompressed_bytes
            FROM archive_inventories WHERE archive_blob_sha256=?
            """,
            (plan.archive_sha256,),
        ).fetchone()
        source_item_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM items i
                JOIN item_blobs ib ON ib.item_id = i.id
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
            LEFT JOIN blobs b ON b.sha256 = am.extracted_blob_sha256
            WHERE am.archive_blob_sha256=?
            """,
            (plan.archive_sha256,),
        ).fetchall()

    inventory_present = inventory is not None
    inventory_matches = bool(
        inventory is not None
        and int(inventory["member_count"]) == plan.archive_member_count
        and int(inventory["file_count"]) == plan.archive_regular_file_count
        and int(inventory["total_uncompressed_bytes"]) == plan.archive_expanded_member_bytes
    )
    members: dict[str, sqlite3.Row] = {}
    for row in rows:
        members[str(row["normalized_path"])] = row
        members[str(row["member_path"])] = row
    missing_paths = tuple(path for path in required_paths if path not in members)
    missing_image_blobs: list[str] = []
    hash_mismatches: list[str] = []
    present_image_blobs = 0
    for member_path, image in sorted(required_images.items()):
        row = members.get(member_path)
        if row is None:
            continue
        actual = row["extracted_blob_sha256"]
        registered = row["registered_blob_sha256"]
        if actual is None or registered is None:
            missing_image_blobs.append(member_path)
            continue
        present_image_blobs += 1
        if str(actual) != image.sha256:
            hash_mismatches.append(f"{member_path}: expected {image.sha256}, indexed {actual}")
    return FlareProjectionReadiness(
        database_path=str(Path(database_path).resolve()),
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=plan.projection_manifest_sha256,
        archive_blob_present=archive_blob_present,
        archive_inventory_present=inventory_present,
        archive_inventory_matches_audit=inventory_matches,
        source_item_count=source_item_count,
        required_member_count=len(required_paths),
        present_member_count=len(required_paths) - len(missing_paths),
        required_source_image_count=len(required_images),
        present_source_image_blob_count=present_image_blobs,
        missing_member_paths=missing_paths,
        missing_source_image_blobs=tuple(missing_image_blobs),
        source_image_hash_mismatches=tuple(hash_mismatches),
    )


def _archive_members(database: IndexDB, archive_sha256: str) -> dict[str, sqlite3.Row]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT ordinal, member_path, normalized_path, extracted_blob_sha256
            FROM archive_members
            WHERE archive_blob_sha256=?
            ORDER BY ordinal
            """,
            (archive_sha256,),
        ).fetchall()
    members: dict[str, sqlite3.Row] = {}
    for row in rows:
        members[str(row["normalized_path"])] = row
        members[str(row["member_path"])] = row
    return members


def _preflight(
    database: IndexDB,
    plan: FlareProjectionPlan,
) -> tuple[str, dict[str, sqlite3.Row]]:
    readiness = check_flare_projection_readiness(database.path, plan)
    if not readiness.ready:
        problems: list[str] = []
        if not readiness.archive_blob_present:
            problems.append("archive blob is not registered")
        if not readiness.archive_inventory_present:
            problems.append("archive inventory is missing")
        elif not readiness.archive_inventory_matches_audit:
            problems.append("archive inventory counts do not match the audit")
        if readiness.source_item_count != 1:
            problems.append(
                f"expected one {SOURCE_ID!r} source item, found {readiness.source_item_count}"
            )
        if readiness.missing_member_paths:
            problems.append(
                "missing evidence members: " + ", ".join(readiness.missing_member_paths[:5])
            )
        if readiness.missing_source_image_blobs:
            problems.append(
                "source PNGs are not extracted/registered: "
                + ", ".join(readiness.missing_source_image_blobs[:5])
            )
        if readiness.source_image_hash_mismatches:
            problems.append(
                "source PNG hashes differ: " + ", ".join(readiness.source_image_hash_mismatches[:3])
            )
        raise ValueError("Flare projection prerequisites are not ready: " + "; ".join(problems))
    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT i.id
            FROM items i
            JOIN item_blobs ib ON ib.item_id=i.id
            WHERE i.source_id=? AND ib.blob_sha256=?
            ORDER BY i.id
            """,
            (SOURCE_ID, plan.archive_sha256),
        ).fetchone()
    if row is None:
        raise AssertionError("ready Flare projection has no source item")
    return str(row["id"]), _archive_members(database, plan.archive_sha256)


def _applicable_rights_evidence(
    plan: FlareProjectionPlan,
    record: FlareProjectionRecord,
) -> tuple[EvidenceDocument, ...]:
    result: list[EvidenceDocument] = []
    for evidence in plan.rights_evidence:
        if not evidence.relative_path.startswith("mods/"):
            result.append(evidence)
            continue
        if any(
            evidence.relative_path.startswith(f"mods/{mod}/") for mod in record.evidence_source_mods
        ):
            result.append(evidence)
    return tuple(result)


def _rights_scope_metadata(
    plan: FlareProjectionPlan,
    record: FlareProjectionRecord,
) -> dict[str, Any]:
    return {
        "scope": "repository_and_relevant_mod_claims_not_asset_level",
        "caveat": RIGHTS_SCOPE_CAVEAT,
        "evidence_source_mods": list(record.evidence_source_mods),
        "evidence": [
            {
                "relative_path": item.relative_path,
                "member_path": item.member_path,
                "sha256": item.sha256,
                "detected_license_identifiers": list(item.detected_license_identifiers),
                "scope": item.scope,
                "notes": item.notes,
            }
            for item in _applicable_rights_evidence(plan, record)
        ],
        "per_asset_manifest_present": False,
        "asset_license_expression": None,
        "asset_creator": None,
        "rights_observation_added": False,
    }


def _frame_phase(record: FlareProjectionRecord, ordinal: int) -> float | None:
    if record.animation_type == "back_forth":
        return None
    if record.frame_count <= 1:
        return 0.0
    if record.animation_type == "looped":
        return ordinal / record.frame_count
    return ordinal / (record.frame_count - 1)


def _sequence_metadata(
    plan: FlareProjectionPlan,
    record: FlareProjectionRecord,
    projection_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": projection_manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "engine_semantics_commit": plan.engine_semantics_commit,
        "active_mods": list(plan.active_mods),
        "optional_mods_excluded": list(plan.optional_mods_excluded),
        "definition_logical_path": record.definition_logical_path,
        "definition_member_path": record.definition_member_path,
        "definition_source_mod": record.definition_source_mod,
        "definition_sha256": record.definition_sha256,
        "definition_size_bytes": record.definition_size_bytes,
        "physical_definition": {
            "member_path": record.definition_member_path,
            "sha256": record.definition_sha256,
            "size_bytes": record.definition_size_bytes,
        },
        "effective_include_expansion": {
            "source_documents_in_engine_order": list(record.source_documents),
            "include_directives": [asdict(item) for item in record.includes],
            "image_bindings_in_engine_order": [asdict(item) for item in record.image_bindings],
            "action_declaration_location": asdict(record.action_declaration_location),
            "effective_raw_frame_record_count": (record.effective_raw_frame_record_count),
        },
        "render_size": asdict(record.render_size) if record.render_size else None,
        "render_offset": asdict(record.render_offset),
        "blend_mode": record.blend_mode,
        "alpha_mod": record.alpha_mod,
        "color_mod": list(record.color_mod),
        "entity_family": record.entity_family,
        "identity": record.identity,
        "body_variant": record.body_variant,
        "attachment_id": record.attachment_id,
        "action_ordinal": record.action_ordinal,
        "source_action": record.source_action,
        "adapter_normalized_action": record.adapter_normalized_action,
        "normalized_action_basis": record.normalized_action_basis,
        "declared_frame_count": record.declared_frame_count,
        "duration_literal": record.duration_literal,
        "duration_milliseconds": record.duration_milliseconds,
        "nominal_fps": record.nominal_fps,
        "animation_type": record.animation_type,
        "loop_mode": record.loop_mode,
        "position": record.position,
        "active_frames": (
            list(record.active_frames)
            if isinstance(record.active_frames, tuple)
            else record.active_frames
        ),
        "active_sub_frame": record.active_sub_frame,
        "layout_mode": record.layout_mode,
        "direction": {
            "index": record.direction,
            "token": record.direction_token,
            "name": record.direction_name,
        },
        "default_60hz_tick_schedule": asdict(record.default_tick_schedule),
        "runtime_fps_configurable": True,
        "source_images": [asdict(item) for item in record.source_images],
        "anchor_relative_union_envelope": {
            "origin": asdict(record.envelope_origin),
            "width": record.envelope_width,
            "height": record.envelope_height,
            "derived_summary_only": True,
            "uniform_source_frame_size_claim": False,
        },
        "entity_relations": [asdict(item) for item in record.entity_relations],
        "usage_relations": [asdict(item) for item in record.usage_relations],
        "attachment_candidates": [asdict(item) for item in record.attachment_candidates],
        "hero_layer_order": (asdict(record.hero_layer_order) if record.hero_layer_order else None),
        "geometry_coordinate_space": "source_png",
        "exact_rectangles_authoritative": True,
        "frame_order_preserved": True,
        "direction_zero_fallback_preserved_not_reauthored": True,
        "default_engine_timing_exact_for_60hz": True,
        "individual_frame_pixels_materialized": False,
        "clipping_grid_inference_or_repair_applied": False,
        "schema_limitations": {
            "sequence_width_height": (
                "Required scalar dimensions store only the anchor-relative union "
                "envelope; exact varying crop rectangles and offsets are frame metadata."
            ),
            "runtime_timing": (
                "Frame duration stores the paired engine's default 60 Hz schedule; "
                "engine FPS and animation-speed effects are runtime configurable."
            ),
            "layer_composition": (
                "Attachment edges are candidates; no body variant, equipment state, "
                "or unresolved layer position is selected."
            ),
        },
        "rights_scope": _rights_scope_metadata(plan, record),
    }


def _occurrence_specs(
    plan: FlareProjectionPlan,
    record: FlareProjectionRecord,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    common = {
        "projection_version": PROJECTION_VERSION,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "definition_logical_path": record.definition_logical_path,
        "source_action": record.source_action,
        "direction": record.direction_name,
    }
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    def add(member_path: str | None, role: str, evidence: dict[str, Any]) -> None:
        if member_path is not None:
            grouped[(member_path, role)].append(evidence)

    for image in record.source_images:
        add(
            image.member_path,
            "flare_source_png",
            {
                "logical_path": image.logical_path,
                "sha256": image.sha256,
                "dimensions": [image.width, image.height],
                "source_mod": image.source_mod,
            },
        )
    for index, member_path in enumerate(record.source_documents):
        add(
            member_path,
            "flare_effective_animation_source_document",
            {
                "engine_include_order": index,
                "is_physical_definition_root": (member_path == record.definition_member_path),
            },
        )
    add(
        record.action_declaration_location.member_path,
        "flare_effective_action_declaration",
        {
            "logical_path": record.action_declaration_location.logical_path,
            "line_number": record.action_declaration_location.line_number,
            "action_ordinal": record.action_ordinal,
        },
    )
    for relation in record.entity_relations:
        add(
            relation.member_path,
            "flare_entity_animation_binding",
            asdict(relation),
        )
    for relation in record.usage_relations:
        add(
            relation.location.member_path,
            "flare_animation_usage",
            asdict(relation),
        )
    for candidate in record.attachment_candidates:
        add(
            candidate.location.member_path,
            "flare_avatar_attachment_candidate",
            asdict(candidate),
        )
    if record.hero_layer_order:
        add(
            record.hero_layer_order.location.member_path,
            "flare_hero_layer_order",
            asdict(record.hero_layer_order),
        )
    for evidence in _applicable_rights_evidence(plan, record):
        add(
            evidence.member_path,
            "flare_scoped_rights_evidence",
            {
                **asdict(evidence),
                "asset_level_claim": False,
                "rights_scope_caveat": RIGHTS_SCOPE_CAVEAT,
            },
        )
    return tuple(
        (
            member_path,
            role,
            {**common, "evidence": grouped[(member_path, role)]},
        )
        for member_path, role in sorted(grouped)
    )


def _resource_entity_metadata(
    plan: FlareProjectionPlan,
    record: FlareProjectionRecord,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "engine_semantics_commit": plan.engine_semantics_commit,
        "identity_kind": "physical_animation_definition_resource",
        "semantic_identity_claim": False,
        "definition_logical_path": record.definition_logical_path,
        "definition_member_path": record.definition_member_path,
        "definition_sha256": record.definition_sha256,
        "definition_source_mod": record.definition_source_mod,
        "entity_family": record.entity_family,
        "identity": record.identity,
        "body_variant": record.body_variant,
        "attachment_id": record.attachment_id,
        "source_documents": list(record.source_documents),
        "rights_scope": _rights_scope_metadata(plan, record),
    }


def _binding_entity_metadata(
    plan: FlareProjectionPlan,
    relations: tuple[FlareProjectionEntityRelation, ...],
    manifest_sha256: str,
) -> dict[str, Any]:
    first = relations[0]
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "identity_kind": "source_enemy_or_npc_definition",
        "source_entity_kind": first.entity_kind,
        "definition_path": first.definition_path,
        "member_path": first.member_path,
        "source_mod": first.source_mod,
        "humanoid": first.humanoid,
        "categories": list(first.categories),
        "is_template": first.is_template,
        "animation_bindings": [asdict(item) for item in relations],
        "classification_caveat": (
            "Only explicit humanoid=true is mapped to humanoid; gameplay enemy/NPC "
            "role and raw categories are not silently treated as morphology."
        ),
    }


def _usage_entity_metadata(
    plan: FlareProjectionPlan,
    relations: tuple[FlareProjectionUsageRelation, ...],
    manifest_sha256: str,
) -> dict[str, Any]:
    first = relations[0]
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "identity_kind": "source_animation_usage_owner",
        "usage_kind": first.usage_kind,
        "owner_id": first.owner_id,
        "owner_name": first.owner_name,
        "usage_occurrences": [asdict(item) for item in relations],
    }


def _attachment_entity_metadata(
    plan: FlareProjectionPlan,
    candidates: tuple[FlareAttachmentCandidate, ...],
    manifest_sha256: str,
) -> dict[str, Any]:
    first = candidates[0]
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "identity_kind": "source_equipment_item_attachment_candidate",
        "item_id": first.item_id,
        "item_name": first.item_name,
        "layer_slot": first.layer_slot,
        "gfx_id": first.gfx_id,
        "candidate_animation_paths": list(first.candidate_animation_paths),
        "candidate_edges": [asdict(item) for item in candidates],
        "resolved_body_variant": None,
        "candidate_edge_is_runtime_selection": False,
    }


def project_flare_audit(
    database: IndexDB,
    plan: FlareProjectionPlan,
    taxonomy: Taxonomy,
) -> FlareProjectionResult:
    """Idempotently project a precomputed safe Flare plan into core DB tables.

    Required archive/source-image rows must already exist.  Exact rectangles stay
    in ``sequence_frames`` metadata; no crops or ``rights_observations`` are made.
    """

    database.initialize()
    item_id, members = _preflight(database, plan)
    manifest_sha256 = plan.projection_manifest_sha256
    entity_ids: dict[str, str] = {}
    created_sequences = 0
    reused_sequences = 0
    occurrence_links = 0

    binding_groups: defaultdict[str, dict[FlareProjectionEntityRelation, None]] = defaultdict(dict)
    usage_groups: defaultdict[str, dict[FlareProjectionUsageRelation, None]] = defaultdict(dict)
    attachment_groups: defaultdict[str, dict[FlareAttachmentCandidate, None]] = defaultdict(dict)
    for record in plan.records:
        for relation in record.entity_relations:
            binding_groups[relation.external_entity_key][relation] = None
        for relation in record.usage_relations:
            usage_groups[relation.external_entity_key][relation] = None
        for candidate in record.attachment_candidates:
            attachment_groups[candidate.external_entity_key][candidate] = None

    for record in plan.records:
        resource_id = entity_ids.get(record.resource_entity_external_key)
        if resource_id is None:
            resource_id = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=record.resource_entity_external_key,
                representative_item_id=item_id,
                display_name=record.identity,
                entity_class="unknown",
                entity_subclass=record.entity_family,
                species_or_type=None,
                taxonomy_version=taxonomy.version,
                metadata=_resource_entity_metadata(plan, record, manifest_sha256),
            )
            entity_ids[record.resource_entity_external_key] = resource_id

        for relation in record.entity_relations:
            if relation.external_entity_key in entity_ids:
                continue
            group = tuple(binding_groups[relation.external_entity_key])
            normalized = taxonomy.normalize_entity_class(relation.projected_entity_class)
            entity_ids[relation.external_entity_key] = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=relation.external_entity_key,
                representative_item_id=item_id,
                display_name=relation.display_name or relation.definition_path,
                entity_class=normalized.value,
                entity_subclass=relation.entity_kind,
                species_or_type=None,
                taxonomy_version=taxonomy.version,
                metadata=_binding_entity_metadata(plan, group, manifest_sha256),
            )
        for relation in record.usage_relations:
            if relation.external_entity_key in entity_ids:
                continue
            group = tuple(usage_groups[relation.external_entity_key])
            normalized = taxonomy.normalize_entity_class(relation.projected_entity_class)
            entity_ids[relation.external_entity_key] = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=relation.external_entity_key,
                representative_item_id=item_id,
                display_name=relation.owner_name or relation.owner_id or relation.usage_kind,
                entity_class=normalized.value,
                entity_subclass=relation.usage_kind,
                species_or_type=None,
                taxonomy_version=taxonomy.version,
                metadata=_usage_entity_metadata(plan, group, manifest_sha256),
            )
        for candidate in record.attachment_candidates:
            if candidate.external_entity_key in entity_ids:
                continue
            group = tuple(attachment_groups[candidate.external_entity_key])
            entity_ids[candidate.external_entity_key] = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=candidate.external_entity_key,
                representative_item_id=item_id,
                display_name=candidate.item_name or candidate.item_id,
                entity_class="object",
                entity_subclass="avatar_attachment_item",
                species_or_type=None,
                taxonomy_version=taxonomy.version,
                metadata=_attachment_entity_metadata(plan, group, manifest_sha256),
            )

        motion = taxonomy.motion_condition(
            action=record.adapter_normalized_action,
            direction=record.direction_token,
            view=None,
        )
        sequence_id = database.find_sequence_by_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
        )
        sequence_arguments = {
            "source_blob_sha256": record.source_blob_sha256,
            "extraction_method": PROJECTION_VERSION,
            "extraction_confidence": 1.0,
            "width": record.envelope_width,
            "height": record.envelope_height,
            "frame_count": record.frame_count,
            "loop_mode": record.loop_mode,
            "action": motion.normalized_action,
            "direction": motion.direction,
            "quality_tier": QUALITY_TIER,
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
            entity_id=resource_id,
            role="primary",
            metadata={
                "identity_kind": "physical_animation_definition_resource",
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
        usage_by_subject: defaultdict[tuple[str, str], list[FlareProjectionUsageRelation]] = (
            defaultdict(list)
        )
        for relation in record.usage_relations:
            usage_by_subject[(relation.external_entity_key, relation.usage_kind)].append(relation)
        for (external_key, usage_kind), relations in sorted(usage_by_subject.items()):
            database.link_sequence_subject(
                sequence_id=sequence_id,
                entity_id=entity_ids[external_key],
                role=f"animation_usage_{usage_kind}",
                metadata={"usage_occurrences": [asdict(item) for item in relations]},
            )
        attachment_by_subject: defaultdict[str, list[FlareAttachmentCandidate]] = defaultdict(list)
        for candidate in record.attachment_candidates:
            attachment_by_subject[candidate.external_entity_key].append(candidate)
        for external_key, candidates in sorted(attachment_by_subject.items()):
            database.link_sequence_subject(
                sequence_id=sequence_id,
                entity_id=entity_ids[external_key],
                role="avatar_attachment_candidate",
                metadata={
                    "candidate_edges": [asdict(item) for item in candidates],
                    "runtime_selection_resolved": False,
                },
            )
        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version=taxonomy.version,
            source_action=record.source_action,
            normalized_action=motion.normalized_action,
            action_family=motion.action_family,
            annotation_method=PROJECTION_VERSION,
            view=motion.view,
            direction=motion.direction,
            loopable=record.animation_type != "play_once",
            cycle_frames=(
                record.declared_frame_count if record.animation_type == "looped" else None
            ),
            phase_zero_frame=(0 if record.animation_type == "looped" else None),
            confidence=(motion.confidence if record.adapter_normalized_action is not None else 0.0),
            conditioning={
                "source_action": record.source_action,
                "adapter_normalized_action": record.adapter_normalized_action,
                "normalized_action_basis": record.normalized_action_basis,
                "source_direction": {
                    "index": record.direction,
                    "token": record.direction_token,
                    "name": record.direction_name,
                },
                "view_evidence": "absent",
                "animation_type": record.animation_type,
                "loop_mode": record.loop_mode,
                "duration_literal": record.duration_literal,
                "duration_milliseconds": record.duration_milliseconds,
                "nominal_fps": record.nominal_fps,
                "default_60hz_tick_schedule": asdict(record.default_tick_schedule),
                "runtime_fps_configurable": True,
                "active_frames": (
                    list(record.active_frames)
                    if isinstance(record.active_frames, tuple)
                    else record.active_frames
                ),
                "active_sub_frame": record.active_sub_frame,
                "ping_pong_cycle_fields_withheld": (record.animation_type == "back_forth"),
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
            database.add_sequence_frame(
                sequence_id=sequence_id,
                ordinal=frame.ordinal,
                source_blob_sha256=frame.image_sha256,
                source_frame_index=frame.source_frame_index,
                duration_ms=frame.default_60hz_duration_ms,
                phase=_frame_phase(record, frame.ordinal),
                direction=motion.direction,
                view=motion.view,
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "source_frame_index_semantics": "flare_action_frame_index",
                    "effective_direction": {
                        "index": frame.effective_direction,
                        "token": frame.effective_direction_token,
                        "name": frame.effective_direction_name,
                    },
                    "authored_direction": {
                        "index": frame.authored_direction,
                        "name": frame.authored_direction_name,
                    },
                    "explicit": frame.explicit,
                    "fallback_from_direction": frame.fallback_from_direction,
                    "fallback_is_not_source_authored_for_effective_direction": (
                        frame.fallback_from_direction is not None
                    ),
                    "rectangle": asdict(frame.rectangle),
                    "offset": asdict(frame.offset),
                    "coordinate_space": "source_png",
                    "image_id": frame.image_id,
                    "image_logical_path": frame.image_logical_path,
                    "image_member_path": frame.image_member_path,
                    "image_sha256": frame.image_sha256,
                    "image_dimensions": [frame.image_width, frame.image_height],
                    "source_location": asdict(frame.source_location),
                    "default_60hz_tick_count": frame.default_60hz_tick_count,
                    "default_60hz_duration_ms": frame.default_60hz_duration_ms,
                    "runtime_fps_configurable": True,
                    "clipping_grid_inference_or_repair_applied": False,
                    "rights_scope": _rights_scope_metadata(plan, record),
                },
            )

    return FlareProjectionResult(
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=manifest_sha256,
        projected_entities=plan.projected_subject_entity_count,
        projected_definitions=plan.projected_definition_count,
        projected_actions=plan.projected_action_count,
        projected_sequences=plan.projected_sequence_count,
        projected_frames=plan.projected_frame_count,
        projected_explicit_frames=plan.projected_explicit_frame_count,
        projected_fallback_frames=plan.projected_fallback_frame_count,
        created_sequences=created_sequences,
        reused_sequences=reused_sequences,
        occurrence_links=occurrence_links,
        excluded_direction_tracks=plan.excluded_direction_track_count,
        excluded_unresolved_slots=plan.excluded_unresolved_slot_count,
        excluded_usages=len(plan.usage_exclusions),
        quarantined_attachments=len(plan.attachment_quarantines),
    )


def ingest_known_flare_sequences(
    database: IndexDB,
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> FlareProjectionResult:
    """Audit and project only the archive matching every Flare snapshot pin."""

    plan = plan_known_flare_projection(archive_path)
    return project_flare_audit(database, plan, taxonomy)


plan_flare_empyrean_projection = plan_flare_projection
plan_known_flare_empyrean_projection = plan_known_flare_projection
check_flare_empyrean_projection_readiness = check_flare_projection_readiness
project_flare_empyrean_audit = project_flare_audit


__all__ = [
    "OPTIONAL_MODS_EXCLUDED",
    "PROJECTION_VERSION",
    "QUALITY_TIER",
    "RIGHTS_SCOPE_CAVEAT",
    "SOURCE_ID",
    "FlareAttachmentCandidate",
    "FlareAttachmentQuarantine",
    "FlareProjectionEntityRelation",
    "FlareProjectionExclusion",
    "FlareProjectionFrame",
    "FlareProjectionPlan",
    "FlareProjectionReadiness",
    "FlareProjectionRecord",
    "FlareProjectionResult",
    "FlareProjectionUsageRelation",
    "FlareUsageExclusion",
    "check_flare_empyrean_projection_readiness",
    "check_flare_projection_readiness",
    "ingest_known_flare_sequences",
    "plan_flare_empyrean_projection",
    "plan_flare_projection",
    "plan_known_flare_empyrean_projection",
    "plan_known_flare_projection",
    "project_flare_audit",
    "project_flare_empyrean_audit",
]
