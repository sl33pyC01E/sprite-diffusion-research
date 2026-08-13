"""Conservative projection of audited Widelands worker/critter tracks.

Planning is pure and partitions every audited action/direction track.  A track
is projected only when its mandatory scale-1 pixels are already the complete
runtime image.  If Widelands associates a player-color mask with the body, the
body and mask remain an exact modular pair in the exclusion ledger: the
projector does not invent a player color or approximate ``blit_blended``.

Readiness opens SQLite in read-only/query-only mode.  Database writes happen
only through the explicit projection functions after complete evidence and CAS
preflight checks.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from spritelab.adapters.widelands import (
    EXPECTED_WIDELANDS_ARCHIVE_SHA256,
    WIDELANDS_COMMIT,
    AnimationRecord,
    EntityRecord,
    EvidenceDocument,
    RightsAudit,
    SourceImage,
    WidelandsArchiveAudit,
    audit_known_widelands_archive,
)
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

SOURCE_ID = "widelands"
PROJECTION_VERSION = "widelands_exact_unmasked_complete_entity_projection_v1"

EXPECTED_PINNED_PROJECTED_SEQUENCE_COUNT = 193
EXPECTED_PINNED_PROJECTED_FRAME_COUNT = 3_272
EXPECTED_PINNED_PROJECTED_ENTITY_COUNT = 22
EXPECTED_PINNED_MODULAR_EXCLUSION_COUNT = 2_082
EXPECTED_PINNED_MODULAR_FRAME_COUNT = 31_142
EXPECTED_PINNED_REQUIRED_MEMBER_COUNT = 4_180
EXPECTED_PINNED_REQUIRED_SOURCE_LAYER_COUNT = 3_994
EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256 = (
    "6e9566274dbd1e2b7159814da7b2ef60532b5888a885b5373253fa7d89ca3fab"
)

_SOURCE_DIRECTION_TO_CANONICAL: dict[str | None, str | None] = {
    None: None,
    "northeast": "up_right",
    "east": "right",
    "southeast": "down_right",
    "southwest": "down_left",
    "west": "left",
    "northwest": "up_left",
}


@dataclass(frozen=True, slots=True)
class WidelandsProjectionEntity:
    """One stable entity identity shared by actions, directions, and variants."""

    entity_external_key: str
    entity_id: str
    tribe: str | None
    constructor_role: str
    entity_class: str
    entity_class_basis: str
    manifest_path: str
    manifest_member_path: str
    manifest_line_number: int
    animation_directory: str


@dataclass(frozen=True, slots=True)
class WidelandsProjectionLayer:
    """One exact scale-1 body or player-color-mask source image."""

    role: Literal["body", "playercolor_mask"]
    logical_path: str
    member_path: str
    sha256: str
    width: int
    height: int
    payload_deduplication_key: str
    geometry_basis: str


@dataclass(frozen=True, slots=True)
class WidelandsProjectionFrame:
    """One exact temporal occurrence and its body/mask layer pairing."""

    ordinal: int
    source_frame_index: int
    duration_milliseconds: int
    body_layer_index: int
    playercolor_mask_layer_index: int | None
    left: int
    top: int
    right: int
    bottom: int
    layer_pair_deduplication_key: str


@dataclass(frozen=True, slots=True)
class WidelandsProjectionRecord:
    """One admitted complete runtime-resolved action/direction timeline."""

    sequence_source_key: str
    track_content_deduplication_key: str
    appearance_variant_key: str
    entity: WidelandsProjectionEntity
    declared_name: str
    normalized_action: str
    normalized_action_basis: str
    variant_hint: str | None
    representation: Literal["spritesheet", "numbered_files"]
    source_direction: str | None
    canonical_direction: str | None
    direction_basis: str
    basename: str
    source_directory: str
    manifest_line_number: int
    fps: int | None
    frame_duration_milliseconds: int
    frame_duration_basis: str
    hotspot: tuple[int, int]
    loop_mode: Literal["loop", "one_shot"]
    body_layers: tuple[WidelandsProjectionLayer, ...]
    playercolor_mask_layers: tuple[WidelandsProjectionLayer, ...]
    frames: tuple[WidelandsProjectionFrame, ...]
    runtime_composite_status: Literal["exact_unmasked_complete_entity"]
    exact_runtime_composite: Literal[True]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def width(self) -> int:
        return self.frames[0].right - self.frames[0].left

    @property
    def height(self) -> int:
        return self.frames[0].bottom - self.frames[0].top

    @property
    def total_duration_milliseconds(self) -> int:
        return sum(frame.duration_milliseconds for frame in self.frames)

    @property
    def primary_source_blob_sha256(self) -> str | None:
        hashes = {layer.sha256 for layer in self.body_layers}
        return next(iter(hashes)) if len(hashes) == 1 else None


@dataclass(frozen=True, slots=True)
class WidelandsProjectionExclusion:
    """One modular or otherwise unsafe track retained without false compositing."""

    track_source_key: str
    track_content_deduplication_key: str
    appearance_variant_key: str
    entity_external_key: str
    entity_id: str
    entity_class: str
    constructor_role: str
    manifest_path: str
    manifest_member_path: str
    manifest_line_number: int
    declared_name: str
    normalized_action: str | None
    normalized_action_basis: str
    variant_hint: str | None
    representation: Literal["spritesheet", "numbered_files"]
    source_direction: str | None
    canonical_direction: str | None
    basename: str
    source_directory: str
    loop_mode: Literal["loop", "one_shot"]
    body_layers: tuple[WidelandsProjectionLayer, ...]
    playercolor_mask_layers: tuple[WidelandsProjectionLayer, ...]
    frames: tuple[WidelandsProjectionFrame, ...]
    runtime_composite_status: str
    exact_runtime_composite: Literal[False]
    required_runtime_parameters: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def has_exact_modular_pairs(self) -> bool:
        return bool(self.frames) and all(
            frame.playercolor_mask_layer_index is not None for frame in self.frames
        )


@dataclass(frozen=True, slots=True)
class WidelandsProjectionPlan:
    """Pure deterministic plan, including the complete modular quarantine ledger."""

    archive_sha256: str
    repository_commit: str
    archive_root: str
    source_audit_record_sha256: str
    taxonomy_version: str
    taxonomy_action_values: tuple[str, ...]
    taxonomy_direction_values: tuple[str, ...]
    taxonomy_entity_values: tuple[str, ...]
    records: tuple[WidelandsProjectionRecord, ...]
    exclusions: tuple[WidelandsProjectionExclusion, ...]
    rights: RightsAudit
    engine_evidence: tuple[EvidenceDocument, ...]

    @property
    def projected_sequence_count(self) -> int:
        return len(self.records)

    @property
    def projected_entity_count(self) -> int:
        return len({record.entity.entity_external_key for record in self.records})

    @property
    def projected_frame_count(self) -> int:
        return sum(record.frame_count for record in self.records)

    @property
    def projected_animated_sequence_count(self) -> int:
        return sum(record.frame_count > 1 for record in self.records)

    @property
    def projected_static_sequence_count(self) -> int:
        return sum(record.frame_count == 1 for record in self.records)

    @property
    def projected_loop_count(self) -> int:
        return sum(record.loop_mode == "loop" for record in self.records)

    @property
    def projected_one_shot_count(self) -> int:
        return sum(record.loop_mode == "one_shot" for record in self.records)

    @property
    def modular_exclusion_count(self) -> int:
        return sum(exclusion.has_exact_modular_pairs for exclusion in self.exclusions)

    @property
    def excluded_frame_count(self) -> int:
        return sum(exclusion.frame_count for exclusion in self.exclusions)

    @property
    def projected_body_layer_hashes(self) -> tuple[tuple[str, str], ...]:
        return _layer_hashes(layer for record in self.records for layer in record.body_layers)

    @property
    def modular_body_layer_hashes(self) -> tuple[tuple[str, str], ...]:
        return _layer_hashes(layer for row in self.exclusions for layer in row.body_layers)

    @property
    def modular_mask_layer_hashes(self) -> tuple[tuple[str, str], ...]:
        return _layer_hashes(
            layer for row in self.exclusions for layer in row.playercolor_mask_layers
        )

    @property
    def required_source_layer_hashes(self) -> tuple[tuple[str, str], ...]:
        return _merge_hash_sets(
            self.projected_body_layer_hashes,
            self.modular_body_layer_hashes,
            self.modular_mask_layer_hashes,
        )

    @property
    def rights_documents(self) -> tuple[EvidenceDocument, ...]:
        return (
            self.rights.root_license,
            self.rights.in_game_license,
            self.rights.credits,
            self.rights.developers,
        )

    @property
    def required_member_paths(self) -> tuple[str, ...]:
        paths = {path for path, _ in self.required_source_layer_hashes}
        paths.update(record.entity.manifest_member_path for record in self.records)
        paths.update(exclusion.manifest_member_path for exclusion in self.exclusions)
        paths.update(document.member_path for document in self.rights_documents)
        paths.update(document.member_path for document in self.engine_evidence)
        return tuple(sorted(paths))

    @property
    def required_evidence_hashes(self) -> tuple[tuple[str, str], ...]:
        return _merge_hash_sets(
            self.required_source_layer_hashes,
            tuple((document.member_path, document.sha256) for document in self.rights_documents),
            tuple((document.member_path, document.sha256) for document in self.engine_evidence),
        )

    @property
    def projected_occurrence_link_count(self) -> int:
        fixed = 1 + len(self.rights_documents) + len(self.engine_evidence)
        return sum(fixed + len(record.body_layers) for record in self.records)

    @property
    def duplicate_source_layer_hash_groups(self) -> int:
        counts: dict[str, int] = {}
        for _, digest in self.required_source_layer_hashes:
            counts[digest] = counts.get(digest, 0) + 1
        return sum(value > 1 for value in counts.values())

    @property
    def duplicate_source_layer_hash_excess(self) -> int:
        counts: dict[str, int] = {}
        for _, digest in self.required_source_layer_hashes:
            counts[digest] = counts.get(digest, 0) + 1
        return sum(value - 1 for value in counts.values() if value > 1)

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
            "taxonomy_direction_values": list(self.taxonomy_direction_values),
            "taxonomy_entity_values": list(self.taxonomy_entity_values),
            "records": [asdict(record) for record in self.records],
            "exclusions": [asdict(exclusion) for exclusion in self.exclusions],
            "rights": asdict(self.rights),
            "engine_evidence": [asdict(document) for document in self.engine_evidence],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class WidelandsProjectionReadiness:
    """Query-only evidence-closure status for one projection plan."""

    database_path: str
    archive_sha256: str
    projection_manifest_sha256: str
    source_registered: bool
    archive_inventory_present: bool
    source_item_count: int
    required_member_count: int
    present_member_count: int
    required_source_layer_count: int
    present_source_layer_blob_count: int
    required_projected_body_count: int
    present_projected_body_blob_count: int
    required_modular_body_count: int
    present_modular_body_blob_count: int
    required_modular_mask_count: int
    present_modular_mask_blob_count: int
    missing_member_paths: tuple[str, ...]
    missing_source_layer_blobs: tuple[str, ...]
    source_layer_hash_mismatches: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.source_registered
            and self.archive_inventory_present
            and self.source_item_count > 0
            and not self.missing_member_paths
            and not self.missing_source_layer_blobs
            and not self.source_layer_hash_mismatches
        )


@dataclass(frozen=True, slots=True)
class WidelandsProjectionResult:
    """Effects of one idempotent projection call."""

    archive_sha256: str
    projection_manifest_sha256: str
    projected_entities: int
    projected_sequences: int
    projected_frames: int
    projected_animated_sequences: int
    projected_static_sequences: int
    projected_loops: int
    projected_one_shots: int
    created_sequences: int
    reused_sequences: int
    occurrence_links: int
    modular_exclusions: int
    excluded_frames: int
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


def _layer_hashes(
    layers: Iterable[WidelandsProjectionLayer],
) -> tuple[tuple[str, str], ...]:
    values: dict[str, str] = {}
    for layer in layers:
        previous = values.setdefault(layer.member_path, layer.sha256)
        if previous != layer.sha256:
            raise ValueError(
                f"One Widelands layer member has multiple audited hashes: {layer.member_path!r}"
            )
    return tuple(sorted(values.items()))


def _merge_hash_sets(
    *sets: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    values: dict[str, str] = {}
    for rows in sets:
        for member_path, digest in rows:
            previous = values.setdefault(member_path, digest)
            if previous != digest:
                raise ValueError(f"Conflicting Widelands evidence hash: {member_path!r}")
    return tuple(sorted(values.items()))


def _entity_external_key(audit: WidelandsArchiveAudit, entity: EntityRecord) -> str:
    return _stable_json_key(
        "widelands_entity",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "entity_id": entity.entity_id,
            "manifest_path": entity.manifest_path,
            "constructor_role": entity.constructor_role,
        },
    )


def _projection_entity(
    audit: WidelandsArchiveAudit, entity: EntityRecord
) -> WidelandsProjectionEntity:
    return WidelandsProjectionEntity(
        entity_external_key=_entity_external_key(audit, entity),
        entity_id=entity.entity_id,
        tribe=entity.tribe,
        constructor_role=entity.constructor_role,
        entity_class=entity.entity_class,
        entity_class_basis=entity.entity_class_basis,
        manifest_path=entity.manifest_path,
        manifest_member_path=entity.member_path,
        manifest_line_number=entity.location.line_number,
        animation_directory=entity.animation_directory,
    )


def _track_source_key(
    audit: WidelandsArchiveAudit,
    entity: EntityRecord,
    animation: AnimationRecord,
) -> str:
    return _stable_json_key(
        "widelands_track",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "entity_id": entity.entity_id,
            "manifest_path": entity.manifest_path,
            "manifest_line_number": animation.location.line_number,
            "declared_name": animation.declared_name,
            "representation": animation.representation,
            "source_direction": animation.direction,
            "basename": animation.basename,
            "source_directory": animation.source_directory,
        },
    )


def _appearance_variant_key(
    audit: WidelandsArchiveAudit,
    entity: EntityRecord,
    animation: AnimationRecord,
) -> str:
    return _stable_json_key(
        "widelands_appearance",
        {
            "archive_sha256": audit.archive_sha256,
            "entity_external_key": _entity_external_key(audit, entity),
            "source_directory": animation.source_directory,
            "variant_hint": animation.variant_hint or "default",
        },
    )


def _mask_member_path(audit: WidelandsArchiveAudit, source: SourceImage) -> str | None:
    if source.playercolor_mask_path is None:
        return None
    return f"{audit.archive_root}/{source.playercolor_mask_path}"


def _body_layer(source: SourceImage) -> WidelandsProjectionLayer:
    return WidelandsProjectionLayer(
        role="body",
        logical_path=source.logical_path,
        member_path=source.member_path,
        sha256=source.sha256,
        width=source.width,
        height=source.height,
        payload_deduplication_key=f"sha256:{source.sha256}",
        geometry_basis="audited_scale_1_source_dimensions",
    )


def _mask_layer(
    audit: WidelandsArchiveAudit, source: SourceImage
) -> WidelandsProjectionLayer | None:
    member_path = _mask_member_path(audit, source)
    if member_path is None or source.playercolor_mask_sha256 is None:
        return None
    return WidelandsProjectionLayer(
        role="playercolor_mask",
        logical_path=source.playercolor_mask_path or "",
        member_path=member_path,
        sha256=source.playercolor_mask_sha256,
        width=source.width,
        height=source.height,
        payload_deduplication_key=f"sha256:{source.playercolor_mask_sha256}",
        geometry_basis="adapter_validated_equal_to_body_dimensions",
    )


def _ordered_unique_layers(
    layers: list[WidelandsProjectionLayer],
) -> tuple[WidelandsProjectionLayer, ...]:
    values: dict[str, WidelandsProjectionLayer] = {}
    for layer in layers:
        previous = values.setdefault(layer.member_path, layer)
        if previous != layer:
            raise ValueError(f"Conflicting Widelands layer facts: {layer.member_path!r}")
    return tuple(values.values())


def _track_layers_and_frames(
    audit: WidelandsArchiveAudit,
    animation: AnimationRecord,
) -> tuple[
    tuple[WidelandsProjectionLayer, ...],
    tuple[WidelandsProjectionLayer, ...],
    tuple[WidelandsProjectionFrame, ...],
]:
    primary_sources = tuple(source for source in animation.source_images if source.scale == 1.0)
    source_by_path = {source.logical_path: source for source in primary_sources}
    body_layers = _ordered_unique_layers([_body_layer(source) for source in primary_sources])
    mask_layers = _ordered_unique_layers(
        [layer for source in primary_sources if (layer := _mask_layer(audit, source)) is not None]
    )
    body_indices = {layer.logical_path: index for index, layer in enumerate(body_layers)}
    mask_indices = {layer.logical_path: index for index, layer in enumerate(mask_layers)}
    frames: list[WidelandsProjectionFrame] = []
    for source_frame in animation.frames:
        try:
            source = source_by_path[source_frame.source_logical_path]
        except KeyError as error:
            raise ValueError(
                "Widelands frame source is absent from its scale-1 source list: "
                f"{source_frame.source_logical_path!r}"
            ) from error
        mask_path = source.playercolor_mask_path
        mask_index = mask_indices.get(mask_path) if mask_path is not None else None
        body_index = body_indices[source.logical_path]
        pair_key = _stable_json_key(
            "widelands_frame_layers",
            {
                "body_sha256": source.sha256,
                "mask_sha256": source.playercolor_mask_sha256,
                "rect": [
                    source_frame.x,
                    source_frame.y,
                    source_frame.width,
                    source_frame.height,
                ],
            },
        )
        frames.append(
            WidelandsProjectionFrame(
                ordinal=source_frame.ordinal,
                source_frame_index=(
                    source_frame.ordinal if animation.representation == "spritesheet" else 0
                ),
                duration_milliseconds=source_frame.duration_milliseconds,
                body_layer_index=body_index,
                playercolor_mask_layer_index=mask_index,
                left=source_frame.x,
                top=source_frame.y,
                right=source_frame.x + source_frame.width,
                bottom=source_frame.y + source_frame.height,
                layer_pair_deduplication_key=pair_key,
            )
        )
    return body_layers, mask_layers, tuple(frames)


def _track_content_key(
    body_layers: tuple[WidelandsProjectionLayer, ...],
    mask_layers: tuple[WidelandsProjectionLayer, ...],
    frames: tuple[WidelandsProjectionFrame, ...],
) -> str:
    return _stable_json_key(
        "widelands_track_content",
        {
            "body_hashes": [layer.sha256 for layer in body_layers],
            "mask_hashes": [layer.sha256 for layer in mask_layers],
            "frames": [asdict(frame) for frame in frames],
        },
    )


def _canonical_direction(source_direction: str | None) -> str | None:
    try:
        return _SOURCE_DIRECTION_TO_CANONICAL[source_direction]
    except KeyError as error:
        raise ValueError(f"Unsupported Widelands source direction: {source_direction!r}") from error


def _exclusion_reasons(
    entity: EntityRecord,
    animation: AnimationRecord,
    taxonomy: Taxonomy,
    mask_layers: tuple[WidelandsProjectionLayer, ...],
    frames: tuple[WidelandsProjectionFrame, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not entity.complete_entity:
        reasons.append("entity:not_complete_entity")
    reasons.extend(f"entity_quarantine:{reason}" for reason in entity.quarantine_reasons)
    if not animation.exact_source_sequence:
        reasons.append("track:not_exact_source_sequence")
    reasons.extend(f"track_quarantine:{reason}" for reason in animation.quarantine_reasons)
    if animation.normalized_action is None:
        reasons.append("action:unmapped")
    elif animation.normalized_action not in taxonomy.action_to_family:
        reasons.append(f"action:noncanonical:{animation.normalized_action}")
    try:
        canonical_direction = _canonical_direction(animation.direction)
    except ValueError:
        canonical_direction = "unknown"
        reasons.append(f"direction:unsupported:{animation.direction}")
    if canonical_direction is not None and canonical_direction not in taxonomy.directions:
        reasons.append(f"direction:noncanonical:{canonical_direction}")
    if not frames:
        reasons.append("track:no_exact_frames")
    elif len({(frame.right - frame.left, frame.bottom - frame.top) for frame in frames}) != 1:
        reasons.append("track:inconsistent_frame_dimensions")
    mask_indices = [frame.playercolor_mask_layer_index for frame in frames]
    if mask_layers and all(index is not None for index in mask_indices):
        reasons.extend(
            (
                "runtime_composite:playercolor_mask_required",
                "runtime_composite:player_color_parameter_unbound",
                "runtime_composite:engine_blend_not_materialized",
            )
        )
    elif mask_layers or any(index is not None for index in mask_indices):
        reasons.append("runtime_composite:incomplete_playercolor_mask_pairing")
    return tuple(dict.fromkeys(reasons))


def _projection_record(
    audit: WidelandsArchiveAudit,
    entity: EntityRecord,
    animation: AnimationRecord,
    body_layers: tuple[WidelandsProjectionLayer, ...],
    frames: tuple[WidelandsProjectionFrame, ...],
) -> WidelandsProjectionRecord:
    if animation.normalized_action is None:
        raise ValueError("Unmapped Widelands action reached record construction")
    if any(frame.playercolor_mask_layer_index is not None for frame in frames):
        raise ValueError("Player-color Widelands track reached complete projection")
    track_key = _track_source_key(audit, entity, animation)
    return WidelandsProjectionRecord(
        sequence_source_key=track_key,
        track_content_deduplication_key=_track_content_key(body_layers, (), frames),
        appearance_variant_key=_appearance_variant_key(audit, entity, animation),
        entity=_projection_entity(audit, entity),
        declared_name=animation.declared_name,
        normalized_action=animation.normalized_action,
        normalized_action_basis=animation.normalized_action_basis,
        variant_hint=animation.variant_hint,
        representation=animation.representation,
        source_direction=animation.direction,
        canonical_direction=_canonical_direction(animation.direction),
        direction_basis=animation.direction_basis,
        basename=animation.basename,
        source_directory=animation.source_directory,
        manifest_line_number=animation.location.line_number,
        fps=animation.fps,
        frame_duration_milliseconds=animation.frame_duration_milliseconds,
        frame_duration_basis=animation.frame_duration_basis,
        hotspot=animation.hotspot,
        loop_mode=animation.loop_mode,
        body_layers=body_layers,
        playercolor_mask_layers=(),
        frames=frames,
        runtime_composite_status="exact_unmasked_complete_entity",
        exact_runtime_composite=True,
    )


def _projection_exclusion(
    audit: WidelandsArchiveAudit,
    entity: EntityRecord,
    animation: AnimationRecord,
    body_layers: tuple[WidelandsProjectionLayer, ...],
    mask_layers: tuple[WidelandsProjectionLayer, ...],
    frames: tuple[WidelandsProjectionFrame, ...],
    reasons: tuple[str, ...],
) -> WidelandsProjectionExclusion:
    if mask_layers and all(frame.playercolor_mask_layer_index is not None for frame in frames):
        status = "modular_body_plus_playercolor_mask_unresolved"
        parameters = ("player_color",)
    else:
        status = "unsafe_or_incomplete_track"
        parameters = ()
    return WidelandsProjectionExclusion(
        track_source_key=_track_source_key(audit, entity, animation),
        track_content_deduplication_key=_track_content_key(body_layers, mask_layers, frames),
        appearance_variant_key=_appearance_variant_key(audit, entity, animation),
        entity_external_key=_entity_external_key(audit, entity),
        entity_id=entity.entity_id,
        entity_class=entity.entity_class,
        constructor_role=entity.constructor_role,
        manifest_path=entity.manifest_path,
        manifest_member_path=entity.member_path,
        manifest_line_number=animation.location.line_number,
        declared_name=animation.declared_name,
        normalized_action=animation.normalized_action,
        normalized_action_basis=animation.normalized_action_basis,
        variant_hint=animation.variant_hint,
        representation=animation.representation,
        source_direction=animation.direction,
        canonical_direction=_SOURCE_DIRECTION_TO_CANONICAL.get(animation.direction),
        basename=animation.basename,
        source_directory=animation.source_directory,
        loop_mode=animation.loop_mode,
        body_layers=body_layers,
        playercolor_mask_layers=mask_layers,
        frames=frames,
        runtime_composite_status=status,
        exact_runtime_composite=False,
        required_runtime_parameters=parameters,
        reasons=reasons,
    )


def plan_widelands_projection(
    audit: WidelandsArchiveAudit,
    taxonomy: Taxonomy,
) -> WidelandsProjectionPlan:
    """Partition every audited track into exact output or explicit quarantine."""

    records: list[WidelandsProjectionRecord] = []
    exclusions: list[WidelandsProjectionExclusion] = []
    audited_track_count = 0
    audited_frame_count = 0
    for entity in audit.entities:
        for animation in entity.animations:
            audited_track_count += 1
            audited_frame_count += len(animation.frames)
            body_layers, mask_layers, frames = _track_layers_and_frames(audit, animation)
            reasons = _exclusion_reasons(entity, animation, taxonomy, mask_layers, frames)
            if reasons:
                exclusions.append(
                    _projection_exclusion(
                        audit,
                        entity,
                        animation,
                        body_layers,
                        mask_layers,
                        frames,
                        reasons,
                    )
                )
            else:
                records.append(_projection_record(audit, entity, animation, body_layers, frames))
    records.sort(key=lambda record: record.sequence_source_key)
    exclusions.sort(key=lambda exclusion: exclusion.track_source_key)
    if len(records) + len(exclusions) != audited_track_count:
        raise AssertionError("Widelands plan does not partition every audited track")
    if (
        sum(record.frame_count for record in records)
        + sum(exclusion.frame_count for exclusion in exclusions)
        != audited_frame_count
    ):
        raise AssertionError("Widelands plan does not partition every audited frame")
    record_keys = [record.sequence_source_key for record in records]
    exclusion_keys = [exclusion.track_source_key for exclusion in exclusions]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("Widelands projected track keys are not unique")
    if len(exclusion_keys) != len(set(exclusion_keys)):
        raise ValueError("Widelands exclusion track keys are not unique")
    if set(record_keys).intersection(exclusion_keys):
        raise ValueError("Widelands track appears in both projection and exclusion ledgers")
    if any(record.playercolor_mask_layers for record in records):
        raise AssertionError("Widelands complete projection retained a player-color layer")
    if any(
        frame.playercolor_mask_layer_index is not None
        for record in records
        for frame in record.frames
    ):
        raise AssertionError("Widelands complete projection retained a modular frame")
    return WidelandsProjectionPlan(
        archive_sha256=audit.archive_sha256,
        repository_commit=audit.commit,
        archive_root=audit.archive_root,
        source_audit_record_sha256=audit.audit_record_sha256,
        taxonomy_version=taxonomy.version,
        taxonomy_action_values=tuple(sorted(taxonomy.action_to_family)),
        taxonomy_direction_values=tuple(sorted(taxonomy.directions)),
        taxonomy_entity_values=tuple(sorted(taxonomy.entity_classes)),
        records=tuple(records),
        exclusions=tuple(exclusions),
        rights=audit.rights,
        engine_evidence=audit.engine_evidence,
    )


def plan_known_widelands_projection(
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> WidelandsProjectionPlan:
    """Audit the pinned CAS snapshot and enforce exact projection counts."""

    plan = plan_widelands_projection(audit_known_widelands_archive(Path(archive_path)), taxonomy)
    expected = (
        EXPECTED_PINNED_PROJECTED_SEQUENCE_COUNT,
        EXPECTED_PINNED_PROJECTED_FRAME_COUNT,
        EXPECTED_PINNED_PROJECTED_ENTITY_COUNT,
        EXPECTED_PINNED_MODULAR_EXCLUSION_COUNT,
        EXPECTED_PINNED_MODULAR_FRAME_COUNT,
        EXPECTED_PINNED_REQUIRED_MEMBER_COUNT,
        EXPECTED_PINNED_REQUIRED_SOURCE_LAYER_COUNT,
    )
    actual = (
        plan.projected_sequence_count,
        plan.projected_frame_count,
        plan.projected_entity_count,
        plan.modular_exclusion_count,
        plan.excluded_frame_count,
        len(plan.required_member_paths),
        len(plan.required_source_layer_hashes),
    )
    if actual != expected:
        raise ValueError(
            f"Pinned Widelands projection count drift: expected {expected}, got {actual}"
        )
    if plan.projection_manifest_sha256 != EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256:
        raise ValueError(
            "Pinned Widelands projection manifest drift: expected "
            f"{EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256}, got "
            f"{plan.projection_manifest_sha256}"
        )
    return plan


def _readonly_connection(database_path: str | Path) -> sqlite3.Connection:
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _readiness_group_count(
    expected: tuple[tuple[str, str], ...],
    members: dict[str, sqlite3.Row],
) -> int:
    count = 0
    for member_path, expected_hash in expected:
        row = members.get(member_path)
        if row is None:
            continue
        if (
            row["extracted_blob_sha256"] == expected_hash
            and row["registered_blob_sha256"] == expected_hash
        ):
            count += 1
    return count


def check_widelands_projection_readiness(
    database_path: str | Path,
    plan: WidelandsProjectionPlan,
) -> WidelandsProjectionReadiness:
    """Inspect live or temporary prerequisites without any SQLite write."""

    required_paths = plan.required_member_paths
    expected_layers = dict(plan.required_source_layer_hashes)
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
    present_layers = 0
    for member_path, expected_hash in sorted(expected_layers.items()):
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
            present_layers += 1
    return WidelandsProjectionReadiness(
        database_path=str(Path(database_path).resolve()),
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=plan.projection_manifest_sha256,
        source_registered=source_registered,
        archive_inventory_present=archive_inventory_present,
        source_item_count=source_item_count,
        required_member_count=len(required_paths),
        present_member_count=len(required_paths) - len(missing_paths),
        required_source_layer_count=len(expected_layers),
        present_source_layer_blob_count=present_layers,
        required_projected_body_count=len(plan.projected_body_layer_hashes),
        present_projected_body_blob_count=_readiness_group_count(
            plan.projected_body_layer_hashes, members
        ),
        required_modular_body_count=len(plan.modular_body_layer_hashes),
        present_modular_body_blob_count=_readiness_group_count(
            plan.modular_body_layer_hashes, members
        ),
        required_modular_mask_count=len(plan.modular_mask_layer_hashes),
        present_modular_mask_blob_count=_readiness_group_count(
            plan.modular_mask_layer_hashes, members
        ),
        missing_member_paths=missing_paths,
        missing_source_layer_blobs=tuple(missing_blobs),
        source_layer_hash_mismatches=tuple(mismatches),
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
            f"Widelands archive has no indexed source item for {SOURCE_ID!r}: {archive_sha256}"
        )
    return str(row[0])


def _preflight(
    database: IndexDB,
    plan: WidelandsProjectionPlan,
) -> tuple[str, dict[str, sqlite3.Row]]:
    with database.connect() as connection:
        source = connection.execute("SELECT 1 FROM sources WHERE id=?", (SOURCE_ID,)).fetchone()
        inventory = connection.execute(
            "SELECT 1 FROM archive_inventories WHERE archive_blob_sha256=?",
            (plan.archive_sha256,),
        ).fetchone()
    if source is None:
        raise ValueError(f"Widelands source registry row is missing: {SOURCE_ID}")
    if inventory is None:
        raise ValueError(f"Widelands archive inventory is missing: {plan.archive_sha256}")
    item_id = _item_id(database, plan.archive_sha256)
    members = _archive_members(database, plan.archive_sha256)
    missing = [path for path in plan.required_member_paths if path not in members]
    if missing:
        raise ValueError(
            "Widelands projection evidence members are missing: " + ", ".join(missing[:10])
        )
    for member_path, expected_hash in plan.required_source_layer_hashes:
        member = members[member_path]
        actual_hash = member["extracted_blob_sha256"]
        if actual_hash is None:
            raise ValueError(f"Widelands source layer is not extracted into CAS: {member_path}")
        if str(actual_hash) != expected_hash:
            raise ValueError(
                "Widelands source layer CAS hash mismatch for "
                f"{member_path}: expected {expected_hash}, indexed {actual_hash}"
            )
        if member["registered_blob_sha256"] is None:
            raise ValueError(f"Widelands source layer CAS blob is not registered: {member_path}")
    return item_id, members


def _rights_metadata(plan: WidelandsProjectionPlan) -> dict[str, Any]:
    return {
        "license_expression": plan.rights.license_expression,
        "license_basis": plan.rights.license_basis,
        "caveat": plan.rights.caveat,
        "documents": [asdict(document) for document in plan.rights_documents],
        "file_level_creator_attribution_claimed": False,
        "rights_observation_added": False,
    }


def _sequence_metadata(
    plan: WidelandsProjectionPlan,
    record: WidelandsProjectionRecord,
    manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "source_audit_record_sha256": plan.source_audit_record_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "sequence_source_key": record.sequence_source_key,
        "track_content_deduplication_key": record.track_content_deduplication_key,
        "appearance_variant_key": record.appearance_variant_key,
        "declared_name": record.declared_name,
        "normalized_action": record.normalized_action,
        "normalized_action_basis": record.normalized_action_basis,
        "variant_hint": record.variant_hint,
        "representation": record.representation,
        "source_direction": record.source_direction,
        "canonical_direction": record.canonical_direction,
        "direction_basis": record.direction_basis,
        "basename": record.basename,
        "source_directory": record.source_directory,
        "manifest_line_number": record.manifest_line_number,
        "fps": record.fps,
        "frame_duration_milliseconds": record.frame_duration_milliseconds,
        "frame_duration_basis": record.frame_duration_basis,
        "duration_ms_per_occurrence": [frame.duration_milliseconds for frame in record.frames],
        "total_duration_ms": record.total_duration_milliseconds,
        "hotspot": list(record.hotspot),
        "loop_mode": record.loop_mode,
        "loop_policy_inferred": False,
        "runtime_composite_status": record.runtime_composite_status,
        "exact_runtime_composite": record.exact_runtime_composite,
        "playercolor_mask_required": False,
        "playercolor_mask_layers": [],
        "body_layers": [asdict(layer) for layer in record.body_layers],
        "source_image_hash_order": [
            record.body_layers[frame.body_layer_index].sha256 for frame in record.frames
        ],
        "source_frame_index_order": [frame.source_frame_index for frame in record.frames],
        "native_source_rectangles_preserved": True,
        "frame_order_preserved": True,
        "exact_engine_timing": True,
        "clipping_or_repair_applied": False,
        "rights_scope": _rights_metadata(plan),
        "engine_evidence": [asdict(document) for document in plan.engine_evidence],
    }


def _occurrence_specs(
    plan: WidelandsProjectionPlan,
    record: WidelandsProjectionRecord,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    common = {
        "projection_version": PROJECTION_VERSION,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "entity_id": record.entity.entity_id,
        "declared_name": record.declared_name,
        "source_direction": record.source_direction,
        "sequence_source_key": record.sequence_source_key,
    }
    specs: list[tuple[str, str, dict[str, Any]]] = []
    for index, layer in enumerate(record.body_layers):
        ordinals = [frame.ordinal for frame in record.frames if frame.body_layer_index == index]
        specs.append(
            (
                layer.member_path,
                "widelands_complete_unmasked_body_source",
                {
                    **common,
                    "layer": asdict(layer),
                    "sequence_ordinals": ordinals,
                    "runtime_composite_status": record.runtime_composite_status,
                    "playercolor_mask_required": False,
                },
            )
        )
    specs.append(
        (
            record.entity.manifest_member_path,
            "widelands_entity_animation_manifest",
            {
                **common,
                "manifest_path": record.entity.manifest_path,
                "manifest_line_number": record.manifest_line_number,
                "source_audit_record_sha256": plan.source_audit_record_sha256,
            },
        )
    )
    specs.extend(
        (
            document.member_path,
            "widelands_collection_rights_evidence",
            {
                **common,
                "evidence": asdict(document),
                "rights_scope": _rights_metadata(plan),
            },
        )
        for document in plan.rights_documents
    )
    specs.extend(
        (
            document.member_path,
            "widelands_engine_animation_semantics",
            {**common, "evidence": asdict(document)},
        )
        for document in plan.engine_evidence
    )
    return tuple(specs)


def _frame_metadata(
    plan: WidelandsProjectionPlan,
    record: WidelandsProjectionRecord,
    frame: WidelandsProjectionFrame,
) -> dict[str, Any]:
    body = record.body_layers[frame.body_layer_index]
    return {
        "projection_version": PROJECTION_VERSION,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "entity_id": record.entity.entity_id,
        "declared_name": record.declared_name,
        "normalized_action": record.normalized_action,
        "variant_hint": record.variant_hint,
        "source_direction": record.source_direction,
        "canonical_direction": record.canonical_direction,
        "body_layer": asdict(body),
        "playercolor_mask_layer": None,
        "playercolor_mask_required": False,
        "runtime_composite_status": record.runtime_composite_status,
        "exact_runtime_composite": True,
        "source_frame_index": frame.source_frame_index,
        "sequence_ordinal": frame.ordinal,
        "duration_milliseconds": frame.duration_milliseconds,
        "frame_rect": {
            "left": frame.left,
            "top": frame.top,
            "right": frame.right,
            "bottom": frame.bottom,
            "width": frame.right - frame.left,
            "height": frame.bottom - frame.top,
            "coordinate_space": "source_image",
        },
        "layer_pair_deduplication_key": frame.layer_pair_deduplication_key,
        "track_content_deduplication_key": record.track_content_deduplication_key,
        "exact_engine_timing": True,
        "native_source_rectangle": True,
        "clipping_or_repair_applied": False,
        "rights_scope": _rights_metadata(plan),
    }


def _validate_taxonomy_contract(plan: WidelandsProjectionPlan, taxonomy: Taxonomy) -> None:
    if taxonomy.version != plan.taxonomy_version:
        raise ValueError(
            "Widelands projection taxonomy version mismatch: "
            f"plan {plan.taxonomy_version!r}, runtime {taxonomy.version!r}"
        )
    if tuple(sorted(taxonomy.action_to_family)) != plan.taxonomy_action_values:
        raise ValueError("Widelands projection taxonomy action vocabulary has changed")
    if tuple(sorted(taxonomy.directions)) != plan.taxonomy_direction_values:
        raise ValueError("Widelands projection taxonomy direction vocabulary has changed")
    if tuple(sorted(taxonomy.entity_classes)) != plan.taxonomy_entity_values:
        raise ValueError("Widelands projection taxonomy entity vocabulary has changed")


def project_widelands_audit(
    database: IndexDB,
    plan: WidelandsProjectionPlan,
    taxonomy: Taxonomy,
) -> WidelandsProjectionResult:
    """Idempotently project a precomputed, complete-pixels-only plan."""

    _validate_taxonomy_contract(plan, taxonomy)
    database.initialize()
    item_id, members = _preflight(database, plan)
    manifest_sha256 = plan.projection_manifest_sha256
    rights_scope = _rights_metadata(plan)
    entity_ids: dict[str, str] = {}
    created_sequences = 0
    reused_sequences = 0
    occurrence_links = 0

    for record in plan.records:
        if not record.exact_runtime_composite or record.playercolor_mask_layers:
            raise ValueError("Unsafe modular Widelands track reached DB projection")
        entity = record.entity
        entity_id = entity_ids.get(entity.entity_external_key)
        if entity_id is None:
            normalized_entity = taxonomy.normalize_entity_class(entity.entity_class)
            if normalized_entity.value == "unknown":
                raise ValueError(
                    f"Widelands entity class became ambiguous: {entity.entity_class!r}"
                )
            entity_id = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=entity.entity_external_key,
                representative_item_id=item_id,
                display_name=entity.entity_id,
                entity_class=normalized_entity.value,
                entity_subclass=entity.constructor_role,
                species_or_type=entity.entity_id,
                taxonomy_version=taxonomy.version,
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "projection_manifest_sha256": manifest_sha256,
                    "source_audit_record_sha256": plan.source_audit_record_sha256,
                    "archive_sha256": plan.archive_sha256,
                    "repository_commit": plan.repository_commit,
                    "entity_id": entity.entity_id,
                    "tribe": entity.tribe,
                    "constructor_role": entity.constructor_role,
                    "adapter_entity_class": entity.entity_class,
                    "normalized_entity_class": normalized_entity.value,
                    "entity_class_basis": entity.entity_class_basis,
                    "manifest_path": entity.manifest_path,
                    "manifest_member_path": entity.manifest_member_path,
                    "manifest_line_number": entity.manifest_line_number,
                    "animation_directory": entity.animation_directory,
                    "classification_method": normalized_entity.method,
                    "classification_confidence": normalized_entity.confidence,
                    "rights_scope": rights_scope,
                },
            )
            entity_ids[entity.entity_external_key] = entity_id

        motion = taxonomy.motion_condition(
            action=record.normalized_action,
            direction=record.canonical_direction,
            view=None,
        )
        if motion.normalized_action != record.normalized_action:
            raise ValueError(
                "Widelands action was not taxonomy-canonical at write time: "
                f"{record.normalized_action!r}"
            )
        expected_direction = record.canonical_direction or "unknown"
        if motion.direction != expected_direction:
            raise ValueError(
                "Widelands direction was not taxonomy-canonical at write time: "
                f"{record.canonical_direction!r} -> {motion.direction!r}"
            )
        sequence_id = database.find_sequence_by_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
        )
        sequence_arguments = {
            "source_blob_sha256": (record.primary_source_blob_sha256 or plan.archive_sha256),
            "extraction_method": PROJECTION_VERSION,
            "extraction_confidence": 1.0,
            "width": record.width,
            "height": record.height,
            "frame_count": record.frame_count,
            "loop_mode": record.loop_mode,
            "action": motion.normalized_action,
            "direction": motion.direction,
            "quality_tier": "F0_lossless_widelands_exact_unmasked_complete_entity",
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
                "entity_id": entity.entity_id,
                "manifest_path": entity.manifest_path,
                "manifest_line_number": record.manifest_line_number,
                "appearance_variant_key": record.appearance_variant_key,
                "variant_hint": record.variant_hint,
                "complete_entity": True,
                "runtime_composite_status": record.runtime_composite_status,
                "exact_runtime_composite": True,
                "playercolor_mask_required": False,
                "rights_scope": rights_scope,
            },
        )
        loopable = record.loop_mode == "loop"
        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version=taxonomy.version,
            source_action=record.declared_name,
            normalized_action=motion.normalized_action,
            action_family=motion.action_family,
            annotation_method=PROJECTION_VERSION,
            view=motion.view,
            direction=motion.direction,
            loopable=loopable,
            cycle_frames=record.frame_count if loopable else None,
            phase_zero_frame=0,
            confidence=motion.confidence,
            conditioning={
                "declared_name": record.declared_name,
                "adapter_normalized_action": record.normalized_action,
                "normalized_action_basis": record.normalized_action_basis,
                "taxonomy_normalization_method": motion.method,
                "source_direction": record.source_direction,
                "canonical_direction": record.canonical_direction,
                "direction_basis": record.direction_basis,
                "appearance_variant_key": record.appearance_variant_key,
                "variant_hint": record.variant_hint,
                "basename": record.basename,
                "source_directory": record.source_directory,
                "hotspot": list(record.hotspot),
                "timing_known": True,
                "exact_engine_timing": True,
                "frame_duration_milliseconds": record.frame_duration_milliseconds,
                "duration_ms_per_occurrence": [
                    frame.duration_milliseconds for frame in record.frames
                ],
                "loop_mode": record.loop_mode,
                "loop_policy_inferred": False,
                "runtime_composite_status": record.runtime_composite_status,
                "exact_runtime_composite": True,
                "playercolor_mask_required": False,
                "track_content_deduplication_key": (record.track_content_deduplication_key),
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
            body = record.body_layers[frame.body_layer_index]
            phase = frame.ordinal / record.frame_count if record.loop_mode == "loop" else None
            database.add_sequence_frame(
                sequence_id=sequence_id,
                ordinal=frame.ordinal,
                source_blob_sha256=body.sha256,
                source_frame_index=frame.source_frame_index,
                duration_ms=frame.duration_milliseconds,
                phase=phase,
                direction=motion.direction,
                view=motion.view,
                metadata=_frame_metadata(plan, record, frame),
            )

    return WidelandsProjectionResult(
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=manifest_sha256,
        projected_entities=plan.projected_entity_count,
        projected_sequences=plan.projected_sequence_count,
        projected_frames=plan.projected_frame_count,
        projected_animated_sequences=plan.projected_animated_sequence_count,
        projected_static_sequences=plan.projected_static_sequence_count,
        projected_loops=plan.projected_loop_count,
        projected_one_shots=plan.projected_one_shot_count,
        created_sequences=created_sequences,
        reused_sequences=reused_sequences,
        occurrence_links=occurrence_links,
        modular_exclusions=plan.modular_exclusion_count,
        excluded_frames=plan.excluded_frame_count,
    )


def ingest_known_widelands_sequences(
    database: IndexDB,
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> WidelandsProjectionResult:
    """Audit and project only the exact pinned Widelands snapshot."""

    plan = plan_known_widelands_projection(archive_path, taxonomy)
    if (
        plan.archive_sha256 != EXPECTED_WIDELANDS_ARCHIVE_SHA256
        or plan.repository_commit != WIDELANDS_COMMIT
    ):
        raise ValueError("Refusing Widelands projection for an unexpected archive or commit")
    return project_widelands_audit(database, plan, taxonomy)


__all__ = [
    "EXPECTED_PINNED_MODULAR_EXCLUSION_COUNT",
    "EXPECTED_PINNED_MODULAR_FRAME_COUNT",
    "EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256",
    "EXPECTED_PINNED_PROJECTED_ENTITY_COUNT",
    "EXPECTED_PINNED_PROJECTED_FRAME_COUNT",
    "EXPECTED_PINNED_PROJECTED_SEQUENCE_COUNT",
    "EXPECTED_PINNED_REQUIRED_MEMBER_COUNT",
    "EXPECTED_PINNED_REQUIRED_SOURCE_LAYER_COUNT",
    "PROJECTION_VERSION",
    "SOURCE_ID",
    "WidelandsProjectionEntity",
    "WidelandsProjectionExclusion",
    "WidelandsProjectionFrame",
    "WidelandsProjectionLayer",
    "WidelandsProjectionPlan",
    "WidelandsProjectionReadiness",
    "WidelandsProjectionRecord",
    "WidelandsProjectionResult",
    "check_widelands_projection_readiness",
    "ingest_known_widelands_sequences",
    "plan_known_widelands_projection",
    "plan_widelands_projection",
    "project_widelands_audit",
]
