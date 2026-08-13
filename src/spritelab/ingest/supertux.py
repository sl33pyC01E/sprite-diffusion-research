"""Deterministic DB projection for the pinned SuperTux creature audit.

The pure planner partitions every declared sprite action.  Only effective,
exact, complete-entity timelines are admitted; components, effects, deprecated
manifests, superseded declarations, and incomplete aliases remain in a
serialized exclusion ledger.  Mirroring/flipping is preserved as an explicit
lossless transform recipe over immutable PNG blobs rather than silently
materialized pixels.

Readiness and preparation inspection opens SQLite read-only and query-only.
Writing is available only through the explicit projector, which performs a
complete evidence/CAS preflight before an atomic, idempotent transaction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from spritelab.adapters.supertux import (
    EXPECTED_SUPERTUX_ARCHIVE_SHA256,
    SUPERTUX_COMMIT,
    AcquisitionEvidence,
    ActionRecord,
    EvidenceDocument,
    ManifestRecord,
    RightsAudit,
    SuperTuxArchiveAudit,
    audit_known_supertux_archive,
)
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

SOURCE_ID = "supertux"
PROJECTION_VERSION = "supertux_exact_complete_entity_transform_recipe_v1"

# Filled from the exact pinned archive and intentionally enforced by the known
# helper.  Changes require reviewing the audit and exclusion ledger, not merely
# accepting newly observed values.
EXPECTED_PINNED_PROJECTED_SEQUENCE_COUNT = 1_010
EXPECTED_PINNED_PROJECTED_FRAME_COUNT = 7_103
EXPECTED_PINNED_PROJECTED_ENTITY_COUNT = 96
EXPECTED_PINNED_EXCLUSION_COUNT = 141
EXPECTED_PINNED_EXCLUDED_FRAME_COUNT = 588
EXPECTED_PINNED_REQUIRED_MEMBER_COUNT = 1_982
EXPECTED_PINNED_REQUIRED_SOURCE_IMAGE_COUNT = 1_841
EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256 = (
    "78680f69abc1ffe442eab73fd8e9c34e7dd9106ba7540de73af4593e2e96109d"
)

_ENTITY_CLASS_MAP = {
    "animal": "animal",
    "humanoid": "humanoid",
    "monster": "monster",
    # The shared taxonomy does not claim plant/elemental/construct subclasses.
    # Creature is the least-assumptive animate superclass; the exact adapter
    # label and classification basis remain first-class fields.
    "plant": "creature",
    "elemental": "creature",
    "construct": "creature",
}


@dataclass(frozen=True, slots=True)
class SuperTuxProjectionEntity:
    """Stable complete-entity identity shared by all admitted actions."""

    entity_external_key: str
    manifest_id: str
    entity_group: str
    display_name: str
    adapter_entity_class: str
    normalized_entity_class: str
    entity_class_basis: str
    manifest_logical_path: str
    manifest_member_path: str
    manifest_sha256: str
    role: str
    role_basis: str
    manifest_quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SuperTuxProjectionLayer:
    """One immutable PNG payload used by one or more temporal occurrences."""

    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    mode: str
    alpha_kind: str
    payload_deduplication_key: str


@dataclass(frozen=True, slots=True)
class SuperTuxProjectionFrame:
    """One ordered source occurrence, including an optional quarantined miss."""

    ordinal: int
    source_layer_index: int | None
    source_frame_index: int
    duration_milliseconds: float
    origin_action: str
    requested_path: str
    logical_path: str
    member_path: str | None
    exists: bool
    width: int | None
    height: int | None
    transform: str
    temporal_occurrence_key: str


LoopMode = Literal[
    "runtime_controlled",
    "engine_custom_infinite",
    "engine_custom_finite",
    "engine_custom_terminal",
]


@dataclass(frozen=True, slots=True)
class SuperTuxProjectionRecord:
    """One admitted complete-entity action with exact runtime source facts."""

    sequence_source_key: str
    track_content_deduplication_key: str
    appearance_variant_key: str
    entity: SuperTuxProjectionEntity
    declaration_ordinal: int
    manifest_line_number: int
    declared_name: str
    adapter_normalized_action: str
    adapter_normalized_action_basis: str
    normalized_action: str
    normalized_action_basis: str
    action_family: str
    source_direction: str | None
    canonical_direction: str
    direction_basis: str
    action_stem: str
    alias_kind: str | None
    alias_target: str | None
    alias_chain: tuple[str, ...]
    declared_image_paths: tuple[str, ...]
    declared_fps: float | None
    effective_fps: float
    frame_duration_milliseconds: float
    declared_loops: int | None
    effective_loops: int
    has_custom_loops: bool
    declared_loop_frame: int | None
    effective_loop_frame: int
    loop_start_ordinal: int
    loop_mode: LoopMode
    hitbox: tuple[float, float, float, float]
    unisolid: bool
    family_name: str
    layers: tuple[SuperTuxProjectionLayer, ...]
    frames: tuple[SuperTuxProjectionFrame, ...]
    variable_frame_geometry: bool
    runtime_composite_status: Literal["exact_complete_entity_transform_recipe"]
    exact_runtime_source_recipe: Literal[True]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def width(self) -> int:
        return max(layer.width for layer in self.layers)

    @property
    def height(self) -> int:
        return max(layer.height for layer in self.layers)

    @property
    def total_duration_milliseconds(self) -> float:
        return round(sum(frame.duration_milliseconds for frame in self.frames), 9)

    @property
    def primary_source_blob_sha256(self) -> str | None:
        digests = {layer.sha256 for layer in self.layers}
        return next(iter(digests)) if len(digests) == 1 else None

    @property
    def has_deferred_transform(self) -> bool:
        return any(frame.transform != "identity" for frame in self.frames)

    @property
    def loopable(self) -> bool | None:
        if not self.has_custom_loops:
            return None
        return self.effective_loops < 0

    @property
    def cycle_frames(self) -> int | None:
        if self.loopable is True:
            count = self.frame_count - self.loop_start_ordinal
            return count if count > 0 else None
        return None


@dataclass(frozen=True, slots=True)
class SuperTuxProjectionExclusion:
    """One rejected declaration with all auditable source and quarantine facts."""

    track_source_key: str
    track_content_deduplication_key: str
    appearance_variant_key: str
    manifest_id: str
    entity_group: str
    display_name: str
    role: str
    role_basis: str
    parent_entity_hint: str | None
    adapter_entity_class: str
    entity_class_basis: str
    manifest_logical_path: str
    manifest_member_path: str
    manifest_sha256: str
    manifest_quarantine_reasons: tuple[str, ...]
    declaration_ordinal: int
    manifest_line_number: int
    declared_name: str
    adapter_normalized_action: str
    adapter_normalized_action_basis: str
    source_direction: str | None
    direction_basis: str
    action_stem: str
    alias_kind: str | None
    alias_target: str | None
    alias_chain: tuple[str, ...]
    declared_image_paths: tuple[str, ...]
    declared_fps: float | None
    effective_fps: float
    frame_duration_milliseconds: float
    declared_loops: int | None
    effective_loops: int
    has_custom_loops: bool
    declared_loop_frame: int | None
    effective_loop_frame: int
    hitbox: tuple[float, float, float, float]
    unisolid: bool
    family_name: str
    effective_declaration: bool
    exact_source_sequence: bool
    layers: tuple[SuperTuxProjectionLayer, ...]
    frames: tuple[SuperTuxProjectionFrame, ...]
    action_quarantine_reasons: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True, slots=True)
class SuperTuxProjectionPlan:
    """Pure deterministic projection plus the complete declaration ledger."""

    archive_sha256: str
    repository_commit: str
    archive_root: str
    source_inventory_sha256: str
    source_audit_record_sha256: str
    taxonomy_version: str
    taxonomy_action_values: tuple[str, ...]
    taxonomy_action_families: tuple[tuple[str, str], ...]
    taxonomy_entity_aliases: tuple[tuple[str, str], ...]
    taxonomy_direction_values: tuple[str, ...]
    taxonomy_entity_values: tuple[str, ...]
    records: tuple[SuperTuxProjectionRecord, ...]
    exclusions: tuple[SuperTuxProjectionExclusion, ...]
    manifests: tuple[tuple[str, str, str], ...]
    rights: RightsAudit
    acquisition_evidence: tuple[AcquisitionEvidence, ...]
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
    def projected_runtime_controlled_count(self) -> int:
        return sum(record.loop_mode == "runtime_controlled" for record in self.records)

    @property
    def projected_custom_finite_count(self) -> int:
        return sum(record.loop_mode == "engine_custom_finite" for record in self.records)

    @property
    def projected_custom_infinite_count(self) -> int:
        return sum(record.loop_mode == "engine_custom_infinite" for record in self.records)

    @property
    def excluded_frame_count(self) -> int:
        return sum(exclusion.frame_count for exclusion in self.exclusions)

    @property
    def projected_source_image_hashes(self) -> tuple[tuple[str, str], ...]:
        return _layer_hashes(layer for record in self.records for layer in record.layers)

    @property
    def excluded_source_image_hashes(self) -> tuple[tuple[str, str], ...]:
        return _layer_hashes(layer for row in self.exclusions for layer in row.layers)

    @property
    def required_source_image_hashes(self) -> tuple[tuple[str, str], ...]:
        return _merge_hash_sets(
            self.projected_source_image_hashes,
            self.excluded_source_image_hashes,
        )

    @property
    def rights_documents(self) -> tuple[EvidenceDocument, ...]:
        return (
            self.rights.root_license,
            self.rights.readme,
            self.rights.authors,
            self.rights.credits,
        )

    @property
    def required_member_paths(self) -> tuple[str, ...]:
        paths = {path for path, _ in self.required_source_image_hashes}
        paths.update(member_path for _, member_path, _ in self.manifests)
        paths.update(document.member_path for document in self.rights_documents)
        paths.update(document.member_path for document in self.engine_evidence)
        return tuple(sorted(paths))

    @property
    def required_evidence_hashes(self) -> tuple[tuple[str, str], ...]:
        return _merge_hash_sets(
            self.required_source_image_hashes,
            tuple((path, digest) for _, path, digest in self.manifests),
            tuple((document.member_path, document.sha256) for document in self.rights_documents),
            tuple((document.member_path, document.sha256) for document in self.engine_evidence),
        )

    @property
    def projected_occurrence_link_count(self) -> int:
        fixed = 1 + len(self.rights_documents) + len(self.engine_evidence)
        return sum(fixed + len(record.layers) for record in self.records)

    @property
    def deferred_transform_sequence_count(self) -> int:
        return sum(record.has_deferred_transform for record in self.records)

    @property
    def duplicate_track_content_groups(self) -> int:
        counts: dict[str, int] = {}
        for record in self.records:
            key = record.track_content_deduplication_key
            counts[key] = counts.get(key, 0) + 1
        return sum(count > 1 for count in counts.values())

    @property
    def duplicate_track_content_excess(self) -> int:
        counts: dict[str, int] = {}
        for record in self.records:
            key = record.track_content_deduplication_key
            counts[key] = counts.get(key, 0) + 1
        return sum(count - 1 for count in counts.values() if count > 1)

    @property
    def projection_version(self) -> str:
        return PROJECTION_VERSION

    @property
    def projection_manifest_sha256(self) -> str:
        encoded = self.canonical_json().encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively serializable and self-identifying plan."""

        return {"projection_version": PROJECTION_VERSION, **asdict(self)}

    def canonical_json(self) -> str:
        """Serialize the pure plan deterministically for review or archival."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class SuperTuxProjectionPreparation:
    """Query-only preparation plan; no extraction or registration is performed."""

    database_path: str
    archive_sha256: str
    projection_manifest_sha256: str
    source_registered: bool
    archive_inventory_present: bool
    archive_inventory_sha256: str | None
    archive_inventory_matches: bool
    source_item_count: int
    required_member_count: int
    present_member_count: int
    required_source_image_count: int
    present_extracted_source_image_count: int
    present_registered_source_image_count: int
    required_non_image_evidence_count: int
    present_verified_non_image_evidence_count: int
    missing_member_paths: tuple[str, ...]
    members_requiring_extraction: tuple[str, ...]
    blobs_requiring_registration: tuple[str, ...]
    source_image_hash_mismatches: tuple[str, ...]
    non_image_evidence_hash_mismatches: tuple[str, ...]
    next_steps: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.source_registered
            and self.archive_inventory_present
            and self.archive_inventory_matches
            and self.source_item_count > 0
            and not self.missing_member_paths
            and not self.members_requiring_extraction
            and not self.blobs_requiring_registration
            and not self.source_image_hash_mismatches
            and not self.non_image_evidence_hash_mismatches
            and self.present_verified_non_image_evidence_count
            == self.required_non_image_evidence_count
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# Readiness and preparation are intentionally the same complete, query-only
# fact set.  The alias keeps call sites semantically clear without duplicating
# a schema that could drift.
SuperTuxProjectionReadiness = SuperTuxProjectionPreparation


@dataclass(frozen=True, slots=True)
class SuperTuxProjectionResult:
    archive_sha256: str
    projection_manifest_sha256: str
    projected_entities: int
    projected_sequences: int
    projected_frames: int
    projected_animated_sequences: int
    projected_static_sequences: int
    runtime_controlled_sequences: int
    custom_finite_sequences: int
    custom_infinite_sequences: int
    deferred_transform_sequences: int
    created_sequences: int
    reused_sequences: int
    occurrence_links: int
    exclusions: int
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
    layers: Iterable[SuperTuxProjectionLayer],
) -> tuple[tuple[str, str], ...]:
    values: dict[str, str] = {}
    for layer in layers:
        previous = values.setdefault(layer.member_path, layer.sha256)
        if previous != layer.sha256:
            raise ValueError(
                f"One SuperTux source member has multiple audited hashes: {layer.member_path!r}"
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
                raise ValueError(f"Conflicting SuperTux evidence hash: {member_path!r}")
    return tuple(sorted(values.items()))


def _entity_external_key(audit: SuperTuxArchiveAudit, manifest: ManifestRecord) -> str:
    return _stable_json_key(
        "supertux_entity",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "manifest_id": manifest.manifest_id,
            "manifest_path": manifest.logical_path,
            "manifest_sha256": manifest.sha256,
        },
    )


def _track_source_key(
    audit: SuperTuxArchiveAudit,
    manifest: ManifestRecord,
    action: ActionRecord,
) -> str:
    return _stable_json_key(
        "supertux_track",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "manifest_path": manifest.logical_path,
            "manifest_sha256": manifest.sha256,
            "declaration_ordinal": action.declaration_ordinal,
            "line_number": action.line_number,
            "declared_name": action.name,
        },
    )


def _appearance_variant_key(audit: SuperTuxArchiveAudit, manifest: ManifestRecord) -> str:
    return _stable_json_key(
        "supertux_appearance",
        {
            "archive_sha256": audit.archive_sha256,
            "repository_commit": audit.commit,
            "manifest_id": manifest.manifest_id,
            "manifest_path": manifest.logical_path,
            "manifest_sha256": manifest.sha256,
        },
    )


def _projection_entity(
    audit: SuperTuxArchiveAudit, manifest: ManifestRecord
) -> SuperTuxProjectionEntity:
    normalized = _ENTITY_CLASS_MAP.get(manifest.entity_class)
    if normalized is None:
        raise ValueError(f"Unmapped SuperTux complete-entity class: {manifest.entity_class!r}")
    return SuperTuxProjectionEntity(
        entity_external_key=_entity_external_key(audit, manifest),
        manifest_id=manifest.manifest_id,
        entity_group=manifest.entity_group,
        display_name=manifest.display_name,
        adapter_entity_class=manifest.entity_class,
        normalized_entity_class=normalized,
        entity_class_basis=manifest.entity_class_basis,
        manifest_logical_path=manifest.logical_path,
        manifest_member_path=manifest.member_path,
        manifest_sha256=manifest.sha256,
        role=manifest.role,
        role_basis=manifest.role_basis,
        manifest_quarantine_reasons=manifest.quarantine_reasons,
    )


def _loop_mode(action: ActionRecord) -> LoopMode:
    if not action.has_custom_loops:
        return "runtime_controlled"
    if action.effective_loops < 0:
        return "engine_custom_infinite"
    if action.effective_loops == 0:
        return "engine_custom_terminal"
    return "engine_custom_finite"


def _action_taxonomy(action: ActionRecord, taxonomy: Taxonomy) -> tuple[str, str, str]:
    normalized = taxonomy.normalize_action(action.normalized_action)
    if normalized.value == "unknown" and action.normalized_action != "unknown":
        # Unknown is reserved for missing/ambiguous input.  These labels are
        # known exact SuperTux facts outside the shared vocabulary, so retain
        # the source label and map only the shared conditioning slot to other.
        return "other", f"{normalized.method}:explicit_other_preserve_source", "other"
    return (
        normalized.value,
        normalized.method,
        taxonomy.action_to_family.get(normalized.value, "other"),
    )


def _canonical_direction(action: ActionRecord, taxonomy: Taxonomy) -> str:
    value = "none" if action.direction is None else action.direction
    normalized = taxonomy.normalize_direction(value)
    if normalized.value == "unknown" and value != "unknown":
        raise ValueError(f"Unmapped SuperTux direction: {value!r}")
    return normalized.value


def _layers_and_frames(
    action: ActionRecord,
) -> tuple[tuple[SuperTuxProjectionLayer, ...], tuple[SuperTuxProjectionFrame, ...]]:
    layers: list[SuperTuxProjectionLayer] = []
    layer_indices: dict[str, int] = {}
    frames: list[SuperTuxProjectionFrame] = []
    for frame in action.frames:
        layer_index: int | None = None
        if frame.exists:
            if (
                frame.member_path is None
                or frame.sha256 is None
                or frame.size_bytes is None
                or frame.width is None
                or frame.height is None
                or frame.mode is None
                or frame.alpha_kind is None
            ):
                raise ValueError(f"Incomplete existing SuperTux frame facts: {frame.logical_path}")
            layer_index = layer_indices.get(frame.member_path)
            if layer_index is None:
                layer_index = len(layers)
                layer_indices[frame.member_path] = layer_index
                layers.append(
                    SuperTuxProjectionLayer(
                        logical_path=frame.logical_path,
                        member_path=frame.member_path,
                        sha256=frame.sha256,
                        size_bytes=frame.size_bytes,
                        width=frame.width,
                        height=frame.height,
                        mode=frame.mode,
                        alpha_kind=frame.alpha_kind,
                        payload_deduplication_key=f"sha256:{frame.sha256}",
                    )
                )
        occurrence_key = _stable_json_key(
            "supertux_frame_occurrence",
            {
                "ordinal": frame.ordinal,
                "sha256": frame.sha256,
                "width": frame.width,
                "height": frame.height,
                "transform": frame.transform,
                "duration_milliseconds": action.frame_duration_milliseconds,
            },
        )
        frames.append(
            SuperTuxProjectionFrame(
                ordinal=frame.ordinal,
                source_layer_index=layer_index,
                source_frame_index=0,
                duration_milliseconds=action.frame_duration_milliseconds,
                origin_action=frame.origin_action,
                requested_path=frame.requested_path,
                logical_path=frame.logical_path,
                member_path=frame.member_path,
                exists=frame.exists,
                width=frame.width,
                height=frame.height,
                transform=frame.transform,
                temporal_occurrence_key=occurrence_key,
            )
        )
    return tuple(layers), tuple(frames)


def _track_content_key(
    action: ActionRecord,
    layers: tuple[SuperTuxProjectionLayer, ...],
    frames: tuple[SuperTuxProjectionFrame, ...],
) -> str:
    return _stable_json_key(
        "supertux_track_content",
        {
            # Provenance paths deliberately do not enter the byte-content key.
            # They remain in layers/frames and archive occurrences.  This key
            # can therefore identify exact duplicate timelines without
            # collapsing their entity, manifest, or attribution identities.
            "frames": [
                {
                    "sha256": (
                        layers[frame.source_layer_index].sha256
                        if frame.source_layer_index is not None
                        else None
                    ),
                    "source_frame_index": frame.source_frame_index,
                    "duration_milliseconds": frame.duration_milliseconds,
                    "width": frame.width,
                    "height": frame.height,
                    "transform": frame.transform,
                    "exists": frame.exists,
                }
                for frame in frames
            ],
            "effective_fps": action.effective_fps,
            "frame_duration_milliseconds": action.frame_duration_milliseconds,
            "effective_loops": action.effective_loops,
            "has_custom_loops": action.has_custom_loops,
            "effective_loop_frame": action.effective_loop_frame,
        },
    )


def _exclusion_reasons(manifest: ManifestRecord, action: ActionRecord) -> tuple[str, ...]:
    reasons: list[str] = []
    if not manifest.complete_entity:
        reasons.append(f"manifest:role:{manifest.role}")
        reasons.extend(f"manifest:{reason}" for reason in manifest.quarantine_reasons)
    if not action.effective_declaration:
        reasons.append("action:superseded_duplicate_declaration")
    if not action.exact_source_sequence:
        reasons.append("action:not_exact_source_sequence")
    if not action.frames:
        reasons.append("action:no_source_frames")
    reasons.extend(f"action:{reason}" for reason in action.quarantine_reasons)
    return tuple(dict.fromkeys(reasons))


def _projection_record(
    audit: SuperTuxArchiveAudit,
    manifest: ManifestRecord,
    action: ActionRecord,
    taxonomy: Taxonomy,
    layers: tuple[SuperTuxProjectionLayer, ...],
    frames: tuple[SuperTuxProjectionFrame, ...],
) -> SuperTuxProjectionRecord:
    normalized_action, normalized_basis, action_family = _action_taxonomy(action, taxonomy)
    dimensions = {(layer.width, layer.height) for layer in layers}
    return SuperTuxProjectionRecord(
        sequence_source_key=_track_source_key(audit, manifest, action),
        track_content_deduplication_key=_track_content_key(action, layers, frames),
        appearance_variant_key=_appearance_variant_key(audit, manifest),
        entity=_projection_entity(audit, manifest),
        declaration_ordinal=action.declaration_ordinal,
        manifest_line_number=action.line_number,
        declared_name=action.name,
        adapter_normalized_action=action.normalized_action,
        adapter_normalized_action_basis=action.normalized_action_basis,
        normalized_action=normalized_action,
        normalized_action_basis=normalized_basis,
        action_family=action_family,
        source_direction=action.direction,
        canonical_direction=_canonical_direction(action, taxonomy),
        direction_basis=action.direction_basis,
        action_stem=action.action_stem,
        alias_kind=action.alias_kind,
        alias_target=action.alias_target,
        alias_chain=action.alias_chain,
        declared_image_paths=action.declared_image_paths,
        declared_fps=action.declared_fps,
        effective_fps=action.effective_fps,
        frame_duration_milliseconds=action.frame_duration_milliseconds,
        declared_loops=action.declared_loops,
        effective_loops=action.effective_loops,
        has_custom_loops=action.has_custom_loops,
        declared_loop_frame=action.declared_loop_frame,
        effective_loop_frame=action.effective_loop_frame,
        loop_start_ordinal=action.effective_loop_frame - 1,
        loop_mode=_loop_mode(action),
        hitbox=action.hitbox,
        unisolid=action.unisolid,
        family_name=action.family_name,
        layers=layers,
        frames=frames,
        variable_frame_geometry=len(dimensions) > 1,
        runtime_composite_status="exact_complete_entity_transform_recipe",
        exact_runtime_source_recipe=True,
    )


def _projection_exclusion(
    audit: SuperTuxArchiveAudit,
    manifest: ManifestRecord,
    action: ActionRecord,
    layers: tuple[SuperTuxProjectionLayer, ...],
    frames: tuple[SuperTuxProjectionFrame, ...],
    reasons: tuple[str, ...],
) -> SuperTuxProjectionExclusion:
    return SuperTuxProjectionExclusion(
        track_source_key=_track_source_key(audit, manifest, action),
        track_content_deduplication_key=_track_content_key(action, layers, frames),
        appearance_variant_key=_appearance_variant_key(audit, manifest),
        manifest_id=manifest.manifest_id,
        entity_group=manifest.entity_group,
        display_name=manifest.display_name,
        role=manifest.role,
        role_basis=manifest.role_basis,
        parent_entity_hint=manifest.parent_entity_hint,
        adapter_entity_class=manifest.entity_class,
        entity_class_basis=manifest.entity_class_basis,
        manifest_logical_path=manifest.logical_path,
        manifest_member_path=manifest.member_path,
        manifest_sha256=manifest.sha256,
        manifest_quarantine_reasons=manifest.quarantine_reasons,
        declaration_ordinal=action.declaration_ordinal,
        manifest_line_number=action.line_number,
        declared_name=action.name,
        adapter_normalized_action=action.normalized_action,
        adapter_normalized_action_basis=action.normalized_action_basis,
        source_direction=action.direction,
        direction_basis=action.direction_basis,
        action_stem=action.action_stem,
        alias_kind=action.alias_kind,
        alias_target=action.alias_target,
        alias_chain=action.alias_chain,
        declared_image_paths=action.declared_image_paths,
        declared_fps=action.declared_fps,
        effective_fps=action.effective_fps,
        frame_duration_milliseconds=action.frame_duration_milliseconds,
        declared_loops=action.declared_loops,
        effective_loops=action.effective_loops,
        has_custom_loops=action.has_custom_loops,
        declared_loop_frame=action.declared_loop_frame,
        effective_loop_frame=action.effective_loop_frame,
        hitbox=action.hitbox,
        unisolid=action.unisolid,
        family_name=action.family_name,
        effective_declaration=action.effective_declaration,
        exact_source_sequence=action.exact_source_sequence,
        layers=layers,
        frames=frames,
        action_quarantine_reasons=action.quarantine_reasons,
        reasons=reasons,
    )


def plan_supertux_projection(
    audit: SuperTuxArchiveAudit,
    taxonomy: Taxonomy,
) -> SuperTuxProjectionPlan:
    """Partition every audited action declaration into output or quarantine."""

    records: list[SuperTuxProjectionRecord] = []
    exclusions: list[SuperTuxProjectionExclusion] = []
    audited_action_count = 0
    audited_frame_count = 0
    for manifest in audit.manifests:
        for action in manifest.actions:
            audited_action_count += 1
            audited_frame_count += len(action.frames)
            layers, frames = _layers_and_frames(action)
            reasons = _exclusion_reasons(manifest, action)
            if reasons:
                exclusions.append(
                    _projection_exclusion(audit, manifest, action, layers, frames, reasons)
                )
            else:
                if not layers or any(frame.source_layer_index is None for frame in frames):
                    raise AssertionError("Exact SuperTux track lacks complete source layers")
                records.append(
                    _projection_record(audit, manifest, action, taxonomy, layers, frames)
                )
    records.sort(key=lambda record: record.sequence_source_key)
    exclusions.sort(key=lambda exclusion: exclusion.track_source_key)
    if len(records) + len(exclusions) != audited_action_count:
        raise AssertionError("SuperTux plan does not partition every action declaration")
    if (
        sum(record.frame_count for record in records)
        + sum(exclusion.frame_count for exclusion in exclusions)
        != audited_frame_count
    ):
        raise AssertionError("SuperTux plan does not partition every declared frame occurrence")
    record_keys = [record.sequence_source_key for record in records]
    exclusion_keys = [row.track_source_key for row in exclusions]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("SuperTux projected track keys are not unique")
    if len(exclusion_keys) != len(set(exclusion_keys)):
        raise ValueError("SuperTux exclusion track keys are not unique")
    if set(record_keys).intersection(exclusion_keys):
        raise ValueError("SuperTux track appears in projection and exclusion ledgers")
    if any(record.entity.role != "complete_entity" for record in records):
        raise AssertionError("Non-complete SuperTux manifest reached exact projection")
    return SuperTuxProjectionPlan(
        archive_sha256=audit.archive_sha256,
        repository_commit=audit.commit,
        archive_root=audit.archive_root,
        source_inventory_sha256=audit.inventory_sha256,
        source_audit_record_sha256=audit.audit_record_sha256,
        taxonomy_version=taxonomy.version,
        taxonomy_action_values=tuple(sorted(taxonomy.action_to_family)),
        taxonomy_action_families=tuple(sorted(taxonomy.action_to_family.items())),
        taxonomy_entity_aliases=tuple(sorted(taxonomy.entity_aliases.items())),
        taxonomy_direction_values=tuple(sorted(taxonomy.directions)),
        taxonomy_entity_values=tuple(sorted(taxonomy.entity_classes)),
        records=tuple(records),
        exclusions=tuple(exclusions),
        manifests=tuple(
            sorted(
                (manifest.logical_path, manifest.member_path, manifest.sha256)
                for manifest in audit.manifests
            )
        ),
        rights=audit.rights,
        acquisition_evidence=audit.acquisition_evidence,
        engine_evidence=audit.engine_evidence,
    )


def plan_known_supertux_projection(
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> SuperTuxProjectionPlan:
    """Audit the exact pinned CAS snapshot and enforce reviewed regression facts."""

    plan = plan_supertux_projection(audit_known_supertux_archive(Path(archive_path)), taxonomy)
    expected = (
        EXPECTED_PINNED_PROJECTED_SEQUENCE_COUNT,
        EXPECTED_PINNED_PROJECTED_FRAME_COUNT,
        EXPECTED_PINNED_PROJECTED_ENTITY_COUNT,
        EXPECTED_PINNED_EXCLUSION_COUNT,
        EXPECTED_PINNED_EXCLUDED_FRAME_COUNT,
        EXPECTED_PINNED_REQUIRED_MEMBER_COUNT,
        EXPECTED_PINNED_REQUIRED_SOURCE_IMAGE_COUNT,
    )
    actual = (
        plan.projected_sequence_count,
        plan.projected_frame_count,
        plan.projected_entity_count,
        len(plan.exclusions),
        plan.excluded_frame_count,
        len(plan.required_member_paths),
        len(plan.required_source_image_hashes),
    )
    if actual != expected:
        raise ValueError(
            f"Pinned SuperTux projection count drift: expected {expected}, got {actual}"
        )
    if plan.projection_manifest_sha256 != EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256:
        raise ValueError(
            "Pinned SuperTux projection manifest drift: expected "
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


def plan_supertux_projection_preparation(
    database_path: str | Path,
    plan: SuperTuxProjectionPlan,
) -> SuperTuxProjectionPreparation:
    """Return exact non-mutating extraction/registration work still required."""

    required_paths = plan.required_member_paths
    expected_images = dict(plan.required_source_image_hashes)
    expected_evidence = dict(plan.required_evidence_hashes)
    expected_non_images = {
        path: digest for path, digest in expected_evidence.items() if path not in expected_images
    }
    with _readonly_connection(database_path) as connection:
        source_registered = (
            connection.execute("SELECT 1 FROM sources WHERE id=? LIMIT 1", (SOURCE_ID,)).fetchone()
            is not None
        )
        inventory_row = connection.execute(
            "SELECT inventory_sha256 FROM archive_inventories WHERE archive_blob_sha256=? LIMIT 1",
            (plan.archive_sha256,),
        ).fetchone()
        archive_inventory_present = inventory_row is not None
        archive_inventory_sha256 = (
            str(inventory_row["inventory_sha256"]) if inventory_row is not None else None
        )
        archive_inventory_matches = archive_inventory_sha256 == plan.source_inventory_sha256
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
        registered_digests = {
            str(row[0]) for row in connection.execute("SELECT sha256 FROM blobs").fetchall()
        }
    members: dict[str, sqlite3.Row] = {}
    for row in rows:
        members[str(row["normalized_path"])] = row
        members[str(row["member_path"])] = row
    missing_paths = tuple(path for path in required_paths if path not in members)
    extraction: list[str] = []
    registration: list[str] = []
    mismatches: list[str] = []
    evidence_mismatches: list[str] = []
    present_extracted = 0
    present_registered = 0
    present_verified_evidence = 0
    for member_path, expected_hash in sorted(expected_images.items()):
        row = members.get(member_path)
        if row is None:
            extraction.append(member_path)
            if expected_hash not in registered_digests:
                registration.append(member_path)
            continue
        actual_hash = row["extracted_blob_sha256"]
        if actual_hash is None:
            extraction.append(member_path)
        elif str(actual_hash) != expected_hash:
            mismatches.append(f"{member_path}: expected {expected_hash}, indexed {actual_hash}")
            extraction.append(member_path)
        else:
            present_extracted += 1
        if expected_hash not in registered_digests:
            registration.append(member_path)
        else:
            present_registered += 1
    for member_path, expected_hash in sorted(expected_non_images.items()):
        row = members.get(member_path)
        if row is None:
            extraction.append(member_path)
            if expected_hash not in registered_digests:
                registration.append(member_path)
            continue
        actual_hash = row["extracted_blob_sha256"]
        if actual_hash is None:
            extraction.append(member_path)
        elif str(actual_hash) != expected_hash:
            evidence_mismatches.append(
                f"{member_path}: expected {expected_hash}, indexed {actual_hash}"
            )
            extraction.append(member_path)
        if expected_hash not in registered_digests:
            registration.append(member_path)
        elif actual_hash is not None and str(actual_hash) == expected_hash:
            present_verified_evidence += 1
    steps: list[str] = []
    if not source_registered:
        steps.append(f"register source {SOURCE_ID!r}")
    if source_item_count == 0:
        steps.append("register and link the pinned source archive item")
    if not archive_inventory_present:
        steps.append("index the pinned ZIP central-directory inventory")
    elif not archive_inventory_matches:
        steps.append("investigate the pinned ZIP inventory digest mismatch")
    if missing_paths:
        steps.append(f"index {len(missing_paths)} required archive members")
    if extraction:
        steps.append(f"guardedly extract {len(extraction)} audited evidence members into CAS")
    if registration:
        steps.append(f"register {len(registration)} audited evidence blobs")
    if mismatches:
        steps.append(f"investigate {len(mismatches)} immutable hash mismatches")
    if evidence_mismatches:
        steps.append(f"investigate {len(evidence_mismatches)} non-image evidence hash mismatches")
    return SuperTuxProjectionPreparation(
        database_path=str(Path(database_path).resolve()),
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=plan.projection_manifest_sha256,
        source_registered=source_registered,
        archive_inventory_present=archive_inventory_present,
        archive_inventory_sha256=archive_inventory_sha256,
        archive_inventory_matches=archive_inventory_matches,
        source_item_count=source_item_count,
        required_member_count=len(required_paths),
        present_member_count=len(required_paths) - len(missing_paths),
        required_source_image_count=len(expected_images),
        present_extracted_source_image_count=present_extracted,
        present_registered_source_image_count=present_registered,
        required_non_image_evidence_count=len(expected_non_images),
        present_verified_non_image_evidence_count=present_verified_evidence,
        missing_member_paths=missing_paths,
        members_requiring_extraction=tuple(extraction),
        blobs_requiring_registration=tuple(registration),
        source_image_hash_mismatches=tuple(mismatches),
        non_image_evidence_hash_mismatches=tuple(evidence_mismatches),
        next_steps=tuple(steps),
    )


def check_supertux_projection_readiness(
    database_path: str | Path,
    plan: SuperTuxProjectionPlan,
) -> SuperTuxProjectionReadiness:
    """Compatibility name for the same query-only complete preparation plan."""

    return plan_supertux_projection_preparation(database_path, plan)


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
            f"SuperTux archive has no indexed source item for {SOURCE_ID!r}: {archive_sha256}"
        )
    return str(row[0])


def _preflight(
    database: IndexDB,
    plan: SuperTuxProjectionPlan,
) -> tuple[str, dict[str, sqlite3.Row]]:
    with database.connect() as connection:
        source = connection.execute("SELECT 1 FROM sources WHERE id=?", (SOURCE_ID,)).fetchone()
        inventory = connection.execute(
            "SELECT inventory_sha256 FROM archive_inventories WHERE archive_blob_sha256=?",
            (plan.archive_sha256,),
        ).fetchone()
    if source is None:
        raise ValueError(f"SuperTux source registry row is missing: {SOURCE_ID}")
    if inventory is None:
        raise ValueError(f"SuperTux archive inventory is missing: {plan.archive_sha256}")
    if str(inventory["inventory_sha256"]) != plan.source_inventory_sha256:
        raise ValueError(
            "SuperTux archive inventory hash mismatch: expected "
            f"{plan.source_inventory_sha256}, indexed {inventory['inventory_sha256']}"
        )
    item_id = _item_id(database, plan.archive_sha256)
    members = _archive_members(database, plan.archive_sha256)
    missing = [path for path in plan.required_member_paths if path not in members]
    if missing:
        raise ValueError(
            "SuperTux projection evidence members are missing: " + ", ".join(missing[:10])
        )
    source_image_paths = {path for path, _ in plan.required_source_image_hashes}
    for member_path, expected_hash in plan.required_evidence_hashes:
        member = members[member_path]
        actual_hash = member["extracted_blob_sha256"]
        if actual_hash is None:
            raise ValueError(f"SuperTux evidence member is not extracted into CAS: {member_path}")
        if str(actual_hash) != expected_hash:
            raise ValueError(
                "SuperTux evidence CAS hash mismatch for "
                f"{member_path}: expected {expected_hash}, indexed {actual_hash}"
            )
        if member["registered_blob_sha256"] is None:
            role = "source image" if member_path in source_image_paths else "non-image evidence"
            raise ValueError(f"SuperTux {role} CAS blob is not registered: {member_path}")
    return item_id, members


def _require_initialized_database(database: IndexDB) -> None:
    """Fail without creating or migrating a caller-supplied database."""

    database_path = database.path
    if not database_path.is_file():
        raise ValueError(
            f"SuperTux projection requires an existing initialized index database: {database_path}"
        )
    try:
        with _readonly_connection(database_path) as connection:
            versions = {
                int(row[0])
                for row in connection.execute(
                    "SELECT version FROM schema_migrations ORDER BY version"
                ).fetchall()
            }
            required_tables = {
                "sources",
                "items",
                "item_blobs",
                "blobs",
                "entities",
                "sequences",
                "sequence_source_keys",
                "sequence_subjects",
                "motion_annotations",
                "archive_inventories",
                "archive_members",
                "sequence_occurrences",
                "sequence_frames",
            }
            present_tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
    except sqlite3.Error as error:
        raise ValueError(
            "SuperTux projection requires a compatible initialized index database"
        ) from error
    if not set(range(1, 6)).issubset(versions):
        raise ValueError(
            "SuperTux projection requires schema migrations 1 through 5; "
            f"observed {sorted(versions)}"
        )
    missing_tables = sorted(required_tables - present_tables)
    if missing_tables:
        raise ValueError(
            "SuperTux projection index schema is missing tables: " + ", ".join(missing_tables)
        )


def _require_readonly_preflight(
    database_path: str | Path,
    plan: SuperTuxProjectionPlan,
) -> None:
    """Complete prerequisite verification before opening a write connection."""

    preparation = plan_supertux_projection_preparation(database_path, plan)
    if not preparation.source_registered:
        raise ValueError(f"SuperTux source registry row is missing: {SOURCE_ID}")
    if preparation.source_item_count == 0:
        raise ValueError(
            f"SuperTux archive has no indexed source item for {SOURCE_ID!r}: {plan.archive_sha256}"
        )
    if not preparation.archive_inventory_present:
        raise ValueError(f"SuperTux archive inventory is missing: {plan.archive_sha256}")
    if not preparation.archive_inventory_matches:
        raise ValueError(
            "SuperTux archive inventory hash mismatch: expected "
            f"{plan.source_inventory_sha256}, indexed "
            f"{preparation.archive_inventory_sha256}"
        )
    if preparation.missing_member_paths:
        raise ValueError(
            "SuperTux projection evidence members are missing: "
            + ", ".join(preparation.missing_member_paths[:10])
        )
    if preparation.source_image_hash_mismatches:
        raise ValueError(
            "SuperTux source image CAS hash mismatch: "
            + preparation.source_image_hash_mismatches[0]
        )
    if preparation.non_image_evidence_hash_mismatches:
        raise ValueError(
            "SuperTux evidence CAS hash mismatch: "
            + preparation.non_image_evidence_hash_mismatches[0]
        )
    if preparation.members_requiring_extraction:
        raise ValueError(
            "SuperTux evidence member is not extracted into CAS: "
            + preparation.members_requiring_extraction[0]
        )
    if preparation.blobs_requiring_registration:
        raise ValueError(
            "SuperTux evidence CAS blob is not registered: "
            + preparation.blobs_requiring_registration[0]
        )
    if not preparation.ready:
        raise ValueError("SuperTux projection preparation is unexpectedly incomplete")


def _rights_metadata(plan: SuperTuxProjectionPlan) -> dict[str, Any]:
    return {
        "license_expression": plan.rights.repository_license_expression,
        "repository_license_expression": plan.rights.repository_license_expression,
        "license_basis": plan.rights.license_basis,
        "attribution_summary": plan.rights.attribution_summary,
        "caveat": plan.rights.caveat,
        "documents": [asdict(document) for document in plan.rights_documents],
        "per_file_license_or_creator_mapping_claimed": False,
        "rights_observation_added": False,
    }


def _loop_semantics_metadata(record: SuperTuxProjectionRecord) -> dict[str, Any]:
    if record.loop_mode == "runtime_controlled":
        interpretation = (
            "No custom loop count is declared; the pinned engine accepts a caller-supplied "
            "loop policy, so loopability is intentionally not inferred."
        )
    else:
        interpretation = (
            "Custom engine loop value is preserved verbatim; no conversion to a generic "
            "repeat-count convention is asserted."
        )
    return {
        "declared_loops": record.declared_loops,
        "effective_loops": record.effective_loops,
        "has_custom_loops": record.has_custom_loops,
        "declared_loop_frame": record.declared_loop_frame,
        "effective_loop_frame_1_based": record.effective_loop_frame,
        "loop_start_ordinal_0_based": record.loop_start_ordinal,
        "loop_mode": record.loop_mode,
        "loopable_annotation": record.loopable,
        "cycle_frames_when_engine_infinite": record.cycle_frames,
        "policy_inferred": False,
        "interpretation": interpretation,
    }


def _sequence_metadata(
    plan: SuperTuxProjectionPlan,
    record: SuperTuxProjectionRecord,
    manifest_sha256: str,
) -> dict[str, Any]:
    model_ready_reasons = ["supertux_engine_loop_mode_not_fixed_phase_normalized"]
    if record.has_deferred_transform:
        model_ready_reasons.append("geometric_transform_materializer_not_implemented")
    return {
        "projection_version": PROJECTION_VERSION,
        "projection_manifest_sha256": manifest_sha256,
        "source_inventory_sha256": plan.source_inventory_sha256,
        "source_audit_record_sha256": plan.source_audit_record_sha256,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "sequence_source_key": record.sequence_source_key,
        "track_content_deduplication_key": record.track_content_deduplication_key,
        "appearance_variant_key": record.appearance_variant_key,
        "manifest_id": record.entity.manifest_id,
        "entity_group": record.entity.entity_group,
        "display_name": record.entity.display_name,
        "manifest_logical_path": record.entity.manifest_logical_path,
        "manifest_member_path": record.entity.manifest_member_path,
        "manifest_sha256": record.entity.manifest_sha256,
        "manifest_role": record.entity.role,
        "manifest_role_basis": record.entity.role_basis,
        "manifest_quarantine_reasons": list(record.entity.manifest_quarantine_reasons),
        "declaration_ordinal": record.declaration_ordinal,
        "manifest_line_number": record.manifest_line_number,
        "declared_name": record.declared_name,
        "adapter_normalized_action": record.adapter_normalized_action,
        "adapter_normalized_action_basis": record.adapter_normalized_action_basis,
        "normalized_action": record.normalized_action,
        "normalized_action_basis": record.normalized_action_basis,
        "action_family": record.action_family,
        "source_direction": record.source_direction,
        "canonical_direction": record.canonical_direction,
        "direction_basis": record.direction_basis,
        "action_stem": record.action_stem,
        "alias_kind": record.alias_kind,
        "alias_target": record.alias_target,
        "alias_chain": list(record.alias_chain),
        "declared_image_paths": list(record.declared_image_paths),
        "declared_fps": record.declared_fps,
        "effective_fps": record.effective_fps,
        "frame_duration_milliseconds": record.frame_duration_milliseconds,
        "duration_ms_per_occurrence": [frame.duration_milliseconds for frame in record.frames],
        "total_duration_ms": record.total_duration_milliseconds,
        "loop_semantics": _loop_semantics_metadata(record),
        "hitbox": list(record.hitbox),
        "unisolid": record.unisolid,
        "family_name": record.family_name,
        "source_layers": [asdict(layer) for layer in record.layers],
        "source_image_hash_order": [
            record.layers[frame.source_layer_index].sha256
            for frame in record.frames
            if frame.source_layer_index is not None
        ],
        "source_transform_order": [frame.transform for frame in record.frames],
        "source_frame_index_order": [frame.source_frame_index for frame in record.frames],
        "native_frame_geometry": [
            {"width": frame.width, "height": frame.height} for frame in record.frames
        ],
        "sequence_geometry_policy": "max_native_frame_envelope",
        "variable_frame_geometry": record.variable_frame_geometry,
        "runtime_composite_status": record.runtime_composite_status,
        "exact_runtime_source_recipe": record.exact_runtime_source_recipe,
        "transforms_materialized_in_source_blobs": False,
        "transform_materialization_required": record.has_deferred_transform,
        "current_canonical_materializer_compatible": not record.has_deferred_transform,
        "required_geometric_transform_operations": sorted(
            {frame.transform for frame in record.frames if frame.transform != "identity"}
        ),
        "model_ready_materialization_eligible": False,
        "model_ready_exclusion_reasons": model_ready_reasons,
        "frame_order_preserved": True,
        "exact_engine_timing": True,
        "clipping_alignment_or_repair_applied": False,
        "rights_scope": _rights_metadata(plan),
        "acquisition_evidence": [asdict(row) for row in plan.acquisition_evidence],
        "engine_evidence": [asdict(document) for document in plan.engine_evidence],
    }


def _occurrence_specs(
    plan: SuperTuxProjectionPlan,
    record: SuperTuxProjectionRecord,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    common = {
        "projection_version": PROJECTION_VERSION,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "manifest_id": record.entity.manifest_id,
        "declared_name": record.declared_name,
        "declaration_ordinal": record.declaration_ordinal,
        "manifest_line_number": record.manifest_line_number,
        "sequence_source_key": record.sequence_source_key,
    }
    specs: list[tuple[str, str, dict[str, Any]]] = []
    for index, layer in enumerate(record.layers):
        occurrences = [
            {
                "ordinal": frame.ordinal,
                "transform": frame.transform,
                "origin_action": frame.origin_action,
                "requested_path": frame.requested_path,
            }
            for frame in record.frames
            if frame.source_layer_index == index
        ]
        specs.append(
            (
                layer.member_path,
                "supertux_complete_entity_source_image",
                {
                    **common,
                    "layer": asdict(layer),
                    "temporal_occurrences": occurrences,
                    "exact_runtime_source_recipe": True,
                    "transforms_materialized_in_source_blob": False,
                },
            )
        )
    specs.append(
        (
            record.entity.manifest_member_path,
            "supertux_sprite_manifest",
            {
                **common,
                "manifest_logical_path": record.entity.manifest_logical_path,
                "manifest_sha256": record.entity.manifest_sha256,
                "manifest_role": record.entity.role,
                "manifest_role_basis": record.entity.role_basis,
                "manifest_quarantine_reasons": list(record.entity.manifest_quarantine_reasons),
                "source_audit_record_sha256": plan.source_audit_record_sha256,
            },
        )
    )
    specs.extend(
        (
            document.member_path,
            "supertux_collection_rights_and_credits_evidence",
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
            "supertux_engine_animation_semantics",
            {**common, "evidence": asdict(document)},
        )
        for document in plan.engine_evidence
    )
    return tuple(specs)


def _frame_metadata(
    plan: SuperTuxProjectionPlan,
    record: SuperTuxProjectionRecord,
    frame: SuperTuxProjectionFrame,
) -> dict[str, Any]:
    if frame.source_layer_index is None or frame.width is None or frame.height is None:
        raise ValueError("Quarantined SuperTux frame reached exact DB projection")
    layer = record.layers[frame.source_layer_index]
    return {
        "projection_version": PROJECTION_VERSION,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "manifest_id": record.entity.manifest_id,
        "declared_name": record.declared_name,
        "declaration_ordinal": record.declaration_ordinal,
        "manifest_line_number": record.manifest_line_number,
        "adapter_normalized_action": record.adapter_normalized_action,
        "normalized_action": record.normalized_action,
        "source_direction": record.source_direction,
        "canonical_direction": record.canonical_direction,
        "source_layer": asdict(layer),
        "source_frame_index": frame.source_frame_index,
        "sequence_ordinal": frame.ordinal,
        "duration_milliseconds": frame.duration_milliseconds,
        "origin_action": frame.origin_action,
        "requested_path": frame.requested_path,
        "logical_path": frame.logical_path,
        "member_path": frame.member_path,
        "frame_rect": {
            "left": 0,
            "top": 0,
            "right": frame.width,
            "bottom": frame.height,
            "width": frame.width,
            "height": frame.height,
            "coordinate_space": "source_image",
        },
        "source_transform": frame.transform,
        "transform_recipe": {
            "operation": frame.transform,
            "apply_to_source_pixels": frame.transform != "identity",
            "materialized_in_source_blob": False,
            "lossless": True,
        },
        "current_canonical_materializer_compatible": frame.transform == "identity",
        "model_ready_materialization_eligible": False,
        "model_ready_exclusion_reasons": [
            "supertux_engine_loop_mode_not_fixed_phase_normalized",
            *(
                ["geometric_transform_materializer_not_implemented"]
                if frame.transform != "identity"
                else []
            ),
        ],
        "temporal_occurrence_key": frame.temporal_occurrence_key,
        "track_content_deduplication_key": record.track_content_deduplication_key,
        "loop_semantics": _loop_semantics_metadata(record),
        "exact_engine_timing": True,
        "native_source_rectangle": True,
        "clipping_alignment_or_repair_applied": False,
        "rights_scope": _rights_metadata(plan),
    }


def _validate_taxonomy_contract(plan: SuperTuxProjectionPlan, taxonomy: Taxonomy) -> None:
    if taxonomy.version != plan.taxonomy_version:
        raise ValueError(
            "SuperTux projection taxonomy version mismatch: "
            f"plan {plan.taxonomy_version!r}, runtime {taxonomy.version!r}"
        )
    if tuple(sorted(taxonomy.action_to_family)) != plan.taxonomy_action_values:
        raise ValueError("SuperTux projection taxonomy action vocabulary has changed")
    if tuple(sorted(taxonomy.action_to_family.items())) != plan.taxonomy_action_families:
        raise ValueError("SuperTux projection taxonomy action-family mapping has changed")
    if tuple(sorted(taxonomy.entity_aliases.items())) != plan.taxonomy_entity_aliases:
        raise ValueError("SuperTux projection taxonomy entity aliases have changed")
    if tuple(sorted(taxonomy.directions)) != plan.taxonomy_direction_values:
        raise ValueError("SuperTux projection taxonomy direction vocabulary has changed")
    if tuple(sorted(taxonomy.entity_classes)) != plan.taxonomy_entity_values:
        raise ValueError("SuperTux projection taxonomy entity vocabulary has changed")


def project_supertux_audit(
    database: IndexDB,
    plan: SuperTuxProjectionPlan,
    taxonomy: Taxonomy,
) -> SuperTuxProjectionResult:
    """Atomically and idempotently project a precomputed exact-source plan."""

    _validate_taxonomy_contract(plan, taxonomy)
    _require_initialized_database(database)
    _require_readonly_preflight(database.path, plan)
    manifest_sha256 = plan.projection_manifest_sha256
    rights_scope = _rights_metadata(plan)
    entity_ids: dict[str, str] = {}
    created_sequences = 0
    reused_sequences = 0
    occurrence_links = 0

    # IndexDB reuses this connection in every nested helper.  Complete
    # preflight precedes the first mutation, and any later exception rolls the
    # entire projection back rather than leaving a partial corpus projection.
    with database.transaction():
        item_id, members = _preflight(database, plan)
        for record in plan.records:
            if not record.exact_runtime_source_recipe or record.entity.role != "complete_entity":
                raise ValueError("Unsafe SuperTux track reached DB projection")
            entity = record.entity
            entity_id = entity_ids.get(entity.entity_external_key)
            if entity_id is None:
                normalized_entity = taxonomy.normalize_entity_class(entity.normalized_entity_class)
                if normalized_entity.value != entity.normalized_entity_class:
                    raise ValueError(
                        "SuperTux entity class was not taxonomy-canonical at write time: "
                        f"{entity.normalized_entity_class!r}"
                    )
                entity_id = database.upsert_entity(
                    source_id=SOURCE_ID,
                    external_identity_key=entity.entity_external_key,
                    representative_item_id=item_id,
                    display_name=entity.display_name,
                    entity_class=entity.normalized_entity_class,
                    entity_subclass=entity.adapter_entity_class,
                    species_or_type=entity.entity_group,
                    taxonomy_version=taxonomy.version,
                    metadata={
                        "projection_version": PROJECTION_VERSION,
                        "projection_manifest_sha256": manifest_sha256,
                        "source_inventory_sha256": plan.source_inventory_sha256,
                        "source_audit_record_sha256": plan.source_audit_record_sha256,
                        "archive_sha256": plan.archive_sha256,
                        "repository_commit": plan.repository_commit,
                        "manifest_id": entity.manifest_id,
                        "entity_group": entity.entity_group,
                        "display_name": entity.display_name,
                        "adapter_entity_class": entity.adapter_entity_class,
                        "normalized_entity_class": entity.normalized_entity_class,
                        "entity_class_basis": entity.entity_class_basis,
                        "classification_mapping": "explicit_supertux_adapter_to_taxonomy",
                        "classification_confidence": 1.0,
                        "manifest_logical_path": entity.manifest_logical_path,
                        "manifest_member_path": entity.manifest_member_path,
                        "manifest_sha256": entity.manifest_sha256,
                        "manifest_role": entity.role,
                        "manifest_role_basis": entity.role_basis,
                        "manifest_quarantine_reasons": list(entity.manifest_quarantine_reasons),
                        "complete_entity": True,
                        "rights_scope": rights_scope,
                    },
                )
                entity_ids[entity.entity_external_key] = entity_id

            motion = taxonomy.motion_condition(
                action=record.normalized_action,
                direction=record.canonical_direction,
                view="platformer",
            )
            if motion.normalized_action != record.normalized_action:
                raise ValueError(
                    "SuperTux action was not taxonomy-canonical at write time: "
                    f"{record.normalized_action!r}"
                )
            if motion.action_family != record.action_family:
                raise ValueError(
                    "SuperTux action family disagrees with the pure plan: "
                    f"{record.normalized_action!r}: plan {record.action_family!r}, "
                    f"runtime {motion.action_family!r}"
                )
            if motion.direction != record.canonical_direction or motion.view != "platformer":
                raise ValueError("SuperTux direction/view became non-canonical at write time")
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
                "quality_tier": (
                    "P0_exact_supertux_geometric_transform_materializer_required"
                    if record.has_deferred_transform
                    else "F0_lossless_supertux_exact_source_pixels"
                ),
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
                    "manifest_id": entity.manifest_id,
                    "entity_group": entity.entity_group,
                    "manifest_logical_path": entity.manifest_logical_path,
                    "manifest_line_number": record.manifest_line_number,
                    "appearance_variant_key": record.appearance_variant_key,
                    "complete_entity": True,
                    "manifest_role": entity.role,
                    "runtime_composite_status": record.runtime_composite_status,
                    "exact_runtime_source_recipe": True,
                    "transform_materialization_required": record.has_deferred_transform,
                    "model_ready_materialization_eligible": False,
                    "model_ready_exclusion_reasons": [
                        "supertux_engine_loop_mode_not_fixed_phase_normalized",
                        *(
                            ["geometric_transform_materializer_not_implemented"]
                            if record.has_deferred_transform
                            else []
                        ),
                    ],
                    "rights_scope": rights_scope,
                },
            )
            annotation_confidence = (
                0.5
                if "explicit_other_preserve_source" in record.normalized_action_basis
                else motion.confidence
            )
            database.annotate_motion(
                sequence_id=sequence_id,
                vocabulary_version=taxonomy.version,
                source_action=record.declared_name,
                normalized_action=motion.normalized_action,
                action_family=motion.action_family,
                annotation_method=PROJECTION_VERSION,
                view=motion.view,
                direction=motion.direction,
                loopable=record.loopable,
                cycle_frames=record.cycle_frames,
                phase_zero_frame=(record.loop_start_ordinal if record.loopable is True else None),
                confidence=annotation_confidence,
                conditioning={
                    "declared_name": record.declared_name,
                    "adapter_normalized_action": record.adapter_normalized_action,
                    "adapter_normalized_action_basis": (record.adapter_normalized_action_basis),
                    "normalized_action_basis": record.normalized_action_basis,
                    "taxonomy_normalization_method": motion.method,
                    "source_direction": record.source_direction,
                    "canonical_direction": record.canonical_direction,
                    "direction_basis": record.direction_basis,
                    "view": "platformer",
                    "appearance_variant_key": record.appearance_variant_key,
                    "action_stem": record.action_stem,
                    "alias_kind": record.alias_kind,
                    "alias_target": record.alias_target,
                    "alias_chain": list(record.alias_chain),
                    "hitbox": list(record.hitbox),
                    "unisolid": record.unisolid,
                    "family_name": record.family_name,
                    "timing_known": True,
                    "exact_engine_timing": True,
                    "effective_fps": record.effective_fps,
                    "frame_duration_milliseconds": record.frame_duration_milliseconds,
                    "duration_ms_per_occurrence": [
                        frame.duration_milliseconds for frame in record.frames
                    ],
                    "loop_semantics": _loop_semantics_metadata(record),
                    "runtime_composite_status": record.runtime_composite_status,
                    "exact_runtime_source_recipe": True,
                    "transform_materialization_required": record.has_deferred_transform,
                    "source_transform_order": [frame.transform for frame in record.frames],
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
                if frame.source_layer_index is None:
                    raise ValueError("Quarantined SuperTux frame reached DB projection")
                layer = record.layers[frame.source_layer_index]
                phase: float | None = None
                if (
                    record.loopable is True
                    and record.cycle_frames is not None
                    and frame.ordinal >= record.loop_start_ordinal
                ):
                    phase = (frame.ordinal - record.loop_start_ordinal) / record.cycle_frames
                database.add_sequence_frame(
                    sequence_id=sequence_id,
                    ordinal=frame.ordinal,
                    source_blob_sha256=layer.sha256,
                    source_frame_index=frame.source_frame_index,
                    duration_ms=frame.duration_milliseconds,
                    phase=phase,
                    direction=motion.direction,
                    view=motion.view,
                    metadata=_frame_metadata(plan, record, frame),
                )

    return SuperTuxProjectionResult(
        archive_sha256=plan.archive_sha256,
        projection_manifest_sha256=manifest_sha256,
        projected_entities=plan.projected_entity_count,
        projected_sequences=plan.projected_sequence_count,
        projected_frames=plan.projected_frame_count,
        projected_animated_sequences=plan.projected_animated_sequence_count,
        projected_static_sequences=plan.projected_static_sequence_count,
        runtime_controlled_sequences=plan.projected_runtime_controlled_count,
        custom_finite_sequences=plan.projected_custom_finite_count,
        custom_infinite_sequences=plan.projected_custom_infinite_count,
        deferred_transform_sequences=plan.deferred_transform_sequence_count,
        created_sequences=created_sequences,
        reused_sequences=reused_sequences,
        occurrence_links=occurrence_links,
        exclusions=len(plan.exclusions),
        excluded_frames=plan.excluded_frame_count,
    )


def ingest_known_supertux_sequences(
    database: IndexDB,
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> SuperTuxProjectionResult:
    """Audit and project only the exact pinned SuperTux snapshot."""

    plan = plan_known_supertux_projection(archive_path, taxonomy)
    if (
        plan.archive_sha256 != EXPECTED_SUPERTUX_ARCHIVE_SHA256
        or plan.repository_commit != SUPERTUX_COMMIT
    ):
        raise ValueError("Refusing SuperTux projection for an unexpected archive or commit")
    return project_supertux_audit(database, plan, taxonomy)


__all__ = [
    "EXPECTED_PINNED_EXCLUDED_FRAME_COUNT",
    "EXPECTED_PINNED_EXCLUSION_COUNT",
    "EXPECTED_PINNED_PROJECTED_ENTITY_COUNT",
    "EXPECTED_PINNED_PROJECTED_FRAME_COUNT",
    "EXPECTED_PINNED_PROJECTED_SEQUENCE_COUNT",
    "EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256",
    "EXPECTED_PINNED_REQUIRED_MEMBER_COUNT",
    "EXPECTED_PINNED_REQUIRED_SOURCE_IMAGE_COUNT",
    "PROJECTION_VERSION",
    "SOURCE_ID",
    "SuperTuxProjectionEntity",
    "SuperTuxProjectionExclusion",
    "SuperTuxProjectionFrame",
    "SuperTuxProjectionLayer",
    "SuperTuxProjectionPlan",
    "SuperTuxProjectionPreparation",
    "SuperTuxProjectionReadiness",
    "SuperTuxProjectionRecord",
    "SuperTuxProjectionResult",
    "check_supertux_projection_readiness",
    "ingest_known_supertux_sequences",
    "plan_known_supertux_projection",
    "plan_supertux_projection",
    "plan_supertux_projection_preparation",
    "project_supertux_audit",
]
