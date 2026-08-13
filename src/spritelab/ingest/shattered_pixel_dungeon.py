"""Project an audited Shattered Pixel Dungeon archive into the provenance DB.

This module deliberately separates the pure projection plan from database writes.
Only animations with one unambiguous source sheet and absolute source-sheet frame
rectangles are materialized.  In particular, HeroSprite tier-patch coordinates
remain audit evidence rather than being misrepresented as absolute sheet crops.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from spritelab.adapters.shattered_pixel_dungeon import (
    EXPECTED_SHATTERED_PIXEL_DUNGEON_ARCHIVE_SHA256,
    AnimationAudit,
    FilmAudit,
    FrameCell,
    ShatteredPixelDungeonAudit,
    SpriteClassAudit,
    audit_known_shattered_pixel_dungeon_archive,
)
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

SOURCE_ID = "shattered_pixel_dungeon"
PROJECTION_VERSION = "shattered_pixel_dungeon_java_projection_v1"
RIGHTS_SCOPE_CAVEAT = (
    "License and attribution evidence is repository-level. The audited archive "
    "does not provide a per-PNG author/license manifest; do not claim that every "
    "sprite sheet has independently verified asset-level rights metadata."
)

TimingMode = Literal["exact_positive_fps", "pose_only_zero_fps", "ambiguous_fps"]


@dataclass(frozen=True)
class ShatteredPixelDungeonProjectionRecord:
    """One safe, materializable animation sequence from the Java audit."""

    sequence_source_key: str
    entity_external_key: str
    class_name: str
    display_name: str
    entity_class: str
    species_or_type: str
    morphology: tuple[str, ...]
    source_action: str
    normalized_action: str | None
    defined_in_class: str
    evidence_member_path: str
    evidence_line_number: int
    class_evidence_member_path: str
    class_evidence_line_number: int
    clone_of: str | None
    source_asset_key: str
    source_sheet_path: str
    source_sheet_member_path: str
    source_sheet_sha256: str
    sheet_width: int
    sheet_height: int
    film: FilmAudit | None
    variant_kind: Literal["frame_indices", "direct_uv_rect"]
    variant_ordinal: int
    frame_indices: tuple[int | None, ...]
    frame_cells: tuple[FrameCell, ...]
    source_fps_values: tuple[float, ...]
    source_fps_expression: str | None
    source_looping: bool
    source_looping_expression: str | None
    frame_expression_order: tuple[str, ...]
    frame_variable_expressions: tuple[tuple[str, tuple[str, ...]], ...]
    source_context: str
    inherited: bool
    ambiguity_reasons: tuple[str, ...]
    timing_mode: TimingMode
    duration_ms: float | None

    @property
    def frame_count(self) -> int:
        return len(self.frame_cells)

    @property
    def frame_width(self) -> int:
        cell = self.frame_cells[0]
        return cell.right - cell.left

    @property
    def frame_height(self) -> int:
        cell = self.frame_cells[0]
        return cell.bottom - cell.top


@dataclass(frozen=True)
class ShatteredPixelDungeonProjectionExclusion:
    """An audited animation intentionally omitted from materialization."""

    class_name: str
    source_action: str
    evidence_member_path: str
    evidence_line_number: int
    reason: str
    candidate_sequence_count: int
    candidate_frame_occurrence_count: int


@dataclass(frozen=True)
class ShatteredPixelDungeonProjectionPlan:
    """Pure, deterministic plan; constructing it performs no database writes."""

    archive_sha256: str
    repository_commit: str | None
    records: tuple[ShatteredPixelDungeonProjectionRecord, ...]
    exclusions: tuple[ShatteredPixelDungeonProjectionExclusion, ...]
    assets_java_member_path: str
    license_evidence_member_paths: tuple[str, ...]
    attribution_evidence_member_paths: tuple[str, ...]

    @property
    def projected_sequence_count(self) -> int:
        return len(self.records)

    @property
    def projected_frame_occurrence_count(self) -> int:
        return sum(record.frame_count for record in self.records)

    @property
    def projected_entity_count(self) -> int:
        return len({record.entity_external_key for record in self.records})

    @property
    def exact_timing_sequence_count(self) -> int:
        return sum(record.timing_mode == "exact_positive_fps" for record in self.records)

    @property
    def pose_only_sequence_count(self) -> int:
        return sum(record.timing_mode == "pose_only_zero_fps" for record in self.records)

    @property
    def ambiguous_timing_sequence_count(self) -> int:
        return sum(record.timing_mode == "ambiguous_fps" for record in self.records)

    @property
    def excluded_candidate_sequence_count(self) -> int:
        return sum(exclusion.candidate_sequence_count for exclusion in self.exclusions)

    @property
    def excluded_candidate_frame_occurrence_count(self) -> int:
        return sum(exclusion.candidate_frame_occurrence_count for exclusion in self.exclusions)

    @property
    def required_member_paths(self) -> tuple[str, ...]:
        paths = {self.assets_java_member_path}
        paths.update(self.license_evidence_member_paths)
        paths.update(self.attribution_evidence_member_paths)
        for record in self.records:
            paths.add(record.source_sheet_member_path)
            paths.add(record.evidence_member_path)
            paths.add(record.class_evidence_member_path)
        return tuple(sorted(paths))


@dataclass(frozen=True)
class ShatteredPixelDungeonReadiness:
    """Read-only check that a DB can accept a projection without missing evidence."""

    database_path: str
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
class ShatteredPixelDungeonProjectionResult:
    """Core-row effects of an idempotent projection run."""

    archive_sha256: str
    projected_entities: int
    projected_sequences: int
    projected_frame_occurrences: int
    created_sequences: int
    reused_sequences: int
    occurrence_links: int
    exact_timing_sequences: int
    pose_only_sequences: int
    ambiguous_timing_sequences: int
    excluded_candidate_sequences: int
    rights_observations_added: int = 0


def _stable_json_key(prefix: str, payload: dict[str, Any]) -> str:
    return f"{prefix}:" + json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _entity_external_key(
    audit: ShatteredPixelDungeonAudit,
    sprite_class: SpriteClassAudit,
    *,
    source_asset_key: str,
    source_sheet_path: str,
) -> str:
    return _stable_json_key(
        "spd-entity-v1",
        {
            "archive_sha256": audit.archive_sha256,
            "class_name": sprite_class.class_name,
            "repository_commit": audit.repository_commit,
            "source_asset_key": source_asset_key,
            "source_sheet_path": source_sheet_path,
        },
    )


def _sequence_source_key(
    audit: ShatteredPixelDungeonAudit,
    sprite_class: SpriteClassAudit,
    animation: AnimationAudit,
    *,
    source_asset_key: str,
    source_sheet_path: str,
    variant_kind: str,
    variant_ordinal: int,
    frame_indices: tuple[int | None, ...],
    frame_cells: tuple[FrameCell, ...],
) -> str:
    return _stable_json_key(
        "spd-sequence-v1",
        {
            "archive_sha256": audit.archive_sha256,
            "class_name": sprite_class.class_name,
            "defined_in_class": animation.defined_in_class,
            "source_context": animation.context,
            "evidence_line_number": animation.line_number,
            "evidence_member_path": animation.evidence_member_path,
            "fps_expression": animation.fps_expression,
            "frame_expression_order": list(animation.frame_expression_order),
            "frame_variable_expressions": [
                [name, list(expressions)]
                for name, expressions in animation.frame_variable_expressions
            ],
            "frame_indices": list(frame_indices),
            "frame_rectangles": [
                [cell.left, cell.top, cell.right, cell.bottom] for cell in frame_cells
            ],
            "repository_commit": audit.repository_commit,
            "clone_of": animation.clone_of,
            "inherited": animation.inherited,
            "looping_expression": animation.looping_expression,
            "source_action": animation.source_action,
            "source_asset_key": source_asset_key,
            "source_sheet_path": source_sheet_path,
            "variant_kind": variant_kind,
            "variant_ordinal": variant_ordinal,
        },
    )


def _timing_mode(animation: AnimationAudit) -> tuple[TimingMode, float | None] | None:
    fps_values = animation.fps_values
    if len(fps_values) > 1:
        if all(value > 0 for value in fps_values):
            return "ambiguous_fps", None
        return None
    if not fps_values:
        return None
    fps = fps_values[0]
    if fps > 0:
        return "exact_positive_fps", 1000.0 / fps
    if fps == 0:
        return "pose_only_zero_fps", None
    return None


def _candidate_counts(animation: AnimationAudit) -> tuple[int, int]:
    if animation.frame_index_variants:
        return (
            len(animation.frame_index_variants),
            sum(len(variant) for variant in animation.frame_index_variants),
        )
    return (
        len(animation.direct_uv_rect_variants),
        len(animation.direct_uv_rect_variants),
    )


def _exclude(
    sprite_class: SpriteClassAudit,
    animation: AnimationAudit,
    reason: str,
) -> ShatteredPixelDungeonProjectionExclusion:
    sequences, frames = _candidate_counts(animation)
    return ShatteredPixelDungeonProjectionExclusion(
        class_name=sprite_class.class_name,
        source_action=animation.source_action,
        evidence_member_path=animation.evidence_member_path,
        evidence_line_number=animation.line_number,
        reason=reason,
        candidate_sequence_count=sequences,
        candidate_frame_occurrence_count=frames,
    )


def _record(
    audit: ShatteredPixelDungeonAudit,
    sprite_class: SpriteClassAudit,
    animation: AnimationAudit,
    *,
    source_asset_key: str,
    source_sheet_path: str,
    variant_kind: Literal["frame_indices", "direct_uv_rect"],
    variant_ordinal: int,
    frame_indices: tuple[int | None, ...],
    frame_cells: tuple[FrameCell, ...],
    timing_mode: TimingMode,
    duration_ms: float | None,
) -> ShatteredPixelDungeonProjectionRecord:
    sheet = next(sheet for sheet in audit.sprite_sheets if sheet.relative_path == source_sheet_path)
    entity_key = _entity_external_key(
        audit,
        sprite_class,
        source_asset_key=source_asset_key,
        source_sheet_path=source_sheet_path,
    )
    return ShatteredPixelDungeonProjectionRecord(
        sequence_source_key=_sequence_source_key(
            audit,
            sprite_class,
            animation,
            source_asset_key=source_asset_key,
            source_sheet_path=source_sheet_path,
            variant_kind=variant_kind,
            variant_ordinal=variant_ordinal,
            frame_indices=frame_indices,
            frame_cells=frame_cells,
        ),
        entity_external_key=entity_key,
        class_name=sprite_class.class_name,
        display_name=sprite_class.entity_label,
        entity_class=sprite_class.entity_class,
        species_or_type=sprite_class.entity_label,
        morphology=sprite_class.morphology_tags,
        source_action=animation.source_action,
        normalized_action=animation.normalized_action,
        defined_in_class=animation.defined_in_class,
        evidence_member_path=animation.evidence_member_path,
        evidence_line_number=animation.line_number,
        class_evidence_member_path=sprite_class.evidence_member_path,
        class_evidence_line_number=sprite_class.line_number,
        clone_of=animation.clone_of,
        source_asset_key=source_asset_key,
        source_sheet_path=source_sheet_path,
        source_sheet_member_path=sheet.member_path,
        source_sheet_sha256=sheet.sha256,
        sheet_width=sheet.width,
        sheet_height=sheet.height,
        film=animation.film,
        variant_kind=variant_kind,
        variant_ordinal=variant_ordinal,
        frame_indices=frame_indices,
        frame_cells=frame_cells,
        source_fps_values=animation.fps_values,
        source_fps_expression=animation.fps_expression,
        source_looping=animation.looping_values[0],
        source_looping_expression=animation.looping_expression,
        frame_expression_order=animation.frame_expression_order,
        frame_variable_expressions=animation.frame_variable_expressions,
        source_context=animation.context,
        inherited=animation.inherited,
        ambiguity_reasons=animation.ambiguity_reasons,
        timing_mode=timing_mode,
        duration_ms=duration_ms,
    )


def plan_shattered_pixel_dungeon_projection(
    audit: ShatteredPixelDungeonAudit,
) -> ShatteredPixelDungeonProjectionPlan:
    """Build a deterministic, write-free projection plan from an adapter audit.

    The plan never combines FPS alternatives with frame alternatives.  Multiple
    frame orders become distinct sequences whose complete FPS candidate tuple is
    retained.  An FPS of zero is represented as an untimed pose, never as a
    zero-duration frame.
    """

    records: list[ShatteredPixelDungeonProjectionRecord] = []
    exclusions: list[ShatteredPixelDungeonProjectionExclusion] = []

    for sprite_class in audit.sprite_classes:
        if sprite_class.abstract:
            continue
        for animation in sprite_class.animations:
            if len(animation.source_sheet_paths) != 1 or len(animation.source_asset_keys) != 1:
                exclusions.append(
                    _exclude(
                        sprite_class,
                        animation,
                        "multiple_runtime_source_sheets_are_not_safely_correlated",
                    )
                )
                continue
            if len(animation.looping_values) != 1:
                exclusions.append(_exclude(sprite_class, animation, "loop_flag_is_ambiguous"))
                continue
            timing = _timing_mode(animation)
            if timing is None:
                exclusions.append(
                    _exclude(sprite_class, animation, "fps_is_missing_or_unsupported")
                )
                continue
            timing_mode, duration_ms = timing
            source_sheet_path = animation.source_sheet_paths[0]
            source_asset_key = animation.source_asset_keys[0]

            if animation.frame_index_variants:
                if len(animation.frame_cell_variants) != len(animation.frame_index_variants):
                    exclusions.append(
                        _exclude(
                            sprite_class,
                            animation,
                            "frame_order_and_geometry_variant_counts_do_not_match",
                        )
                    )
                    continue
                for ordinal, (indices, cells) in enumerate(
                    zip(
                        animation.frame_index_variants,
                        animation.frame_cell_variants,
                        strict=True,
                    )
                ):
                    if len(indices) != len(cells):
                        exclusions.append(
                            _exclude(
                                sprite_class,
                                animation,
                                "frame_order_and_geometry_lengths_do_not_match",
                            )
                        )
                        break
                    if any(cell.coordinate_space != "source_sheet" for cell in cells):
                        exclusions.append(
                            _exclude(
                                sprite_class,
                                animation,
                                "frame_geometry_is_relative_to_a_runtime_patch",
                            )
                        )
                        break
                    records.append(
                        _record(
                            audit,
                            sprite_class,
                            animation,
                            source_asset_key=source_asset_key,
                            source_sheet_path=source_sheet_path,
                            variant_kind="frame_indices",
                            variant_ordinal=ordinal,
                            frame_indices=tuple(indices),
                            frame_cells=tuple(cells),
                            timing_mode=timing_mode,
                            duration_ms=duration_ms,
                        )
                    )
            elif animation.direct_uv_rect_variants:
                for ordinal, cell in enumerate(animation.direct_uv_rect_variants):
                    if not cell.coordinate_space.startswith("source_sheet"):
                        exclusions.append(
                            _exclude(
                                sprite_class,
                                animation,
                                "direct_uv_geometry_is_not_in_source_sheet_coordinates",
                            )
                        )
                        break
                    records.append(
                        _record(
                            audit,
                            sprite_class,
                            animation,
                            source_asset_key=source_asset_key,
                            source_sheet_path=source_sheet_path,
                            variant_kind="direct_uv_rect",
                            variant_ordinal=ordinal,
                            frame_indices=(None,),
                            frame_cells=(cell,),
                            timing_mode=timing_mode,
                            duration_ms=duration_ms,
                        )
                    )
            else:
                exclusions.append(_exclude(sprite_class, animation, "no_resolved_frame_geometry"))

    records.sort(key=lambda record: record.sequence_source_key)
    exclusions.sort(
        key=lambda item: (
            item.class_name,
            item.source_action,
            item.evidence_member_path,
            item.evidence_line_number,
            item.reason,
        )
    )
    return ShatteredPixelDungeonProjectionPlan(
        archive_sha256=audit.archive_sha256,
        repository_commit=audit.repository_commit,
        records=tuple(records),
        exclusions=tuple(exclusions),
        assets_java_member_path=audit.assets_java_member_path,
        license_evidence_member_paths=tuple(
            evidence.member_path
            for evidence in audit.evidence_documents
            if evidence.detected_license_identifiers
        ),
        attribution_evidence_member_paths=tuple(
            sorted(
                {
                    evidence.member_path
                    for evidence in audit.evidence_documents
                    if not evidence.detected_license_identifiers
                }
                | {attribution.evidence_member_path for attribution in audit.attributions}
            )
        ),
    )


def plan_known_shattered_pixel_dungeon_projection(
    archive_path: str | Path,
) -> ShatteredPixelDungeonProjectionPlan:
    """Audit the pinned CAS archive and produce a write-free projection plan."""

    return plan_shattered_pixel_dungeon_projection(
        audit_known_shattered_pixel_dungeon_archive(archive_path)
    )


def _readonly_connection(database_path: str | Path) -> sqlite3.Connection:
    uri = Path(database_path).resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def check_shattered_pixel_dungeon_projection_readiness(
    database_path: str | Path,
    plan: ShatteredPixelDungeonProjectionPlan,
) -> ShatteredPixelDungeonReadiness:
    """Inspect exact projection prerequisites through a query-only SQLite URI."""

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
            SELECT normalized_path, member_path, extracted_blob_sha256
            FROM archive_members
            WHERE archive_blob_sha256 = ?
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
    for path, expected_hash in sorted(expected_sheet_hashes.items()):
        row = members.get(path)
        if row is None:
            continue
        actual_hash = row["extracted_blob_sha256"]
        if actual_hash is None:
            missing_sheet_blobs.append(path)
        elif str(actual_hash) != expected_hash:
            hash_mismatches.append(f"{path}: expected {expected_hash}, indexed {actual_hash}")
    return ShatteredPixelDungeonReadiness(
        database_path=str(Path(database_path).resolve()),
        archive_blob_present=archive_blob_present,
        source_item_count=source_item_count,
        required_member_count=len(required_paths),
        present_member_count=len(required_paths) - len(missing_paths),
        missing_member_paths=missing_paths,
        missing_sheet_blobs=tuple(missing_sheet_blobs),
        sheet_hash_mismatches=tuple(hash_mismatches),
    )


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
            "The exact Shattered Pixel Dungeon archive has no indexed source item "
            f"for source_id={SOURCE_ID!r}: {archive_sha256}"
        )
    return str(row[0])


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


def _preflight(
    database: IndexDB,
    plan: ShatteredPixelDungeonProjectionPlan,
) -> tuple[str, dict[str, sqlite3.Row]]:
    item_id = _item_id(database, plan.archive_sha256)
    members = _archive_members(database, plan.archive_sha256)
    missing = [path for path in plan.required_member_paths if path not in members]
    if missing:
        raise ValueError(
            "Projection evidence members are missing from archive_members: "
            + ", ".join(missing[:10])
        )
    for record in plan.records:
        row = members[record.source_sheet_member_path]
        indexed_hash = row["extracted_blob_sha256"]
        if indexed_hash is None:
            raise ValueError(
                "Source sprite sheet has not been extracted into CAS: "
                f"{record.source_sheet_member_path}"
            )
        if str(indexed_hash) != record.source_sheet_sha256:
            raise ValueError(
                "Source sprite sheet CAS hash does not match audited ZIP bytes for "
                f"{record.source_sheet_member_path}: expected "
                f"{record.source_sheet_sha256}, indexed {indexed_hash}"
            )
    return item_id, members


def _rights_scope_metadata(plan: ShatteredPixelDungeonProjectionPlan) -> dict[str, Any]:
    return {
        "scope": "repository_level_only",
        "caveat": RIGHTS_SCOPE_CAVEAT,
        "license_evidence_member_paths": list(plan.license_evidence_member_paths),
        "attribution_evidence_member_paths": list(plan.attribution_evidence_member_paths),
        "per_png_manifest_present": False,
    }


def _loop_semantics(
    record: ShatteredPixelDungeonProjectionRecord,
) -> tuple[str, bool | None, int | None]:
    if record.timing_mode == "pose_only_zero_fps":
        return "unknown", None, None
    if record.source_looping:
        return "loop", True, record.frame_count
    return "one_shot", False, None


def _frame_phase(
    ordinal: int,
    frame_count: int,
    *,
    loop_mode: str,
) -> float:
    if frame_count <= 1:
        return 0.0
    if loop_mode == "loop":
        return ordinal / frame_count
    return ordinal / (frame_count - 1)


def _sequence_metadata(
    plan: ShatteredPixelDungeonProjectionPlan,
    record: ShatteredPixelDungeonProjectionRecord,
) -> dict[str, Any]:
    timing_known = record.timing_mode == "exact_positive_fps"
    return {
        "projection_version": PROJECTION_VERSION,
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "source_class": record.class_name,
        "defined_in_class": record.defined_in_class,
        "source_action": record.source_action,
        "normalized_action": record.normalized_action,
        "source_asset_key": record.source_asset_key,
        "source_sheet_path": record.source_sheet_path,
        "source_sheet_member_path": record.source_sheet_member_path,
        "source_sheet_sha256": record.source_sheet_sha256,
        "sheet_dimensions": [record.sheet_width, record.sheet_height],
        "variant_kind": record.variant_kind,
        "variant_ordinal": record.variant_ordinal,
        "frame_indices": list(record.frame_indices),
        "film": asdict(record.film) if record.film is not None else None,
        "source_fps_values": list(record.source_fps_values),
        "source_fps_expression": record.source_fps_expression,
        "source_looping_values": [record.source_looping],
        "source_looping_expression": record.source_looping_expression,
        "frame_expression_order": list(record.frame_expression_order),
        "frame_variable_expressions": [
            [name, list(expressions)] for name, expressions in record.frame_variable_expressions
        ],
        "source_context": record.source_context,
        "inherited": record.inherited,
        "ambiguity_reasons": list(record.ambiguity_reasons),
        "timing_mode": record.timing_mode,
        "timing_known": timing_known,
        "exact_engine_timing": timing_known,
        "pose_only": record.timing_mode == "pose_only_zero_fps",
        "state_occurrence_order_preserved": True,
        "geometry_coordinate_space": "source_sheet",
        "geometry_absolute": True,
        "direction_evidence": (
            "no_explicit_direction_track; CharSprite_default_can_horizontal_flip"
        ),
        "evidence": {
            "animation_member_path": record.evidence_member_path,
            "animation_line_number": record.evidence_line_number,
            "class_member_path": record.class_evidence_member_path,
            "class_line_number": record.class_evidence_line_number,
            "clone_of": record.clone_of,
            "assets_java_member_path": plan.assets_java_member_path,
        },
        "rights_scope": _rights_scope_metadata(plan),
    }


def _occurrence_specs(
    plan: ShatteredPixelDungeonProjectionPlan,
    record: ShatteredPixelDungeonProjectionRecord,
) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    common = {
        "archive_sha256": plan.archive_sha256,
        "repository_commit": plan.repository_commit,
        "projection_version": PROJECTION_VERSION,
    }
    specs: list[tuple[str, str, dict[str, Any]]] = [
        (
            record.source_sheet_member_path,
            "spd_source_sprite_sheet",
            {
                **common,
                "source_asset_key": record.source_asset_key,
                "source_sheet_path": record.source_sheet_path,
                "source_sheet_sha256": record.source_sheet_sha256,
                "sheet_dimensions": [record.sheet_width, record.sheet_height],
            },
        ),
        (
            record.evidence_member_path,
            "spd_animation_definition",
            {
                **common,
                "class_name": record.class_name,
                "defined_in_class": record.defined_in_class,
                "source_action": record.source_action,
                "line_number": record.evidence_line_number,
            },
        ),
        (
            plan.assets_java_member_path,
            "spd_asset_mapping_definition",
            {
                **common,
                "source_asset_key": record.source_asset_key,
                "source_sheet_path": record.source_sheet_path,
            },
        ),
    ]
    if record.class_evidence_member_path != record.evidence_member_path:
        specs.append(
            (
                record.class_evidence_member_path,
                "spd_entity_definition",
                {
                    **common,
                    "class_name": record.class_name,
                    "line_number": record.class_evidence_line_number,
                },
            )
        )
    for path in plan.license_evidence_member_paths:
        specs.append(
            (
                path,
                "spd_repository_license_evidence",
                {**common, "rights_scope_caveat": RIGHTS_SCOPE_CAVEAT},
            )
        )
    for path in plan.attribution_evidence_member_paths:
        specs.append(
            (
                path,
                "spd_repository_attribution_evidence",
                {**common, "rights_scope_caveat": RIGHTS_SCOPE_CAVEAT},
            )
        )
    return tuple(specs)


def project_shattered_pixel_dungeon_audit(
    database: IndexDB,
    plan: ShatteredPixelDungeonProjectionPlan,
    taxonomy: Taxonomy,
) -> ShatteredPixelDungeonProjectionResult:
    """Idempotently project a precomputed plan into core provenance tables.

    The caller must have already indexed the source item, archive members, and
    extracted sprite sheet blobs.  This function adds no rights observation: it
    preserves the repository-scope caveat on entities, sequences, frames, and
    occurrence links without upgrading repository evidence to per-asset rights.
    """

    database.initialize()
    item_id, members = _preflight(database, plan)
    entity_ids: dict[str, str] = {}
    created_sequences = 0
    reused_sequences = 0
    occurrence_links = 0
    rights_scope = _rights_scope_metadata(plan)

    for record in plan.records:
        entity_id = entity_ids.get(record.entity_external_key)
        if entity_id is None:
            entity_class = taxonomy.normalize_entity_class(record.entity_class)
            entity_id = database.upsert_entity(
                source_id=SOURCE_ID,
                external_identity_key=record.entity_external_key,
                representative_item_id=item_id,
                display_name=record.display_name,
                entity_class=entity_class.value,
                species_or_type=record.species_or_type,
                taxonomy_version=taxonomy.version,
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "archive_sha256": plan.archive_sha256,
                    "repository_commit": plan.repository_commit,
                    "source_class": record.class_name,
                    "source_asset_key": record.source_asset_key,
                    "source_sheet_path": record.source_sheet_path,
                    "morphology": list(record.morphology),
                    "default_view": "top_down",
                    "class_evidence_member_path": record.class_evidence_member_path,
                    "class_evidence_line_number": record.class_evidence_line_number,
                    "classification_basis": (
                        "conservative lexical class-name classification from the audit"
                    ),
                    "rights_scope": rights_scope,
                },
            )
            entity_ids[record.entity_external_key] = entity_id

        loop_mode, semantic_loopable, cycle_frames = _loop_semantics(record)
        motion = taxonomy.motion_condition(
            action=record.normalized_action,
            direction="unknown",
            view="top_down",
        )
        existing_sequence_id = database.find_sequence_by_source_key(
            source_id=SOURCE_ID,
            external_sequence_key=record.sequence_source_key,
        )
        sequence_metadata = _sequence_metadata(plan, record)
        quality = {
            "exact_positive_fps": "F0_lossless_source_sheet_exact_timing",
            "pose_only_zero_fps": "F1_lossless_source_sheet_pose_only",
            "ambiguous_fps": "F1_lossless_source_sheet_ambiguous_timing",
        }[record.timing_mode]
        if existing_sequence_id is None:
            sequence_id = database.create_sequence(
                item_id=item_id,
                source_blob_sha256=record.source_sheet_sha256,
                width=record.frame_width,
                height=record.frame_height,
                frame_count=record.frame_count,
                loop_mode=loop_mode,
                action=motion.normalized_action,
                direction="unknown",
                extraction_method=PROJECTION_VERSION,
                quality_tier=quality,
                extraction_confidence=(1.0 if record.timing_mode != "ambiguous_fps" else 0.9),
                metadata=sequence_metadata,
            )
            database.register_sequence_source_key(
                source_id=SOURCE_ID,
                external_sequence_key=record.sequence_source_key,
                sequence_id=sequence_id,
            )
            created_sequences += 1
        else:
            sequence_id = existing_sequence_id
            database.update_sequence_facts(
                sequence_id=sequence_id,
                source_blob_sha256=record.source_sheet_sha256,
                width=record.frame_width,
                height=record.frame_height,
                frame_count=record.frame_count,
                loop_mode=loop_mode,
                action=motion.normalized_action,
                direction="unknown",
                extraction_method=PROJECTION_VERSION,
                quality_tier=quality,
                extraction_confidence=(1.0 if record.timing_mode != "ambiguous_fps" else 0.9),
                metadata=sequence_metadata,
            )
            reused_sequences += 1

        database.link_sequence_subject(
            sequence_id=sequence_id,
            entity_id=entity_id,
            role="primary",
            metadata={
                "source_class": record.class_name,
                "source_sheet_path": record.source_sheet_path,
                "rights_scope": rights_scope,
            },
        )

        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version=taxonomy.version,
            source_action=record.source_action,
            normalized_action=motion.normalized_action,
            action_family=motion.action_family,
            annotation_method=PROJECTION_VERSION,
            view="top_down",
            direction="unknown",
            loopable=semantic_loopable,
            cycle_frames=cycle_frames,
            phase_zero_frame=0,
            confidence=1.0 if record.normalized_action is not None else 0.0,
            conditioning={
                "normalization_method": motion.method,
                "normalization_confidence": motion.confidence,
                "source_fps_values": list(record.source_fps_values),
                "source_looping_values": [record.source_looping],
                "timing_mode": record.timing_mode,
                "timing_known": record.timing_mode == "exact_positive_fps",
                "exact_engine_timing": record.timing_mode == "exact_positive_fps",
                "pose_only": record.timing_mode == "pose_only_zero_fps",
                "state_occurrence_order_preserved": True,
                "direction_evidence": (
                    "no_explicit_direction_track; CharSprite_default_can_horizontal_flip"
                ),
                "geometry_coordinate_space": "source_sheet",
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

        for ordinal, (frame_index, cell) in enumerate(
            zip(record.frame_indices, record.frame_cells, strict=True)
        ):
            database.add_sequence_frame(
                sequence_id=sequence_id,
                ordinal=ordinal,
                source_blob_sha256=record.source_sheet_sha256,
                source_frame_index=frame_index,
                duration_ms=record.duration_ms,
                phase=_frame_phase(
                    ordinal,
                    record.frame_count,
                    loop_mode=loop_mode,
                ),
                direction="unknown",
                view="top_down",
                metadata={
                    "projection_version": PROJECTION_VERSION,
                    "source_sheet_path": record.source_sheet_path,
                    "source_sheet_member_path": record.source_sheet_member_path,
                    "source_asset_key": record.source_asset_key,
                    "source_frame_index": frame_index,
                    "frame_rect": {
                        "left": cell.left,
                        "top": cell.top,
                        "right": cell.right,
                        "bottom": cell.bottom,
                        "width": cell.right - cell.left,
                        "height": cell.bottom - cell.top,
                        "column": cell.column,
                        "row": cell.row,
                        "coordinate_space": cell.coordinate_space,
                    },
                    "timing_mode": record.timing_mode,
                    "source_fps_values": list(record.source_fps_values),
                    "source_looping": record.source_looping,
                    "rights_scope": rights_scope,
                },
            )

    return ShatteredPixelDungeonProjectionResult(
        archive_sha256=plan.archive_sha256,
        projected_entities=len(entity_ids),
        projected_sequences=plan.projected_sequence_count,
        projected_frame_occurrences=plan.projected_frame_occurrence_count,
        created_sequences=created_sequences,
        reused_sequences=reused_sequences,
        occurrence_links=occurrence_links,
        exact_timing_sequences=plan.exact_timing_sequence_count,
        pose_only_sequences=plan.pose_only_sequence_count,
        ambiguous_timing_sequences=plan.ambiguous_timing_sequence_count,
        excluded_candidate_sequences=plan.excluded_candidate_sequence_count,
    )


def ingest_known_shattered_pixel_dungeon_sequences(
    database: IndexDB,
    archive_path: str | Path,
    taxonomy: Taxonomy,
) -> ShatteredPixelDungeonProjectionResult:
    """Audit and project only the exact pinned Shattered Pixel Dungeon archive."""

    plan = plan_known_shattered_pixel_dungeon_projection(archive_path)
    if plan.archive_sha256 != EXPECTED_SHATTERED_PIXEL_DUNGEON_ARCHIVE_SHA256:
        raise ValueError(
            "Refusing Shattered Pixel Dungeon projection for unexpected archive: "
            f"{plan.archive_sha256}"
        )
    return project_shattered_pixel_dungeon_audit(database, plan, taxonomy)


__all__ = [
    "PROJECTION_VERSION",
    "RIGHTS_SCOPE_CAVEAT",
    "SOURCE_ID",
    "ShatteredPixelDungeonProjectionExclusion",
    "ShatteredPixelDungeonProjectionPlan",
    "ShatteredPixelDungeonProjectionRecord",
    "ShatteredPixelDungeonProjectionResult",
    "ShatteredPixelDungeonReadiness",
    "check_shattered_pixel_dungeon_projection_readiness",
    "ingest_known_shattered_pixel_dungeon_sequences",
    "plan_known_shattered_pixel_dungeon_projection",
    "plan_shattered_pixel_dungeon_projection",
    "project_shattered_pixel_dungeon_audit",
]
