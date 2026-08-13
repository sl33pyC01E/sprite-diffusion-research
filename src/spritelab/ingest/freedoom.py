from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from spritelab.adapters.freedoom import (
    ActionSequenceAudit,
    DoomFrameRotation,
    FamilyAudit,
    SpriteImageAudit,
    audit_freedoom_archive,
)
from spritelab.db import IndexDB
from spritelab.taxonomy import Taxonomy

_TRAINABLE_ACTIONS = frozenset({"idle", "run", "attack", "hurt", "death"})
_HUMANOID_FAMILIES = frozenset({"PLAY", "POSS", "SPOS", "CPOS", "SSWV"})
_ROBOT_FAMILIES = frozenset({"BSPI", "CYBR", "SPID"})


@dataclass(frozen=True)
class FreedoomIngestResult:
    entities: int
    sequences: int
    frames: int
    occurrence_edges: int
    skipped_ambiguous_actions: int
    skipped_unmapped_families: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def ingest_freedoom_sequences(
    *,
    database: IndexDB,
    archive_blob_sha256: str,
    archive_path: Path,
    taxonomy: Taxonomy,
) -> FreedoomIngestResult:
    """Index conservative Doom-state action/view clips from a pinned archive.

    The compatibility table provides unique pose order, not exact engine state
    repetition or tic timing. Directional mirror requirements stay explicit and
    no transformed pixels are silently substituted for the raw source blob.
    """
    database.initialize()
    item_id = _item_id(database, "freedoom")
    audit = audit_freedoom_archive(
        archive_path,
        expected_sha256=archive_blob_sha256,
    )
    member_rows = _member_rows(database, archive_blob_sha256)
    entity_ids: dict[str, str] = {}
    sequence_count = 0
    frame_count = 0
    occurrence_count = 0
    skipped_ambiguous = 0
    skipped_unmapped = 0

    for family in audit.families:
        action_sequences = [
            sequence for sequence in family.sequences if sequence.action in _TRAINABLE_ACTIONS
        ]
        if not action_sequences:
            skipped_unmapped += 1
            continue
        entity_id = _entity(database, item_id, taxonomy, family)
        entity_ids[family.identity_key] = entity_id
        files_by_name = {file.raw_filename: file for file in audit.sprite_files}
        for action_sequence in action_sequences:
            for source_rotation in _sequence_rotations(action_sequence, files_by_name):
                selected = _select_frames(action_sequence, source_rotation, files_by_name)
                if not selected:
                    continue
                if action_sequence.overlaps_other_action_groups:
                    skipped_ambiguous += len(action_sequence.ambiguous_frame_tokens)
                external_key = _external_sequence_key(
                    family=family.family,
                    action=action_sequence.action,
                    rotation=source_rotation,
                    commit=audit.repository_commit,
                )
                sequence_id = database.find_sequence_by_source_key(
                    source_id="freedoom",
                    external_sequence_key=external_key,
                )
                first_file = selected[0][0]
                sequence_metadata = {
                    "source_family": family.family,
                    "source_rotation": source_rotation,
                    "source_sequence_key": action_sequence.sequence_key,
                    "frame_tokens": action_sequence.frame_tokens,
                    "basis": action_sequence.basis,
                    "sequence_semantics": action_sequence.sequence_semantics,
                    "exact_engine_timing": action_sequence.timing_preserved,
                    "state_occurrence_order_preserved": (
                        action_sequence.state_occurrence_order_preserved
                    ),
                    "overlaps_other_action_groups": (action_sequence.overlaps_other_action_groups),
                    "ambiguous_frame_tokens": action_sequence.ambiguous_frame_tokens,
                    "repository_commit": audit.repository_commit,
                }
                if sequence_id is None:
                    sequence_id = database.create_sequence(
                        item_id=item_id,
                        source_blob_sha256=first_file.sha256,
                        extraction_method="freedoom_doom_state_unique_pose_projection_v2",
                        extraction_confidence=0.55,
                        width=max(file.width for file, _reference, _alternates in selected),
                        height=max(file.height for file, _reference, _alternates in selected),
                        frame_count=len(selected),
                        loop_mode="unknown",
                        action=action_sequence.action,
                        direction="unknown",
                        quality_tier="F1_lossless_unique_poses_unknown_timing",
                        metadata=sequence_metadata,
                    )
                else:
                    database.update_sequence_facts(
                        sequence_id=sequence_id,
                        source_blob_sha256=first_file.sha256,
                        extraction_method="freedoom_doom_state_unique_pose_projection_v2",
                        extraction_confidence=0.55,
                        width=max(file.width for file, _reference, _alternates in selected),
                        height=max(file.height for file, _reference, _alternates in selected),
                        frame_count=len(selected),
                        loop_mode="unknown",
                        action=action_sequence.action,
                        direction="unknown",
                        quality_tier="F1_lossless_unique_poses_unknown_timing",
                        metadata=sequence_metadata,
                    )
                database.register_sequence_source_key(
                    source_id="freedoom",
                    external_sequence_key=external_key,
                    sequence_id=sequence_id,
                )
                database.link_sequence_subject(
                    sequence_id=sequence_id,
                    entity_id=entity_id,
                )
                motion = taxonomy.motion_condition(action=action_sequence.action)
                database.annotate_motion(
                    sequence_id=sequence_id,
                    vocabulary_version=taxonomy.version,
                    source_action=action_sequence.action,
                    normalized_action=motion.normalized_action,
                    action_family=motion.action_family,
                    annotation_method="canonical_doom_1.10_unique_artwork_projection",
                    view="unknown",
                    direction="unknown",
                    loopable=None,
                    cycle_frames=None,
                    phase_zero_frame=0,
                    confidence=0.55,
                    conditioning={
                        "doom_rotation": source_rotation,
                        "timing_known": False,
                        "state_occurrence_order_preserved": False,
                        "sequence_semantics": action_sequence.sequence_semantics,
                        "frame_overlap_ambiguity": (action_sequence.overlaps_other_action_groups),
                    },
                )
                for ordinal, (file, reference, alternates) in enumerate(selected):
                    member = member_rows[file.member_path]
                    database.link_sequence_occurrence(
                        sequence_id=sequence_id,
                        archive_blob_sha256=archive_blob_sha256,
                        archive_member_ordinal=int(member["ordinal"]),
                        occurrence_role="doom_sprite_lump",
                        metadata={
                            "raw_filename": file.raw_filename,
                            "frame_token": reference.frame_token,
                            "rotation": reference.rotation,
                            "mirror_required": reference.mirrored,
                        },
                    )
                    occurrence_count += 1
                    database.add_sequence_frame(
                        sequence_id=sequence_id,
                        ordinal=ordinal,
                        source_blob_sha256=file.sha256,
                        source_frame_index=None,
                        duration_ms=None,
                        phase=_phase(ordinal, len(selected), None),
                        direction="unknown",
                        view="unknown",
                        metadata={
                            "frame_token": reference.frame_token,
                            "doom_rotation": reference.rotation,
                            "mirror_required": reference.mirrored,
                            "source_filename": file.raw_filename,
                            "alternate_source_filenames": [
                                candidate.raw_filename for candidate in alternates
                            ],
                            "timing_unknown": True,
                        },
                    )
                    frame_count += 1
                sequence_count += 1

    return FreedoomIngestResult(
        entities=len(entity_ids),
        sequences=sequence_count,
        frames=frame_count,
        occurrence_edges=occurrence_count,
        skipped_ambiguous_actions=skipped_ambiguous,
        skipped_unmapped_families=skipped_unmapped,
    )


def _entity(
    database: IndexDB,
    item_id: str,
    taxonomy: Taxonomy,
    family: FamilyAudit,
) -> str:
    if family.family in _HUMANOID_FAMILIES:
        entity_class = "humanoid"
        candidates = ("humanoid", "monster") if family.family != "PLAY" else ("humanoid",)
    elif family.family in _ROBOT_FAMILIES:
        entity_class = "robot"
        candidates = ("robot", "monster")
    else:
        entity_class = "monster"
        candidates = ("monster", "creature")
    labels = [hint.label for hint in family.label_hints]
    return database.upsert_entity(
        source_id="freedoom",
        external_identity_key=family.identity_key,
        representative_item_id=item_id,
        display_name=labels[0] if labels else family.family,
        entity_class=entity_class,
        species_or_type=labels[0] if labels else None,
        taxonomy_version=taxonomy.version,
        metadata={
            "doom_family": family.family,
            "entity_class_candidates": candidates,
            "label_hints": labels,
            "label_basis": "dehacked_cc_cast_call" if labels else "unresolved",
        },
    )


def _sequence_rotations(
    sequence: ActionSequenceAudit,
    files_by_name: dict[str, SpriteImageAudit],
) -> tuple[int, ...]:
    rotations = {
        reference.rotation
        for filename in sequence.raw_filenames
        for reference in _references(files_by_name[filename])
        if reference.frame_token in sequence.frame_tokens and reference.rotation != 0
    }
    return tuple(sorted(rotations)) if rotations else (0,)


def _select_frames(
    sequence: ActionSequenceAudit,
    rotation: int,
    files_by_name: dict[str, SpriteImageAudit],
) -> list[tuple[SpriteImageAudit, DoomFrameRotation, tuple[SpriteImageAudit, ...]]]:
    result = []
    for frame_token in sequence.frame_tokens:
        candidates: list[tuple[SpriteImageAudit, DoomFrameRotation]] = []
        for filename in sequence.raw_filenames:
            file = files_by_name[filename]
            for reference in _references(file):
                if reference.frame_token != frame_token:
                    continue
                if reference.rotation == rotation or (rotation != 0 and reference.rotation == 0):
                    candidates.append((file, reference))
        if not candidates:
            continue
        candidates.sort(
            key=lambda pair: (
                pair[1].rotation != rotation,
                pair[1].mirrored,
                pair[0].raw_filename.encode("utf-8"),
            )
        )
        chosen_file, chosen_reference = candidates[0]
        alternates = tuple(file for file, _reference in candidates[1:])
        result.append((chosen_file, chosen_reference, alternates))
    return result


def _references(file: SpriteImageAudit) -> tuple[DoomFrameRotation, ...]:
    return file.parsed_name.references if file.parsed_name else ()


def _member_rows(database: IndexDB, archive_sha256: str) -> dict[str, sqlite3.Row]:
    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT ordinal, normalized_path, extracted_blob_sha256
            FROM archive_members WHERE archive_blob_sha256=?
            """,
            (archive_sha256,),
        ).fetchall()
    return {str(row["normalized_path"]): row for row in rows}


def _item_id(database: IndexDB, source_id: str) -> str:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT id FROM items WHERE source_id=? ORDER BY id LIMIT 1",
            (source_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"No indexed item for source {source_id}")
    return str(row["id"])


def _external_sequence_key(*, family: str, action: str, rotation: int, commit: str | None) -> str:
    return json.dumps(
        {
            "family": family,
            "action": action,
            "rotation": rotation,
            "commit": commit,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _phase(ordinal: int, frame_count: int, loop_hint: bool | None) -> float:
    if frame_count <= 1:
        return 0.0
    if loop_hint is True:
        return ordinal / frame_count
    return ordinal / (frame_count - 1)
