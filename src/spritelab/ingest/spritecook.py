from __future__ import annotations

import json
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spritelab.adapters.spritecook import (
    SpriteCookIndex,
    classify_member,
    parse_index_metadata,
)
from spritelab.db import IndexDB
from spritelab.media import extract_animation
from spritelab.taxonomy import Taxonomy


@dataclass(frozen=True)
class SpriteCookIngestResult:
    entities: int
    sequences: int
    frames: int
    skipped_duplicate_occurrences: int
    unresolved_actions: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def ingest_spritecook_sequences(
    *,
    database: IndexDB,
    archive_blob_sha256: str,
    archive_path: Path,
    taxonomy: Taxonomy,
) -> SpriteCookIngestResult:
    """Index logical SpriteCook animations without generating frame files.

    Each sequence points to an immutable animated WebP carrier and records frame
    indices/durations/phases. Mirrored archive occurrences are retained as
    occurrence edges but collapsed to one logical identity/action sequence.
    """
    database.initialize()
    item_id = _item_id(database, "spritecook_free")
    index = _load_index(archive_path)
    candidates = _candidate_rows(database, archive_blob_sha256)
    logical: dict[tuple[str, str, str], dict[str, Any]] = {}
    occurrences: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
    unresolved_actions = 0

    for row in candidates:
        hint = classify_member(row["normalized_path"], index=index)
        if hint.identity_key is None:
            continue
        action = (
            hint.normalized_action_candidates[0]
            if len(hint.normalized_action_candidates) == 1
            else "unknown"
        )
        if action == "unknown":
            unresolved_actions += 1
        key = (hint.identity_key, action, str(row["extracted_blob_sha256"]))
        occurrences.setdefault(key, []).append(row)
        logical.setdefault(
            key,
            {
                "hint": hint,
                "row": row,
                "action": action,
            },
        )

    entities: dict[str, str] = {}
    existing_sequences = _existing_sequence_keys(database, item_id)
    sequences = 0
    frames = 0
    duplicate_occurrences = 0
    for key in sorted(logical):
        record = logical[key]
        hint = record["hint"]
        row = record["row"]
        action = record["action"]
        identity_key = hint.identity_key
        if identity_key not in entities:
            entity_class = _primary_entity_class(hint.normalized_entity_class_candidates)
            entities[identity_key] = database.upsert_entity(
                source_id="spritecook_free",
                external_identity_key=identity_key,
                representative_item_id=item_id,
                display_name=hint.raw_entity_hint,
                entity_class=entity_class,
                taxonomy_version=taxonomy.version,
                metadata={
                    "entity_class_candidates": hint.normalized_entity_class_candidates,
                    "entity_basis": hint.entity_basis,
                    "source_example": hint.example_slug,
                },
            )

        blob_path = Path(str(row["storage_path"]))
        animation = extract_animation(blob_path)
        motion = taxonomy.motion_condition(
            action=action,
            view=_source_view(hint),
        )
        loop_semantics = "loop" if motion.loopable_default is True else "one_shot"
        if action == "unknown":
            loop_semantics = "unknown"
        external_sequence_key = json.dumps(key, ensure_ascii=False, separators=(",", ":"))
        sequence_id = database.find_sequence_by_source_key(
            source_id="spritecook_free",
            external_sequence_key=external_sequence_key,
        ) or existing_sequences.get(external_sequence_key)
        if sequence_id is None:
            sequence_id = database.create_sequence(
                item_id=item_id,
                source_blob_sha256=str(row["extracted_blob_sha256"]),
                extraction_method="spritecook_animated_container_v1",
                extraction_confidence=1.0,
                width=animation.canvas_size[0],
                height=animation.canvas_size[1],
                frame_count=animation.frame_count,
                loop_mode=loop_semantics,
                action=action,
                direction="unknown",
                quality_tier="F0_lossless_container",
                metadata={
                    "container_format": animation.format,
                    "container_loop_count": animation.loop_count,
                    "source_action": hint.raw_action_hint,
                    "prompt": hint.provenance.prompt,
                    "prompt_scope": hint.provenance.prompt_scope,
                    "entity_class_candidates": hint.normalized_entity_class_candidates,
                    "logical_key": key,
                    "external_sequence_key": external_sequence_key,
                },
            )
        database.register_sequence_source_key(
            source_id="spritecook_free",
            external_sequence_key=external_sequence_key,
            sequence_id=sequence_id,
        )
        database.link_sequence_subject(
            sequence_id=sequence_id,
            entity_id=entities[identity_key],
        )
        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version=taxonomy.version,
            source_action=hint.raw_action_hint,
            normalized_action=action,
            action_family=motion.action_family,
            annotation_method=f"spritecook_adapter:{hint.action_basis}",
            view=motion.view,
            direction=motion.direction,
            loopable=motion.loopable_default,
            cycle_frames=(animation.frame_count if motion.loopable_default is True else None),
            phase_zero_frame=0,
            confidence=motion.confidence,
            conditioning={
                "container_loop_count": animation.loop_count,
                "semantic_loop_mode": loop_semantics,
            },
        )
        for occurrence in sorted(occurrences[key], key=lambda value: str(value["normalized_path"])):
            database.link_sequence_occurrence(
                sequence_id=sequence_id,
                archive_blob_sha256=archive_blob_sha256,
                archive_member_ordinal=int(occurrence["ordinal"]),
                occurrence_role="animation_container",
                metadata={"archive_member": occurrence["normalized_path"]},
            )
        duplicate_occurrences += len(occurrences[key]) - 1
        for ordinal, frame in enumerate(animation.frames):
            phase = _phase(ordinal, animation.frame_count, loop_semantics)
            database.add_sequence_frame(
                sequence_id=sequence_id,
                ordinal=ordinal,
                source_blob_sha256=str(row["extracted_blob_sha256"]),
                source_frame_index=frame.source_index,
                duration_ms=frame.duration_ms,
                phase=phase,
                direction="unknown",
                view=motion.view,
                metadata={
                    "disposal": frame.disposal,
                    "blend": frame.blend,
                    "source_extent": frame.source_extent,
                },
            )
            frames += 1
        sequences += 1

    return SpriteCookIngestResult(
        entities=len(entities),
        sequences=sequences,
        frames=frames,
        skipped_duplicate_occurrences=duplicate_occurrences,
        unresolved_actions=unresolved_actions,
    )


def _candidate_rows(database: IndexDB, archive_sha256: str) -> list[sqlite3.Row]:
    with database.connect() as connection:
        return connection.execute(
            """
            SELECT am.ordinal, am.normalized_path, am.extracted_blob_sha256,
                   b.storage_path
            FROM archive_members am
            JOIN blobs b ON b.sha256=am.extracted_blob_sha256
            WHERE am.archive_blob_sha256=?
              AND lower(am.normalized_path) LIKE '%.webp'
            ORDER BY am.normalized_path COLLATE BINARY
            """,
            (archive_sha256,),
        ).fetchall()


def _item_id(database: IndexDB, source_id: str) -> str:
    with database.connect() as connection:
        row = connection.execute(
            "SELECT id FROM items WHERE source_id=? ORDER BY id LIMIT 1",
            (source_id,),
        ).fetchone()
    if row is None:
        raise KeyError(f"No indexed item for source {source_id}")
    return str(row["id"])


def _existing_sequence_keys(database: IndexDB, item_id: str) -> dict[str, str]:
    result: dict[str, str] = {}
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT id, metadata_json FROM sequences WHERE item_id=? ORDER BY id",
            (item_id,),
        ).fetchall()
    for row in rows:
        metadata = json.loads(str(row["metadata_json"]))
        logical_key = metadata.get("logical_key")
        if isinstance(logical_key, list) and len(logical_key) == 3:
            key = json.dumps(logical_key, ensure_ascii=False, separators=(",", ":"))
            result.setdefault(key, str(row["id"]))
    return result


def _load_index(archive_path: Path) -> SpriteCookIndex:
    with zipfile.ZipFile(archive_path) as archive:
        matches = [
            name
            for name in archive.namelist()
            if name.casefold().endswith("/index.json") or name.casefold() == "index.json"
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one SpriteCook index.json, found {len(matches)}")
        payload = json.loads(archive.read(matches[0]))
    return parse_index_metadata(payload)


def _primary_entity_class(candidates: tuple[str, ...]) -> str:
    return candidates[0] if candidates else "unknown"


def _source_view(hint: Any) -> str | None:
    return "isometric" if hint.example_slug == "isometric-buildings" else None


def _phase(ordinal: int, frame_count: int, loop_mode: str) -> float:
    if frame_count <= 1:
        return 0.0
    if loop_mode == "loop":
        return ordinal / frame_count
    return ordinal / (frame_count - 1)
