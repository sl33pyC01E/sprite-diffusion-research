from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from spritelab.adapters.shattered_pixel_dungeon import FrameCell
from spritelab.db import IndexDB
from spritelab.ingest.shattered_pixel_dungeon import (
    ShatteredPixelDungeonProjectionPlan,
    ShatteredPixelDungeonProjectionRecord,
    check_shattered_pixel_dungeon_projection_readiness,
    plan_known_shattered_pixel_dungeon_projection,
    project_shattered_pixel_dungeon_audit,
)
from spritelab.taxonomy import load_taxonomy

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "de/ed/deed31527c24b2a500f6c79e1cf2502c4c9d8b4391529033da7015cd73c97544"
)
TAXONOMY = load_taxonomy(Path("configs/taxonomy.toml"))


def _cell(index: int) -> FrameCell:
    column = index % 8
    row = index // 8
    return FrameCell(
        frame_index=index,
        column=column,
        row=row,
        left=column * 8,
        top=row * 8,
        right=(column + 1) * 8,
        bottom=(row + 1) * 8,
        coordinate_space="source_sheet",
    )


def _record(
    *,
    key: str,
    action: str,
    normalized_action: str,
    indices: tuple[int, ...],
    fps_values: tuple[float, ...],
    timing_mode: str,
    duration_ms: float | None,
    source_looping: bool,
    sheet_sha256: str,
) -> ShatteredPixelDungeonProjectionRecord:
    return ShatteredPixelDungeonProjectionRecord(
        sequence_source_key=key,
        entity_external_key="spd-entity-v1:test-rat",
        class_name="RatSprite",
        display_name="rat",
        entity_class="animal",
        species_or_type="rat",
        morphology=("quadruped",),
        source_action=action,
        normalized_action=normalized_action,
        defined_in_class="RatSprite",
        evidence_member_path="root/src/RatSprite.java",
        evidence_line_number=10,
        class_evidence_member_path="root/src/RatSprite.java",
        class_evidence_line_number=1,
        clone_of=None,
        source_asset_key="RAT",
        source_sheet_path="sprites/rat.png",
        source_sheet_member_path="root/assets/sprites/rat.png",
        source_sheet_sha256=sheet_sha256,
        sheet_width=64,
        sheet_height=16,
        film=None,
        variant_kind="frame_indices",
        variant_ordinal=0,
        frame_indices=indices,
        frame_cells=tuple(_cell(index) for index in indices),
        source_fps_values=fps_values,
        source_fps_expression=str(fps_values),
        source_looping=source_looping,
        source_looping_expression=str(source_looping).lower(),
        frame_expression_order=tuple(str(index) for index in indices),
        frame_variable_expressions=(),
        source_context="constructor",
        inherited=False,
        ambiguity_reasons=(),
        timing_mode=timing_mode,  # type: ignore[arg-type]
        duration_ms=duration_ms,
    )


def _plan(sheet_sha256: str, archive_sha256: str) -> ShatteredPixelDungeonProjectionPlan:
    records = (
        _record(
            key="spd-sequence-v1:exact-repeat",
            action="idle",
            normalized_action="idle",
            indices=(0, 0, 1),
            fps_values=(4,),
            timing_mode="exact_positive_fps",
            duration_ms=250.0,
            source_looping=True,
            sheet_sha256=sheet_sha256,
        ),
        _record(
            key="spd-sequence-v1:pose",
            action="idle",
            normalized_action="idle",
            indices=(2,),
            fps_values=(0,),
            timing_mode="pose_only_zero_fps",
            duration_ms=None,
            source_looping=True,
            sheet_sha256=sheet_sha256,
        ),
        _record(
            key="spd-sequence-v1:ambiguous-a",
            action="run",
            normalized_action="run",
            indices=(3, 4),
            fps_values=(10, 15),
            timing_mode="ambiguous_fps",
            duration_ms=None,
            source_looping=False,
            sheet_sha256=sheet_sha256,
        ),
        _record(
            key="spd-sequence-v1:ambiguous-b",
            action="run",
            normalized_action="run",
            indices=(5, 6),
            fps_values=(10, 15),
            timing_mode="ambiguous_fps",
            duration_ms=None,
            source_looping=False,
            sheet_sha256=sheet_sha256,
        ),
    )
    return ShatteredPixelDungeonProjectionPlan(
        archive_sha256=archive_sha256,
        repository_commit="test-commit",
        records=records,
        exclusions=(),
        assets_java_member_path="root/src/Assets.java",
        license_evidence_member_paths=("root/LICENSE.txt",),
        attribution_evidence_member_paths=("root/README.md",),
    )


def _indexed_database(
    tmp_path: Path,
) -> tuple[IndexDB, ShatteredPixelDungeonProjectionPlan]:
    archive_bytes = b"synthetic archive bytes"
    sheet_bytes = b"synthetic lossless sheet bytes"
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    sheet_sha256 = hashlib.sha256(sheet_bytes).hexdigest()
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    database.register_source(
        source_id="shattered_pixel_dungeon",
        kind="git_archive",
        name="Shattered Pixel Dungeon test fixture",
        root_url="https://example.invalid/spd",
        adapter_version="test",
    )
    item_id = database.upsert_item(
        source_id="shattered_pixel_dungeon",
        external_id="test-commit",
        canonical_url="https://example.invalid/spd/test-commit",
    )
    database.register_blob(
        sha256=archive_sha256,
        size_bytes=len(archive_bytes),
        storage_path=tmp_path / "archive.zip",
        mime_type="application/zip",
    )
    database.register_blob(
        sha256=sheet_sha256,
        size_bytes=len(sheet_bytes),
        storage_path=tmp_path / "rat.png",
        mime_type="image/png",
    )
    database.link_item_blob(
        item_id=item_id,
        blob_sha256=archive_sha256,
        role="source_archive",
    )
    paths = (
        "root/LICENSE.txt",
        "root/README.md",
        "root/src/Assets.java",
        "root/src/RatSprite.java",
        "root/assets/sprites/rat.png",
    )
    database.upsert_archive_inventory(
        archive_blob_sha256=archive_sha256,
        archive_format="zip",
        member_count=len(paths),
        file_count=len(paths),
        total_uncompressed_bytes=1,
        total_compressed_bytes=1,
        inventory_sha256="0" * 64,
    )
    database.upsert_archive_members(
        archive_blob_sha256=archive_sha256,
        members=[
            {
                "ordinal": ordinal,
                "member_path": path,
                "normalized_path": path,
                "member_kind": "file",
                "size_bytes": 1,
                "compressed_bytes": 1,
                "extracted_blob_sha256": (sheet_sha256 if path.endswith("rat.png") else None),
            }
            for ordinal, path in enumerate(paths)
        ],
    )
    return database, _plan(sheet_sha256, archive_sha256)


def test_projection_is_idempotent_and_preserves_exact_source_facts(
    tmp_path: Path,
) -> None:
    database, plan = _indexed_database(tmp_path)
    readiness = check_shattered_pixel_dungeon_projection_readiness(database.path, plan)
    assert readiness.ready
    assert readiness.required_member_count == 5

    first = project_shattered_pixel_dungeon_audit(database, plan, TAXONOMY)
    second = project_shattered_pixel_dungeon_audit(database, plan, TAXONOMY)
    assert first.created_sequences == 4
    assert first.reused_sequences == 0
    assert second.created_sequences == 0
    assert second.reused_sequences == 4
    assert first.projected_entities == second.projected_entities == 1
    assert first.projected_frame_occurrences == second.projected_frame_occurrences == 8
    assert first.rights_observations_added == second.rights_observations_added == 0

    with database.connect() as connection:
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "entities",
                "sequences",
                "sequence_source_keys",
                "sequence_subjects",
                "motion_annotations",
                "sequence_frames",
                "sequence_occurrences",
                "rights_observations",
            )
        }
        assert counts == {
            "entities": 1,
            "sequences": 4,
            "sequence_source_keys": 4,
            "sequence_subjects": 4,
            "motion_annotations": 4,
            "sequence_frames": 8,
            "sequence_occurrences": 20,
            "rights_observations": 0,
        }

        exact_id = connection.execute(
            """
            SELECT sequence_id FROM sequence_source_keys
            WHERE external_sequence_key='spd-sequence-v1:exact-repeat'
            """
        ).fetchone()[0]
        exact = connection.execute("SELECT * FROM sequences WHERE id=?", (exact_id,)).fetchone()
        assert exact["source_blob_sha256"] == plan.records[0].source_sheet_sha256
        assert (exact["width"], exact["height"], exact["frame_count"]) == (8, 8, 3)
        assert (exact["loop_mode"], exact["direction"]) == ("loop", "unknown")
        assert json.loads(exact["metadata_json"])["rights_scope"]["scope"] == (
            "repository_level_only"
        )
        exact_frames = connection.execute(
            """
            SELECT * FROM sequence_frames
            WHERE sequence_id=? ORDER BY ordinal
            """,
            (exact_id,),
        ).fetchall()
        assert [row["source_frame_index"] for row in exact_frames] == [0, 0, 1]
        assert [row["duration_ms"] for row in exact_frames] == [250.0, 250.0, 250.0]
        assert [row["phase"] for row in exact_frames] == [0.0, 1 / 3, 2 / 3]
        assert {row["direction"] for row in exact_frames} == {"unknown"}
        assert {row["view"] for row in exact_frames} == {"top_down"}

        pose_id = connection.execute(
            """
            SELECT sequence_id FROM sequence_source_keys
            WHERE external_sequence_key='spd-sequence-v1:pose'
            """
        ).fetchone()[0]
        pose = connection.execute(
            "SELECT loop_mode, metadata_json FROM sequences WHERE id=?", (pose_id,)
        ).fetchone()
        assert pose["loop_mode"] == "unknown"
        assert json.loads(pose["metadata_json"])["pose_only"] is True
        pose_frame = connection.execute(
            "SELECT duration_ms FROM sequence_frames WHERE sequence_id=?", (pose_id,)
        ).fetchone()
        assert pose_frame["duration_ms"] is None
        pose_motion = connection.execute(
            "SELECT loopable, conditioning_json FROM motion_annotations WHERE sequence_id=?",
            (pose_id,),
        ).fetchone()
        assert pose_motion["loopable"] is None
        pose_conditioning = json.loads(pose_motion["conditioning_json"])
        assert pose_conditioning["source_fps_values"] == [0]
        assert pose_conditioning["source_looping_values"] == [True]

        ambiguous = connection.execute(
            """
            SELECT s.id, s.metadata_json
            FROM sequences s
            JOIN sequence_source_keys sk ON sk.sequence_id=s.id
            WHERE sk.external_sequence_key LIKE 'spd-sequence-v1:ambiguous-%'
            ORDER BY sk.external_sequence_key
            """
        ).fetchall()
        assert len(ambiguous) == 2
        assert all(
            json.loads(row["metadata_json"])["source_fps_values"] == [10, 15] for row in ambiguous
        )
        assert all(
            connection.execute(
                """
                SELECT COUNT(*) FROM sequence_frames
                WHERE sequence_id=? AND duration_ms IS NULL
                """,
                (row["id"],),
            ).fetchone()[0]
            == 2
            for row in ambiguous
        )


def test_readiness_reports_hash_mismatch_without_writing(tmp_path: Path) -> None:
    database, plan = _indexed_database(tmp_path)
    broken_record = plan.records[0]
    broken_plan = ShatteredPixelDungeonProjectionPlan(
        archive_sha256=plan.archive_sha256,
        repository_commit=plan.repository_commit,
        records=(replace(broken_record, source_sheet_sha256="f" * 64),),
        exclusions=(),
        assets_java_member_path=plan.assets_java_member_path,
        license_evidence_member_paths=plan.license_evidence_member_paths,
        attribution_evidence_member_paths=plan.attribution_evidence_member_paths,
    )
    before = database.path.stat().st_mtime_ns
    readiness = check_shattered_pixel_dungeon_projection_readiness(database.path, broken_plan)
    after = database.path.stat().st_mtime_ns
    assert not readiness.ready
    assert readiness.sheet_hash_mismatches
    assert before == after


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact CAS archive is not present")
def test_exact_archive_projection_plan_is_conservative_and_complete() -> None:
    plan = plan_known_shattered_pixel_dungeon_projection(EXACT_ARCHIVE)
    assert plan.projected_entity_count == 103
    assert plan.projected_sequence_count == 631
    assert plan.projected_frame_occurrence_count == 2439
    assert plan.exact_timing_sequence_count == 615
    assert plan.pose_only_sequence_count == 10
    assert plan.ambiguous_timing_sequence_count == 6
    assert plan.excluded_candidate_sequence_count == 28
    assert plan.excluded_candidate_frame_occurrence_count == 138
    assert len({record.sequence_source_key for record in plan.records}) == 631
    assert {exclusion.reason for exclusion in plan.exclusions} == {
        "multiple_runtime_source_sheets_are_not_safely_correlated"
    }
    assert all(record.frame_count == len(record.frame_indices) for record in plan.records)
    assert all(
        record.duration_ms is not None and record.duration_ms > 0
        for record in plan.records
        if record.timing_mode == "exact_positive_fps"
    )
    assert all(
        record.duration_ms is None
        for record in plan.records
        if record.timing_mode != "exact_positive_fps"
    )
    assert all(record.sequence_source_key.startswith("spd-sequence-v1:") for record in plan.records)
