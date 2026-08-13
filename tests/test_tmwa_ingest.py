from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pytest

from spritelab.adapters.tmwa import SOURCE_ID, audit_known_tmwa_archive
from spritelab.db import IndexDB
from spritelab.ingest.tmwa import (
    EXPECTED_TMWA_PROJECTION_MANIFEST_SHA256,
    TmwaProjectionPlan,
    check_tmwa_projection_readiness,
    plan_known_tmwa_projection,
    plan_tmwa_projection,
    project_tmwa_audit,
)
from spritelab.taxonomy import load_taxonomy

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "7b/7a/7b7a8c61eca284b35ddedaa1af4210a1694933e36ebdc4299bd73395a6532152"
)
LIVE_INDEX = Path("data/index/spritelab.sqlite3")
TAXONOMY = load_taxonomy(Path("configs/taxonomy.toml"))


@lru_cache(maxsize=1)
def _known_plan() -> TmwaProjectionPlan:
    return plan_known_tmwa_projection(EXACT_ARCHIVE)


def _modified_at(value: tuple[int, int, int, int, int, int]) -> str:
    return (
        f"{value[0]:04d}-{value[1]:02d}-{value[2]:02d}T{value[3]:02d}:{value[4]:02d}:{value[5]:02d}"
    )


def _indexed_database(tmp_path: Path, plan: TmwaProjectionPlan) -> IndexDB:
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    database.register_source(
        source_id=SOURCE_ID,
        kind="git_archive",
        name="TMWA projection fixture",
        root_url="https://github.com/themanaworld/tmwa-client-data",
        adapter_version="test",
    )
    item_id = database.upsert_item(
        source_id=SOURCE_ID,
        external_id=plan.repository_commit,
        canonical_url=(
            f"https://github.com/themanaworld/tmwa-client-data/commit/{plan.repository_commit}"
        ),
    )
    database.register_blob(
        sha256=plan.archive_sha256,
        size_bytes=65_557_370,
        storage_path=EXACT_ARCHIVE,
        mime_type="application/zip",
    )
    database.link_item_blob(
        item_id=item_id,
        blob_sha256=plan.archive_sha256,
        role="source_archive",
    )
    database.upsert_archive_inventory(
        archive_blob_sha256=plan.archive_sha256,
        archive_format="zip",
        member_count=plan.counts.zip_member_count,
        file_count=plan.counts.regular_file_member_count,
        total_uncompressed_bytes=plan.counts.expanded_member_bytes,
        total_compressed_bytes=plan.counts.compressed_member_bytes,
        inventory_sha256=plan.archive_inventory_sha256,
    )
    database.upsert_archive_members(
        archive_blob_sha256=plan.archive_sha256,
        members=[
            {
                "ordinal": member.ordinal,
                "member_path": member.member_path,
                "normalized_path": member.normalized_path,
                "member_kind": member.member_kind,
                "size_bytes": member.size_bytes,
                "compressed_bytes": member.compressed_bytes,
                "crc32": member.crc32,
                "compression_method": member.compression_method,
                "modified_at": _modified_at(member.modified_at),
            }
            for member in plan.archive_members
        ],
    )
    database.register_archive_extractions(
        archive_blob_sha256=plan.archive_sha256,
        extracted=[
            {
                "ordinal": member.ordinal,
                "sha256": member.content_sha256,
                "size_bytes": member.size_bytes,
                "storage_path": tmp_path / str(member.content_sha256),
            }
            for member in plan.required_members
        ],
        selected_role="tmwa_projection_evidence",
    )
    return database


def test_exact_plan_is_pure_deterministic_and_conservative() -> None:
    archive_stat = EXACT_ARCHIVE.stat()
    audit = audit_known_tmwa_archive(EXACT_ARCHIVE)
    first = plan_tmwa_projection(audit)
    second = plan_tmwa_projection(audit)
    after = EXACT_ARCHIVE.stat()

    assert (archive_stat.st_size, archive_stat.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert first.projection_manifest_sha256 == second.projection_manifest_sha256
    assert first.projection_manifest_sha256 == EXPECTED_TMWA_PROJECTION_MANIFEST_SHA256
    assert (
        first.projected_sequence_count,
        first.projected_frame_count,
        first.projected_definition_count,
        first.projected_source_image_count,
        first.projected_semantic_entity_count,
        len(first.exclusions),
        len(first.required_members),
    ) == (853, 4_153, 53, 53, 54, 19_865, 4_169)
    assert Counter(item.source_action for item in first.records) == {
        "stand": 247,
        "attack": 238,
        "walk": 213,
        "dead": 121,
        "spawn": 17,
        "attack_magic": 8,
        "attack_splash": 8,
        "hurt": 1,
    }
    assert Counter(item.loop_mode for item in first.records) == {
        "loop": 280,
        "hold": 305,
        "one_shot_return_to_stand": 268,
    }
    assert all(item.variant_index == 0 for item in first.records)
    assert all(item.source_image == item.frames[0].source_image for item in first.records)
    assert all(frame.palette_expression is None for item in first.records for frame in item.frames)
    assert all(
        relation.layer_role == "complete_single_layer_entity"
        and relation.layer_count == 1
        and relation.palette_expression is None
        for item in first.records
        for relation in item.entity_relations
    )
    assert all(item.rights_assessment.status == "documented_path_claim" for item in first.records)
    reasons = Counter(reason for item in first.exclusions for reason in item.reasons)
    assert reasons["no_safe_complete_single_layer_monster_binding"] == 19_350
    assert reasons["imageset_palette_transform_unresolved"] == 10_774
    assert reasons["source_image_rights_license_missing"] == 7_333
    assert reasons["source_image_rights_unclaimed"] == 1_070
    assert reasons["source_image_rights_unresolved_contributor_or_license"] == 754
    assert reasons["source_image_rights_contradictory"] == 169
    assert reasons["runtime_control_flow_track"] == 7


def test_live_readiness_is_exact_and_query_only() -> None:
    readiness = check_tmwa_projection_readiness(LIVE_INDEX, _known_plan())
    assert readiness.query_only_enabled
    assert readiness.ready
    assert readiness.archive_inventory_exact
    assert readiness.source_item_count == 1
    assert (
        readiness.planned_archive_member_count,
        readiness.present_archive_member_count,
    ) == (5_082, 5_082)
    assert (
        readiness.required_member_count,
        readiness.present_member_count,
        readiness.extracted_member_blob_count,
    ) == (4_169, 4_169, 4_169)
    assert not readiness.missing_member_paths
    assert not readiness.missing_archive_member_paths
    assert not readiness.member_metadata_mismatches
    assert not readiness.member_hash_mismatches
    assert not readiness.unregistered_member_blobs


def test_exact_projection_is_idempotent_on_temp_database_only(tmp_path: Path) -> None:
    plan = _known_plan()
    database = _indexed_database(tmp_path, plan)
    readiness = check_tmwa_projection_readiness(database.path, plan)
    assert readiness.ready
    assert readiness.query_only_enabled

    with database.transaction():
        first = project_tmwa_audit(database, plan, TAXONOMY)
    with database.transaction():
        second = project_tmwa_audit(database, plan, TAXONOMY)

    assert (first.created_sequences, first.reused_sequences) == (853, 0)
    assert (second.created_sequences, second.reused_sequences) == (0, 853)
    assert first.projected_frames == second.projected_frames == 4_153
    assert first.projected_resource_entities == second.projected_resource_entities == 53
    assert first.projected_semantic_entities == second.projected_semantic_entities == 54
    assert first.rights_observations_added == second.rights_observations_added == 0

    with database.connect() as connection:
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "entities",
                "sequences",
                "sequence_source_keys",
                "motion_annotations",
                "sequence_frames",
                "rights_observations",
            )
        }
        assert counts == {
            "entities": 107,
            "sequences": 853,
            "sequence_source_keys": 853,
            "motion_annotations": 853,
            "sequence_frames": 4_153,
            "rights_observations": 0,
        }
        rows = connection.execute(
            """
            SELECT s.id, s.loop_mode, s.action, s.direction, s.metadata_json,
                   e.entity_class AS primary_entity_class,
                   e.display_name AS primary_display_name
            FROM sequences s
            JOIN sequence_subjects ss ON ss.sequence_id=s.id AND ss.role='primary'
            JOIN entities e ON e.id=ss.entity_id
            """
        ).fetchall()
        bat_attack = next(
            row
            for row in rows
            if (metadata := json.loads(row["metadata_json"]))["definition_logical_path"]
            == "graphics/sprites/monsters/bat.xml"
            and metadata["source_action"] == "attack"
            and metadata["direction_literal"] == "down"
        )
        metadata = json.loads(bat_attack["metadata_json"])
        assert (bat_attack["loop_mode"], bat_attack["action"], bat_attack["direction"]) == (
            "one_shot",
            "attack",
            "down",
        )
        assert metadata["loop_mode"] == "one_shot_return_to_stand"
        assert bat_attack["primary_entity_class"] == "animal"
        assert isinstance(bat_attack["primary_display_name"], str)
        assert bat_attack["primary_display_name"]
        assert (
            metadata["source_image"]["logical_path"]
            == metadata["rights"]["assessment"]["table_claims"][0]["scope_path"]
        )
        assert not metadata["geometry"]["crops_materialized"]
        assert not metadata["geometry"]["compositing_performed"]
        assert not metadata["geometry"]["recoloring_performed"]

        frames = connection.execute(
            """
            SELECT duration_ms, phase, metadata_json
            FROM sequence_frames WHERE sequence_id=? ORDER BY ordinal
            """,
            (bat_attack["id"],),
        ).fetchall()
        assert len(frames) == 4
        assert all(row["duration_ms"] > 0 for row in frames)
        assert [row["phase"] for row in frames] == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
        frame_metadata = json.loads(frames[0]["metadata_json"])
        assert frame_metadata["source_cell_xywh"] == [185, 0, 37, 58]
        left, top, width, height = frame_metadata["source_cell_xywh"]
        assert frame_metadata["frame_rect"] == {
            "bottom": top + height,
            "coordinate_space": "source_image",
            "height": height,
            "left": left,
            "right": left + width,
            "top": top,
            "width": width,
        }
        assert frame_metadata["source_coordinate_space_literal"] == "source_png"
        assert frame_metadata["normalized_loop_mode"] == "one_shot"
        assert frame_metadata["palette_expression"] is None
        assert not frame_metadata["pixel_transform_applied"]
        assert not frame_metadata["composite_applied"]


def test_strict_preflight_rejects_temp_member_hash_mismatch_before_writes(
    tmp_path: Path,
) -> None:
    plan = _known_plan()
    database = _indexed_database(tmp_path, plan)
    first_member = plan.required_members[0]
    replacement_hash = next(
        member.content_sha256
        for member in plan.required_members[1:]
        if member.content_sha256 != first_member.content_sha256
    )
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE archive_members SET extracted_blob_sha256=?
            WHERE archive_blob_sha256=? AND ordinal=?
            """,
            (replacement_hash, plan.archive_sha256, first_member.ordinal),
        )
    readiness = check_tmwa_projection_readiness(database.path, plan)
    assert not readiness.ready
    assert readiness.member_hash_mismatches == (first_member.member_path,)
    with pytest.raises(ValueError, match="extracted member hashes differ"):
        project_tmwa_audit(database, plan, TAXONOMY)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM sequences").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 0
