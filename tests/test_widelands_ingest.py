from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.widelands import (
    EXPECTED_WIDELANDS_ARCHIVE_SHA256,
    WIDELANDS_COMMIT,
    audit_widelands_archive,
    known_widelands_cas_path,
)
from spritelab.db import IndexDB
from spritelab.ingest.widelands import (
    EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256,
    SOURCE_ID,
    WidelandsProjectionPlan,
    check_widelands_projection_readiness,
    plan_known_widelands_projection,
    plan_widelands_projection,
    project_widelands_audit,
)
from spritelab.taxonomy import load_taxonomy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXACT_ARCHIVE = known_widelands_cas_path(PROJECT_ROOT / "data" / "raw")
LIVE_INDEX = PROJECT_ROOT / "data" / "index" / "spritelab.sqlite3"
TAXONOMY = load_taxonomy(PROJECT_ROOT / "configs" / "taxonomy.toml")

_ENGINE_FILES = (
    "src/logic/map_objects/map_object.cc",
    "src/graphic/animation/animation.cc",
    "src/graphic/animation/nonpacked_animation.cc",
    "src/graphic/animation/spritesheet_animation.cc",
    "src/io/filesystem/filesystem.cc",
)


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _fixture_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / "widelands-projection-fixture.zip"
    root = "widelands-projection-fixture"
    worker_root = "data/tribes/workers/test/fixture_worker"
    critter_root = "data/world/critters/fixture_wolf"
    worker_manifest = r"""
local dirname = path.dirname(__file__)
wl.Descriptions():new_worker_type {
   name = "fixture_worker",
   animation_directory = dirname,
   spritesheets = {
      idle = {
         fps = 5,
         frames = 2,
         rows = 1,
         columns = 2,
         hotspot = { 2, 4 },
         play_once = true,
      },
   },
}
"""
    critter_manifest = r"""
local dirname = path.dirname(__file__)
wl.Descriptions():new_critter_type {
   name = "fixture_wolf",
   animation_directory = dirname,
   spritesheets = {
      idle = {
         frames = 1,
         rows = 1,
         columns = 1,
         hotspot = { 2, 4 },
      },
      eating = {
         basename = "idle",
         frames = 1,
         rows = 1,
         columns = 1,
         hotspot = { 2, 4 },
      },
      walk = {
         fps = 10,
         frames = 2,
         rows = 1,
         columns = 2,
         directional = true,
         hotspot = { 3, 5 },
      },
   },
}
"""
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for logical_path, payload in {
            "COPYING": b"GNU GENERAL PUBLIC LICENSE Version 2\n",
            "CREDITS": b"Fixture contributors\n",
            "data/txts/LICENSE.lua": b"GPL V2.0 or any later version\n",
            "data/txts/developers.json": b'{"developers": []}\n',
        }.items():
            archive.writestr(f"{root}/{logical_path}", payload)
        for logical_path in _ENGINE_FILES:
            archive.writestr(f"{root}/{logical_path}", f"evidence:{logical_path}\n")
        archive.writestr(f"{root}/{worker_root}/init.lua", worker_manifest)
        archive.writestr(f"{root}/{critter_root}/init.lua", critter_manifest)
        archive.writestr(
            f"{root}/{worker_root}/idle_1.png",
            _png_bytes((8, 4), (20, 40, 60, 255)),
        )
        archive.writestr(
            f"{root}/{worker_root}/idle_1_pc.png",
            _png_bytes((8, 4), (255, 0, 0, 255)),
        )
        archive.writestr(
            f"{root}/{critter_root}/idle_1.png",
            _png_bytes((4, 6), (70, 80, 90, 255)),
        )
        for ordinal, direction in enumerate(("ne", "e", "se", "sw", "w", "nw")):
            archive.writestr(
                f"{root}/{critter_root}/walk_{direction}_1.png",
                _png_bytes((8, 6), (90 + ordinal, 100, 110, 255)),
            )
    return archive_path


def _indexed_database(
    tmp_path: Path,
    archive_path: Path,
) -> tuple[IndexDB, WidelandsProjectionPlan]:
    audit = audit_widelands_archive(archive_path)
    plan = plan_widelands_projection(audit, TAXONOMY)
    database = IndexDB(tmp_path / "widelands-index.sqlite3")
    database.initialize()
    database.register_source(
        source_id=SOURCE_ID,
        kind="git_archive",
        name="Widelands projection fixture",
        root_url="https://example.invalid/widelands",
        adapter_version="test",
    )
    item_id = database.upsert_item(
        source_id=SOURCE_ID,
        external_id=audit.commit,
        canonical_url=f"https://example.invalid/widelands/{audit.commit}",
    )
    database.register_blob(
        sha256=audit.archive_sha256,
        size_bytes=archive_path.stat().st_size,
        storage_path=archive_path,
        mime_type="application/zip",
    )
    database.link_item_blob(
        item_id=item_id,
        blob_sha256=audit.archive_sha256,
        role="source_archive",
    )
    layer_hashes = dict(plan.required_source_layer_hashes)
    for layer_sha256 in sorted(set(layer_hashes.values())):
        database.register_blob(
            sha256=layer_sha256,
            size_bytes=1,
            storage_path=tmp_path / f"{layer_sha256}.png",
            mime_type="image/png",
        )
    database.upsert_archive_inventory(
        archive_blob_sha256=audit.archive_sha256,
        archive_format="zip",
        member_count=len(plan.required_member_paths),
        file_count=len(plan.required_member_paths),
        total_uncompressed_bytes=1,
        total_compressed_bytes=1,
        inventory_sha256="0" * 64,
    )
    database.upsert_archive_members(
        archive_blob_sha256=audit.archive_sha256,
        members=[
            {
                "ordinal": ordinal,
                "member_path": member_path,
                "normalized_path": member_path,
                "member_kind": "file",
                "size_bytes": 1,
                "compressed_bytes": 1,
                "extracted_blob_sha256": layer_hashes.get(member_path),
            }
            for ordinal, member_path in enumerate(plan.required_member_paths)
        ],
    )
    return database, plan


def test_plan_projects_only_complete_pixels_and_retains_exact_modular_pairs(
    tmp_path: Path,
) -> None:
    audit = audit_widelands_archive(_fixture_archive(tmp_path))
    plan = plan_widelands_projection(audit, TAXONOMY)

    assert plan.projected_sequence_count == 8
    assert plan.projected_entity_count == 1
    assert plan.projected_frame_count == 14
    assert plan.projected_animated_sequence_count == 6
    assert plan.projected_static_sequence_count == 2
    assert plan.projected_loop_count == 8
    assert plan.projected_one_shot_count == 0
    assert plan.modular_exclusion_count == 1
    assert plan.excluded_frame_count == 2
    assert len(plan.exclusions) == 1
    assert len(plan.required_source_layer_hashes) == 9
    assert len(plan.required_member_paths) == 20
    assert plan.projected_occurrence_link_count == 88

    modular = plan.exclusions[0]
    assert modular.entity_id == "fixture_worker"
    assert modular.declared_name == "idle"
    assert modular.loop_mode == "one_shot"
    assert modular.runtime_composite_status == ("modular_body_plus_playercolor_mask_unresolved")
    assert modular.required_runtime_parameters == ("player_color",)
    assert modular.reasons == (
        "runtime_composite:playercolor_mask_required",
        "runtime_composite:player_color_parameter_unbound",
        "runtime_composite:engine_blend_not_materialized",
    )
    assert len(modular.body_layers) == len(modular.playercolor_mask_layers) == 1
    assert [frame.body_layer_index for frame in modular.frames] == [0, 0]
    assert [frame.playercolor_mask_layer_index for frame in modular.frames] == [0, 0]
    assert [frame.duration_milliseconds for frame in modular.frames] == [200, 200]

    northeast = next(
        record
        for record in plan.records
        if record.declared_name == "walk" and record.source_direction == "northeast"
    )
    assert northeast.entity.entity_id == "fixture_wolf"
    assert northeast.entity.entity_class == "animal"
    assert northeast.canonical_direction == "up_right"
    assert northeast.normalized_action == "walk"
    assert northeast.playercolor_mask_layers == ()
    assert northeast.runtime_composite_status == "exact_unmasked_complete_entity"
    assert [frame.source_frame_index for frame in northeast.frames] == [0, 1]
    assert [frame.duration_milliseconds for frame in northeast.frames] == [100, 100]
    assert [(frame.left, frame.top, frame.right, frame.bottom) for frame in northeast.frames] == [
        (0, 0, 4, 6),
        (4, 0, 8, 6),
    ]

    idle = next(record for record in plan.records if record.declared_name == "idle")
    eating = next(record for record in plan.records if record.declared_name == "eating")
    assert idle.normalized_action == "idle"
    assert eating.normalized_action == "eat"
    assert idle.track_content_deduplication_key == eating.track_content_deduplication_key
    assert idle.sequence_source_key != eating.sequence_source_key
    assert (
        plan.projection_manifest_sha256
        == plan_widelands_projection(audit, TAXONOMY).projection_manifest_sha256
    )
    changed = plan_widelands_projection(replace(audit, commit="different-source-fact"), TAXONOMY)
    assert changed.projection_manifest_sha256 != plan.projection_manifest_sha256


def test_temp_projection_is_idempotent_and_preserves_order_timing_and_provenance(
    tmp_path: Path,
) -> None:
    archive_path = _fixture_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    before = database.path.stat().st_mtime_ns
    readiness = check_widelands_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert readiness.ready
    assert readiness.required_member_count == readiness.present_member_count == 20
    assert readiness.required_source_layer_count == 9
    assert readiness.present_source_layer_blob_count == 9
    assert readiness.required_projected_body_count == 7
    assert readiness.present_projected_body_blob_count == 7
    assert readiness.required_modular_body_count == 1
    assert readiness.present_modular_body_blob_count == 1
    assert readiness.required_modular_mask_count == 1
    assert readiness.present_modular_mask_blob_count == 1
    assert before == after

    first = project_widelands_audit(database, plan, TAXONOMY)
    second = project_widelands_audit(database, plan, TAXONOMY)
    assert (first.created_sequences, first.reused_sequences) == (8, 0)
    assert (second.created_sequences, second.reused_sequences) == (0, 8)
    assert first.projected_entities == second.projected_entities == 1
    assert first.projected_frames == second.projected_frames == 14
    assert first.occurrence_links == second.occurrence_links == 88
    assert first.modular_exclusions == second.modular_exclusions == 1
    assert first.excluded_frames == second.excluded_frames == 2
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
            "sequences": 8,
            "sequence_source_keys": 8,
            "sequence_subjects": 8,
            "motion_annotations": 8,
            "sequence_frames": 14,
            "sequence_occurrences": 88,
            "rights_observations": 0,
        }
        sequence_rows = connection.execute(
            "SELECT id, loop_mode, action, direction, metadata_json FROM sequences"
        ).fetchall()
        northeast = next(
            row
            for row in sequence_rows
            if json.loads(row["metadata_json"])["declared_name"] == "walk"
            and json.loads(row["metadata_json"])["source_direction"] == "northeast"
        )
        metadata = json.loads(northeast["metadata_json"])
        assert northeast["loop_mode"] == "loop"
        assert northeast["action"] == "walk"
        assert northeast["direction"] == "up_right"
        assert metadata["source_frame_index_order"] == [0, 1]
        assert metadata["duration_ms_per_occurrence"] == [100, 100]
        assert metadata["runtime_composite_status"] == ("exact_unmasked_complete_entity")
        assert metadata["playercolor_mask_required"] is False
        assert metadata["playercolor_mask_layers"] == []
        assert metadata["rights_scope"]["license_expression"] == "GPL-2.0-or-later"

        frames = connection.execute(
            "SELECT ordinal, source_blob_sha256, source_frame_index, duration_ms, phase, "
            "direction, metadata_json FROM sequence_frames WHERE sequence_id=? "
            "ORDER BY ordinal",
            (northeast["id"],),
        ).fetchall()
        assert [row["ordinal"] for row in frames] == [0, 1]
        assert [row["source_frame_index"] for row in frames] == [0, 1]
        assert [row["duration_ms"] for row in frames] == [100, 100]
        assert [row["phase"] for row in frames] == [0.0, 0.5]
        assert {row["direction"] for row in frames} == {"up_right"}
        assert len({row["source_blob_sha256"] for row in frames}) == 1
        frame_metadata = json.loads(frames[0]["metadata_json"])
        assert frame_metadata["frame_rect"] == {
            "bottom": 6,
            "coordinate_space": "source_image",
            "height": 6,
            "left": 0,
            "right": 4,
            "top": 0,
            "width": 4,
        }
        assert frame_metadata["playercolor_mask_layer"] is None

        occurrences = connection.execute(
            "SELECT occurrence_role, metadata_json FROM sequence_occurrences "
            "WHERE sequence_id=? ORDER BY occurrence_role",
            (northeast["id"],),
        ).fetchall()
        assert {row["occurrence_role"] for row in occurrences} == {
            "widelands_collection_rights_evidence",
            "widelands_complete_unmasked_body_source",
            "widelands_engine_animation_semantics",
            "widelands_entity_animation_manifest",
        }
        assert len(occurrences) == 11
        assert all("playercolor_mask" not in row["occurrence_role"] for row in occurrences)


def test_readiness_reports_hash_mismatch_without_writing(tmp_path: Path) -> None:
    archive_path = _fixture_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    wrong_hash = "f" * 64
    database.register_blob(
        sha256=wrong_hash,
        size_bytes=1,
        storage_path=tmp_path / "wrong.png",
        mime_type="image/png",
    )
    member_path = plan.required_source_layer_hashes[0][0]
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE archive_members SET extracted_blob_sha256=?
            WHERE archive_blob_sha256=? AND member_path=?
            """,
            (wrong_hash, plan.archive_sha256, member_path),
        )
    before = database.path.stat().st_mtime_ns
    readiness = check_widelands_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert not readiness.ready
    assert readiness.missing_source_layer_blobs == ()
    assert len(readiness.source_layer_hash_mismatches) == 1
    assert member_path in readiness.source_layer_hash_mismatches[0]
    assert before == after
    with pytest.raises(ValueError, match="CAS hash mismatch"):
        project_widelands_audit(database, plan, TAXONOMY)


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact Widelands CAS archive missing")
def test_exact_pinned_plan_is_deterministic_and_conservative() -> None:
    plan = plan_known_widelands_projection(EXACT_ARCHIVE, TAXONOMY)

    assert plan.archive_sha256 == EXPECTED_WIDELANDS_ARCHIVE_SHA256
    assert plan.repository_commit == WIDELANDS_COMMIT
    assert plan.source_audit_record_sha256 == (
        "2208e5ef94bbe6adbe80e2c668336ef04bcd79997e5794d9b81df5ded9ad9a86"
    )
    assert plan.projection_manifest_sha256 == EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256
    assert plan.projected_sequence_count == 193
    assert plan.projected_entity_count == 22
    assert plan.projected_frame_count == 3_272
    assert plan.projected_animated_sequence_count == 183
    assert plan.projected_static_sequence_count == 10
    assert plan.projected_loop_count == 193
    assert plan.projected_one_shot_count == 0
    assert plan.modular_exclusion_count == len(plan.exclusions) == 2_082
    assert plan.excluded_frame_count == 31_142
    assert plan.projected_occurrence_link_count == 2_123
    assert len(plan.required_member_paths) == 4_180
    assert len(plan.required_source_layer_hashes) == 3_994
    assert len(set(dict(plan.required_source_layer_hashes).values())) == 3_548
    assert len(plan.projected_body_layer_hashes) == 154
    assert len(plan.modular_body_layer_hashes) == 1_920
    assert len(plan.modular_mask_layer_hashes) == 1_920
    assert len(plan.required_evidence_hashes) == 4_003
    assert len(set(dict(plan.required_evidence_hashes).values())) == 3_557
    assert plan.duplicate_source_layer_hash_groups == 207
    assert plan.duplicate_source_layer_hash_excess == 446
    assert len({record.sequence_source_key for record in plan.records}) == 193
    assert len({record.entity.entity_external_key for record in plan.records}) == 22
    assert {record.normalized_action for record in plan.records} == {
        "carry",
        "eat",
        "idle",
        "walk",
    }
    assert all(exclusion.has_exact_modular_pairs for exclusion in plan.exclusions)
    assert all(
        exclusion.required_runtime_parameters == ("player_color",) for exclusion in plan.exclusions
    )


@pytest.mark.skipif(
    not (EXACT_ARCHIVE.is_file() and LIVE_INDEX.is_file()),
    reason="exact Widelands CAS archive or live index missing",
)
def test_live_readiness_is_query_only() -> None:
    plan = plan_known_widelands_projection(EXACT_ARCHIVE, TAXONOMY)
    readiness = check_widelands_projection_readiness(LIVE_INDEX, plan)

    assert readiness.archive_sha256 == EXPECTED_WIDELANDS_ARCHIVE_SHA256
    assert readiness.projection_manifest_sha256 == (EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256)
    assert readiness.required_member_count == 4_180
    assert readiness.required_source_layer_count == 3_994
    if not readiness.source_registered:
        assert not readiness.ready
