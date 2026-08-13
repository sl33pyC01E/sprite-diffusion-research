from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.wesnoth import (
    EXPECTED_WESNOTH_ARCHIVE_SHA256,
    WESNOTH_COMMIT,
    audit_wesnoth_archive,
    known_wesnoth_cas_path,
    primary_declarations_for,
)
from spritelab.db import IndexDB
from spritelab.ingest.wesnoth import (
    WesnothProjectionPlan,
    check_wesnoth_projection_readiness,
    plan_known_wesnoth_projection,
    plan_wesnoth_projection,
    project_wesnoth_audit,
)
from spritelab.taxonomy import load_taxonomy

EXACT_ARCHIVE = known_wesnoth_cas_path(Path("data/raw"))
LIVE_INDEX = Path("data/index/spritelab.sqlite3")
TAXONOMY = load_taxonomy(Path("configs/taxonomy.toml"))
PINNED_PROJECTION_MANIFEST_SHA256 = (
    "1712326c432e1f143857e8d41ef03889dd91d0ad0566a6e43c040b9aaf8d1da8"
)


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGBA", size, color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _fixture_wml() -> str:
    return """\
[unit_type]
    id=Fixture Ranger
    name= _ "Fixture Ranger"
    race=human
    image="units/humans/fixture.png"
    [standing_anim]
        direction=n
        start_time=0
        [frame]
            image="units/humans/fixture-stand-[1~2].png:[100,120]"
            layer=42
            offset="0~0.1"
            x=2
            y=-1
            directional_x=3
            directional_y=4
            auto_hflip=no
            auto_vflip=yes
        [/frame]
    [/standing_anim]
    [movement_anim]
        direction=s
        [frame]
            image="units/humans/fixture-move-[1~2].png:[80,140]"
        [/frame]
        [missile_frame]
            image="projectiles/fixture-bolt.png:70"
            layer=60
            offset=0.5
        [/missile_frame]
    [/movement_anim]
    [attack_anim]
        [frame]
            image="units/humans/fixture-attack.png:50~RC(magenta>red)"
        [/frame]
    [/attack_anim]
    [death]
        {SOUND:HIT fixture.ogg 0}
        [frame]
            image="{DEATH_IMAGE}:100"
        [/frame]
    [/death]
[/unit_type]
"""


def _synthetic_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / "wesnoth-fixture.zip"
    root = "wesnoth-fixture"
    images = {
        "data/core/images/units/humans/fixture.png": ((4, 5), (10, 20, 30, 255)),
        "data/core/images/units/humans/fixture-stand-1.png": (
            (4, 5),
            (20, 30, 40, 255),
        ),
        "data/core/images/units/humans/fixture-stand-2.png": (
            (4, 5),
            (30, 40, 50, 255),
        ),
        "data/core/images/units/humans/fixture-move-1.png": (
            (3, 4),
            (40, 50, 60, 255),
        ),
        "data/core/images/units/humans/fixture-move-2.png": (
            (5, 6),
            (50, 60, 70, 255),
        ),
        "data/core/images/units/humans/fixture-attack.png": (
            (4, 5),
            (60, 70, 80, 255),
        ),
        "data/core/images/projectiles/fixture-bolt.png": (
            (2, 2),
            (70, 80, 90, 255),
        ),
    }
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/data/core/units/humans/Fixture.cfg", _fixture_wml())
        archive.writestr(
            f"{root}/README.md",
            "Most art is GPL v2+; newer contributions are CC BY-SA 4.0.\n",
        )
        archive.writestr(f"{root}/COPYING", "GNU GENERAL PUBLIC LICENSE Version 2\n")
        archive.writestr(
            f"{root}/data/COPYING.txt",
            "GNU GENERAL PUBLIC LICENSE Version 2\n",
        )
        archive.writestr(
            f"{root}/copyrights.csv",
            "Date,File,License,Author,Notes,Needs Update,MD5\n"
            "2026/01/01,data/core/music/example.ogg,CC0,Artist,,,\n",
        )
        archive.writestr(f"{root}/src/units/animation.cpp", 'anim["cycles"] = true;\n')
        archive.writestr(f"{root}/src/units/frame.cpp", "result.auto_hflip = true;\n")
        for member_path, (size, color) in images.items():
            archive.writestr(f"{root}/{member_path}", _png_bytes(size, color))
    return archive_path


def _indexed_database(
    tmp_path: Path,
    archive_path: Path,
) -> tuple[IndexDB, WesnothProjectionPlan]:
    audit = audit_wesnoth_archive(archive_path)
    plan = plan_wesnoth_projection(audit)
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    database.register_source(
        source_id="wesnoth",
        kind="git_archive",
        name="Wesnoth projection fixture",
        root_url="https://example.invalid/wesnoth",
        adapter_version="test",
    )
    item_id = database.upsert_item(
        source_id="wesnoth",
        external_id=WESNOTH_COMMIT,
        canonical_url=f"https://example.invalid/wesnoth/{WESNOTH_COMMIT}",
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
    source_image_hashes = dict(plan.required_source_image_hashes)
    for image_sha256 in sorted(set(source_image_hashes.values())):
        database.register_blob(
            sha256=image_sha256,
            size_bytes=1,
            storage_path=tmp_path / f"{image_sha256}.png",
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
                "extracted_blob_sha256": source_image_hashes.get(member_path),
            }
            for ordinal, member_path in enumerate(plan.required_member_paths)
        ],
    )
    return database, plan


def test_plan_preserves_literal_tracks_and_quarantines_unsafe_without_repair(
    tmp_path: Path,
) -> None:
    audit = audit_wesnoth_archive(_synthetic_archive(tmp_path))
    plan = plan_wesnoth_projection(audit)

    assert plan.projected_entity_count == 1
    assert plan.projected_sequence_count == 2
    assert plan.projected_frame_occurrence_count == 4
    assert plan.projected_loop_count == 1
    assert plan.projected_one_shot_count == 1
    assert plan.projected_auxiliary_declaration_count == 1
    assert plan.projected_occurrence_link_count == 18
    assert plan.excluded_candidate_sequence_count == 2
    assert plan.excluded_candidate_frame_occurrence_count == 2
    assert plan.excluded_transformed_primary_frame_count == 1

    standing = next(record for record in plan.records if record.source_tag == "standing_anim")
    assert standing.entity.unit_id == "Fixture Ranger"
    assert standing.normalized_action == "idle"
    assert standing.direction_hint == "n"
    assert standing.loop_mode == "loop"
    assert [frame.duration_milliseconds for frame in standing.frames] == [100, 120]
    assert [frame.ordinal for frame in standing.frames] == [0, 1]
    assert all(frame.layer_literal == "42" for frame in standing.frames)
    assert all(frame.offset_literal == "0~0.1" for frame in standing.frames)
    assert all(frame.x_literal == "2" and frame.y_literal == "-1" for frame in standing.frames)
    assert all(frame.directional_x_literal == "3" for frame in standing.frames)
    assert all(frame.directional_y_literal == "4" for frame in standing.frames)
    assert all(not frame.effective_auto_hflip for frame in standing.frames)
    assert all(frame.effective_auto_vflip for frame in standing.frames)
    assert all(frame.inline_modifiers is None for frame in standing.frames)
    assert all(frame.lossless_source_pixels and frame.exact_timing for frame in standing.frames)

    movement = next(record for record in plan.records if record.source_tag == "movement_anim")
    assert movement.normalized_action == "move"
    assert movement.direction_hint == "s"
    assert movement.loop_mode == "one_shot"
    assert [frame.duration_milliseconds for frame in movement.frames] == [80, 140]
    assert (movement.sequence_width, movement.sequence_height) == (5, 6)
    assert not movement.source_dimensions_consistent
    assert len(movement.auxiliary_frame_declarations) == 1
    auxiliary = movement.auxiliary_frame_declarations[0]
    assert auxiliary.render_role == "projectile"
    assert auxiliary.expression == "projectiles/fixture-bolt.png:70"
    assert auxiliary.layer_literal == "60"
    assert auxiliary.offset_literal == "0.5"

    excluded_by_tag = {exclusion.animation.source_tag: exclusion for exclusion in plan.exclusions}
    attack = excluded_by_tag["attack_anim"]
    assert attack.transformed_primary_frame_count == 1
    assert "inline_image_path_function" in attack.reasons
    attack_frame = primary_declarations_for(attack.animation)[0].frames[0]
    assert attack_frame.inline_modifiers == "~RC(magenta>red)"
    assert not attack_frame.lossless_source_pixels
    death = excluded_by_tag["death"]
    assert "unexpanded_wml_macro" in death.reasons
    assert set(death.animation.macro_invocations) == {
        "DEATH_IMAGE",
        "SOUND:HIT fixture.ogg 0",
    }
    assert all(record.source_tag not in excluded_by_tag for record in plan.records)

    assert plan.rights.per_asset_license is None
    assert plan.rights.per_asset_attribution is None
    assert (
        plan.projection_manifest_sha256 == plan_wesnoth_projection(audit).projection_manifest_sha256
    )
    changed = plan_wesnoth_projection(replace(audit, commit="different-source-fact"))
    assert changed.projection_manifest_sha256 != plan.projection_manifest_sha256


def test_projection_is_idempotent_and_retains_timing_facing_context_and_rights_scope(
    tmp_path: Path,
) -> None:
    archive_path = _synthetic_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    before = database.path.stat().st_mtime_ns
    readiness = check_wesnoth_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns
    assert readiness.ready
    assert readiness.required_source_image_count == 4
    assert readiness.present_source_image_blob_count == 4
    assert before == after

    first = project_wesnoth_audit(database, plan, TAXONOMY)
    second = project_wesnoth_audit(database, plan, TAXONOMY)
    assert (first.created_sequences, first.reused_sequences) == (2, 0)
    assert (second.created_sequences, second.reused_sequences) == (0, 2)
    assert first.projected_entities == second.projected_entities == 1
    assert first.projected_frame_occurrences == second.projected_frame_occurrences == 4
    assert first.occurrence_links == second.occurrence_links == 18
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
            "sequences": 2,
            "sequence_source_keys": 2,
            "sequence_subjects": 2,
            "motion_annotations": 2,
            "sequence_frames": 4,
            "sequence_occurrences": 18,
            "rights_observations": 0,
        }

        sequence_rows = connection.execute(
            "SELECT id, width, height, loop_mode, action, direction, metadata_json FROM sequences"
        ).fetchall()
        standing = next(
            row
            for row in sequence_rows
            if json.loads(row["metadata_json"])["source_tag"] == "standing_anim"
        )
        standing_metadata = json.loads(standing["metadata_json"])
        assert (standing["width"], standing["height"]) == (4, 5)
        assert standing["loop_mode"] == "loop"
        assert standing["action"] == "idle"
        assert standing["direction"] == "up"
        assert standing_metadata["adapter_normalized_action"] == "idle"
        assert standing_metadata["source_direction_groups"] == [["n"]]
        assert standing_metadata["duration_ms_per_occurrence"] == [100, 120]
        assert standing_metadata["source_image_member_order"] == [
            frame.source_member_path
            for frame in next(
                record for record in plan.records if record.source_tag == "standing_anim"
            ).frames
        ]
        rights_scope = standing_metadata["rights_scope"]
        assert rights_scope["scope"] == "repository_and_art_collection_only_not_asset_level"
        assert rights_scope["asset_license_expression"] is None
        assert rights_scope["asset_creator"] is None
        assert not rights_scope["rights_observation_added"]

        standing_frames = connection.execute(
            "SELECT duration_ms, phase, direction, view, metadata_json "
            "FROM sequence_frames WHERE sequence_id=? ORDER BY ordinal",
            (standing["id"],),
        ).fetchall()
        assert [row["duration_ms"] for row in standing_frames] == [100.0, 120.0]
        assert [row["phase"] for row in standing_frames] == [0.0, 0.5]
        assert {row["direction"] for row in standing_frames} == {"up"}
        assert {row["view"] for row in standing_frames} == {"unknown"}
        frame_metadata = json.loads(standing_frames[0]["metadata_json"])
        assert frame_metadata["layer_literal"] == "42"
        assert frame_metadata["offset_literal"] == "0~0.1"
        assert frame_metadata["effective_auto_hflip"] is False
        assert frame_metadata["effective_auto_vflip"] is True
        assert frame_metadata["inline_modifiers"] is None
        assert frame_metadata["frame_rect"] == {
            "bottom": 5,
            "coordinate_space": "source_sheet",
            "height": 5,
            "left": 0,
            "right": 4,
            "top": 0,
            "width": 4,
        }

        movement = next(
            row
            for row in sequence_rows
            if json.loads(row["metadata_json"])["source_tag"] == "movement_anim"
        )
        movement_metadata = json.loads(movement["metadata_json"])
        assert (movement["width"], movement["height"]) == (5, 6)
        assert movement["loop_mode"] == "one_shot"
        assert movement["action"] == "unknown"
        assert movement["direction"] == "down"
        assert movement_metadata["adapter_normalized_action"] == "move"
        assert not movement_metadata["source_dimensions_consistent"]
        assert not movement_metadata["runtime_composite_complete"]
        assert not movement_metadata["auxiliary_tracks_composited"]
        assert movement_metadata["auxiliary_frame_declarations"][0]["render_role"] == ("projectile")
        movement_phases = connection.execute(
            "SELECT phase FROM sequence_frames WHERE sequence_id=? ORDER BY ordinal",
            (movement["id"],),
        ).fetchall()
        assert [row["phase"] for row in movement_phases] == [0.0, 1.0]

        motion = connection.execute(
            "SELECT * FROM motion_annotations WHERE sequence_id=?",
            (movement["id"],),
        ).fetchone()
        conditioning = json.loads(motion["conditioning_json"])
        assert motion["source_action"] == "movement_anim"
        assert motion["normalized_action"] == "unknown"
        assert motion["direction"] == "down"
        assert motion["loopable"] == 0
        assert motion["cycle_frames"] is None
        assert conditioning["adapter_normalized_action"] == "move"
        assert conditioning["duration_ms_per_occurrence"] == [80, 140]


def test_readiness_reports_hash_mismatch_without_writing(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    wrong_hash = "f" * 64
    database.register_blob(
        sha256=wrong_hash,
        size_bytes=1,
        storage_path=tmp_path / "wrong.png",
        mime_type="image/png",
    )
    member_path = plan.required_source_image_hashes[0][0]
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE archive_members SET extracted_blob_sha256=?
            WHERE archive_blob_sha256=? AND member_path=?
            """,
            (wrong_hash, plan.archive_sha256, member_path),
        )
    before = database.path.stat().st_mtime_ns
    readiness = check_wesnoth_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert not readiness.ready
    assert readiness.missing_source_image_blobs == ()
    assert len(readiness.source_image_hash_mismatches) == 1
    assert member_path in readiness.source_image_hash_mismatches[0]
    assert before == after
    with pytest.raises(ValueError, match="CAS hash mismatch"):
        project_wesnoth_audit(database, plan, TAXONOMY)


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact CAS archive is not present")
def test_exact_pinned_projection_plan_is_deterministic_and_conservative() -> None:
    plan = plan_known_wesnoth_projection(EXACT_ARCHIVE)

    assert plan.archive_sha256 == EXPECTED_WESNOTH_ARCHIVE_SHA256
    assert plan.repository_commit == WESNOTH_COMMIT
    assert plan.source_audit_record_sha256 == (
        "21c4d48184cb99495609de0c238a107d23e9c19d4000009df4f6893447b1e9e8"
    )
    assert plan.projection_manifest_sha256 == PINNED_PROJECTION_MANIFEST_SHA256
    assert plan.projected_entity_count == 248
    assert plan.projected_sequence_count == 604
    assert plan.projected_frame_occurrence_count == 2_526
    assert plan.projected_loop_count == 224
    assert plan.projected_one_shot_count == 380
    assert plan.projected_normalized_action_count == 598
    assert plan.projected_unknown_action_count == 6
    assert plan.projected_exact_single_direction_count == 6
    assert plan.projected_auxiliary_declaration_count == 370
    assert plan.projected_occurrence_link_count == 6_066
    assert plan.excluded_candidate_sequence_count == 1_073
    assert plan.excluded_candidate_frame_occurrence_count == 6_114
    assert plan.excluded_transformed_primary_frame_count == 41
    assert plan.excluded_macro_animation_count == 1_004
    assert plan.excluded_conditional_animation_count == 140
    assert len(plan.required_member_paths) == 1_526
    assert len(plan.required_source_image_hashes) == 1_275
    assert len({record.sequence_source_key for record in plan.records}) == 604
    assert len({record.entity.entity_external_key for record in plan.records}) == 248
    assert all(record.safe_primary_source_sequence for record in plan.records)
    assert all(record.primary_timeline_exact for record in plan.records)
    assert all(
        frame.lossless_source_pixels
        and frame.exact_timing
        and frame.inline_modifiers is None
        and frame.separate_image_mod is None
        for record in plan.records
        for frame in record.frames
    )
    assert all(
        not exclusion.animation.safe_primary_source_sequence for exclusion in plan.exclusions
    )
    assert plan.rights.per_asset_license is None
    assert plan.rights.per_asset_attribution is None


@pytest.mark.skipif(
    not EXACT_ARCHIVE.is_file() or not LIVE_INDEX.is_file(),
    reason="exact CAS archive or local index is not present",
)
def test_exact_live_index_is_ready_via_query_only_dry_run() -> None:
    plan = plan_known_wesnoth_projection(EXACT_ARCHIVE)
    readiness = check_wesnoth_projection_readiness(LIVE_INDEX, plan)

    assert readiness.ready
    assert readiness.archive_blob_present
    assert readiness.source_item_count == 1
    assert readiness.required_member_count == readiness.present_member_count == 1_526
    assert readiness.required_source_image_count == 1_275
    assert readiness.present_source_image_blob_count == 1_275
    assert readiness.missing_member_paths == ()
    assert readiness.missing_source_image_blobs == ()
    assert readiness.source_image_hash_mismatches == ()
    assert readiness.projection_manifest_sha256 == PINNED_PROJECTION_MANIFEST_SHA256
