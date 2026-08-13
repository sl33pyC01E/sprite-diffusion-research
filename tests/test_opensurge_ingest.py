from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.opensurge import (
    EXPECTED_OPEN_SURGE_ARCHIVE_SHA256,
    OPEN_SURGE_COMMIT,
    audit_open_surge_archive,
)
from spritelab.db import IndexDB
from spritelab.ingest.opensurge import (
    OpenSurgeProjectionPlan,
    check_open_surge_projection_readiness,
    plan_known_open_surge_projection,
    plan_open_surge_projection,
    project_open_surge_audit,
)
from spritelab.taxonomy import load_taxonomy

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "14/48/1448d51c3e8bc8122ae893aad96e73f9c9ef1ffb74457109fa8cfa0ef10d0206"
)
LIVE_INDEX = Path("data/index/spritelab.sqlite3")
TAXONOMY = load_taxonomy(Path("configs/taxonomy.toml"))


def _png_bytes(size: tuple[int, int]) -> bytes:
    image = Image.new("P", size)
    image.putpalette([0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254)
    output = BytesIO()
    image.save(output, format="PNG", transparency=0)
    return output.getvalue()


def _script(*, include_unsafe: bool = False) -> str:
    unsafe = (
        """
sprite "Non Integral Grid"
{
    source_file "images/test.png"
    source_rect 0 0 24 9
    frame_size 8 8
    animation 0
    {
        repeat FALSE
        fps 8
        data 0
    }
}

sprite "Out Of Image"
{
    source_file "images/test.png"
    source_rect 32 0 16 8
    frame_size 16 8
    animation 0
    {
        repeat FALSE
        fps 8
        data 0
    }
}
"""
        if include_unsafe
        else ""
    )
    return f"""
// File: test.spr
// Description: projection fixture
// Author: Ada Artist
// License: MIT

// art by Pixel Person
sprite "Test Creature"
{{
    source_file "images/test.png"
    source_rect 8 4 24 8
    frame_size 8 8
    hot_spot 4 8
    action_spot 6 4

    // charging
    animation 0
    {{
        repeat TRUE
        fps 12.5
        data 0 1 1 2
        repeat_from 1
        action_spot 7 3
    }}

    animation 7 // running
    {{
        repeat TRUE
        fps 20
        data 2 1 0
    }}

    // running to charging
    transition 7 to 0
    {{
        repeat FALSE
        fps 10
        data 1 1
    }}
}}
{unsafe}
"""


def _synthetic_archive(tmp_path: Path, *, include_unsafe: bool = False) -> Path:
    archive_path = tmp_path / "opensurge.zip"
    root = f"opensurge-{OPEN_SURGE_COMMIT}"
    copyright_csv = (
        "Type;File;License;Author;Website;Notes\n"
        "image;images/test.png;CC-BY-4.0;Ada Artist;example.test;fixture art\n"
    )
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/LICENSE", "GNU GENERAL PUBLIC LICENSE\nVersion 3\n")
        archive.writestr(f"{root}/README.md", "Open Surge test. License: GPLv3.\n")
        archive.writestr(f"{root}/licenses/MIT-license.txt", "MIT License\n")
        archive.writestr(f"{root}/src/misc/copyright_data.csv", copyright_csv)
        archive.writestr(
            f"{root}/src/core/sprite.c",
            "GPL version 3 or later; row-major sprite fixture.\n",
        )
        archive.writestr(
            f"{root}/src/core/animation.c",
            "GPL version 3 or later; repeat_from fixture.\n",
        )
        archive.writestr(
            f"{root}/src/core/color.c",
            "bool color_is_transparent(unsigned char r, unsigned char g, "
            "unsigned char b, unsigned char a) {\n"
            "    return (a == 0) || (r == 255 && g == 0 && b == 255);\n"
            "}\n",
        )
        archive.writestr(
            f"{root}/src/core/shader.c",
            '"const vec3 MASK_COLOR = vec3(1.0, 0.0, 1.0);\\n"\n'
            '"p *= float(p.rgb != MASK_COLOR);\\n"\n',
        )
        archive.writestr(
            f"{root}/sprites/enemies/test.spr",
            _script(include_unsafe=include_unsafe),
        )
        archive.writestr(f"{root}/images/test.png", _png_bytes((40, 16)))
    return archive_path


def _indexed_database(
    tmp_path: Path,
    archive_path: Path,
) -> tuple[IndexDB, OpenSurgeProjectionPlan]:
    audit = audit_open_surge_archive(archive_path)
    plan = plan_open_surge_projection(audit)
    sheet = audit.source_sheets[0]
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    database.register_source(
        source_id="open_surge",
        kind="git_archive",
        name="Open Surge projection fixture",
        root_url="https://example.invalid/opensurge",
        adapter_version="test",
    )
    item_id = database.upsert_item(
        source_id="open_surge",
        external_id=OPEN_SURGE_COMMIT,
        canonical_url=f"https://example.invalid/opensurge/{OPEN_SURGE_COMMIT}",
    )
    database.register_blob(
        sha256=audit.archive_sha256,
        size_bytes=archive_path.stat().st_size,
        storage_path=archive_path,
        mime_type="application/zip",
    )
    database.register_blob(
        sha256=sheet.sha256,
        size_bytes=sheet.size_bytes,
        storage_path=tmp_path / "test.png",
        mime_type="image/png",
    )
    database.link_item_blob(
        item_id=item_id,
        blob_sha256=audit.archive_sha256,
        role="source_archive",
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
                "extracted_blob_sha256": (
                    sheet.sha256 if member_path == sheet.member_path else None
                ),
            }
            for ordinal, member_path in enumerate(plan.required_member_paths)
        ],
    )
    return database, plan


def test_plan_excludes_non_integral_grids_and_out_of_image_occurrences(
    tmp_path: Path,
) -> None:
    audit = audit_open_surge_archive(_synthetic_archive(tmp_path, include_unsafe=True))
    plan = plan_open_surge_projection(audit)

    assert plan.projected_entity_count == 1
    assert plan.projected_sequence_count == 3
    assert plan.projected_frame_occurrence_count == 9
    assert plan.excluded_candidate_sequence_count == 2
    assert plan.excluded_candidate_frame_occurrence_count == 2
    assert plan.excluded_unsafe_occurrence_count == 1
    assert {reason for item in plan.exclusions for reason in item.reasons} == {
        "source_rect_is_not_an_integral_frame_grid",
        "frame_occurrence_is_outside_source_image",
    }
    assert {item.sprite_identity for item in plan.exclusions} == {
        "Non Integral Grid",
        "Out Of Image",
    }
    assert all(
        occurrence.within_source_image is True
        for record in plan.records
        for occurrence in record.frame_occurrences
    )


def test_projection_is_idempotent_and_preserves_exact_timing_loops_and_credit(
    tmp_path: Path,
) -> None:
    archive_path = _synthetic_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    readiness = check_open_surge_projection_readiness(database.path, plan)
    assert readiness.ready
    assert readiness.required_member_count == 5

    first = project_open_surge_audit(database, plan, TAXONOMY)
    second = project_open_surge_audit(database, plan, TAXONOMY)
    assert (first.created_sequences, first.reused_sequences) == (3, 0)
    assert (second.created_sequences, second.reused_sequences) == (0, 3)
    assert first.projected_entities == second.projected_entities == 1
    assert first.projected_frame_occurrences == second.projected_frame_occurrences == 9
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
            "sequences": 3,
            "sequence_source_keys": 3,
            "sequence_subjects": 3,
            "motion_annotations": 3,
            "sequence_frames": 9,
            "sequence_occurrences": 15,
            "rights_observations": 0,
        }

        rows = connection.execute(
            "SELECT id, loop_mode, action, direction, metadata_json FROM sequences"
        ).fetchall()
        charging_row = next(
            row for row in rows if json.loads(row["metadata_json"])["animation_id"] == 0
        )
        charging_metadata = json.loads(charging_row["metadata_json"])
        assert charging_row["loop_mode"] == "intro_then_loop"
        assert charging_row["action"] == "unknown"
        assert charging_row["direction"] == "unknown"
        assert charging_metadata["data_frame_index_order"] == [0, 1, 1, 2]
        assert charging_metadata["intro_data"] == [0]
        assert charging_metadata["loop_data"] == [1, 1, 2]
        assert charging_metadata["fps"] == 12.5
        assert charging_metadata["duration_ms_per_occurrence"] == 80.0
        assert charging_metadata["asset_credit"]["license_expression"] == "CC-BY-4.0"
        assert charging_metadata["asset_credit"]["author"] == "Ada Artist"
        (pixel_transform,) = charging_metadata["pixel_transforms"]
        assert pixel_transform["schema"] == "spritelab.pixel_transform.v1"
        assert pixel_transform["op"] == "exact_uint8_rgb_to_rgba_zero"
        assert pixel_transform["rgb"] == [255, 0, 255]
        assert len(pixel_transform["evidence"]) == 2

        frames = connection.execute(
            """
            SELECT source_frame_index, duration_ms, phase, direction, view, metadata_json
            FROM sequence_frames WHERE sequence_id=? ORDER BY ordinal
            """,
            (charging_row["id"],),
        ).fetchall()
        assert [row["source_frame_index"] for row in frames] == [0, 1, 1, 2]
        assert [row["duration_ms"] for row in frames] == [80.0, 80.0, 80.0, 80.0]
        assert [row["phase"] for row in frames] == [None, 0.0, 1 / 3, 2 / 3]
        assert {row["direction"] for row in frames} == {"unknown"}
        assert {row["view"] for row in frames} == {"unknown"}
        assert [json.loads(row["metadata_json"])["in_loop_tail"] for row in frames] == [
            False,
            True,
            True,
            True,
        ]
        assert all(
            json.loads(row["metadata_json"])["pixel_transforms"] == [pixel_transform]
            for row in frames
        )

        occurrence_roles = {
            row[0]
            for row in connection.execute(
                "SELECT occurrence_role FROM sequence_occurrences WHERE sequence_id=?",
                (charging_row["id"],),
            )
        }
        assert occurrence_roles == {
            "opensurge_asset_credit_manifest",
            "opensurge_engine_color_key_predicate",
            "opensurge_engine_color_key_shader",
            "opensurge_source_sprite_sheet",
            "opensurge_sprite_definition",
        }

        motion = connection.execute(
            "SELECT * FROM motion_annotations WHERE sequence_id=?",
            (charging_row["id"],),
        ).fetchone()
        assert motion["source_action"] == "charging"
        assert motion["normalized_action"] == "unknown"
        assert motion["loopable"] == 1
        assert motion["cycle_frames"] == 3
        assert motion["phase_zero_frame"] == 1

        running = next(row for row in rows if json.loads(row["metadata_json"])["animation_id"] == 7)
        assert running["action"] == "run"
        assert running["loop_mode"] == "loop"
        transition = next(
            row
            for row in rows
            if json.loads(row["metadata_json"])["declaration_kind"] == "transition"
        )
        assert transition["action"] == "unknown"
        assert transition["loop_mode"] == "one_shot"

        entity_metadata = json.loads(
            connection.execute("SELECT metadata_json FROM entities").fetchone()[0]
        )
        assert entity_metadata["asset_credit"]["evidence_line_number"] == 2
        assert entity_metadata["source_header_authors"] == ["Ada Artist"]


def test_readiness_hash_mismatch_is_reported_without_writing(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    database.register_blob(
        sha256="f" * 64,
        size_bytes=1,
        storage_path=tmp_path / "wrong.png",
        mime_type="image/png",
    )
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE archive_members SET extracted_blob_sha256=?
            WHERE archive_blob_sha256=? AND normalized_path LIKE '%/images/test.png'
            """,
            ("f" * 64, plan.archive_sha256),
        )
    before = database.path.stat().st_mtime_ns
    readiness = check_open_surge_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert not readiness.ready
    assert len(readiness.sheet_hash_mismatches) == 1
    assert before == after


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact CAS archive is not present")
def test_exact_archive_projection_plan_is_deterministic_and_conservative() -> None:
    plan = plan_known_open_surge_projection(EXACT_ARCHIVE)

    assert plan.archive_sha256 == EXPECTED_OPEN_SURGE_ARCHIVE_SHA256
    assert plan.repository_commit == OPEN_SURGE_COMMIT
    assert plan.projected_entity_count == 353
    assert plan.projected_sequence_count == 878
    assert plan.projected_regular_animation_count == 854
    assert plan.projected_transition_count == 24
    assert plan.projected_frame_occurrence_count == 3484
    assert plan.projected_normalized_action_count == 199
    assert plan.projected_unknown_action_count == 679
    assert plan.projected_explicit_direction_count == 26
    assert plan.projected_loop_count == 631
    assert plan.projected_intro_then_loop_count == 9
    assert plan.projected_one_shot_count == 238
    assert plan.projected_oversized_source_rect_count == 45
    assert plan.excluded_candidate_sequence_count == 39
    assert plan.excluded_candidate_frame_occurrence_count == 56
    assert plan.excluded_unsafe_occurrence_count == 3
    assert len(plan.required_member_paths) == 214
    assert plan.projection_manifest_sha256 == (
        "7ec1fabd908bb195cae4ae2a50374c08336f3a9d861a24fd209f2773e8d53f43"
    )
    assert plan.pixel_transform.transform_sha256 == (
        "d0860d86c815d0a4b6f7c116f7a2f31faedaaac4b7d9877f028e5545364fa306"
    )
    assert len({record.sequence_source_key for record in plan.records}) == 878
    assert {reason for item in plan.exclusions for reason in item.reasons} == {
        "source_rect_is_not_an_integral_frame_grid",
        "frame_occurrence_is_outside_source_image",
    }
    assert {item.sprite_identity for item in plan.exclusions} == {
        "Animal",
        "Power Pluggy Clockwise",
        "Power Pluggy Counterclockwise",
        "SD_VERTICALDANGER",
    }
    assert all(record.asset_credit.file_path == record.source_file for record in plan.records)
    assert all(
        occurrence.within_declared_source_rect and occurrence.within_source_image is True
        for record in plan.records
        for occurrence in record.frame_occurrences
    )


@pytest.mark.skipif(
    not EXACT_ARCHIVE.is_file() or not LIVE_INDEX.is_file(),
    reason="exact CAS archive or local index is not present",
)
def test_exact_live_index_is_ready_via_read_only_dry_run() -> None:
    plan = plan_known_open_surge_projection(EXACT_ARCHIVE)
    before = LIVE_INDEX.stat().st_mtime_ns
    readiness = check_open_surge_projection_readiness(LIVE_INDEX, plan)
    after = LIVE_INDEX.stat().st_mtime_ns

    assert readiness.ready
    assert readiness.archive_blob_present
    assert readiness.source_item_count == 1
    assert readiness.required_member_count == readiness.present_member_count == 214
    assert readiness.missing_member_paths == ()
    assert readiness.missing_sheet_blobs == ()
    assert readiness.sheet_hash_mismatches == ()
    assert readiness.projection_manifest_sha256 == plan.projection_manifest_sha256
    assert before == after


def test_projection_manifest_hash_changes_with_source_fact(tmp_path: Path) -> None:
    audit = audit_open_surge_archive(_synthetic_archive(tmp_path))
    first = plan_open_surge_projection(audit)
    second = plan_open_surge_projection(audit)
    assert len(first.projection_manifest_sha256) == 64
    assert first.projection_manifest_sha256 == second.projection_manifest_sha256

    changed = plan_open_surge_projection(
        audit_open_surge_archive(_synthetic_archive(tmp_path, include_unsafe=True))
    )
    assert first.projection_manifest_sha256 != changed.projection_manifest_sha256
