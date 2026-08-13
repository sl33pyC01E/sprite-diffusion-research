from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from PIL import Image

from spritelab.adapters.flare import (
    EXPECTED_FLARE_ARCHIVE_SHA256,
    FLARE_ENGINE_COMMIT,
    FLARE_GAME_COMMIT,
    audit_flare_archive,
)
from spritelab.db import IndexDB
from spritelab.ingest.flare import (
    FlareProjectionPlan,
    check_flare_projection_readiness,
    plan_flare_projection,
    plan_known_flare_projection,
    project_flare_audit,
)
from spritelab.taxonomy import load_taxonomy

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "9c/8e/9c8e58bb704f55174928990a1a375c213ffb2847d7b928d6117a84db3e1215cc"
)
LIVE_INDEX = Path("data/index/spritelab.sqlite3")
TAXONOMY = load_taxonomy(Path("configs/taxonomy.toml"))


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGBA", size, color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _compressed_animation(image: str) -> str:
    return f"""image={image}

[stance]
frames=2
duration=200ms
type=looped
active_frame=all
frame=0,SW,0,0,8,8,3,7
frame=1,SW,8,0,8,8,4,7
"""


def _geometry_free_parent() -> str:
    return """[stance]
frames=2
duration=200ms
type=looped
"""


def _symlink_info(name: str) -> ZipInfo:
    info = ZipInfo(name)
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    return info


def _synthetic_archive(tmp_path: Path) -> Path:
    path = tmp_path / "flare.zip"
    root = f"flare-game-{FLARE_GAME_COMMIT}"
    layer_lines = "\n".join(
        f"layer={token},main,body" for token in ("SW", "W", "NW", "N", "NE", "E", "SE", "S")
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/", b"")
        archive.writestr(f"{root}/README", "Art/data CC-BY-SA 3.0 or later.\n")
        archive.writestr(_symlink_info(f"{root}/README.md"), "README")
        archive.writestr(f"{root}/LICENSE.txt", "Attribution-ShareAlike 3.0 Unported\n")
        archive.writestr(f"{root}/CREDITS.txt", "Art\nFixture Artist\n")
        archive.writestr(f"{root}/mods/fantasycore/settings.txt", "description=fixture base\n")
        archive.writestr(f"{root}/mods/empyrean_campaign/settings.txt", "requires=fantasycore\n")
        archive.writestr(
            f"{root}/mods/fantasycore/animations/enemies/wolf.txt",
            _compressed_animation("images/wolf.png"),
        )
        archive.writestr(
            f"{root}/mods/fantasycore/animations/hero.txt",
            _geometry_free_parent(),
        )
        for variant in ("female", "female_dark", "male"):
            archive.writestr(
                f"{root}/mods/fantasycore/animations/avatar/{variant}/sword.txt",
                _compressed_animation("images/sword.png"),
            )
        archive.writestr(
            f"{root}/mods/fantasycore/images/wolf.png",
            _png_bytes((16, 8), (255, 0, 0, 127)),
        )
        archive.writestr(
            f"{root}/mods/fantasycore/images/sword.png",
            _png_bytes((16, 8), (0, 255, 0, 127)),
        )
        archive.writestr(f"{root}/mods/fantasycore/engine/hero_layers.txt", layer_lines)
        archive.writestr(
            f"{root}/mods/empyrean_campaign/enemies/wolf.txt",
            "name=Fixture Wolf\ncategories=wolf,animal\nhumanoid=true\n"
            "animations=animations/enemies/wolf.txt\n",
        )
        archive.writestr(
            f"{root}/mods/empyrean_campaign/items/items.txt",
            "[item]\nid=7\nname=Fixture Sword\nitem_type=main\ngfx=sword\n"
            "loot_animation=animations/enemies/wolf.txt\n",
        )
        archive.writestr(
            f"{root}/mods/empyrean_campaign/powers/powers.txt",
            "[power]\nid=9\nname=Howl\nanimation=animations/enemies/wolf.txt\n",
        )
    return path


def _indexed_database(
    tmp_path: Path,
    archive_path: Path,
) -> tuple[IndexDB, FlareProjectionPlan]:
    audit = audit_flare_archive(archive_path)
    plan = plan_flare_projection(audit)
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    database.register_source(
        source_id="flare_empyrean",
        kind="git_archive",
        name="Flare projection fixture",
        root_url="https://example.invalid/flare",
        adapter_version="test",
    )
    item_id = database.upsert_item(
        source_id="flare_empyrean",
        external_id=FLARE_GAME_COMMIT,
        canonical_url=f"https://example.invalid/flare/{FLARE_GAME_COMMIT}",
    )
    database.register_blob(
        sha256=audit.archive_sha256,
        size_bytes=archive_path.stat().st_size,
        storage_path=archive_path,
        mime_type="application/zip",
    )
    for image in plan.required_source_images:
        database.register_blob(
            sha256=image.sha256,
            size_bytes=image.size_bytes,
            storage_path=tmp_path / f"{image.sha256}.png",
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
        member_count=plan.archive_member_count,
        file_count=plan.archive_regular_file_count,
        total_uncompressed_bytes=plan.archive_expanded_member_bytes,
        total_compressed_bytes=archive_path.stat().st_size,
        inventory_sha256="0" * 64,
    )
    image_hashes = {image.member_path: image.sha256 for image in plan.required_source_images}
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
                "extracted_blob_sha256": image_hashes.get(member_path),
            }
            for ordinal, member_path in enumerate(plan.required_member_paths)
        ],
    )
    return database, plan


def test_plan_admits_only_complete_tracks_and_preserves_candidate_ambiguity(
    tmp_path: Path,
) -> None:
    archive_path = _synthetic_archive(tmp_path)
    audit = audit_flare_archive(archive_path)
    first = plan_flare_projection(audit)
    second = plan_flare_projection(audit)

    assert first.projection_manifest_sha256 == second.projection_manifest_sha256
    assert first.projected_definition_count == 4
    assert first.projected_action_count == 4
    assert first.projected_sequence_count == 32
    assert first.projected_frame_count == 64
    assert first.projected_explicit_frame_count == 8
    assert first.projected_fallback_frame_count == 56
    assert first.excluded_direction_track_count == 8
    assert first.excluded_unresolved_slot_count == 16
    assert {reason for item in first.exclusions for reason in item.reasons} == {
        "action_missing_exact_geometry",
        "direction_track_has_unresolved_slots",
    }
    assert {item.definition_logical_path for item in first.exclusions} == {"animations/hero.txt"}
    assert first.projected_entity_binding_count == 1
    assert first.projected_usage_count == 3
    assert first.attachment_candidate_edge_count == 3
    assert len(first.attachment_quarantines) == 1
    assert first.attachment_quarantines[0].reasons == ("body_variant_selection_is_ambiguous",)

    southwest = next(
        item
        for item in first.records
        if item.definition_logical_path == "animations/enemies/wolf.txt" and item.direction == 0
    )
    northwest = next(
        item
        for item in first.records
        if item.definition_logical_path == "animations/enemies/wolf.txt" and item.direction == 2
    )
    assert (southwest.envelope_origin.x, southwest.envelope_origin.y) == (3, 7)
    assert (southwest.envelope_width, southwest.envelope_height) == (9, 8)
    assert [frame.default_60hz_tick_count for frame in southwest.frames] == [6, 6]
    assert [frame.explicit for frame in southwest.frames] == [True, True]
    assert [frame.fallback_from_direction for frame in northwest.frames] == [0, 0]
    assert all(frame.authored_direction == 0 for frame in northwest.frames)

    changed = replace(first, geometry_missing_action_count=2)
    assert changed.projection_manifest_sha256 != first.projection_manifest_sha256


def test_projection_is_idempotent_and_retains_geometry_timing_relations_and_rights(
    tmp_path: Path,
) -> None:
    archive_path = _synthetic_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    readiness = check_flare_projection_readiness(database.path, plan)
    assert readiness.ready
    assert readiness.required_source_image_count == 2

    first = project_flare_audit(database, plan, TAXONOMY)
    second = project_flare_audit(database, plan, TAXONOMY)
    assert (first.created_sequences, first.reused_sequences) == (32, 0)
    assert (second.created_sequences, second.reused_sequences) == (0, 32)
    assert first.projected_entities == second.projected_entities == 8
    assert first.projected_frames == second.projected_frames == 64
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
            "entities": 8,
            "sequences": 32,
            "sequence_source_keys": 32,
            "motion_annotations": 32,
            "sequence_frames": 64,
            "rights_observations": 0,
        }
        rows = connection.execute(
            "SELECT id, width, height, loop_mode, action, direction, metadata_json FROM sequences"
        ).fetchall()
        wolf_northwest = next(
            row
            for row in rows
            if (metadata := json.loads(row["metadata_json"]))["definition_logical_path"]
            == "animations/enemies/wolf.txt"
            and metadata["direction"]["index"] == 2
        )
        metadata = json.loads(wolf_northwest["metadata_json"])
        assert (wolf_northwest["width"], wolf_northwest["height"]) == (9, 8)
        assert wolf_northwest["loop_mode"] == "loop"
        assert wolf_northwest["action"] == "idle"
        assert wolf_northwest["direction"] == "up_left"
        assert metadata["duration_literal"] == "200ms"
        assert metadata["duration_milliseconds"] == 200
        assert metadata["default_60hz_tick_schedule"]["frame_indices"] == [
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
        ]
        assert metadata["anchor_relative_union_envelope"]["derived_summary_only"]
        assert metadata["entity_relations"][0]["humanoid"] is True
        assert metadata["rights_scope"]["asset_creator"] is None
        assert not metadata["rights_scope"]["per_asset_manifest_present"]

        frames = connection.execute(
            """
            SELECT duration_ms, direction, view, metadata_json
            FROM sequence_frames WHERE sequence_id=? ORDER BY ordinal
            """,
            (wolf_northwest["id"],),
        ).fetchall()
        assert [row["duration_ms"] for row in frames] == [100.0, 100.0]
        assert {row["direction"] for row in frames} == {"up_left"}
        assert {row["view"] for row in frames} == {"unknown"}
        frame_metadata = json.loads(frames[0]["metadata_json"])
        assert frame_metadata["rectangle"] == {
            "height": 8,
            "width": 8,
            "x": 0,
            "y": 0,
        }
        assert frame_metadata["offset"] == {"x": 3, "y": 7}
        assert frame_metadata["explicit"] is False
        assert frame_metadata["fallback_from_direction"] == 0
        assert frame_metadata["authored_direction"]["index"] == 0

        sword = next(
            row
            for row in rows
            if (metadata := json.loads(row["metadata_json"]))["definition_logical_path"]
            == "animations/avatar/female/sword.txt"
            and metadata["direction"]["index"] == 0
        )
        sword_metadata = json.loads(sword["metadata_json"])
        candidate = sword_metadata["attachment_candidates"][0]
        assert candidate["candidate_body_variant"] == "female"
        assert candidate["body_variant_choice_resolved"] is False
        assert candidate["layer_slot"] == "main"
        assert candidate["layer_index_back_to_front"] == 0


def test_readiness_is_query_only_and_reports_image_hash_mismatch(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    before_hash = hashlib.sha256(database.path.read_bytes()).hexdigest()
    before_mtime = database.path.stat().st_mtime_ns
    readiness = check_flare_projection_readiness(database.path, plan)
    after_mtime = database.path.stat().st_mtime_ns
    after_hash = hashlib.sha256(database.path.read_bytes()).hexdigest()
    assert readiness.ready
    assert (before_mtime, before_hash) == (after_mtime, after_hash)

    wrong_hash = "f" * 64
    database.register_blob(
        sha256=wrong_hash,
        size_bytes=1,
        storage_path=tmp_path / "wrong.png",
        mime_type="image/png",
    )
    image_path = plan.required_source_images[0].member_path
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE archive_members SET extracted_blob_sha256=?
            WHERE archive_blob_sha256=? AND member_path=?
            """,
            (wrong_hash, plan.archive_sha256, image_path),
        )
    mismatch = check_flare_projection_readiness(database.path, plan)
    assert not mismatch.ready
    assert len(mismatch.source_image_hash_mismatches) == 1
    assert mismatch.missing_source_image_blobs == ()


@pytest.fixture(scope="module")
def exact_plan() -> FlareProjectionPlan:
    if not EXACT_ARCHIVE.is_file():
        pytest.skip("exact Flare CAS archive is not present")
    return plan_known_flare_projection(EXACT_ARCHIVE)


def test_exact_projection_plan_is_deterministic_and_conservative(
    exact_plan: FlareProjectionPlan,
) -> None:
    plan = exact_plan
    assert plan.archive_sha256 == EXPECTED_FLARE_ARCHIVE_SHA256
    assert plan.repository_commit == FLARE_GAME_COMMIT
    assert plan.engine_semantics_commit == FLARE_ENGINE_COMMIT
    assert plan.active_mods == ("fantasycore", "empyrean_campaign")
    assert plan.optional_mods_excluded == (
        "minicore",
        "alpha_demo",
        "minicore_alpha",
        "devlab",
    )
    assert plan.projected_definition_count == 328
    assert plan.projected_action_count == 1_980
    assert plan.projected_sequence_count == 15_840
    assert plan.projected_frame_count == 71_064
    assert plan.projected_explicit_frame_count == 70_897
    assert plan.projected_fallback_frame_count == 167
    assert plan.projected_source_image_count == 296
    assert plan.projected_entity_binding_count == 176
    assert plan.projected_usage_count == 975
    assert plan.attachment_candidate_edge_count == 1_047
    assert len(plan.attachment_quarantines) == 349
    assert plan.unresolved_attachment_layer_count == 1
    assert plan.projected_subject_entity_count == 1_650
    assert plan.excluded_direction_track_count == 128
    assert plan.excluded_unresolved_slot_count == 544
    assert len(plan.usage_exclusions) == 1
    assert plan.usage_exclusions[0].usage.animation_path == ("animations/enemies/wyvern_adult.txt")
    assert {item.definition_logical_path for item in plan.exclusions} == {
        "animations/hero.txt",
        "animations/avatar/default_unpacked.txt",
    }
    assert len(plan.required_member_paths) == 934
    assert plan.projection_manifest_sha256 == (
        "e0b232b3158a21546d8f7512b18d98306d2a7a66fc993b0bf986ad7618acbc9b"
    )
    assert all(
        frame.rectangle.x >= 0
        and frame.rectangle.y >= 0
        and frame.rectangle.right <= frame.image_width
        and frame.rectangle.bottom <= frame.image_height
        for record in plan.records
        for frame in record.frames
    )
    relic = next(item for item in plan.attachment_quarantines if item.item_id == "1102")
    assert relic.item_name == "Knife of Sacrifices"
    assert relic.layer_slot == "relic"
    assert len(relic.missing_layer_directions) == 8


def test_exact_live_index_is_ready_via_query_only_check(
    exact_plan: FlareProjectionPlan,
) -> None:
    if not LIVE_INDEX.is_file():
        pytest.skip("local provenance index is not present")
    readiness = check_flare_projection_readiness(LIVE_INDEX, exact_plan)
    assert readiness.ready
    assert readiness.archive_blob_present
    assert readiness.archive_inventory_present
    assert readiness.archive_inventory_matches_audit
    assert readiness.source_item_count == 1
    assert readiness.required_member_count == readiness.present_member_count == 934
    assert readiness.required_source_image_count == 296
    assert readiness.present_source_image_blob_count == 296
    assert readiness.missing_member_paths == ()
    assert readiness.missing_source_image_blobs == ()
    assert readiness.source_image_hash_mismatches == ()
    assert readiness.projection_manifest_sha256 == exact_plan.projection_manifest_sha256
