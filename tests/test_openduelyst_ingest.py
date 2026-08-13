from __future__ import annotations

import hashlib
import json
import plistlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.openduelyst import (
    EXPECTED_OPENDUELYST_ARCHIVE_SHA256,
    OPENDUELYST_COMMIT,
    audit_openduelyst_archive,
)
from spritelab.db import IndexDB
from spritelab.ingest.openduelyst import (
    OpenDuelystProjectionPlan,
    check_openduelyst_projection_readiness,
    plan_known_openduelyst_projection,
    plan_openduelyst_projection,
    project_openduelyst_audit,
)
from spritelab.taxonomy import load_taxonomy

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "9d/90/9d907a2d299b0f1598984192e3d4832aeb770e75fa2507370ff8e66428282f8e"
)
LIVE_INDEX = Path("data/index/spritelab.sqlite3")
TAXONOMY = load_taxonomy(Path("configs/taxonomy.toml"))


def _png_bytes(size: tuple[int, int] = (16, 16)) -> bytes:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _frame(
    rect: str,
    *,
    offset: str = "{0,0}",
    rotated: bool = False,
    source_rect: str = "{{0,0},{4,4}}",
) -> dict[str, object]:
    return {
        "frame": rect,
        "offset": offset,
        "rotated": rotated,
        "sourceColorRect": source_rect,
        "sourceSize": "{4,4}",
    }


def _plist_bytes() -> bytes:
    return plistlib.dumps(
        {
            "frames": {
                "wolf_run_10.png": _frame("{{0,0},{4,4}}"),
                "wolf_run_2.png": _frame(
                    "{{4,0},{3,4}}",
                    offset="{0.5,-0.5}",
                    rotated=True,
                    source_rect="{{1,0},{3,4}}",
                ),
                "broken_0.png": _frame(
                    "{{7,0},{3,4}}",
                    source_rect="{{0,0},{4,4}}",
                ),
            },
            "metadata": {
                "format": 2,
                "size": "{16,16}",
                "textureFileName": "wolf.png",
                "realTextureFileName": "wolf.png",
            },
        },
        sort_keys=False,
    )


def _resources_source() -> str:
    return """
const RSX = {
  wolfRun: {
    name: 'wolfRun', img: 'resources/units/wolf.png',
    plist: 'resources/units/wolf.plist', framePrefix: 'wolf_run_', frameDelay: .10,
  },
  wolfWalk: {
    name: 'wolfWalk', img: 'resources/units/wolf.png',
    plist: 'resources/units/wolf.plist', framePrefix: 'wolf_run_', frameDelay: .10,
  },
  wolfEmpty: {
    name: 'wolfEmpty', img: 'resources/units/wolf.png',
    plist: 'resources/units/wolf.plist', framePrefix: 'missing_', frameDelay: .10,
  },
  wolfBroken: {
    name: 'wolfBroken', img: 'resources/units/wolf.png',
    plist: 'resources/units/wolf.plist', framePrefix: 'broken_', frameDelay: .10,
  },
};
"""


def _factory_source() -> str:
    return """
class CardFactory_Test
  @cardForIdentifier: (identifier,gameSession) ->
    if (identifier == Cards.Neutral.Wolf)
      card = new Unit(gameSession)
      card.factionId = Factions.Neutral
      card.raceId = Races.Vespyr
      card.name = i18next.t("cards.wolf_name")
      card.setBaseAnimResource(
        walk : RSX.wolfRun.name
        idle : RSX.wolfWalk.name
        attackDelay: 0.4
      )
    if (identifier == Cards.Neutral.SpiritWolf)
      card = new Unit(gameSession)
      card.factionId = Factions.Neutral
      card.name = i18next.t("cards.spirit_wolf_name")
      card.setBaseAnimResource(
        death : RSX.wolfRun.name
        idle : RSX.wolfWalk.name
      )
"""


def _synthetic_archive(tmp_path: Path) -> Path:
    root = f"duelyst-{OPENDUELYST_COMMIT}"
    archive_path = tmp_path / "duelyst.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/LICENSE", "CC0 1.0 Universal\n")
        archive.writestr(f"{root}/app/data/resources.js", _resources_source())
        archive.writestr(
            f"{root}/app/sdk/cards/cardsLookup.coffee",
            ("class Cards\n  @Neutral:{\n    Wolf: 900\n    SpiritWolf: 901\n  }\n"),
        )
        archive.writestr(
            f"{root}/app/sdk/cards/factory/test/neutral.coffee",
            _factory_source(),
        )
        archive.writestr(
            f"{root}/app/localization/locales/en/cards.json",
            json.dumps(
                {
                    "wolf_name": "Snow Wolf",
                    "spirit_wolf_name": "Spirit Wolf",
                }
            ),
        )
        archive.writestr(f"{root}/app/resources/units/wolf.plist", _plist_bytes())
        archive.writestr(f"{root}/app/resources/units/wolf.png", _png_bytes())
    return archive_path


def _indexed_database(
    tmp_path: Path,
    archive_path: Path,
) -> tuple[IndexDB, OpenDuelystProjectionPlan]:
    audit = audit_openduelyst_archive(archive_path)
    plan = plan_openduelyst_projection(audit)
    image_hash = plan.records[0].atlas_image_sha256
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    database.register_source(
        source_id="openduelyst",
        kind="git_archive",
        name="OpenDuelyst projection fixture",
        root_url="https://example.invalid/openduelyst",
        adapter_version="test",
    )
    item_id = database.upsert_item(
        source_id="openduelyst",
        external_id="open-duelyst/duelyst",
        canonical_url="https://example.invalid/openduelyst/duelyst",
    )
    database.register_blob(
        sha256=audit.archive_sha256,
        size_bytes=archive_path.stat().st_size,
        storage_path=archive_path,
        mime_type="application/zip",
    )
    database.register_blob(
        sha256=image_hash,
        size_bytes=len(_png_bytes()),
        storage_path=tmp_path / "wolf.png",
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
    image_member = plan.records[0].atlas_image_member_path
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
                "extracted_blob_sha256": (image_hash if member_path == image_member else None),
            }
            for ordinal, member_path in enumerate(plan.required_member_paths)
        ],
    )
    return database, plan


def test_plan_quarantines_empty_and_unsafe_geometry_without_repairs(
    tmp_path: Path,
) -> None:
    audit = audit_openduelyst_archive(_synthetic_archive(tmp_path))
    plan = plan_openduelyst_projection(audit)

    assert plan.projected_sequence_count == 2
    assert plan.projected_frame_occurrence_count == 4
    assert plan.projected_physical_entity_count == 1
    assert plan.projected_mapped_entity_count == 2
    assert plan.projected_entity_count == 3
    assert plan.projected_subject_link_count == 6
    assert plan.projected_exact_action_count == 1
    assert plan.projected_ambiguous_or_unmapped_action_count == 1
    assert plan.projected_loop_count == 1
    assert plan.projected_role_dependent_loop_count == 1
    assert plan.excluded_candidate_sequence_count == 2
    assert plan.excluded_candidate_frame_occurrence_count == 1

    exclusions = {exclusion.resource_alias: exclusion for exclusion in plan.exclusions}
    assert exclusions["wolfEmpty"].reasons == ("frame_prefix_matches_no_plist_key",)
    assert exclusions["wolfEmpty"].unsafe_frame_keys == ()
    assert exclusions["wolfBroken"].reasons == ("packed_rect_size_differs_from_source_color_rect",)
    assert exclusions["wolfBroken"].unsafe_frame_keys == ("broken_0.png",)

    records = {record.resource_alias: record for record in plan.records}
    assert records["wolfRun"].source_roles == ("walk", "death")
    assert records["wolfRun"].normalized_action is None
    assert records["wolfRun"].loop_mode == "role_dependent"
    assert records["wolfWalk"].source_roles == ("idle",)
    assert records["wolfWalk"].loop_mode == "loop"
    assert records["wolfRun"].exact_timeline_aliases == (
        "wolfRun",
        "wolfWalk",
    )
    assert records["wolfRun"].frames[0].key == "wolf_run_2.png"
    assert records["wolfRun"].frames[0].rotated is True
    assert records["wolfRun"].frames[0].is_trimmed is True


def test_projection_is_idempotent_and_preserves_geometry_roles_and_scoped_rights(
    tmp_path: Path,
) -> None:
    archive_path = _synthetic_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    readiness = check_openduelyst_projection_readiness(database.path, plan)
    assert readiness.ready
    assert readiness.required_source_image_count == 1
    assert readiness.present_source_image_blob_count == 1

    first = project_openduelyst_audit(database, plan, TAXONOMY)
    second = project_openduelyst_audit(database, plan, TAXONOMY)
    assert (first.created_sequences, first.reused_sequences) == (2, 0)
    assert (second.created_sequences, second.reused_sequences) == (0, 2)
    assert first.projected_entities == second.projected_entities == 3
    assert first.projected_frame_occurrences == 4
    assert first.projected_subject_links == 6
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
                "rights_observations",
            )
        }
        assert counts == {
            "entities": 3,
            "sequences": 2,
            "sequence_source_keys": 2,
            "sequence_subjects": 6,
            "motion_annotations": 2,
            "sequence_frames": 4,
            "rights_observations": 0,
        }

        rows = connection.execute(
            "SELECT id, width, height, loop_mode, action, direction, metadata_json FROM sequences"
        ).fetchall()
        run = next(
            row for row in rows if json.loads(row["metadata_json"])["resource_alias"] == "wolfRun"
        )
        run_metadata = json.loads(run["metadata_json"])
        assert (run["width"], run["height"]) == (4, 4)
        assert run["loop_mode"] == "role_dependent"
        assert run["action"] == "unknown"
        assert run["direction"] == "unknown"
        assert run_metadata["source_roles"] == ["walk", "death"]
        assert run_metadata["runtime_frame_key_order"] == [
            "wolf_run_2.png",
            "wolf_run_10.png",
        ]
        assert run_metadata["declared_frame_delay_expression"] == ".10"
        assert run_metadata["runtime_delay_multiplier"] == 0.8
        assert run_metadata["duration_ms_per_occurrence"] == pytest.approx(80.0)
        rights = run_metadata["rights_scope"]
        assert rights["scope"] == "repository_project_claim_only_not_asset_level"
        assert rights["repository_claim_identifiers"] == ["CC0-1.0"]
        assert rights["asset_license_expression"] is None
        assert rights["asset_creator"] is None
        assert rights["per_asset_manifest_present"] is False

        frames = connection.execute(
            "SELECT duration_ms, phase, metadata_json FROM sequence_frames "
            "WHERE sequence_id=? ORDER BY ordinal",
            (run["id"],),
        ).fetchall()
        assert [frame["duration_ms"] for frame in frames] == [
            pytest.approx(80.0),
            pytest.approx(80.0),
        ]
        assert [frame["phase"] for frame in frames] == [None, None]
        first_frame = json.loads(frames[0]["metadata_json"])
        assert first_frame["frame_key"] == "wolf_run_2.png"
        assert first_frame["packed_frame_rect"]["raw"] == "{{4,0},{3,4}}"
        assert first_frame["rotated"] is True
        assert first_frame["source_color_rect"]["raw"] == "{{1,0},{3,4}}"
        assert first_frame["source_size"]["raw"] == "{4,4}"
        assert first_frame["offset"]["raw"] == "{0.5,-0.5}"
        assert first_frame["physical_frame_aliases"] == [
            "wolfRun",
            "wolfWalk",
        ]

        subjects = connection.execute(
            "SELECT role, metadata_json FROM sequence_subjects "
            "WHERE sequence_id=? ORDER BY role, entity_id",
            (run["id"],),
        ).fetchall()
        assert [subject["role"] for subject in subjects] == [
            "primary",
            "source_entity_mapping",
            "source_entity_mapping",
        ]
        mapped_roles = sorted(
            json.loads(subject["metadata_json"])["roles_for_sequence"]
            for subject in subjects
            if subject["role"] == "source_entity_mapping"
        )
        assert mapped_roles == [["death"], ["walk"]]

        motion = connection.execute(
            "SELECT * FROM motion_annotations WHERE sequence_id=?",
            (run["id"],),
        ).fetchone()
        assert motion["source_action"] is None
        assert motion["normalized_action"] == "unknown"
        assert motion["loopable"] is None
        assert motion["cycle_frames"] is None


def test_readiness_reports_hash_mismatch_without_writing(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    database.register_blob(
        sha256="f" * 64,
        size_bytes=1,
        storage_path=tmp_path / "wrong.png",
        mime_type="image/png",
    )
    image_member = plan.records[0].atlas_image_member_path
    with database.connect() as connection:
        connection.execute(
            """
            UPDATE archive_members SET extracted_blob_sha256=?
            WHERE archive_blob_sha256=? AND normalized_path=?
            """,
            ("f" * 64, plan.archive_sha256, image_member),
        )
    before = database.path.stat().st_mtime_ns
    readiness = check_openduelyst_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert not readiness.ready
    assert len(readiness.source_image_hash_mismatches) == 1
    assert readiness.missing_source_image_blobs == ()
    assert before == after


def test_manifest_hash_changes_with_source_facts(tmp_path: Path) -> None:
    archive = _synthetic_archive(tmp_path)
    audit = audit_openduelyst_archive(archive)
    first = plan_openduelyst_projection(audit)
    second = plan_openduelyst_projection(audit)
    assert len(first.projection_manifest_sha256) == 64
    assert first.projection_manifest_sha256 == second.projection_manifest_sha256

    changed_path = tmp_path / "changed" / "duelyst.zip"
    changed_path.parent.mkdir()
    with (
        ZipFile(archive) as source,
        ZipFile(changed_path, "w", compression=ZIP_DEFLATED) as changed,
    ):
        for info in source.infolist():
            payload = source.read(info)
            if info.filename.endswith("app/data/resources.js"):
                payload = payload.replace(b"frameDelay: .10", b"frameDelay: .20", 1)
            changed.writestr(info, payload)
    changed = plan_openduelyst_projection(audit_openduelyst_archive(changed_path))
    assert first.projection_manifest_sha256 != changed.projection_manifest_sha256


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact CAS archive is not present")
def test_exact_archive_projection_plan_is_deterministic_and_conservative() -> None:
    plan = plan_known_openduelyst_projection(EXACT_ARCHIVE)

    assert plan.archive_sha256 == EXPECTED_OPENDUELYST_ARCHIVE_SHA256
    assert plan.repository_commit == OPENDUELYST_COMMIT
    assert plan.projected_sequence_count == 5_302
    assert plan.projected_frame_occurrence_count == 69_020
    assert plan.projected_physical_entity_count == 1_276
    assert plan.projected_mapped_entity_count == 1_076
    assert plan.projected_entity_count == 2_352
    assert plan.projected_subject_link_count == 10_247
    assert plan.projected_exact_action_count == 4_672
    assert plan.projected_ambiguous_or_unmapped_action_count == 630
    assert plan.projected_loop_count == 2_692
    assert plan.projected_one_shot_count == 2_006
    assert plan.projected_role_dependent_loop_count == 3
    assert plan.projected_unknown_loop_count == 601
    assert plan.excluded_candidate_sequence_count == 10
    assert plan.excluded_candidate_frame_occurrence_count == 71
    assert len(plan.required_member_paths) == 2_638
    assert plan.projection_manifest_sha256 == (
        "ff7411d9a1dcd4aa76bb40dc8dbc087f563983352355684d8b37ab67847b2719"
    )

    exclusions = {exclusion.resource_alias: exclusion for exclusion in plan.exclusions}
    assert {alias for alias, item in exclusions.items() if item.frame_count == 0} == {
        "iconSuperMaliceActive",
        "iconThoughtExchangeActive",
        "f2TwilightFoxHit",
        "f1ThirdGeneralCast",
        "f3DuplicatorObelyskRun",
        "f4AbominationDeath",
        "f5OrphanAspectDeath",
        "f5OrphanAspectHit",
    }
    assert exclusions["f3GeneralFestiveIdle"].reasons == (
        "descriptor_image_path_differs_from_plist_texture_path",
    )
    assert set(exclusions["fx_fluid_sphere"].reasons) == {
        "packed_rect_size_differs_from_source_color_rect",
        "source_color_rect_is_outside_source_canvas",
    }
    assert all(record.frame_count > 0 for record in plan.records)
    assert all(
        frame.within_image_bounds is True for record in plan.records for frame in record.frames
    )
    assert all(
        len({(frame.source_size.width, frame.source_size.height) for frame in record.frames}) == 1
        for record in plan.records
    )


@pytest.mark.skipif(
    not EXACT_ARCHIVE.is_file() or not LIVE_INDEX.is_file(),
    reason="exact CAS archive or local index is not present",
)
def test_exact_live_index_is_ready_via_read_only_dry_run() -> None:
    plan = plan_known_openduelyst_projection(EXACT_ARCHIVE)
    before_hash = hashlib.sha256(LIVE_INDEX.read_bytes()).hexdigest()
    before_mtime = LIVE_INDEX.stat().st_mtime_ns
    readiness = check_openduelyst_projection_readiness(LIVE_INDEX, plan)
    after_mtime = LIVE_INDEX.stat().st_mtime_ns
    after_hash = hashlib.sha256(LIVE_INDEX.read_bytes()).hexdigest()

    assert readiness.ready
    assert readiness.archive_blob_present
    assert readiness.source_item_count == 1
    assert readiness.required_member_count == readiness.present_member_count == 2_638
    assert readiness.required_source_image_count == 1_276
    assert readiness.present_source_image_blob_count == 1_276
    assert readiness.missing_member_paths == ()
    assert readiness.missing_source_image_blobs == ()
    assert readiness.source_image_hash_mismatches == ()
    assert readiness.projection_manifest_sha256 == plan.projection_manifest_sha256
    assert (before_mtime, before_hash) == (after_mtime, after_hash)
