from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.ss14 import (
    EXPECTED_SS14_ARCHIVE_SHA256,
    SS14_ARCHIVE_URL,
    SS14_COMMIT,
    audit_ss14_archive,
    known_ss14_cas_path,
)
from spritelab.archive import ArchiveLimits, extract_zip_to_cas, inspect_zip
from spritelab.db import IndexDB
from spritelab.indexing import (
    index_zip_extraction,
    index_zip_manifest,
    inspect_media_observation,
)
from spritelab.ingest.ss14 import (
    EXPECTED_PINNED_ARCHIVE_INVENTORY_SHA256,
    EXPECTED_PINNED_ARCHIVE_MEMBER_COUNT,
    EXPECTED_PINNED_PREPARATION_MANIFEST_SHA256,
    SOURCE_ID,
    SS14_ITEM_EXTERNAL_ID,
    SS14_MEDIA_INSPECTOR_VERSION,
    SS14_SELECTED_IMAGE_ROLE,
    Ss14PreparationPlan,
    Ss14ProjectionPlan,
    check_ss14_preparation_readiness,
    check_ss14_projection_readiness,
    plan_known_ss14_preparation,
    plan_known_ss14_projection,
    plan_ss14_preparation,
    plan_ss14_projection,
    project_ss14_audit,
)
from spritelab.sources import load_source_registry, sync_source_registry
from spritelab.storage import ContentAddressedStore, DiskGuard
from spritelab.taxonomy import load_taxonomy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXACT_ARCHIVE = known_ss14_cas_path(PROJECT_ROOT / "data" / "raw")
LIVE_INDEX = PROJECT_ROOT / "data" / "index" / "spritelab.sqlite3"
TAXONOMY = load_taxonomy(PROJECT_ROOT / "configs" / "taxonomy.toml")
PINNED_PROJECTION_MANIFEST_SHA256 = (
    "1e8bb0e67924b57ecf67ab3523a7f3c37987fbf13ad179feab1236f3370aafe4"
)


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    image = Image.new("RGBA", size, color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _write_rsi(
    archive: ZipFile,
    *,
    root: str,
    path: str,
    metadata: dict[str, object],
    images: dict[str, tuple[tuple[int, int], tuple[int, int, int, int]]],
) -> None:
    archive.writestr(f"{root}/{path}/meta.json", json.dumps(metadata))
    for name, (size, color) in images.items():
        archive.writestr(f"{root}/{path}/{name}.png", _png_bytes(size, color))


def _fixture_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / "ss14-projection-fixture.zip"
    root = "space-station-14-projection-fixture"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/LICENSE.TXT", "MIT License\n")
        archive.writestr(f"{root}/.github/rsi-schema.json", '{"type":"object"}\n')
        _write_rsi(
            archive,
            root=root,
            path="Resources/Textures/Mobs/Animals/fox.rsi",
            metadata={
                "version": 1,
                "license": "CC-BY-SA-3.0",
                "copyright": (
                    "Fixture Artist; from https://github.com/tgstation/tgstation/blob/"
                    "53d1f1477d22a11a99c6c6924977cd431075761b/icons/mob/animal.dmi"
                ),
                "size": {"x": 4, "y": 4},
                "states": [
                    {
                        "name": "fox-running",
                        "directions": 4,
                        "delays": [[0.2, 0.3], [0.1, 0.4], [0.25, 0.25], [0.5]],
                    },
                    {"name": "fox-dead", "directions": 4},
                    {"name": "fox-moving", "delays": [[0.2, 0.2]]},
                    {"name": "fox-idle", "delays": [[0.2, 0.2]]},
                    {"name": "eyes-running", "delays": [[0.2, 0.2]]},
                ],
            },
            images={
                "fox-running": ((28, 4), (10, 20, 30, 255)),
                "fox-dead": ((16, 4), (20, 30, 40, 255)),
                "fox-moving": ((8, 4), (30, 40, 50, 255)),
                "fox-idle": ((12, 4), (40, 50, 60, 255)),
                "eyes-running": ((8, 4), (50, 60, 70, 255)),
            },
        )
        _write_rsi(
            archive,
            root=root,
            path="Resources/Textures/Mobs/Pets/wolf.rsi",
            metadata={
                "version": 1,
                "license": "CC-BY-NC-SA-3.0",
                "copyright": "NC Fixture Artist",
                "size": {"x": 4, "y": 4},
                "states": [{"name": "wolf-running", "delays": [[0.2, 0.2]]}],
            },
            images={"wolf-running": ((8, 4), (60, 70, 80, 255))},
        )
        _write_rsi(
            archive,
            root=root,
            path="Resources/Textures/Mobs/Species/Human/parts.rsi",
            metadata={
                "version": 1,
                "license": "CC0-1.0",
                "copyright": "Fixture Artist",
                "size": {"x": 4, "y": 4},
                "states": [{"name": "human-dead", "directions": 4}],
            },
            images={"human-dead": ((16, 4), (70, 80, 90, 255))},
        )
    return archive_path


def _indexed_database(
    tmp_path: Path,
    archive_path: Path,
) -> tuple[IndexDB, Ss14ProjectionPlan]:
    audit = audit_ss14_archive(archive_path)
    plan = plan_ss14_projection(audit, TAXONOMY)
    database = IndexDB(tmp_path / "ss14-index.sqlite3")
    database.initialize()
    database.register_source(
        source_id=SOURCE_ID,
        kind="git_archive",
        name="SS14 projection fixture",
        root_url="https://example.invalid/ss14",
        adapter_version="test",
    )
    item_id = database.upsert_item(
        source_id=SOURCE_ID,
        external_id=SS14_COMMIT,
        canonical_url=f"https://example.invalid/ss14/{SS14_COMMIT}",
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


def _prepare_with_generic_apis(
    database: IndexDB,
    store: ContentAddressedStore,
    plan: Ss14PreparationPlan,
    *,
    inspect_media: bool,
) -> None:
    limits = ArchiveLimits()
    registry = load_source_registry(PROJECT_ROOT / "configs" / "sources.toml")
    sync_source_registry(database, registry)
    source = registry.by_id(SOURCE_ID)
    database.register_blob(
        sha256=plan.archive_sha256,
        size_bytes=plan.archive_size_bytes,
        storage_path=Path(plan.archive_path),
        mime_type="application/zip",
    )
    item_id = database.upsert_item(
        source_id=SOURCE_ID,
        external_id=SS14_ITEM_EXTERNAL_ID,
        canonical_url=source.root_url,
        title="space-station-14",
        creator_name="space-wizards",
        creator_url="https://github.com/space-wizards",
        metadata={
            "full_name": SS14_ITEM_EXTERNAL_ID,
            "resolved_ref": plan.repository_commit,
            "commit_sha": plan.repository_commit,
            "commit_url": plan.commit_url,
            "archive_sha256": plan.archive_sha256,
            "archive_url": plan.archive_url,
            "acquisition_state": "adopted_existing_cas_archive",
        },
    )
    with database.connect() as connection:
        existing_link = connection.execute(
            """
            SELECT 1 FROM item_blobs
            WHERE item_id=? AND blob_sha256=? AND role='source_archive'
            LIMIT 1
            """,
            (item_id, plan.archive_sha256),
        ).fetchone()
    if existing_link is None:
        database.link_item_blob(
            item_id=item_id,
            blob_sha256=plan.archive_sha256,
            role="source_archive",
            original_url=SS14_ARCHIVE_URL,
            original_filename=f"space-station-14-{plan.repository_commit}.zip",
        )

    manifest = inspect_zip(plan.archive_path, limits=limits)
    assert manifest.inventory_sha256 == plan.archive_inventory_sha256
    index_zip_manifest(
        database,
        archive_blob_sha256=plan.archive_sha256,
        manifest=manifest,
        limits=limits,
    )
    extraction = extract_zip_to_cas(
        plan.archive_path,
        store,
        limits=limits,
        select=plan.selected_image_member_paths,
    )
    assert {member.member.normalized_name for member in extraction.extracted} == set(
        plan.selected_image_member_paths
    )
    assert {
        member.member.normalized_name: member.blob.sha256 for member in extraction.extracted
    } == {member.member_path: member.expected_sha256 for member in plan.selected_image_members}
    index_zip_extraction(
        database,
        archive_blob_sha256=plan.archive_sha256,
        extraction=extraction,
        selected_role=SS14_SELECTED_IMAGE_ROLE,
    )
    if not inspect_media:
        return

    observations_by_hash = {}
    for extracted in extraction.extracted:
        observations_by_hash.setdefault(
            extracted.blob.sha256,
            inspect_media_observation(
                blob_sha256=extracted.blob.sha256,
                path=extracted.blob.path,
                original_name=extracted.member.normalized_name,
                inspector_version=SS14_MEDIA_INSPECTOR_VERSION,
            ),
        )
    database.record_media_observations(list(observations_by_hash.values()))
    database.mark_archive_member_inspections(
        archive_blob_sha256=plan.archive_sha256,
        inspections=[
            {"ordinal": extracted.member.archive_index, "status": "media_inspected"}
            for extracted in extraction.extracted
        ],
    )


def test_plan_projects_exact_direction_timelines_and_partitions_every_state(
    tmp_path: Path,
) -> None:
    audit = audit_ss14_archive(_fixture_archive(tmp_path))
    plan = plan_ss14_projection(audit, TAXONOMY)

    assert plan.projected_state_count == 2
    assert plan.projected_sequence_count == 8
    assert plan.projected_entity_count == 1
    assert plan.projected_frame_occurrence_count == 20
    assert plan.projected_animated_sequence_count == 4
    assert plan.projected_static_sequence_count == 4
    assert plan.excluded_state_count == 5
    assert plan.projected_state_count + plan.excluded_state_count == audit.counts.states

    south = next(
        record
        for record in plan.records
        if record.state_name == "fox-running" and record.source_direction == "south"
    )
    assert south.engine_delays_seconds == pytest.approx((0.1, 0.1, 0.05, 0.25))
    assert south.engine_source_cell_indices == (0, 0, 1, 1)
    assert [frame.source_cell_index for frame in south.frames] == [0, 0, 1, 1]
    assert [frame.source_delay_seconds for frame in south.frames] == pytest.approx(
        [0.2, 0.2, 0.3, 0.3]
    )
    assert [frame.duration_milliseconds for frame in south.frames] == pytest.approx(
        [100.0, 100.0, 50.0, 250.0]
    )
    assert [(frame.left, frame.top, frame.right, frame.bottom) for frame in south.frames] == [
        (0, 0, 4, 4),
        (0, 0, 4, 4),
        (4, 0, 8, 4),
        (4, 0, 8, 4),
    ]
    assert south.rights.license_expression == "CC-BY-SA-3.0"
    assert south.rights.upstream_references[0].revision == (
        "53d1f1477d22a11a99c6c6924977cd431075761b"
    )
    assert south.image_payload_deduplication_key == f"sha256:{south.source_sheet_sha256}"
    assert south.loop_semantics == "not_encoded_in_rsi_caller_controls_playback"

    exclusions = {exclusion.state_name: exclusion for exclusion in plan.exclusions}
    assert exclusions["fox-moving"].reasons == ("action:noncanonical:move",)
    assert exclusions["fox-idle"].reasons == ("image:surplus_capacity",)
    assert exclusions["eyes-running"].reasons == ("state_role:modular_component",)
    assert exclusions["wolf-running"].reasons == ("quarantine:noncommercial_asset_license",)
    assert exclusions["human-dead"].reasons == ("state_role:modular_component",)

    assert (
        plan.projection_manifest_sha256
        == plan_ss14_projection(audit, TAXONOMY).projection_manifest_sha256
    )
    changed = plan_ss14_projection(replace(audit, commit="different-source-fact"), TAXONOMY)
    assert changed.projection_manifest_sha256 != plan.projection_manifest_sha256


def test_generic_apis_prepare_exact_allowlist_and_readiness_is_query_only(
    tmp_path: Path,
) -> None:
    fixture_archive = _fixture_archive(tmp_path)
    guard = DiskGuard(tmp_path, min_free_bytes=0)
    store = ContentAddressedStore(tmp_path / "data", guard)
    archive_blob = store.put_file(fixture_archive)
    projection = plan_ss14_projection(audit_ss14_archive(archive_blob.path), TAXONOMY)
    plan = plan_ss14_preparation(archive_blob.path, projection)
    database = IndexDB(tmp_path / "prepared-index.sqlite3")
    database.initialize()

    initial = check_ss14_preparation_readiness(database.path, plan)
    assert not initial.projection_prerequisites_ready
    assert not initial.ready
    assert not initial.source_registered
    assert not initial.archive_blob_registered
    assert initial.archive_retrieval_count == 0

    _prepare_with_generic_apis(database, store, plan, inspect_media=False)
    extracted = check_ss14_preparation_readiness(database.path, plan)
    assert extracted.projection_prerequisites_ready
    assert extracted.extraction_complete
    assert not extracted.media_inspection_complete
    assert not extracted.ready
    assert extracted.source_item_count == 1
    assert extracted.pinned_source_item_count == 1
    assert extracted.source_archive_link_count == 1
    assert extracted.archive_retrieval_count == 0
    assert extracted.indexed_archive_member_count == plan.archive_member_count
    assert extracted.present_required_member_count == plan.required_member_count
    assert extracted.present_selected_image_blob_count == plan.selected_image_member_count
    assert extracted.present_unique_selected_image_blob_count == plan.unique_selected_image_count
    assert extracted.present_unique_selected_image_file_count == plan.unique_selected_image_count
    assert check_ss14_projection_readiness(database.path, projection).ready

    before = database.path.stat().st_mtime_ns
    observed = check_ss14_preparation_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns
    assert observed.extraction_complete
    assert before == after

    _prepare_with_generic_apis(database, store, plan, inspect_media=True)
    completed = check_ss14_preparation_readiness(database.path, plan)
    assert completed.ready
    assert completed.all_media_valid
    assert completed.media_observation_count == plan.unique_selected_image_count
    assert completed.media_inspected_member_count == plan.selected_image_member_count
    assert completed.media_invalid_member_count == 0

    _prepare_with_generic_apis(database, store, plan, inspect_media=True)
    rerun = check_ss14_preparation_readiness(database.path, plan)
    assert rerun.ready
    assert rerun.source_archive_link_count == 1

    invalid = next(
        member
        for member in plan.selected_image_members
        if sum(
            candidate.expected_sha256 == member.expected_sha256
            for candidate in plan.selected_image_members
        )
        == 1
    )
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM media_observations WHERE blob_sha256=? AND inspector_version=?",
            (invalid.expected_sha256, SS14_MEDIA_INSPECTOR_VERSION),
        )
        connection.execute(
            """
            UPDATE archive_members SET inspection_status='media_invalid', error=?
            WHERE archive_blob_sha256=? AND ordinal=?
            """,
            ("InvalidPNGError: fixture rejection", plan.archive_sha256, invalid.ordinal),
        )
    terminal_invalid = check_ss14_preparation_readiness(database.path, plan)
    assert terminal_invalid.ready
    assert terminal_invalid.media_inspection_complete
    assert not terminal_invalid.all_media_valid
    assert terminal_invalid.media_invalid_member_count == 1
    assert terminal_invalid.media_terminal_unique_image_count == plan.unique_selected_image_count
    assert invalid.member_path in terminal_invalid.media_invalid_members[0]


def test_projection_is_idempotent_and_preserves_rights_geometry_timing_and_unknown_loop(
    tmp_path: Path,
) -> None:
    archive_path = _fixture_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    before = database.path.stat().st_mtime_ns
    readiness = check_ss14_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns
    assert readiness.ready
    assert readiness.source_registered
    assert readiness.required_source_image_count == 2
    assert readiness.present_source_image_blob_count == 2
    assert before == after

    first = project_ss14_audit(database, plan, TAXONOMY)
    second = project_ss14_audit(database, plan, TAXONOMY)
    assert (first.created_sequences, first.reused_sequences) == (8, 0)
    assert (second.created_sequences, second.reused_sequences) == (0, 8)
    assert first.projected_entities == second.projected_entities == 1
    assert first.projected_frame_occurrences == second.projected_frame_occurrences == 20
    assert first.occurrence_links == second.occurrence_links == 32
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
            "sequence_frames": 20,
            "sequence_occurrences": 32,
            "rights_observations": 0,
        }
        rows = connection.execute(
            "SELECT id, loop_mode, action, direction, metadata_json FROM sequences"
        ).fetchall()
        south = next(
            row
            for row in rows
            if json.loads(row["metadata_json"])["state_name"] == "fox-running"
            and json.loads(row["metadata_json"])["source_direction"] == "south"
        )
        metadata = json.loads(south["metadata_json"])
        assert south["loop_mode"] == "unknown"
        assert south["action"] == "run"
        assert south["direction"] == "down"
        assert metadata["loop_policy_inferred"] is False
        assert metadata["engine_source_cell_indices"] == [0, 0, 1, 1]
        assert metadata["rights_scope"]["license_expression"] == "CC-BY-SA-3.0"
        assert metadata["rights_scope"]["upstream_asset_deduplication_keys"] == [
            "github:tgstation/tgstation@"
            "53d1f1477d22a11a99c6c6924977cd431075761b:icons/mob/animal.dmi"
        ]

        frames = connection.execute(
            "SELECT source_frame_index, duration_ms, phase, direction, metadata_json "
            "FROM sequence_frames WHERE sequence_id=? ORDER BY ordinal",
            (south["id"],),
        ).fetchall()
        assert [row["source_frame_index"] for row in frames] == [0, 0, 1, 1]
        assert [row["duration_ms"] for row in frames] == pytest.approx([100.0, 100.0, 50.0, 250.0])
        assert [row["phase"] for row in frames] == [None, None, None, None]
        assert {row["direction"] for row in frames} == {"down"}
        frame_metadata = json.loads(frames[0]["metadata_json"])
        assert frame_metadata["frame_rect"] == {
            "bottom": 4,
            "column": 0,
            "coordinate_space": "source_sheet",
            "height": 4,
            "left": 0,
            "right": 4,
            "row": 0,
            "top": 0,
            "width": 4,
        }
        assert frame_metadata["source_delay_seconds"] == pytest.approx(0.2)
        assert frame_metadata["engine_delay_seconds"] == pytest.approx(0.1)
        assert frame_metadata["source_action_cue"] == "run"

        occurrences = connection.execute(
            "SELECT occurrence_role, metadata_json FROM sequence_occurrences "
            "WHERE sequence_id=? ORDER BY occurrence_role",
            (south["id"],),
        ).fetchall()
        assert {row["occurrence_role"] for row in occurrences} == {
            "ss14_repository_license_scope_evidence",
            "ss14_rsi_metadata_and_per_pack_rights",
            "ss14_rsi_schema_evidence",
            "ss14_rsi_state_image",
        }
        image_occurrence = next(
            row for row in occurrences if row["occurrence_role"] == "ss14_rsi_state_image"
        )
        image_occurrence_metadata = json.loads(image_occurrence["metadata_json"])
        assert image_occurrence_metadata["source_sheet_sha256"] == metadata["source_sheet_sha256"]
        assert image_occurrence_metadata["rights_scope"]["copyright"].startswith("Fixture Artist")

        motion = connection.execute(
            "SELECT * FROM motion_annotations WHERE sequence_id=?",
            (south["id"],),
        ).fetchone()
        conditioning = json.loads(motion["conditioning_json"])
        assert motion["source_action"] == "fox-running"
        assert motion["normalized_action"] == "run"
        assert motion["direction"] == "down"
        assert motion["loopable"] is None
        assert motion["cycle_frames"] is None
        assert conditioning["source_action_cue"] == "run"
        assert conditioning["loop_policy_inferred"] is False


def test_readiness_reports_hash_mismatch_without_writing(
    tmp_path: Path,
) -> None:
    archive_path = _fixture_archive(tmp_path)
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
    readiness = check_ss14_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert not readiness.ready
    assert readiness.source_registered
    assert readiness.missing_source_image_blobs == ()
    assert len(readiness.source_image_hash_mismatches) == 1
    assert member_path in readiness.source_image_hash_mismatches[0]
    assert before == after
    with pytest.raises(ValueError, match="CAS hash mismatch"):
        project_ss14_audit(database, plan, TAXONOMY)


def test_readiness_reports_missing_source_registration_without_writing(tmp_path: Path) -> None:
    plan = plan_ss14_projection(audit_ss14_archive(_fixture_archive(tmp_path)), TAXONOMY)
    database = IndexDB(tmp_path / "empty-index.sqlite3")
    database.initialize()
    before = database.path.stat().st_mtime_ns
    readiness = check_ss14_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert not readiness.source_registered
    assert not readiness.archive_inventory_present
    assert readiness.source_item_count == 0
    assert not readiness.ready
    assert before == after


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact SS14 CAS archive is not present")
def test_exact_pinned_projection_plan_is_deterministic_and_conservative() -> None:
    plan = plan_known_ss14_projection(EXACT_ARCHIVE, TAXONOMY)

    assert plan.archive_sha256 == EXPECTED_SS14_ARCHIVE_SHA256
    assert plan.repository_commit == SS14_COMMIT
    assert plan.source_audit_record_sha256 == (
        "0804bd63eed162bd09c42716f7a1ef46f712e2aa702f6a75c38552ec18fac973"
    )
    assert plan.projection_manifest_sha256 == PINNED_PROJECTION_MANIFEST_SHA256
    assert plan.projected_state_count == 189
    assert plan.projected_sequence_count == 246
    assert plan.projected_entity_count == 139
    assert plan.projected_frame_occurrence_count == 297
    assert plan.projected_animated_sequence_count == 13
    assert plan.projected_static_sequence_count == 233
    assert plan.excluded_state_count == 1_791
    assert plan.projected_occurrence_link_count == 2_706
    assert len(plan.required_member_paths) == 287
    assert len(plan.required_source_image_hashes) == 189
    assert len(set(dict(plan.required_source_image_hashes).values())) == 187
    assert len(plan.required_evidence_hashes) == 287
    assert len(set(dict(plan.required_evidence_hashes).values())) == 282
    assert plan.duplicate_image_payload_group_count == 1
    assert plan.duplicate_image_payload_excess == 2
    assert len({record.sequence_source_key for record in plan.records}) == 246
    assert len({record.entity.entity_external_key for record in plan.records}) == 139
    assert {record.source_action_cue for record in plan.records} == {
        "attack",
        "death",
        "emote",
        "hurt",
        "idle",
        "run",
        "sleep",
        "spawn",
        "walk",
    }
    reason_counts: dict[str, int] = {}
    for exclusion in plan.exclusions:
        for reason in exclusion.reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    assert reason_counts["action:unmapped"] == 1_726
    assert reason_counts["action:noncanonical:move"] == 30
    assert reason_counts["action:noncanonical:rest"] == 6
    assert reason_counts["action:noncanonical:sit"] == 6
    assert reason_counts["action:noncanonical:stun"] == 2
    assert reason_counts["state_role:modular_component"] == 1_261
    assert reason_counts["state_role:effect_or_overlay"] == 147
    assert reason_counts["state_role:icon_or_item_view"] == 65
    assert reason_counts["quarantine:noncommercial_asset_license"] == 13
    assert reason_counts["image:surplus_capacity"] == 47


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact SS14 CAS archive is not present")
def test_exact_pinned_preparation_plan_limits_extraction_to_projected_images(
    tmp_path: Path,
) -> None:
    plan = plan_known_ss14_preparation(EXACT_ARCHIVE, TAXONOMY)

    assert plan.preparation_manifest_sha256 == EXPECTED_PINNED_PREPARATION_MANIFEST_SHA256
    assert plan.archive_inventory_sha256 == EXPECTED_PINNED_ARCHIVE_INVENTORY_SHA256
    assert plan.archive_member_count == EXPECTED_PINNED_ARCHIVE_MEMBER_COUNT == 49_472
    assert plan.archive_file_count == 43_004
    assert plan.archive_directory_count == 6_468
    assert plan.archive_symlink_count == 0
    assert plan.archive_total_compressed_bytes == 218_134_591
    assert plan.archive_total_uncompressed_bytes == 340_248_827
    assert plan.all_png_member_count == 25_832
    assert plan.all_png_compressed_bytes == 40_779_680
    assert plan.all_png_uncompressed_bytes == 61_014_321
    assert plan.required_member_count == 287
    assert plan.required_member_compressed_bytes == 414_157
    assert plan.required_member_uncompressed_bytes == 813_186
    assert plan.selected_image_member_count == 189
    assert plan.selected_image_compressed_bytes == 356_198
    assert plan.selected_image_uncompressed_bytes == 536_519
    assert plan.unique_selected_image_count == 187
    assert plan.unique_selected_image_bytes == 530_803
    assert set(plan.selected_image_member_paths) == {
        member.member_path for member in plan.required_members if member.extraction_required
    }

    store = ContentAddressedStore(tmp_path / "exact-media", DiskGuard(tmp_path, 0))
    extraction = extract_zip_to_cas(
        EXACT_ARCHIVE,
        store,
        limits=ArchiveLimits(),
        select=plan.selected_image_member_paths,
    )
    by_hash = {}
    for extracted in extraction.extracted:
        by_hash.setdefault(extracted.blob.sha256, extracted)
    valid_hashes = set()
    invalid = {}
    for digest, extracted in by_hash.items():
        try:
            inspect_media_observation(
                blob_sha256=digest,
                path=extracted.blob.path,
                original_name=extracted.member.normalized_name,
                inspector_version=SS14_MEDIA_INSPECTOR_VERSION,
            )
            valid_hashes.add(digest)
        except ValueError as error:
            invalid[digest] = (extracted.member.normalized_name, str(error))
    assert len(valid_hashes) == 176
    assert len(invalid) == 11
    assert all(path.endswith("dead.png") for path, _ in invalid.values())
    assert {message for _, message in invalid.values()} == {"data follows the PNG IEND chunk"}
    expected_trailing_zero_bytes = {
        "barrier_dead.png": 2,
        "barrier_naked_dead.png": 1,
        "bear_dead.png": 2,
        "fossilegg_dead.png": 1,
        "glider_dead.png": 2,
        "harvester_dead.png": 1,
        "leviathing_dead.png": 2,
        "molder_dead.png": 1,
        "narsian_dead.png": 2,
        "pouncer_dead.png": 1,
        "skitter_dead.png": 1,
    }
    observed_trailing_zero_bytes = {}
    for digest, (member_path, _) in invalid.items():
        payload = by_hash[digest].blob.path.read_bytes()
        iend_type_offset = payload.rfind(b"IEND")
        assert iend_type_offset >= 4
        trailing = payload[iend_type_offset + 8 :]
        assert trailing == b"\0" * len(trailing)
        observed_trailing_zero_bytes[Path(member_path).name] = len(trailing)
    assert observed_trailing_zero_bytes == expected_trailing_zero_bytes


@pytest.mark.skipif(
    not (EXACT_ARCHIVE.is_file() and LIVE_INDEX.is_file()),
    reason="exact CAS archive or live index is not present",
)
def test_live_readiness_query_reports_state_without_projection_write() -> None:
    plan = plan_known_ss14_projection(EXACT_ARCHIVE, TAXONOMY)
    readiness = check_ss14_projection_readiness(LIVE_INDEX, plan)

    assert readiness.archive_sha256 == EXPECTED_SS14_ARCHIVE_SHA256
    assert readiness.projection_manifest_sha256 == PINNED_PROJECTION_MANIFEST_SHA256
    assert readiness.required_member_count == 287
    assert readiness.required_source_image_count == 189
    if not readiness.source_registered:
        assert not readiness.ready
