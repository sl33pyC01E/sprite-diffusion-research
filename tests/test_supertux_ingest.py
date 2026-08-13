from __future__ import annotations

import hashlib
import io
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.supertux import (
    EXPECTED_SUPERTUX_ARCHIVE_SHA256,
    SUPERTUX_COMMIT,
    audit_supertux_archive,
    known_supertux_cas_path,
)
from spritelab.db import IndexDB
from spritelab.ingest.supertux import (
    EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256,
    SOURCE_ID,
    SuperTuxProjectionPlan,
    check_supertux_projection_readiness,
    plan_known_supertux_projection,
    plan_supertux_projection,
    plan_supertux_projection_preparation,
    project_supertux_audit,
)
from spritelab.snapshot import SnapshotFilters, load_sequence_samples
from spritelab.taxonomy import load_taxonomy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXACT_ARCHIVE = known_supertux_cas_path(PROJECT_ROOT / "data" / "raw")
LIVE_INDEX = PROJECT_ROOT / "data" / "index" / "spritelab.sqlite3"
TAXONOMY = load_taxonomy(PROJECT_ROOT / "configs" / "taxonomy.toml")

_EVIDENCE = (
    "LICENSE.txt",
    "README.md",
    "data/AUTHORS",
    "data/credits.stxt",
    "src/sprite/sprite_data.cpp",
    "src/sprite/sprite.cpp",
)


def _png_bytes(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _fixture_archive(tmp_path: Path) -> Path:
    path = tmp_path / "supertux-projection-fixture.zip"
    root = "supertux-projection-fixture"
    hero_manifest = r"""
(supertux-sprite
  (action
    (name "walk-left")
    (fps 20)
    (loops 1)
    (loop-frame 2)
    (hitbox 1 2 8 6)
    (images "a.png" "b.png" "a.png"))
  (action
    (name "walk-right")
    (fps 20)
    (hitbox 2 2 8 6)
    (mirror-action "walk-left"))
  (action
    (name "roof-left")
    (fps 20)
    (hitbox 1 2 8 6)
    (flip-action "walk-left"))
  (action
    (name "slow-left")
    (fps 5)
    (hitbox 9 9 1 1)
    (clone-action "walk-left"))
  (action
    (name "idle")
    (images "a.png")))
"""
    effect_manifest = r"""
(supertux-sprite
  (action (name "default") (hitbox 0 0 4 4) (images "glow.png")))
"""
    module_manifest = r"""
(supertux-sprite
  (action (name "default") (hitbox 0 0 4 4) (images "hat.png")))
"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for logical_path in _EVIDENCE:
            archive.writestr(f"{root}/{logical_path}", f"evidence:{logical_path}\n")
        archive.writestr(
            f"{root}/data/images/creatures/tux/tux.sprite",
            hero_manifest,
        )
        archive.writestr(
            f"{root}/data/images/creatures/crystallo/crystallo-overlay.sprite",
            effect_manifest,
        )
        archive.writestr(
            f"{root}/data/images/creatures/tux/santahat.sprite",
            module_manifest,
        )
        for logical_path, payload in {
            "data/images/creatures/tux/a.png": _png_bytes((12, 8), (20, 40, 60, 255)),
            "data/images/creatures/tux/b.png": _png_bytes((10, 7), (70, 80, 90, 180)),
            "data/images/creatures/tux/unreferenced.png": _png_bytes((3, 5), (1, 2, 3, 255)),
            "data/images/creatures/crystallo/glow.png": _png_bytes((4, 4), (0, 200, 255, 90)),
            "data/images/creatures/tux/hat.png": _png_bytes((4, 4), (200, 10, 10, 255)),
        }.items():
            archive.writestr(f"{root}/{logical_path}", payload)
    return path


def _indexed_database(
    tmp_path: Path,
    archive_path: Path,
) -> tuple[IndexDB, SuperTuxProjectionPlan]:
    audit = audit_supertux_archive(archive_path)
    plan = plan_supertux_projection(audit, TAXONOMY)
    database = IndexDB(tmp_path / "supertux-index.sqlite3")
    database.initialize()
    database.register_source(
        source_id=SOURCE_ID,
        kind="git_archive",
        name="SuperTux projection fixture",
        root_url="https://example.invalid/supertux",
        adapter_version="test",
    )
    item_id = database.upsert_item(
        source_id=SOURCE_ID,
        external_id=audit.commit,
        canonical_url=f"https://example.invalid/supertux/{audit.commit}",
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
    evidence_hashes = dict(plan.required_evidence_hashes)
    for evidence_sha256 in sorted(set(evidence_hashes.values())):
        database.register_blob(
            sha256=evidence_sha256,
            size_bytes=1,
            storage_path=tmp_path / evidence_sha256,
            mime_type="application/octet-stream",
        )
    database.upsert_archive_inventory(
        archive_blob_sha256=audit.archive_sha256,
        archive_format="zip",
        member_count=len(plan.required_member_paths),
        file_count=len(plan.required_member_paths),
        total_uncompressed_bytes=1,
        total_compressed_bytes=1,
        inventory_sha256=audit.inventory_sha256,
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
                "extracted_blob_sha256": evidence_hashes.get(member_path),
            }
            for ordinal, member_path in enumerate(plan.required_member_paths)
        ],
    )
    return database, plan


def test_plan_preserves_alias_timing_loop_geometry_roles_and_quarantine(
    tmp_path: Path,
) -> None:
    audit = audit_supertux_archive(_fixture_archive(tmp_path))
    plan = plan_supertux_projection(audit, TAXONOMY)

    assert plan.projected_sequence_count == 5
    assert plan.projected_entity_count == 1
    assert plan.projected_frame_count == 13
    assert plan.projected_animated_sequence_count == 4
    assert plan.projected_static_sequence_count == 1
    assert plan.projected_runtime_controlled_count == 1
    assert plan.projected_custom_finite_count == 4
    assert plan.deferred_transform_sequence_count == 2
    assert len(plan.exclusions) == 2
    assert plan.excluded_frame_count == 2
    assert len(plan.required_source_image_hashes) == 4
    assert len(plan.required_member_paths) == 13
    assert plan.projected_occurrence_link_count == 44

    walk_left = next(record for record in plan.records if record.declared_name == "walk-left")
    assert walk_left.entity.adapter_entity_class == "humanoid"
    assert walk_left.entity.normalized_entity_class == "humanoid"
    assert walk_left.normalized_action == "walk"
    assert walk_left.canonical_direction == "left"
    assert walk_left.frame_duration_milliseconds == 50
    assert [frame.duration_milliseconds for frame in walk_left.frames] == [50, 50, 50]
    assert [frame.source_frame_index for frame in walk_left.frames] == [0, 0, 0]
    assert [frame.source_layer_index for frame in walk_left.frames] == [0, 1, 0]
    assert walk_left.loop_mode == "engine_custom_finite"
    assert walk_left.effective_loops == 1
    assert walk_left.effective_loop_frame == 2
    assert walk_left.loop_start_ordinal == 1
    assert walk_left.loopable is False
    assert walk_left.cycle_frames is None
    assert walk_left.variable_frame_geometry
    assert (walk_left.width, walk_left.height) == (12, 8)

    walk_right = next(record for record in plan.records if record.declared_name == "walk-right")
    assert walk_right.alias_kind == "mirror"
    assert walk_right.alias_target == "walk-left"
    assert walk_right.alias_chain == ("mirror:walk-left",)
    assert {frame.transform for frame in walk_right.frames} == {"horizontal_flip"}
    assert walk_right.has_deferred_transform

    roof = next(record for record in plan.records if record.declared_name == "roof-left")
    assert roof.normalized_action == "other"
    assert "explicit_other_preserve_source" in roof.normalized_action_basis
    assert {frame.transform for frame in roof.frames} == {"vertical_flip"}

    slow = next(record for record in plan.records if record.declared_name == "slow-left")
    assert slow.declared_fps == 5
    assert slow.effective_fps == 20
    assert slow.hitbox == walk_left.hitbox
    assert slow.alias_chain == ("clone:walk-left",)

    assert {row.role for row in plan.exclusions} == {"effect_layer", "modular_component"}
    assert all("manifest:not_a_complete_entity_manifest" in row.reasons for row in plan.exclusions)
    assert len(plan.records) + len(plan.exclusions) == audit.counts.action_declarations
    assert json.loads(plan.canonical_json())["projection_version"] == plan.projection_version
    assert (
        plan.projection_manifest_sha256
        == plan_supertux_projection(audit, TAXONOMY).projection_manifest_sha256
    )
    changed = plan_supertux_projection(replace(audit, commit="different-source-fact"), TAXONOMY)
    assert changed.projection_manifest_sha256 != plan.projection_manifest_sha256


def test_temp_projection_is_query_only_then_atomic_idempotent_and_exact(
    tmp_path: Path,
) -> None:
    archive_path = _fixture_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    before = database.path.stat().st_mtime_ns
    preparation = plan_supertux_projection_preparation(database.path, plan)
    readiness = check_supertux_projection_readiness(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert preparation == readiness
    assert preparation.ready
    assert preparation.archive_inventory_matches
    assert preparation.archive_inventory_sha256 == plan.source_inventory_sha256
    assert preparation.next_steps == ()
    assert preparation.required_member_count == preparation.present_member_count == 13
    assert preparation.required_source_image_count == 4
    assert preparation.present_extracted_source_image_count == 4
    assert preparation.present_registered_source_image_count == 4
    assert preparation.required_non_image_evidence_count == 9
    assert preparation.present_verified_non_image_evidence_count == 9
    assert before == after

    first = project_supertux_audit(database, plan, TAXONOMY)
    second = project_supertux_audit(database, plan, TAXONOMY)
    assert (first.created_sequences, first.reused_sequences) == (5, 0)
    assert (second.created_sequences, second.reused_sequences) == (0, 5)
    assert first.projected_entities == second.projected_entities == 1
    assert first.projected_frames == second.projected_frames == 13
    assert first.occurrence_links == second.occurrence_links == 44
    assert first.exclusions == second.exclusions == 2
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
            "sequences": 5,
            "sequence_source_keys": 5,
            "sequence_subjects": 5,
            "motion_annotations": 5,
            "sequence_frames": 13,
            "sequence_occurrences": 44,
            "rights_observations": 0,
        }
        sequence_rows = connection.execute(
            "SELECT id, width, height, loop_mode, action, direction, quality_tier, "
            "metadata_json FROM sequences"
        ).fetchall()
        mirrored = next(
            row
            for row in sequence_rows
            if json.loads(row["metadata_json"])["declared_name"] == "walk-right"
        )
        metadata = json.loads(mirrored["metadata_json"])
        assert (mirrored["width"], mirrored["height"]) == (12, 8)
        assert mirrored["loop_mode"] == "engine_custom_finite"
        assert mirrored["action"] == "walk"
        assert mirrored["direction"] == "right"
        assert mirrored["quality_tier"] == (
            "P0_exact_supertux_geometric_transform_materializer_required"
        )
        assert metadata["source_transform_order"] == [
            "horizontal_flip",
            "horizontal_flip",
            "horizontal_flip",
        ]
        assert metadata["alias_chain"] == ["mirror:walk-left"]
        assert metadata["loop_semantics"]["effective_loops"] == 1
        # Mirror copies source frames and custom loop count, while retaining
        # the declaration's default loop-frame under the pinned engine path.
        assert metadata["loop_semantics"]["effective_loop_frame_1_based"] == 1
        assert metadata["transform_materialization_required"] is True
        assert metadata["current_canonical_materializer_compatible"] is False
        assert metadata["required_geometric_transform_operations"] == ["horizontal_flip"]
        assert metadata["model_ready_materialization_eligible"] is False
        assert metadata["model_ready_exclusion_reasons"] == [
            "supertux_engine_loop_mode_not_fixed_phase_normalized",
            "geometric_transform_materializer_not_implemented",
        ]
        assert metadata["rights_scope"]["repository_license_expression"] == "GPL-3.0"

        frames = connection.execute(
            "SELECT ordinal, source_frame_index, duration_ms, phase, direction, view, "
            "metadata_json FROM sequence_frames WHERE sequence_id=? ORDER BY ordinal",
            (mirrored["id"],),
        ).fetchall()
        assert [row["ordinal"] for row in frames] == [0, 1, 2]
        assert [row["source_frame_index"] for row in frames] == [0, 0, 0]
        assert [row["duration_ms"] for row in frames] == [50, 50, 50]
        assert [row["phase"] for row in frames] == [None, None, None]
        assert {row["direction"] for row in frames} == {"right"}
        assert {row["view"] for row in frames} == {"platformer"}
        first_frame_metadata = json.loads(frames[0]["metadata_json"])
        second_frame_metadata = json.loads(frames[1]["metadata_json"])
        assert first_frame_metadata["frame_rect"] == {
            "bottom": 8,
            "coordinate_space": "source_image",
            "height": 8,
            "left": 0,
            "right": 12,
            "top": 0,
            "width": 12,
        }
        assert second_frame_metadata["frame_rect"]["width"] == 10
        assert first_frame_metadata["transform_recipe"] == {
            "apply_to_source_pixels": True,
            "lossless": True,
            "materialized_in_source_blob": False,
            "operation": "horizontal_flip",
        }
        assert first_frame_metadata["current_canonical_materializer_compatible"] is False

        occurrences = connection.execute(
            "SELECT occurrence_role FROM sequence_occurrences WHERE sequence_id=?",
            (mirrored["id"],),
        ).fetchall()
        assert len(occurrences) == 9
        assert {row["occurrence_role"] for row in occurrences} == {
            "supertux_collection_rights_and_credits_evidence",
            "supertux_complete_entity_source_image",
            "supertux_engine_animation_semantics",
            "supertux_sprite_manifest",
        }

        idle = next(
            row
            for row in sequence_rows
            if json.loads(row["metadata_json"])["declared_name"] == "idle"
        )
        assert idle["quality_tier"] == "F0_lossless_supertux_exact_source_pixels"
        annotation = connection.execute(
            "SELECT loopable, cycle_frames, phase_zero_frame FROM motion_annotations "
            "WHERE sequence_id=?",
            (idle["id"],),
        ).fetchone()
        assert annotation is not None
        assert annotation["loopable"] is None
        assert annotation["cycle_frames"] is None
        assert annotation["phase_zero_frame"] is None

    # This is the executable fail-closed gate for current model inputs.  The
    # fixed-phase model-ready selector rejects every engine-specific loop mode;
    # transform-deferred rows additionally carry the geometric incompatibility.
    assert (
        load_sequence_samples(
            database.path,
            SnapshotFilters(temporal_mode="model_ready", include_source_ids=(SOURCE_ID,)),
        )
        == ()
    )


def test_readiness_reports_preparation_and_hash_mismatch_without_writing(
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
    preparation = plan_supertux_projection_preparation(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert not preparation.ready
    assert member_path in preparation.members_requiring_extraction
    assert len(preparation.source_image_hash_mismatches) == 1
    assert member_path in preparation.source_image_hash_mismatches[0]
    assert preparation.next_steps[-1] == "investigate 1 immutable hash mismatches"
    assert before == after
    with pytest.raises(ValueError, match="CAS hash mismatch"):
        project_supertux_audit(database, plan, TAXONOMY)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM sequences").fetchone()[0] == 0


def test_readiness_and_preflight_reject_inventory_digest_drift(tmp_path: Path) -> None:
    archive_path = _fixture_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    with database.connect() as connection:
        connection.execute(
            "UPDATE archive_inventories SET inventory_sha256=? WHERE archive_blob_sha256=?",
            ("e" * 64, plan.archive_sha256),
        )
    before = database.path.stat().st_mtime_ns
    preparation = plan_supertux_projection_preparation(database.path, plan)
    after = database.path.stat().st_mtime_ns

    assert preparation.archive_inventory_present
    assert not preparation.archive_inventory_matches
    assert preparation.archive_inventory_sha256 == "e" * 64
    assert preparation.next_steps == ("investigate the pinned ZIP inventory digest mismatch",)
    assert not preparation.ready
    assert before == after
    with pytest.raises(ValueError, match="inventory hash mismatch"):
        project_supertux_audit(database, plan, TAXONOMY)


def test_readiness_and_preflight_bind_rights_evidence_hash(tmp_path: Path) -> None:
    archive_path = _fixture_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    wrong_hash = "d" * 64
    database.register_blob(
        sha256=wrong_hash,
        size_bytes=1,
        storage_path=tmp_path / "wrong-rights-evidence",
        mime_type="application/octet-stream",
    )
    member_path = plan.rights_documents[0].member_path
    with database.connect() as connection:
        connection.execute(
            "UPDATE archive_members SET extracted_blob_sha256=? "
            "WHERE archive_blob_sha256=? AND member_path=?",
            (wrong_hash, plan.archive_sha256, member_path),
        )
    preparation = plan_supertux_projection_preparation(database.path, plan)

    assert not preparation.ready
    assert preparation.present_verified_non_image_evidence_count == 8
    assert member_path in preparation.members_requiring_extraction
    assert len(preparation.non_image_evidence_hash_mismatches) == 1
    assert member_path in preparation.non_image_evidence_hash_mismatches[0]
    with pytest.raises(ValueError, match="evidence CAS hash mismatch"):
        project_supertux_audit(database, plan, TAXONOMY)


def test_projection_does_not_initialize_or_mutate_a_blank_database(tmp_path: Path) -> None:
    archive_path = _fixture_archive(tmp_path)
    plan = plan_supertux_projection(audit_supertux_archive(archive_path), TAXONOMY)
    database = IndexDB(tmp_path / "absent-index.sqlite3")

    assert not database.path.exists()
    with pytest.raises(ValueError, match="existing initialized index database"):
        project_supertux_audit(database, plan, TAXONOMY)
    assert not database.path.exists()


def test_readonly_preflight_does_not_mutate_an_initialized_not_ready_database(
    tmp_path: Path,
) -> None:
    archive_path = _fixture_archive(tmp_path)
    plan = plan_supertux_projection(audit_supertux_archive(archive_path), TAXONOMY)
    database = IndexDB(tmp_path / "initialized-not-ready.sqlite3")
    database.initialize()
    before = hashlib.sha256(database.path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="source registry row is missing"):
        project_supertux_audit(database, plan, TAXONOMY)
    after = hashlib.sha256(database.path.read_bytes()).hexdigest()
    assert before == after


def test_taxonomy_action_family_drift_is_rejected_before_writes(tmp_path: Path) -> None:
    archive_path = _fixture_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    families = dict(TAXONOMY.action_families)
    families["locomotion"] = tuple(action for action in families["locomotion"] if action != "walk")
    families["other"] = (*families["other"], "walk")
    drifted = replace(TAXONOMY, action_families=families)

    with pytest.raises(ValueError, match="action-family mapping has changed"):
        project_supertux_audit(database, plan, drifted)
    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM sequences").fetchone()[0] == 0


def test_projection_rolls_back_as_one_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = _fixture_archive(tmp_path)
    database, plan = _indexed_database(tmp_path, archive_path)
    original = database.add_sequence_frame
    calls = 0

    def fail_during_second_frame(**kwargs: object) -> None:
        nonlocal calls
        calls += 1
        original(**kwargs)  # type: ignore[arg-type]
        if calls == 2:
            raise RuntimeError("fixture interruption")

    monkeypatch.setattr(database, "add_sequence_frame", fail_during_second_frame)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        project_supertux_audit(database, plan, TAXONOMY)

    with database.connect() as connection:
        for table in (
            "entities",
            "sequences",
            "sequence_source_keys",
            "sequence_subjects",
            "motion_annotations",
            "sequence_frames",
            "sequence_occurrences",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact SuperTux CAS archive missing")
def test_exact_pinned_plan_is_deterministic_and_conservative() -> None:
    plan = plan_known_supertux_projection(EXACT_ARCHIVE, TAXONOMY)

    assert plan.archive_sha256 == EXPECTED_SUPERTUX_ARCHIVE_SHA256
    assert plan.repository_commit == SUPERTUX_COMMIT
    assert plan.source_inventory_sha256 == (
        "2da2740e59deeb960db9d24505171e7a97ab2cc5b3968b82d353f643927c48d2"
    )
    assert plan.source_audit_record_sha256 == (
        "1b5fd92ffbfe2dc7fbd9ca7f53d0c7fd2b540b84f8a3da6f0fbe722f09183703"
    )
    assert plan.projection_manifest_sha256 == EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256
    assert plan.projected_sequence_count == 1_010
    assert plan.projected_entity_count == 96
    assert plan.projected_frame_count == 7_103
    assert plan.projected_animated_sequence_count == 657
    assert plan.projected_static_sequence_count == 353
    assert plan.projected_runtime_controlled_count == 736
    assert plan.projected_custom_finite_count == 274
    assert plan.projected_custom_infinite_count == 0
    assert plan.deferred_transform_sequence_count == 437
    assert len(plan.exclusions) == 141
    assert plan.excluded_frame_count == 588
    assert len(plan.required_member_paths) == 1_982
    assert len(plan.required_source_image_hashes) == 1_841
    assert len(plan.projected_source_image_hashes) == 1_612
    assert len(plan.excluded_source_image_hashes) == 245
    assert len(set(dict(plan.projected_source_image_hashes).values())) == 1_609
    assert len(plan.required_evidence_hashes) == 1_982
    assert plan.projected_occurrence_link_count == 11_735
    assert plan.duplicate_track_content_groups == 150
    assert plan.duplicate_track_content_excess == 296
    assert sum(record.variable_frame_geometry for record in plan.records) == 4
    assert len({record.sequence_source_key for record in plan.records}) == 1_010
    assert len({record.entity.entity_external_key for record in plan.records}) == 96
    assert Counter(
        record.effective_loops for record in plan.records if record.has_custom_loops
    ) == {1: 274}
    assert Counter(frame.transform for record in plan.records for frame in record.frames) == {
        "identity": 3_752,
        "horizontal_flip": 3_303,
        "vertical_flip": 32,
        "horizontal_vertical_flip": 16,
    }
    assert Counter(record.canonical_direction for record in plan.records) == {
        "left": 430,
        "right": 429,
        "none": 110,
        "up_left": 8,
        "up_right": 8,
        "down_right": 7,
        "down": 7,
        "down_left": 7,
        "up": 4,
    }
    assert Counter(row.role for row in plan.exclusions) == {
        "modular_component": 79,
        "effect_layer": 49,
        "complete_entity": 10,
        "deprecated": 3,
    }


@pytest.mark.skipif(
    not (EXACT_ARCHIVE.is_file() and LIVE_INDEX.is_file()),
    reason="exact SuperTux CAS archive or live index missing",
)
def test_live_preparation_is_query_only_and_exact() -> None:
    plan = plan_known_supertux_projection(EXACT_ARCHIVE, TAXONOMY)
    before = LIVE_INDEX.stat().st_mtime_ns
    preparation = plan_supertux_projection_preparation(LIVE_INDEX, plan)
    after = LIVE_INDEX.stat().st_mtime_ns

    assert preparation.archive_sha256 == EXPECTED_SUPERTUX_ARCHIVE_SHA256
    assert preparation.projection_manifest_sha256 == (EXPECTED_PINNED_PROJECTION_MANIFEST_SHA256)
    assert preparation.required_member_count == 1_982
    assert preparation.required_source_image_count == 1_841
    assert preparation.required_non_image_evidence_count == 141
    assert 0 <= preparation.present_member_count <= 1_982
    assert 0 <= preparation.present_extracted_source_image_count <= 1_841
    assert 0 <= preparation.present_registered_source_image_count <= 1_841
    assert 0 <= preparation.present_verified_non_image_evidence_count <= 141
    if preparation.archive_inventory_present:
        assert preparation.archive_inventory_sha256 is not None
    else:
        assert preparation.archive_inventory_sha256 is None
        assert not preparation.archive_inventory_matches
    if preparation.ready:
        assert preparation.present_member_count == 1_982
        assert preparation.present_extracted_source_image_count == 1_841
        assert preparation.present_registered_source_image_count == 1_841
        assert preparation.present_verified_non_image_evidence_count == 141
        assert preparation.next_steps == ()
    assert before == after
