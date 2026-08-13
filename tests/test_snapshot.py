import hashlib
import json
from pathlib import Path

import pytest

from spritelab.dataset import SplitPolicy
from spritelab.db import IndexDB
from spritelab.snapshot import (
    SnapshotFilters,
    build_snapshot_from_index,
    export_snapshot,
    load_sequence_samples,
    write_snapshot,
)

SOURCE = "source-fixture"
ITEM_WOLF = "item-wolf"
ITEM_KNIGHT = "item-knight"
ENTITY_WOLF = "entity-wolf"
ENTITY_KNIGHT = "entity-knight"
A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
E = "e" * 64
F = "f" * 64
SECONDARY_SOURCE = "source-secondary"


def _fixture_database(tmp_path: Path) -> Path:
    path = tmp_path / "index.sqlite3"
    database = IndexDB(path)
    database.initialize()
    now = "2026-01-01T00:00:00+00:00"
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO sources(
                id, kind, name, root_url, adapter_version, config_json, created_at
            ) VALUES (?, 'fixture', 'Fixture sprites', 'https://example.invalid/', '7', '{}', ?)
            """,
            (SOURCE, now),
        )
        connection.executemany(
            """
            INSERT INTO items(
                id, source_id, external_id, canonical_url, title, metadata_json,
                first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, '{}', ?, ?)
            """,
            (
                (
                    ITEM_WOLF,
                    SOURCE,
                    "wolf-pack",
                    "https://example.invalid/wolf",
                    "Wolf",
                    now,
                    now,
                ),
                (
                    ITEM_KNIGHT,
                    SOURCE,
                    "knight-pack",
                    "https://example.invalid/knight",
                    "Knight",
                    now,
                    now,
                ),
            ),
        )
        connection.executemany(
            """
            INSERT INTO blobs(sha256, size_bytes, mime_type, storage_path, first_seen_at)
            VALUES (?, 100, 'image/png', ?, ?)
            """,
            ((digest, f"objects/{digest}", now) for digest in (F, D, B, E, A, C)),
        )
        connection.executemany(
            """
            INSERT INTO entities(
                id, source_id, external_identity_key, display_name, entity_class,
                taxonomy_version, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, '1.0', '{}', ?, ?)
            """,
            (
                (ENTITY_WOLF, SOURCE, "wolf", "Wolf", "animal", now, now),
                (ENTITY_KNIGHT, SOURCE, "knight", "Knight", "humanoid", now, now),
            ),
        )
        connection.executemany(
            """
            INSERT INTO sequences(
                id, item_id, source_blob_sha256, extraction_method,
                extraction_confidence, width, height, frame_count, loop_mode,
                action, direction, quality_tier, metadata_json, created_at
            ) VALUES (?, ?, ?, 'fixture-v1', 1.0, 16, 16, ?, ?, ?, 'right', 'F0', ?, ?)
            """,
            (
                (
                    "seq-pose-run",
                    ITEM_WOLF,
                    E,
                    2,
                    "unknown",
                    "run",
                    '{"exact_engine_timing":false}',
                    now,
                ),
                ("seq-single", ITEM_KNIGHT, D, 1, "unknown", "idle", "{}", now),
                ("seq-timed-attack", ITEM_KNIGHT, C, 2, "one_shot", "attack", "{}", now),
                ("seq-timed-idle", ITEM_WOLF, A, 2, "loop", "idle", "{}", now),
            ),
        )
        connection.executemany(
            """
            INSERT INTO sequence_subjects(sequence_id, entity_id, role, metadata_json)
            VALUES (?, ?, 'primary', '{}')
            """,
            (
                ("seq-pose-run", ENTITY_WOLF),
                ("seq-single", ENTITY_KNIGHT),
                ("seq-timed-attack", ENTITY_KNIGHT),
                ("seq-timed-idle", ENTITY_WOLF),
            ),
        )
        connection.executemany(
            """
            INSERT INTO motion_annotations(
                sequence_id, vocabulary_version, source_action, normalized_action,
                action_family, view, direction, loopable, cycle_frames,
                phase_zero_frame, confidence, annotation_method, conditioning_json,
                created_at, updated_at
            ) VALUES (?, '1.0', ?, ?, ?, 'side', 'right', ?, ?, 0, 1.0,
                      'fixture', ?, ?, ?)
            """,
            (
                (
                    "seq-pose-run",
                    "RunPoseSet",
                    "run",
                    "locomotion",
                    None,
                    None,
                    '{"timing_known":false}',
                    now,
                    now,
                ),
                (
                    "seq-single",
                    "IdlePose",
                    "idle",
                    "stationary",
                    None,
                    None,
                    "{}",
                    now,
                    now,
                ),
                (
                    "seq-timed-attack",
                    "Attack",
                    "attack",
                    "combat",
                    0,
                    None,
                    '{"timing_known":true}',
                    now,
                    now,
                ),
                (
                    "seq-timed-idle",
                    "Idle",
                    "idle",
                    "stationary",
                    1,
                    2,
                    '{"timing_known":true}',
                    now,
                    now,
                ),
            ),
        )
        connection.executemany(
            """
            INSERT INTO sequence_frames(
                sequence_id, ordinal, source_blob_sha256, source_frame_index,
                duration_ms, phase, direction, view, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, 'right', 'side', '{}')
            """,
            (
                ("seq-timed-idle", 1, B, 1, 90.0, 0.5),
                ("seq-timed-idle", 0, A, 0, 80.0, 0.0),
                ("seq-timed-attack", 1, D, 1, 100.0, 1.0),
                ("seq-timed-attack", 0, C, 0, 120.0, 0.0),
                ("seq-pose-run", 1, E, 1, 75.0, 1.0),
                ("seq-pose-run", 0, E, 0, 75.0, 0.0),
                ("seq-single", 0, D, 0, 100.0, 0.0),
            ),
        )
        connection.executemany(
            """
            INSERT INTO sequence_source_keys(
                source_id, external_sequence_key, sequence_id, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            (
                (SOURCE, "wolf:run", "seq-pose-run", now),
                (SOURCE, "knight:idle", "seq-single", now),
                (SOURCE, "knight:attack", "seq-timed-attack", now),
                (SOURCE, "wolf:idle", "seq-timed-idle", now),
            ),
        )
        connection.executemany(
            """
            INSERT INTO duplicate_edges(
                id, left_blob_sha256, right_blob_sha256, method,
                distance, parameters_json, created_at
            ) VALUES (?, ?, ?, 'fixture-near', 0.1, '{}', ?)
            """,
            (
                ("duplicate-a-e", A, E, now),
                ("duplicate-e-c", E, C, now),
            ),
        )
        connection.execute(
            """
            INSERT INTO retrievals(
                id, item_id, url, requested_at, completed_at, status_code, blob_sha256
            ) VALUES ('retrieval-wolf', ?, 'https://example.invalid/wolf.zip', ?, ?, 200, ?)
            """,
            (ITEM_WOLF, now, now, F),
        )
        connection.execute(
            """
            INSERT INTO rights_observations(
                id, item_id, observed_at, license_raw, license_expression,
                basis, metadata_json
            ) VALUES ('rights-wolf', ?, ?, 'CC0', 'CC0-1.0', 'fixture', '{}')
            """,
            (ITEM_WOLF, now),
        )
        connection.execute(
            """
            INSERT INTO item_blobs(
                id, item_id, blob_sha256, role, original_url, observed_at
            ) VALUES ('itemblob-wolf', ?, ?, 'archive',
                      'https://example.invalid/wolf.zip', ?)
            """,
            (ITEM_WOLF, F, now),
        )
        connection.execute(
            """
            INSERT INTO archive_members(
                archive_blob_sha256, ordinal, member_path, normalized_path,
                member_kind, size_bytes, compressed_bytes, extracted_blob_sha256,
                selected_role, inspection_status, metadata_json, observed_at
            ) VALUES (?, 0, 'wolf/idle.webp', 'wolf/idle.webp', 'file', 100, 80, ?,
                      'sprite', 'media_inspected', '{}', ?)
            """,
            (F, A, now),
        )
        connection.execute(
            """
            INSERT INTO sequence_occurrences(
                sequence_id, archive_blob_sha256, archive_member_ordinal,
                occurrence_role, metadata_json, created_at
            ) VALUES ('seq-timed-idle', ?, 0, 'animated_source', '{}', ?)
            """,
            (F, now),
        )
    return path


def _add_source_key_only_sequence(database_path: Path) -> None:
    now = "2026-01-01T00:00:00+00:00"
    with IndexDB(database_path).connect() as connection:
        connection.execute(
            """
            INSERT INTO sources(
                id, kind, name, root_url, adapter_version, config_json, created_at
            ) VALUES (?, 'fixture', 'Secondary sprites',
                      'https://secondary.example.invalid/', '1', '{}', ?)
            """,
            (SECONDARY_SOURCE, now),
        )
        connection.execute(
            """
            INSERT INTO sequences(
                id, item_id, source_blob_sha256, extraction_method,
                extraction_confidence, width, height, frame_count, loop_mode,
                action, direction, quality_tier, metadata_json, created_at
            ) VALUES ('seq-source-key-only', NULL, ?, 'fixture-v1', 1.0,
                      16, 16, 2, 'loop', 'walk', 'right', 'F0', '{}', ?)
            """,
            (A, now),
        )
        connection.execute(
            """
            INSERT INTO motion_annotations(
                sequence_id, vocabulary_version, source_action, normalized_action,
                action_family, view, direction, loopable, cycle_frames,
                phase_zero_frame, confidence, annotation_method, conditioning_json,
                created_at, updated_at
            ) VALUES ('seq-source-key-only', '1.0', 'Walk', 'walk', 'locomotion',
                      'side', 'right', 1, 2, 0, 1.0, 'fixture',
                      '{"timing_known":true}', ?, ?)
            """,
            (now, now),
        )
        connection.executemany(
            """
            INSERT INTO sequence_frames(
                sequence_id, ordinal, source_blob_sha256, source_frame_index,
                duration_ms, phase, direction, view, metadata_json
            ) VALUES ('seq-source-key-only', ?, ?, ?, 100.0, ?, 'right', 'side', '{}')
            """,
            ((0, A, 0, 0.0), (1, B, 1, 0.5)),
        )
        connection.execute(
            """
            INSERT INTO sequence_source_keys(
                source_id, external_sequence_key, sequence_id, created_at
            ) VALUES (?, 'source-key-only:walk', 'seq-source-key-only', ?)
            """,
            (SECONDARY_SOURCE, now),
        )


def _add_timed_unknown_phase_sequence(database_path: Path) -> None:
    now = "2026-01-01T00:00:00+00:00"
    with IndexDB(database_path).connect() as connection:
        connection.execute(
            """
            INSERT INTO sequences(
                id, item_id, source_blob_sha256, extraction_method,
                extraction_confidence, width, height, frame_count, loop_mode,
                action, direction, quality_tier, metadata_json, created_at
            ) VALUES ('seq-timed-unknown-phase', ?, ?, 'fixture-v1', 1.0,
                      16, 16, 2, 'unknown', 'attack', 'right', 'F0', '{}', ?)
            """,
            (ITEM_KNIGHT, C, now),
        )
        connection.executemany(
            """
            INSERT INTO sequence_frames(
                sequence_id, ordinal, source_blob_sha256, source_frame_index,
                duration_ms, phase, direction, view, metadata_json
            ) VALUES ('seq-timed-unknown-phase', ?, ?, ?, 100.0, NULL,
                      'right', 'side', '{}')
            """,
            ((0, C, 0), (1, D, 1)),
        )
        connection.execute(
            """
            INSERT INTO sequence_source_keys(
                source_id, external_sequence_key, sequence_id, created_at
            ) VALUES (?, 'knight:unknown-phase', 'seq-timed-unknown-phase', ?)
            """,
            (SOURCE, now),
        )


def test_default_selection_requires_genuine_recorded_timing(tmp_path: Path) -> None:
    database_path = _fixture_database(tmp_path)
    before = database_path.read_bytes()

    samples = load_sequence_samples(database_path)

    assert [sample.sequence_id for sample in samples] == [
        "seq-timed-attack",
        "seq-timed-idle",
    ]
    idle = samples[1]
    assert idle.identity_id == ENTITY_WOLF
    assert idle.source_id == SOURCE
    assert idle.source_pack_id == ITEM_WOLF
    assert idle.entity_class == "animal"
    assert idle.action == "idle"
    assert idle.source_blob_sha256 == (A, B)
    assert f"entity:{ENTITY_WOLF}" in idle.duplicate_group_ids
    assert any(group.startswith("duplicate-component:") for group in idle.duplicate_group_ids)
    assert idle.metadata["temporal_evidence"]["known"] is True
    assert idle.metadata["source"]["id"] == SOURCE
    assert idle.metadata["item"]["id"] == ITEM_WOLF
    assert idle.metadata["sequence_source_keys"] == [
        {"external_sequence_key": "wolf:idle", "source_id": SOURCE}
    ]
    assert idle.metadata["retrieval_ids"] == ["retrieval-wolf"]
    assert idle.metadata["rights_observation_ids"] == ["rights-wolf"]
    assert idle.metadata["item_blob_occurrence_ids"] == ["itemblob-wolf"]
    assert idle.metadata["archive_occurrences"][0]["archive_blob_sha256"] == F
    assert database_path.read_bytes() == before


def test_model_ready_mode_requires_explicit_loop_mode_and_complete_phases(
    tmp_path: Path,
) -> None:
    database_path = _fixture_database(tmp_path)
    _add_timed_unknown_phase_sequence(database_path)

    known = load_sequence_samples(database_path, SnapshotFilters(temporal_mode="known"))
    model_ready = load_sequence_samples(
        database_path,
        SnapshotFilters(temporal_mode="model_ready"),
    )

    assert [sample.sequence_id for sample in known] == [
        "seq-timed-attack",
        "seq-timed-idle",
        "seq-timed-unknown-phase",
    ]
    assert [sample.sequence_id for sample in model_ready] == [
        "seq-timed-attack",
        "seq-timed-idle",
    ]


def test_temporal_and_action_filters_expose_pose_only_sequences(tmp_path: Path) -> None:
    database_path = _fixture_database(tmp_path)
    filters = SnapshotFilters(
        minimum_frame_count=2,
        actions=(" RUN ", "run"),
        temporal_mode="pose_only",
    )

    assert filters.actions == ("run",)
    samples = load_sequence_samples(database_path, filters)

    assert [sample.sequence_id for sample in samples] == ["seq-pose-run"]
    evidence = samples[0].metadata["temporal_evidence"]
    assert evidence["duration_source"] == "sequence_frames.duration_ms"
    assert evidence["known"] is False
    assert evidence["explicit_negative_claims"] == (
        "conditioning.timing_known=false",
        "sequence_metadata.exact_engine_timing=false",
    )

    all_samples = load_sequence_samples(
        database_path,
        SnapshotFilters(minimum_frame_count=1, temporal_mode="all"),
    )
    assert {sample.sequence_id for sample in all_samples} == {
        "seq-pose-run",
        "seq-single",
        "seq-timed-attack",
        "seq-timed-idle",
    }


def test_transitive_duplicate_edges_share_a_split_and_export_is_stable(tmp_path: Path) -> None:
    database_path = _fixture_database(tmp_path)
    policy = SplitPolicy(seed="snapshot-v1", group_source_pack=False)

    first = build_snapshot_from_index(database_path, policy=policy)
    second = build_snapshot_from_index(database_path, policy=policy)

    assert first == second
    assert first.sha256 == "7c1b69e197c83f96dae38c83126ab7b821e8b1223c2cb6b8aec40865994f0be1"
    assert first.sha256 == hashlib.sha256(first.canonical_json.encode()).hexdigest()
    assignments = {row.sequence_id: row for row in first.manifest.assignments}
    assert (
        assignments["seq-timed-idle"].component_id == assignments["seq-timed-attack"].component_id
    )
    assert assignments["seq-timed-idle"].split == assignments["seq-timed-attack"].split
    assert first.coverage.sample_count == 2
    assert first.timing_counts == {"known": 2, "pose_only": 0}

    one = write_snapshot(first, tmp_path / "one.json")
    exported = export_snapshot(
        database_path,
        tmp_path / "two.json",
        policy=policy,
    )
    assert exported == first
    assert one.read_bytes() == (tmp_path / "two.json").read_bytes()
    payload = json.loads(one.read_text(encoding="utf-8"))
    assert payload["manifest_sha256"] == first.manifest.sha256
    assert payload["filters"]["temporal_mode"] == "known"
    assert payload["filters"] == {
        "actions": [],
        "minimum_frame_count": 2,
        "temporal_mode": "known",
    }

    with IndexDB(database_path).connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM dataset_snapshots").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM dataset_members").fetchone()[0] == 0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"minimum_frame_count": 0},
        {"temporal_mode": "maybe"},
        {"include_source_ids": "source-fixture"},
        {"exclude_source_ids": ("source-fixture", 7)},
        {
            "include_source_ids": ("source-fixture",),
            "exclude_source_ids": (" source-fixture ",),
        },
    ),
)
def test_snapshot_filters_reject_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SnapshotFilters(**kwargs)  # type: ignore[arg-type]


def test_source_filters_use_all_resolved_associations_including_source_keys(
    tmp_path: Path,
) -> None:
    database_path = _fixture_database(tmp_path)
    _add_source_key_only_sequence(database_path)

    normalized = SnapshotFilters(
        include_source_ids=(f" {SECONDARY_SOURCE} ", SECONDARY_SOURCE),
        exclude_source_ids=("missing-source", " missing-source "),
    )
    assert normalized.include_source_ids == (SECONDARY_SOURCE,)
    assert normalized.exclude_source_ids == ("missing-source",)

    included = load_sequence_samples(database_path, normalized)
    assert [sample.sequence_id for sample in included] == ["seq-source-key-only"]
    assert included[0].source_id == SECONDARY_SOURCE
    assert included[0].metadata["source_ids"] == (SECONDARY_SOURCE,)
    assert included[0].metadata["item"] is None
    assert included[0].metadata["subjects"] == []
    assert included[0].metadata["sequence_source_keys"] == [
        {
            "external_sequence_key": "source-key-only:walk",
            "source_id": SECONDARY_SOURCE,
        }
    ]

    excluded = load_sequence_samples(
        database_path,
        SnapshotFilters(exclude_source_ids=(SOURCE,)),
    )
    assert [sample.sequence_id for sample in excluded] == ["seq-source-key-only"]

    artifact = build_snapshot_from_index(
        database_path,
        policy=SplitPolicy(seed="source-filter-v1", group_source_pack=False),
        filters=normalized,
    )
    payload = json.loads(artifact.canonical_json)
    assert payload["filters"]["include_source_ids"] == [SECONDARY_SOURCE]
    assert payload["filters"]["exclude_source_ids"] == ["missing-source"]
