import json
import sqlite3
from pathlib import Path

import pytest

from spritelab.db import IndexDB


def test_schema_and_source_registration(tmp_path: Path) -> None:
    database = IndexDB(tmp_path / "index.sqlite3")
    database.register_source(
        source_id="test",
        kind="fixture",
        name="Fixture",
        root_url="https://example.invalid/",
        adapter_version="1",
    )

    counts = database.counts()
    assert counts["sources"] == 1
    with database.connect() as connection:
        events = connection.execute(
            "SELECT event_type, entity_id FROM events ORDER BY id"
        ).fetchall()
    assert [(row["event_type"], row["entity_id"]) for row in events] == [
        ("source_registered", "test")
    ]


def test_identity_groups_multiple_action_sequences(tmp_path: Path) -> None:
    database = IndexDB(tmp_path / "index.sqlite3")
    database.register_source(
        source_id="pack",
        kind="fixture",
        name="Pack",
        root_url="https://example.invalid/pack/",
        adapter_version="1",
    )
    item_id = database.upsert_item(
        source_id="pack",
        external_id="hero",
        canonical_url="https://example.invalid/pack/hero",
    )
    entity_id = database.upsert_entity(
        source_id="pack",
        external_identity_key="hero-blue",
        representative_item_id=item_id,
        display_name="Blue hero",
        entity_class="humanoid",
        entity_subclass="adventurer",
        taxonomy_version="1.0",
    )

    for sequence_id, action in (("seq_idle", "idle"), ("seq_run", "run")):
        with database.connect() as connection:
            connection.execute(
                """
                INSERT INTO sequences(
                    id, item_id, extraction_method, width, height, frame_count,
                    quality_tier, metadata_json, created_at
                ) VALUES (?, ?, 'fixture', 16, 24, 4, 'F0', '{}', datetime('now'))
                """,
                (sequence_id, item_id),
            )
        database.link_sequence_subject(sequence_id=sequence_id, entity_id=entity_id)
        database.annotate_motion(
            sequence_id=sequence_id,
            vocabulary_version="1.0",
            normalized_action=action,
            action_family="stationary" if action == "idle" else "locomotion",
            annotation_method="fixture",
            source_action=f"Hero_{action.title()}",
            loopable=True,
            cycle_frames=4,
        )

    with database.connect() as connection:
        rows = connection.execute(
            """
            SELECT e.external_identity_key, m.normalized_action, m.source_action
            FROM sequence_subjects AS ss
            JOIN entities AS e ON e.id = ss.entity_id
            JOIN motion_annotations AS m ON m.sequence_id = ss.sequence_id
            ORDER BY m.normalized_action
            """
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("hero-blue", "idle", "Hero_Idle"),
        ("hero-blue", "run", "Hero_Run"),
    ]


def test_archive_and_media_index_round_trip(tmp_path: Path) -> None:
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    with database.connect() as connection:
        for digest in ("a" * 64, "b" * 64):
            connection.execute(
                """
                INSERT INTO blobs(sha256, size_bytes, storage_path, first_seen_at)
                VALUES (?, 10, ?, datetime('now'))
                """,
                (digest, f"objects/{digest}"),
            )
    database.upsert_archive_inventory(
        archive_blob_sha256="a" * 64,
        archive_format="zip",
        member_count=1,
        file_count=1,
        total_uncompressed_bytes=10,
        total_compressed_bytes=8,
        inventory_sha256="c" * 64,
        policy={"max_member_bytes": 100},
    )
    database.upsert_archive_members(
        archive_blob_sha256="a" * 64,
        members=[
            {
                "ordinal": 0,
                "member_path": "sprites/hero.png",
                "normalized_path": "sprites/hero.png",
                "member_kind": "file",
                "size_bytes": 10,
                "compressed_bytes": 8,
                "crc32": 42,
                "compression_method": 8,
            }
        ],
    )
    database.attach_archive_member_blob(
        archive_blob_sha256="a" * 64,
        ordinal=0,
        extracted_blob_sha256="b" * 64,
        selected_role="sprite",
    )
    database.record_media_observation(
        blob_sha256="b" * 64,
        inspector_version="png-v1",
        media_format="PNG",
        width=16,
        height=24,
        mode="RGBA",
        has_alpha=True,
        is_animated=False,
        frame_count=1,
        pixel_sha256="d" * 64,
    )

    counts = database.counts()
    assert counts["archive_members"] == 1
    assert counts["media_observations"] == 1
    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT normalized_path, extracted_blob_sha256, selected_role
            FROM archive_members
            """
        ).fetchone()
    assert tuple(row) == ("sprites/hero.png", "b" * 64, "sprite")


def test_sequence_source_keys_are_idempotent_and_stable(tmp_path: Path) -> None:
    database = IndexDB(tmp_path / "index.sqlite3")
    database.register_source(
        source_id="pack",
        kind="fixture",
        name="Pack",
        root_url="https://example.invalid/",
        adapter_version="1",
    )
    sequence_id = database.create_sequence(
        extraction_method="fixture",
        width=8,
        height=8,
        frame_count=2,
        quality_tier="F0",
    )
    database.register_sequence_source_key(
        source_id="pack", external_sequence_key="hero:run", sequence_id=sequence_id
    )
    database.register_sequence_source_key(
        source_id="pack", external_sequence_key="hero:run", sequence_id=sequence_id
    )

    assert (
        database.find_sequence_by_source_key(source_id="pack", external_sequence_key="hero:run")
        == sequence_id
    )

    database.update_sequence_facts(
        sequence_id=sequence_id,
        extraction_method="fixture-v2",
        extraction_confidence=0.55,
        width=16,
        height=24,
        frame_count=3,
        loop_mode="unknown",
        action="run",
        direction="east",
        quality_tier="F1",
        metadata={"timing_known": False},
    )
    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT extraction_method, extraction_confidence, width, height,
                   frame_count, loop_mode, action, direction, quality_tier,
                   metadata_json
            FROM sequences WHERE id=?
            """,
            (sequence_id,),
        ).fetchone()
    assert tuple(row[:9]) == (
        "fixture-v2",
        0.55,
        16,
        24,
        3,
        "unknown",
        "run",
        "east",
        "F1",
    )
    assert json.loads(row[9]) == {"timing_known": False}


def test_bulk_archive_registration_handles_more_than_sql_variable_limit(
    tmp_path: Path,
) -> None:
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    archive_sha = "a" * 64
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO blobs(sha256, size_bytes, storage_path, first_seen_at)
            VALUES (?, 1, 'archive.zip', datetime('now'))
            """,
            (archive_sha,),
        )
    members = [
        {
            "ordinal": ordinal,
            "member_path": f"sprites/{ordinal}.png",
            "normalized_path": f"sprites/{ordinal}.png",
            "member_kind": "file",
            "size_bytes": 1,
            "compressed_bytes": 1,
        }
        for ordinal in range(1_200)
    ]
    database.upsert_archive_members(
        archive_blob_sha256=archive_sha,
        members=members,
    )
    extracted = [
        {
            "ordinal": ordinal,
            "sha256": f"{ordinal + 1:064x}",
            "size_bytes": 1,
            "storage_path": f"objects/{ordinal}",
        }
        for ordinal in range(1_200)
    ]

    assert (
        database.register_archive_extractions(
            archive_blob_sha256=archive_sha,
            extracted=extracted,
            selected_role="sprite",
        )
        == 1_200
    )


def test_transaction_reuses_one_connection_and_commits_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    original_connect = sqlite3.connect
    connection_calls = 0

    def counted_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        nonlocal connection_calls
        connection_calls += 1
        return original_connect(*args, **kwargs)

    monkeypatch.setattr("spritelab.db.sqlite3.connect", counted_connect)
    with database.transaction() as outer_connection:
        sequence_id = database.create_sequence(
            extraction_method="fixture",
            width=8,
            height=8,
            frame_count=1,
            quality_tier="F0",
        )
        with database.connect() as nested_connection:
            assert nested_connection is outer_connection
        with database.transaction() as nested_transaction:
            assert nested_transaction is outer_connection
        assert (
            database.find_sequence_by_source_key(
                source_id="fixture",
                external_sequence_key="missing",
            )
            is None
        )

    assert connection_calls == 1
    with original_connect(database.path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM sequences WHERE id=?", (sequence_id,)
            ).fetchone()[0]
            == 1
        )


def test_transaction_rolls_back_all_nested_helper_writes(tmp_path: Path) -> None:
    database = IndexDB(tmp_path / "index.sqlite3")
    database.register_source(
        source_id="fixture",
        kind="fixture",
        name="Fixture",
        root_url="https://example.invalid/",
        adapter_version="1",
    )

    with pytest.raises(RuntimeError, match="abort batch"), database.transaction():
        sequence_id = database.create_sequence(
            extraction_method="fixture",
            width=8,
            height=8,
            frame_count=1,
            quality_tier="F0",
        )
        database.register_sequence_source_key(
            source_id="fixture",
            external_sequence_key="one",
            sequence_id=sequence_id,
        )
        raise RuntimeError("abort batch")

    with database.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM sequences").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM sequence_source_keys").fetchone()[0] == 0
