from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from spritelab.db import IndexDB
from spritelab.reporting import export_provenance_reports


@pytest.fixture
def provenance_database(tmp_path: Path) -> Path:
    path = tmp_path / "index.sqlite3"
    database = IndexDB(path)
    database.initialize()
    asset_sha = "a" * 64
    terms_sha = "b" * 64
    robots_sha = "c" * 64
    orphan_sha = "d" * 64
    with database.connect() as connection:
        connection.executemany(
            """
            INSERT INTO sources(
                id, kind, name, root_url, adapter_version, config_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    "alpha",
                    "catalog",
                    "Créations Æther",
                    "https://example.test/alpha",
                    "1",
                    '{"default_license":"LicenseRef-SourceDefault","rate":2}',
                    "2026-01-01T00:00:00+00:00",
                ),
                (
                    "beta",
                    "repository",
                    "Beta sprites",
                    "https://example.test/beta",
                    "1",
                    "{}",
                    "2026-01-01T00:00:00+00:00",
                ),
            ),
        )
        connection.executemany(
            """
            INSERT INTO items(
                id, source_id, external_id, canonical_url, title, description,
                creator_name, creator_url, published_at, metadata_json,
                first_seen_at, last_seen_at, tombstoned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    "item_conflict",
                    "alpha",
                    "sprite-z",
                    "https://example.test/alpha/z",
                    "Renard animé",
                    "A running fox",
                    "Zoë Artist",
                    "https://example.test/zoe",
                    "2025-01-01",
                    '{"tags":["fox","run"]}',
                    "2026-01-02T00:00:00+00:00",
                    "2026-01-03T00:00:00+00:00",
                    None,
                ),
                (
                    "item_unknown_observed",
                    "alpha",
                    "sprite-a",
                    "https://example.test/alpha/a",
                    "Unknown license observation",
                    None,
                    None,
                    None,
                    None,
                    "{}",
                    "2026-01-02T00:00:00+00:00",
                    "2026-01-03T00:00:00+00:00",
                    None,
                ),
                (
                    "item_unobserved",
                    "alpha",
                    "sprite-m",
                    "https://example.test/alpha/m",
                    "No rights evidence",
                    None,
                    None,
                    None,
                    None,
                    "{}",
                    "2026-01-02T00:00:00+00:00",
                    "2026-01-03T00:00:00+00:00",
                    None,
                ),
                (
                    "item_single",
                    "beta",
                    "one",
                    "https://example.test/beta/one",
                    "Single license",
                    None,
                    "Beta Author",
                    None,
                    None,
                    "{}",
                    "2026-01-02T00:00:00+00:00",
                    "2026-01-03T00:00:00+00:00",
                    None,
                ),
            ),
        )
        connection.executemany(
            """
            INSERT INTO blobs(sha256, size_bytes, mime_type, storage_path, first_seen_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                (asset_sha, 100, "image/png", "objects/aa/asset", "2026-01-02"),
                (terms_sha, 50, "text/html", "objects/bb/terms", "2026-01-02"),
                (robots_sha, 20, "text/plain", "objects/cc/robots", "2026-01-02"),
                (orphan_sha, 999, None, "objects/dd/orphan", "2026-01-02"),
            ),
        )
        connection.execute(
            """
            INSERT INTO crawl_runs(
                id, source_id, started_at, completed_at, status, parameters_json,
                free_bytes_start, free_bytes_end, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_1",
                "alpha",
                "2026-01-02T00:00:00+00:00",
                "2026-01-02T00:01:00+00:00",
                "completed",
                "{}",
                200_000,
                199_000,
                None,
            ),
        )
        connection.executemany(
            """
            INSERT INTO retrievals(
                id, run_id, item_id, url, requested_at, completed_at, status_code,
                etag, last_modified, mime_type, content_length, blob_sha256, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    "retrieval_ok",
                    "run_1",
                    "item_conflict",
                    "https://cdn.example.test/fox.png",
                    "2026-01-02T00:00:01+00:00",
                    "2026-01-02T00:00:02+00:00",
                    200,
                    '"abc"',
                    None,
                    "image/png",
                    100,
                    asset_sha,
                    None,
                ),
                (
                    "retrieval_failed",
                    "run_1",
                    "item_unknown_observed",
                    "https://cdn.example.test/missing.png",
                    "2026-01-02T00:00:03+00:00",
                    "2026-01-02T00:00:04+00:00",
                    404,
                    None,
                    None,
                    "text/html",
                    None,
                    None,
                    "not found",
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO item_blobs(
                id, item_id, blob_sha256, role, original_url, original_filename,
                archive_member, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "item_blob_1",
                "item_conflict",
                asset_sha,
                "original",
                "https://cdn.example.test/fox.png",
                "fox.png",
                "sprites/fox.png",
                "2026-01-02T00:00:02+00:00",
            ),
        )
        connection.executemany(
            """
            INSERT INTO rights_observations(
                id, item_id, observed_at, license_raw, license_expression,
                license_url, attribution_raw, terms_url, terms_blob_sha256,
                robots_url, robots_blob_sha256, basis, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    "rights_cc_by",
                    "item_conflict",
                    "2026-01-02T00:10:00+00:00",
                    "Creative Commons Attribution 4.0",
                    "CC-BY-4.0",
                    "https://creativecommons.org/licenses/by/4.0/",
                    "Credit Zoë Artist",
                    "https://example.test/terms/old",
                    terms_sha,
                    "https://example.test/robots.txt",
                    robots_sha,
                    "asset page badge",
                    '{"selector":".license-old"}',
                ),
                (
                    "rights_restrictive",
                    "item_conflict",
                    "2026-02-02T00:10:00+00:00",
                    "All rights reserved",
                    "LicenseRef-Proprietary",
                    "https://example.test/terms/new",
                    "Publisher attribution required",
                    "https://example.test/terms/new",
                    terms_sha,
                    "https://example.test/robots.txt",
                    robots_sha,
                    "later terms capture",
                    '{"selector":".license-new"}',
                ),
                (
                    "rights_unknown",
                    "item_unknown_observed",
                    "2026-01-02T00:10:00+00:00",
                    None,
                    None,
                    None,
                    "Creator field was blank",
                    "https://example.test/terms/unknown",
                    terms_sha,
                    None,
                    None,
                    "page inspected; no license statement found",
                    '{"absence_recorded":true}',
                ),
                (
                    "rights_cc0",
                    "item_single",
                    "2026-01-02T00:10:00+00:00",
                    "CC0",
                    "CC0-1.0",
                    "https://creativecommons.org/publicdomain/zero/1.0/",
                    "Beta Author",
                    None,
                    None,
                    None,
                    None,
                    "repository file header",
                    "{}",
                ),
            ),
        )
        connection.execute(
            """
            INSERT INTO entities(
                id, source_id, external_identity_key, representative_item_id,
                display_name, entity_class, entity_subclass, species_or_type,
                taxonomy_version, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "entity_fox",
                "alpha",
                "fox",
                "item_conflict",
                "Fox",
                "animal",
                "mammal",
                "fox",
                "1",
                "{}",
                "2026-01-02",
                "2026-01-02",
            ),
        )
        connection.execute(
            """
            INSERT INTO sequences(
                id, item_id, source_blob_sha256, extraction_method,
                extraction_confidence, width, height, frame_count, loop_mode,
                action, direction, quality_tier, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "sequence_run",
                "item_conflict",
                asset_sha,
                "sheet_grid",
                1.0,
                16,
                16,
                1,
                "loop",
                "run",
                "right",
                "F0",
                "{}",
                "2026-01-02",
            ),
        )
        connection.execute(
            """
            INSERT INTO frames(
                sequence_id, ordinal, blob_sha256, duration_ms, bbox_json, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("sequence_run", 0, asset_sha, 100, "[0,0,16,16]", "{}"),
        )
        connection.execute(
            """
            INSERT INTO sequence_frames(
                sequence_id, ordinal, source_blob_sha256, source_frame_index,
                duration_ms, phase, direction, view, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("sequence_run", 0, asset_sha, 0, 100, 0.0, "right", "side", "{}"),
        )
        connection.execute(
            """
            INSERT INTO sequence_subjects(sequence_id, entity_id, role, metadata_json)
            VALUES (?, ?, ?, ?)
            """,
            ("sequence_run", "entity_fox", "primary", "{}"),
        )
        connection.execute(
            """
            INSERT INTO sequence_source_keys(
                source_id, external_sequence_key, sequence_id, created_at
            ) VALUES (?, ?, ?, ?)
            """,
            ("alpha", "fox/run/right", "sequence_run", "2026-01-02"),
        )
        connection.execute(
            """
            INSERT INTO archive_inventories(
                archive_blob_sha256, archive_format, member_count, file_count,
                total_uncompressed_bytes, total_compressed_bytes, policy_json,
                inventory_sha256, inspected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (asset_sha, "zip", 1, 1, 100, 80, "{}", "e" * 64, "2026-01-02"),
        )
        connection.execute(
            """
            INSERT INTO archive_members(
                archive_blob_sha256, ordinal, member_path, normalized_path,
                member_kind, size_bytes, compressed_bytes, crc32,
                compression_method, modified_at, extracted_blob_sha256,
                selected_role, inspection_status, error, metadata_json, observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_sha,
                0,
                "fox.png",
                "fox.png",
                "file",
                100,
                80,
                1,
                8,
                None,
                asset_sha,
                "sprite",
                "extracted",
                None,
                "{}",
                "2026-01-02",
            ),
        )
        connection.execute(
            """
            INSERT INTO sequence_occurrences(
                sequence_id, archive_blob_sha256, archive_member_ordinal,
                occurrence_role, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("sequence_run", asset_sha, 0, "source_sprite", "{}", "2026-01-02"),
        )
        connection.execute(
            """
            INSERT INTO motion_annotations(
                sequence_id, vocabulary_version, source_action, normalized_action,
                action_family, view, direction, loopable, cycle_frames,
                phase_zero_frame, confidence, annotation_method, conditioning_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "sequence_run",
                "1",
                "sprint",
                "run",
                "locomotion",
                "side",
                "right",
                1,
                1,
                0,
                1.0,
                "fixture",
                "{}",
                "2026-01-02",
                "2026-01-02",
            ),
        )
        connection.execute(
            """
            INSERT INTO dataset_snapshots(
                id, name, manifest_sha256, parameters_json, code_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("dataset_1", "fixture", "e" * 64, '{"quality":"F0"}', "test", "2026-01-03"),
        )
        connection.execute(
            """
            INSERT INTO dataset_members(
                snapshot_id, sequence_id, split, sample_weight, decision, reason
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("dataset_1", "sequence_run", "train", 1.0, "include", None),
        )
    return path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_exports_are_utf8_atomic_and_byte_stable(provenance_database: Path, tmp_path: Path) -> None:
    output_directory = tmp_path / "reports"
    paths = export_provenance_reports(provenance_database, output_directory)
    first_bytes = {path: path.read_bytes() for path in paths.__dict__.values()}

    export_provenance_reports(provenance_database, output_directory)

    assert {path: path.read_bytes() for path in paths.__dict__.values()} == first_bytes
    assert "Créations Æther".encode() in paths.sources_jsonl.read_bytes()
    assert "Renard animé".encode() in paths.inventory_jsonl.read_bytes()
    assert not list(output_directory.glob(".*.tmp"))

    with paths.sources_csv.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    jsonl_rows = _read_jsonl(paths.sources_jsonl)
    assert [row["id"] for row in csv_rows] == ["alpha", "beta"]
    assert [row["id"] for row in jsonl_rows] == ["alpha", "beta"]
    assert jsonl_rows[0]["config"]["default_license"] == "LicenseRef-SourceDefault"
    assert jsonl_rows[0]["rights_scope"].startswith("item_observations_only")
    assert {
        key: jsonl_rows[0][key]
        for key in (
            "entity_count",
            "sequence_count",
            "sequence_frame_count",
            "sequence_occurrence_count",
            "sequence_subject_count",
        )
    } == {
        "entity_count": 1,
        "sequence_count": 1,
        "sequence_frame_count": 1,
        "sequence_occurrence_count": 1,
        "sequence_subject_count": 1,
    }
    assert all(
        jsonl_rows[1][key] == 0
        for key in (
            "entity_count",
            "sequence_count",
            "sequence_frame_count",
            "sequence_occurrence_count",
            "sequence_subject_count",
        )
    )

    bundle = json.loads(paths.bundle_manifest_json.read_text(encoding="utf-8"))
    assert bundle["artifact_kind"] == "spritelab_provenance_report_bundle"
    assert bundle["read_consistency"] == "single_sqlite_query_only_read_transaction"
    assert bundle["report_schema_version"] == 3
    assert bundle["database_schema_versions"]
    bundled_paths = {row["relative_path"]: row for row in bundle["files"]}
    assert set(bundled_paths) == {
        "ATTRIBUTION.md",
        "corpus_summary.json",
        "inventory.jsonl",
        "sources.csv",
        "sources.jsonl",
    }
    for relative_path, row in bundled_paths.items():
        payload = (output_directory / relative_path).read_bytes()
        assert row["size_bytes"] == len(payload)
        assert row["sha256"] == hashlib.sha256(payload).hexdigest()


def test_inventory_preserves_every_rights_observation_without_inference(
    provenance_database: Path, tmp_path: Path
) -> None:
    paths = export_provenance_reports(provenance_database, tmp_path / "reports")
    records = _read_jsonl(paths.inventory_jsonl)
    items = {record["id"]: record for record in records if record["record_type"] == "item"}
    rights = [record for record in records if record["record_type"] == "rights_observation"]

    assert {record["id"] for record in rights} == {
        "rights_cc0",
        "rights_cc_by",
        "rights_restrictive",
        "rights_unknown",
    }
    conflict = items["item_conflict"]["rights_summary"]
    assert conflict["observation_state"] == "multiple"
    assert conflict["license_state"] == "multiple_observed_licenses"
    assert conflict["observed_license_labels"] == ["CC-BY-4.0", "LicenseRef-Proprietary"]
    assert conflict["resolution"] == "not_resolved_or_inferred"

    unknown_observed = items["item_unknown_observed"]["rights_summary"]
    assert unknown_observed["license_state"] == "unknown_observed"
    assert unknown_observed["unknown_license_observation_count"] == 1
    unobserved = items["item_unobserved"]["rights_summary"]
    assert unobserved["license_state"] == "unknown_no_observation"
    assert unobserved["observed_license_labels"] == []

    by_id = {record["id"]: record for record in rights}
    assert by_id["rights_cc_by"]["attribution_raw"] == "Credit Zoë Artist"
    assert by_id["rights_restrictive"]["attribution_raw"] == "Publisher attribution required"
    assert by_id["rights_unknown"]["license_known"] is False
    assert by_id["rights_cc_by"]["terms_snapshot"] == {
        "sha256": "b" * 64,
        "storage_path": "objects/bb/terms",
    }
    assert by_id["rights_cc_by"]["robots_snapshot"]["sha256"] == "c" * 64

    blobs = [record for record in records if record["record_type"] == "blob"]
    assert any(record["sha256"] == "d" * 64 for record in blobs)
    retrieval = next(record for record in records if record.get("id") == "retrieval_ok")
    assert retrieval["blob"]["sha256"] == "a" * 64
    assert retrieval["blob"]["storage_path"] == "objects/aa/asset"
    item_blob = next(record for record in records if record.get("id") == "item_blob_1")
    assert item_blob["blob"]["sha256"] == "a" * 64
    assert item_blob["provenance_links"]["original_url"].endswith("fox.png")


def test_attribution_and_summary_expose_conflicts_unknowns_and_corpus_counts(
    provenance_database: Path, tmp_path: Path
) -> None:
    paths = export_provenance_reports(provenance_database, tmp_path / "reports")
    attribution = paths.attribution_markdown.read_text(encoding="utf-8")
    summary = json.loads(paths.corpus_summary_json.read_text(encoding="utf-8"))

    assert "CC-BY-4.0" in attribution
    assert "LicenseRef-Proprietary" in attribution
    assert "Credit Zoë Artist" in attribution
    assert "Publisher attribution required" in attribution
    assert "multiple observations are preserved separately" in attribution
    assert "unknown (no item-level observation recorded)" in attribution
    assert "LicenseRef-SourceDefault" not in attribution

    assert summary["provenance_policy"]["source_license_inheritance"] is False
    assert summary["rights"]["item_license_states"] == {
        "multiple_observed_licenses": 1,
        "single_observed_license": 1,
        "unknown_no_observation": 1,
        "unknown_observed": 1,
    }
    assert summary["rights"]["item_observation_states"] == {
        "multiple": 1,
        "none": 1,
        "single": 2,
    }
    assert summary["rights"]["observed_license_labels"] == {
        "CC-BY-4.0": 1,
        "CC0-1.0": 1,
        "LicenseRef-Proprietary": 1,
    }
    assert summary["rights"]["unknown_license_observation_count"] == 1
    assert summary["corpus"]["counts"]["blobs"] == 4
    assert summary["corpus"]["counts"]["frames"] == 1
    assert summary["corpus"]["counts"]["sequence_frames"] == 1
    assert summary["corpus"]["counts"]["sequence_occurrences"] == 1
    assert summary["corpus"]["counts"]["sequence_source_keys"] == 1
    assert summary["corpus"]["counts"]["sequence_subjects"] == 1
    assert summary["corpus"]["total_blob_bytes"] == 1169
    assert summary["entity_classes"] == {"animal": 1}
    assert summary["motion"]["normalized_actions"] == {"run": 1}
    assert summary["motion"]["action_families"] == {"locomotion": 1}
    assert summary["datasets"][0]["split_counts"] == {"train": 1}
