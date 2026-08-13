from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from spritelab.db import IndexDB
from spritelab.lpc_snapshot import export_lpc_manifest


def _fixture(tmp_path: Path) -> tuple[IndexDB, Path, str]:
    root = "Universal-LPC-Spritesheet-Character-Generator-deadbeef"
    archive_path = tmp_path / "lpc.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{root}/CREDITS.csv",
            "filename,notes,authors,licenses,urls\n"
            "body/bodies/male/run.png,source,Artist,CC0,https://example.test\n",
        )
        archive.writestr(
            f"{root}/sheet_definitions/body/body.json",
            json.dumps(
                {
                    "name": "Body",
                    "type_name": "body",
                    "layer_1": {"male": "body/bodies/male/"},
                }
            ),
        )
        archive.writestr(f"{root}/spritesheets/body/bodies/male/run.png", b"fixture")
    archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    with database.connect() as connection:
        connection.execute(
            """
            INSERT INTO blobs(sha256, size_bytes, storage_path, first_seen_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (archive_sha, archive_path.stat().st_size, str(archive_path)),
        )
    paths = (
        f"{root}/CREDITS.csv",
        f"{root}/sheet_definitions/body/body.json",
        f"{root}/spritesheets/body/bodies/male/run.png",
    )
    database.upsert_archive_members(
        archive_blob_sha256=archive_sha,
        members=[
            {
                "ordinal": ordinal,
                "member_path": path,
                "normalized_path": path,
                "member_kind": "file",
                "size_bytes": 1,
                "compressed_bytes": 1,
            }
            for ordinal, path in enumerate(paths)
        ],
    )
    media_sha = "b" * 64
    database.register_archive_extractions(
        archive_blob_sha256=archive_sha,
        extracted=[
            {
                "ordinal": 2,
                "sha256": media_sha,
                "size_bytes": 7,
                "storage_path": str(tmp_path / "fake-media"),
            }
        ],
        selected_role="sprite",
    )
    database.record_media_observation(
        blob_sha256=media_sha,
        inspector_version="fixture-v1",
        media_format="PNG",
        width=512,
        height=256,
        mode="RGBA",
        has_alpha=True,
        is_animated=False,
        frame_count=1,
        pixel_sha256="c" * 64,
    )
    database.mark_archive_member_inspection(
        archive_blob_sha256=archive_sha,
        ordinal=2,
        status="media_inspected",
    )
    return database, archive_path, archive_sha


def test_lpc_export_is_deterministic_and_retains_slice_and_credit_evidence(
    tmp_path: Path,
) -> None:
    database, archive_path, archive_sha = _fixture(tmp_path)
    first_path = tmp_path / "first.jsonl.gz"
    second_path = tmp_path / "second.jsonl.gz"

    first = export_lpc_manifest(
        database_path=database.path,
        archive_path=archive_path,
        archive_blob_sha256=archive_sha,
        output_path=first_path,
    )
    second = export_lpc_manifest(
        database_path=database.path,
        archive_path=archive_path,
        archive_blob_sha256=archive_sha,
        output_path=second_path,
    )

    assert first.compressed_sha256 == second.compressed_sha256
    assert first.canonical_jsonl_sha256 == second.canonical_jsonl_sha256
    assert first.record_count == 1
    assert first.slice_count == 4
    assert first.cell_count == 32
    assert first.credit_row_count == 1
    assert first.definition_count == 1
    assert first.geometry_counts == {"canonical": 1}
    assert first.credit_match_counts == {"credits_csv_exact_filename": 1}
    with gzip.open(first_path, "rt", encoding="utf-8") as stream:
        record = json.loads(stream.readline())
        assert stream.readline() == ""
    assert record["record_kind"] == "modular_compositing_layer_sheet"
    assert record["is_complete_entity"] is False
    assert record["slices"][0]["cells"][0] == {
        "column_index": 0,
        "frame_index": 0,
        "height": 64,
        "row_index": 0,
        "source_grid_index": 0,
        "width": 64,
        "x": 0,
        "y": 0,
    }
    assert record["credit"]["license_tokens"] == ["CC0"]


def test_lpc_export_preserves_existing_output_and_verifies_archive(tmp_path: Path) -> None:
    database, archive_path, archive_sha = _fixture(tmp_path)
    output = tmp_path / "manifest.jsonl.gz"
    output.write_bytes(b"preserve")

    with pytest.raises(FileExistsError):
        export_lpc_manifest(
            database_path=database.path,
            archive_path=archive_path,
            archive_blob_sha256=archive_sha,
            output_path=output,
        )
    assert output.read_bytes() == b"preserve"

    with pytest.raises(ValueError, match="do not match"):
        export_lpc_manifest(
            database_path=database.path,
            archive_path=archive_path,
            archive_blob_sha256="a" * 64,
            output_path=tmp_path / "other.jsonl.gz",
        )
