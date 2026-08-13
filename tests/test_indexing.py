import zipfile
from pathlib import Path

from spritelab.archive import ArchiveLimits, extract_zip_to_cas, inspect_zip
from spritelab.db import IndexDB
from spritelab.indexing import index_zip_extraction, index_zip_manifest
from spritelab.storage import ContentAddressedStore, DiskGuard


def test_index_zip_manifest_and_selected_extraction(tmp_path: Path) -> None:
    archive_path = tmp_path / "sprites.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("hero/idle.png", b"pixels")
        archive.writestr("LICENSE.txt", b"license")
    database = IndexDB(tmp_path / "index.sqlite3")
    database.initialize()
    store = ContentAddressedStore(tmp_path / "data", DiskGuard(tmp_path, 0))
    archive_blob = store.put_file(archive_path)
    database.register_blob(
        sha256=archive_blob.sha256,
        size_bytes=archive_blob.size_bytes,
        storage_path=archive_blob.path,
    )
    limits = ArchiveLimits()
    manifest = inspect_zip(archive_blob.path, limits=limits)

    assert (
        index_zip_manifest(
            database,
            archive_blob_sha256=archive_blob.sha256,
            manifest=manifest,
            limits=limits,
        )
        == 2
    )
    extraction = extract_zip_to_cas(
        archive_blob.path,
        store,
        limits=limits,
        select=["hero/idle.png"],
    )
    assert (
        index_zip_extraction(
            database,
            archive_blob_sha256=archive_blob.sha256,
            extraction=extraction,
            selected_role="sprite_candidate",
        )
        == 1
    )

    with database.connect() as connection:
        row = connection.execute(
            """
            SELECT normalized_path, extracted_blob_sha256, selected_role
            FROM archive_members WHERE normalized_path='hero/idle.png'
            """
        ).fetchone()
    assert tuple(row) == (
        "hero/idle.png",
        extraction.extracted[0].blob.sha256,
        "sprite_candidate",
    )
