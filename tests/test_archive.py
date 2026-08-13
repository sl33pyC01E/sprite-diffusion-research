from __future__ import annotations

import stat
import struct
import warnings
import zipfile
from pathlib import Path

import pytest

from spritelab.archive import (
    ArchiveIntegrityError,
    ArchiveLimitExceeded,
    ArchiveLimits,
    ArchiveSelectionError,
    UnsafeArchiveError,
    extract_zip_to_cas,
    inspect_zip,
)
from spritelab.storage import ContentAddressedStore, DiskFloorReached, DiskGuard


def _write_zip(
    path: Path,
    members: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _store(tmp_path: Path, *, floor: int = 0) -> ContentAddressedStore:
    return ContentAddressedStore(tmp_path / "data", DiskGuard(tmp_path, floor))


def test_inspect_zip_is_metadata_only_and_summarizes_extensions(tmp_path: Path) -> None:
    archive_path = tmp_path / "sprites.zip"
    _write_zip(
        archive_path,
        {
            "hero/idle.PNG": b"idle",
            "hero/run.gif": b"run",
            "README": b"credits",
            "empty/": b"",
        },
    )

    manifest = inspect_zip(archive_path)

    assert [member.normalized_name for member in manifest.members] == [
        "hero/idle.PNG",
        "hero/run.gif",
        "README",
        "empty",
    ]
    assert manifest.regular_file_count == 3
    assert manifest.directory_count == 1
    assert manifest.total_uncompressed_bytes == len(b"idleruncredits")
    assert manifest.extension_counts == {"": 1, ".gif": 1, ".png": 1}
    assert len(manifest.inventory_sha256) == 64
    assert manifest.inventory_sha256 == inspect_zip(archive_path).inventory_sha256
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../escape.png",
        "sprites/../../escape.png",
        "/absolute.png",
        "C:/drive.png",
        "C:drive-relative.png",
        r"\\server\share\unc.png",
    ],
)
def test_inspect_rejects_traversal_absolute_and_drive_paths(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    archive_path = tmp_path / "unsafe.zip"
    _write_zip(archive_path, {unsafe_name: b"sprite"})

    with pytest.raises(UnsafeArchiveError):
        inspect_zip(archive_path)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("hero/run.png", "hero/./run.png"),
        ("hero/run.png", "HERO/RUN.PNG"),
        (r"hero\run.png", "hero/run.png"),
    ],
)
def test_inspect_rejects_normalized_and_case_collisions(
    tmp_path: Path,
    first: str,
    second: str,
) -> None:
    archive_path = tmp_path / "collision.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(first, b"one")
            archive.writestr(second, b"two")

    with pytest.raises(UnsafeArchiveError):
        inspect_zip(archive_path)


def test_inspect_rejects_symlink_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("hero-link.png")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, b"hero.png")

    with pytest.raises(UnsafeArchiveError, match="Symlink or special"):
        inspect_zip(archive_path)


def test_opt_in_indexes_symlink_as_non_extractable_metadata(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("README.md")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README", b"real readme")
        archive.writestr(link, b"README")

    limits = ArchiveLimits(allow_symlink_metadata=True)
    manifest = inspect_zip(archive_path, limits=limits)

    assert manifest.regular_file_count == 1
    assert manifest.symlink_count == 1
    assert manifest.directory_count == 0
    symlink = manifest.members[1]
    assert symlink.is_symlink
    assert not symlink.is_regular_file
    assert symlink.extension == ""
    store = _store(tmp_path)
    extraction = extract_zip_to_cas(archive_path, store, limits=limits)
    assert [item.member.normalized_name for item in extraction.extracted] == ["README"]
    with pytest.raises(ArchiveSelectionError, match="non-regular"):
        extract_zip_to_cas(archive_path, store, limits=limits, select=["README.md"])


def test_inspect_rejects_encrypted_flag(tmp_path: Path) -> None:
    archive_path = tmp_path / "encrypted-flag.zip"
    _write_zip(archive_path, {"hero.png": b"sprite"})
    payload = bytearray(archive_path.read_bytes())
    local_offset = payload.index(b"PK\x03\x04")
    central_offset = payload.index(b"PK\x01\x02")
    local_flags = struct.unpack_from("<H", payload, local_offset + 6)[0]
    central_flags = struct.unpack_from("<H", payload, central_offset + 8)[0]
    struct.pack_into("<H", payload, local_offset + 6, local_flags | 1)
    struct.pack_into("<H", payload, central_offset + 8, central_flags | 1)
    archive_path.write_bytes(payload)

    with pytest.raises(UnsafeArchiveError, match="Encrypted"):
        inspect_zip(archive_path)


def test_inspect_enforces_member_count_size_total_and_ratio_limits(tmp_path: Path) -> None:
    count_path = tmp_path / "count.zip"
    _write_zip(count_path, {"one.png": b"1", "two.png": b"2"})
    with pytest.raises(ArchiveLimitExceeded, match="members"):
        inspect_zip(count_path, limits=ArchiveLimits(max_members=1))

    size_path = tmp_path / "size.zip"
    _write_zip(size_path, {"large.png": b"12345"})
    with pytest.raises(ArchiveLimitExceeded, match="per-member"):
        inspect_zip(size_path, limits=ArchiveLimits(max_member_bytes=4))

    total_path = tmp_path / "total.zip"
    _write_zip(total_path, {"one.png": b"123", "two.png": b"456"})
    with pytest.raises(ArchiveLimitExceeded, match="total limit"):
        inspect_zip(total_path, limits=ArchiveLimits(max_total_expanded_bytes=5))

    bomb_path = tmp_path / "bomb.zip"
    _write_zip(bomb_path, {"repeat.bin": b"A" * 20_000}, compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(ArchiveLimitExceeded, match="compression ratio"):
        inspect_zip(bomb_path, limits=ArchiveLimits(max_compression_ratio=2))


def test_extract_streams_only_selected_regular_files_to_cas(tmp_path: Path) -> None:
    archive_path = tmp_path / "sprites.zip"
    _write_zip(
        archive_path,
        {
            "hero/idle.png": b"idle pixels",
            "hero/run.gif": b"run pixels",
            "docs/LICENSE.txt": b"license",
            "empty/": b"",
        },
        compression=zipfile.ZIP_DEFLATED,
    )
    store = _store(tmp_path)

    result = extract_zip_to_cas(
        archive_path,
        store,
        select=lambda member: member.extension in {".png", ".gif"},
        chunk_bytes=3,
    )

    assert [item.member.normalized_name for item in result.extracted] == [
        "hero/idle.png",
        "hero/run.gif",
    ]
    assert [item.blob.path.read_bytes() for item in result.extracted] == [
        b"idle pixels",
        b"run pixels",
    ]
    assert all(item.blob.size_bytes == item.member.uncompressed_bytes for item in result.extracted)
    assert result.extracted[0].member.crc32 != 0
    object_files = [path for path in store.objects_root.rglob("*") if path.is_file()]
    assert len(object_files) == 2


def test_extract_accepts_exact_normalized_names_and_rejects_unknown_names(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "sprites.zip"
    _write_zip(archive_path, {"hero/idle.png": b"idle", "hero/run.png": b"run"})
    store = _store(tmp_path)

    result = extract_zip_to_cas(archive_path, store, select=["hero/run.png"])
    assert [item.member.normalized_name for item in result.extracted] == ["hero/run.png"]

    with pytest.raises(ArchiveSelectionError, match="did not match"):
        extract_zip_to_cas(archive_path, store, select=["missing.png"])


def test_extract_obeys_store_disk_floor_before_member_write(tmp_path: Path) -> None:
    archive_path = tmp_path / "sprite.zip"
    _write_zip(archive_path, {"hero.png": b"sprite"})
    free_bytes = DiskGuard(tmp_path, 0).status().free_bytes
    store = _store(tmp_path, floor=free_bytes)

    with pytest.raises(DiskFloorReached):
        extract_zip_to_cas(archive_path, store)


def test_extract_rejects_corrupt_crc_without_committing_blob(tmp_path: Path) -> None:
    archive_path = tmp_path / "corrupt.zip"
    original = b"unique sprite payload"
    _write_zip(archive_path, {"hero.png": original})
    payload = bytearray(archive_path.read_bytes())
    data_offset = payload.index(original)
    payload[data_offset] ^= 0xFF
    archive_path.write_bytes(payload)
    store = _store(tmp_path)

    with pytest.raises(ArchiveIntegrityError, match="integrity check failed"):
        extract_zip_to_cas(archive_path, store)

    object_files = [path for path in store.objects_root.rglob("*") if path.is_file()]
    assert object_files == []
