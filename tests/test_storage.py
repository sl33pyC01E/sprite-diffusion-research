from pathlib import Path

import pytest

from spritelab.storage import ContentAddressedStore, DiskFloorReached, DiskGuard


def test_content_addressed_store_deduplicates(tmp_path: Path) -> None:
    guard = DiskGuard(tmp_path, 0)
    store = ContentAddressedStore(tmp_path, guard)

    first = store.put_bytes(b"sprite")
    second = store.put_bytes(b"sprite")

    assert first.sha256 == second.sha256
    assert first.path.read_bytes() == b"sprite"
    assert first.existed is False
    assert second.existed is True


def test_disk_guard_rejects_impossible_floor(tmp_path: Path) -> None:
    free = DiskGuard(tmp_path, 0).status().free_bytes
    guard = DiskGuard(tmp_path, free + 1)
    with pytest.raises(DiskFloorReached):
        guard.require_capacity(label="test")


def test_same_volume_file_ingest_needs_only_one_payload_of_headroom(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"sprite")
    free = DiskGuard(tmp_path, 0).status().free_bytes
    store = ContentAddressedStore(tmp_path / "data", DiskGuard(tmp_path, free - 1024 * 1024))

    blob = store.put_file(payload, source_is_on_store_volume=True)

    assert blob.path.read_bytes() == b"sprite"
