from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


class DiskFloorReached(RuntimeError):
    """Raised before a write would cross the configured free-space floor."""


class HashMismatch(RuntimeError):
    """Raised when a completed payload does not match its declared digest."""


@dataclass(frozen=True)
class DiskStatus:
    volume_path: Path
    total_bytes: int
    used_bytes: int
    free_bytes: int
    floor_bytes: int

    @property
    def writable_budget_bytes(self) -> int:
        return max(0, self.free_bytes - self.floor_bytes)


class DiskGuard:
    def __init__(self, volume_path: Path, min_free_bytes: int) -> None:
        self.volume_path = volume_path.resolve()
        self.min_free_bytes = min_free_bytes

    def status(self) -> DiskStatus:
        usage = shutil.disk_usage(self.volume_path)
        return DiskStatus(
            volume_path=self.volume_path,
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
            floor_bytes=self.min_free_bytes,
        )

    def require_capacity(self, additional_bytes: int = 0, *, label: str = "write") -> None:
        if additional_bytes < 0:
            raise ValueError("additional_bytes must be non-negative")
        status = self.status()
        remaining = status.free_bytes - additional_bytes
        if remaining < self.min_free_bytes:
            raise DiskFloorReached(
                f"Refusing {label}: {additional_bytes} bytes would leave "
                f"{remaining} bytes free, below floor {self.min_free_bytes}"
            )


@dataclass(frozen=True)
class StoredBlob:
    sha256: str
    size_bytes: int
    path: Path
    existed: bool


class ContentAddressedStore:
    """Immutable SHA-256 object store with guarded, atomic writes."""

    def __init__(self, data_root: Path, guard: DiskGuard) -> None:
        self.data_root = data_root.resolve()
        self.guard = guard
        self.objects_root = self.data_root / "raw" / "objects" / "sha256"
        self.temp_root = self.data_root / "raw" / ".partial"

    def initialize(self) -> None:
        self.guard.require_capacity(label="store initialization")
        self.objects_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def object_path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"Invalid SHA-256 digest: {sha256!r}")
        return self.objects_root / sha256[:2] / sha256[2:4] / sha256

    def partial_path(self, key: str) -> Path:
        """Return a stable partial path for resumable external acquisition."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.temp_root / f"http-{digest}.part"

    def commit_partial(
        self,
        partial_path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> StoredBlob:
        """Hash and atomically promote a completed in-store partial file."""
        partial_path = partial_path.resolve()
        if partial_path.parent != self.temp_root.resolve():
            raise ValueError(f"Partial file is outside store temp directory: {partial_path}")
        size = partial_path.stat().st_size
        digest = hashlib.sha256()
        with partial_path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        hexdigest = digest.hexdigest()
        if expected_sha256 is not None and hexdigest != expected_sha256.lower():
            raise HashMismatch(f"Expected SHA-256 {expected_sha256.lower()}, received {hexdigest}")
        destination = self.object_path(hexdigest)
        if destination.exists():
            partial_path.unlink(missing_ok=True)
            return StoredBlob(hexdigest, size, destination, True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial_path.replace(destination)
        return StoredBlob(hexdigest, size, destination, False)

    def put_bytes(self, payload: bytes) -> StoredBlob:
        digest = hashlib.sha256(payload).hexdigest()
        destination = self.object_path(digest)
        if destination.exists():
            return StoredBlob(digest, len(payload), destination, True)
        self.guard.require_capacity(len(payload), label="blob write")
        self.initialize()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="blob-", suffix=".part", dir=self.temp_root)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if destination.exists():
                temp_path.unlink(missing_ok=True)
                return StoredBlob(digest, len(payload), destination, True)
            temp_path.replace(destination)
            return StoredBlob(digest, len(payload), destination, False)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    def put_stream(
        self,
        chunks: Iterable[bytes],
        *,
        expected_bytes: int | None = None,
    ) -> StoredBlob:
        if expected_bytes is not None:
            self.guard.require_capacity(expected_bytes, label="streamed blob write")
        self.initialize()
        digest = hashlib.sha256()
        size = 0
        fd, temp_name = tempfile.mkstemp(prefix="blob-", suffix=".part", dir=self.temp_root)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                for chunk in chunks:
                    if not chunk:
                        continue
                    self.guard.require_capacity(len(chunk), label="streamed blob chunk")
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            hexdigest = digest.hexdigest()
            destination = self.object_path(hexdigest)
            if destination.exists():
                temp_path.unlink(missing_ok=True)
                return StoredBlob(hexdigest, size, destination, True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temp_path.replace(destination)
            return StoredBlob(hexdigest, size, destination, False)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise

    def put_file(
        self,
        source: Path,
        *,
        chunk_bytes: int = 1024 * 1024,
        source_is_on_store_volume: bool = False,
    ) -> StoredBlob:
        size = source.stat().st_size

        def chunks(handle: BinaryIO) -> Iterable[bytes]:
            while block := handle.read(chunk_bytes):
                yield block

        if not source_is_on_store_volume:
            with source.open("rb") as handle:
                return self.put_stream(chunks(handle), expected_bytes=size)

        digest = hashlib.sha256()
        with source.open("rb") as handle:
            while block := handle.read(chunk_bytes):
                digest.update(block)
        hexdigest = digest.hexdigest()
        destination = self.object_path(hexdigest)
        if destination.exists():
            return StoredBlob(hexdigest, size, destination, True)
        self.guard.require_capacity(size, label="same-volume blob copy")
        self.initialize()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="blob-", suffix=".part", dir=self.temp_root)
        temp_path = Path(temp_name)
        try:
            os.close(fd)
            shutil.copyfile(source, temp_path)
            with temp_path.open("rb+") as handle:
                os.fsync(handle.fileno())
            if destination.exists():
                temp_path.unlink(missing_ok=True)
                return StoredBlob(hexdigest, size, destination, True)
            temp_path.replace(destination)
            return StoredBlob(hexdigest, size, destination, False)
        except BaseException:
            temp_path.unlink(missing_ok=True)
            raise
