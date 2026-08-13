from __future__ import annotations

import binascii
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from spritelab.storage import ContentAddressedStore, StoredBlob


class ArchiveError(RuntimeError):
    """Base class for archive inspection and extraction failures."""


class UnsafeArchiveError(ArchiveError):
    """Raised when an archive contains an unsafe or ambiguous member."""


class ArchiveLimitExceeded(ArchiveError):
    """Raised when declared archive expansion exceeds a configured limit."""


class ArchiveIntegrityError(ArchiveError):
    """Raised when ZIP structure, size, or CRC validation fails."""


class ArchiveSelectionError(ArchiveError):
    """Raised for a named selection that does not identify regular files."""


class UnsupportedArchiveError(ArchiveError):
    """Raised when the Python runtime cannot decompress a selected member."""


@dataclass(frozen=True)
class ArchiveLimits:
    """Limits applied to ZIP metadata before any member is decompressed."""

    max_members: int = 250_000
    max_member_bytes: int = 4 * 1024**3
    max_total_expanded_bytes: int = 64 * 1024**3
    max_compression_ratio: float = 1_000.0
    allow_symlink_metadata: bool = False

    def __post_init__(self) -> None:
        integer_limits = {
            "max_members": self.max_members,
            "max_member_bytes": self.max_member_bytes,
            "max_total_expanded_bytes": self.max_total_expanded_bytes,
        }
        for name, value in integer_limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            isinstance(self.max_compression_ratio, bool)
            or not isinstance(self.max_compression_ratio, (int, float))
            or not math.isfinite(self.max_compression_ratio)
            or self.max_compression_ratio <= 0
        ):
            raise ValueError("max_compression_ratio must be finite and positive")
        if not isinstance(self.allow_symlink_metadata, bool):
            raise ValueError("allow_symlink_metadata must be a boolean")


@dataclass(frozen=True)
class ZipMember:
    """A validated, filesystem-independent ZIP central-directory entry."""

    archive_index: int
    original_name: str
    normalized_name: str
    is_directory: bool
    is_symlink: bool
    compressed_bytes: int
    uncompressed_bytes: int
    crc32: int
    compression_method: int
    flag_bits: int
    modified_at: tuple[int, int, int, int, int, int]
    create_system: int
    create_version: int
    extract_version: int
    external_attributes: int
    internal_attributes: int
    unix_mode: int | None
    comment: bytes
    extra: bytes
    header_offset: int

    @property
    def extension(self) -> str:
        if not self.is_regular_file:
            return ""
        return PurePosixPath(self.normalized_name).suffix.casefold()

    @property
    def is_regular_file(self) -> bool:
        return not self.is_directory and not self.is_symlink

    @property
    def compression_ratio(self) -> float:
        return _compression_ratio(self.uncompressed_bytes, self.compressed_bytes)


@dataclass(frozen=True)
class ZipManifest:
    """Metadata-only result of validating a ZIP archive."""

    members: tuple[ZipMember, ...]
    total_compressed_bytes: int
    total_uncompressed_bytes: int
    extension_counts: Mapping[str, int]

    @property
    def regular_file_count(self) -> int:
        return sum(member.is_regular_file for member in self.members)

    @property
    def directory_count(self) -> int:
        return sum(member.is_directory for member in self.members)

    @property
    def symlink_count(self) -> int:
        return sum(member.is_symlink for member in self.members)

    @property
    def compression_ratio(self) -> float:
        return _compression_ratio(self.total_uncompressed_bytes, self.total_compressed_bytes)

    @property
    def inventory_sha256(self) -> str:
        """Stable digest of central-directory facts, independent of object location."""
        payload = [
            {
                "ordinal": member.archive_index,
                "path": member.normalized_name,
                "directory": member.is_directory,
                "symlink": member.is_symlink,
                "compressed_bytes": member.compressed_bytes,
                "uncompressed_bytes": member.uncompressed_bytes,
                "crc32": member.crc32,
                "compression_method": member.compression_method,
                "modified_at": member.modified_at,
            }
            for member in self.members
        ]
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ExtractedZipMember:
    """A ZIP member and its immutable content-addressed object."""

    member: ZipMember
    blob: StoredBlob


@dataclass(frozen=True)
class ZipExtraction:
    """Validated manifest plus the members selected for CAS ingestion."""

    manifest: ZipManifest
    extracted: tuple[ExtractedZipMember, ...]


type ZipSource = str | os.PathLike[str] | BinaryIO
type MemberSelector = Callable[[ZipMember], bool] | Iterable[str] | None

_DRIVE_PATH = re.compile(r"^[A-Za-z]:")
_ENCRYPTION_FLAGS = 0x0001 | 0x0040


def inspect_zip(
    source: ZipSource,
    *,
    limits: ArchiveLimits | None = None,
) -> ZipManifest:
    """Validate and list a ZIP without decompressing or writing its members."""

    active_limits = limits or ArchiveLimits()
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            return _inspect_open_zip(archive, active_limits)
    except ArchiveError:
        raise
    except zipfile.BadZipFile as exc:
        raise ArchiveIntegrityError(f"Invalid ZIP archive: {exc}") from exc


def extract_zip_to_cas(
    source: ZipSource,
    store: ContentAddressedStore,
    *,
    limits: ArchiveLimits | None = None,
    select: MemberSelector = None,
    chunk_bytes: int = 1024 * 1024,
) -> ZipExtraction:
    """Stream selected regular ZIP members directly into an immutable CAS.

    ``select`` may be a predicate or an iterable of exact normalized member names.
    The archive is fully inspected before the first selected member is read.
    """

    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be a positive integer")

    active_limits = limits or ArchiveLimits()
    try:
        with zipfile.ZipFile(source, mode="r") as archive:
            manifest = _inspect_open_zip(archive, active_limits)
            selected = _select_regular_members(manifest, select)
            infos = archive.infolist()
            extracted: list[ExtractedZipMember] = []

            for member in selected:
                info = infos[member.archive_index]
                if info.filename != member.original_name:
                    raise ArchiveIntegrityError(
                        f"ZIP member metadata changed during inspection: {member.normalized_name!r}"
                    )
                store.guard.require_capacity(
                    member.uncompressed_bytes,
                    label=f"ZIP member staging {member.normalized_name}",
                )
                store.initialize()
                fd, temp_name = tempfile.mkstemp(
                    prefix="zip-member-", suffix=".part", dir=store.temp_root
                )
                temp_path = os.fspath(temp_name)
                try:
                    with os.fdopen(fd, "wb") as output, archive.open(info, mode="r") as handle:
                        observed_size, observed_crc = _copy_verified_member(
                            handle,
                            output,
                            member,
                            chunk_bytes,
                            store,
                        )
                        output.flush()
                        os.fsync(output.fileno())
                    if observed_size != member.uncompressed_bytes:
                        raise ArchiveIntegrityError(
                            f"ZIP member {member.normalized_name!r} yielded "
                            f"{observed_size} bytes; expected {member.uncompressed_bytes}"
                        )
                    if observed_crc & 0xFFFFFFFF != member.crc32:
                        raise ArchiveIntegrityError(
                            f"CRC mismatch for ZIP member {member.normalized_name!r}"
                        )
                    blob = store.commit_partial(Path(temp_path))
                except BaseException:
                    Path(temp_path).unlink(missing_ok=True)
                    raise
                if blob.size_bytes != member.uncompressed_bytes:
                    raise ArchiveIntegrityError(
                        f"Size mismatch for {member.normalized_name!r}: "
                        f"expected {member.uncompressed_bytes}, received {blob.size_bytes}"
                    )
                extracted.append(ExtractedZipMember(member=member, blob=blob))

            return ZipExtraction(manifest=manifest, extracted=tuple(extracted))
    except ArchiveError:
        raise
    except zipfile.BadZipFile as exc:
        raise ArchiveIntegrityError(f"ZIP integrity check failed: {exc}") from exc
    except NotImplementedError as exc:
        raise UnsupportedArchiveError(f"Unsupported ZIP compression method: {exc}") from exc


def _inspect_open_zip(archive: zipfile.ZipFile, limits: ArchiveLimits) -> ZipManifest:
    infos = archive.infolist()
    if len(infos) > limits.max_members:
        raise ArchiveLimitExceeded(f"ZIP has {len(infos)} members; limit is {limits.max_members}")

    members: list[ZipMember] = []
    seen_exact: dict[str, str] = {}
    seen_casefolded: dict[str, str] = {}
    extensions: Counter[str] = Counter()
    total_compressed = 0
    total_uncompressed = 0

    for archive_index, info in enumerate(infos):
        normalized_name, has_directory_suffix = _normalize_member_name(info.filename)
        prior_exact = seen_exact.get(normalized_name)
        if prior_exact is not None:
            raise UnsafeArchiveError(
                f"Duplicate normalized ZIP member {normalized_name!r}: "
                f"{prior_exact!r} and {info.filename!r}"
            )
        folded = normalized_name.casefold()
        prior_case = seen_casefolded.get(folded)
        if prior_case is not None:
            raise UnsafeArchiveError(
                f"Case-colliding ZIP members: {prior_case!r} and {normalized_name!r}"
            )
        seen_exact[normalized_name] = info.filename
        seen_casefolded[folded] = normalized_name

        if info.flag_bits & _ENCRYPTION_FLAGS:
            raise UnsafeArchiveError(f"Encrypted ZIP member is not accepted: {normalized_name!r}")
        if info.file_size < 0 or info.compress_size < 0:
            raise ArchiveIntegrityError(f"Negative size for ZIP member {normalized_name!r}")

        is_directory, is_symlink, unix_mode = _classify_member(
            info,
            normalized_name,
            has_directory_suffix,
            allow_symlink_metadata=limits.allow_symlink_metadata,
        )
        if is_directory and info.file_size != 0:
            raise UnsafeArchiveError(f"Directory ZIP member carries file data: {normalized_name!r}")
        if info.file_size > limits.max_member_bytes:
            raise ArchiveLimitExceeded(
                f"ZIP member {normalized_name!r} expands to {info.file_size} bytes; "
                f"per-member limit is {limits.max_member_bytes}"
            )

        member_ratio = _compression_ratio(info.file_size, info.compress_size)
        if member_ratio > limits.max_compression_ratio:
            raise ArchiveLimitExceeded(
                f"ZIP member {normalized_name!r} has compression ratio "
                f"{member_ratio:.2f}; limit is {limits.max_compression_ratio:g}"
            )

        total_compressed += info.compress_size
        total_uncompressed += info.file_size
        if total_uncompressed > limits.max_total_expanded_bytes:
            raise ArchiveLimitExceeded(
                f"ZIP expands to at least {total_uncompressed} bytes; "
                f"total limit is {limits.max_total_expanded_bytes}"
            )

        total_ratio = _compression_ratio(total_uncompressed, total_compressed)
        if total_ratio > limits.max_compression_ratio:
            raise ArchiveLimitExceeded(
                f"ZIP aggregate compression ratio is {total_ratio:.2f}; "
                f"limit is {limits.max_compression_ratio:g}"
            )

        member = ZipMember(
            archive_index=archive_index,
            original_name=info.filename,
            normalized_name=normalized_name,
            is_directory=is_directory,
            is_symlink=is_symlink,
            compressed_bytes=info.compress_size,
            uncompressed_bytes=info.file_size,
            crc32=info.CRC,
            compression_method=info.compress_type,
            flag_bits=info.flag_bits,
            modified_at=info.date_time,
            create_system=info.create_system,
            create_version=info.create_version,
            extract_version=info.extract_version,
            external_attributes=info.external_attr,
            internal_attributes=info.internal_attr,
            unix_mode=unix_mode,
            comment=bytes(info.comment),
            extra=bytes(info.extra),
            header_offset=info.header_offset,
        )
        members.append(member)
        if not is_directory and not is_symlink:
            extensions[member.extension] += 1

    return ZipManifest(
        members=tuple(members),
        total_compressed_bytes=total_compressed,
        total_uncompressed_bytes=total_uncompressed,
        extension_counts=dict(sorted(extensions.items())),
    )


def _normalize_member_name(raw_name: str) -> tuple[str, bool]:
    if not isinstance(raw_name, str) or not raw_name:
        raise UnsafeArchiveError("ZIP member name must be non-empty text")
    if "\x00" in raw_name:
        raise UnsafeArchiveError("ZIP member name contains a NUL byte")

    portable_name = unicodedata.normalize("NFC", raw_name.replace("\\", "/"))
    if portable_name.startswith("/") or _DRIVE_PATH.match(portable_name):
        raise UnsafeArchiveError(f"Absolute or drive-qualified ZIP member: {raw_name!r}")

    has_directory_suffix = portable_name.endswith("/")
    parts: list[str] = []
    for part in portable_name.split("/"):
        if part in {"", "."}:
            continue
        if part == "..":
            raise UnsafeArchiveError(f"Traversal segment in ZIP member: {raw_name!r}")
        parts.append(part)

    if not parts:
        raise UnsafeArchiveError(f"ZIP member has no usable normalized name: {raw_name!r}")
    return "/".join(parts), has_directory_suffix


def _classify_member(
    info: zipfile.ZipInfo,
    normalized_name: str,
    has_directory_suffix: bool,
    *,
    allow_symlink_metadata: bool,
) -> tuple[bool, bool, int | None]:
    unix_mode: int | None = None
    unix_type = 0
    if info.create_system == 3:
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        unix_type = stat.S_IFMT(unix_mode)
        if unix_type == stat.S_IFLNK and allow_symlink_metadata:
            if has_directory_suffix:
                raise UnsafeArchiveError(
                    f"Symlink mode conflicts with directory name: {normalized_name!r}"
                )
            return False, True, unix_mode
        if unix_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise UnsafeArchiveError(
                f"Symlink or special ZIP member is not accepted: {normalized_name!r}"
            )

    if unix_type == stat.S_IFDIR and not has_directory_suffix:
        raise UnsafeArchiveError(
            f"Directory mode conflicts with ZIP member name: {normalized_name!r}"
        )
    if unix_type == stat.S_IFREG and has_directory_suffix:
        raise UnsafeArchiveError(
            f"Regular-file mode conflicts with directory name: {normalized_name!r}"
        )

    return has_directory_suffix, False, unix_mode


def _select_regular_members(
    manifest: ZipManifest,
    select: MemberSelector,
) -> tuple[ZipMember, ...]:
    regular = tuple(member for member in manifest.members if member.is_regular_file)
    if select is None:
        return regular
    if callable(select):
        return tuple(member for member in regular if select(member))

    requested_values = (select,) if isinstance(select, str) else tuple(select)

    requested: set[str] = set()
    for raw_name in requested_values:
        if not isinstance(raw_name, str):
            raise TypeError("named ZIP selections must contain only strings")
        normalized, is_directory_name = _normalize_member_name(raw_name)
        if is_directory_name:
            raise ArchiveSelectionError(
                f"Named ZIP selection is a directory, not a regular file: {raw_name!r}"
            )
        requested.add(normalized)

    by_name = {member.normalized_name: member for member in manifest.members}
    unknown = requested.difference(by_name)
    if unknown:
        rendered = ", ".join(repr(name) for name in sorted(unknown))
        raise ArchiveSelectionError(f"ZIP selection did not match member(s): {rendered}")
    nonregular = sorted(name for name in requested if not by_name[name].is_regular_file)
    if nonregular:
        rendered = ", ".join(repr(name) for name in nonregular)
        raise ArchiveSelectionError(f"ZIP selection names non-regular member(s): {rendered}")
    return tuple(member for member in regular if member.normalized_name in requested)


def _copy_verified_member(
    handle: BinaryIO,
    output: BinaryIO,
    member: ZipMember,
    chunk_bytes: int,
    store: ContentAddressedStore,
) -> tuple[int, int]:
    observed_size = 0
    observed_crc = 0
    while block := handle.read(chunk_bytes):
        store.guard.require_capacity(len(block), label=f"ZIP member chunk {member.normalized_name}")
        observed_size += len(block)
        if observed_size > member.uncompressed_bytes:
            raise ArchiveIntegrityError(
                f"ZIP member {member.normalized_name!r} exceeded its declared size"
            )
        observed_crc = binascii.crc32(block, observed_crc)
        output.write(block)
    return observed_size, observed_crc


def _compression_ratio(uncompressed_bytes: int, compressed_bytes: int) -> float:
    if uncompressed_bytes == 0:
        return 0.0
    if compressed_bytes == 0:
        return math.inf
    return uncompressed_bytes / compressed_bytes
