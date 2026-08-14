"""Join extracted M.U.G.E.N DEF/AIR metadata to an archive SFF inventory.

This module never extracts or executes archive members. It supports the metadata-first
workflow used for very large RAR collections: small DEF/AIR files live in a verified
staging tree while SFF payloads remain in the immutable source container.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from spritelab.adapters.mugen import (
    MugenAirAction,
    MugenAirParseExclusion,
    MugenCharacterDefinition,
    parse_air,
    parse_character_def,
)
from spritelab.mugen_directory import MugenDirectoryDefinitionFailure, _resolve_reference


@dataclass(frozen=True, slots=True)
class MugenArchiveMember:
    path: str
    size_bytes: int
    crc32: str | None


@dataclass(frozen=True, slots=True)
class MugenArchiveMetadataVariant:
    definition_paths: tuple[str, ...]
    definitions: tuple[MugenCharacterDefinition, ...]
    air_path: str
    air_sha256: str
    sff_member: MugenArchiveMember
    actions: tuple[MugenAirAction, ...]
    air_parse_exclusions: tuple[MugenAirParseExclusion, ...]


@dataclass(frozen=True, slots=True)
class MugenArchiveMetadataAudit:
    root: str
    definition_count: int
    sff_inventory_count: int
    variants: tuple[MugenArchiveMetadataVariant, ...]
    failures: tuple[MugenDirectoryDefinitionFailure, ...]


def parse_7z_slt_members(text: str) -> tuple[MugenArchiveMember, ...]:
    """Parse member path/size/CRC facts from 7-Zip ``l -slt -ba`` output."""

    rows: list[MugenArchiveMember] = []
    current: dict[str, object] = {}

    def finish() -> None:
        if "path" not in current or "size_bytes" not in current:
            return
        rows.append(
            MugenArchiveMember(
                path=_normalize_member_path(str(current["path"])),
                size_bytes=int(current["size_bytes"]),
                crc32=(str(current["crc32"]).casefold() if current.get("crc32") else None),
            )
        )

    for line in text.splitlines():
        if line.startswith("Path = "):
            finish()
            current = {"path": line[7:]}
        elif line.startswith("Size = "):
            current["size_bytes"] = int(line[7:])
        elif line.startswith("CRC = "):
            value = line[6:].strip()
            current["crc32"] = value or None
    finish()
    return tuple(rows)


def audit_mugen_archive_metadata_directory(
    root: Path,
    sff_inventory: tuple[MugenArchiveMember, ...],
) -> MugenArchiveMetadataAudit:
    """Resolve every DEF-selected AIR/SFF pair without reading SFF payloads."""

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"MUGEN root is not a directory: {root}")
    files = tuple(sorted((path for path in root.rglob("*") if path.is_file()), key=_path_key))
    relative = {path: path.relative_to(root).as_posix() for path in files}
    local_folded: dict[str, list[Path]] = {}
    for path, name in relative.items():
        local_folded.setdefault(name.casefold(), []).append(path)
    archive_folded: dict[str, list[MugenArchiveMember]] = {}
    for member in sff_inventory:
        archive_folded.setdefault(member.path.casefold(), []).append(member)

    definitions = tuple(path for path in files if path.suffix.casefold() == ".def")
    failures: list[MugenDirectoryDefinitionFailure] = []
    grouped: dict[
        tuple[str, str],
        list[tuple[Path, MugenCharacterDefinition, Path, MugenArchiveMember]],
    ] = {}
    for definition_path in definitions:
        definition_name = relative[definition_path]
        try:
            definition = parse_character_def(definition_path.read_bytes())
        except (OSError, ValueError) as error:
            failures.append(
                MugenDirectoryDefinitionFailure(
                    definition_name,
                    "invalid_definition",
                    f"{type(error).__name__}: {error}",
                )
            )
            continue
        anim = definition.file("anim")
        sprite = definition.file("sprite")
        if not anim or not sprite:
            failures.append(
                MugenDirectoryDefinitionFailure(
                    definition_name,
                    "missing_media_reference",
                    "DEF does not declare both [Files] anim and sprite",
                )
            )
            continue
        try:
            air_path = _resolve_reference(root, definition_path, anim, local_folded)
            sff_member = _resolve_archive_reference(
                definition_name,
                sprite,
                archive_folded,
            )
        except ValueError as error:
            failures.append(
                MugenDirectoryDefinitionFailure(
                    definition_name,
                    "unresolved_media_reference",
                    str(error),
                )
            )
            continue
        grouped.setdefault((relative[air_path].casefold(), sff_member.path.casefold()), []).append(
            (definition_path, definition, air_path, sff_member)
        )

    variants: list[MugenArchiveMetadataVariant] = []
    for key in sorted(grouped, key=lambda value: tuple(part.encode("utf-8") for part in value)):
        rows = sorted(grouped[key], key=lambda row: _path_key(row[0]))
        air_path, sff_member = rows[0][2], rows[0][3]
        exclusions: list[MugenAirParseExclusion] = []
        try:
            actions = parse_air(
                air_path.read_bytes(),
                reject_duplicate_actions=False,
                recover_invalid_elements=True,
                exclusions=exclusions,
            )
        except (OSError, ValueError) as error:
            for definition_path, _, _, _ in rows:
                failures.append(
                    MugenDirectoryDefinitionFailure(
                        relative[definition_path],
                        "invalid_media",
                        f"{type(error).__name__}: {error}",
                    )
                )
            continue
        variants.append(
            MugenArchiveMetadataVariant(
                definition_paths=tuple(relative[row[0]] for row in rows),
                definitions=tuple(row[1] for row in rows),
                air_path=relative[air_path],
                air_sha256=_file_sha256(air_path),
                sff_member=sff_member,
                actions=actions,
                air_parse_exclusions=tuple(exclusions),
            )
        )
    return MugenArchiveMetadataAudit(
        root=str(root),
        definition_count=len(definitions),
        sff_inventory_count=len(sff_inventory),
        variants=tuple(variants),
        failures=tuple(sorted(failures, key=lambda row: row.definition_path.encode("utf-8"))),
    )


def _resolve_archive_reference(
    definition_path: str,
    reference: str,
    folded: dict[str, list[MugenArchiveMember]],
) -> MugenArchiveMember:
    normalized = reference.replace("\\", "/").strip()
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or (candidate.parts and candidate.parts[0].endswith(":")):
        raise ValueError(f"absolute media reference is forbidden: {reference!r}")
    parts = list(PurePosixPath(definition_path).parent.parts)
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"media reference escapes collection root: {reference!r}")
            parts.pop()
        else:
            parts.append(part)
    name = PurePosixPath(*parts).as_posix()
    matches = folded.get(name.casefold(), [])
    if len(matches) != 1:
        raise ValueError(f"media reference {reference!r} resolved to {len(matches)} files")
    return matches[0]


def _normalize_member_path(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\\", "/"))
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or (candidate.parts and candidate.parts[0].endswith(":")):
        raise ValueError(f"absolute archive member is forbidden: {value!r}")
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"archive member traversal is forbidden: {value!r}")
    return candidate.as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_key(path: Path) -> bytes:
    return path.as_posix().encode("utf-8")
