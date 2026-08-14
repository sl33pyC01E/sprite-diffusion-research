"""Resolve declarative M.U.G.E.N character media from an extracted collection.

Only DEF, AIR, and SFF files are interpreted. Runtime CMD/CNS/ST files are never
loaded or executed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from spritelab.adapters.mugen import (
    MugenAirAction,
    MugenCharacterDefinition,
    MugenSffHeader,
    inspect_sff_header,
    parse_air,
    parse_character_def,
)


@dataclass(frozen=True)
class MugenDirectoryDefinitionFailure:
    definition_path: str
    reason: str
    detail: str


@dataclass(frozen=True)
class MugenDirectoryVariant:
    definition_paths: tuple[str, ...]
    definitions: tuple[MugenCharacterDefinition, ...]
    air_path: str
    air_sha256: str
    sff_path: str
    sff_sha256: str
    sff_bytes: int
    sff_header: MugenSffHeader
    actions: tuple[MugenAirAction, ...]


@dataclass(frozen=True)
class MugenDirectoryAudit:
    root: str
    definition_count: int
    variants: tuple[MugenDirectoryVariant, ...]
    failures: tuple[MugenDirectoryDefinitionFailure, ...]


def audit_mugen_directory(root: Path) -> MugenDirectoryAudit:
    """Resolve every distinct DEF-selected AIR/SFF pair below ``root``."""

    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"MUGEN root is not a directory: {root}")
    files = tuple(sorted((path for path in root.rglob("*") if path.is_file()), key=_path_key))
    relative = {path: path.relative_to(root).as_posix() for path in files}
    folded: dict[str, list[Path]] = {}
    for path, name in relative.items():
        folded.setdefault(name.casefold(), []).append(path)
    definitions = tuple(path for path in files if path.suffix.casefold() == ".def")
    failures: list[MugenDirectoryDefinitionFailure] = []
    grouped: dict[tuple[str, str], list[tuple[Path, MugenCharacterDefinition, Path, Path]]] = {}
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
            air_path = _resolve_reference(root, definition_path, anim, folded)
            sff_path = _resolve_reference(root, definition_path, sprite, folded)
        except ValueError as error:
            failures.append(
                MugenDirectoryDefinitionFailure(
                    definition_name,
                    "unresolved_media_reference",
                    str(error),
                )
            )
            continue
        grouped.setdefault(
            (relative[air_path].casefold(), relative[sff_path].casefold()), []
        ).append((definition_path, definition, air_path, sff_path))

    variants: list[MugenDirectoryVariant] = []
    for key in sorted(grouped, key=lambda value: tuple(part.encode("utf-8") for part in value)):
        rows = sorted(grouped[key], key=lambda row: _path_key(row[0]))
        air_path, sff_path = rows[0][2], rows[0][3]
        try:
            actions = parse_air(air_path.read_bytes(), reject_duplicate_actions=False)
            sff_payload = sff_path.read_bytes()
            header = inspect_sff_header(sff_payload)
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
            MugenDirectoryVariant(
                definition_paths=tuple(relative[row[0]] for row in rows),
                definitions=tuple(row[1] for row in rows),
                air_path=relative[air_path],
                air_sha256=_file_sha256(air_path),
                sff_path=relative[sff_path],
                sff_sha256=header.sha256,
                sff_bytes=len(sff_payload),
                sff_header=header,
                actions=actions,
            )
        )
    return MugenDirectoryAudit(
        root=str(root),
        definition_count=len(definitions),
        variants=tuple(variants),
        failures=tuple(sorted(failures, key=lambda row: row.definition_path.encode("utf-8"))),
    )


def _resolve_reference(
    root: Path,
    definition_path: Path,
    reference: str,
    folded: dict[str, list[Path]],
) -> Path:
    normalized = reference.replace("\\", "/").strip()
    candidate = PurePosixPath(normalized)
    if candidate.is_absolute() or (candidate.parts and candidate.parts[0].endswith(":")):
        raise ValueError(f"absolute media reference is forbidden: {reference!r}")
    parts = list(PurePosixPath(definition_path.relative_to(root).parent.as_posix()).parts)
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
    resolved = matches[0].resolve(strict=True)
    if root not in resolved.parents:
        raise ValueError(f"media reference escapes collection root: {reference!r}")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_key(path: Path) -> bytes:
    return path.as_posix().encode("utf-8")
