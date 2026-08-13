"""Read-only, exact-snapshot audit for Widelands worker and critter sprites.

The adapter parses the literal Lua tables used by Widelands' worker/critter
descriptions.  It does not execute Lua.  Only a deliberately small expression
subset is resolved: literal tables/scalars, references to earlier literal
tables, and animation directories built from ``dirname`` plus string literals.

Runtime animation semantics are taken from engine files in the same commit:
directional declarations expand in the engine's six-direction order, packed
spritesheets are read row-major, ``fps`` is converted with integer division,
and animations loop unless ``play_once`` is true.  Every accepted frame points
back to an exact ZIP member and byte digest.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image

EXPECTED_WIDELANDS_ARCHIVE_SHA256 = (
    "51093e130b2f4098716bc442ce3d8566d94b4590cfa233ba1c74e5d173a48186"
)
WIDELANDS_COMMIT = "fbe33f1b96e877ebe7352c29ad3bd06770bd5e0a"
WIDELANDS_REPOSITORY_URL = "https://github.com/widelands/widelands"
WIDELANDS_COMMIT_URL = f"{WIDELANDS_REPOSITORY_URL}/tree/{WIDELANDS_COMMIT}"
WIDELANDS_ARCHIVE_URL = f"https://codeload.github.com/widelands/widelands/zip/{WIDELANDS_COMMIT}"
_EXPECTED_ROOT = f"widelands-{WIDELANDS_COMMIT}"

_WORKER_MANIFEST_RE = re.compile(r"^data/tribes/workers/[^/]+/[^/]+/init\.lua$")
_CRITTER_MANIFEST_RE = re.compile(r"^data/world/critters/[^/]+/init\.lua$")
_PNG_RE = re.compile(
    r"^data/(?:tribes/workers/[^/]+/[^/]+|world/critters/[^/]+)/.*\.png$",
    re.IGNORECASE,
)
_CONSTRUCTORS = frozenset(
    {
        "new_worker_type",
        "new_carrier_type",
        "new_soldier_type",
        "new_ferry_type",
        "new_critter_type",
    }
)
_DIRECTIONS = ("ne", "e", "se", "sw", "w", "nw")
_DIRECTION_LABELS: Mapping[str, str] = {
    "ne": "northeast",
    "e": "east",
    "se": "southeast",
    "sw": "southwest",
    "w": "west",
    "nw": "northwest",
}
_SCALES: tuple[tuple[float, str], ...] = (
    (0.5, "_0.5"),
    (1.0, "_1"),
    (2.0, "_2"),
    (4.0, "_4"),
)
_ENGINE_EVIDENCE_PATHS = (
    "src/logic/map_objects/map_object.cc",
    "src/graphic/animation/animation.cc",
    "src/graphic/animation/nonpacked_animation.cc",
    "src/graphic/animation/spritesheet_animation.cc",
    "src/io/filesystem/filesystem.cc",
)
_RIGHTS_EVIDENCE_PATHS = (
    "COPYING",
    "CREDITS",
    "data/txts/LICENSE.lua",
    "data/txts/developers.json",
)
_ANIMAL_WORKER_NAMES = frozenset(
    {
        "amazons_tapir",
        "atlanteans_horse",
        "barbarians_ox",
        "empire_donkey",
        "frisians_reindeer",
    }
)
_EXPLICIT_WORK_ACTION_NAMES = frozenset(
    {
        "beeswarm",
        "collecting",
        "fetch_water",
        "freeing",
        "gather",
        "gathering",
        "harvesting",
        "planting",
        "planting_harvesting",
        "release",
        "releasein",
        "releaseout",
        "sawing",
        "stacking_1",
        "stacking_2",
        "water",
    }
)


class WidelandsArchiveError(ValueError):
    """Raised when an archive is unsafe or is not an auditable Widelands tree."""


class WidelandsParseError(ValueError):
    """Raised when a relevant literal Lua declaration is structurally invalid."""


@dataclass(frozen=True, slots=True)
class SourceLocation:
    manifest_path: str
    member_path: str
    line_number: int


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SourceImage:
    scale: float
    ordinal: int
    logical_path: str
    member_path: str
    sha256: str
    width: int
    height: int
    image_format: str
    playercolor_mask_path: str | None
    playercolor_mask_sha256: str | None


@dataclass(frozen=True, slots=True)
class FrameRecord:
    ordinal: int
    source_logical_path: str
    source_member_path: str
    source_sha256: str
    source_width: int
    source_height: int
    x: int
    y: int
    width: int
    height: int
    duration_milliseconds: int


@dataclass(frozen=True, slots=True)
class AnimationRecord:
    declared_name: str
    normalized_action: str | None
    normalized_action_basis: str
    variant_hint: str | None
    representation: Literal["spritesheet", "numbered_files"]
    direction: str | None
    direction_basis: str
    basename: str
    source_directory: str
    location: SourceLocation
    fps: int | None
    frame_duration_milliseconds: int
    frame_duration_basis: str
    declared_frame_count: int | None
    rows: int | None
    columns: int | None
    hotspot: tuple[int, int]
    play_once: bool
    loop_mode: Literal["loop", "one_shot"]
    source_images: tuple[SourceImage, ...]
    frames: tuple[FrameRecord, ...]
    available_scales: tuple[float, ...]
    exact_source_sequence: bool
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EntityRecord:
    entity_id: str
    tribe: str | None
    constructor_role: str
    entity_class: str
    entity_class_basis: str
    manifest_path: str
    member_path: str
    location: SourceLocation
    animation_directory: str
    animations: tuple[AnimationRecord, ...]
    complete_entity: bool
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuxiliaryImage:
    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    image_format: str
    role: Literal[
        "playercolor_mask",
        "ui_icon",
        "equipment_or_status_icon",
        "unreferenced_layer_or_effect",
    ]


@dataclass(frozen=True, slots=True)
class AuditIssue:
    code: str
    count: int
    message: str


@dataclass(frozen=True, slots=True)
class WidelandsArchiveCounts:
    archive_members: int
    archive_files: int
    archive_directories: int
    worker_manifests: int
    critter_manifests: int
    constructor_role_counts: tuple[tuple[str, int], ...]
    entities: int
    complete_entities: int
    animation_declarations: int
    direction_tracks: int
    exact_tracks: int
    quarantined_tracks: int
    primary_frames: int
    primary_animation_images: int
    playercolor_mask_images: int
    ui_icon_images: int
    equipment_or_status_images: int
    unreferenced_layer_or_effect_images: int
    worker_tree_pngs: int
    critter_tree_pngs: int
    action_counts: tuple[tuple[str, int], ...]
    entity_class_counts: tuple[tuple[str, int], ...]
    direction_counts: tuple[tuple[str, int], ...]
    representation_counts: tuple[tuple[str, int], ...]
    scale_counts: tuple[tuple[str, int], ...]
    source_image_format_counts: tuple[tuple[str, int], ...]
    surplus_spritesheet_cells: int
    duplicate_primary_image_hash_groups: int
    duplicate_primary_image_hash_excess: int


@dataclass(frozen=True, slots=True)
class RightsAudit:
    license_expression: str
    license_basis: str
    root_license: EvidenceDocument
    in_game_license: EvidenceDocument
    credits: EvidenceDocument
    developers: EvidenceDocument
    caveat: str


@dataclass(frozen=True, slots=True)
class WidelandsArchiveAudit:
    archive_sha256: str
    archive_size_bytes: int
    repository_url: str
    commit: str
    commit_url: str
    archive_url: str
    archive_root: str
    counts: WidelandsArchiveCounts
    entities: tuple[EntityRecord, ...]
    auxiliary_images: tuple[AuxiliaryImage, ...]
    rights: RightsAudit
    engine_evidence: tuple[EvidenceDocument, ...]
    issues: tuple[AuditIssue, ...]
    projection_policy: tuple[str, ...]
    audit_record_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class _ArchiveMember:
    logical_path: str
    member_path: str
    info: ZipInfo


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str
    value: str | int | float | bool | None
    line: int


@dataclass(frozen=True, slots=True)
class _LuaEntry:
    key: str | int | None
    value: _LuaValue
    line: int


@dataclass(frozen=True, slots=True)
class _LuaTable:
    entries: tuple[_LuaEntry, ...]
    line: int


@dataclass(frozen=True, slots=True)
class _LuaScalar:
    value: str | int | float | bool | None
    line: int


@dataclass(frozen=True, slots=True)
class _LuaIdentifier:
    name: str
    line: int


@dataclass(frozen=True, slots=True)
class _LuaExpression:
    tokens: tuple[_Token, ...]
    line: int


_LuaValue = _LuaTable | _LuaScalar | _LuaIdentifier | _LuaExpression


@dataclass(frozen=True, slots=True)
class _ParsedObject:
    constructor_role: str
    table: _LuaTable
    literal_tables: Mapping[str, _LuaTable]
    line: int


@dataclass(frozen=True, slots=True)
class _ImageInfo:
    sha256: str
    size_bytes: int
    width: int
    height: int
    image_format: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise WidelandsArchiveError(f"unsafe archive member path: {name!r}")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise WidelandsArchiveError(f"unsafe archive member path: {name!r}")
    return pure.as_posix()


def _validate_members(infos: Sequence[ZipInfo]) -> tuple[str, tuple[_ArchiveMember, ...]]:
    if not infos:
        raise WidelandsArchiveError("archive is empty")
    seen: set[str] = set()
    seen_casefolded: dict[str, str] = {}
    roots: set[str] = set()
    prepared: list[tuple[str, ZipInfo]] = []
    for info in infos:
        name = _normalize_member_name(info.filename)
        if name in seen:
            raise WidelandsArchiveError(f"duplicate archive member: {name}")
        folded = name.casefold()
        prior = seen_casefolded.get(folded)
        if prior is not None:
            raise WidelandsArchiveError(f"case-colliding archive members: {prior!r}, {name!r}")
        seen.add(name)
        seen_casefolded[folded] = name
        roots.add(PurePosixPath(name).parts[0])
        if info.flag_bits & 0x1:
            raise WidelandsArchiveError(f"encrypted archive member: {name}")
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise WidelandsArchiveError(f"non-regular archive member: {name}")
        prepared.append((name, info))
    if len(roots) != 1:
        raise WidelandsArchiveError(f"expected one archive root, found {sorted(roots)!r}")
    root = next(iter(roots))
    prefix = root + "/"
    files: list[_ArchiveMember] = []
    for name, info in prepared:
        if info.is_dir() or name == root:
            continue
        if not name.startswith(prefix):
            raise WidelandsArchiveError(f"member escaped archive root: {name}")
        files.append(_ArchiveMember(name[len(prefix) :], name, info))
    files.sort(key=lambda member: member.logical_path.encode("utf-8"))
    return root, tuple(files)


def _long_bracket_end(text: str, start: int) -> tuple[int, str] | None:
    match = re.match(r"\[(=*)\[", text[start:])
    if not match:
        return None
    equals = match.group(1)
    close = "]" + equals + "]"
    body_start = start + len(match.group(0))
    close_at = text.find(close, body_start)
    if close_at < 0:
        raise WidelandsParseError("unterminated Lua long-bracket literal")
    return close_at + len(close), text[body_start:close_at]


def _decode_quoted(text: str, start: int) -> tuple[int, str]:
    quote = text[start]
    chars: list[str] = []
    i = start + 1
    escape_map = {
        "a": "\a",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
        "v": "\v",
        "\\": "\\",
        '"': '"',
        "'": "'",
    }
    while i < len(text):
        char = text[i]
        if char == quote:
            return i + 1, "".join(chars)
        if char != "\\":
            chars.append(char)
            i += 1
            continue
        i += 1
        if i >= len(text):
            break
        escaped = text[i]
        if escaped == "z":
            i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            continue
        if escaped.isdigit():
            match = re.match(r"\d{1,3}", text[i:])
            assert match is not None
            chars.append(chr(int(match.group(0))))
            i += len(match.group(0))
            continue
        chars.append(escape_map.get(escaped, escaped))
        i += 1
    raise WidelandsParseError("unterminated Lua quoted string")


def _tokenize_lua(text: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    i = 0
    line = 1
    while i < len(text):
        char = text[i]
        if char.isspace():
            line += char == "\n"
            i += 1
            continue
        if text.startswith("--", i):
            long_comment = _long_bracket_end(text, i + 2)
            if long_comment is not None:
                end, _value = long_comment
                line += text.count("\n", i, end)
                i = end
            else:
                end = text.find("\n", i + 2)
                i = len(text) if end < 0 else end
            continue
        if char in {'"', "'"}:
            start_line = line
            end, value = _decode_quoted(text, i)
            tokens.append(_Token("string", text[i:end], value, start_line))
            line += text.count("\n", i, end)
            i = end
            continue
        long_string = _long_bracket_end(text, i) if char == "[" else None
        if long_string is not None:
            start_line = line
            end, value = long_string
            tokens.append(_Token("string", text[i:end], value, start_line))
            line += text.count("\n", i, end)
            i = end
            continue
        identifier = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[i:])
        if identifier:
            raw = identifier.group(0)
            value: str | bool | None = raw
            kind = "identifier"
            if raw == "true":
                kind, value = "scalar", True
            elif raw == "false":
                kind, value = "scalar", False
            elif raw == "nil":
                kind, value = "scalar", None
            tokens.append(_Token(kind, raw, value, line))
            i += len(raw)
            continue
        number = re.match(r"(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?", text[i:])
        if number:
            raw = number.group(0)
            value = float(raw) if any(marker in raw for marker in ".eE") else int(raw)
            tokens.append(_Token("number", raw, value, line))
            i += len(raw)
            continue
        operator = next(
            (
                candidate
                for candidate in ("...", "..", "==", "~=", "<=", ">=", "//")
                if text.startswith(candidate, i)
            ),
            None,
        )
        if operator:
            tokens.append(_Token("symbol", operator, operator, line))
            i += len(operator)
            continue
        tokens.append(_Token("symbol", char, char, line))
        i += 1
    return tuple(tokens)


def _scalar_or_expression(tokens: Sequence[_Token]) -> _LuaValue:
    if not tokens:
        raise WidelandsParseError("empty Lua value expression")
    first = tokens[0]
    if len(tokens) == 1:
        if first.kind in {"string", "number", "scalar"}:
            return _LuaScalar(first.value, first.line)
        if first.kind == "identifier":
            return _LuaIdentifier(str(first.value), first.line)
    if len(tokens) == 2 and tokens[0].text == "-" and tokens[1].kind == "number":
        number = tokens[1].value
        assert isinstance(number, (int, float)) and not isinstance(number, bool)
        return _LuaScalar(-number, tokens[0].line)
    return _LuaExpression(tuple(tokens), first.line)


def _parse_table(tokens: Sequence[_Token], start: int) -> tuple[_LuaTable, int]:
    if start >= len(tokens) or tokens[start].text != "{":
        raise WidelandsParseError("expected Lua table literal")
    entries: list[_LuaEntry] = []
    i = start + 1
    array_index = 1
    while i < len(tokens):
        while i < len(tokens) and tokens[i].text in {",", ";"}:
            i += 1
        if i >= len(tokens):
            break
        if tokens[i].text == "}":
            return _LuaTable(tuple(entries), tokens[start].line), i + 1

        key: str | int | None = None
        line = tokens[i].line
        if i + 1 < len(tokens) and tokens[i].kind == "identifier" and tokens[i + 1].text == "=":
            key = str(tokens[i].value)
            i += 2
        elif tokens[i].text == "[":
            close = i + 1
            while close < len(tokens) and tokens[close].text != "]":
                close += 1
            if close + 1 < len(tokens) and tokens[close + 1].text == "=":
                key_value = _scalar_or_expression(tokens[i + 1 : close])
                if not isinstance(key_value, _LuaScalar) or not isinstance(
                    key_value.value, (str, int)
                ):
                    raise WidelandsParseError(f"unsupported Lua table key at line {line}")
                key = key_value.value
                i = close + 2

        value_start = i
        if i < len(tokens) and tokens[i].text == "{":
            value, i = _parse_table(tokens, i)
        else:
            paren = bracket = brace = 0
            while i < len(tokens):
                token = tokens[i].text
                if token == "(":
                    paren += 1
                elif token == ")":
                    paren -= 1
                elif token == "[":
                    bracket += 1
                elif token == "]":
                    bracket -= 1
                elif token == "{":
                    brace += 1
                elif token == "}":
                    if paren == bracket == brace == 0:
                        break
                    brace -= 1
                elif token in {",", ";"} and paren == bracket == brace == 0:
                    break
                i += 1
            value = _scalar_or_expression(tokens[value_start:i])
        if key is None:
            key = array_index
            array_index += 1
        entries.append(_LuaEntry(key, value, line))
        if i < len(tokens) and tokens[i].text in {",", ";"}:
            i += 1
    raise WidelandsParseError(f"unterminated Lua table starting at line {tokens[start].line}")


def _parse_manifest(text: str) -> tuple[_ParsedObject, ...]:
    tokens = _tokenize_lua(text)
    literal_tables: dict[str, _LuaTable] = {}
    brace_depth = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.text == "{":
            brace_depth += 1
        elif token.text == "}":
            brace_depth -= 1
        if (
            brace_depth == 0
            and token.kind == "identifier"
            and i + 2 < len(tokens)
            and tokens[i + 1].text == "="
            and tokens[i + 2].text == "{"
        ):
            table, end = _parse_table(tokens, i + 2)
            literal_tables[str(token.value)] = table
            i = end
            continue
        i += 1

    objects: list[_ParsedObject] = []
    for i, token in enumerate(tokens):
        if token.kind != "identifier" or token.value not in _CONSTRUCTORS:
            continue
        if i + 1 >= len(tokens) or tokens[i + 1].text != "{":
            continue
        table, _end = _parse_table(tokens, i + 1)
        objects.append(
            _ParsedObject(
                constructor_role=str(token.value).removeprefix("new_").removesuffix("_type"),
                table=table,
                literal_tables=literal_tables,
                line=token.line,
            )
        )
    return tuple(objects)


def _mapping(table: _LuaTable) -> dict[str, _LuaValue]:
    result: dict[str, _LuaValue] = {}
    for entry in table.entries:
        if not isinstance(entry.key, str):
            continue
        if entry.key in result:
            raise WidelandsParseError(f"duplicate Lua table key {entry.key!r} at line {entry.line}")
        result[entry.key] = entry.value
    return result


def _require_table(value: _LuaValue, *, label: str) -> _LuaTable:
    if not isinstance(value, _LuaTable):
        raise WidelandsParseError(f"{label} must be a literal table")
    return value


def _resolve_table(
    value: _LuaValue, literal_tables: Mapping[str, _LuaTable], *, label: str
) -> _LuaTable:
    if isinstance(value, _LuaTable):
        return value
    if isinstance(value, _LuaIdentifier) and value.name in literal_tables:
        return literal_tables[value.name]
    raise WidelandsParseError(f"{label} is not a resolvable literal table")


def _require_string(value: _LuaValue, *, label: str) -> str:
    if isinstance(value, _LuaScalar) and isinstance(value.value, str):
        return value.value
    raise WidelandsParseError(f"{label} must be a literal string")


def _optional_string(value: _LuaValue | None, *, label: str) -> str | None:
    return None if value is None else _require_string(value, label=label)


def _require_int(value: _LuaValue, *, label: str, positive: bool = False) -> int:
    if (
        not isinstance(value, _LuaScalar)
        or isinstance(value.value, bool)
        or not isinstance(value.value, int)
    ):
        raise WidelandsParseError(f"{label} must be a literal integer")
    if positive and value.value <= 0:
        raise WidelandsParseError(f"{label} must be positive")
    return value.value


def _optional_int(
    value: _LuaValue | None, *, label: str, default: int | None = None, positive: bool = False
) -> int | None:
    return default if value is None else _require_int(value, label=label, positive=positive)


def _optional_bool(value: _LuaValue | None, *, label: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, _LuaScalar) and isinstance(value.value, bool):
        return value.value
    raise WidelandsParseError(f"{label} must be a literal boolean")


def _int_pair(value: _LuaValue, *, label: str) -> tuple[int, int]:
    table = _require_table(value, label=label)
    values = [
        entry.value.value
        for entry in table.entries
        if isinstance(entry.key, int)
        and isinstance(entry.value, _LuaScalar)
        and isinstance(entry.value.value, int)
        and not isinstance(entry.value.value, bool)
    ]
    if len(values) != 2:
        raise WidelandsParseError(f"{label} must contain exactly two literal integers")
    return values[0], values[1]


def _expression_text(value: _LuaExpression) -> str:
    return " ".join(token.text for token in value.tokens)


def _resolve_directory(value: _LuaValue | None, manifest_directory: str, *, label: str) -> str:
    if value is None:
        return ""
    if isinstance(value, _LuaIdentifier) and value.name == "dirname":
        return manifest_directory
    if isinstance(value, _LuaScalar) and isinstance(value.value, str):
        path = value.value.replace("\\", "/").strip("/")
        return path
    if isinstance(value, _LuaExpression):
        tokens = value.tokens
        # Literal pinned manifests use only: dirname .. "subdirectory[/]".
        if tokens and tokens[0].kind == "identifier" and tokens[0].value == "dirname":
            suffixes: list[str] = []
            cursor = 1
            while cursor < len(tokens):
                if cursor + 1 >= len(tokens) or tokens[cursor].text != "..":
                    break
                part = tokens[cursor + 1]
                if part.kind != "string" or not isinstance(part.value, str):
                    break
                suffixes.append(part.value)
                cursor += 2
            if cursor == len(tokens):
                suffix = "".join(suffixes).replace("\\", "/")
                return f"{manifest_directory.rstrip('/')}/{suffix.lstrip('/')}".rstrip("/")
        if _expression_text(value) == "path . dirname ( __file__ )":
            return manifest_directory
    raise WidelandsParseError(f"{label} is not a supported literal dirname expression")


def _normalize_action(name: str) -> tuple[str | None, str, str | None]:
    lowered = name.casefold()
    tokens = tuple(part for part in re.split(r"[^a-z0-9]+", lowered) if part)
    joined = "_".join(tokens)
    if joined in _EXPLICIT_WORK_ACTION_NAMES:
        return "work", "explicit_widelands_worker_action_name", None
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("death", ("die", "death")),
        ("dodge", ("eva", "evade", "dodge")),
        ("attack", ("atk", "attack", "fight")),
        ("carry", ("walkload", "carry", "loaded")),
        ("walk", ("walk",)),
        ("idle", ("idle",)),
        ("eat", ("eat", "eating")),
        ("sleep", ("sleep", "sleeping")),
        (
            "work",
            (
                "work",
                "working",
                "hack",
                "hacking",
                "dig",
                "digging",
                "fish",
                "fishing",
                "harvest",
                "plant",
                "sow",
                "reap",
                "mine",
                "cut",
                "build",
            ),
        ),
        ("hurt", ("hurt", "wounded")),
        ("emote", ("dance", "celebrate", "cheer", "greet")),
    )
    matched: str | None = None
    marker: str | None = None
    for action, markers in rules:
        for candidate in markers:
            if (
                candidate in tokens
                or joined.startswith(candidate + "_")
                or joined.endswith("_" + candidate)
            ):
                matched, marker = action, candidate
                break
        if matched is not None:
            break
    if matched is None:
        return None, "unmapped_literal_animation_name", None
    marker_index = joined.find(marker or "")
    prefix = joined[:marker_index].strip("_") if marker_index >= 0 else ""
    variant = prefix or None
    return matched, f"literal_animation_name_marker:{marker}", variant


def _entity_class(entity_id: str, role: str) -> tuple[str, str]:
    if role == "critter":
        return "animal", "new_critter_type_constructor"
    if role == "ferry":
        return "vehicle", "new_ferry_type_constructor"
    if entity_id in _ANIMAL_WORKER_NAMES:
        return "animal", "explicit_known_pack_animal_worker_id"
    return "humanoid", f"new_{role}_type_worker_family_constructor"


def _image_info(archive: ZipFile, member: _ArchiveMember) -> _ImageInfo:
    payload = archive.read(member.info)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
            image_format = (image.format or "UNKNOWN").upper()
    except Exception as error:
        raise WidelandsArchiveError(
            f"invalid image member {member.logical_path!r}: {error}"
        ) from error
    return _ImageInfo(_sha256_bytes(payload), len(payload), width, height, image_format)


def _join_logical(directory: str, filename: str) -> str:
    joined = PurePosixPath(directory) / filename if directory else PurePosixPath(filename)
    normalized = joined.as_posix()
    if joined.is_absolute() or ".." in joined.parts:
        raise WidelandsParseError(f"unsafe resolved animation path: {normalized!r}")
    return normalized


def _mask_path(logical_path: str) -> str:
    path = PurePosixPath(logical_path)
    return (path.parent / f"{path.stem}_pc{path.suffix}").as_posix()


def _sequential_paths(
    directory: str, basename: str, member_paths: frozenset[str]
) -> tuple[str, ...]:
    for digits in (0, 1, 2, 3):
        if digits == 0:
            candidate = _join_logical(directory, f"{basename}.png")
            if candidate in member_paths:
                return (candidate,)
            continue
        paths: list[str] = []
        for ordinal in range(10**digits):
            candidate = _join_logical(directory, f"{basename}_{ordinal:0{digits}d}.png")
            if candidate not in member_paths:
                break
            paths.append(candidate)
        if paths:
            return tuple(paths)
    return ()


def _source_image(
    *,
    archive: ZipFile,
    by_path: Mapping[str, _ArchiveMember],
    image_cache: dict[str, _ImageInfo],
    logical_path: str,
    scale: float,
    ordinal: int,
) -> SourceImage:
    member = by_path[logical_path]
    info = image_cache.get(logical_path)
    if info is None:
        info = _image_info(archive, member)
        image_cache[logical_path] = info
    candidate_mask = _mask_path(logical_path)
    mask_sha: str | None = None
    if candidate_mask in by_path:
        mask_info = image_cache.get(candidate_mask)
        if mask_info is None:
            mask_info = _image_info(archive, by_path[candidate_mask])
            image_cache[candidate_mask] = mask_info
        if (mask_info.width, mask_info.height) != (info.width, info.height):
            raise WidelandsArchiveError(
                f"player-color mask dimensions differ for {logical_path!r}: "
                f"{mask_info.width}x{mask_info.height} vs {info.width}x{info.height}"
            )
        mask_sha = mask_info.sha256
    return SourceImage(
        scale=scale,
        ordinal=ordinal,
        logical_path=logical_path,
        member_path=member.member_path,
        sha256=info.sha256,
        width=info.width,
        height=info.height,
        image_format=info.image_format,
        playercolor_mask_path=candidate_mask if mask_sha is not None else None,
        playercolor_mask_sha256=mask_sha,
    )


def _animation_records(
    *,
    archive: ZipFile,
    by_path: Mapping[str, _ArchiveMember],
    image_cache: dict[str, _ImageInfo],
    manifest_path: str,
    member_path: str,
    manifest_directory: str,
    animation_directory: str,
    representation: Literal["spritesheet", "numbered_files"],
    declarations: _LuaTable,
) -> tuple[AnimationRecord, ...]:
    records: list[AnimationRecord] = []
    member_paths = frozenset(by_path)
    for entry in declarations.entries:
        if not isinstance(entry.key, str):
            continue
        declaration = _require_table(entry.value, label=f"{manifest_path}:{entry.key}")
        fields = _mapping(declaration)
        basename = (
            _optional_string(fields.get("basename"), label=f"{entry.key}.basename") or entry.key
        )
        directory = (
            _resolve_directory(
                fields.get("directory"), manifest_directory, label=f"{entry.key}.directory"
            )
            if "directory" in fields
            else animation_directory
        )
        fps = _optional_int(fields.get("fps"), label=f"{entry.key}.fps", positive=True)
        frame_duration = 250 if fps is None else 1000 // fps
        if frame_duration <= 0:
            raise WidelandsParseError(f"{entry.key}.fps produces a non-positive frame time")
        hotspot = _int_pair(fields["hotspot"], label=f"{entry.key}.hotspot")
        directional = _optional_bool(fields.get("directional"), label=f"{entry.key}.directional")
        play_once = _optional_bool(fields.get("play_once"), label=f"{entry.key}.play_once")
        rows = _optional_int(fields.get("rows"), label=f"{entry.key}.rows", positive=True)
        columns = _optional_int(fields.get("columns"), label=f"{entry.key}.columns", positive=True)
        declared_frames = _optional_int(
            fields.get("frames"), label=f"{entry.key}.frames", positive=True
        )
        if representation == "spritesheet" and None in {rows, columns, declared_frames}:
            raise WidelandsParseError(
                f"packed animation {manifest_path}:{entry.key} lacks frames/rows/columns"
            )
        action, action_basis, variant_hint = _normalize_action(entry.key)
        directions: tuple[str | None, ...] = _DIRECTIONS if directional else (None,)
        for direction in directions:
            effective_basename = basename + (f"_{direction}" if direction is not None else "")
            all_images: list[SourceImage] = []
            primary_paths: tuple[str, ...] = ()
            available_scales: list[float] = []
            for scale, suffix in _SCALES:
                if representation == "spritesheet":
                    candidate = _join_logical(directory, f"{effective_basename}{suffix}.png")
                    scale_paths = (candidate,) if candidate in member_paths else ()
                else:
                    scale_paths = _sequential_paths(
                        directory, effective_basename + suffix, member_paths
                    )
                if scale == 1.0 and not scale_paths:
                    if representation == "spritesheet":
                        fallback = _join_logical(directory, f"{effective_basename}.png")
                        scale_paths = (fallback,) if fallback in member_paths else ()
                    else:
                        scale_paths = _sequential_paths(directory, effective_basename, member_paths)
                if not scale_paths:
                    continue
                available_scales.append(scale)
                if scale == 1.0:
                    primary_paths = scale_paths
                for ordinal, logical_path in enumerate(scale_paths):
                    all_images.append(
                        _source_image(
                            archive=archive,
                            by_path=by_path,
                            image_cache=image_cache,
                            logical_path=logical_path,
                            scale=scale,
                            ordinal=ordinal,
                        )
                    )
            quarantine: list[str] = []
            frames: list[FrameRecord] = []
            if 1.0 not in available_scales or not primary_paths:
                quarantine.append("missing_mandatory_scale_1_source")
            else:
                primary_by_path = {
                    image.logical_path: image for image in all_images if image.scale == 1.0
                }
                if representation == "spritesheet":
                    source = primary_by_path[primary_paths[0]]
                    assert rows is not None and columns is not None and declared_frames is not None
                    if source.width % columns or source.height % rows:
                        quarantine.append("spritesheet_dimensions_not_divisible_by_grid")
                    elif rows * columns < declared_frames:
                        quarantine.append("declared_frames_exceed_spritesheet_capacity")
                    elif (rows - 1) * columns > declared_frames:
                        quarantine.append("spritesheet_has_engine_rejected_extra_row")
                    else:
                        frame_width = source.width // columns
                        frame_height = source.height // rows
                        for ordinal in range(declared_frames):
                            frames.append(
                                FrameRecord(
                                    ordinal=ordinal,
                                    source_logical_path=source.logical_path,
                                    source_member_path=source.member_path,
                                    source_sha256=source.sha256,
                                    source_width=source.width,
                                    source_height=source.height,
                                    x=(ordinal % columns) * frame_width,
                                    y=(ordinal // columns) * frame_height,
                                    width=frame_width,
                                    height=frame_height,
                                    duration_milliseconds=frame_duration,
                                )
                            )
                else:
                    primary = [primary_by_path[path] for path in primary_paths]
                    dimensions = {(image.width, image.height) for image in primary}
                    if len(dimensions) != 1:
                        quarantine.append("numbered_frames_have_mismatched_dimensions")
                    else:
                        for ordinal, source in enumerate(primary):
                            frames.append(
                                FrameRecord(
                                    ordinal=ordinal,
                                    source_logical_path=source.logical_path,
                                    source_member_path=source.member_path,
                                    source_sha256=source.sha256,
                                    source_width=source.width,
                                    source_height=source.height,
                                    x=0,
                                    y=0,
                                    width=source.width,
                                    height=source.height,
                                    duration_milliseconds=frame_duration,
                                )
                            )
            records.append(
                AnimationRecord(
                    declared_name=entry.key,
                    normalized_action=action,
                    normalized_action_basis=action_basis,
                    variant_hint=variant_hint,
                    representation=representation,
                    direction=_DIRECTION_LABELS[direction] if direction is not None else None,
                    direction_basis=(
                        "engine_direction_suffix_order"
                        if direction is not None
                        else "not_directional"
                    ),
                    basename=effective_basename,
                    source_directory=directory,
                    location=SourceLocation(manifest_path, member_path, entry.line),
                    fps=fps,
                    frame_duration_milliseconds=frame_duration,
                    frame_duration_basis=(
                        "integer_1000_divided_by_declared_fps"
                        if fps is not None
                        else "engine_default_250_milliseconds"
                    ),
                    declared_frame_count=declared_frames,
                    rows=rows,
                    columns=columns,
                    hotspot=hotspot,
                    play_once=play_once,
                    loop_mode="one_shot" if play_once else "loop",
                    source_images=tuple(
                        sorted(
                            all_images,
                            key=lambda image: (image.scale, image.ordinal, image.logical_path),
                        )
                    ),
                    frames=tuple(frames),
                    available_scales=tuple(available_scales),
                    exact_source_sequence=not quarantine,
                    quarantine_reasons=tuple(sorted(set(quarantine))),
                )
            )
    return tuple(records)


def _evidence_document(
    archive: ZipFile, by_path: Mapping[str, _ArchiveMember], logical_path: str
) -> EvidenceDocument:
    try:
        member = by_path[logical_path]
    except KeyError as error:
        raise WidelandsArchiveError(
            f"required evidence member is missing: {logical_path}"
        ) from error
    payload = archive.read(member.info)
    return EvidenceDocument(logical_path, member.member_path, _sha256_bytes(payload), len(payload))


def _counter_tuple(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counter.items(), key=lambda item: item[0].encode("utf-8")))


def _audit_payload_without_hash(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    archive_root: str,
    counts: WidelandsArchiveCounts,
    entities: tuple[EntityRecord, ...],
    auxiliary_images: tuple[AuxiliaryImage, ...],
    rights: RightsAudit,
    engine_evidence: tuple[EvidenceDocument, ...],
    issues: tuple[AuditIssue, ...],
) -> dict[str, Any]:
    projection_policy = (
        "Only complete worker/critter body tracks with exact scale-1 sources are candidates.",
        "Player-color masks remain separate layers and are never silently composited.",
        "Menu, level/status, unreferenced layer, and effect images remain auxiliary evidence.",
        "Directional declarations expand in pinned engine order: ne,e,se,sw,w,nw.",
        "Spritesheet frames use pinned row-major cell order and declared frame count.",
        "No Lua code is executed; unsupported expressions are rejected or quarantined.",
        "Root GPL and credit evidence must travel with every projected sequence.",
    )
    return {
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "repository_url": WIDELANDS_REPOSITORY_URL,
        "commit": WIDELANDS_COMMIT,
        "commit_url": WIDELANDS_COMMIT_URL,
        "archive_url": WIDELANDS_ARCHIVE_URL,
        "archive_root": archive_root,
        "counts": asdict(counts),
        "entities": [asdict(entity) for entity in entities],
        "auxiliary_images": [asdict(image) for image in auxiliary_images],
        "rights": asdict(rights),
        "engine_evidence": [asdict(document) for document in engine_evidence],
        "issues": [asdict(issue) for issue in issues],
        "projection_policy": list(projection_policy),
    }


def audit_widelands_archive(
    archive_path: Path, *, archive_sha256: str | None = None
) -> WidelandsArchiveAudit:
    """Audit a Widelands repository ZIP without extracting it or writing a database."""

    archive_path = Path(archive_path)
    digest = archive_sha256 or _sha256_file(archive_path)
    try:
        with ZipFile(archive_path) as archive:
            infos = archive.infolist()
            root, members = _validate_members(infos)
            by_path = {member.logical_path: member for member in members}
            worker_manifests = tuple(
                path for path in by_path if _WORKER_MANIFEST_RE.fullmatch(path)
            )
            critter_manifests = tuple(
                path for path in by_path if _CRITTER_MANIFEST_RE.fullmatch(path)
            )
            manifest_paths = tuple(
                sorted(worker_manifests + critter_manifests, key=lambda path: path.encode("utf-8"))
            )
            if not manifest_paths:
                raise WidelandsArchiveError("no canonical Widelands worker/critter manifests found")

            image_cache: dict[str, _ImageInfo] = {}
            entities: list[EntityRecord] = []
            constructor_counts: Counter[str] = Counter()
            for manifest_path in manifest_paths:
                member = by_path[manifest_path]
                payload = archive.read(member.info)
                try:
                    text = payload.decode("utf-8-sig")
                except UnicodeDecodeError as error:
                    raise WidelandsParseError(f"manifest is not UTF-8: {manifest_path}") from error
                parsed_objects = _parse_manifest(text)
                if not parsed_objects:
                    raise WidelandsParseError(
                        f"manifest has no supported constructor: {manifest_path}"
                    )
                manifest_directory = PurePosixPath(manifest_path).parent.as_posix()
                for parsed in parsed_objects:
                    fields = _mapping(parsed.table)
                    if "name" not in fields:
                        raise WidelandsParseError(
                            f"constructor at {manifest_path}:{parsed.line} has no literal name"
                        )
                    entity_id = _require_string(fields["name"], label=f"{manifest_path}.name")
                    animation_directory = _resolve_directory(
                        fields.get("animation_directory"),
                        manifest_directory,
                        label=f"{manifest_path}.animation_directory",
                    )
                    animations: list[AnimationRecord] = []
                    for field, representation in (
                        ("animations", "numbered_files"),
                        ("spritesheets", "spritesheet"),
                    ):
                        value = fields.get(field)
                        if value is None:
                            continue
                        declarations = _resolve_table(
                            value, parsed.literal_tables, label=f"{manifest_path}.{field}"
                        )
                        animations.extend(
                            _animation_records(
                                archive=archive,
                                by_path=by_path,
                                image_cache=image_cache,
                                manifest_path=manifest_path,
                                member_path=member.member_path,
                                manifest_directory=manifest_directory,
                                animation_directory=animation_directory,
                                representation=representation,  # type: ignore[arg-type]
                                declarations=declarations,
                            )
                        )
                    role = parsed.constructor_role
                    constructor_counts[role] += 1
                    entity_class, class_basis = _entity_class(entity_id, role)
                    quarantine: list[str] = []
                    if not animations:
                        quarantine.append("no_literal_animation_tables")
                    if not any(animation.declared_name == "idle" for animation in animations):
                        quarantine.append("missing_engine_required_idle_animation")
                    if any(not animation.exact_source_sequence for animation in animations):
                        quarantine.append("contains_quarantined_animation_track")
                    tribe = None
                    parts = PurePosixPath(manifest_path).parts
                    if manifest_path.startswith("data/tribes/workers/"):
                        tribe = parts[3]
                    entities.append(
                        EntityRecord(
                            entity_id=entity_id,
                            tribe=tribe,
                            constructor_role=role,
                            entity_class=entity_class,
                            entity_class_basis=class_basis,
                            manifest_path=manifest_path,
                            member_path=member.member_path,
                            location=SourceLocation(manifest_path, member.member_path, parsed.line),
                            animation_directory=animation_directory,
                            animations=tuple(animations),
                            complete_entity=True,
                            quarantine_reasons=tuple(sorted(set(quarantine))),
                        )
                    )
            entities.sort(
                key=lambda entity: (entity.entity_id.encode("utf-8"), entity.manifest_path)
            )

            primary_paths = {
                image.logical_path
                for entity in entities
                for animation in entity.animations
                for image in animation.source_images
            }
            mask_paths = {
                image.playercolor_mask_path
                for entity in entities
                for animation in entity.animations
                for image in animation.source_images
                if image.playercolor_mask_path is not None
            }
            corpus_png_paths = tuple(path for path in by_path if _PNG_RE.fullmatch(path))
            auxiliaries: list[AuxiliaryImage] = []
            for logical_path in sorted(
                set(corpus_png_paths).difference(primary_paths), key=lambda p: p.encode("utf-8")
            ):
                member = by_path[logical_path]
                info = image_cache.get(logical_path)
                if info is None:
                    info = _image_info(archive, member)
                    image_cache[logical_path] = info
                leaf = PurePosixPath(logical_path).name.casefold()
                if logical_path in mask_paths or leaf.endswith("_pc.png"):
                    role: Literal[
                        "playercolor_mask",
                        "ui_icon",
                        "equipment_or_status_icon",
                        "unreferenced_layer_or_effect",
                    ] = "playercolor_mask"
                elif leaf == "menu.png":
                    role = "ui_icon"
                elif "_level" in leaf or leaf.startswith(
                    ("attack_", "defense_", "health_", "evade_")
                ):
                    role = "equipment_or_status_icon"
                else:
                    role = "unreferenced_layer_or_effect"
                auxiliaries.append(
                    AuxiliaryImage(
                        logical_path=logical_path,
                        member_path=member.member_path,
                        sha256=info.sha256,
                        size_bytes=info.size_bytes,
                        width=info.width,
                        height=info.height,
                        image_format=info.image_format,
                        role=role,
                    )
                )

            all_tracks = [animation for entity in entities for animation in entity.animations]
            primary_frames = [frame for animation in all_tracks for frame in animation.frames]
            primary_images_by_path = {
                image.logical_path: image
                for animation in all_tracks
                for image in animation.source_images
            }
            primary_hash_counts = Counter(image.sha256 for image in primary_images_by_path.values())
            action_counts = Counter(
                animation.normalized_action or "unknown" for animation in all_tracks
            )
            entity_class_counts = Counter(entity.entity_class for entity in entities)
            direction_counts = Counter(animation.direction or "none" for animation in all_tracks)
            representation_counts = Counter(animation.representation for animation in all_tracks)
            scale_counts = Counter(f"{image.scale:g}" for image in primary_images_by_path.values())
            format_counts = Counter(image.image_format for image in primary_images_by_path.values())
            auxiliary_role_counts = Counter(image.role for image in auxiliaries)
            surplus_cells = sum(
                (animation.rows or 0) * (animation.columns or 0)
                - (animation.declared_frame_count or 0)
                for animation in all_tracks
                if animation.representation == "spritesheet"
            )
            counts = WidelandsArchiveCounts(
                archive_members=len(infos),
                archive_files=len(members),
                archive_directories=len(infos) - len(members),
                worker_manifests=len(worker_manifests),
                critter_manifests=len(critter_manifests),
                constructor_role_counts=_counter_tuple(constructor_counts),
                entities=len(entities),
                complete_entities=sum(entity.complete_entity for entity in entities),
                animation_declarations=len(all_tracks),
                direction_tracks=sum(animation.direction is not None for animation in all_tracks),
                exact_tracks=sum(animation.exact_source_sequence for animation in all_tracks),
                quarantined_tracks=sum(
                    not animation.exact_source_sequence for animation in all_tracks
                ),
                primary_frames=len(primary_frames),
                primary_animation_images=len(primary_images_by_path),
                playercolor_mask_images=auxiliary_role_counts["playercolor_mask"],
                ui_icon_images=auxiliary_role_counts["ui_icon"],
                equipment_or_status_images=auxiliary_role_counts["equipment_or_status_icon"],
                unreferenced_layer_or_effect_images=auxiliary_role_counts[
                    "unreferenced_layer_or_effect"
                ],
                worker_tree_pngs=sum(
                    path.startswith("data/tribes/workers/") for path in corpus_png_paths
                ),
                critter_tree_pngs=sum(
                    path.startswith("data/world/critters/") for path in corpus_png_paths
                ),
                action_counts=_counter_tuple(action_counts),
                entity_class_counts=_counter_tuple(entity_class_counts),
                direction_counts=_counter_tuple(direction_counts),
                representation_counts=_counter_tuple(representation_counts),
                scale_counts=_counter_tuple(scale_counts),
                source_image_format_counts=_counter_tuple(format_counts),
                surplus_spritesheet_cells=surplus_cells,
                duplicate_primary_image_hash_groups=sum(
                    count > 1 for count in primary_hash_counts.values()
                ),
                duplicate_primary_image_hash_excess=sum(
                    max(0, count - 1) for count in primary_hash_counts.values()
                ),
            )
            rights_documents = {
                path: _evidence_document(archive, by_path, path) for path in _RIGHTS_EVIDENCE_PATHS
            }
            rights = RightsAudit(
                license_expression="GPL-2.0-or-later",
                license_basis="pinned_data/txts/LICENSE.lua_game_wide_statement_plus_COPYING",
                root_license=rights_documents["COPYING"],
                in_game_license=rights_documents["data/txts/LICENSE.lua"],
                credits=rights_documents["CREDITS"],
                developers=rights_documents["data/txts/developers.json"],
                caveat=(
                    "The game-wide GPL statement and project credits are preserved, but this audit "
                    "does not claim that every historical image has file-level creator attribution."
                ),
            )
            engine_evidence = tuple(
                _evidence_document(archive, by_path, path) for path in _ENGINE_EVIDENCE_PATHS
            )
            issues = (
                AuditIssue(
                    "unmapped_action_name",
                    action_counts["unknown"],
                    "Animation names outside the explicit vocabulary retain no guessed action.",
                ),
                AuditIssue(
                    "quarantined_animation_track",
                    counts.quarantined_tracks,
                    "Tracks lacking exact mandatory sources or valid geometry are not candidates.",
                ),
                AuditIssue(
                    "unreferenced_layer_or_effect_image",
                    counts.unreferenced_layer_or_effect_images,
                    "Unconsumed PNGs remain separate evidence, never complete entity sequences.",
                ),
                AuditIssue(
                    "duplicate_primary_image_payload",
                    counts.duplicate_primary_image_hash_excess,
                    "Exact byte duplicates require identity-aware downstream deduplication.",
                ),
            )
            entity_tuple = tuple(entities)
            auxiliary_tuple = tuple(auxiliaries)
            payload = _audit_payload_without_hash(
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                archive_root=root,
                counts=counts,
                entities=entity_tuple,
                auxiliary_images=auxiliary_tuple,
                rights=rights,
                engine_evidence=engine_evidence,
                issues=issues,
            )
            record_hash = _sha256_bytes(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
            return WidelandsArchiveAudit(
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                repository_url=WIDELANDS_REPOSITORY_URL,
                commit=WIDELANDS_COMMIT,
                commit_url=WIDELANDS_COMMIT_URL,
                archive_url=WIDELANDS_ARCHIVE_URL,
                archive_root=root,
                counts=counts,
                entities=entity_tuple,
                auxiliary_images=auxiliary_tuple,
                rights=rights,
                engine_evidence=engine_evidence,
                issues=issues,
                projection_policy=tuple(payload["projection_policy"]),
                audit_record_sha256=record_hash,
            )
    except BadZipFile as error:
        raise WidelandsArchiveError(f"not a valid ZIP archive: {archive_path}") from error


def audit_known_widelands_archive(archive_path: Path) -> WidelandsArchiveAudit:
    """Hash-check and audit the exact pinned Widelands snapshot."""

    archive_path = Path(archive_path)
    digest = _sha256_file(archive_path)
    if digest != EXPECTED_WIDELANDS_ARCHIVE_SHA256:
        raise WidelandsArchiveError(
            "Widelands archive SHA-256 mismatch: "
            f"expected {EXPECTED_WIDELANDS_ARCHIVE_SHA256}, got {digest}"
        )
    audit = audit_widelands_archive(archive_path, archive_sha256=digest)
    if audit.archive_root != _EXPECTED_ROOT:
        raise WidelandsArchiveError(
            "Widelands archive root mismatch: "
            f"expected {_EXPECTED_ROOT!r}, got {audit.archive_root!r}"
        )
    return audit


def known_widelands_cas_path(raw_root: Path) -> Path:
    """Return the immutable CAS path for the pinned archive digest."""

    digest = EXPECTED_WIDELANDS_ARCHIVE_SHA256
    return Path(raw_root) / "objects" / "sha256" / digest[:2] / digest[2:4] / digest
