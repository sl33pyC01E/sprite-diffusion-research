"""Read-only exact-snapshot audit for SuperTux creature sprite manifests.

SuperTux stores animation declarations as small S-expression ``.sprite``
documents.  This adapter parses that declarative subset without executing game
code and resolves image, mirror, vertical-flip, and clone actions in the same
order used by the pinned engine.  Every usable frame remains tied to an exact
ZIP member and byte digest; effects, components, stale manifests, and missing
references remain explicit quarantine evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile, ZipInfo

from spritelab.media.png import InvalidPNGError, inspect_png

EXPECTED_SUPERTUX_ARCHIVE_SHA256 = (
    "98ea15f57224ab3374fb5a3a1bfc538fa33790eecf60c5f2193d782e96b1abc5"
)
EXPECTED_SUPERTUX_INVENTORY_SHA256 = (
    "2da2740e59deeb960db9d24505171e7a97ab2cc5b3968b82d353f643927c48d2"
)
EXPECTED_SUPERTUX_AUDIT_RECORD_SHA256 = (
    "1b5fd92ffbfe2dc7fbd9ca7f53d0c7fd2b540b84f8a3da6f0fbe722f09183703"
)
SUPERTUX_COMMIT = "958bb9873c77f4063166d382076d4b19feb8a9c8"
SUPERTUX_REPOSITORY_URL = "https://github.com/SuperTux/supertux"
SUPERTUX_COMMIT_URL = f"{SUPERTUX_REPOSITORY_URL}/tree/{SUPERTUX_COMMIT}"
SUPERTUX_ARCHIVE_URL = f"https://codeload.github.com/SuperTux/supertux/zip/{SUPERTUX_COMMIT}"
_EXPECTED_ROOT = f"supertux-{SUPERTUX_COMMIT}"

_CREATURE_MANIFEST_RE = re.compile(r"^data/images/creatures/.+\.sprite$")
_CREATURE_PNG_RE = re.compile(r"^data/images/creatures/.+\.png$", re.IGNORECASE)
_RIGHTS_EVIDENCE_PATHS = ("LICENSE.txt", "README.md", "data/AUTHORS", "data/credits.stxt")
_ENGINE_EVIDENCE_PATHS = ("src/sprite/sprite_data.cpp", "src/sprite/sprite.cpp")
_SUPPORTED_ACTION_FIELDS = frozenset(
    {
        "name",
        "hitbox",
        "unisolid",
        "fps",
        "loops",
        "loop-frame",
        "family_name",
        "images",
        "mirror-action",
        "flip-action",
        "clone-action",
    }
)
_ALIAS_FIELDS = ("mirror-action", "flip-action", "clone-action")
_MAX_ARCHIVE_MEMBERS = 100_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024**3
_MAX_MEMBER_BYTES = 256 * 1024**2
_MAX_MANIFEST_BYTES = 4 * 1024**2
_MAX_SEXP_TOKENS = 1_000_000
_MAX_SEXP_DEPTH = 128

# These paths are layers, lights, or glows drawn with another sprite.  Keeping
# the list explicit makes the pinned classification reviewable and prevents a
# filename heuristic from silently promoting an effect to a complete entity.
_EFFECT_MANIFESTS = frozenset(
    {
        "data/images/creatures/crusher/corrupted/krosh_eye_glow.sprite",
        "data/images/creatures/crusher/corrupted/krush_eye_glow.sprite",
        "data/images/creatures/crystallo/crystallo-overlay.sprite",
        "data/images/creatures/darttrap/granito/dart_light.sprite",
        "data/images/creatures/darttrap/skull/dart_light.sprite",
        "data/images/creatures/dive_mine/ticking_glow/ticking_glow.sprite",
        "data/images/creatures/ghosttree/blue_root_light.sprite",
        "data/images/creatures/haywire/ticking_glow/ticking_glow.sprite",
        "data/images/creatures/mole/corrupted/core_glow/core_glow.sprite",
        "data/images/creatures/mr_bomb/ticking_glow/ticking_glow.sprite",
        "data/images/creatures/mr_tree/corrupted/eye_glow.sprite",
        "data/images/creatures/overlays/fireoverlay/fireoverlay.sprite",
        "data/images/creatures/overlays/iceoverlay/iceoverlay.sprite",
        "data/images/creatures/vicious_ivy/corrupted/eye_glow.sprite",
        "data/images/creatures/walkingleaf/corrupted/eye_glow.sprite",
    }
)

# Projectiles, boss parts, accessories, and placeholders do not depict the
# complete entity by themselves.  They are valuable compositing/effect
# evidence, but are not candidates for complete-body projection.
_MODULAR_MANIFESTS = frozenset(
    {
        "data/images/creatures/crusher/roots/crusher_root.sprite",
        "data/images/creatures/crusher/roots/crusher_root_side.sprite",
        "data/images/creatures/crystallo/shard.sprite",
        "data/images/creatures/darttrap/granito/root_dart.sprite",
        "data/images/creatures/darttrap/skull/skull_dart.sprite",
        "data/images/creatures/dispenser/invisible.sprite",
        "data/images/creatures/ghosttree/blue_root.sprite",
        "data/images/creatures/ghosttree/granito_root.sprite",
        "data/images/creatures/ghosttree/green_root.sprite",
        "data/images/creatures/ghosttree/main_root.sprite",
        "data/images/creatures/ghosttree/pinch_root.sprite",
        "data/images/creatures/ghosttree/red_root.sprite",
        "data/images/creatures/granito/corrupted/big/rock_mine.sprite",
        "data/images/creatures/granito/corrupted/big/root_spike.sprite",
        "data/images/creatures/granito/corrupted/hive/granito_hive.sprite",
        "data/images/creatures/mole/corrupted/root.sprite",
        "data/images/creatures/mole/corrupted/root_sapling.sprite",
        "data/images/creatures/mole/mole_rock.sprite",
        "data/images/creatures/mr_cherry/cherry.sprite",
        "data/images/creatures/mr_cherry/juicebox.sprite",
        "data/images/creatures/skullyhop/darttrap.sprite",
        "data/images/creatures/skullyhop/skull_dart.sprite",
        "data/images/creatures/tux/santahat.sprite",
    }
)

_ANIMAL_GROUPS = frozenset(
    {
        "fatbat",
        "fish",
        "flame_fish",
        "igel",
        "mole",
        "owl",
        "snail",
        "spidermite",
        "tarantula",
        "toad",
        "zeekling",
    }
)
_HUMANOID_GROUPS = frozenset({"nolok", "penny", "tux"})
_PLANT_GROUPS = frozenset(
    {"ghosttree", "leafshot", "mr_tree", "plant", "pumpkin", "vicious_ivy", "walkingleaf"}
)
_ELEMENTAL_GROUPS = frozenset({"crystallo", "flame", "kugelblitz", "livefire", "willowisp"})
_CONSTRUCT_GROUPS = frozenset(
    {
        "angrystone",
        "bag",
        "bsod",
        "crusher",
        "darttrap",
        "dispenser",
        "dive_mine",
        "gold_bomb",
        "granito",
        "haywire",
        "iceblock",
        "laptop",
        "mr_bomb",
        "mr_candle",
        "short_fuse",
        "stalactite",
        "totem",
    }
)

AssetRole = Literal["complete_entity", "modular_component", "effect_layer", "deprecated"]
FrameTransform = Literal["identity", "horizontal_flip", "vertical_flip", "horizontal_vertical_flip"]


class SuperTuxArchiveError(ValueError):
    """Raised when a ZIP is unsafe or is not an auditable SuperTux tree."""


class SuperTuxParseError(ValueError):
    """Raised when the supported sprite S-expression subset is malformed."""


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class AcquisitionEvidence:
    role: Literal["repository_metadata", "commit_metadata", "license_metadata"]
    url: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SymlinkEvidence:
    logical_path: str
    member_path: str
    target: str
    resolved_logical_path: str
    target_exists: bool


@dataclass(frozen=True, slots=True)
class FrameReference:
    ordinal: int
    origin_action: str
    requested_path: str
    logical_path: str
    member_path: str | None
    exists: bool
    sha256: str | None
    size_bytes: int | None
    width: int | None
    height: int | None
    mode: str | None
    alpha_kind: str | None
    transform: FrameTransform


@dataclass(frozen=True, slots=True)
class ActionRecord:
    declaration_ordinal: int
    line_number: int
    name: str
    normalized_action: str
    normalized_action_basis: str
    direction: str | None
    direction_basis: str
    action_stem: str
    alias_kind: Literal["mirror", "flip", "clone"] | None
    alias_target: str | None
    alias_chain: tuple[str, ...]
    declared_image_paths: tuple[str, ...]
    declared_fps: float | None
    effective_fps: float
    frame_duration_milliseconds: float
    declared_loops: int | None
    effective_loops: int
    has_custom_loops: bool
    declared_loop_frame: int | None
    effective_loop_frame: int
    hitbox: tuple[float, float, float, float]
    unisolid: bool
    family_name: str
    frames: tuple[FrameReference, ...]
    effective_declaration: bool
    exact_source_sequence: bool
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    manifest_id: str
    entity_group: str
    display_name: str
    role: AssetRole
    role_basis: str
    parent_entity_hint: str | None
    entity_class: str
    entity_class_basis: str
    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int
    actions: tuple[ActionRecord, ...]
    effective_action_names: tuple[str, ...]
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
    mode: str
    alpha_kind: str
    role: Literal["unreferenced_creature_image"]


@dataclass(frozen=True, slots=True)
class DuplicateImageGroup:
    sha256: str
    logical_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditIssue:
    code: str
    count: int
    message: str


@dataclass(frozen=True, slots=True)
class SuperTuxArchiveCounts:
    archive_members: int
    archive_files: int
    archive_directories: int
    archive_symlinks: int
    archive_compressed_bytes: int
    archive_uncompressed_bytes: int
    creature_manifests: int
    complete_entity_manifests: int
    modular_component_manifests: int
    effect_layer_manifests: int
    deprecated_manifests: int
    action_declarations: int
    effective_actions: int
    direct_image_actions: int
    mirror_alias_actions: int
    flip_alias_actions: int
    clone_alias_actions: int
    exact_complete_tracks: int
    quarantined_effective_tracks: int
    resolved_frame_occurrences: int
    direct_image_occurrences: int
    unique_referenced_images: int
    missing_image_reference_occurrences: int
    unique_missing_images: int
    creature_tree_pngs: int
    referenced_creature_tree_pngs: int
    referenced_external_pngs: int
    unreferenced_creature_tree_pngs: int
    duplicate_creature_image_hash_groups: int
    duplicate_creature_image_hash_excess: int
    duplicate_action_name_excess: int
    empty_effective_actions: int
    entity_class_counts: tuple[tuple[str, int], ...]
    action_counts: tuple[tuple[str, int], ...]
    direction_counts: tuple[tuple[str, int], ...]
    transform_counts: tuple[tuple[str, int], ...]
    source_mode_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class RightsAudit:
    repository_license_expression: str
    license_basis: str
    root_license: EvidenceDocument
    readme: EvidenceDocument
    authors: EvidenceDocument
    credits: EvidenceDocument
    attribution_summary: str
    caveat: str


@dataclass(frozen=True, slots=True)
class SuperTuxArchiveAudit:
    archive_sha256: str
    archive_size_bytes: int
    inventory_sha256: str
    repository_url: str
    commit: str
    commit_url: str
    archive_url: str
    archive_root: str
    counts: SuperTuxArchiveCounts
    manifests: tuple[ManifestRecord, ...]
    auxiliary_images: tuple[AuxiliaryImage, ...]
    duplicate_image_groups: tuple[DuplicateImageGroup, ...]
    rights: RightsAudit
    acquisition_evidence: tuple[AcquisitionEvidence, ...]
    engine_evidence: tuple[EvidenceDocument, ...]
    symlinks: tuple[SymlinkEvidence, ...]
    issues: tuple[AuditIssue, ...]
    projection_policy: tuple[str, ...]
    audit_record_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class _ArchiveEntry:
    logical_path: str
    member_path: str
    kind: Literal["file", "directory", "symlink"]
    unix_mode: int
    info: ZipInfo


@dataclass(frozen=True, slots=True)
class _ImageInfo:
    sha256: str
    size_bytes: int
    width: int
    height: int
    mode: str
    alpha_kind: str


@dataclass(frozen=True, slots=True)
class _Token:
    kind: Literal["left", "right", "string", "atom"]
    value: str
    line: int
    column: int


@dataclass(frozen=True, slots=True)
class _Atom:
    value: str
    quoted: bool
    line: int


@dataclass(frozen=True, slots=True)
class _ListNode:
    items: tuple[_Node, ...]
    line: int


_Node = _Atom | _ListNode


@dataclass(frozen=True, slots=True)
class _DeclaredAction:
    ordinal: int
    line: int
    name: str
    images: tuple[str, ...]
    alias_kind: Literal["mirror", "flip", "clone"] | None
    alias_target: str | None
    hitbox: tuple[float, ...] | None
    fps: float | None
    loops: int | None
    loop_frame: int | None
    unisolid: bool | None
    family_name: str | None


@dataclass(slots=True)
class _ActionState:
    name: str
    x_offset: float = 0.0
    y_offset: float = 0.0
    hitbox_w: float = 0.0
    hitbox_h: float = 0.0
    unisolid: bool = False
    fps: float = 10.0
    loops: int = -1
    loop_frame: int = 1
    has_custom_loops: bool = False
    family_name: str = ""
    frames: tuple[FrameReference, ...] = ()
    alias_chain: tuple[str, ...] = ()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _normalize_member_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise SuperTuxArchiveError(f"unsafe archive member path: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or re.match(r"^[A-Za-z]:", name):
        raise SuperTuxArchiveError(f"unsafe archive member path: {name!r}")
    normalized = pure.as_posix()
    if name.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _entry_kind(info: ZipInfo) -> tuple[Literal["file", "directory", "symlink"], int]:
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if info.is_dir():
        if file_type not in (0, stat.S_IFDIR):
            raise SuperTuxArchiveError(f"directory has incompatible mode: {info.filename!r}")
        return "directory", mode
    if file_type == stat.S_IFLNK:
        return "symlink", mode
    if file_type in (0, stat.S_IFREG):
        return "file", mode
    raise SuperTuxArchiveError(f"unsupported special ZIP member: {info.filename!r}")


def _validate_archive_members(
    archive: ZipFile,
) -> tuple[str, tuple[_ArchiveEntry, ...], tuple[SymlinkEvidence, ...], str]:
    infos = archive.infolist()
    if not infos:
        raise SuperTuxArchiveError("archive is empty")
    if len(infos) > _MAX_ARCHIVE_MEMBERS:
        raise SuperTuxArchiveError(f"archive has too many members: {len(infos)}")
    if sum(info.file_size for info in infos) > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise SuperTuxArchiveError("archive uncompressed size exceeds the audit limit")

    seen: set[str] = set()
    folded: dict[str, str] = {}
    roots: set[str] = set()
    entries: list[_ArchiveEntry] = []
    inventory_rows: list[dict[str, Any]] = []
    for info in infos:
        member_path = _normalize_member_name(info.filename)
        collision_key = member_path.rstrip("/")
        if collision_key in seen:
            raise SuperTuxArchiveError(f"duplicate archive member: {member_path!r}")
        seen.add(collision_key)
        prior = folded.get(collision_key.casefold())
        if prior is not None and prior != collision_key:
            raise SuperTuxArchiveError(
                f"case-colliding archive members: {prior!r}, {collision_key!r}"
            )
        folded[collision_key.casefold()] = collision_key
        if info.flag_bits & 0x1:
            raise SuperTuxArchiveError(f"encrypted ZIP member is not accepted: {member_path!r}")
        if info.file_size > _MAX_MEMBER_BYTES:
            raise SuperTuxArchiveError(f"ZIP member exceeds the audit limit: {member_path!r}")
        parts = PurePosixPath(collision_key).parts
        if not parts:
            raise SuperTuxArchiveError(f"member has no archive root: {member_path!r}")
        roots.add(parts[0])
        kind, mode = _entry_kind(info)
        logical_path = PurePosixPath(*parts[1:]).as_posix() if len(parts) > 1 else ""
        entry = _ArchiveEntry(logical_path, collision_key, kind, mode, info)
        entries.append(entry)
        inventory_rows.append(
            {
                "member_path": collision_key,
                "logical_path": logical_path,
                "kind": kind,
                "unix_mode": mode,
                "compressed_size": info.compress_size,
                "uncompressed_size": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "compression": info.compress_type,
            }
        )
    if len(roots) != 1:
        raise SuperTuxArchiveError(f"archive must have exactly one root, found {sorted(roots)!r}")
    root = next(iter(roots))

    by_logical = {entry.logical_path: entry for entry in entries if entry.logical_path}
    symlinks: list[SymlinkEvidence] = []
    for entry in entries:
        if entry.kind != "symlink":
            continue
        try:
            target = archive.read(entry.info).decode("utf-8")
        except UnicodeDecodeError as error:
            raise SuperTuxArchiveError(
                f"symlink target is not UTF-8: {entry.member_path!r}"
            ) from error
        if not target or "\x00" in target or "\\" in target or target.startswith("/"):
            raise SuperTuxArchiveError(f"unsafe symlink target {target!r}: {entry.logical_path}")
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(entry.logical_path), target))
        if resolved == ".." or resolved.startswith("../") or re.match(r"^[A-Za-z]:", resolved):
            raise SuperTuxArchiveError(f"escaping symlink target {target!r}: {entry.logical_path}")
        target_entry = by_logical.get(resolved)
        symlinks.append(
            SymlinkEvidence(
                logical_path=entry.logical_path,
                member_path=entry.member_path,
                target=target,
                resolved_logical_path=resolved,
                target_exists=target_entry is not None and target_entry.kind == "file",
            )
        )
    inventory_sha256 = _canonical_hash(inventory_rows)
    return root, tuple(entries), tuple(symlinks), inventory_sha256


def _tokenize(text: str, *, logical_path: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    line = 1
    column = 1
    while index < len(text):
        char = text[index]
        if char in " \t\r":
            index += 1
            column += 1
            continue
        if char == "\n":
            index += 1
            line += 1
            column = 1
            continue
        if char == ";":
            while index < len(text) and text[index] != "\n":
                index += 1
                column += 1
            continue
        if char == "(":
            tokens.append(_Token("left", char, line, column))
            index += 1
            column += 1
            continue
        if char == ")":
            tokens.append(_Token("right", char, line, column))
            index += 1
            column += 1
            continue
        if char == '"':
            start_line, start_column = line, column
            index += 1
            column += 1
            value: list[str] = []
            while index < len(text):
                char = text[index]
                if char == '"':
                    index += 1
                    column += 1
                    break
                if char == "\\":
                    if index + 1 >= len(text):
                        raise SuperTuxParseError(
                            f"unterminated escape at {logical_path}:{line}:{column}"
                        )
                    escaped = text[index + 1]
                    value.append({"n": "\n", "r": "\r", "t": "\t"}.get(escaped, escaped))
                    index += 2
                    column += 2
                    continue
                if char == "\n":
                    raise SuperTuxParseError(
                        f"newline in string at {logical_path}:{start_line}:{start_column}"
                    )
                value.append(char)
                index += 1
                column += 1
            else:
                raise SuperTuxParseError(
                    f"unterminated string at {logical_path}:{start_line}:{start_column}"
                )
            tokens.append(_Token("string", "".join(value), start_line, start_column))
            continue
        start = index
        start_column = column
        while index < len(text) and not text[index].isspace() and text[index] not in "();":
            index += 1
            column += 1
        if start == index:
            raise SuperTuxParseError(f"unexpected token at {logical_path}:{line}:{column}")
        tokens.append(_Token("atom", text[start:index], line, start_column))
    return tuple(tokens)


def _parse_document(text: str, *, logical_path: str) -> _ListNode:
    tokens = _tokenize(text, logical_path=logical_path)
    if len(tokens) > _MAX_SEXP_TOKENS:
        raise SuperTuxParseError(f"sprite document has too many tokens: {logical_path}")
    index = 0

    def parse_one(depth: int = 0) -> _Node:
        nonlocal index
        if depth > _MAX_SEXP_DEPTH:
            raise SuperTuxParseError(f"sprite document is nested too deeply: {logical_path}")
        if index >= len(tokens):
            raise SuperTuxParseError(f"unexpected end of document: {logical_path}")
        token = tokens[index]
        index += 1
        if token.kind == "right":
            raise SuperTuxParseError(
                f"unexpected ')' at {logical_path}:{token.line}:{token.column}"
            )
        if token.kind in {"atom", "string"}:
            return _Atom(token.value, token.kind == "string", token.line)
        items: list[_Node] = []
        while True:
            if index >= len(tokens):
                raise SuperTuxParseError(
                    f"unclosed list at {logical_path}:{token.line}:{token.column}"
                )
            if tokens[index].kind == "right":
                index += 1
                return _ListNode(tuple(items), token.line)
            items.append(parse_one(depth + 1))

    root = parse_one()
    if index != len(tokens):
        token = tokens[index]
        raise SuperTuxParseError(
            f"multiple root expressions at {logical_path}:{token.line}:{token.column}"
        )
    if not isinstance(root, _ListNode):
        raise SuperTuxParseError(f"sprite root is not a list: {logical_path}")
    return root


def _atom(node: _Node, *, label: str) -> _Atom:
    if not isinstance(node, _Atom):
        raise SuperTuxParseError(f"{label} must be an atom")
    return node


def _field_map(action: _ListNode, *, logical_path: str) -> dict[str, _ListNode]:
    fields: dict[str, _ListNode] = {}
    for node in action.items[1:]:
        if not isinstance(node, _ListNode) or not node.items:
            raise SuperTuxParseError(f"malformed action field at {logical_path}:{action.line}")
        key_atom = _atom(node.items[0], label=f"{logical_path}:{node.line} field name")
        key = key_atom.value
        if key not in _SUPPORTED_ACTION_FIELDS:
            raise SuperTuxParseError(
                f"unsupported action field {key!r}: {logical_path}:{node.line}"
            )
        if key in fields:
            raise SuperTuxParseError(f"duplicate action field {key!r}: {logical_path}:{node.line}")
        fields[key] = node
    return fields


def _one_text(fields: Mapping[str, _ListNode], key: str, *, label: str) -> str | None:
    node = fields.get(key)
    if node is None:
        return None
    if len(node.items) != 2:
        raise SuperTuxParseError(f"{label}.{key} must contain one value")
    value = _atom(node.items[1], label=f"{label}.{key}")
    return value.value


def _float_value(atom: _Atom, *, label: str) -> float:
    try:
        value = float(atom.value)
    except ValueError as error:
        raise SuperTuxParseError(f"{label} is not numeric: {atom.value!r}") from error
    if not math.isfinite(value):
        raise SuperTuxParseError(f"{label} must be finite")
    return value


def _one_float(fields: Mapping[str, _ListNode], key: str, *, label: str) -> float | None:
    node = fields.get(key)
    if node is None:
        return None
    if len(node.items) != 2:
        raise SuperTuxParseError(f"{label}.{key} must contain one value")
    return _float_value(_atom(node.items[1], label=f"{label}.{key}"), label=f"{label}.{key}")


def _one_int(fields: Mapping[str, _ListNode], key: str, *, label: str) -> int | None:
    value = _one_float(fields, key, label=label)
    if value is None:
        return None
    if not value.is_integer():
        raise SuperTuxParseError(f"{label}.{key} must be an integer")
    return int(value)


def _one_bool(fields: Mapping[str, _ListNode], key: str, *, label: str) -> bool | None:
    raw = _one_text(fields, key, label=label)
    if raw is None:
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise SuperTuxParseError(f"{label}.{key} must be true or false")


def _parse_actions(text: str, *, logical_path: str) -> tuple[_DeclaredAction, ...]:
    root = _parse_document(text, logical_path=logical_path)
    if (
        not root.items
        or _atom(root.items[0], label=f"{logical_path} root").value != "supertux-sprite"
    ):
        raise SuperTuxParseError(f"not a supertux-sprite document: {logical_path}")
    actions: list[_DeclaredAction] = []
    for node in root.items[1:]:
        if not isinstance(node, _ListNode) or not node.items:
            raise SuperTuxParseError(f"malformed sprite entry at {logical_path}:{root.line}")
        tag = _atom(node.items[0], label=f"{logical_path}:{node.line} entry").value
        if tag != "action":
            raise SuperTuxParseError(
                f"unsupported sprite entry {tag!r}: {logical_path}:{node.line}"
            )
        fields = _field_map(node, logical_path=logical_path)
        label = f"{logical_path}:{node.line}"
        name = _one_text(fields, "name", label=label)
        if not name:
            raise SuperTuxParseError(f"action has no non-empty name: {label}")
        image_node = fields.get("images")
        images: tuple[str, ...] = ()
        if image_node is not None:
            images = tuple(
                _atom(item, label=f"{label}.images").value for item in image_node.items[1:]
            )
            if not images or any(not path for path in images):
                raise SuperTuxParseError(f"{label}.images must contain non-empty paths")
        aliases = [key for key in _ALIAS_FIELDS if key in fields]
        if len(aliases) + int(bool(images)) != 1:
            raise SuperTuxParseError(
                f"{label} must contain exactly one images/mirror/flip/clone source"
            )
        alias_field = aliases[0] if aliases else None
        alias_kind = {
            "mirror-action": "mirror",
            "flip-action": "flip",
            "clone-action": "clone",
        }.get(alias_field)
        alias_target = _one_text(fields, alias_field, label=label) if alias_field else None
        if alias_field and not alias_target:
            raise SuperTuxParseError(f"{label}.{alias_field} has no target")
        hitbox_node = fields.get("hitbox")
        hitbox: tuple[float, ...] | None = None
        if hitbox_node is not None:
            hitbox = tuple(
                _float_value(_atom(item, label=f"{label}.hitbox"), label=f"{label}.hitbox")
                for item in hitbox_node.items[1:]
            )
            if len(hitbox) not in (2, 4):
                raise SuperTuxParseError(f"{label}.hitbox must contain two or four values")
        fps = _one_float(fields, "fps", label=label)
        if fps is not None and fps < 0:
            raise SuperTuxParseError(f"{label}.fps must be non-negative")
        loop_frame = _one_int(fields, "loop-frame", label=label)
        actions.append(
            _DeclaredAction(
                ordinal=len(actions),
                line=node.line,
                name=name,
                images=images,
                alias_kind=alias_kind,  # type: ignore[arg-type]
                alias_target=alias_target,
                hitbox=hitbox,
                fps=fps,
                loops=_one_int(fields, "loops", label=label),
                loop_frame=loop_frame,
                unisolid=_one_bool(fields, "unisolid", label=label),
                family_name=_one_text(fields, "family_name", label=label),
            )
        )
    if not actions:
        raise SuperTuxParseError(f"sprite document has no actions: {logical_path}")
    return tuple(actions)


def _image_info(archive: ZipFile, entry: _ArchiveEntry) -> _ImageInfo:
    if entry.kind != "file":
        raise SuperTuxParseError(f"sprite image is not a regular file: {entry.logical_path}")
    payload = archive.read(entry.info)
    try:
        inspection = inspect_png(payload)
    except InvalidPNGError as error:
        raise SuperTuxParseError(f"invalid PNG {entry.logical_path}: {error}") from error
    if inspection.is_animated:
        raise SuperTuxParseError(
            f"APNG is not accepted as a SuperTux source surface: {entry.logical_path}"
        )
    return _ImageInfo(
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        width=inspection.size[0],
        height=inspection.size[1],
        mode=inspection.mode,
        alpha_kind=inspection.alpha_kind,
    )


def _resolve_image_path(manifest_path: str, requested_path: str) -> str:
    if (
        not requested_path
        or "\x00" in requested_path
        or "\\" in requested_path
        or requested_path.startswith("/")
        or re.match(r"^[A-Za-z]:", requested_path)
    ):
        raise SuperTuxParseError(f"unsafe image reference {requested_path!r} in {manifest_path}")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(manifest_path), requested_path))
    if resolved == ".." or resolved.startswith("../"):
        raise SuperTuxParseError(
            f"image reference escapes archive root: {manifest_path}: {requested_path!r}"
        )
    if not resolved.casefold().endswith(".png"):
        raise SuperTuxParseError(
            f"sprite image reference is not PNG: {manifest_path}: {requested_path!r}"
        )
    return PurePosixPath(resolved).as_posix()


def _toggle_transform(transform: FrameTransform, *, horizontal: bool) -> FrameTransform:
    if horizontal:
        return {
            "identity": "horizontal_flip",
            "horizontal_flip": "identity",
            "vertical_flip": "horizontal_vertical_flip",
            "horizontal_vertical_flip": "vertical_flip",
        }[transform]  # type: ignore[return-value]
    return {
        "identity": "vertical_flip",
        "horizontal_flip": "horizontal_vertical_flip",
        "vertical_flip": "identity",
        "horizontal_vertical_flip": "horizontal_flip",
    }[transform]  # type: ignore[return-value]


def _direction_and_stem(name: str) -> tuple[str | None, str, str]:
    tokens = [token for token in re.split(r"[-_]+", name.casefold()) if token]
    horizontal = next((token for token in tokens if token in {"left", "right"}), None)
    vertical = next(
        (
            {"upwards": "up", "downwards": "down"}.get(token, token)
            for token in tokens
            if token in {"up", "down", "upwards", "downwards"}
        ),
        None,
    )
    direction = None
    if horizontal and vertical:
        direction = f"{vertical}_{horizontal}"
    elif horizontal:
        direction = horizontal
    elif vertical:
        direction = vertical
    stem_tokens = [
        token
        for token in tokens
        if token not in {"left", "right", "up", "down", "upwards", "downwards", "middle"}
    ]
    stem = "-".join(stem_tokens)
    return direction, ("action_name_direction_tokens" if direction else "none"), stem


def _normalized_action(name: str, stem: str) -> tuple[str, str]:
    tokens = set(token for token in re.split(r"[-_]+", stem) if token)
    ordered_rules: tuple[tuple[str, frozenset[str]], ...] = (
        ("death", frozenset({"dead", "death", "die", "dying", "melting", "shattered"})),
        ("hurt", frozenset({"busted", "dizzy", "flat", "iced", "squished", "stunned"})),
        ("run", frozenset({"run", "running"})),
        ("walk", frozenset({"walk", "walking"})),
        ("idle", frozenset({"default", "idle", "stand", "standing"})),
        ("sleep", frozenset({"sleep", "sleeping"})),
        ("wake", frozenset({"wake", "waking"})),
        ("jump", frozenset({"jump", "jumping", "leap", "walljump"})),
        ("fall", frozenset({"fall", "falling"})),
        ("fly", frozenset({"fly", "flying"})),
        ("swim", frozenset({"swim", "swimming"})),
        ("climb", frozenset({"climb", "climbing"})),
        ("slide", frozenset({"skid", "slide", "sliding"})),
        ("attack", frozenset({"attack", "attacking", "bite", "kick", "peck", "shoot"})),
        ("throw", frozenset({"throw", "throwing"})),
        ("stomp", frozenset({"stomp", "stomping"})),
        ("emote", frozenset({"celebrate", "rage", "scratch", "taunt", "wave"})),
        ("interact", frozenset({"grab", "push"})),
        ("spawn", frozenset({"appear", "loading", "spawn"})),
    )
    for normalized, vocabulary in ordered_rules:
        if tokens.intersection(vocabulary):
            return normalized, f"explicit_action_vocabulary:{normalized}"
    if not stem:
        return "walk", "direction_only_action_name"
    normalized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return normalized or name.casefold(), "preserved_action_stem"


def _manifest_role(path: str) -> tuple[AssetRole, str]:
    if path.endswith(".deprecated.sprite"):
        return "deprecated", "deprecated_filename"
    if path in _EFFECT_MANIFESTS:
        return "effect_layer", "pinned_effect_layer_review"
    if path in _MODULAR_MANIFESTS:
        return "modular_component", "pinned_component_or_projectile_review"
    return "complete_entity", "pinned_complete_body_review"


def _entity_class(group: str, role: AssetRole) -> tuple[str, str]:
    if role in {"effect_layer", "modular_component"}:
        return "component", "non_complete_manifest_role"
    if group in _ANIMAL_GROUPS:
        return "animal", "pinned_creature_family_review"
    if group in _HUMANOID_GROUPS:
        return "humanoid", "pinned_creature_family_review"
    if group in _PLANT_GROUPS:
        return "plant", "pinned_creature_family_review"
    if group in _ELEMENTAL_GROUPS:
        return "elemental", "pinned_creature_family_review"
    if group in _CONSTRUCT_GROUPS:
        return "construct", "pinned_creature_family_review"
    return "monster", "conservative_complete_creature_default"


def _resolve_manifest_actions(
    *,
    archive: ZipFile,
    by_path: Mapping[str, _ArchiveEntry],
    image_cache: dict[str, _ImageInfo],
    manifest_path: str,
    declarations: Sequence[_DeclaredAction],
) -> tuple[ActionRecord, ...]:
    states: dict[str, _ActionState] = {}
    records: list[ActionRecord] = []
    for declaration in declarations:
        quarantine: list[str] = []
        state = states.get(declaration.name)
        if state is None:
            state = _ActionState(name=declaration.name)
            states[declaration.name] = state

        # Matches SpriteData::parse_action: a redefinition clears dimensions
        # and surfaces, but preserves several other fields until overwritten.
        state.hitbox_w = 0.0
        state.hitbox_h = 0.0
        state.frames = ()
        state.alias_chain = ()
        if declaration.hitbox is not None:
            if len(declaration.hitbox) == 4:
                state.hitbox_w = declaration.hitbox[2]
                state.hitbox_h = declaration.hitbox[3]
            state.x_offset = declaration.hitbox[0]
            state.y_offset = declaration.hitbox[1]
        if declaration.unisolid is not None:
            state.unisolid = declaration.unisolid
        if declaration.fps is not None:
            state.fps = declaration.fps
        if declaration.loops is not None:
            state.loops = declaration.loops
            state.has_custom_loops = True
        if declaration.loop_frame is not None:
            state.loop_frame = declaration.loop_frame
            if state.loop_frame < 1:
                state.loop_frame = 1
                quarantine.append("loop_frame_clamped_to_one")
        state.family_name = declaration.family_name or f"::{declaration.name}"

        if declaration.alias_kind is not None:
            source = states.get(declaration.alias_target or "")
            if source is None:
                quarantine.append("alias_target_not_previously_declared")
                state.frames = ()
            elif declaration.alias_kind == "clone":
                old_name = state.name
                old_family = state.family_name
                state = _ActionState(
                    name=source.name,
                    x_offset=source.x_offset,
                    y_offset=source.y_offset,
                    hitbox_w=source.hitbox_w,
                    hitbox_h=source.hitbox_h,
                    unisolid=source.unisolid,
                    fps=source.fps,
                    loops=source.loops,
                    loop_frame=source.loop_frame,
                    has_custom_loops=source.has_custom_loops,
                    family_name=source.family_name,
                    frames=source.frames,
                    alias_chain=source.alias_chain,
                )
                state.name = old_name
                state.family_name = old_family
                if state.family_name == f"::{state.name}":
                    state.family_name = source.family_name
                state.alias_chain = source.alias_chain + (f"clone:{declaration.alias_target}",)
                states[declaration.name] = state
            else:
                horizontal = declaration.alias_kind == "mirror"
                state.frames = tuple(
                    replace(
                        frame,
                        ordinal=index,
                        transform=_toggle_transform(frame.transform, horizontal=horizontal),
                    )
                    for index, frame in enumerate(source.frames)
                )
                if state.hitbox_w < 1 and state.hitbox_h < 1:
                    state.hitbox_w = source.hitbox_w
                    state.hitbox_h = source.hitbox_h
                    state.x_offset = source.x_offset
                    state.y_offset = source.y_offset
                if not state.has_custom_loops and source.has_custom_loops:
                    state.has_custom_loops = True
                    state.loops = source.loops
                if state.fps == 0:
                    state.fps = source.fps
                if state.family_name == f"::{state.name}":
                    state.family_name = source.family_name
                state.alias_chain = source.alias_chain + (
                    f"{declaration.alias_kind}:{declaration.alias_target}",
                )
                if source is state:
                    quarantine.append("self_alias_clears_source_frames")
        else:
            frames: list[FrameReference] = []
            max_width = 0
            max_height = 0
            for ordinal, requested in enumerate(declaration.images):
                logical_path = _resolve_image_path(manifest_path, requested)
                entry = by_path.get(logical_path)
                info = None
                if entry is not None and entry.kind == "file":
                    info = image_cache.get(logical_path)
                    if info is None:
                        info = _image_info(archive, entry)
                        image_cache[logical_path] = info
                    max_width = max(max_width, info.width)
                    max_height = max(max_height, info.height)
                else:
                    quarantine.append("missing_source_image")
                frames.append(
                    FrameReference(
                        ordinal=ordinal,
                        origin_action=declaration.name,
                        requested_path=requested,
                        logical_path=logical_path,
                        member_path=entry.member_path if entry is not None else None,
                        exists=info is not None,
                        sha256=info.sha256 if info is not None else None,
                        size_bytes=info.size_bytes if info is not None else None,
                        width=info.width if info is not None else None,
                        height=info.height if info is not None else None,
                        mode=info.mode if info is not None else None,
                        alpha_kind=info.alpha_kind if info is not None else None,
                        transform="identity",
                    )
                )
            state.frames = tuple(frames)
            if state.hitbox_w < 1:
                state.hitbox_w = max_width - state.x_offset
            if state.hitbox_h < 1:
                state.hitbox_h = max_height - state.y_offset

        if state.loop_frame > len(state.frames) and state.frames:
            state.loop_frame = 1
            quarantine.append("loop_frame_clamped_to_one")
        if not state.frames:
            quarantine.append("empty_resolved_frame_sequence")
        if any(not frame.exists for frame in state.frames):
            quarantine.append("missing_source_image")
        if state.fps <= 0:
            quarantine.append("nonpositive_effective_fps")
        if state.hitbox_w <= 0 or state.hitbox_h <= 0:
            quarantine.append("nonpositive_effective_hitbox")

        direction, direction_basis, stem = _direction_and_stem(declaration.name)
        normalized_action, normalized_basis = _normalized_action(declaration.name, stem)
        unique_quarantine = tuple(sorted(set(quarantine)))
        records.append(
            ActionRecord(
                declaration_ordinal=declaration.ordinal,
                line_number=declaration.line,
                name=declaration.name,
                normalized_action=normalized_action,
                normalized_action_basis=normalized_basis,
                direction=direction,
                direction_basis=direction_basis,
                action_stem=stem,
                alias_kind=declaration.alias_kind,
                alias_target=declaration.alias_target,
                alias_chain=state.alias_chain,
                declared_image_paths=declaration.images,
                declared_fps=declaration.fps,
                effective_fps=state.fps,
                frame_duration_milliseconds=(
                    round(1000.0 / state.fps, 9) if state.fps > 0 else 0.0
                ),
                declared_loops=declaration.loops,
                effective_loops=state.loops,
                has_custom_loops=state.has_custom_loops,
                declared_loop_frame=declaration.loop_frame,
                effective_loop_frame=state.loop_frame,
                hitbox=(state.x_offset, state.y_offset, state.hitbox_w, state.hitbox_h),
                unisolid=state.unisolid,
                family_name=state.family_name,
                frames=state.frames,
                effective_declaration=True,
                exact_source_sequence=not unique_quarantine,
                quarantine_reasons=unique_quarantine,
            )
        )

    last_ordinal = {record.name: record.declaration_ordinal for record in records}
    final_records: list[ActionRecord] = []
    for record in records:
        if last_ordinal[record.name] == record.declaration_ordinal:
            final_records.append(record)
            continue
        reasons = tuple(sorted(set(record.quarantine_reasons + ("superseded_duplicate_action",))))
        final_records.append(
            replace(
                record,
                effective_declaration=False,
                exact_source_sequence=False,
                quarantine_reasons=reasons,
            )
        )
    return tuple(final_records)


def _manifest_identity(path: str) -> tuple[str, str, str]:
    relative = path.removeprefix("data/images/creatures/")
    manifest_id = relative.removesuffix(".sprite")
    parts = PurePosixPath(relative).parts
    group = parts[0]
    display_name = PurePosixPath(relative).name.removesuffix(".sprite").removesuffix(".deprecated")
    return manifest_id, group, display_name


def _evidence_document(
    archive: ZipFile, by_path: Mapping[str, _ArchiveEntry], logical_path: str
) -> EvidenceDocument:
    entry = by_path.get(logical_path)
    if entry is None or entry.kind != "file":
        raise SuperTuxArchiveError(f"required evidence member is missing: {logical_path}")
    payload = archive.read(entry.info)
    return EvidenceDocument(logical_path, entry.member_path, _sha256_bytes(payload), len(payload))


def _counter_tuple(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counter.items(), key=lambda item: item[0].encode("utf-8")))


def _acquisition_evidence() -> tuple[AcquisitionEvidence, ...]:
    return (
        AcquisitionEvidence(
            role="repository_metadata",
            url="https://api.github.com/repos/SuperTux/supertux",
            sha256="7d0f3b826ea40dd48352590ce30851ee6c7aaae7baae03c8e198e9db78ca1f75",
            size_bytes=6_267,
        ),
        AcquisitionEvidence(
            role="commit_metadata",
            url=f"https://api.github.com/repos/SuperTux/supertux/commits/{SUPERTUX_COMMIT}",
            sha256="b4128e1bb1bbc46ed82a4abd758805e5397287b65fba6e6ce01ebe77a645cd3a",
            size_bytes=7_566,
        ),
        AcquisitionEvidence(
            role="license_metadata",
            url=(f"https://api.github.com/repos/SuperTux/supertux/license?ref={SUPERTUX_COMMIT}"),
            sha256="54554de21bafb5a8dfaa81d359ecf1a1684f0990a2b15d232817856d439e528b",
            size_bytes=49_550,
        ),
    )


def _audit_payload_without_hash(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    inventory_sha256: str,
    archive_root: str,
    counts: SuperTuxArchiveCounts,
    manifests: tuple[ManifestRecord, ...],
    auxiliary_images: tuple[AuxiliaryImage, ...],
    duplicate_image_groups: tuple[DuplicateImageGroup, ...],
    rights: RightsAudit,
    acquisition_evidence: tuple[AcquisitionEvidence, ...],
    engine_evidence: tuple[EvidenceDocument, ...],
    symlinks: tuple[SymlinkEvidence, ...],
    issues: tuple[AuditIssue, ...],
) -> dict[str, Any]:
    projection_policy = (
        "Only complete_entity manifests with effective exact_source_sequence actions qualify.",
        "Effect layers, modular components, projectiles, accessories, and deprecated data "
        "stay separate.",
        "Declared frame order and repetition are preserved; byte-identical frames are not "
        "collapsed.",
        "Mirror and vertical-flip aliases retain explicit transforms over their exact source PNGs.",
        "Clone aliases follow pinned engine semantics, including replacement of declared "
        "timing fields.",
        "Missing image references and empty effective actions are auditable quarantine records.",
        "Project license, README data-license note, AUTHORS, and credits evidence travel "
        "with exports.",
        "No Squirrel/C++ code is executed and the repository ZIP is never extracted by "
        "this adapter.",
    )
    return {
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "inventory_sha256": inventory_sha256,
        "repository_url": SUPERTUX_REPOSITORY_URL,
        "commit": SUPERTUX_COMMIT,
        "commit_url": SUPERTUX_COMMIT_URL,
        "archive_url": SUPERTUX_ARCHIVE_URL,
        "archive_root": archive_root,
        "counts": asdict(counts),
        "manifests": [asdict(manifest) for manifest in manifests],
        "auxiliary_images": [asdict(image) for image in auxiliary_images],
        "duplicate_image_groups": [asdict(group) for group in duplicate_image_groups],
        "rights": asdict(rights),
        "acquisition_evidence": [asdict(evidence) for evidence in acquisition_evidence],
        "engine_evidence": [asdict(document) for document in engine_evidence],
        "symlinks": [asdict(symlink) for symlink in symlinks],
        "issues": [asdict(issue) for issue in issues],
        "projection_policy": list(projection_policy),
    }


def audit_supertux_archive(
    archive_path: Path, *, archive_sha256: str | None = None
) -> SuperTuxArchiveAudit:
    """Audit a SuperTux repository ZIP without extraction or database writes."""

    archive_path = Path(archive_path)
    digest = archive_sha256 or _sha256_file(archive_path)
    try:
        with ZipFile(archive_path) as archive:
            root, entries, symlinks, inventory_sha256 = _validate_archive_members(archive)
            by_path = {entry.logical_path: entry for entry in entries if entry.logical_path}
            manifest_paths = tuple(
                sorted(
                    (
                        path
                        for path, entry in by_path.items()
                        if entry.kind == "file" and _CREATURE_MANIFEST_RE.fullmatch(path)
                    ),
                    key=lambda path: path.encode("utf-8"),
                )
            )
            if not manifest_paths:
                raise SuperTuxArchiveError("no creature sprite manifests found")

            image_cache: dict[str, _ImageInfo] = {}
            manifests: list[ManifestRecord] = []
            for manifest_path in manifest_paths:
                entry = by_path[manifest_path]
                if entry.info.file_size > _MAX_MANIFEST_BYTES:
                    raise SuperTuxArchiveError(
                        f"sprite manifest exceeds the parse limit: {manifest_path}"
                    )
                payload = archive.read(entry.info)
                try:
                    text = payload.decode("utf-8-sig")
                except UnicodeDecodeError as error:
                    raise SuperTuxParseError(
                        f"sprite manifest is not UTF-8: {manifest_path}"
                    ) from error
                declarations = _parse_actions(text, logical_path=manifest_path)
                actions = _resolve_manifest_actions(
                    archive=archive,
                    by_path=by_path,
                    image_cache=image_cache,
                    manifest_path=manifest_path,
                    declarations=declarations,
                )
                manifest_id, group, display_name = _manifest_identity(manifest_path)
                role, role_basis = _manifest_role(manifest_path)
                entity_class, class_basis = _entity_class(group, role)
                effective = tuple(action for action in actions if action.effective_declaration)
                manifest_quarantine: list[str] = []
                if role == "deprecated":
                    manifest_quarantine.append("deprecated_manifest")
                if role != "complete_entity":
                    manifest_quarantine.append("not_a_complete_entity_manifest")
                if any(not action.exact_source_sequence for action in effective):
                    manifest_quarantine.append("contains_quarantined_effective_action")
                if len(actions) != len(effective):
                    manifest_quarantine.append("contains_duplicate_action_name")
                manifests.append(
                    ManifestRecord(
                        manifest_id=manifest_id,
                        entity_group=group,
                        display_name=display_name,
                        role=role,
                        role_basis=role_basis,
                        parent_entity_hint=(
                            group if role in {"modular_component", "effect_layer"} else None
                        ),
                        entity_class=entity_class,
                        entity_class_basis=class_basis,
                        logical_path=manifest_path,
                        member_path=entry.member_path,
                        sha256=_sha256_bytes(payload),
                        size_bytes=len(payload),
                        actions=actions,
                        effective_action_names=tuple(action.name for action in effective),
                        complete_entity=role == "complete_entity",
                        quarantine_reasons=tuple(sorted(set(manifest_quarantine))),
                    )
                )

            creature_png_paths = tuple(
                sorted(
                    (
                        path
                        for path, entry in by_path.items()
                        if entry.kind == "file" and _CREATURE_PNG_RE.fullmatch(path)
                    ),
                    key=lambda path: path.encode("utf-8"),
                )
            )
            for path in creature_png_paths:
                if path not in image_cache:
                    image_cache[path] = _image_info(archive, by_path[path])

            all_actions = [action for manifest in manifests for action in manifest.actions]
            effective_actions = [action for action in all_actions if action.effective_declaration]
            direct_actions = [action for action in all_actions if action.alias_kind is None]
            complete_exact_actions = [
                action
                for manifest in manifests
                if manifest.complete_entity
                for action in manifest.actions
                if action.effective_declaration and action.exact_source_sequence
            ]
            referenced_paths = {
                frame.logical_path
                for action in all_actions
                for frame in action.frames
                if frame.exists
            }
            direct_missing_frames = [
                frame for action in direct_actions for frame in action.frames if not frame.exists
            ]
            creature_png_set = set(creature_png_paths)
            referenced_creature_paths = referenced_paths.intersection(creature_png_set)
            external_paths = referenced_paths.difference(creature_png_set)

            auxiliaries = tuple(
                AuxiliaryImage(
                    logical_path=path,
                    member_path=by_path[path].member_path,
                    sha256=image_cache[path].sha256,
                    size_bytes=image_cache[path].size_bytes,
                    width=image_cache[path].width,
                    height=image_cache[path].height,
                    mode=image_cache[path].mode,
                    alpha_kind=image_cache[path].alpha_kind,
                    role="unreferenced_creature_image",
                )
                for path in creature_png_paths
                if path not in referenced_creature_paths
            )
            paths_by_hash: defaultdict[str, list[str]] = defaultdict(list)
            for path in creature_png_paths:
                paths_by_hash[image_cache[path].sha256].append(path)
            duplicate_groups = tuple(
                DuplicateImageGroup(sha256=digest_value, logical_paths=tuple(paths))
                for digest_value, paths in sorted(paths_by_hash.items())
                if len(paths) > 1
            )

            role_counts = Counter(manifest.role for manifest in manifests)
            class_counts = Counter(
                manifest.entity_class for manifest in manifests if manifest.complete_entity
            )
            action_counts = Counter(action.normalized_action for action in complete_exact_actions)
            direction_counts = Counter(
                action.direction or "none" for action in complete_exact_actions
            )
            transform_counts = Counter(
                frame.transform for action in complete_exact_actions for frame in action.frames
            )
            source_mode_counts = Counter(
                image_cache[path].mode for path in referenced_paths if path in image_cache
            )
            counts = SuperTuxArchiveCounts(
                archive_members=len(entries),
                archive_files=sum(entry.kind == "file" for entry in entries),
                archive_directories=sum(entry.kind == "directory" for entry in entries),
                archive_symlinks=sum(entry.kind == "symlink" for entry in entries),
                archive_compressed_bytes=sum(entry.info.compress_size for entry in entries),
                archive_uncompressed_bytes=sum(entry.info.file_size for entry in entries),
                creature_manifests=len(manifests),
                complete_entity_manifests=role_counts["complete_entity"],
                modular_component_manifests=role_counts["modular_component"],
                effect_layer_manifests=role_counts["effect_layer"],
                deprecated_manifests=role_counts["deprecated"],
                action_declarations=len(all_actions),
                effective_actions=len(effective_actions),
                direct_image_actions=sum(action.alias_kind is None for action in all_actions),
                mirror_alias_actions=sum(action.alias_kind == "mirror" for action in all_actions),
                flip_alias_actions=sum(action.alias_kind == "flip" for action in all_actions),
                clone_alias_actions=sum(action.alias_kind == "clone" for action in all_actions),
                exact_complete_tracks=len(complete_exact_actions),
                quarantined_effective_tracks=sum(
                    not action.exact_source_sequence for action in effective_actions
                ),
                resolved_frame_occurrences=sum(len(action.frames) for action in effective_actions),
                direct_image_occurrences=sum(
                    len(action.declared_image_paths) for action in direct_actions
                ),
                unique_referenced_images=len(referenced_paths),
                missing_image_reference_occurrences=len(direct_missing_frames),
                unique_missing_images=len({frame.logical_path for frame in direct_missing_frames}),
                creature_tree_pngs=len(creature_png_paths),
                referenced_creature_tree_pngs=len(referenced_creature_paths),
                referenced_external_pngs=len(external_paths),
                unreferenced_creature_tree_pngs=len(auxiliaries),
                duplicate_creature_image_hash_groups=len(duplicate_groups),
                duplicate_creature_image_hash_excess=sum(
                    len(group.logical_paths) - 1 for group in duplicate_groups
                ),
                duplicate_action_name_excess=len(all_actions) - len(effective_actions),
                empty_effective_actions=sum(not action.frames for action in effective_actions),
                entity_class_counts=_counter_tuple(class_counts),
                action_counts=_counter_tuple(action_counts),
                direction_counts=_counter_tuple(direction_counts),
                transform_counts=_counter_tuple(transform_counts),
                source_mode_counts=_counter_tuple(source_mode_counts),
            )

            rights_documents = {
                path: _evidence_document(archive, by_path, path) for path in _RIGHTS_EVIDENCE_PATHS
            }
            rights = RightsAudit(
                repository_license_expression="GPL-3.0",
                license_basis=(
                    "GitHub repository declaration plus pinned LICENSE.txt; README.md states that "
                    "most of data/ is also CC-by-SA without making that a per-file grant"
                ),
                root_license=rights_documents["LICENSE.txt"],
                readme=rights_documents["README.md"],
                authors=rights_documents["data/AUTHORS"],
                credits=rights_documents["data/credits.stxt"],
                attribution_summary=(
                    "data/AUTHORS credits most graphics as of 0.7 to Rustybox, Eauix, WeLuvGoatz, "
                    "FrostC, FilipOK, and Bruhmoent and directs readers to history for details."
                ),
                caveat=(
                    "No creature-tree per-PNG license or author manifest exists. Preserve project "
                    "evidence and commit history; do not upgrade broad project notes to file-level "
                    "authorship or assume the README's secondary CC-by-SA note applies uniformly."
                ),
            )
            engine_evidence = tuple(
                _evidence_document(archive, by_path, path) for path in _ENGINE_EVIDENCE_PATHS
            )
            acquisition_evidence = _acquisition_evidence()
            issues = (
                AuditIssue(
                    "missing_source_image_reference",
                    counts.missing_image_reference_occurrences,
                    "Declared image occurrences absent from the exact commit remain quarantined.",
                ),
                AuditIssue(
                    "duplicate_action_name",
                    counts.duplicate_action_name_excess,
                    "A later declaration replaces the same action name under engine semantics.",
                ),
                AuditIssue(
                    "empty_effective_action",
                    counts.empty_effective_actions,
                    "Effective actions resolving to zero source surfaces are quarantined.",
                ),
                AuditIssue(
                    "deprecated_manifest",
                    counts.deprecated_manifests,
                    "Deprecated manifests stay in inventory and are never projected as "
                    "current data.",
                ),
                AuditIssue(
                    "unreferenced_creature_image",
                    counts.unreferenced_creature_tree_pngs,
                    "Creature-tree PNGs not consumed by manifests remain auxiliary evidence.",
                ),
                AuditIssue(
                    "duplicate_creature_image_payload",
                    counts.duplicate_creature_image_hash_excess,
                    "Byte-identical files require provenance-aware, identity-safe deduplication.",
                ),
            )
            manifest_tuple = tuple(manifests)
            payload = _audit_payload_without_hash(
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                inventory_sha256=inventory_sha256,
                archive_root=root,
                counts=counts,
                manifests=manifest_tuple,
                auxiliary_images=auxiliaries,
                duplicate_image_groups=duplicate_groups,
                rights=rights,
                acquisition_evidence=acquisition_evidence,
                engine_evidence=engine_evidence,
                symlinks=symlinks,
                issues=issues,
            )
            audit_record_sha256 = _canonical_hash(payload)
            return SuperTuxArchiveAudit(
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                inventory_sha256=inventory_sha256,
                repository_url=SUPERTUX_REPOSITORY_URL,
                commit=SUPERTUX_COMMIT,
                commit_url=SUPERTUX_COMMIT_URL,
                archive_url=SUPERTUX_ARCHIVE_URL,
                archive_root=root,
                counts=counts,
                manifests=manifest_tuple,
                auxiliary_images=auxiliaries,
                duplicate_image_groups=duplicate_groups,
                rights=rights,
                acquisition_evidence=acquisition_evidence,
                engine_evidence=engine_evidence,
                symlinks=symlinks,
                issues=issues,
                projection_policy=tuple(payload["projection_policy"]),
                audit_record_sha256=audit_record_sha256,
            )
    except BadZipFile as error:
        raise SuperTuxArchiveError(f"not a valid ZIP archive: {archive_path}") from error


def audit_known_supertux_archive(archive_path: Path) -> SuperTuxArchiveAudit:
    """Hash-check and audit the exact pinned SuperTux snapshot."""

    archive_path = Path(archive_path)
    digest = _sha256_file(archive_path)
    if digest != EXPECTED_SUPERTUX_ARCHIVE_SHA256:
        raise SuperTuxArchiveError(
            "SuperTux archive SHA-256 mismatch: "
            f"expected {EXPECTED_SUPERTUX_ARCHIVE_SHA256}, got {digest}"
        )
    audit = audit_supertux_archive(archive_path, archive_sha256=digest)
    if audit.archive_root != _EXPECTED_ROOT:
        raise SuperTuxArchiveError(
            f"SuperTux archive root mismatch: expected {_EXPECTED_ROOT!r}, "
            f"got {audit.archive_root!r}"
        )
    if audit.inventory_sha256 != EXPECTED_SUPERTUX_INVENTORY_SHA256:
        raise SuperTuxArchiveError(
            "SuperTux central-directory inventory mismatch: expected "
            f"{EXPECTED_SUPERTUX_INVENTORY_SHA256}, got {audit.inventory_sha256}"
        )
    if audit.audit_record_sha256 != EXPECTED_SUPERTUX_AUDIT_RECORD_SHA256:
        raise SuperTuxArchiveError(
            "SuperTux canonical audit record mismatch: expected "
            f"{EXPECTED_SUPERTUX_AUDIT_RECORD_SHA256}, got {audit.audit_record_sha256}"
        )
    return audit


def known_supertux_cas_path(raw_root: Path) -> Path:
    """Return the immutable CAS path for the pinned archive digest."""

    digest = EXPECTED_SUPERTUX_ARCHIVE_SHA256
    return Path(raw_root) / "objects" / "sha256" / digest[:2] / digest[2:4] / digest
