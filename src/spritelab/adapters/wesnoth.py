"""Read-only, exact-snapshot audit for Battle for Wesnoth unit animations.

The adapter reads WML and referenced images directly from a repository ZIP in
the content-addressed store.  It deliberately does not run the WML
preprocessor: literal animation declarations are retained exactly, while
macro-generated or conditional runtime structure is marked as such.  This is
safer than guessing the result of campaign-dependent preprocessing.

The pinned engine source is also part of the evidence surface.  In this
snapshot ``standing_anim`` is forced to cycle, other particles default to not
cycling, image timelines default to one millisecond per image when no duration
is supplied, and horizontal flipping defaults to enabled.  Those rules are
represented explicitly in the audit records below.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import stat
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile, ZipInfo

EXPECTED_WESNOTH_ARCHIVE_SHA256 = "fd10c38abfe3406fbc1e4dfdbc03762c576e5c9376173a7f09120040cbccba3e"
WESNOTH_COMMIT = "52858e8fa4ae3c0427f5ad12ec11cfdf22fe2b2b"
WESNOTH_REPOSITORY_URL = "https://github.com/wesnoth/wesnoth"
WESNOTH_COMMIT_URL = f"{WESNOTH_REPOSITORY_URL}/tree/{WESNOTH_COMMIT}"
_EXPECTED_ROOT = f"wesnoth-{WESNOTH_COMMIT}"

_UNIT_CONFIG_MARKER = "/units/"
_IMAGE_EXTENSIONS = (".png", ".webp", ".jpg", ".jpeg")
_FRAME_TAG_RE = re.compile(r"^(?:(?P<prefix>[A-Za-z0-9_]+)_)?frame$")
_TAG_RE = re.compile(r"^\s*\[\s*(?P<close>/)?(?P<tag>\+?[A-Za-z0-9_]+)\s*\]\s*$")
_ASSIGNMENT_RE = re.compile(r"^\s*(?P<key>[A-Za-z0-9_,]+)\s*=\s*(?P<value>.*)$")
_MACRO_RE = re.compile(r"\{(?P<body>[^{}\r\n]+)\}")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_SIMPLE_IMAGE_RE = re.compile(r"^[A-Za-z0-9_./@+\-]+$")

_ANIMATION_TAGS = frozenset(
    {
        "animation",
        "attack_anim",
        "death",
        "defend",
        "healed_anim",
        "healing_anim",
        "idle_anim",
        "leading_anim",
        "levelin_anim",
        "levelout_anim",
        "movement_anim",
        "poison_anim",
        "post_movement_anim",
        "post_teleport_anim",
        "pre_movement_anim",
        "pre_teleport_anim",
        "recruit_anim",
        "recruiting_anim",
        "resistance_anim",
        "standing_anim",
        "teaching_anim",
        "victory_anim",
    }
)

_ACTION_MAP: Mapping[str, str] = {
    "attack_anim": "attack",
    "death": "death",
    "defend": "defend",
    "healed_anim": "heal",
    "healing_anim": "heal",
    "idle_anim": "idle",
    "leading_anim": "emote",
    "levelin_anim": "transform",
    "levelout_anim": "transform",
    "movement_anim": "move",
    "poison_anim": "hurt",
    "post_movement_anim": "move_transition",
    "post_teleport_anim": "teleport",
    "pre_movement_anim": "move_transition",
    "pre_teleport_anim": "teleport",
    "recruit_anim": "spawn",
    "recruiting_anim": "emote",
    "resistance_anim": "defend",
    "standing_anim": "idle",
    "teaching_anim": "emote",
    "victory_anim": "emote",
}

_GENERIC_APPLY_TO_ACTION: Mapping[str, str] = {
    "attack": "attack",
    "death": "death",
    "defend": "defend",
    "healed": "heal",
    "healing": "heal",
    "idling": "idle",
    "leading": "emote",
    "levelin": "transform",
    "levelout": "transform",
    "movement": "move",
    "poisoned": "hurt",
    "post_movement": "move_transition",
    "post_teleport": "teleport",
    "pre_movement": "move_transition",
    "pre_teleport": "teleport",
    "recruited": "spawn",
    "recruiting": "emote",
    "resistance": "defend",
    "standing": "idle",
    "teaching": "emote",
    "victory": "emote",
}

_ANIMAL_RACES = frozenset(
    {
        "bat",
        "falcon",
        "gryphon",
        "horse",
        "rat",
        "wolf",
    }
)
_HUMANOID_RACES = frozenset(
    {
        "dunefolk",
        "dwarf",
        "elf",
        "goblin",
        "human",
        "orc",
    }
)
_CREATURE_RACES = frozenset({"drake", "lizard", "merman", "naga", "ogre", "saurian", "troll"})
_VEHICLE_WORDS = ("boat", "caravan", "cart", "ship", "wagon")


class WesnothArchiveError(ValueError):
    """Raised when an archive is not a safe Wesnoth repository ZIP."""


class WesnothParseError(ValueError):
    """Raised when a literal WML expression is structurally invalid."""


@dataclass(frozen=True)
class SourceLocation:
    config_path: str
    member_path: str
    line_number: int


@dataclass(frozen=True)
class WmlAttribute:
    name: str
    value: str
    location: SourceLocation


@dataclass(frozen=True)
class ImageResolution:
    logical_path: str | None
    selected_member_path: str | None
    resolution_basis: str
    candidate_member_paths: tuple[str, ...]
    sha256: str | None
    width: int | None
    height: int | None


@dataclass(frozen=True)
class ExpandedImageFrame:
    ordinal_in_expression: int
    source_expression: str
    logical_path: str | None
    inline_modifiers: str | None
    separate_image_mod: str | None
    duration_milliseconds: int | None
    resolution: ImageResolution
    exact_timing: bool
    lossless_source_pixels: bool
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FrameDeclaration:
    frame_tag: str
    render_role: str
    image_attribute: str
    expression: str
    location: SourceLocation
    raw_attributes: tuple[WmlAttribute, ...]
    context_attributes: tuple[WmlAttribute, ...]
    branch_path: tuple[str, ...]
    directions: tuple[str, ...]
    start_time_literal: str | None
    duration_literal: str | None
    begin_literal: str | None
    end_literal: str | None
    layer_literal: str | None
    offset_literal: str | None
    x_literal: str | None
    y_literal: str | None
    directional_x_literal: str | None
    directional_y_literal: str | None
    auto_hflip_literal: str | None
    effective_auto_hflip: bool
    auto_vflip_literal: str | None
    effective_auto_vflip: bool
    primary_literal: str | None
    frames: tuple[ExpandedImageFrame, ...]
    declaration_exact: bool
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True)
class AnimationRecord:
    source_tag: str
    variant_path: tuple[str, ...]
    normalized_action: str | None
    normalized_action_basis: str
    location: SourceLocation
    raw_attributes: tuple[WmlAttribute, ...]
    apply_to_literal: str | None
    attack_name_filters: tuple[str, ...]
    attack_range_filters: tuple[str, ...]
    directions: tuple[str, ...]
    start_time_literal: str | None
    cycles_literal: str | None
    effective_cycles: bool
    loop_mode: Literal["loop", "one_shot"]
    loop_basis: str
    macro_invocations: tuple[str, ...]
    frame_declarations: tuple[FrameDeclaration, ...]
    primary_timeline_exact: bool
    safe_primary_source_sequence: bool
    primary_frame_count: int
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True)
class EntityRecord:
    unit_id: str
    name_literal: str | None
    race_literal: str | None
    entity_class: str
    entity_class_basis: str
    config_path: str
    member_path: str
    location: SourceLocation
    raw_attributes: tuple[WmlAttribute, ...]
    base_unit_ids: tuple[str, ...]
    base_image_literal: str | None
    profile_literal: str | None
    macro_invocations: tuple[str, ...]
    animations: tuple[AnimationRecord, ...]
    unresolved_inheritance: bool
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ImageInventoryCounts:
    unit_sprite_pngs: int
    portrait_images: int
    effect_images: int
    terrain_images: int
    ui_images: int
    map_or_story_images: int
    other_images: int


@dataclass(frozen=True)
class RightsEvidence:
    member_path: str
    sha256: str
    evidence_scope: str
    finding: str


@dataclass(frozen=True)
class RightsAudit:
    repository_license_expression: str
    art_scope_statement: str
    per_asset_license: None
    per_asset_attribution: None
    projection_status: str
    copyrights_csv_rows: int
    copyrights_csv_image_rows: int
    evidence: tuple[RightsEvidence, ...]


@dataclass(frozen=True)
class AuditIssue:
    code: str
    count: int
    detail: str


@dataclass(frozen=True)
class WesnothArchiveCounts:
    archive_members: int
    archive_files: int
    cfg_files: int
    unit_cfg_files: int
    png_files: int
    image_files: int
    unit_type_declarations: int
    entity_records: int
    unique_unit_ids: int
    duplicate_unit_id_groups: int
    duplicate_unit_id_excess: int
    unresolved_entity_ids: int
    base_unit_inheritances: int
    animation_records: int
    variant_animation_records: int
    looping_animation_records: int
    one_shot_animation_records: int
    animations_with_primary_frames: int
    safe_primary_animations: int
    quarantined_primary_animations: int
    primary_frame_declarations: int
    expanded_primary_frames: int
    resolved_primary_frames: int
    unresolved_primary_frames: int
    transformed_primary_frames: int
    safe_primary_frame_occurrences: int
    unique_resolved_primary_image_members: int
    unique_safe_primary_image_members: int
    auxiliary_frame_declarations: int
    conditional_animations: int
    macro_affected_animations: int
    action_counts: tuple[tuple[str, int], ...]
    animation_tag_counts: tuple[tuple[str, int], ...]
    entity_class_counts: tuple[tuple[str, int], ...]
    image_inventory: ImageInventoryCounts


@dataclass(frozen=True)
class WesnothArchiveAudit:
    archive_sha256: str
    archive_size_bytes: int
    repository_url: str
    commit: str
    commit_url: str
    archive_root: str
    counts: WesnothArchiveCounts
    entities: tuple[EntityRecord, ...]
    rights: RightsAudit
    engine_evidence_paths: tuple[str, ...]
    issues: tuple[AuditIssue, ...]
    projection_policy: tuple[str, ...]
    audit_record_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)

    def canonical_json(self) -> str:
        """Serialize deterministically for evidence and regression checks."""

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass
class _Node:
    tag: str
    line_number: int
    end_line_number: int | None = None
    attrs: list[tuple[str, str, int]] = field(default_factory=list)
    children: list[_Node] = field(default_factory=list)
    parent: _Node | None = None

    def attr(self, key: str) -> str | None:
        values = [value for name, value, _ in self.attrs if name == key]
        return values[-1] if values else None

    def child_nodes(self, tag: str | None = None) -> tuple[_Node, ...]:
        if tag is None:
            return tuple(self.children)
        return tuple(child for child in self.children if child.tag == tag)


@dataclass(frozen=True)
class _ArchiveMember:
    logical_path: str
    member_path: str
    info: ZipInfo


@dataclass(frozen=True)
class _ExpandedExpression:
    logical_path: str | None
    inline_modifiers: str | None
    duration_milliseconds: int | None
    exact_timing: bool
    issues: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalize_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise WesnothArchiveError(f"unsafe archive member path: {name!r}")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise WesnothArchiveError(f"unsafe archive member path: {name!r}")
    return pure.as_posix()


def _validate_members(infos: Sequence[ZipInfo]) -> tuple[str, tuple[_ArchiveMember, ...]]:
    seen: set[str] = set()
    roots: set[str] = set()
    prepared: list[tuple[str, ZipInfo]] = []
    for info in infos:
        normalized = _normalize_member_name(info.filename)
        if normalized in seen:
            raise WesnothArchiveError(f"duplicate archive member: {normalized}")
        seen.add(normalized)
        roots.add(PurePosixPath(normalized).parts[0])
        if info.flag_bits & 0x1:
            raise WesnothArchiveError(f"encrypted archive member: {normalized}")
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise WesnothArchiveError(f"non-regular archive member: {normalized}")
        prepared.append((normalized, info))
    if len(roots) != 1:
        raise WesnothArchiveError(f"expected one archive root, found {sorted(roots)!r}")
    root = next(iter(roots))
    members: list[_ArchiveMember] = []
    prefix = f"{root}/"
    for normalized, info in prepared:
        if info.is_dir() or normalized == root:
            continue
        if not normalized.startswith(prefix):
            raise WesnothArchiveError(f"member is outside root {root!r}: {normalized!r}")
        logical = normalized[len(prefix) :]
        members.append(_ArchiveMember(logical_path=logical, member_path=normalized, info=info))
    members.sort(key=lambda member: member.logical_path)
    return root, tuple(members)


def _decode_wml(payload: bytes, path: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise WesnothParseError(f"{path}: WML is not UTF-8") from exc


def _strip_comment(line: str) -> str:
    quoted = False
    escaped = False
    result: list[str] = []
    for char in line:
        if escaped:
            result.append(char)
            escaped = False
            continue
        if char == "\\" and quoted:
            result.append(char)
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            result.append(char)
            continue
        if char == "#" and not quoted:
            break
        result.append(char)
    return "".join(result).rstrip()


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace(r"\"", '"')
    return value


def _parse_wml(text: str, config_path: str) -> tuple[_Node, tuple[str, ...]]:
    root = _Node(tag="__root__", line_number=1)
    stack = [root]
    in_definition = 0
    macro_invocations: list[str] = []
    lines = text.splitlines()
    for line_number, raw_line in enumerate(lines, 1):
        stripped_raw = raw_line.strip()
        if stripped_raw.startswith("#define "):
            in_definition += 1
            continue
        if stripped_raw.startswith("#enddef"):
            in_definition = max(0, in_definition - 1)
            continue
        if in_definition:
            continue
        line = _strip_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        for match in _MACRO_RE.finditer(line):
            macro_invocations.append(match.group("body").strip())
        tag_match = _TAG_RE.match(line)
        if tag_match:
            tag = tag_match.group("tag").lstrip("+")
            if tag_match.group("close"):
                if len(stack) == 1 or stack[-1].tag != tag:
                    # Raw WML can contain preprocessor-dependent closing tags.  Keep
                    # the rest of this document auditable instead of fabricating a
                    # tree across the mismatch.
                    continue
                stack[-1].end_line_number = line_number
                stack.pop()
            else:
                node = _Node(tag=tag, line_number=line_number, parent=stack[-1])
                stack[-1].children.append(node)
                stack.append(node)
            continue
        assignment = _ASSIGNMENT_RE.match(line)
        if assignment and len(stack) > 1:
            value = _unquote(assignment.group("value"))
            stack[-1].attrs.append((assignment.group("key"), value, line_number))
    for node in stack[1:]:
        node.end_line_number = len(lines)
    root.end_line_number = len(lines)
    return root, tuple(sorted(set(macro_invocations)))


def _descendants(node: _Node, tag: str | None = None) -> Iterable[_Node]:
    for child in node.children:
        if tag is None or child.tag == tag:
            yield child
        yield from _descendants(child, tag)


def _ancestor_nodes(node: _Node) -> Iterable[_Node]:
    current = node.parent
    while current is not None:
        yield current
        current = current.parent


def _node_text(lines: Sequence[str], node: _Node) -> str:
    end = node.end_line_number or node.line_number
    return "\n".join(lines[node.line_number - 1 : end])


def _macro_invocations(text: str) -> tuple[str, ...]:
    return tuple(sorted({match.group("body").strip() for match in _MACRO_RE.finditer(text)}))


def _split_top_level(value: str, delimiter: str = ",") -> tuple[str, ...]:
    result: list[str] = []
    current: list[str] = []
    square = 0
    round_depth = 0
    quoted = False
    escaped = False
    for char in value:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\" and quoted:
            current.append(char)
            escaped = True
            continue
        if char == '"':
            quoted = not quoted
            current.append(char)
            continue
        if not quoted:
            if char == "[":
                square += 1
            elif char == "]":
                square -= 1
            elif char == "(":
                round_depth += 1
            elif char == ")":
                round_depth -= 1
            elif char == delimiter and square == 0 and round_depth == 0:
                result.append("".join(current).strip())
                current = []
                continue
        current.append(char)
    result.append("".join(current).strip())
    return tuple(part for part in result if part)


def _find_top_level_duration_colon(value: str) -> int | None:
    square = 0
    round_depth = 0
    candidate: int | None = None
    for index, char in enumerate(value):
        if char == "[":
            square += 1
        elif char == "]":
            square -= 1
        elif char == "(":
            round_depth += 1
        elif char == ")":
            round_depth -= 1
        elif char == ":" and square == 0 and round_depth == 0:
            candidate = index
    return candidate


def _expand_range_term(term: str) -> tuple[str, ...]:
    term = term.strip()
    if "*" in term and "~" not in term:
        pieces = tuple(piece.strip() for piece in term.split("*"))
        if len(pieces) != 2 or not _INTEGER_RE.match(pieces[1]):
            raise WesnothParseError(f"unsupported WML repetition term: {term!r}")
        repeat = int(pieces[1])
        if repeat < 1:
            raise WesnothParseError(f"invalid WML repetition term: {term!r}")
        return tuple(pieces[0] for _ in range(repeat))
    if "~" not in term:
        return (term,)
    pieces = term.split("~")
    if len(pieces) != 2 or not all(_INTEGER_RE.match(piece.strip()) for piece in pieces):
        raise WesnothParseError(f"unsupported WML range term: {term!r}")
    start_text, end_text = (piece.strip() for piece in pieces)
    start = int(start_text)
    end = int(end_text)
    step = 1 if end >= start else -1
    start_unsigned = start_text.lstrip("+-")
    end_unsigned = end_text.lstrip("+-")
    width = max(
        len(start_unsigned) if len(start_unsigned) > 1 and start_unsigned.startswith("0") else 0,
        len(end_unsigned) if len(end_unsigned) > 1 and end_unsigned.startswith("0") else 0,
    )
    values: list[str] = []
    for value in range(start, end + step, step):
        sign = "-" if value < 0 else ""
        digits = str(abs(value)).zfill(width) if width else str(abs(value))
        values.append(f"{sign}{digits}")
    return tuple(values)


def _expand_bracket_groups(path: str) -> tuple[str, ...]:
    groups: list[tuple[int, int, tuple[str, ...]]] = []
    depth = 0
    start: int | None = None
    for index, char in enumerate(path):
        if char == "[":
            if depth == 0:
                start = index
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise WesnothParseError(f"unmatched WML image expansion: {path!r}")
            if depth == 0 and start is not None:
                terms = _split_top_level(path[start + 1 : index])
                values = tuple(value for term in terms for value in _expand_range_term(term))
                groups.append((start, index, values))
                start = None
    if depth:
        raise WesnothParseError(f"unclosed WML image expansion: {path!r}")
    if not groups:
        return (path,)
    cardinalities = {len(values) for _, _, values in groups}
    if len(cardinalities) != 1:
        raise WesnothParseError(f"mismatched WML expansion cardinalities: {path!r}")
    cardinality = next(iter(cardinalities))
    expanded: list[str] = []
    for ordinal in range(cardinality):
        cursor = 0
        pieces: list[str] = []
        for left, right, values in groups:
            pieces.append(path[cursor:left])
            pieces.append(values[ordinal])
            cursor = right + 1
        pieces.append(path[cursor:])
        expanded.append("".join(pieces))
    return tuple(expanded)


def _parse_duration_list(value: str) -> tuple[int, ...]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    durations: list[int] = []
    for term in _split_top_level(value):
        pieces = tuple(piece.strip() for piece in term.split("*"))
        if len(pieces) == 1 and _INTEGER_RE.match(pieces[0]):
            duration = int(pieces[0])
            repeat = 1
        elif len(pieces) == 2 and all(_INTEGER_RE.match(piece) for piece in pieces):
            duration = int(pieces[0])
            repeat = int(pieces[1])
        else:
            raise WesnothParseError(f"unsupported duration term: {term!r}")
        if duration < 0 or repeat < 1:
            raise WesnothParseError(f"invalid duration term: {term!r}")
        durations.extend([max(duration, 1)] * repeat)
    if not durations:
        raise WesnothParseError("empty duration list")
    return tuple(durations)


def _split_inline_modifiers(value: str) -> tuple[str, str | None]:
    square = 0
    for index, char in enumerate(value):
        if char == "[":
            square += 1
        elif char == "]":
            square -= 1
        elif char == "~" and square == 0:
            return value[:index], value[index:]
    return value, None


def expand_image_expression(
    expression: str,
    *,
    explicit_duration_literal: str | None = None,
    begin_literal: str | None = None,
    end_literal: str | None = None,
) -> tuple[_ExpandedExpression, ...]:
    """Expand the literal subset of Wesnoth's progressive-image syntax.

    Unsupported variables and malformed duration cardinalities remain explicit
    unresolved records; the function never guesses a filename or timing.
    """

    expression = _unquote(expression).strip()
    if not expression:
        return ()
    if '","' in expression:
        # Some old internal WML uses a single assignment with individually
        # quoted comma-separated values.  Once the outer quote is removed,
        # these separators are literal and can be normalized safely.
        expression = expression.replace('","', ",")
    explicit_duration: int | None = None
    explicit_duration_issue: str | None = None
    if explicit_duration_literal is not None:
        try:
            parsed = _parse_duration_list(explicit_duration_literal)
            if len(parsed) != 1:
                raise WesnothParseError("frame duration must be a single value")
            explicit_duration = parsed[0]
        except WesnothParseError:
            explicit_duration_issue = "unsupported_explicit_duration"
    elif end_literal is not None:
        if _INTEGER_RE.match(end_literal.strip()) and (
            begin_literal is None or _INTEGER_RE.match(begin_literal.strip())
        ):
            begin = int(begin_literal or "0")
            explicit_duration = max(int(end_literal) - begin, 1)
        else:
            explicit_duration_issue = "unsupported_begin_end_duration"

    # Each temporary item is path, modifiers, inline duration, and issues.
    prepared: list[tuple[str | None, str | None, int | None, tuple[str, ...]]] = []
    for source_part in _split_top_level(expression):
        colon = _find_top_level_duration_colon(source_part)
        duration_text: str | None = None
        image_text = source_part
        if colon is not None:
            possible_duration = source_part[colon + 1 :].strip()
            try:
                _parse_duration_list(possible_duration)
            except WesnothParseError:
                pass
            else:
                image_text = source_part[:colon].strip()
                duration_text = possible_duration
        base_expression, modifiers = _split_inline_modifiers(_unquote(image_text.strip()))
        issues: list[str] = []
        if any(token in base_expression for token in ("{", "}", "$")):
            paths: tuple[str | None, ...] = (None,)
            issues.append("unexpanded_image_variable_or_macro")
        else:
            try:
                paths = _expand_bracket_groups(base_expression)
            except WesnothParseError:
                paths = (None,)
                issues.append("unsupported_image_expansion")
        inline_durations: tuple[int | None, ...]
        if duration_text is not None:
            try:
                parsed_durations = _parse_duration_list(duration_text)
            except WesnothParseError:
                parsed_durations = ()
            if len(parsed_durations) == 1:
                inline_durations = tuple(parsed_durations[0] for _ in paths)
            elif len(parsed_durations) == len(paths):
                inline_durations = parsed_durations
            else:
                inline_durations = tuple(None for _ in paths)
                issues.append("image_duration_cardinality_mismatch")
        else:
            inline_durations = tuple(None for _ in paths)
        for path, inline_duration in zip(paths, inline_durations, strict=True):
            path_issues = list(issues)
            logical: str | None = None
            if path is not None:
                logical = path.strip().replace("\\", "/")
                if logical.startswith("../") or "/../" in logical or logical.startswith("/"):
                    path_issues.append("unsafe_logical_image_path")
                    logical = None
                elif not _SIMPLE_IMAGE_RE.match(logical):
                    path_issues.append("unsupported_image_path_literal")
                    logical = None
            prepared.append((logical, modifiers, inline_duration, tuple(sorted(set(path_issues)))))

    if not prepared:
        return ()
    if explicit_duration_issue:
        prepared = [
            (path, modifiers, inline_duration, tuple(sorted((*issues, explicit_duration_issue))))
            for path, modifiers, inline_duration, issues in prepared
        ]
    total_specified = sum(inline or 0 for _, _, inline, _ in prepared)
    if explicit_duration is not None:
        # This follows progressive_single in src/units/frame_private.hpp: the
        # residual duration is divided by *all* expanded items, then clamped to
        # one millisecond.  Inline durations still override that time chunk.
        unspecified_chunk = max((explicit_duration - total_specified) // len(prepared), 1)
    else:
        unspecified_chunk = 1

    output: list[_ExpandedExpression] = []
    for logical, modifiers, inline_duration, issues in prepared:
        timing_issue = any(
            issue
            in {
                "unsupported_explicit_duration",
                "unsupported_begin_end_duration",
                "image_duration_cardinality_mismatch",
            }
            for issue in issues
        )
        duration = inline_duration if inline_duration is not None else unspecified_chunk
        output.append(
            _ExpandedExpression(
                logical_path=logical,
                inline_modifiers=modifiers,
                duration_milliseconds=None if timing_issue else duration,
                exact_timing=not timing_issue,
                issues=issues,
            )
        )
    return tuple(output)


def _png_dimensions(payload: bytes) -> tuple[int | None, int | None]:
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        return None, None
    width, height = struct.unpack(">II", payload[16:24])
    return width, height


def _image_index(members: Sequence[_ArchiveMember]) -> dict[str, tuple[_ArchiveMember, ...]]:
    index: dict[str, list[_ArchiveMember]] = defaultdict(list)
    for member in members:
        lower = member.logical_path.lower()
        if not lower.endswith(_IMAGE_EXTENSIONS):
            continue
        marker = "/images/"
        position = lower.find(marker)
        if position >= 0:
            logical = member.logical_path[position + len(marker) :]
            index[logical].append(member)
        index[member.logical_path].append(member)
    return {
        key: tuple(sorted(value, key=lambda item: item.logical_path))
        for key, value in index.items()
    }


def _campaign_scope(config_path: str) -> str | None:
    match = re.match(r"^data/campaigns/([^/]+)/", config_path)
    return match.group(1) if match else None


def _resolve_image(
    logical_path: str | None,
    *,
    config_path: str,
    index: Mapping[str, tuple[_ArchiveMember, ...]],
    archive: ZipFile,
    payload_cache: dict[str, tuple[str, int | None, int | None]],
) -> ImageResolution:
    if logical_path is None:
        return ImageResolution(None, None, "unresolved_literal", (), None, None, None)
    logical_path = logical_path.removeprefix("./")
    preferred: list[tuple[str, str]] = []
    campaign = _campaign_scope(config_path)
    if campaign:
        preferred.append(
            (f"data/campaigns/{campaign}/images/{logical_path}", "campaign_binary_path")
        )
    preferred.append((f"data/core/images/{logical_path}", "core_binary_path"))
    preferred.append((logical_path, "repository_relative"))
    all_candidates = index.get(logical_path, ())
    selected: _ArchiveMember | None = None
    basis = "missing"
    for preferred_path, preferred_basis in preferred:
        matches = tuple(
            member
            for member in index.get(preferred_path, ())
            if member.logical_path == preferred_path
        )
        if len(matches) == 1:
            selected = matches[0]
            basis = preferred_basis
            break
        if len(matches) > 1:
            basis = "ambiguous_duplicate_member"
            break
    if selected is None and basis == "missing":
        unique = {member.member_path: member for member in all_candidates}
        if len(unique) == 1:
            selected = next(iter(unique.values()))
            basis = "unique_images_suffix"
        elif len(unique) > 1:
            basis = "ambiguous_binary_path"
    candidate_paths = tuple(sorted({member.member_path for member in all_candidates}))
    if selected is None:
        return ImageResolution(logical_path, None, basis, candidate_paths, None, None, None)
    cached = payload_cache.get(selected.member_path)
    if cached is None:
        payload = archive.read(selected.member_path)
        width, height = _png_dimensions(payload)
        cached = (_sha256_bytes(payload), width, height)
        payload_cache[selected.member_path] = cached
    return ImageResolution(
        logical_path=logical_path,
        selected_member_path=selected.member_path,
        resolution_basis=basis,
        candidate_member_paths=candidate_paths,
        sha256=cached[0],
        width=cached[1],
        height=cached[2],
    )


def _directions(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def _inherited_attr(node: _Node, animation: _Node, key: str) -> str | None:
    current: _Node | None = node
    while current is not None:
        value = current.attr(key)
        if value is not None:
            return value
        if current is animation:
            break
        current = current.parent
    return None


def _wml_attributes(node: _Node, *, config_path: str, member_path: str) -> tuple[WmlAttribute, ...]:
    return tuple(
        WmlAttribute(name, value, SourceLocation(config_path, member_path, line_number))
        for name, value, line_number in node.attrs
    )


def _context_attributes(
    node: _Node,
    animation: _Node,
    *,
    config_path: str,
    member_path: str,
) -> tuple[WmlAttribute, ...]:
    chain: list[_Node] = []
    current: _Node | None = node
    while current is not None:
        chain.append(current)
        if current is animation:
            break
        current = current.parent
    attributes: list[WmlAttribute] = []
    for context_node in reversed(chain):
        attributes.extend(
            _wml_attributes(context_node, config_path=config_path, member_path=member_path)
        )
    return tuple(attributes)


def _branch_path(node: _Node, animation: _Node) -> tuple[str, ...]:
    result: list[str] = []
    current = node.parent
    while current is not None and current is not animation:
        if current.tag in {"if", "else"}:
            result.append(f"{current.tag}@{current.line_number}")
        current = current.parent
    return tuple(reversed(result))


def _inherited_directions(node: _Node, animation: _Node) -> tuple[str, ...]:
    current: _Node | None = node
    while current is not None:
        value = current.attr("direction")
        if value:
            return _directions(value)
        if current is animation:
            break
        current = current.parent
    return ()


def _render_role(frame_tag: str, image_attribute: str) -> str:
    if frame_tag == "frame" and image_attribute == "image":
        return "primary_unit"
    if "missile" in frame_tag or image_attribute == "image_diagonal":
        return "projectile"
    if image_attribute == "halo" or "halo" in frame_tag:
        return "effect_overlay"
    return "auxiliary_layer"


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"yes", "true", "on", "1"}:
        return True
    if lowered in {"no", "false", "off", "0"}:
        return False
    return default


def _frame_declarations(
    animation: _Node,
    *,
    config_path: str,
    member_path: str,
    index: Mapping[str, tuple[_ArchiveMember, ...]],
    archive: ZipFile,
    payload_cache: dict[str, tuple[str, int | None, int | None]],
) -> tuple[FrameDeclaration, ...]:
    declarations: list[FrameDeclaration] = []
    for node in _descendants(animation):
        if not _FRAME_TAG_RE.match(node.tag):
            continue
        prefix = "" if node.tag == "frame" else node.tag.removesuffix("frame")
        prefix = prefix.rstrip("_")
        image_keys = ("image", "image_diagonal", "halo")
        for image_key in image_keys:
            expression = node.attr(image_key)
            if expression is None:
                continue
            duration = node.attr("duration")
            begin = node.attr("begin")
            end = node.attr("end")
            expanded = expand_image_expression(
                expression,
                explicit_duration_literal=duration,
                begin_literal=begin,
                end_literal=end,
            )
            separate_mod = _inherited_attr(
                node, animation, "image_mod" if image_key != "halo" else "halo_mod"
            )
            frames: list[ExpandedImageFrame] = []
            declaration_reasons: set[str] = set()
            for ordinal, item in enumerate(expanded):
                resolution = _resolve_image(
                    item.logical_path,
                    config_path=config_path,
                    index=index,
                    archive=archive,
                    payload_cache=payload_cache,
                )
                reasons = set(item.issues)
                if resolution.selected_member_path is None:
                    reasons.add(f"image_{resolution.resolution_basis}")
                if item.inline_modifiers:
                    reasons.add("inline_image_path_function")
                if separate_mod:
                    reasons.add("separate_image_mod")
                lossless = not reasons and resolution.selected_member_path is not None
                declaration_reasons.update(reasons)
                frames.append(
                    ExpandedImageFrame(
                        ordinal_in_expression=ordinal,
                        source_expression=expression,
                        logical_path=item.logical_path,
                        inline_modifiers=item.inline_modifiers,
                        separate_image_mod=separate_mod,
                        duration_milliseconds=item.duration_milliseconds,
                        resolution=resolution,
                        exact_timing=item.exact_timing,
                        lossless_source_pixels=lossless,
                        quarantine_reasons=tuple(sorted(reasons)),
                    )
                )
            branch = _branch_path(node, animation)
            if branch:
                declaration_reasons.add("conditional_wml_branch")
            auto_hflip = _inherited_attr(node, animation, "auto_hflip")
            auto_vflip = _inherited_attr(node, animation, "auto_vflip")
            declarations.append(
                FrameDeclaration(
                    frame_tag=node.tag,
                    render_role=_render_role(node.tag, image_key),
                    image_attribute=image_key,
                    expression=expression,
                    location=SourceLocation(config_path, member_path, node.line_number),
                    raw_attributes=_wml_attributes(
                        node, config_path=config_path, member_path=member_path
                    ),
                    context_attributes=_context_attributes(
                        node,
                        animation,
                        config_path=config_path,
                        member_path=member_path,
                    ),
                    branch_path=branch,
                    directions=_inherited_directions(node, animation),
                    start_time_literal=animation.attr(f"{prefix + '_' if prefix else ''}start_time")
                    or animation.attr("start_time"),
                    duration_literal=duration,
                    begin_literal=begin,
                    end_literal=end,
                    layer_literal=_inherited_attr(node, animation, "layer"),
                    offset_literal=_inherited_attr(node, animation, "offset"),
                    x_literal=_inherited_attr(node, animation, "x"),
                    y_literal=_inherited_attr(node, animation, "y"),
                    directional_x_literal=_inherited_attr(node, animation, "directional_x"),
                    directional_y_literal=_inherited_attr(node, animation, "directional_y"),
                    auto_hflip_literal=auto_hflip,
                    effective_auto_hflip=_truthy(auto_hflip, default=True),
                    auto_vflip_literal=auto_vflip,
                    effective_auto_vflip=_truthy(
                        auto_vflip,
                        default=_render_role(node.tag, image_key) != "primary_unit",
                    ),
                    primary_literal=_inherited_attr(node, animation, "primary"),
                    frames=tuple(frames),
                    declaration_exact=bool(frames) and not declaration_reasons,
                    quarantine_reasons=tuple(sorted(declaration_reasons)),
                )
            )
    declarations.sort(
        key=lambda declaration: (
            declaration.location.line_number,
            declaration.frame_tag,
            declaration.image_attribute,
        )
    )
    return tuple(declarations)


def _attack_filters(animation: _Node) -> tuple[tuple[str, ...], tuple[str, ...]]:
    names: list[str] = []
    ranges: list[str] = []
    for node in _descendants(animation, "filter_attack"):
        names.extend(part.strip() for part in (node.attr("name") or "").split(",") if part.strip())
        ranges.extend(
            part.strip() for part in (node.attr("range") or "").split(",") if part.strip()
        )
    return tuple(names), tuple(ranges)


def _animation_action(node: _Node) -> tuple[str | None, str]:
    if node.tag != "animation":
        action = _ACTION_MAP.get(node.tag)
        return action, "source_tag" if action else "unmapped_source_tag"
    apply_to = node.attr("apply_to")
    if apply_to is None:
        return None, "generic_animation_without_apply_to"
    action = _GENERIC_APPLY_TO_ACTION.get(apply_to.strip().lower())
    return action, "apply_to_literal" if action else "unmapped_apply_to_literal"


def _animation_variant_path(node: _Node) -> tuple[str, ...]:
    components: list[str] = []
    current = node.parent
    while current is not None and current.tag != "unit_type":
        identity = (
            current.attr("variation_id")
            or current.attr("variation_name")
            or current.attr("id")
            or current.attr("name")
        )
        suffix = f":{identity}" if identity else ""
        components.append(f"{current.tag}{suffix}@{current.line_number}")
        current = current.parent
    return tuple(reversed(components))


def _animation_record(
    node: _Node,
    *,
    lines: Sequence[str],
    config_path: str,
    member_path: str,
    index: Mapping[str, tuple[_ArchiveMember, ...]],
    archive: ZipFile,
    payload_cache: dict[str, tuple[str, int | None, int | None]],
) -> AnimationRecord:
    declarations = _frame_declarations(
        node,
        config_path=config_path,
        member_path=member_path,
        index=index,
        archive=archive,
        payload_cache=payload_cache,
    )
    primary = tuple(
        declaration for declaration in declarations if declaration.render_role == "primary_unit"
    )
    macros = _macro_invocations(_node_text(lines, node))
    conditional = any(declaration.branch_path for declaration in primary)
    reasons: set[str] = set()
    if not primary:
        reasons.add("no_literal_primary_frames")
    if macros:
        reasons.add("unexpanded_wml_macro")
    if conditional:
        reasons.add("conditional_runtime_track")
    for declaration in primary:
        reasons.update(declaration.quarantine_reasons)
    primary_frames = tuple(frame for declaration in primary for frame in declaration.frames)
    exact_timeline = (
        bool(primary)
        and not macros
        and not conditional
        and all(declaration.declaration_exact for declaration in primary)
    )
    safe_source = exact_timeline and all(frame.lossless_source_pixels for frame in primary_frames)
    action, action_basis = _animation_action(node)
    cycles_literal = node.attr("cycles") or node.attr("frame_cycles")
    if node.tag == "standing_anim":
        cycles = True
        loop_basis = "engine_forces_standing_cycles"
    elif cycles_literal is not None:
        cycles = _truthy(cycles_literal, default=False)
        loop_basis = "literal_cycles_attribute"
    else:
        cycles = False
        loop_basis = "engine_particle_default_no_cycle"
    attack_names, attack_ranges = _attack_filters(node)
    return AnimationRecord(
        source_tag=node.tag,
        variant_path=_animation_variant_path(node),
        normalized_action=action,
        normalized_action_basis=action_basis,
        location=SourceLocation(config_path, member_path, node.line_number),
        raw_attributes=_wml_attributes(node, config_path=config_path, member_path=member_path),
        apply_to_literal=node.attr("apply_to"),
        attack_name_filters=attack_names,
        attack_range_filters=attack_ranges,
        directions=_directions(node.attr("direction")),
        start_time_literal=node.attr("start_time"),
        cycles_literal=cycles_literal,
        effective_cycles=cycles,
        loop_mode="loop" if cycles else "one_shot",
        loop_basis=loop_basis,
        macro_invocations=macros,
        frame_declarations=declarations,
        primary_timeline_exact=exact_timeline,
        safe_primary_source_sequence=safe_source,
        primary_frame_count=len(primary_frames),
        quarantine_reasons=tuple(sorted(reasons)),
    )


def _entity_class(unit_id: str, race: str | None, config_path: str) -> tuple[str, str]:
    race_value = (race or "").strip().lower()
    lower_id = unit_id.lower()
    lower_path = config_path.lower()
    if any(word in lower_id for word in _VEHICLE_WORDS):
        return "vehicle", "unit_id_vehicle_term"
    if race_value in _ANIMAL_RACES:
        return "animal", "explicit_race"
    if race_value in _HUMANOID_RACES:
        return "humanoid", "explicit_race"
    if race_value in _CREATURE_RACES:
        return "creature", "explicit_race"
    if race_value == "undead" or "/undead" in lower_path:
        return "undead", "explicit_race_or_unit_tree"
    if race_value == "mechanical" or "/mechanical" in lower_path:
        return "construct", "explicit_race_or_unit_tree"
    if "/monsters/" in lower_path or race_value == "monster":
        return "monster", "unit_tree_or_explicit_race"
    if any(word in lower_id for word in ("dummy", "gate", "wall", "egg sac")):
        return "object", "unit_id_object_term"
    return "unknown", "insufficient_literal_evidence"


def _entities_from_config(
    *,
    text: str,
    config_path: str,
    member_path: str,
    index: Mapping[str, tuple[_ArchiveMember, ...]],
    archive: ZipFile,
    payload_cache: dict[str, tuple[str, int | None, int | None]],
) -> tuple[EntityRecord, ...]:
    root, _ = _parse_wml(text, config_path)
    lines = text.splitlines()
    records: list[EntityRecord] = []
    for unit in _descendants(root, "unit_type"):
        unit_id = (unit.attr("id") or "").strip()
        unit_macros = _macro_invocations(_node_text(lines, unit))
        animation_nodes = tuple(
            node
            for node in _descendants(unit)
            if node.tag in _ANIMATION_TAGS
            and next(
                (ancestor for ancestor in _ancestor_nodes(node) if ancestor.tag == "unit_type"),
                None,
            )
            is unit
        )
        animations = tuple(
            _animation_record(
                animation_node,
                lines=lines,
                config_path=config_path,
                member_path=member_path,
                index=index,
                archive=archive,
                payload_cache=payload_cache,
            )
            for animation_node in animation_nodes
        )
        base_ids: list[str] = []
        for base in unit.child_nodes("base_unit"):
            base_ids.extend(
                part.strip() for part in (base.attr("id") or "").split(",") if part.strip()
            )
        entity_reasons: set[str] = set()
        if not unit_id or any(token in unit_id for token in ("{", "}", "$")):
            entity_reasons.add("unresolved_unit_id")
        if base_ids:
            entity_reasons.add("base_unit_inheritance_not_expanded")
        entity_class, class_basis = _entity_class(unit_id, unit.attr("race"), config_path)
        records.append(
            EntityRecord(
                unit_id=unit_id,
                name_literal=unit.attr("name"),
                race_literal=unit.attr("race"),
                entity_class=entity_class,
                entity_class_basis=class_basis,
                config_path=config_path,
                member_path=member_path,
                location=SourceLocation(config_path, member_path, unit.line_number),
                raw_attributes=_wml_attributes(
                    unit, config_path=config_path, member_path=member_path
                ),
                base_unit_ids=tuple(base_ids),
                base_image_literal=unit.attr("image"),
                profile_literal=unit.attr("profile"),
                macro_invocations=unit_macros,
                animations=animations,
                unresolved_inheritance=bool(base_ids),
                quarantine_reasons=tuple(sorted(entity_reasons)),
            )
        )
    return tuple(records)


def _image_inventory(members: Sequence[_ArchiveMember]) -> ImageInventoryCounts:
    counts: Counter[str] = Counter()
    for member in members:
        lower = member.logical_path.lower()
        if not lower.endswith(_IMAGE_EXTENSIONS):
            continue
        if "/images/units/" in lower:
            counts["unit"] += 1
        elif "/images/portraits/" in lower:
            counts["portrait"] += 1
        elif any(marker in lower for marker in ("/images/halo/", "/images/projectiles/")):
            counts["effect"] += 1
        elif "/images/terrain/" in lower:
            counts["terrain"] += 1
        elif any(
            marker in lower
            for marker in ("/images/buttons/", "/images/icons/", "/images/misc/", "/images/themes/")
        ):
            counts["ui"] += 1
        elif any(marker in lower for marker in ("/images/maps/", "/images/story/")):
            counts["map_story"] += 1
        else:
            counts["other"] += 1
    return ImageInventoryCounts(
        unit_sprite_pngs=counts["unit"],
        portrait_images=counts["portrait"],
        effect_images=counts["effect"],
        terrain_images=counts["terrain"],
        ui_images=counts["ui"],
        map_or_story_images=counts["map_story"],
        other_images=counts["other"],
    )


def _rights_audit(archive: ZipFile, by_logical: Mapping[str, _ArchiveMember]) -> RightsAudit:
    evidence_specs = (
        (
            "README.md",
            "repository_and_art_collection",
            "Repository GPL-2.0 badge; most art/music GPL-2.0-or-later, "
            "newer contributions CC-BY-SA-4.0.",
        ),
        ("COPYING", "repository", "Full GNU GPL version 2 license text."),
        ("data/COPYING.txt", "data_tree", "Full GNU GPL version 2 license text under data/."),
        (
            "copyrights.csv",
            "listed_exception_files_only",
            "File-level license/author table; absence is not a per-file license assertion.",
        ),
    )
    evidence: list[RightsEvidence] = []
    for logical, scope, finding in evidence_specs:
        member = by_logical.get(logical)
        if member is None:
            continue
        payload = archive.read(member.member_path)
        evidence.append(RightsEvidence(member.member_path, _sha256_bytes(payload), scope, finding))
    rows = 0
    image_rows = 0
    csv_member = by_logical.get("copyrights.csv")
    if csv_member is not None:
        text = archive.read(csv_member.member_path).decode("utf-8-sig", errors="strict")
        parsed = tuple(csv.DictReader(io.StringIO(text)))
        rows = len(parsed)
        image_rows = sum(
            str(row.get("File", "")).lower().endswith(_IMAGE_EXTENSIONS) for row in parsed
        )
    return RightsAudit(
        repository_license_expression="GPL-2.0-or-later",
        art_scope_statement="most GPL-2.0-or-later; newer contributions CC-BY-SA-4.0",
        per_asset_license=None,
        per_asset_attribution=None,
        projection_status="repository_scope_only_mixed_art_license_requires_asset_review",
        copyrights_csv_rows=rows,
        copyrights_csv_image_rows=image_rows,
        evidence=tuple(evidence),
    )


def _sorted_counts(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counter.items()))


def _audit_payload_without_hash(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    archive_root: str,
    counts: WesnothArchiveCounts,
    entities: tuple[EntityRecord, ...],
    rights: RightsAudit,
    issues: tuple[AuditIssue, ...],
) -> dict[str, Any]:
    return {
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "repository_url": WESNOTH_REPOSITORY_URL,
        "commit": WESNOTH_COMMIT,
        "commit_url": WESNOTH_COMMIT_URL,
        "archive_root": archive_root,
        "counts": asdict(counts),
        "entities": [asdict(entity) for entity in entities],
        "rights": asdict(rights),
        "engine_evidence_paths": [
            f"{archive_root}/src/units/animation.cpp",
            f"{archive_root}/src/units/frame.cpp",
        ],
        "issues": [asdict(issue) for issue in issues],
        "projection_policy": [
            "project only literal unit_type IDs and literal primary frame timelines",
            "require safe_primary_source_sequence and a resolved immutable image member "
            "for every frame",
            "retain source WML location, expression, timing, action basis, loop basis, "
            "and facing metadata",
            "do not project auxiliary layers as body frames",
            "quarantine macros, conditional tracks, image path functions, ambiguous "
            "binary paths, and inheritance",
            "attach repository rights evidence only; leave per-asset license and attribution null",
        ],
    }


def audit_wesnoth_archive(
    archive_path: Path,
    *,
    archive_sha256: str | None = None,
) -> WesnothArchiveAudit:
    """Audit a structurally compatible repository ZIP without extracting it."""

    archive_path = Path(archive_path)
    digest = archive_sha256 or _sha256_file(archive_path)
    try:
        with ZipFile(archive_path) as archive:
            root, members = _validate_members(archive.infolist())
            by_logical = {member.logical_path: member for member in members}
            image_index = _image_index(members)
            payload_cache: dict[str, tuple[str, int | None, int | None]] = {}
            unit_cfg_members = tuple(
                member
                for member in members
                if member.logical_path.lower().endswith(".cfg")
                and _UNIT_CONFIG_MARKER in f"/{member.logical_path.lower()}"
            )
            entities: list[EntityRecord] = []
            parse_failures: list[str] = []
            for member in unit_cfg_members:
                try:
                    text = _decode_wml(archive.read(member.member_path), member.logical_path)
                    entities.extend(
                        _entities_from_config(
                            text=text,
                            config_path=member.logical_path,
                            member_path=member.member_path,
                            index=image_index,
                            archive=archive,
                            payload_cache=payload_cache,
                        )
                    )
                except WesnothParseError:
                    parse_failures.append(member.logical_path)
            entities.sort(
                key=lambda entity: (entity.config_path, entity.location.line_number, entity.unit_id)
            )
            entity_tuple = tuple(entities)
            animations = tuple(
                animation for entity in entity_tuple for animation in entity.animations
            )
            primary_declarations = tuple(
                declaration
                for animation in animations
                for declaration in animation.frame_declarations
                if declaration.render_role == "primary_unit"
            )
            auxiliary_declarations = tuple(
                declaration
                for animation in animations
                for declaration in animation.frame_declarations
                if declaration.render_role != "primary_unit"
            )
            primary_frames = tuple(
                frame for declaration in primary_declarations for frame in declaration.frames
            )
            with_primary = tuple(
                animation for animation in animations if animation.primary_frame_count
            )
            safe_primary = tuple(
                animation for animation in with_primary if animation.safe_primary_source_sequence
            )
            unresolved_ids = sum(
                "unresolved_unit_id" in entity.quarantine_reasons for entity in entity_tuple
            )
            unit_id_counts = Counter(entity.unit_id for entity in entity_tuple)
            resolved_primary_members = {
                frame.resolution.selected_member_path
                for frame in primary_frames
                if frame.resolution.selected_member_path is not None
            }
            safe_primary_frames = tuple(
                frame
                for animation in safe_primary
                for declaration in primary_declarations_for(animation)
                for frame in declaration.frames
            )
            safe_primary_members = {
                frame.resolution.selected_member_path
                for frame in safe_primary_frames
                if frame.resolution.selected_member_path is not None
            }
            image_inventory = _image_inventory(members)
            action_counts = Counter(
                animation.normalized_action or "unknown" for animation in animations
            )
            tag_counts = Counter(animation.source_tag for animation in animations)
            class_counts = Counter(entity.entity_class for entity in entity_tuple)
            counts = WesnothArchiveCounts(
                archive_members=len(archive.infolist()),
                archive_files=len(members),
                cfg_files=sum(member.logical_path.lower().endswith(".cfg") for member in members),
                unit_cfg_files=len(unit_cfg_members),
                png_files=sum(member.logical_path.lower().endswith(".png") for member in members),
                image_files=sum(
                    member.logical_path.lower().endswith(_IMAGE_EXTENSIONS) for member in members
                ),
                unit_type_declarations=len(entity_tuple),
                entity_records=len(entity_tuple),
                unique_unit_ids=len(unit_id_counts),
                duplicate_unit_id_groups=sum(count > 1 for count in unit_id_counts.values()),
                duplicate_unit_id_excess=sum(count - 1 for count in unit_id_counts.values()),
                unresolved_entity_ids=unresolved_ids,
                base_unit_inheritances=sum(bool(entity.base_unit_ids) for entity in entity_tuple),
                animation_records=len(animations),
                variant_animation_records=sum(
                    bool(animation.variant_path) for animation in animations
                ),
                looping_animation_records=sum(
                    animation.loop_mode == "loop" for animation in animations
                ),
                one_shot_animation_records=sum(
                    animation.loop_mode == "one_shot" for animation in animations
                ),
                animations_with_primary_frames=len(with_primary),
                safe_primary_animations=len(safe_primary),
                quarantined_primary_animations=sum(
                    not animation.safe_primary_source_sequence for animation in with_primary
                ),
                primary_frame_declarations=len(primary_declarations),
                expanded_primary_frames=len(primary_frames),
                resolved_primary_frames=sum(
                    frame.resolution.selected_member_path is not None for frame in primary_frames
                ),
                unresolved_primary_frames=sum(
                    frame.resolution.selected_member_path is None for frame in primary_frames
                ),
                transformed_primary_frames=sum(
                    bool(frame.inline_modifiers or frame.separate_image_mod)
                    for frame in primary_frames
                ),
                safe_primary_frame_occurrences=len(safe_primary_frames),
                unique_resolved_primary_image_members=len(resolved_primary_members),
                unique_safe_primary_image_members=len(safe_primary_members),
                auxiliary_frame_declarations=len(auxiliary_declarations),
                conditional_animations=sum(
                    any(
                        declaration.branch_path
                        for declaration in primary_declarations_for(animation)
                    )
                    for animation in animations
                ),
                macro_affected_animations=sum(
                    bool(animation.macro_invocations) for animation in animations
                ),
                action_counts=_sorted_counts(action_counts),
                animation_tag_counts=_sorted_counts(tag_counts),
                entity_class_counts=_sorted_counts(class_counts),
                image_inventory=image_inventory,
            )
            rights = _rights_audit(archive, by_logical)
            issues = (
                AuditIssue(
                    "wml_parse_failures",
                    len(parse_failures),
                    "Unit-tree configs that could not be decoded/parsed; no partial "
                    "records projected.",
                ),
                AuditIssue(
                    "unexpanded_macros",
                    counts.macro_affected_animations,
                    "Animations containing WML macro invocations need the exact runtime "
                    "preprocessor context.",
                ),
                AuditIssue(
                    "conditional_tracks",
                    counts.conditional_animations,
                    "Primary frames under [if]/[else] are declarations, not asserted "
                    "complete timelines.",
                ),
                AuditIssue(
                    "runtime_image_transforms",
                    counts.transformed_primary_frames,
                    "Primary frames with IPFs/image_mod are not treated as lossless "
                    "source-pixel renders.",
                ),
                AuditIssue(
                    "mixed_repository_art_rights",
                    counts.resolved_primary_frames,
                    "Resolved image members inherit no guessed per-asset license or "
                    "artist attribution.",
                ),
            )
            payload = _audit_payload_without_hash(
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                archive_root=root,
                counts=counts,
                entities=entity_tuple,
                rights=rights,
                issues=issues,
            )
            record_hash = _sha256_bytes(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
            return WesnothArchiveAudit(
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                repository_url=WESNOTH_REPOSITORY_URL,
                commit=WESNOTH_COMMIT,
                commit_url=WESNOTH_COMMIT_URL,
                archive_root=root,
                counts=counts,
                entities=entity_tuple,
                rights=rights,
                engine_evidence_paths=(
                    f"{root}/src/units/animation.cpp",
                    f"{root}/src/units/frame.cpp",
                ),
                issues=issues,
                projection_policy=tuple(payload["projection_policy"]),
                audit_record_sha256=record_hash,
            )
    except BadZipFile as exc:
        raise WesnothArchiveError(f"not a valid ZIP archive: {archive_path}") from exc


def primary_declarations_for(animation: AnimationRecord) -> tuple[FrameDeclaration, ...]:
    """Return only body-frame declarations for one audited animation."""

    return tuple(
        declaration
        for declaration in animation.frame_declarations
        if declaration.render_role == "primary_unit"
    )


def audit_known_wesnoth_archive(archive_path: Path) -> WesnothArchiveAudit:
    """Hash-check and audit the exact pinned Wesnoth snapshot."""

    archive_path = Path(archive_path)
    digest = _sha256_file(archive_path)
    if digest != EXPECTED_WESNOTH_ARCHIVE_SHA256:
        raise WesnothArchiveError(
            "Wesnoth archive SHA-256 mismatch: expected "
            f"{EXPECTED_WESNOTH_ARCHIVE_SHA256}, got {digest}"
        )
    audit = audit_wesnoth_archive(archive_path, archive_sha256=digest)
    if audit.archive_root != _EXPECTED_ROOT:
        raise WesnothArchiveError(
            "Wesnoth archive root mismatch: expected "
            f"{_EXPECTED_ROOT!r}, got {audit.archive_root!r}"
        )
    return audit


def known_wesnoth_cas_path(raw_root: Path) -> Path:
    """Return the project's deterministic four-character-sharded CAS path."""

    digest = EXPECTED_WESNOTH_ARCHIVE_SHA256
    return Path(raw_root) / "objects" / "sha256" / digest[:2] / digest[2:4] / digest


__all__ = [
    "EXPECTED_WESNOTH_ARCHIVE_SHA256",
    "WESNOTH_COMMIT",
    "WESNOTH_COMMIT_URL",
    "WESNOTH_REPOSITORY_URL",
    "AnimationRecord",
    "AuditIssue",
    "EntityRecord",
    "ExpandedImageFrame",
    "FrameDeclaration",
    "ImageInventoryCounts",
    "ImageResolution",
    "RightsAudit",
    "RightsEvidence",
    "SourceLocation",
    "WesnothArchiveAudit",
    "WesnothArchiveCounts",
    "WesnothArchiveError",
    "WesnothParseError",
    "WmlAttribute",
    "audit_known_wesnoth_archive",
    "audit_wesnoth_archive",
    "expand_image_expression",
    "known_wesnoth_cas_path",
    "primary_declarations_for",
]
