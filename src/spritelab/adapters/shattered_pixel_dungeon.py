from __future__ import annotations

import ast
import hashlib
import io
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

EXPECTED_SHATTERED_PIXEL_DUNGEON_ARCHIVE_SHA256 = (
    "deed31527c24b2a500f6c79e1cf2502c4c9d8b4391529033da7015cd73c97544"
)
SHATTERED_PIXEL_DUNGEON_COMMIT = "7b8b845a76fe76c6b7c031ae9e570852411f56db"
SHATTERED_PIXEL_DUNGEON_REPOSITORY_URL = "https://github.com/00-Evan/shattered-pixel-dungeon"
SHATTERED_PIXEL_DUNGEON_COMMIT_URL = (
    f"{SHATTERED_PIXEL_DUNGEON_REPOSITORY_URL}/tree/{SHATTERED_PIXEL_DUNGEON_COMMIT}"
)

_EXPECTED_ROOT = f"shattered-pixel-dungeon-{SHATTERED_PIXEL_DUNGEON_COMMIT}"
_ASSETS_SUFFIX = "core/src/main/java/com/shatteredpixel/shatteredpixeldungeon/Assets.java"
_HERO_CLASS_SUFFIX = (
    "core/src/main/java/com/shatteredpixel/shatteredpixeldungeon/actors/hero/HeroClass.java"
)
_SPRITE_ASSET_PREFIX = "core/src/main/assets/sprites/"
_LICENSE_RELATIVE_PATH = "LICENSE.txt"
_README_RELATIVE_PATH = "README.md"

_CONTROL_WORDS = frozenset(
    {"if", "for", "while", "switch", "catch", "synchronized", "return", "new", "do"}
)
_NORMALIZED_ACTIONS: Mapping[str, str | None] = {
    "idle": "idle",
    "activeIdle": "idle",
    "run": "run",
    "attack": "attack",
    "cast": "cast",
    "kick": "attack_melee",
    "slam": "attack_melee",
    "stab": "attack_melee",
    "pumpAttack": "attack_melee",
    "zap": "attack_ranged",
    "die": "death",
    "operate": "interact",
    "read": "interact",
    "fly": "fly",
    "charging": None,
    "charge": None,
    "pump": None,
    "prep": None,
    "leap": "jump",
    "hiding": "transform",
    "advancedHiding": "transform",
    "crumple": None,
    "tierIdles": "idle",
}

type Numeric = int | float


class ShatteredPixelDungeonArchiveError(ValueError):
    """Raised when an archive cannot be audited as a Shattered Pixel Dungeon tree."""


class JavaSpriteParseError(ValueError):
    """Raised when pinned Java sprite evidence is structurally contradictory."""


@dataclass(frozen=True)
class SpriteAssetMapping:
    key: str
    relative_path: str
    evidence_member_path: str
    line_number: int


@dataclass(frozen=True)
class HeroClassSheetMapping:
    hero_class: str
    asset_key: str
    relative_path: str
    evidence_member_path: str
    line_number: int


@dataclass(frozen=True)
class SpriteSheetAudit:
    asset_key: str | None
    relative_path: str
    member_path: str
    mapped_by_assets_java: bool
    width: int
    height: int
    image_mode: str
    has_transparency: bool
    size_bytes: int
    compressed_size_bytes: int
    crc32: str
    sha256: str
    embedded_metadata_keys: tuple[str, ...]


@dataclass(frozen=True)
class FrameCell:
    frame_index: int | None
    column: int | None
    row: int | None
    left: int
    top: int
    right: int
    bottom: int
    coordinate_space: str


@dataclass(frozen=True)
class FilmAudit:
    variable_name: str | None
    constructor_expression: str
    layout_kind: str
    frame_width: int | None
    frame_height: int | None
    frame_width_expression: str | None
    frame_height_expression: str | None
    sheet_widths: tuple[int, ...]
    sheet_heights: tuple[int, ...]
    columns: tuple[int, ...]
    rows: tuple[int, ...]
    capacities: tuple[int, ...]
    source_sheet_grid_rows: tuple[int, ...]
    source_sheet_grid_capacities: tuple[int, ...]
    evidence_member_path: str
    line_number: int


@dataclass(frozen=True)
class AnimationAudit:
    source_action: str
    normalized_action: str | None
    fps_values: tuple[Numeric, ...]
    fps_expression: str | None
    looping_values: tuple[bool, ...]
    looping_expression: str | None
    frame_index_variants: tuple[tuple[int, ...], ...]
    frame_expression_order: tuple[str, ...]
    frame_variable_expressions: tuple[tuple[str, tuple[str, ...]], ...]
    frame_cell_variants: tuple[tuple[FrameCell, ...], ...]
    direct_uv_rect_variants: tuple[FrameCell, ...]
    film: FilmAudit | None
    source_asset_keys: tuple[str, ...]
    source_sheet_paths: tuple[str, ...]
    clone_of: str | None
    defined_in_class: str
    inherited: bool
    context: str
    evidence_member_path: str
    line_number: int
    frame_order_preserved: bool
    deliberate_repeats_preserved: bool
    timing_preserved: bool
    loop_semantics_preserved: bool
    geometry_valid: bool | None
    ambiguity_reasons: tuple[str, ...]

    @property
    def resolved_sequence_variant_count(self) -> int:
        if self.frame_index_variants:
            return len(self.frame_index_variants)
        return len(self.direct_uv_rect_variants)


@dataclass(frozen=True)
class SpriteClassAudit:
    class_name: str
    simple_name: str
    parent_name: str
    resolved_parent_class: str | None
    abstract: bool
    evidence_member_path: str
    line_number: int
    source_asset_keys: tuple[str, ...]
    source_sheet_paths: tuple[str, ...]
    texture_resolution: str
    animations: tuple[AnimationAudit, ...]
    entity_class: str
    entity_label: str
    morphology_tags: tuple[str, ...]
    classification_basis: str
    view: str
    direction_semantics: str
    ambiguity_reasons: tuple[str, ...]

    @property
    def action_slots(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(animation.source_action for animation in self.animations))


@dataclass(frozen=True)
class EvidenceDocument:
    relative_path: str
    member_path: str
    sha256: str
    size_bytes: int
    detected_license_identifiers: tuple[str, ...]
    scope: str
    notes: str


@dataclass(frozen=True)
class AttributionEvidence:
    name: str
    role: str
    evidence_member_path: str
    evidence_line_numbers: tuple[int, ...]
    evidence_text: str


@dataclass(frozen=True)
class EntityClassCount:
    entity_class: str
    concrete_class_count: int


@dataclass(frozen=True)
class ActionCount:
    source_action: str
    concrete_class_count: int
    action_slot_count: int
    resolved_sequence_variant_count: int
    frame_occurrence_count: int


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    related_names: tuple[str, ...]


@dataclass(frozen=True)
class ShatteredPixelDungeonCounts:
    zip_member_count: int
    archive_png_file_count: int
    java_file_count: int
    parsed_java_class_count: int
    assets_sprite_mapping_count: int
    hero_class_sheet_mapping_count: int
    sprite_png_file_count: int
    mapped_sprite_png_file_count: int
    unmapped_sprite_png_file_count: int
    missing_mapped_sprite_png_count: int
    sprite_definition_class_count: int
    concrete_sprite_class_count: int
    abstract_sprite_class_count: int
    source_animation_frame_call_count: int
    source_animation_clone_assignment_count: int
    concrete_action_slot_count: int
    resolved_sequence_variant_count: int
    unresolved_animation_count: int
    frame_occurrence_count: int
    invalid_geometry_animation_count: int
    animal_concrete_class_count: int
    quadruped_concrete_class_count: int
    evidence_document_count: int
    license_evidence_document_count: int
    sprite_png_with_embedded_attribution_count: int


@dataclass(frozen=True)
class ShatteredPixelDungeonAudit:
    archive_path: str
    archive_sha256: str
    repository_commit: str | None
    repository_url: str
    commit_url: str | None
    root_prefix: str
    assets_java_member_path: str
    counts: ShatteredPixelDungeonCounts
    asset_mappings: tuple[SpriteAssetMapping, ...]
    hero_class_sheet_mappings: tuple[HeroClassSheetMapping, ...]
    sprite_sheets: tuple[SpriteSheetAudit, ...]
    sprite_classes: tuple[SpriteClassAudit, ...]
    entity_classes: tuple[EntityClassCount, ...]
    actions: tuple[ActionCount, ...]
    evidence_documents: tuple[EvidenceDocument, ...]
    attributions: tuple[AttributionEvidence, ...]
    issues: tuple[AuditIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _CodeContext:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class _Assignment:
    name: str
    expressions: tuple[str, ...]
    position: int
    context_start: int | None


@dataclass(frozen=True)
class _RawFilm:
    variable_name: str
    constructor_expression: str
    args: tuple[str, ...]
    position: int
    context_start: int | None
    line_number: int


@dataclass(frozen=True)
class _RawAnimationInit:
    action: str
    fps_expression: str
    looping_expression: str
    position: int
    context_start: int | None
    line_number: int


@dataclass(frozen=True)
class _RawFrameCall:
    action: str
    args: tuple[str, ...]
    position: int
    context_start: int | None
    context_name: str
    line_number: int


@dataclass(frozen=True)
class _RawClone:
    action: str
    source_action: str
    position: int
    context_start: int | None
    context_name: str
    line_number: int


@dataclass(frozen=True)
class _RawClass:
    canonical_name: str
    simple_name: str
    parent_name: str
    resolved_parent_class: str | None
    abstract: bool
    member_path: str
    line_number: int
    start: int
    end: int
    direct_asset_keys: tuple[str, ...]
    dynamic_texture_expressions: tuple[str, ...]
    assignments: tuple[_Assignment, ...]
    arrays: tuple[tuple[str, tuple[str, ...]], ...]
    tex_offset_expressions: tuple[str, ...]
    films: tuple[_RawFilm, ...]
    animation_inits: tuple[_RawAnimationInit, ...]
    frame_calls: tuple[_RawFrameCall, ...]
    clones: tuple[_RawClone, ...]


@dataclass(frozen=True)
class _ActionTemplate:
    action: str
    owner_class: str
    frame_call: _RawFrameCall | None
    clone: _RawClone | None
    animation_init: _RawAnimationInit | None
    film: _RawFilm | None


def _line_number(source: str, position: int) -> int:
    return source.count("\n", 0, position) + 1


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_java(source: str) -> str:
    """Replace comments and literals with spaces while retaining offsets and newlines."""

    chars = list(source)
    index = 0
    state = "code"
    while index < len(chars):
        char = chars[index]
        next_char = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "code":
            if char == "/" and next_char == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "line_comment"
                continue
            if char == "/" and next_char == "*":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                chars[index] = " "
                index += 1
                state = "string"
                continue
            if char == "'":
                chars[index] = " "
                index += 1
                state = "char"
                continue
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                chars[index] = " "
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                chars[index] = chars[index + 1] = " "
                index += 2
                state = "code"
            else:
                if char != "\n":
                    chars[index] = " "
                index += 1
            continue
        if char == "\\":
            chars[index] = " "
            if index + 1 < len(chars) and chars[index + 1] != "\n":
                chars[index + 1] = " "
            index += 2
            continue
        delimiter = '"' if state == "string" else "'"
        if char == delimiter:
            chars[index] = " "
            state = "code"
        elif char != "\n":
            chars[index] = " "
        index += 1
    return "".join(chars)


def _matching_brace(masked: str, opening: int) -> int:
    depth = 0
    for position in range(opening, len(masked)):
        char = masked[position]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return position
    raise JavaSpriteParseError(f"unclosed Java block beginning at offset {opening}")


def _split_args(expression: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depths = {"(": 0, "[": 0, "{": 0}
    matching = {")": "(", "]": "[", "}": "{"}
    in_string: str | None = None
    escaped = False
    for index, char in enumerate(expression):
        if in_string is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == in_string:
                in_string = None
            continue
        if char in {'"', "'"}:
            in_string = char
        elif char in depths:
            depths[char] += 1
        elif char in matching:
            depths[matching[char]] -= 1
        elif char == "," and all(depth == 0 for depth in depths.values()):
            parts.append(expression[start:index].strip())
            start = index + 1
    tail = expression[start:].strip()
    if tail:
        parts.append(tail)
    return tuple(parts)


def parse_assets_sprite_mappings(
    source: str,
    *,
    member_path: str = "Assets.java",
) -> tuple[SpriteAssetMapping, ...]:
    masked = _mask_java(source)
    declaration = re.search(r"\bclass\s+Sprites\s*\{", masked)
    if declaration is None:
        raise JavaSpriteParseError(f"{member_path} has no Assets.Sprites class")
    opening = masked.index("{", declaration.start())
    closing = _matching_brace(masked, opening)
    body_source = source[opening + 1 : closing]
    body_masked = masked[opening + 1 : closing]
    pattern = re.compile(
        r"\b(?:public\s+)?(?:static\s+)?(?:final\s+)?String\s+"
        r"(?P<key>[A-Z][A-Z0-9_]*)\s*=\s*\"(?P<path>[^\"]+)\"\s*;"
    )
    # String contents are masked, so use the original body for this tightly scoped grammar.
    mappings: list[SpriteAssetMapping] = []
    seen_keys: set[str] = set()
    seen_paths: set[str] = set()
    for match in pattern.finditer(body_source):
        if not body_masked[match.start() : match.end()].strip():
            continue
        key = match.group("key")
        path = PurePosixPath(match.group("path")).as_posix()
        if not path.startswith("sprites/") or PurePosixPath(path).suffix.lower() != ".png":
            continue
        if key in seen_keys:
            raise JavaSpriteParseError(f"duplicate Assets.Sprites key {key!r}")
        if path in seen_paths:
            raise JavaSpriteParseError(f"duplicate Assets.Sprites path {path!r}")
        seen_keys.add(key)
        seen_paths.add(path)
        mappings.append(
            SpriteAssetMapping(
                key=key,
                relative_path=path,
                evidence_member_path=member_path,
                line_number=_line_number(source, opening + 1 + match.start()),
            )
        )
    if not mappings:
        # Reference body_masked so an accidental masking regression is visible to coverage tools.
        raise JavaSpriteParseError(
            f"{member_path} has no sprite mappings ({len(body_masked)} masked body characters)"
        )
    return tuple(mappings)


def parse_hero_class_sheet_mappings(
    source: str,
    asset_mappings: Sequence[SpriteAssetMapping],
    *,
    member_path: str = "HeroClass.java",
) -> tuple[HeroClassSheetMapping, ...]:
    by_key = {mapping.key: mapping for mapping in asset_mappings}
    masked = _mask_java(source)
    method = re.search(r"\bString\s+spritesheet\s*\(\s*\)\s*\{", masked)
    if method is None:
        return ()
    opening = masked.index("{", method.start())
    closing = _matching_brace(masked, opening)
    body = source[opening + 1 : closing]
    pattern = re.compile(
        r"case\s+(?P<hero>[A-Z][A-Z0-9_]*)\s*:\s*"
        r"(?:default\s*:\s*)?return\s+Assets\.Sprites\.(?P<key>[A-Z][A-Z0-9_]*)\s*;",
        re.DOTALL,
    )
    found: list[HeroClassSheetMapping] = []
    for match in pattern.finditer(body):
        key = match.group("key")
        mapping = by_key.get(key)
        if mapping is None:
            raise JavaSpriteParseError(
                f"HeroClass.spritesheet references unknown Assets.Sprites.{key}"
            )
        found.append(
            HeroClassSheetMapping(
                hero_class=match.group("hero"),
                asset_key=key,
                relative_path=mapping.relative_path,
                evidence_member_path=member_path,
                line_number=_line_number(source, opening + 1 + match.start()),
            )
        )
    return tuple(found)


def _class_spans(source: str, member_path: str) -> tuple[tuple[Any, ...], ...]:
    masked = _mask_java(source)
    pattern = re.compile(
        r"(?P<mods>(?:(?:public|protected|private|static|final|abstract)\s+)*)"
        r"\bclass\s+(?P<name>[A-Za-z_]\w*)"
        r"(?:\s*<[^{};]+>)?\s+extends\s+(?P<parent>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"
        r"[^\{;]*\{"
    )
    provisional: list[tuple[int, int, str, str, bool, int]] = []
    for match in pattern.finditer(masked):
        opening = masked.index("{", match.start(), match.end())
        closing = _matching_brace(masked, opening)
        provisional.append(
            (
                match.start(),
                closing,
                match.group("name"),
                match.group("parent").split(".")[-1],
                "abstract" in match.group("mods").split(),
                _line_number(source, match.start()),
            )
        )
    result: list[tuple[Any, ...]] = []
    for start, end, name, parent, abstract, line in provisional:
        containers = [
            candidate for candidate in provisional if candidate[0] < start and candidate[1] > end
        ]
        if containers:
            immediate = min(containers, key=lambda item: item[1] - item[0])
            outer_record = next(item for item in result if item[0] == immediate[0])
            canonical = f"{outer_record[6]}.{name}"
        else:
            canonical = name
        result.append((start, end, name, parent, abstract, line, canonical, masked))
    return tuple(result)


def _blank_nested(
    masked: str, own_start: int, own_end: int, spans: Sequence[tuple[Any, ...]]
) -> str:
    chars = list(masked)
    for span in spans:
        start, end = int(span[0]), int(span[1])
        if start > own_start and end < own_end:
            for index in range(start, end + 1):
                if chars[index] != "\n":
                    chars[index] = " "
    for index in range(0, own_start):
        if chars[index] != "\n":
            chars[index] = " "
    for index in range(own_end + 1, len(chars)):
        if chars[index] != "\n":
            chars[index] = " "
    return "".join(chars)


def _contexts(own_masked: str) -> tuple[_CodeContext, ...]:
    pattern = re.compile(
        r"(?P<name>[A-Za-z_]\w*)\s*\([^;{}]*\)\s*"
        r"(?:throws\s+[^{]+)?\{"
    )
    contexts: list[_CodeContext] = []
    for match in pattern.finditer(own_masked):
        name = match.group("name")
        if name in _CONTROL_WORDS:
            continue
        opening = own_masked.index("{", match.start(), match.end())
        try:
            closing = _matching_brace(own_masked, opening)
        except JavaSpriteParseError:
            continue
        contexts.append(_CodeContext(name=name, start=opening, end=closing))
    return tuple(contexts)


def _context_at(position: int, contexts: Sequence[_CodeContext]) -> _CodeContext | None:
    candidates = [context for context in contexts if context.start < position < context.end]
    return min(candidates, key=lambda context: context.end - context.start) if candidates else None


def _parse_raw_classes(source: str, member_path: str) -> tuple[_RawClass, ...]:
    spans = _class_spans(source, member_path)
    raw_classes: list[_RawClass] = []
    for start, end, simple, parent, abstract, line, canonical, masked in spans:
        own_masked = _blank_nested(masked, start, end, spans)
        contexts = _contexts(own_masked)

        asset_keys: list[str] = []
        dynamic_textures: list[str] = []
        texture_pattern = re.compile(r"(?<![.\w])texture\s*\((?P<expr>.*?)\)\s*;", re.DOTALL)
        for match in texture_pattern.finditer(own_masked, start, end + 1):
            expression = source[match.start("expr") : match.end("expr")].strip()
            asset_match = re.fullmatch(r"Assets\.Sprites\.([A-Z][A-Z0-9_]*)", expression)
            if asset_match:
                asset_keys.append(asset_match.group(1))
            else:
                dynamic_textures.append(re.sub(r"\s+", " ", expression))

        assignments: list[_Assignment] = []
        arrays: list[tuple[str, tuple[str, ...]]] = []
        array_pattern = re.compile(
            r"\b(?:public|protected|private|static|final|\s)*int\s*\[\s*\]\s*"
            r"(?P<name>[A-Za-z_]\w*)\s*=\s*\{(?P<values>[^}]*)}\s*;"
        )
        array_ranges: list[tuple[int, int]] = []
        for match in array_pattern.finditer(own_masked, start, end + 1):
            arrays.append(
                (
                    match.group("name"),
                    _split_args(source[match.start("values") : match.end("values")]),
                )
            )
            array_ranges.append((match.start(), match.end()))

        declaration_pattern = re.compile(
            r"\b(?:public\s+|protected\s+|private\s+|static\s+|final\s+)*"
            r"(?:int|float|double|boolean)\s+(?P<name>[A-Za-z_]\w*)\s*=\s*"
            r"(?P<expr>[^;]+);"
        )
        for match in declaration_pattern.finditer(own_masked, start, end + 1):
            if any(left <= match.start() < right for left, right in array_ranges):
                continue
            context = _context_at(match.start(), contexts)
            assignments.append(
                _Assignment(
                    name=match.group("name"),
                    expressions=(source[match.start("expr") : match.end("expr")].strip(),),
                    position=match.start(),
                    context_start=context.start if context else None,
                )
            )
        bare_pattern = re.compile(r"(?<![.\w])(?P<name>[A-Za-z_]\w*)\s*=(?!=)\s*(?P<expr>[^;]+);")
        declaration_starts = {item.position for item in assignments}
        for match in bare_pattern.finditer(own_masked, start, end + 1):
            if match.start() in declaration_starts:
                continue
            expression = source[match.start("expr") : match.end("expr")].strip()
            if expression.startswith("new ") or ".clone" in expression:
                continue
            context = _context_at(match.start(), contexts)
            assignments.append(
                _Assignment(
                    name=match.group("name"),
                    expressions=(expression,),
                    position=match.start(),
                    context_start=context.start if context else None,
                )
            )

        film_pattern = re.compile(
            r"\bTextureFilm\s+(?P<name>[A-Za-z_]\w*)\s*=\s*new\s+TextureFilm\s*"
            r"\((?P<args>.*?)\)\s*;",
            re.DOTALL,
        )
        films: list[_RawFilm] = []
        for match in film_pattern.finditer(own_masked, start, end + 1):
            context = _context_at(match.start(), contexts)
            arg_text = source[match.start("args") : match.end("args")]
            films.append(
                _RawFilm(
                    variable_name=match.group("name"),
                    constructor_expression=re.sub(r"\s+", " ", arg_text.strip()),
                    args=_split_args(arg_text),
                    position=match.start(),
                    context_start=context.start if context else None,
                    line_number=_line_number(source, match.start()),
                )
            )

        action_receiver = r"[A-Za-z_]\w*(?:\s*\[[^\]]+\])?"
        init_pattern = re.compile(
            rf"(?P<action>{action_receiver})\s*=\s*new\s+(?:MovieClip\.)?Animation\s*"
            r"\((?P<args>.*?)\)\s*;",
            re.DOTALL,
        )
        animation_inits: list[_RawAnimationInit] = []
        for match in init_pattern.finditer(own_masked, start, end + 1):
            args = _split_args(source[match.start("args") : match.end("args")])
            if len(args) != 2:
                continue
            context = _context_at(match.start(), contexts)
            animation_inits.append(
                _RawAnimationInit(
                    action=re.sub(r"\s+", "", match.group("action")),
                    fps_expression=args[0],
                    looping_expression=args[1],
                    position=match.start(),
                    context_start=context.start if context else None,
                    line_number=_line_number(source, match.start()),
                )
            )

        frame_pattern = re.compile(
            rf"(?<![.\w])(?P<action>{action_receiver})\.frames\s*"
            r"\((?P<args>.*?)\)\s*;",
            re.DOTALL,
        )
        frame_calls: list[_RawFrameCall] = []
        for match in frame_pattern.finditer(own_masked, start, end + 1):
            action = match.group("action")
            args_start, args_end = match.span("args")
            if action is None or args_start < 0 or args_end < 0:
                continue
            context = _context_at(match.start(), contexts)
            frame_calls.append(
                _RawFrameCall(
                    action=re.sub(r"\s+", "", action),
                    args=_split_args(source[args_start:args_end]),
                    position=match.start(),
                    context_start=context.start if context else None,
                    context_name=context.name if context else "<class>",
                    line_number=_line_number(source, match.start()),
                )
            )

        clone_pattern = re.compile(
            rf"(?P<action>{action_receiver})\s*=\s*"
            rf"(?P<source>{action_receiver})\.clone\s*\(\s*\)\s*;"
        )
        clones: list[_RawClone] = []
        for match in clone_pattern.finditer(own_masked, start, end + 1):
            context = _context_at(match.start(), contexts)
            clones.append(
                _RawClone(
                    action=re.sub(r"\s+", "", match.group("action")),
                    source_action=re.sub(r"\s+", "", match.group("source")),
                    position=match.start(),
                    context_start=context.start if context else None,
                    context_name=context.name if context else "<class>",
                    line_number=_line_number(source, match.start()),
                )
            )

        tex_offsets: list[str] = []
        for context in contexts:
            if context.name != "texOffset":
                continue
            return_match = re.search(
                r"\breturn\s+(?P<expr>[^;]+);", own_masked[context.start : context.end]
            )
            if return_match:
                absolute_start = context.start + return_match.start("expr")
                absolute_end = context.start + return_match.end("expr")
                tex_offsets.append(source[absolute_start:absolute_end].strip())

        raw_classes.append(
            _RawClass(
                canonical_name=canonical,
                simple_name=simple,
                parent_name=parent,
                resolved_parent_class=None,
                abstract=abstract,
                member_path=member_path,
                line_number=line,
                start=start,
                end=end,
                direct_asset_keys=tuple(dict.fromkeys(asset_keys)),
                dynamic_texture_expressions=tuple(dict.fromkeys(dynamic_textures)),
                assignments=tuple(sorted(assignments, key=lambda item: item.position)),
                arrays=tuple(arrays),
                tex_offset_expressions=tuple(tex_offsets),
                films=tuple(films),
                animation_inits=tuple(animation_inits),
                frame_calls=tuple(frame_calls),
                clones=tuple(clones),
            )
        )
    return tuple(raw_classes)


def _resolve_parent_names(classes: Sequence[_RawClass]) -> tuple[_RawClass, ...]:
    by_simple: dict[str, list[str]] = defaultdict(list)
    by_canonical = {raw.canonical_name: raw for raw in classes}
    for raw in classes:
        by_simple[raw.simple_name].append(raw.canonical_name)
    resolved: list[_RawClass] = []
    for raw in classes:
        parent: str | None = None
        outer_parts = raw.canonical_name.split(".")[:-1]
        for length in range(len(outer_parts), -1, -1):
            candidate = ".".join((*outer_parts[:length], raw.parent_name))
            if candidate != raw.canonical_name and candidate in by_canonical:
                parent = candidate
                break
        if parent is None and len(by_simple.get(raw.parent_name, ())) == 1:
            parent = by_simple[raw.parent_name][0]
        resolved.append(replace(raw, resolved_parent_class=parent))
    return tuple(resolved)


def _top_level_ternary(expression: str) -> tuple[str, str] | None:
    depth = 0
    question = -1
    nested = 0
    for index, char in enumerate(expression):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif depth == 0 and char == "?":
            if question < 0:
                question = index
            nested += 1
        elif depth == 0 and char == ":" and question >= 0:
            nested -= 1
            if nested == 0:
                return expression[question + 1 : index], expression[index + 1 :]
    return None


def _normalize_numeric(value: Numeric) -> Numeric:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _unique_values(values: Iterable[Numeric | bool]) -> tuple[Any, ...]:
    result: list[Numeric | bool] = []
    for value in values:
        normalized = _normalize_numeric(value) if not isinstance(value, bool) else value
        if normalized not in result:
            result.append(normalized)
    return tuple(sorted(result, key=lambda item: (str(type(item)), float(item))))


def _eval_expression(
    expression: str,
    env: Mapping[str, tuple[str, ...]],
    arrays: Mapping[str, tuple[str, ...]],
    *,
    tex_offset: Numeric | None = None,
    stack: frozenset[str] = frozenset(),
) -> tuple[Any, ...]:
    expression = expression.strip()
    expression = re.sub(r"\((?:int|float|double|long)\)\s*", "", expression)
    expression = re.sub(r"(?<=\d)[fFdDlL]\b", "", expression)
    ternary = _top_level_ternary(expression)
    if ternary is not None:
        left, right = ternary
        return _unique_values(
            (
                *_eval_expression(left, env, arrays, tex_offset=tex_offset, stack=stack),
                *_eval_expression(right, env, arrays, tex_offset=tex_offset, stack=stack),
            )
        )
    if expression == "true":
        return (True,)
    if expression == "false":
        return (False,)
    expression = expression.replace("Math.round", "java_round")
    expression = expression.replace("texOffset()", "tex_offset")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        array_match = re.fullmatch(r"([A-Za-z_]\w*)\s*\[.*]", expression, re.DOTALL)
        if array_match and array_match.group(1) in arrays:
            values: list[Any] = []
            for item in arrays[array_match.group(1)]:
                values.extend(
                    _eval_expression(item, env, arrays, tex_offset=tex_offset, stack=stack)
                )
            return _unique_values(values)
        return ()

    def evaluate(node: ast.AST) -> tuple[Any, ...]:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return (node.value,)
        if isinstance(node, ast.Name):
            if node.id == "tex_offset":
                return () if tex_offset is None else (tex_offset,)
            if node.id in stack or node.id not in env:
                return ()
            values: list[Any] = []
            for value_expression in env[node.id]:
                values.extend(
                    _eval_expression(
                        value_expression,
                        env,
                        arrays,
                        tex_offset=tex_offset,
                        stack=stack | {node.id},
                    )
                )
            return _unique_values(values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            values = evaluate(node.operand)
            if isinstance(node.op, ast.USub):
                return _unique_values(-value for value in values)
            return values
        if isinstance(node, ast.BinOp):
            left_values = evaluate(node.left)
            right_values = evaluate(node.right)
            result: list[Any] = []
            for left in left_values:
                for right in right_values:
                    try:
                        if isinstance(node.op, ast.Add):
                            result.append(left + right)
                        elif isinstance(node.op, ast.Sub):
                            result.append(left - right)
                        elif isinstance(node.op, ast.Mult):
                            result.append(left * right)
                        elif isinstance(node.op, ast.Div):
                            result.append(left / right)
                        elif isinstance(node.op, ast.FloorDiv):
                            result.append(left // right)
                        elif isinstance(node.op, ast.Mod):
                            result.append(left % right)
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
            return _unique_values(result)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "java_round"
            and len(node.args) == 1
        ):
            return _unique_values(math.floor(value + 0.5) for value in evaluate(node.args[0]))
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            array = arrays.get(node.value.id)
            if array is None:
                return ()
            indices = evaluate(node.slice)
            chosen = array
            if indices and all(isinstance(index, int) for index in indices):
                chosen = tuple(array[index] for index in indices if 0 <= index < len(array))
            result: list[Any] = []
            for item in chosen:
                result.extend(
                    _eval_expression(item, env, arrays, tex_offset=tex_offset, stack=stack)
                )
            return _unique_values(result)
        return ()

    return evaluate(tree)


def _environment_for(
    raw: _RawClass,
    position: int,
    context_start: int | None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for assignment in raw.assignments:
        if assignment.position > position:
            continue
        if assignment.context_start is not None and assignment.context_start != context_start:
            continue
        grouped[assignment.name].extend(assignment.expressions)
    return (
        {name: tuple(dict.fromkeys(expressions)) for name, expressions in grouped.items()},
        dict(raw.arrays),
    )


def _tex_offsets_for(
    raw: _RawClass,
    by_name: Mapping[str, _RawClass],
    seen: frozenset[str] = frozenset(),
) -> tuple[Numeric, ...]:
    if raw.canonical_name in seen:
        return ()
    if raw.tex_offset_expressions:
        env, arrays = _environment_for(raw, raw.end, None)
        values: list[Numeric] = []
        for expression in raw.tex_offset_expressions:
            values.extend(_eval_expression(expression, env, arrays))
        return tuple(value for value in _unique_values(values) if not isinstance(value, bool))
    if raw.resolved_parent_class and raw.resolved_parent_class in by_name:
        return _tex_offsets_for(
            by_name[raw.resolved_parent_class], by_name, seen | {raw.canonical_name}
        )
    return ()


def _nearest_init(raw: _RawClass, call: _RawFrameCall) -> _RawAnimationInit | None:
    candidates = [
        init
        for init in raw.animation_inits
        if init.action == call.action and init.position < call.position
    ]
    same_context = [init for init in candidates if init.context_start == call.context_start]
    return max(same_context or candidates, key=lambda item: item.position, default=None)


def _nearest_film(raw: _RawClass, call: _RawFrameCall) -> _RawFilm | None:
    if not call.args:
        return None
    variable = call.args[0].strip()
    if not re.fullmatch(r"[A-Za-z_]\w*", variable):
        return None
    candidates = [
        film
        for film in raw.films
        if film.variable_name == variable and film.position < call.position
    ]
    same_context = [film for film in candidates if film.context_start == call.context_start]
    return max(same_context or candidates, key=lambda item: item.position, default=None)


def _direct_templates(raw: _RawClass) -> dict[str, tuple[_ActionTemplate, ...]]:
    grouped: dict[str, list[_ActionTemplate]] = defaultdict(list)
    for call in raw.frame_calls:
        grouped[call.action].append(
            _ActionTemplate(
                action=call.action,
                owner_class=raw.canonical_name,
                frame_call=call,
                clone=None,
                animation_init=_nearest_init(raw, call),
                film=_nearest_film(raw, call),
            )
        )
    for clone in raw.clones:
        grouped[clone.action].append(
            _ActionTemplate(
                action=clone.action,
                owner_class=raw.canonical_name,
                frame_call=None,
                clone=clone,
                animation_init=None,
                film=None,
            )
        )
    return {action: tuple(templates) for action, templates in grouped.items()}


def _effective_templates(
    raw: _RawClass,
    by_name: Mapping[str, _RawClass],
    cache: dict[str, dict[str, tuple[_ActionTemplate, ...]]],
    stack: frozenset[str] = frozenset(),
) -> dict[str, tuple[_ActionTemplate, ...]]:
    if raw.canonical_name in cache:
        return cache[raw.canonical_name]
    if raw.canonical_name in stack:
        raise JavaSpriteParseError(f"inheritance cycle at {raw.canonical_name}")
    inherited: dict[str, tuple[_ActionTemplate, ...]] = {}
    if raw.resolved_parent_class and raw.resolved_parent_class in by_name:
        inherited.update(
            _effective_templates(
                by_name[raw.resolved_parent_class],
                by_name,
                cache,
                stack | {raw.canonical_name},
            )
        )
    inherited.update(_direct_templates(raw))
    cache[raw.canonical_name] = inherited
    return inherited


def _effective_texture_keys(
    raw: _RawClass,
    by_name: Mapping[str, _RawClass],
    hero_keys: tuple[str, ...],
    seen: frozenset[str] = frozenset(),
) -> tuple[tuple[str, ...], str, tuple[str, ...]]:
    if raw.canonical_name in seen:
        return (), "inheritance_cycle", ("texture_inheritance_cycle",)
    if raw.direct_asset_keys:
        return raw.direct_asset_keys, "direct_assets_sprites_constant", ()
    if any("spritesheet()" in expression for expression in raw.dynamic_texture_expressions):
        return (
            hero_keys,
            "dynamic_hero_class_spritesheet_candidates",
            ("runtime_hero_class_selects_one_of_candidate_sheets",),
        )
    if raw.dynamic_texture_expressions:
        return (
            (),
            "unresolved_dynamic_texture_expression",
            tuple(
                f"dynamic_texture:{expression}" for expression in raw.dynamic_texture_expressions
            ),
        )
    if raw.resolved_parent_class and raw.resolved_parent_class in by_name:
        keys, resolution, reasons = _effective_texture_keys(
            by_name[raw.resolved_parent_class],
            by_name,
            hero_keys,
            seen | {raw.canonical_name},
        )
        return keys, f"inherited:{resolution}", reasons
    return (), "no_texture_evidence", ("no_resolved_texture",)


def _numeric_candidates(
    expression: str | None,
    env: Mapping[str, tuple[str, ...]],
    arrays: Mapping[str, tuple[str, ...]],
    tex_offsets: tuple[Numeric, ...],
) -> tuple[Numeric, ...]:
    if expression is None:
        return ()
    offsets: tuple[Numeric | None, ...] = tex_offsets or (None,)
    values: list[Numeric] = []
    for offset in offsets:
        values.extend(_eval_expression(expression, env, arrays, tex_offset=offset))
    return tuple(
        value
        for value in _unique_values(values)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    )


def _boolean_candidates(
    expression: str | None,
    env: Mapping[str, tuple[str, ...]],
    arrays: Mapping[str, tuple[str, ...]],
) -> tuple[bool, ...]:
    if expression is None:
        return ()
    return tuple(
        value for value in _eval_expression(expression, env, arrays) if isinstance(value, bool)
    )


def _frame_variants(
    expressions: tuple[str, ...],
    env: Mapping[str, tuple[str, ...]],
    arrays: Mapping[str, tuple[str, ...]],
    tex_offsets: tuple[Numeric, ...],
) -> tuple[tuple[int, ...], ...]:
    offsets: tuple[Numeric | None, ...] = tex_offsets or (None,)
    referenced = set().union(
        *(set(re.findall(r"\b[A-Za-z_]\w*\b", expression)) for expression in expressions)
    )
    referenced -= {"Math", "round", "texOffset", "true", "false"}
    variable_values: dict[str, tuple[Numeric, ...]] = {}
    for name in sorted(referenced):
        if name not in env:
            continue
        candidates = _numeric_candidates(name, env, arrays, tex_offsets)
        if candidates:
            variable_values[name] = candidates

    environments: list[dict[str, tuple[str, ...]]] = [dict(env)]
    for name, candidates in variable_values.items():
        expanded: list[dict[str, tuple[str, ...]]] = []
        for candidate_env in environments:
            for candidate in candidates:
                next_env = dict(candidate_env)
                next_env[name] = (str(candidate),)
                expanded.append(next_env)
        environments = expanded

    variants: list[tuple[int, ...]] = []
    for offset in offsets:
        for candidate_env in environments:
            frames: list[int] = []
            for expression in expressions:
                values = _eval_expression(
                    expression,
                    candidate_env,
                    arrays,
                    tex_offset=offset,
                )
                integer_values = [
                    int(value)
                    for value in values
                    if isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and float(value).is_integer()
                ]
                if len(integer_values) != 1:
                    frames = []
                    break
                frames.append(integer_values[0])
            if frames and tuple(frames) not in variants:
                variants.append(tuple(frames))
    return tuple(variants)


def _film_audit(
    film: _RawFilm | None,
    owner: _RawClass,
    sheet_audits: Mapping[str, SpriteSheetAudit],
    sheet_paths: tuple[str, ...],
    tex_offsets: tuple[Numeric, ...],
) -> FilmAudit | None:
    if film is None:
        return None
    env, arrays = _environment_for(owner, film.position, film.context_start)
    width_expression: str | None = None
    height_expression: str | None = None
    layout = "unknown_texture_film_constructor"
    if len(film.args) >= 3:
        width_expression, height_expression = film.args[-2:]
        layout = "row_major_grid"
        if len(film.args) >= 4 and "tiers()" in film.args[0]:
            layout = "dynamic_hero_armor_tier_patch_row_major_grid"
    widths = _numeric_candidates(width_expression, env, arrays, tex_offsets)
    heights = _numeric_candidates(height_expression, env, arrays, tex_offsets)
    frame_width = int(widths[0]) if len(widths) == 1 and float(widths[0]).is_integer() else None
    frame_height = int(heights[0]) if len(heights) == 1 and float(heights[0]).is_integer() else None
    sheet_widths = tuple(
        sorted({sheet_audits[path].width for path in sheet_paths if path in sheet_audits})
    )
    sheet_heights = tuple(
        sorted({sheet_audits[path].height for path in sheet_paths if path in sheet_audits})
    )
    columns = (
        tuple(sorted({width // frame_width for width in sheet_widths}))
        if frame_width and frame_width > 0
        else ()
    )
    source_sheet_rows = (
        tuple(sorted({height // frame_height for height in sheet_heights}))
        if frame_height and frame_height > 0
        else ()
    )
    if layout.startswith("dynamic_hero_armor_tier_patch"):
        rows = (1,) if frame_height else ()
    else:
        rows = (
            tuple(sorted({height // frame_height for height in sheet_heights}))
            if frame_height and frame_height > 0
            else ()
        )
    capacities = tuple(sorted({column * row for column in columns for row in rows}))
    source_sheet_capacities = tuple(
        sorted({column * row for column in columns for row in source_sheet_rows})
    )
    return FilmAudit(
        variable_name=film.variable_name,
        constructor_expression=film.constructor_expression,
        layout_kind=layout,
        frame_width=frame_width,
        frame_height=frame_height,
        frame_width_expression=width_expression,
        frame_height_expression=height_expression,
        sheet_widths=sheet_widths,
        sheet_heights=sheet_heights,
        columns=columns,
        rows=rows,
        capacities=capacities,
        source_sheet_grid_rows=source_sheet_rows,
        source_sheet_grid_capacities=source_sheet_capacities,
        evidence_member_path=owner.member_path,
        line_number=film.line_number,
    )


def _cells_for_frames(
    variants: tuple[tuple[int, ...], ...],
    film: FilmAudit | None,
) -> tuple[tuple[FrameCell, ...], ...]:
    if (
        film is None
        or film.frame_width is None
        or film.frame_height is None
        or len(film.columns) != 1
    ):
        return ()
    columns = film.columns[0]
    coordinate_space = (
        "dynamic_hero_armor_tier_patch"
        if film.layout_kind.startswith("dynamic_hero_armor_tier_patch")
        else "source_sheet"
    )
    return tuple(
        tuple(
            FrameCell(
                frame_index=index,
                column=index % columns,
                row=index // columns,
                left=(index % columns) * film.frame_width,
                top=(index // columns) * film.frame_height,
                right=(index % columns + 1) * film.frame_width,
                bottom=(index // columns + 1) * film.frame_height,
                coordinate_space=coordinate_space,
            )
            for index in variant
        )
        for variant in variants
    )


def _uv_rects(
    call: _RawFrameCall,
    owner: _RawClass,
    tex_offsets: tuple[Numeric, ...],
) -> tuple[FrameCell, ...]:
    if len(call.args) != 1:
        return ()
    match = re.fullmatch(r"texture\.uvRect\s*\((.*)\)", call.args[0], re.DOTALL)
    if match is None:
        return ()
    expressions = _split_args(match.group(1))
    if len(expressions) != 4:
        return ()
    env, arrays = _environment_for(owner, call.position, call.context_start)
    values = [
        _numeric_candidates(expression, env, arrays, tex_offsets) for expression in expressions
    ]
    if any(len(item) != 1 or not float(item[0]).is_integer() for item in values):
        return ()
    left, top, right, bottom = (int(item[0]) for item in values)
    return (
        FrameCell(
            frame_index=None,
            column=None,
            row=None,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            coordinate_space="source_sheet_explicit_uv_rect",
        ),
    )


def _base_action_name(action: str) -> str:
    return re.sub(r"\[.*]$", "", action)


def _normalize_action(action: str) -> str | None:
    base = _base_action_name(action)
    return _NORMALIZED_ACTIONS.get(base)


def _classify_entity(
    class_name: str,
    sheet_paths: tuple[str, ...],
) -> tuple[str, str, tuple[str, ...], str]:
    simple_name = class_name.split(".")[-1]
    joined = " ".join((simple_name, *sheet_paths)).lower()
    label = re.sub(r"(?<!^)(?=[A-Z])", " ", simple_name).replace(" Sprite", "")

    rules: tuple[tuple[tuple[str, ...], str, tuple[str, ...]], ...] = (
        (("fungal_spinner",), "creature", ("fungal",)),
        (("rat", "ratking", "sheep"), "animal", ("quadruped",)),
        (("spirit_hawk", "hawk", "bat"), "animal", ("winged",)),
        (("bee", "swarm"), "animal", ("insect", "winged")),
        (("crab",), "animal", ("crustacean", "multi_legged")),
        (("piranha",), "animal", ("fish", "aquatic")),
        (("snake",), "animal", ("serpentine",)),
        (("scorpio", "spinner"), "animal", ("arthropod", "multi_legged")),
        (("larva",), "animal", ("invertebrate",)),
        (("dm100", "dm200", "dm201", "dm300", "pylon", "red_sentry"), "robot", ()),
        (
            (
                "hero",
                "warrior",
                "mage",
                "rogue",
                "huntress",
                "duelist",
                "cleric",
                "thief",
                "bandit",
                "brute",
                "guard",
                "gnoll",
                "monk",
                "necromancer",
                "shaman",
                "shopkeeper",
                "wandmaker",
                "blacksmith",
                "king",
            ),
            "humanoid",
            ("biped",),
        ),
        (("ninja_log", "crystal_spire", "rot_heart", "lotus"), "object", ()),
        (("fungal_core", "fungal_sentry", "spawner", "ward", "wards"), "object", ()),
        (("slime", "goo", "mimic", "wisp", "lasher"), "creature", ()),
        (
            (
                "skeleton",
                "undead",
                "warlock",
                "succubus",
                "tengu",
                "elemental",
                "wraith",
                "ghost",
                "yog",
                "fist",
                "demon",
                "ripper",
                "ghoul",
                "eye",
            ),
            "monster",
            (),
        ),
        (("golem", "statue", "guardian"), "creature", ("biped",)),
    )
    for tokens, entity_class, morphology in rules:
        matched = next(
            (
                token
                for token in tokens
                if re.search(
                    rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])",
                    joined,
                )
            ),
            None,
        )
        if matched is not None:
            return entity_class, label, morphology, f"source_identifier_token:{matched}"
    return "unknown", label, (), "no_high_confidence_identifier_rule"


def _materialize_animation(
    template: _ActionTemplate,
    target: _RawClass,
    owner: _RawClass,
    source_keys: tuple[str, ...],
    source_paths: tuple[str, ...],
    sheet_audits: Mapping[str, SpriteSheetAudit],
    tex_offsets: tuple[Numeric, ...],
) -> AnimationAudit:
    if template.frame_call is None:
        raise JavaSpriteParseError("clone templates must be resolved before materialization")
    call = template.frame_call
    env, arrays = _environment_for(owner, call.position, call.context_start)
    frame_expressions = call.args[1:] if template.film is not None else ()
    referenced_frame_variables = tuple(
        sorted(
            {
                name
                for expression in frame_expressions
                for name in re.findall(r"\b[A-Za-z_]\w*\b", expression)
                if name in env
            }
        )
    )
    frame_variants = _frame_variants(frame_expressions, env, arrays, tex_offsets)
    uv_rects = _uv_rects(call, owner, tex_offsets)
    film = _film_audit(template.film, owner, sheet_audits, source_paths, tex_offsets)
    frame_cells = _cells_for_frames(frame_variants, film)
    init = template.animation_init
    fps_values = _numeric_candidates(
        init.fps_expression if init else None, env, arrays, tex_offsets
    )
    looping_values = _boolean_candidates(init.looping_expression if init else None, env, arrays)
    geometry_valid: bool | None = None
    if frame_variants and film and film.capacities:
        geometry_valid = all(
            index >= 0 and all(index < capacity for capacity in film.capacities)
            for variant in frame_variants
            for index in variant
        )
    elif uv_rects and source_paths:
        dimensions = [
            (sheet_audits[path].width, sheet_audits[path].height)
            for path in source_paths
            if path in sheet_audits
        ]
        if dimensions:
            geometry_valid = all(
                0 <= rect.left < rect.right <= width and 0 <= rect.top < rect.bottom <= height
                for rect in uv_rects
                for width, height in dimensions
            )
    ambiguity: list[str] = []
    if len(frame_variants) > 1:
        ambiguity.append("runtime_branch_or_offset_produces_multiple_exact_frame_orders")
    if not frame_variants and not uv_rects:
        ambiguity.append("frame_expressions_not_statically_resolved")
    if len(fps_values) > 1:
        ambiguity.append("runtime_branch_produces_multiple_fps_values")
    if not fps_values:
        ambiguity.append("fps_not_statically_resolved")
    if not looping_values:
        ambiguity.append("loop_flag_not_statically_resolved")
    if len(source_paths) > 1:
        ambiguity.append("runtime_selects_one_of_multiple_source_sheets")
    if geometry_valid is False:
        ambiguity.append("referenced_frame_outside_derived_film_capacity")
    return AnimationAudit(
        source_action=call.action,
        normalized_action=_normalize_action(call.action),
        fps_values=fps_values,
        fps_expression=init.fps_expression if init else None,
        looping_values=looping_values,
        looping_expression=init.looping_expression if init else None,
        frame_index_variants=frame_variants,
        frame_expression_order=tuple(expression.strip() for expression in frame_expressions),
        frame_variable_expressions=tuple((name, env[name]) for name in referenced_frame_variables),
        frame_cell_variants=frame_cells,
        direct_uv_rect_variants=uv_rects,
        film=film,
        source_asset_keys=source_keys,
        source_sheet_paths=source_paths,
        clone_of=None,
        defined_in_class=owner.canonical_name,
        inherited=owner.canonical_name != target.canonical_name,
        context=call.context_name,
        evidence_member_path=owner.member_path,
        line_number=call.line_number,
        frame_order_preserved=bool(frame_variants or uv_rects),
        deliberate_repeats_preserved=bool(frame_variants),
        timing_preserved=len(fps_values) == 1,
        loop_semantics_preserved=len(looping_values) == 1,
        geometry_valid=geometry_valid,
        ambiguity_reasons=tuple(ambiguity),
    )


def _materialize_templates(
    templates: Mapping[str, tuple[_ActionTemplate, ...]],
    target: _RawClass,
    by_name: Mapping[str, _RawClass],
    source_keys: tuple[str, ...],
    source_paths: tuple[str, ...],
    sheet_audits: Mapping[str, SpriteSheetAudit],
    tex_offsets: tuple[Numeric, ...],
) -> tuple[AnimationAudit, ...]:
    cache: dict[str, tuple[AnimationAudit, ...]] = {}

    def resolve(action: str, stack: frozenset[str] = frozenset()) -> tuple[AnimationAudit, ...]:
        if action in cache:
            return cache[action]
        if action in stack:
            return ()
        results: list[AnimationAudit] = []
        for template in templates.get(action, ()):
            owner = by_name[template.owner_class]
            if template.frame_call is not None:
                results.append(
                    _materialize_animation(
                        template,
                        target,
                        owner,
                        source_keys,
                        source_paths,
                        sheet_audits,
                        tex_offsets,
                    )
                )
            elif template.clone is not None:
                source_animations = resolve(template.clone.source_action, stack | {action})
                for source in source_animations:
                    results.append(
                        replace(
                            source,
                            source_action=template.clone.action,
                            normalized_action=_normalize_action(template.clone.action),
                            clone_of=template.clone.source_action,
                            defined_in_class=owner.canonical_name,
                            inherited=owner.canonical_name != target.canonical_name,
                            context=template.clone.context_name,
                            evidence_member_path=owner.member_path,
                            line_number=template.clone.line_number,
                        )
                    )
        cache[action] = tuple(results)
        return cache[action]

    for action in templates:
        resolve(action)
    return tuple(animation for action in sorted(cache) for animation in cache[action])


def _image_has_transparency(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        extrema = image.getchannel("A").getextrema()
        return bool(extrema and extrema[0] < 255)
    return "transparency" in image.info


def _audit_sprite_sheets(
    archive: ZipFile,
    infos: Mapping[str, ZipInfo],
    root: str,
    mappings: Sequence[SpriteAssetMapping],
) -> tuple[SpriteSheetAudit, ...]:
    by_path = {mapping.relative_path: mapping for mapping in mappings}
    prefix = f"{root}/{_SPRITE_ASSET_PREFIX}"
    sheets: list[SpriteSheetAudit] = []
    for member_path in sorted(
        name for name in infos if name.startswith(prefix) and name.lower().endswith(".png")
    ):
        relative_path = f"sprites/{member_path[len(prefix) :]}"
        info = infos[member_path]
        payload = archive.read(info)
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                width, height = image.size
                mode = image.mode
                transparency = _image_has_transparency(image)
                metadata_keys = tuple(
                    sorted(
                        str(key)
                        for key in image.info
                        if str(key).lower()
                        in {"author", "artist", "copyright", "license", "source", "comment"}
                    )
                )
        except (UnidentifiedImageError, OSError) as error:
            raise ShatteredPixelDungeonArchiveError(
                f"invalid sprite PNG {member_path}: {error}"
            ) from error
        mapping = by_path.get(relative_path)
        sheets.append(
            SpriteSheetAudit(
                asset_key=mapping.key if mapping else None,
                relative_path=relative_path,
                member_path=member_path,
                mapped_by_assets_java=mapping is not None,
                width=width,
                height=height,
                image_mode=mode,
                has_transparency=transparency,
                size_bytes=info.file_size,
                compressed_size_bytes=info.compress_size,
                crc32=f"{info.CRC:08x}",
                sha256=_sha256_bytes(payload),
                embedded_metadata_keys=metadata_keys,
            )
        )
    return tuple(sheets)


def _repository_commit_from_root(root: str) -> str | None:
    match = re.fullmatch(r"shattered-pixel-dungeon-([0-9a-fA-F]{40})", root)
    return match.group(1).lower() if match else None


def _validate_zip_members(archive: ZipFile) -> tuple[dict[str, ZipInfo], str]:
    infos: dict[str, ZipInfo] = {}
    roots: set[str] = set()
    casefolded: dict[str, str] = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ShatteredPixelDungeonArchiveError(f"unsafe archive member path {info.filename!r}")
        normalized = path.as_posix().rstrip("/")
        if not normalized:
            continue
        folded = normalized.casefold()
        if folded in casefolded and casefolded[folded] != normalized:
            raise ShatteredPixelDungeonArchiveError(
                f"case-colliding archive members: {casefolded[folded]!r}, {normalized!r}"
            )
        casefolded[folded] = normalized
        if normalized in infos:
            raise ShatteredPixelDungeonArchiveError(f"duplicate archive member {normalized!r}")
        infos[normalized] = info
        roots.add(path.parts[0])
        if info.flag_bits & 0x1:
            raise ShatteredPixelDungeonArchiveError(
                f"encrypted archive member is unsupported: {normalized!r}"
            )
    if len(roots) != 1:
        raise ShatteredPixelDungeonArchiveError(
            f"expected exactly one archive root, found {sorted(roots)!r}"
        )
    return infos, next(iter(roots))


def _evidence_document(
    archive: ZipFile,
    infos: Mapping[str, ZipInfo],
    root: str,
    relative_path: str,
    *,
    identifiers: tuple[str, ...],
    scope: str,
    notes: str,
) -> EvidenceDocument | None:
    member_path = f"{root}/{relative_path}"
    info = infos.get(member_path)
    if info is None:
        return None
    payload = archive.read(info)
    return EvidenceDocument(
        relative_path=relative_path,
        member_path=member_path,
        sha256=_sha256_bytes(payload),
        size_bytes=info.file_size,
        detected_license_identifiers=identifiers,
        scope=scope,
        notes=notes,
    )


def _attribution_evidence(
    archive: ZipFile,
    infos: Mapping[str, ZipInfo],
    root: str,
) -> tuple[AttributionEvidence, ...]:
    evidence: list[AttributionEvidence] = []
    readme_path = f"{root}/{_README_RELATIVE_PATH}"
    if readme_path in infos:
        text = archive.read(readme_path).decode("utf-8-sig")
        for line_number, line in enumerate(text.splitlines(), 1):
            if "based on" in line.lower() and "Pixel Dungeon" in line:
                evidence.append(
                    AttributionEvidence(
                        name="Pixel Dungeon / Watabou",
                        role="upstream project named by repository README",
                        evidence_member_path=readme_path,
                        evidence_line_numbers=(line_number,),
                        evidence_text=line.strip(),
                    )
                )
                break
    copyright_lines: dict[tuple[str, str], list[int]] = defaultdict(list)
    representative_path: dict[tuple[str, str], str] = {}
    pattern = re.compile(r"Copyright \(C\) (?P<years>[0-9-]+) (?P<name>[^\r\n*]+)")
    for member_path, info in infos.items():
        if not member_path.endswith(".java"):
            continue
        text = archive.read(info).decode("utf-8-sig")
        for match in pattern.finditer(text[:2500]):
            key = (match.group("name").strip(), match.group("years"))
            copyright_lines[key].append(_line_number(text, match.start()))
            representative_path.setdefault(key, member_path)
    for (name, years), lines in sorted(copyright_lines.items()):
        evidence.append(
            AttributionEvidence(
                name=name,
                role=f"copyright notice ({years}); repeated in {len(lines)} Java files",
                evidence_member_path=representative_path[(name, years)],
                evidence_line_numbers=tuple(sorted(set(lines))),
                evidence_text=f"Copyright (C) {years} {name}",
            )
        )
    return tuple(evidence)


def audit_shattered_pixel_dungeon_archive(
    archive_path: str | Path,
) -> ShatteredPixelDungeonAudit:
    path = Path(archive_path)
    if not path.is_file():
        raise ShatteredPixelDungeonArchiveError(f"archive does not exist: {path}")
    archive_sha256 = _sha256_file(path)
    try:
        with ZipFile(path) as archive:
            infos, root = _validate_zip_members(archive)
            assets_path = f"{root}/{_ASSETS_SUFFIX}"
            if assets_path not in infos:
                raise ShatteredPixelDungeonArchiveError(
                    f"archive has no pinned Assets.java at {assets_path!r}"
                )
            assets_source = archive.read(assets_path).decode("utf-8-sig")
            mappings = parse_assets_sprite_mappings(assets_source, member_path=assets_path)
            hero_path = f"{root}/{_HERO_CLASS_SUFFIX}"
            hero_mappings = (
                parse_hero_class_sheet_mappings(
                    archive.read(hero_path).decode("utf-8-sig"),
                    mappings,
                    member_path=hero_path,
                )
                if hero_path in infos
                else ()
            )
            sheets = _audit_sprite_sheets(archive, infos, root, mappings)
            sheets_by_path = {sheet.relative_path: sheet for sheet in sheets}

            raw_classes: list[_RawClass] = []
            java_paths = sorted(name for name in infos if name.endswith(".java"))
            for member_path in java_paths:
                source = archive.read(member_path).decode("utf-8-sig")
                raw_classes.extend(_parse_raw_classes(source, member_path))
            resolved_classes = _resolve_parent_names(raw_classes)
            by_name = {raw.canonical_name: raw for raw in resolved_classes}
            hero_keys = tuple(mapping.asset_key for mapping in hero_mappings)
            mapping_by_key = {mapping.key: mapping for mapping in mappings}

            template_cache: dict[str, dict[str, tuple[_ActionTemplate, ...]]] = {}
            candidate_classes: list[_RawClass] = []
            for raw in resolved_classes:
                templates = _effective_templates(raw, by_name, template_cache)
                keys, _, _ = _effective_texture_keys(raw, by_name, hero_keys)
                if templates and keys:
                    candidate_classes.append(raw)

            class_audits: list[SpriteClassAudit] = []
            for raw in sorted(candidate_classes, key=lambda item: item.canonical_name):
                templates = _effective_templates(raw, by_name, template_cache)
                keys, texture_resolution, texture_ambiguity = _effective_texture_keys(
                    raw, by_name, hero_keys
                )
                paths = tuple(
                    mapping_by_key[key].relative_path for key in keys if key in mapping_by_key
                )
                tex_offsets = _tex_offsets_for(raw, by_name)
                animations = _materialize_templates(
                    templates,
                    raw,
                    by_name,
                    keys,
                    paths,
                    sheets_by_path,
                    tex_offsets,
                )
                entity_class, label, morphology, basis = _classify_entity(raw.canonical_name, paths)
                ambiguities = list(texture_ambiguity)
                if raw.abstract:
                    ambiguities.append("abstract_definition_requires_concrete_subclass")
                if not animations:
                    ambiguities.append("no_materialized_animation_records")
                class_audits.append(
                    SpriteClassAudit(
                        class_name=raw.canonical_name,
                        simple_name=raw.simple_name,
                        parent_name=raw.parent_name,
                        resolved_parent_class=raw.resolved_parent_class,
                        abstract=raw.abstract,
                        evidence_member_path=raw.member_path,
                        line_number=raw.line_number,
                        source_asset_keys=keys,
                        source_sheet_paths=paths,
                        texture_resolution=texture_resolution,
                        animations=animations,
                        entity_class=entity_class,
                        entity_label=label,
                        morphology_tags=morphology,
                        classification_basis=basis,
                        view="top_down",
                        direction_semantics=(
                            "no_explicit_direction_track; CharSprite_default_can_horizontal_flip"
                        ),
                        ambiguity_reasons=tuple(dict.fromkeys(ambiguities)),
                    )
                )

            license_member_path = f"{root}/{_LICENSE_RELATIVE_PATH}"
            license_text = (
                archive.read(license_member_path).decode("utf-8-sig", errors="replace")
                if license_member_path in infos
                else ""
            )
            root_license_identifiers = (
                ("GPL-3.0-only",)
                if "GNU GENERAL PUBLIC LICENSE" in license_text
                and re.search(r"\bVersion\s+3\b", license_text)
                else ()
            )
            java_header_identifiers = (
                ("GPL-3.0-or-later",)
                if re.search(
                    r"either version 3 of the License, or\s*\*?\s*"
                    r"\(at your option\) any later version",
                    assets_source,
                )
                else ()
            )
            license_doc = _evidence_document(
                archive,
                infos,
                root,
                _LICENSE_RELATIVE_PATH,
                identifiers=root_license_identifiers,
                scope="repository_root_license_text",
                notes=(
                    "The root contains the GNU GPL version 3 text. Java file headers grant "
                    "GPL version 3 or later for the program. No sprite-specific license or "
                    "credit manifest was found, so PNG scope remains repository-level evidence."
                ),
            )
            java_header_doc = _evidence_document(
                archive,
                infos,
                root,
                _ASSETS_SUFFIX,
                identifiers=java_header_identifiers,
                scope="representative_java_source_header",
                notes=(
                    "Assets.java carries the program grant for GNU GPL version 3 or, at the "
                    "recipient's option, any later version. The same header occurs throughout "
                    "the audited Java tree."
                ),
            )
            readme_doc = _evidence_document(
                archive,
                infos,
                root,
                _README_RELATIVE_PATH,
                identifiers=(),
                scope="repository_description_and_upstream_attribution",
                notes="Names Pixel Dungeon and Watabou as the upstream project/source.",
            )
            documents = tuple(
                doc for doc in (license_doc, java_header_doc, readme_doc) if doc is not None
            )
            attributions = _attribution_evidence(archive, infos, root)
    except (BadZipFile, OSError) as error:
        raise ShatteredPixelDungeonArchiveError(
            f"cannot read ZIP archive {path}: {error}"
        ) from error

    repository_commit = _repository_commit_from_root(root)
    mapped_paths = {mapping.relative_path for mapping in mappings}
    present_paths = {sheet.relative_path for sheet in sheets}
    missing_mapped = tuple(sorted(mapped_paths - present_paths))
    unmapped = tuple(sorted(present_paths - mapped_paths))
    concrete = tuple(audit for audit in class_audits if not audit.abstract)
    entity_counter = Counter(audit.entity_class for audit in concrete)
    entity_counts = tuple(
        EntityClassCount(entity_class=key, concrete_class_count=value)
        for key, value in sorted(entity_counter.items())
    )
    action_class_sets: dict[str, set[str]] = defaultdict(set)
    action_slots: Counter[str] = Counter()
    action_variants: Counter[str] = Counter()
    action_frames: Counter[str] = Counter()
    for class_audit in concrete:
        by_action: dict[str, list[AnimationAudit]] = defaultdict(list)
        for animation in class_audit.animations:
            by_action[animation.source_action].append(animation)
        for action, animations in by_action.items():
            action_class_sets[action].add(class_audit.class_name)
            action_slots[action] += 1
            action_variants[action] += sum(
                animation.resolved_sequence_variant_count for animation in animations
            )
            action_frames[action] += sum(
                sum(len(variant) for variant in animation.frame_index_variants)
                + len(animation.direct_uv_rect_variants)
                for animation in animations
            )
    actions = tuple(
        ActionCount(
            source_action=action,
            concrete_class_count=len(action_class_sets[action]),
            action_slot_count=action_slots[action],
            resolved_sequence_variant_count=action_variants[action],
            frame_occurrence_count=action_frames[action],
        )
        for action in sorted(action_slots)
    )

    all_animations = tuple(
        animation for class_audit in concrete for animation in class_audit.animations
    )
    source_frame_records = {
        (animation.evidence_member_path, animation.line_number, animation.source_action)
        for class_audit in class_audits
        for animation in class_audit.animations
        if animation.clone_of is None
    }
    source_clone_records = {
        (
            animation.evidence_member_path,
            animation.line_number,
            animation.source_action,
            animation.clone_of,
        )
        for class_audit in class_audits
        for animation in class_audit.animations
        if animation.clone_of is not None
    }
    unresolved = tuple(
        animation
        for animation in all_animations
        if not animation.frame_index_variants and not animation.direct_uv_rect_variants
    )
    invalid_geometry = tuple(
        animation for animation in all_animations if animation.geometry_valid is False
    )
    issues: list[AuditIssue] = []
    if missing_mapped:
        issues.append(
            AuditIssue(
                severity="error",
                code="mapped_sprite_png_missing",
                message="Assets.Sprites mappings point to absent PNG members.",
                related_names=missing_mapped,
            )
        )
    if unmapped:
        issues.append(
            AuditIssue(
                severity="info",
                code="sprite_png_not_mapped_by_assets_java",
                message="Sprite-directory PNGs exist without an Assets.Sprites constant.",
                related_names=unmapped,
            )
        )
    if unresolved:
        issues.append(
            AuditIssue(
                severity="warning",
                code="animation_frame_order_not_statically_resolved",
                message="Some Java animation calls use expressions outside the adapter subset.",
                related_names=tuple(
                    sorted(
                        {
                            f"{animation.defined_in_class}:{animation.source_action}:"
                            f"{animation.line_number}"
                            for animation in unresolved
                        }
                    )
                ),
            )
        )
    if invalid_geometry:
        issues.append(
            AuditIssue(
                severity="error",
                code="animation_frame_outside_sheet_grid",
                message="A resolved frame index exceeds its TextureFilm grid capacity.",
                related_names=tuple(
                    sorted(
                        {
                            f"{animation.defined_in_class}:{animation.source_action}:"
                            f"{animation.line_number}"
                            for animation in invalid_geometry
                        }
                    )
                ),
            )
        )
    issues.append(
        AuditIssue(
            severity="warning",
            code="sprite_asset_license_scope_is_repository_level",
            message=(
                "The archive contains a root GPLv3 license and GPLv3-or-later Java headers, "
                "but no PNG-level license/artist manifest. Preserve this scope caveat in exports."
            ),
            related_names=(_LICENSE_RELATIVE_PATH,),
        )
    )
    dynamic_classes = tuple(
        audit.class_name for audit in concrete if "dynamic_hero_class" in audit.texture_resolution
    )
    if dynamic_classes:
        issues.append(
            AuditIssue(
                severity="info",
                code="runtime_selected_hero_sheet",
                message="Hero-derived sprites select one of six class sheets at runtime.",
                related_names=dynamic_classes,
            )
        )

    counts = ShatteredPixelDungeonCounts(
        zip_member_count=len(infos),
        archive_png_file_count=sum(name.lower().endswith(".png") for name in infos),
        java_file_count=len(java_paths),
        parsed_java_class_count=len(resolved_classes),
        assets_sprite_mapping_count=len(mappings),
        hero_class_sheet_mapping_count=len(hero_mappings),
        sprite_png_file_count=len(sheets),
        mapped_sprite_png_file_count=sum(sheet.mapped_by_assets_java for sheet in sheets),
        unmapped_sprite_png_file_count=len(unmapped),
        missing_mapped_sprite_png_count=len(missing_mapped),
        sprite_definition_class_count=len(class_audits),
        concrete_sprite_class_count=len(concrete),
        abstract_sprite_class_count=sum(audit.abstract for audit in class_audits),
        source_animation_frame_call_count=len(source_frame_records),
        source_animation_clone_assignment_count=len(source_clone_records),
        concrete_action_slot_count=sum(len(audit.action_slots) for audit in concrete),
        resolved_sequence_variant_count=sum(
            animation.resolved_sequence_variant_count for animation in all_animations
        ),
        unresolved_animation_count=len(unresolved),
        frame_occurrence_count=sum(
            sum(len(variant) for variant in animation.frame_index_variants)
            + len(animation.direct_uv_rect_variants)
            for animation in all_animations
        ),
        invalid_geometry_animation_count=len(invalid_geometry),
        animal_concrete_class_count=entity_counter.get("animal", 0),
        quadruped_concrete_class_count=sum(
            "quadruped" in audit.morphology_tags for audit in concrete
        ),
        evidence_document_count=len(documents),
        license_evidence_document_count=sum(
            bool(document.detected_license_identifiers) for document in documents
        ),
        sprite_png_with_embedded_attribution_count=sum(
            bool(sheet.embedded_metadata_keys) for sheet in sheets
        ),
    )
    return ShatteredPixelDungeonAudit(
        archive_path=str(path.resolve()),
        archive_sha256=archive_sha256,
        repository_commit=repository_commit,
        repository_url=SHATTERED_PIXEL_DUNGEON_REPOSITORY_URL,
        commit_url=(
            f"{SHATTERED_PIXEL_DUNGEON_REPOSITORY_URL}/tree/{repository_commit}"
            if repository_commit
            else None
        ),
        root_prefix=root,
        assets_java_member_path=assets_path,
        counts=counts,
        asset_mappings=mappings,
        hero_class_sheet_mappings=hero_mappings,
        sprite_sheets=sheets,
        sprite_classes=tuple(class_audits),
        entity_classes=entity_counts,
        actions=actions,
        evidence_documents=documents,
        attributions=attributions,
        issues=tuple(issues),
    )


def audit_known_shattered_pixel_dungeon_archive(
    archive_path: str | Path,
) -> ShatteredPixelDungeonAudit:
    path = Path(archive_path)
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != EXPECTED_SHATTERED_PIXEL_DUNGEON_ARCHIVE_SHA256:
        raise ShatteredPixelDungeonArchiveError(
            "archive digest mismatch: expected "
            f"{EXPECTED_SHATTERED_PIXEL_DUNGEON_ARCHIVE_SHA256}, got {actual_sha256}"
        )
    audit = audit_shattered_pixel_dungeon_archive(path)
    if audit.repository_commit != SHATTERED_PIXEL_DUNGEON_COMMIT:
        raise ShatteredPixelDungeonArchiveError(
            "archive root commit mismatch: expected "
            f"{SHATTERED_PIXEL_DUNGEON_COMMIT}, got {audit.repository_commit}"
        )
    if audit.root_prefix != _EXPECTED_ROOT:
        raise ShatteredPixelDungeonArchiveError(
            f"archive root mismatch: expected {_EXPECTED_ROOT!r}, got {audit.root_prefix!r}"
        )
    return audit
