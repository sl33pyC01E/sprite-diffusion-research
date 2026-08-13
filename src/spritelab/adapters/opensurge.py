"""Evidence-preserving adapter for Open Surge ``.spr`` sprite definitions.

Open Surge is unusually valuable for animation research because the repository
contains executable sprite metadata rather than only sheets.  This module keeps
the source declarations intact: numeric animation IDs, ordered ``data`` frame
occurrences (including deliberate repeats), FPS, loop tails, transitions, and
per-animation anchor overrides.

The parser is deliberately conservative about semantics.  Human comments are
kept verbatim and only a small, explicit vocabulary is mapped to the project's
conditioning taxonomy.  An unrecognised comment remains an unrecognised comment;
numeric IDs and filenames are never treated as action labels by themselves.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

EXPECTED_OPEN_SURGE_ARCHIVE_SHA256 = (
    "1448d51c3e8bc8122ae893aad96e73f9c9ef1ffb74457109fa8cfa0ef10d0206"
)
OPEN_SURGE_COMMIT = "bcb3466e10913f2d5f34dec848e0c2f3ee944883"
OPEN_SURGE_REPOSITORY_URL = "https://github.com/alemart/opensurge"
OPEN_SURGE_COMMIT_URL = f"{OPEN_SURGE_REPOSITORY_URL}/tree/{OPEN_SURGE_COMMIT}"

_EXPECTED_ROOT = f"opensurge-{OPEN_SURGE_COMMIT}"
_COPYRIGHT_DATA_PATH = "src/misc/copyright_data.csv"
_COLOR_ENGINE_PATH = "src/core/color.c"
_SHADER_ENGINE_PATH = "src/core/shader.c"
_BLOCK_IDENTIFIERS = frozenset(
    {"animation", "custom_properties", "keyframe", "keyframes", "sprite", "transition"}
)
_SPRITE_PROPERTIES = frozenset(
    {
        "action_spot",
        "animation",
        "custom_properties",
        "frame_size",
        "hot_spot",
        "keyframes",
        "source_file",
        "source_rect",
        "transition",
    }
)
_ANIMATION_PROPERTIES = frozenset(
    {"action_spot", "data", "fps", "hot_spot", "play", "repeat", "repeat_from"}
)

Numeric = int | float
TransitionEndpoint = int | Literal["any"]


class OpenSurgeArchiveError(ValueError):
    """Raised when a ZIP cannot be audited as the expected Open Surge tree."""


class OpenSurgeSpriteParseError(ValueError):
    """Raised when a ``.spr`` declaration is structurally contradictory."""


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class RawProperty:
    """A parsed property retained even when the adapter does not interpret it."""

    name: str
    values: tuple[str, ...]
    line_number: int


@dataclass(frozen=True)
class AssetCredit:
    """One exact row from ``src/misc/copyright_data.csv``."""

    asset_type: str
    file_path: str
    license_expression: str
    author: str
    website: str | None
    notes: str | None
    evidence_member_path: str
    line_number: int


@dataclass(frozen=True)
class SpriteScriptEvidence:
    relative_path: str
    member_path: str
    sha256: str
    size_bytes: int
    declared_file: str | None
    description: str | None
    authors: tuple[str, ...]
    license_expressions: tuple[str, ...]
    artwork_comments: tuple[str, ...]
    sprite_count: int


@dataclass(frozen=True)
class FrameOccurrence:
    """One occurrence in ``data``, not one deduplicated sheet cell."""

    occurrence_index: int
    source_frame_index: int
    column: int
    row: int
    left: int
    top: int
    right: int
    bottom: int
    in_loop_tail: bool
    within_declared_source_rect: bool
    within_source_image: bool | None


@dataclass(frozen=True)
class AnimationDefinition:
    declaration_kind: Literal["animation", "transition"]
    animation_id: int | None
    transition_from: TransitionEndpoint | None
    transition_to: TransitionEndpoint | None
    transition_ordinal: int | None
    source_label: str | None
    source_label_basis: str
    normalized_action: str | None
    normalized_action_basis: str
    direction_hint: str | None
    source_variant_hint: str | None
    repeat: bool
    repeat_was_explicit: bool
    effective_repeat: bool
    repeat_from: int
    repeat_from_was_explicit: bool
    effective_repeat_from: int
    fps: float
    fps_source_token: str
    fps_was_explicit: bool
    data: tuple[int, ...]
    intro_data: tuple[int, ...]
    loop_data: tuple[int, ...]
    hot_spot: Point
    hot_spot_overridden: bool
    action_spot: Point
    action_spot_overridden: bool
    programmatic_animation_name: str | None
    frame_occurrences: tuple[FrameOccurrence, ...]
    comments: tuple[str, ...]
    unknown_properties: tuple[RawProperty, ...]
    evidence_member_path: str
    line_number: int

    @property
    def frame_count(self) -> int:
        return len(self.data)

    @property
    def duration_seconds(self) -> float:
        return len(self.data) / self.fps

    @property
    def loop_mode(self) -> str:
        if not self.effective_repeat:
            return "one_shot"
        return "intro_then_loop" if self.effective_repeat_from else "loop"


@dataclass(frozen=True)
class EntityClassification:
    primary_entity_class: str
    entity_class_candidates: tuple[str, ...]
    subject_role: str
    morphology_tags: tuple[str, ...]
    parent_subject: str | None
    classification_basis: str


@dataclass(frozen=True)
class SpriteDefinition:
    identity: str
    source_file: str
    source_rect: Rectangle
    frame_size: Point
    hot_spot: Point
    action_spot: Point
    source_sheet_columns: int
    source_sheet_rows: int
    source_sheet_frame_capacity: int
    source_image_width: int | None
    source_image_height: int | None
    source_file_exists: bool | None
    source_rect_within_image: bool | None
    source_rect_grid_compatible: bool
    referenced_frames_within_declared_grid: bool
    referenced_frames_within_image: bool | None
    animations: tuple[AnimationDefinition, ...]
    transitions: tuple[AnimationDefinition, ...]
    entity: EntityClassification
    asset_credit: AssetCredit | None
    source_header_authors: tuple[str, ...]
    source_header_licenses: tuple[str, ...]
    source_comments: tuple[str, ...]
    unknown_properties: tuple[RawProperty, ...]
    evidence_member_path: str
    relative_script_path: str
    line_number: int


@dataclass(frozen=True)
class SourceSheetAudit:
    relative_path: str
    member_path: str
    width: int
    height: int
    image_mode: str
    image_format: str | None
    has_transparency: bool
    size_bytes: int
    compressed_size_bytes: int
    crc32: str
    sha256: str
    sprite_reference_count: int
    asset_credit: AssetCredit | None


@dataclass(frozen=True)
class EvidenceDocument:
    relative_path: str
    member_path: str
    sha256: str
    size_bytes: int
    detected_license_identifiers: tuple[str, ...]
    scope: str
    notes: str
    relevant_line_numbers: tuple[int, ...] = ()


@dataclass(frozen=True)
class EntityClassCount:
    entity_class: str
    standalone_subject_count: int


@dataclass(frozen=True)
class ActionCount:
    normalized_action: str
    sequence_count: int
    frame_occurrence_count: int


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    related_names: tuple[str, ...]


@dataclass(frozen=True)
class OpenSurgeCounts:
    zip_member_count: int
    file_member_count: int
    archive_png_file_count: int
    sprite_script_file_count: int
    sprite_script_with_header_license_count: int
    sprite_definition_count: int
    regular_animation_count: int
    transition_count: int
    invalid_transition_endpoint_count: int
    total_timeline_count: int
    frame_occurrence_count: int
    repeated_frame_occurrence_count: int
    repeat_true_count: int
    repeat_false_count: int
    repeat_from_declaration_count: int
    comment_labeled_timeline_count: int
    normalized_action_timeline_count: int
    unresolved_action_timeline_count: int
    source_sheet_reference_count: int
    unique_source_sheet_count: int
    missing_source_sheet_count: int
    copyright_data_row_count: int
    copyright_image_row_count: int
    credited_unique_source_sheet_count: int
    uncredited_unique_source_sheet_count: int
    source_rect_out_of_image_count: int
    source_rect_grid_incompatible_count: int
    invalid_declared_frame_index_count: int
    referenced_frame_out_of_image_occurrence_count: int
    standalone_character_subject_count: int
    enemy_character_subject_count: int
    boss_character_subject_count: int
    animal_character_subject_count: int
    creature_character_subject_count: int
    quadruped_character_subject_count: int


@dataclass(frozen=True)
class OpenSurgeAudit:
    archive_path: str
    archive_sha256: str
    repository_commit: str | None
    repository_url: str
    commit_url: str | None
    root_prefix: str
    counts: OpenSurgeCounts
    scripts: tuple[SpriteScriptEvidence, ...]
    sprites: tuple[SpriteDefinition, ...]
    source_sheets: tuple[SourceSheetAudit, ...]
    asset_credits: tuple[AssetCredit, ...]
    entity_classes: tuple[EntityClassCount, ...]
    actions: tuple[ActionCount, ...]
    evidence_documents: tuple[EvidenceDocument, ...]
    issues: tuple[AuditIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    line: int


@dataclass(frozen=True)
class _Statement:
    identifier: str
    args: tuple[str, ...]
    block: tuple[_Statement, ...] | None
    leading_comments: tuple[str, ...]
    inline_comments: tuple[str, ...]
    line: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_relative_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    parsed = PurePosixPath(normalized)
    parts = tuple(part for part in parsed.parts if part not in {"", "."})
    if parsed.is_absolute() or not parts or ".." in parts or ":" in parts[0]:
        raise OpenSurgeArchiveError(f"unsafe or empty Open Surge path: {path!r}")
    return PurePosixPath(*parts).as_posix()


def _lex(source: str) -> tuple[_Token, ...]:
    tokens: list[_Token] = []
    index = 0
    line = 1
    while index < len(source):
        char = source[index]
        if char in " \t\f\v":
            index += 1
            continue
        if char == "\r" or char == "\n":
            if char == "\r" and index + 1 < len(source) and source[index + 1] == "\n":
                index += 1
            tokens.append(_Token("newline", "\n", line))
            line += 1
            index += 1
            continue
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            if end < 0:
                end = len(source)
            text = source[index + 2 : end].rstrip("\r").strip()
            tokens.append(_Token("comment", text, line))
            index = end
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                raise OpenSurgeSpriteParseError(f"unterminated block comment at line {line}")
            text = source[index + 2 : end]
            tokens.append(_Token("comment", text.strip(), line))
            line += text.count("\n")
            index = end + 2
            continue
        if char == "{":
            tokens.append(_Token("lbrace", char, line))
            index += 1
            continue
        if char == "}":
            tokens.append(_Token("rbrace", char, line))
            index += 1
            continue
        if char in {'"', "'"}:
            quote = char
            start_line = line
            index += 1
            value: list[str] = []
            while index < len(source) and source[index] != quote:
                if source[index] == "\\" and index + 1 < len(source):
                    value.extend((source[index], source[index + 1]))
                    index += 2
                    continue
                if source[index] in "\r\n":
                    raise OpenSurgeSpriteParseError(
                        f"newline in quoted string beginning at line {start_line}"
                    )
                value.append(source[index])
                index += 1
            if index >= len(source):
                raise OpenSurgeSpriteParseError(
                    f"unterminated quoted string beginning at line {start_line}"
                )
            index += 1
            tokens.append(_Token("atom", "".join(value), start_line))
            continue

        start = index
        while index < len(source):
            if source[index].isspace() or source[index] in "{}":
                break
            if source.startswith("//", index) or source.startswith("/*", index):
                break
            index += 1
        if index == start:
            raise OpenSurgeSpriteParseError(f"cannot tokenize character at line {line}")
        tokens.append(_Token("atom", source[start:index], line))
    return tuple(tokens)


class _Parser:
    def __init__(self, tokens: Sequence[_Token]) -> None:
        self._tokens = tokens
        self._position = 0

    def parse(self) -> tuple[_Statement, ...]:
        result = self._parse_program(expect_closing_brace=False)
        if self._position != len(self._tokens):
            token = self._tokens[self._position]
            raise OpenSurgeSpriteParseError(f"unexpected token at line {token.line}")
        return result

    def _parse_program(self, *, expect_closing_brace: bool) -> tuple[_Statement, ...]:
        statements: list[_Statement] = []
        pending_comments: list[str] = []
        while self._position < len(self._tokens):
            token = self._tokens[self._position]
            if token.kind == "newline":
                self._position += 1
                continue
            if token.kind == "comment":
                pending_comments.append(token.value)
                self._position += 1
                continue
            if token.kind == "rbrace":
                if not expect_closing_brace:
                    raise OpenSurgeSpriteParseError(
                        f"unexpected closing brace at line {token.line}"
                    )
                self._position += 1
                return tuple(statements)
            if token.kind != "atom":
                raise OpenSurgeSpriteParseError(
                    f"expected statement identifier at line {token.line}"
                )

            identifier = token.value
            line = token.line
            self._position += 1
            args: list[str] = []
            inline_comments: list[str] = []
            while self._position < len(self._tokens):
                current = self._tokens[self._position]
                if current.kind == "atom":
                    args.append(current.value)
                    self._position += 1
                    continue
                if current.kind == "comment":
                    inline_comments.append(current.value)
                    self._position += 1
                    continue
                break

            is_block = identifier.casefold() in _BLOCK_IDENTIFIERS
            if is_block:
                while self._position < len(self._tokens):
                    current = self._tokens[self._position]
                    if current.kind == "newline":
                        self._position += 1
                    elif current.kind == "comment":
                        inline_comments.append(current.value)
                        self._position += 1
                    else:
                        break
                if (
                    self._position >= len(self._tokens)
                    or self._tokens[self._position].kind != "lbrace"
                ):
                    raise OpenSurgeSpriteParseError(
                        f"missing block for {identifier!r} at line {line}"
                    )
                self._position += 1
                block = self._parse_program(expect_closing_brace=True)
            else:
                block = None
                if (
                    self._position < len(self._tokens)
                    and self._tokens[self._position].kind == "lbrace"
                ):
                    raise OpenSurgeSpriteParseError(
                        f"unexpected block for {identifier!r} at line {line}"
                    )

            statements.append(
                _Statement(
                    identifier=identifier,
                    args=tuple(args),
                    block=block,
                    leading_comments=tuple(pending_comments),
                    inline_comments=tuple(inline_comments),
                    line=line,
                )
            )
            pending_comments.clear()

            if not is_block and self._position < len(self._tokens):
                current = self._tokens[self._position]
                if current.kind not in {"newline", "rbrace"}:
                    raise OpenSurgeSpriteParseError(
                        f"property {identifier!r} does not end at line {line}"
                    )

        if expect_closing_brace:
            raise OpenSurgeSpriteParseError("unterminated declaration block")
        return tuple(statements)


def _single_statement(
    statements: Sequence[_Statement],
    name: str,
    *,
    required: bool,
) -> _Statement | None:
    matches = [item for item in statements if item.identifier.casefold() == name.casefold()]
    if len(matches) > 1:
        raise OpenSurgeSpriteParseError(f"duplicate {name!r} property")
    if required and not matches:
        raise OpenSurgeSpriteParseError(f"missing required {name!r} property")
    return matches[0] if matches else None


def _integer_values(statement: _Statement, count: int) -> tuple[int, ...]:
    if statement.block is not None or len(statement.args) != count:
        raise OpenSurgeSpriteParseError(
            f"{statement.identifier!r} at line {statement.line} requires {count} integers"
        )
    try:
        return tuple(int(value, 10) for value in statement.args)
    except ValueError as error:
        raise OpenSurgeSpriteParseError(
            f"non-integer {statement.identifier!r} value at line {statement.line}"
        ) from error


def _point(statement: _Statement) -> Point:
    x, y = _integer_values(statement, 2)
    return Point(x, y)


def _rectangle(statement: _Statement) -> Rectangle:
    x, y, width, height = _integer_values(statement, 4)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise OpenSurgeSpriteParseError(
            f"invalid source rectangle at line {statement.line}: {statement.args!r}"
        )
    return Rectangle(x, y, width, height)


def _source_label(statement: _Statement) -> tuple[str | None, str]:
    for comment in reversed(statement.inline_comments):
        cleaned = _semantic_comment(comment)
        if cleaned is not None:
            return cleaned, "inline_comment"
    for comment in reversed(statement.leading_comments):
        cleaned = _semantic_comment(comment)
        if cleaned is not None:
            return cleaned, "preceding_comment"
    return None, "unlabeled"


def _semantic_comment(comment: str) -> str | None:
    value = " ".join(comment.strip().split())
    if not value:
        return None
    if re.fullmatch(r"[-=_* ]+", value):
        return None
    if not re.search(r"[A-Za-z0-9<>]", value):
        return None
    if re.match(r"(?i)^(file|description|author|license)\s*:", value):
        return None
    if re.match(r"(?i)^(?:sprite |boss )?art(?:work)?\s+by\b", value):
        return None
    if re.match(r"(?i)^concept art\s+by\b", value):
        return None
    if value.casefold().startswith("todo "):
        return None
    return value


def interpret_action_comment(label: str | None) -> tuple[str | None, str, str | None, str | None]:
    """Conservatively interpret a source comment.

    Returns ``(normalized_action, basis, direction_hint, variant_hint)``.  A
    comment outside the explicit vocabulary is returned as unresolved by the
    caller and is never guessed from its animation ID or sprite filename.
    """

    if label is None:
        return None, "unlabeled", None, None
    compact = " ".join(label.casefold().strip().split())
    direction: str | None = None
    if re.search(r"\bright\b", compact):
        direction = "right"
    elif re.search(r"\bleft\b", compact):
        direction = "left"

    animal_match = re.fullmatch(r"animal\s+(\d+)\s*:\s*(appearing|running|stopped)", compact)
    if animal_match:
        action = {"appearing": "spawn", "running": "run", "stopped": "idle"}[animal_match.group(2)]
        return action, "structured_comment_mapping", direction, f"animal {animal_match.group(1)}"

    exact = {
        "appearing": "spawn",
        "back to the ground": "land",
        "breathing": "idle",
        "crouch": "crouch",
        "cry walk": "walk",
        "dead": "death",
        "defeated": "death",
        "disappearing": "despawn",
        "drowned": "death",
        "ducking": "crouch",
        "fall": "fall",
        "falling": "fall",
        "flap (far)": "fly",
        "flap (near)": "fly",
        "gasp": "emote",
        "getting hit": "hurt",
        "got hit": "hurt",
        "hovering (far)": "hover",
        "hovering (near)": "hover",
        "idle": "idle",
        "jump": "jump",
        "jumping": "jump",
        "pushing": "push",
        "right glide": "fly",
        "left glide": "fly",
        "right wing-flap": "fly",
        "left wing-flap": "fly",
        "roar": "emote",
        "running": "run",
        "shooting": "shoot",
        "springing": "jump",
        "stop and wail": "emote",
        "stopped": "idle",
        "waiting": "idle",
        "walking": "walk",
        "winning (victory)": "celebrate",
    }
    if compact in exact:
        return exact[compact], "exact_comment_mapping", direction, None
    if re.fullmatch(r"hit!?(?:\s*\(in use\))?", compact):
        return "hurt", "structured_comment_mapping", direction, None
    if re.fullmatch(r"falling\s*\((?:far|near)\)", compact):
        return "fall", "structured_comment_mapping", direction, None
    if re.fullmatch(r"idle\s*\([^)]+\)", compact):
        return "idle", "structured_comment_mapping", direction, None
    return None, "unresolved_comment", direction, None


def _animation_definition(
    statement: _Statement,
    *,
    source_rect: Rectangle,
    frame_size: Point,
    default_hot_spot: Point,
    default_action_spot: Point,
    member_path: str,
    transition_ordinal: int | None,
) -> AnimationDefinition:
    kind = statement.identifier.casefold()
    if statement.block is None:
        raise OpenSurgeSpriteParseError(f"missing {kind} block at line {statement.line}")
    if kind == "animation":
        if len(statement.args) > 1:
            raise OpenSurgeSpriteParseError(
                f"animation at line {statement.line} accepts zero or one numeric ID"
            )
        try:
            animation_id = int(statement.args[0], 10) if statement.args else 0
        except ValueError as error:
            raise OpenSurgeSpriteParseError(
                f"non-numeric animation ID at line {statement.line}"
            ) from error
        if animation_id < 0:
            raise OpenSurgeSpriteParseError(f"negative animation ID at line {statement.line}")
        transition_from = transition_to = None
    elif kind == "transition":
        if len(statement.args) != 3 or statement.args[1].casefold() != "to":
            raise OpenSurgeSpriteParseError(
                f"transition at line {statement.line} must be FROM to TO"
            )
        animation_id = None
        transition_from = _transition_endpoint(statement.args[0], statement.line)
        transition_to = _transition_endpoint(statement.args[2], statement.line)
        if transition_from == "any" and transition_to == "any":
            raise OpenSurgeSpriteParseError(
                f"transition cannot be any-to-any at line {statement.line}"
            )
        if transition_from == transition_to:
            raise OpenSurgeSpriteParseError(
                f"transition cannot have identical endpoints at line {statement.line}"
            )
    else:
        raise OpenSurgeSpriteParseError(f"unsupported timeline kind {kind!r}")

    repeat_statement = _single_statement(statement.block, "repeat", required=False)
    repeat = False
    if repeat_statement is not None:
        if len(repeat_statement.args) != 1:
            raise OpenSurgeSpriteParseError(
                f"repeat at line {repeat_statement.line} requires one boolean"
            )
        repeat_token = repeat_statement.args[0].casefold()
        if repeat_token not in {"true", "false"}:
            raise OpenSurgeSpriteParseError(f"invalid repeat value at line {repeat_statement.line}")
        repeat = repeat_token == "true"

    fps_statement = _single_statement(statement.block, "fps", required=False)
    fps_source = "8"
    fps = 8.0
    if fps_statement is not None:
        if len(fps_statement.args) != 1:
            raise OpenSurgeSpriteParseError(f"fps at line {fps_statement.line} requires one number")
        fps_source = fps_statement.args[0]
        try:
            fps = float(fps_source)
        except ValueError as error:
            raise OpenSurgeSpriteParseError(
                f"invalid fps value at line {fps_statement.line}"
            ) from error
        if fps <= 0:
            raise OpenSurgeSpriteParseError(f"non-positive fps at line {fps_statement.line}")

    data_statement = _single_statement(statement.block, "data", required=True)
    assert data_statement is not None
    if data_statement.block is not None or not data_statement.args:
        raise OpenSurgeSpriteParseError(f"empty data at line {data_statement.line}")
    try:
        data = tuple(int(value, 10) for value in data_statement.args)
    except ValueError as error:
        raise OpenSurgeSpriteParseError(
            f"non-numeric frame index at line {data_statement.line}"
        ) from error
    if any(index < 0 for index in data):
        raise OpenSurgeSpriteParseError(f"negative frame index at line {data_statement.line}")

    repeat_from_statement = _single_statement(statement.block, "repeat_from", required=False)
    repeat_from = 0
    if repeat_from_statement is not None:
        repeat_from = _integer_values(repeat_from_statement, 1)[0]
        if repeat_from < 0:
            raise OpenSurgeSpriteParseError(
                f"negative repeat_from at line {repeat_from_statement.line}"
            )
    effective_repeat = repeat and kind != "transition"
    effective_repeat_from = repeat_from if effective_repeat else 0
    if effective_repeat_from >= len(data):
        effective_repeat_from = len(data) - 1

    hot_statement = _single_statement(statement.block, "hot_spot", required=False)
    action_statement = _single_statement(statement.block, "action_spot", required=False)
    play_statement = _single_statement(statement.block, "play", required=False)
    if play_statement is not None and len(play_statement.args) != 1:
        raise OpenSurgeSpriteParseError(f"play at line {play_statement.line} requires one name")
    hot_spot = _point(hot_statement) if hot_statement else default_hot_spot
    action_spot = _point(action_statement) if action_statement else default_action_spot

    label, label_basis = _source_label(statement)
    normalized, normalized_basis, direction, variant = interpret_action_comment(label)
    if kind == "transition" and normalized is not None:
        normalized = None
        normalized_basis = "transition_comment_preserved_without_action_projection"

    columns = source_rect.width // frame_size.x
    rows = source_rect.height // frame_size.y
    capacity = columns * rows
    occurrences: list[FrameOccurrence] = []
    for occurrence_index, source_index in enumerate(data):
        column = source_index % columns if columns else 0
        row = source_index // columns if columns else 0
        left = source_rect.x + column * frame_size.x
        top = source_rect.y + row * frame_size.y
        occurrences.append(
            FrameOccurrence(
                occurrence_index=occurrence_index,
                source_frame_index=source_index,
                column=column,
                row=row,
                left=left,
                top=top,
                right=left + frame_size.x,
                bottom=top + frame_size.y,
                in_loop_tail=effective_repeat and occurrence_index >= effective_repeat_from,
                within_declared_source_rect=source_index < capacity,
                within_source_image=None,
            )
        )

    unknown = tuple(
        RawProperty(item.identifier, item.args, item.line)
        for item in statement.block
        if item.identifier.casefold() not in _ANIMATION_PROPERTIES
    )
    comments = tuple(
        value
        for value in (*statement.leading_comments, *statement.inline_comments)
        if value.strip()
    )
    intro_data = data[:effective_repeat_from] if effective_repeat else data
    loop_data = data[effective_repeat_from:] if effective_repeat else ()
    return AnimationDefinition(
        declaration_kind=kind,  # type: ignore[arg-type]
        animation_id=animation_id,
        transition_from=transition_from,
        transition_to=transition_to,
        transition_ordinal=transition_ordinal,
        source_label=label,
        source_label_basis=label_basis,
        normalized_action=normalized,
        normalized_action_basis=normalized_basis,
        direction_hint=direction,
        source_variant_hint=variant,
        repeat=repeat,
        repeat_was_explicit=repeat_statement is not None,
        effective_repeat=effective_repeat,
        repeat_from=repeat_from,
        repeat_from_was_explicit=repeat_from_statement is not None,
        effective_repeat_from=effective_repeat_from,
        fps=fps,
        fps_source_token=fps_source,
        fps_was_explicit=fps_statement is not None,
        data=data,
        intro_data=intro_data,
        loop_data=loop_data,
        hot_spot=hot_spot,
        hot_spot_overridden=hot_statement is not None,
        action_spot=action_spot,
        action_spot_overridden=action_statement is not None,
        programmatic_animation_name=play_statement.args[0] if play_statement else None,
        frame_occurrences=tuple(occurrences),
        comments=comments,
        unknown_properties=unknown,
        evidence_member_path=member_path,
        line_number=statement.line,
    )


def _transition_endpoint(value: str, line: int) -> TransitionEndpoint:
    if value.casefold() == "any":
        return "any"
    try:
        endpoint = int(value, 10)
    except ValueError as error:
        raise OpenSurgeSpriteParseError(f"invalid transition endpoint at line {line}") from error
    if endpoint < 0:
        raise OpenSurgeSpriteParseError(f"negative transition endpoint at line {line}")
    return endpoint


def _header_values(source: str, field: str) -> tuple[str, ...]:
    return tuple(
        match.group(1).strip()
        for match in re.finditer(
            rf"(?im)^\s*//\s*{re.escape(field)}\s*:\s*(.*?)\s*$",
            source,
        )
        if match.group(1).strip()
    )


def _artwork_comments(source: str) -> tuple[str, ...]:
    result: list[str] = []
    for match in re.finditer(r"(?im)^\s*//\s*(.*?)\s*$", source):
        value = " ".join(match.group(1).split())
        if re.search(
            r"(?i)\b(?:art|artwork|sprite art|boss art|concept art)\b.*\bby\b",
            value,
        ):
            result.append(value)
    return tuple(dict.fromkeys(result))


def parse_copyright_data(
    payload: bytes | str,
    *,
    evidence_member_path: str = _COPYRIGHT_DATA_PATH,
) -> tuple[AssetCredit, ...]:
    """Parse the semicolon-delimited source-of-truth attribution manifest."""

    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    expected = ["Type", "File", "License", "Author", "Website", "Notes"]
    if reader.fieldnames != expected:
        raise OpenSurgeSpriteParseError(
            f"unexpected copyright_data.csv columns: {reader.fieldnames!r}"
        )
    rows: list[AssetCredit] = []
    seen: set[tuple[str, str]] = set()
    for line_number, row in enumerate(reader, 2):
        asset_type = (row.get("Type") or "").strip()
        file_path = _normalize_relative_path((row.get("File") or "").strip())
        license_expression = (row.get("License") or "").strip()
        author = (row.get("Author") or "").strip()
        if not asset_type or not license_expression or not author:
            raise OpenSurgeSpriteParseError(
                f"incomplete copyright row at {evidence_member_path}:{line_number}"
            )
        key = (asset_type, file_path)
        if key in seen:
            raise OpenSurgeSpriteParseError(f"duplicate copyright row for {asset_type}:{file_path}")
        seen.add(key)
        website = (row.get("Website") or "").strip() or None
        notes = (row.get("Notes") or "").strip() or None
        rows.append(
            AssetCredit(
                asset_type=asset_type,
                file_path=file_path,
                license_expression=license_expression,
                author=author,
                website=website,
                notes=notes,
                evidence_member_path=evidence_member_path,
                line_number=line_number,
            )
        )
    return tuple(rows)


def classify_sprite_entity(relative_script_path: str, identity: str) -> EntityClassification:
    """Classify a pinned source identity without inferring from animation IDs."""

    path = _normalize_relative_path(relative_script_path)
    lowered = identity.casefold()

    explicit: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
        "giant wolf": (("animal",), ("boss", "quadruped", "wolf"), "boss_character"),
        "hydra": (("creature",), ("boss", "multiheaded", "serpentine"), "boss_character"),
        "salamander boss": (
            ("animal", "creature"),
            ("boss", "salamander"),
            "boss_character",
        ),
        "salamander boss defeated": (
            ("animal", "creature"),
            ("boss", "defeat_pose", "salamander"),
            "boss_character_variant",
        ),
        "crococopter": (
            ("creature", "robot"),
            ("enemy", "flying", "hybrid"),
            "enemy_character",
        ),
        "fish": (("animal",), ("enemy", "aquatic", "fish"), "enemy_character"),
        "swoopharrier": (
            ("animal", "creature"),
            ("enemy", "flying", "bird_like"),
            "enemy_character",
        ),
        "jumping fish": (
            ("animal",),
            ("enemy", "aquatic", "fish", "multi_variant_sheet"),
            "enemy_character_collection",
        ),
        "lady bugsy": (
            ("creature", "animal"),
            ("enemy", "insect_like"),
            "enemy_character",
        ),
        "greenmarmot": (
            ("animal",),
            ("enemy", "marmot", "quadruped"),
            "enemy_character",
        ),
        "redmarmot": (
            ("animal",),
            ("enemy", "marmot", "quadruped"),
            "enemy_character",
        ),
        "mosquito": (
            ("animal",),
            ("enemy", "flying", "insect"),
            "enemy_character",
        ),
        "rulersalamander": (
            ("animal", "creature"),
            ("enemy", "salamander"),
            "enemy_character",
        ),
        "springfling": (("creature",), ("enemy",), "enemy_character"),
        "wolfey": (
            ("animal",),
            ("enemy", "quadruped", "wolf"),
            "enemy_character",
        ),
        "skaterbug": (
            ("creature", "animal"),
            ("friend", "insect_like"),
            "friend_character",
        ),
        "animal": (
            ("animal",),
            ("multi_subject_sheet",),
            "character_collection",
        ),
        "sd_animal": (
            ("animal",),
            ("legacy", "multi_subject_sheet"),
            "character_collection",
        ),
        "surge": (
            ("animal", "humanoid"),
            ("anthropomorphic", "biped", "player", "rabbit"),
            "player_character",
        ),
        "tux": (
            ("animal", "humanoid"),
            ("biped", "penguin", "player"),
            "player_character",
        ),
        "charge": (
            ("animal", "humanoid"),
            ("anthropomorphic", "badger", "biped", "player"),
            "player_character",
        ),
        "neon": (
            ("animal", "humanoid"),
            ("anthropomorphic", "biped", "player", "squirrel"),
            "player_character",
        ),
    }
    if lowered in explicit:
        candidates, tags, role = explicit[lowered]
        if lowered == "surge":
            basis = "repository_readme_names_surge_the_rabbit"
        elif lowered in {"charge", "neon"}:
            basis = "official_project_character_species_statement_and_pinned_player_path"
        else:
            basis = "pinned_source_identity_and_script_path"
        return EntityClassification(candidates[0], candidates, role, tags, None, basis)

    parent: str | None = None
    for prefix in ("giant wolf", "hydra", "salamander boss", "lady bugsy", "rulersalamander"):
        if lowered.startswith(prefix + "'") or lowered.startswith(prefix + " "):
            parent = prefix.title()
            break
    effect_words = ("bullet", "impact", "lightning", "mask", "orb", "spark")
    if parent is not None:
        role = "effect" if any(word in lowered for word in effect_words) else "character_component"
        return EntityClassification(
            "effect" if role == "effect" else "unknown",
            ("effect",) if role == "effect" else (),
            role,
            ("boss_related",),
            parent,
            "component_name_and_pinned_boss_script_path",
        )
    if lowered in {"lady bugsy bullet", "redmarmotchain", "rulersalamandershock"}:
        role = "effect" if lowered != "redmarmotchain" else "character_component"
        return EntityClassification(
            "effect" if role == "effect" else "unknown",
            ("effect",) if role == "effect" else (),
            role,
            ("enemy_related",),
            None,
            "component_name_and_pinned_enemy_script_path",
        )

    if path.startswith("sprites/players/"):
        return EntityClassification(
            "object",
            ("object",),
            "player_related_ui_or_effect",
            (),
            None,
            "pinned_script_directory_and_identity",
        )
    if path.startswith("sprites/enemies/"):
        return EntityClassification(
            "unknown",
            (),
            "enemy_related_unknown",
            ("enemy",),
            None,
            "pinned_enemy_script_path_only",
        )
    if path.startswith("sprites/bosses/"):
        return EntityClassification(
            "unknown",
            (),
            "boss_related_unknown",
            ("boss",),
            None,
            "pinned_boss_script_path_only",
        )
    if path.startswith(("sprites/items/", "sprites/legacy/items/")):
        return EntityClassification(
            "object", ("object",), "object_or_item", (), None, "pinned_script_directory"
        )
    if path.startswith(("sprites/scenes/", "sprites/ui/", "sprites/legacy/hud/")):
        return EntityClassification(
            "object", ("object",), "ui_or_scene", (), None, "pinned_script_directory"
        )
    if path.startswith(("sprites/misc/", "sprites/legacy/fx/")):
        return EntityClassification(
            "effect", ("effect",), "effect", (), None, "pinned_script_directory"
        )
    return EntityClassification(
        "unknown", (), "non_character_or_unresolved", (), None, "unresolved_source_role"
    )


def parse_sprite_script(
    payload: bytes | str,
    *,
    relative_path: str,
    member_path: str | None = None,
    source_image_sizes: Mapping[str, tuple[int, int]] | None = None,
    image_credits: Mapping[str, AssetCredit] | None = None,
) -> tuple[SpriteDefinition, ...]:
    """Parse all sprite blocks in one source file without filesystem access."""

    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    relative_path = _normalize_relative_path(relative_path)
    evidence_path = member_path or relative_path
    statements = _Parser(_lex(text)).parse()
    unexpected_top = [item for item in statements if item.identifier.casefold() != "sprite"]
    if unexpected_top:
        first = unexpected_top[0]
        raise OpenSurgeSpriteParseError(
            f"unexpected top-level {first.identifier!r} at {evidence_path}:{first.line}"
        )

    header_authors = _header_values(text, "Author")
    header_licenses = _header_values(text, "License")
    sprites: list[SpriteDefinition] = []
    seen_identities: set[str] = set()
    for statement in statements:
        if len(statement.args) != 1 or statement.block is None:
            raise OpenSurgeSpriteParseError(
                f"sprite at {evidence_path}:{statement.line} requires one identity and a block"
            )
        identity = statement.args[0]
        folded_identity = identity.casefold()
        if folded_identity in seen_identities:
            raise OpenSurgeSpriteParseError(f"duplicate sprite identity {identity!r}")
        seen_identities.add(folded_identity)

        source_statement = _single_statement(statement.block, "source_file", required=True)
        rect_statement = _single_statement(statement.block, "source_rect", required=True)
        frame_statement = _single_statement(statement.block, "frame_size", required=True)
        hot_statement = _single_statement(statement.block, "hot_spot", required=False)
        action_statement = _single_statement(statement.block, "action_spot", required=False)
        assert source_statement and rect_statement and frame_statement
        if source_statement.block is not None or len(source_statement.args) != 1:
            raise OpenSurgeSpriteParseError(
                f"source_file at {evidence_path}:{source_statement.line} requires one path"
            )
        source_file = _normalize_relative_path(source_statement.args[0])
        source_rect = _rectangle(rect_statement)
        frame_size = _point(frame_statement)
        if frame_size.x <= 0 or frame_size.y <= 0:
            raise OpenSurgeSpriteParseError(
                f"non-positive frame_size at {evidence_path}:{frame_statement.line}"
            )
        hot_spot = _point(hot_statement) if hot_statement else Point(0, 0)
        action_spot = _point(action_statement) if action_statement else Point(0, 0)

        animations: list[AnimationDefinition] = []
        transitions: list[AnimationDefinition] = []
        seen_animation_ids: set[int] = set()
        for child in statement.block:
            kind = child.identifier.casefold()
            if kind == "animation":
                parsed = _animation_definition(
                    child,
                    source_rect=source_rect,
                    frame_size=frame_size,
                    default_hot_spot=hot_spot,
                    default_action_spot=action_spot,
                    member_path=evidence_path,
                    transition_ordinal=None,
                )
                assert parsed.animation_id is not None
                if parsed.animation_id in seen_animation_ids:
                    raise OpenSurgeSpriteParseError(
                        f"duplicate animation ID {parsed.animation_id} in sprite {identity!r}"
                    )
                seen_animation_ids.add(parsed.animation_id)
                animations.append(parsed)
            elif kind == "transition":
                transitions.append(
                    _animation_definition(
                        child,
                        source_rect=source_rect,
                        frame_size=frame_size,
                        default_hot_spot=hot_spot,
                        default_action_spot=action_spot,
                        member_path=evidence_path,
                        transition_ordinal=len(transitions),
                    )
                )
        if not animations:
            raise OpenSurgeSpriteParseError(f"sprite {identity!r} has no regular animation")

        columns = source_rect.width // frame_size.x
        rows = source_rect.height // frame_size.y
        capacity = columns * rows
        if capacity <= 0:
            raise OpenSurgeSpriteParseError(f"sprite {identity!r} has an empty declared frame grid")
        timelines = (*animations, *transitions)
        declared_valid = all(
            occurrence.within_declared_source_rect
            for timeline in timelines
            for occurrence in timeline.frame_occurrences
        )

        image_size = source_image_sizes.get(source_file) if source_image_sizes else None
        source_exists: bool | None = (
            image_size is not None if source_image_sizes is not None else None
        )
        rect_within: bool | None = None
        frames_within: bool | None = None
        if image_size is not None:
            width, height = image_size
            rect_within = source_rect.right <= width and source_rect.bottom <= height
            frames_within = True
            updated_timelines: list[AnimationDefinition] = []
            for timeline in timelines:
                frame_occurrences = tuple(
                    replace(
                        occurrence,
                        within_source_image=(
                            occurrence.left >= 0
                            and occurrence.top >= 0
                            and occurrence.right <= width
                            and occurrence.bottom <= height
                        ),
                    )
                    for occurrence in timeline.frame_occurrences
                )
                frames_within = frames_within and all(
                    occurrence.within_source_image is True for occurrence in frame_occurrences
                )
                updated_timelines.append(replace(timeline, frame_occurrences=frame_occurrences))
            animations = updated_timelines[: len(animations)]
            transitions = updated_timelines[len(animations) :]

        unknown = tuple(
            RawProperty(item.identifier, item.args, item.line)
            for item in statement.block
            if item.identifier.casefold() not in _SPRITE_PROPERTIES
        )
        source_comments = tuple(
            value
            for value in (*statement.leading_comments, *statement.inline_comments)
            if value.strip()
        )
        sprites.append(
            SpriteDefinition(
                identity=identity,
                source_file=source_file,
                source_rect=source_rect,
                frame_size=frame_size,
                hot_spot=hot_spot,
                action_spot=action_spot,
                source_sheet_columns=columns,
                source_sheet_rows=rows,
                source_sheet_frame_capacity=capacity,
                source_image_width=image_size[0] if image_size else None,
                source_image_height=image_size[1] if image_size else None,
                source_file_exists=source_exists,
                source_rect_within_image=rect_within,
                source_rect_grid_compatible=(
                    source_rect.width % frame_size.x == 0 and source_rect.height % frame_size.y == 0
                ),
                referenced_frames_within_declared_grid=declared_valid,
                referenced_frames_within_image=frames_within,
                animations=tuple(animations),
                transitions=tuple(transitions),
                entity=classify_sprite_entity(relative_path, identity),
                asset_credit=image_credits.get(source_file) if image_credits else None,
                source_header_authors=header_authors,
                source_header_licenses=header_licenses,
                source_comments=source_comments,
                unknown_properties=unknown,
                evidence_member_path=evidence_path,
                relative_script_path=relative_path,
                line_number=statement.line,
            )
        )
    return tuple(sprites)


def _zip_root(infos: Sequence[ZipInfo]) -> str:
    roots: set[str] = set()
    seen: set[str] = set()
    for info in infos:
        raw = info.filename.replace("\\", "/")
        path = PurePosixPath(raw)
        parts = tuple(part for part in path.parts if part not in {"", "."})
        if path.is_absolute() or not parts or ".." in parts or ":" in parts[0]:
            raise OpenSurgeArchiveError(f"unsafe ZIP member path: {info.filename!r}")
        normalized = PurePosixPath(*parts).as_posix() + ("/" if info.is_dir() else "")
        folded = normalized.casefold()
        if folded in seen:
            raise OpenSurgeArchiveError(f"duplicate/case-colliding ZIP member: {normalized!r}")
        seen.add(folded)
        roots.add(parts[0])
        if info.flag_bits & 0x1:
            raise OpenSurgeArchiveError(f"encrypted ZIP member: {normalized!r}")
    if len(roots) != 1:
        raise OpenSurgeArchiveError(f"expected one archive root, found {sorted(roots)!r}")
    return next(iter(roots))


def _repository_commit(root: str) -> str | None:
    match = re.fullmatch(r"opensurge-([0-9a-fA-F]{40})", root)
    return match.group(1).lower() if match else None


def _document(
    archive: ZipFile,
    by_relative_path: Mapping[str, ZipInfo],
    root: str,
    relative_path: str,
    *,
    detected: tuple[str, ...],
    scope: str,
    notes: str,
    relevant_line_numbers: tuple[int, ...] = (),
) -> EvidenceDocument:
    info = by_relative_path.get(relative_path)
    if info is None:
        raise OpenSurgeArchiveError(f"missing evidence document: {relative_path}")
    payload = archive.read(info)
    return EvidenceDocument(
        relative_path=relative_path,
        member_path=f"{root}/{relative_path}",
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        detected_license_identifiers=detected,
        scope=scope,
        notes=notes,
        relevant_line_numbers=relevant_line_numbers,
    )


def _pixel_semantics_document(
    archive: ZipFile,
    by_relative_path: Mapping[str, ZipInfo],
    root: str,
    relative_path: str,
    *,
    required_patterns: tuple[tuple[str, re.Pattern[str]], ...],
    scope: str,
    notes: str,
) -> EvidenceDocument:
    """Require and record the exact engine code that establishes color-key behavior."""

    info = by_relative_path.get(relative_path)
    if info is None:
        raise OpenSurgeArchiveError(f"missing evidence document: {relative_path}")
    payload = archive.read(info)
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise OpenSurgeArchiveError(
            f"pixel-semantics evidence is not UTF-8: {relative_path}"
        ) from error
    lines = text.splitlines()
    relevant_lines: list[int] = []
    for claim, pattern in required_patterns:
        matches = tuple(index for index, line in enumerate(lines, start=1) if pattern.search(line))
        if not matches:
            raise OpenSurgeArchiveError(
                f"pixel-semantics evidence {relative_path} does not establish {claim}"
            )
        relevant_lines.extend(matches)
    return EvidenceDocument(
        relative_path=relative_path,
        member_path=f"{root}/{relative_path}",
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        detected_license_identifiers=("GPL-3.0-or-later",),
        scope=scope,
        notes=notes,
        relevant_line_numbers=tuple(sorted(set(relevant_lines))),
    )


def _timeline_occurrence_reference(
    sprite: SpriteDefinition,
    timeline: AnimationDefinition,
    occurrence: FrameOccurrence,
) -> str:
    timeline_id = (
        timeline.animation_id if timeline.animation_id is not None else timeline.transition_ordinal
    )
    return (
        f"{sprite.identity}:{timeline.declaration_kind}:{timeline_id}:"
        f"{occurrence.occurrence_index}={occurrence.source_frame_index}"
    )


def audit_open_surge_archive(archive_path: str | Path) -> OpenSurgeAudit:
    """Audit an Open Surge source archive in place, without extracting it."""

    path = Path(archive_path)
    archive_sha256 = _sha256_file(path)
    try:
        with ZipFile(path) as archive:
            infos = tuple(archive.infolist())
            root = _zip_root(infos)
            prefix = f"{root}/"
            by_relative_path = {
                info.filename.replace("\\", "/")[len(prefix) :]: info
                for info in infos
                if not info.is_dir() and info.filename.replace("\\", "/").startswith(prefix)
            }
            copyright_info = by_relative_path.get(_COPYRIGHT_DATA_PATH)
            if copyright_info is None:
                raise OpenSurgeArchiveError(f"missing {_COPYRIGHT_DATA_PATH}")
            credits = parse_copyright_data(
                archive.read(copyright_info),
                evidence_member_path=f"{root}/{_COPYRIGHT_DATA_PATH}",
            )
            image_credits = {
                credit.file_path: credit for credit in credits if credit.asset_type == "image"
            }

            script_paths = tuple(
                sorted(name for name in by_relative_path if name.casefold().endswith(".spr"))
            )
            script_payloads = {name: archive.read(by_relative_path[name]) for name in script_paths}
            source_paths: list[str] = []
            for payload in script_payloads.values():
                text = payload.decode("utf-8-sig")
                source_paths.extend(
                    match.group(1) or match.group(2)
                    for match in re.finditer(
                        r"(?m)^\s*source_file\s+(?:\"([^\"]+)\"|(\S+))",
                        text,
                    )
                )
            unique_source_paths = tuple(
                sorted({_normalize_relative_path(name) for name in source_paths})
            )

            source_sizes: dict[str, tuple[int, int]] = {}
            sheets: list[SourceSheetAudit] = []
            reference_counts = Counter(source_paths)
            for source_path in unique_source_paths:
                info = by_relative_path.get(source_path)
                if info is None:
                    continue
                payload = archive.read(info)
                try:
                    with Image.open(io.BytesIO(payload)) as image:
                        image.load()
                        width, height = image.size
                        image_mode = image.mode
                        image_format = image.format
                        has_transparency = (
                            "A" in image.getbands()
                            or "transparency" in image.info
                            or image.mode == "P"
                            and "transparency" in image.info
                        )
                except (UnidentifiedImageError, OSError) as error:
                    raise OpenSurgeArchiveError(
                        f"cannot decode referenced source image {source_path}: {error}"
                    ) from error
                source_sizes[source_path] = (width, height)
                sheets.append(
                    SourceSheetAudit(
                        relative_path=source_path,
                        member_path=f"{root}/{source_path}",
                        width=width,
                        height=height,
                        image_mode=image_mode,
                        image_format=image_format,
                        has_transparency=has_transparency,
                        size_bytes=info.file_size,
                        compressed_size_bytes=info.compress_size,
                        crc32=f"{info.CRC:08x}",
                        sha256=_sha256_bytes(payload),
                        sprite_reference_count=reference_counts[source_path],
                        asset_credit=image_credits.get(source_path),
                    )
                )

            sprites: list[SpriteDefinition] = []
            scripts: list[SpriteScriptEvidence] = []
            seen_identities: set[str] = set()
            for script_path in script_paths:
                payload = script_payloads[script_path]
                text = payload.decode("utf-8-sig")
                parsed = parse_sprite_script(
                    payload,
                    relative_path=script_path,
                    member_path=f"{root}/{script_path}",
                    source_image_sizes=source_sizes,
                    image_credits=image_credits,
                )
                for sprite in parsed:
                    key = sprite.identity.casefold()
                    if key in seen_identities:
                        raise OpenSurgeArchiveError(
                            f"duplicate sprite identity across scripts: {sprite.identity!r}"
                        )
                    seen_identities.add(key)
                sprites.extend(parsed)
                scripts.append(
                    SpriteScriptEvidence(
                        relative_path=script_path,
                        member_path=f"{root}/{script_path}",
                        sha256=_sha256_bytes(payload),
                        size_bytes=len(payload),
                        declared_file=(_header_values(text, "File") or (None,))[0],
                        description=(_header_values(text, "Description") or (None,))[0],
                        authors=_header_values(text, "Author"),
                        license_expressions=_header_values(text, "License"),
                        artwork_comments=_artwork_comments(text),
                        sprite_count=len(parsed),
                    )
                )

            license_values = tuple(
                sorted(
                    {
                        credit.license_expression
                        for credit in credits
                        if credit.asset_type == "image"
                    }
                )
            )
            evidence_documents = (
                _document(
                    archive,
                    by_relative_path,
                    root,
                    _COPYRIGHT_DATA_PATH,
                    detected=license_values,
                    scope="asset_level_semicolon_delimited_credit_manifest",
                    notes=(
                        "The image rows are the authoritative per-file license, author, website, "
                        "and notes evidence used for sprite source sheets."
                    ),
                ),
                _document(
                    archive,
                    by_relative_path,
                    root,
                    "LICENSE",
                    detected=("GPL-3.0-only",),
                    scope="repository_root_license_text",
                    notes=(
                        "Repository-level GPL text; do not substitute it for the distinct "
                        "per-image licenses in copyright_data.csv."
                    ),
                ),
                _document(
                    archive,
                    by_relative_path,
                    root,
                    "README.md",
                    detected=("GPL-3.0",),
                    scope="repository_description_and_license_declaration",
                    notes="Also identifies Surge as a rabbit and describes the featured game.",
                ),
                _document(
                    archive,
                    by_relative_path,
                    root,
                    "licenses/MIT-license.txt",
                    detected=("MIT",),
                    scope="bundled_generic_license_template",
                    notes=(
                        "Generic MIT template with placeholder copyright fields; individual "
                        ".spr headers independently declare MIT on 82 scripts."
                    ),
                ),
                _document(
                    archive,
                    by_relative_path,
                    root,
                    "src/core/sprite.c",
                    detected=("GPL-3.0-or-later",),
                    scope="engine_parser_and_row_major_sheet_geometry_evidence",
                    notes=(
                        "Documents source_rect validation, frame-grid capacity, row-major frame "
                        "coordinates, transition parsing, and inherited anchors."
                    ),
                ),
                _document(
                    archive,
                    by_relative_path,
                    root,
                    "src/core/animation.c",
                    detected=("GPL-3.0-or-later",),
                    scope="engine_animation_timing_and_loop_semantics_evidence",
                    notes=(
                        "Documents FPS timing, ordered data lookup, repeat_from behavior, and "
                        "the forced non-repeating transition rule."
                    ),
                ),
                _pixel_semantics_document(
                    archive,
                    by_relative_path,
                    root,
                    _COLOR_ENGINE_PATH,
                    required_patterns=(
                        (
                            "alpha-zero or exact RGB (255,0,255) transparency",
                            re.compile(
                                r"\(\s*a\s*==\s*0\s*\)\s*\|\|\s*\(\s*r\s*==\s*255"
                                r"\s*&&\s*g\s*==\s*0\s*&&\s*b\s*==\s*255\s*\)"
                            ),
                        ),
                    ),
                    scope="engine_exact_magenta_color_key_predicate_evidence",
                    notes=(
                        "color_is_transparent treats alpha zero or exact uint8 RGB "
                        "(255, 0, 255) as transparent; it does not specify a fuzzy threshold."
                    ),
                ),
                _pixel_semantics_document(
                    archive,
                    by_relative_path,
                    root,
                    _SHADER_ENGINE_PATH,
                    required_patterns=(
                        (
                            "the exact normalized magenta mask constant",
                            re.compile(
                                r"MASK_COLOR\s*=\s*vec3\(\s*1\.0\s*,\s*0\.0\s*,\s*1\.0\s*\)"
                            ),
                        ),
                        (
                            "zeroing all sampled components on an exact mask-color match",
                            re.compile(r"p\s*\*=\s*float\(\s*p\.rgb\s*!=\s*MASK_COLOR\s*\)"),
                        ),
                    ),
                    scope="engine_exact_magenta_premultiplied_rgba_zeroing_evidence",
                    notes=(
                        "The default fragment shader compares sampled RGB to exact magenta "
                        "and multiplies all components by zero on a match."
                    ),
                ),
            )
    except (BadZipFile, OSError, UnicodeDecodeError) as error:
        raise OpenSurgeArchiveError(f"cannot read ZIP archive {path}: {error}") from error

    timelines = tuple(
        timeline for sprite in sprites for timeline in (*sprite.animations, *sprite.transitions)
    )
    standalone_roles = {
        "boss_character",
        "boss_character_variant",
        "character_collection",
        "enemy_character",
        "enemy_character_collection",
        "friend_character",
        "player_character",
    }
    standalone = tuple(
        sprite for sprite in sprites if sprite.entity.subject_role in standalone_roles
    )
    entity_counter = Counter(sprite.entity.primary_entity_class for sprite in standalone)
    action_counter: Counter[str] = Counter()
    action_frames: Counter[str] = Counter()
    for timeline in timelines:
        if timeline.normalized_action is not None:
            action_counter[timeline.normalized_action] += 1
            action_frames[timeline.normalized_action] += len(timeline.data)

    missing_source_paths = tuple(sorted(set(unique_source_paths) - set(source_sizes)))
    uncredited_source_paths = tuple(
        sorted(
            source_path for source_path in unique_source_paths if source_path not in image_credits
        )
    )
    source_rect_bad = tuple(
        sprite.identity for sprite in sprites if sprite.source_rect_within_image is False
    )
    grid_bad = tuple(
        sprite.identity for sprite in sprites if not sprite.source_rect_grid_compatible
    )
    declared_index_bad = tuple(
        _timeline_occurrence_reference(sprite, timeline, occurrence)
        for sprite in sprites
        for timeline in (*sprite.animations, *sprite.transitions)
        for occurrence in timeline.frame_occurrences
        if not occurrence.within_declared_source_rect
    )
    image_occurrence_bad = tuple(
        _timeline_occurrence_reference(sprite, timeline, occurrence)
        for sprite in sprites
        for timeline in (*sprite.animations, *sprite.transitions)
        for occurrence in timeline.frame_occurrences
        if occurrence.within_source_image is False
    )
    invalid_transition_endpoints = tuple(
        f"{sprite.identity}:{transition.transition_from}->{transition.transition_to}"
        for sprite in sprites
        for transition in sprite.transitions
        if any(
            endpoint != "any" and endpoint not in {item.animation_id for item in sprite.animations}
            for endpoint in (transition.transition_from, transition.transition_to)
        )
    )
    issues: list[AuditIssue] = []
    if missing_source_paths:
        issues.append(
            AuditIssue(
                "error",
                "referenced_source_sheet_missing",
                "One or more .spr source_file paths are absent from the archive.",
                missing_source_paths,
            )
        )
    if uncredited_source_paths:
        issues.append(
            AuditIssue(
                "warning",
                "referenced_source_sheet_without_asset_credit",
                "A referenced image has no image row in copyright_data.csv.",
                uncredited_source_paths,
            )
        )
    if source_rect_bad:
        issues.append(
            AuditIssue(
                "warning",
                "declared_source_rect_exceeds_image",
                (
                    "Some raw source_rect declarations exceed the current PNG dimensions. "
                    "The engine has runtime adjustment logic; raw declarations remain preserved."
                ),
                tuple(sorted(source_rect_bad)),
            )
        )
    if grid_bad:
        issues.append(
            AuditIssue(
                "warning",
                "source_rect_not_multiple_of_frame_size",
                (
                    "A raw source_rect is not an exact frame grid. Engine adjustment behavior "
                    "must not be confused with the raw declaration."
                ),
                tuple(sorted(grid_bad)),
            )
        )
    if declared_index_bad:
        issues.append(
            AuditIssue(
                "error",
                "data_frame_outside_declared_grid",
                "A data occurrence points beyond the floor-divided declared frame grid.",
                declared_index_bad,
            )
        )
    if image_occurrence_bad:
        issues.append(
            AuditIssue(
                "warning",
                "referenced_frame_cell_exceeds_image",
                (
                    "A row-major cell referenced by data extends beyond the current PNG. Keep "
                    "these records out of pixel extraction until reconciled with engine behavior."
                ),
                image_occurrence_bad,
            )
        )
    if invalid_transition_endpoints:
        issues.append(
            AuditIssue(
                "error",
                "transition_endpoint_animation_missing",
                "A transition endpoint does not resolve to a regular animation in its sprite.",
                invalid_transition_endpoints,
            )
        )
    issues.extend(
        (
            AuditIssue(
                "info",
                "license_scope_is_asset_specific",
                (
                    "Use copyright_data.csv image rows for sheet rights. The repository GPL, "
                    ".spr MIT headers, and underlying PNG licenses have different scopes."
                ),
                (_COPYRIGHT_DATA_PATH, "LICENSE", "licenses/MIT-license.txt"),
            ),
            AuditIssue(
                "info",
                "unresolved_action_comments_preserved",
                (
                    "Comments outside the explicit action vocabulary and unlabeled numeric "
                    "animations remain unnormalized rather than receiving guessed actions."
                ),
                (),
            ),
        )
    )

    counts = OpenSurgeCounts(
        zip_member_count=len(infos),
        file_member_count=sum(not info.is_dir() for info in infos),
        archive_png_file_count=sum(
            not info.is_dir() and info.filename.casefold().endswith(".png") for info in infos
        ),
        sprite_script_file_count=len(scripts),
        sprite_script_with_header_license_count=sum(
            bool(script.license_expressions) for script in scripts
        ),
        sprite_definition_count=len(sprites),
        regular_animation_count=sum(len(sprite.animations) for sprite in sprites),
        transition_count=sum(len(sprite.transitions) for sprite in sprites),
        invalid_transition_endpoint_count=len(invalid_transition_endpoints),
        total_timeline_count=len(timelines),
        frame_occurrence_count=sum(len(timeline.data) for timeline in timelines),
        repeated_frame_occurrence_count=sum(
            len(timeline.data) - len(set(timeline.data)) for timeline in timelines
        ),
        repeat_true_count=sum(timeline.repeat for timeline in timelines),
        repeat_false_count=sum(not timeline.repeat for timeline in timelines),
        repeat_from_declaration_count=sum(
            timeline.repeat_from_was_explicit for timeline in timelines
        ),
        comment_labeled_timeline_count=sum(
            timeline.source_label is not None for timeline in timelines
        ),
        normalized_action_timeline_count=sum(
            timeline.normalized_action is not None for timeline in timelines
        ),
        unresolved_action_timeline_count=sum(
            timeline.normalized_action is None for timeline in timelines
        ),
        source_sheet_reference_count=len(source_paths),
        unique_source_sheet_count=len(unique_source_paths),
        missing_source_sheet_count=len(missing_source_paths),
        copyright_data_row_count=len(credits),
        copyright_image_row_count=sum(credit.asset_type == "image" for credit in credits),
        credited_unique_source_sheet_count=sum(
            source_path in image_credits for source_path in unique_source_paths
        ),
        uncredited_unique_source_sheet_count=len(uncredited_source_paths),
        source_rect_out_of_image_count=len(source_rect_bad),
        source_rect_grid_incompatible_count=len(grid_bad),
        invalid_declared_frame_index_count=len(declared_index_bad),
        referenced_frame_out_of_image_occurrence_count=len(image_occurrence_bad),
        standalone_character_subject_count=len(standalone),
        enemy_character_subject_count=sum(
            sprite.entity.subject_role.startswith("enemy_character") for sprite in standalone
        ),
        boss_character_subject_count=sum(
            sprite.entity.subject_role.startswith("boss_character") for sprite in standalone
        ),
        animal_character_subject_count=entity_counter.get("animal", 0),
        creature_character_subject_count=entity_counter.get("creature", 0),
        quadruped_character_subject_count=sum(
            "quadruped" in sprite.entity.morphology_tags for sprite in standalone
        ),
    )
    return OpenSurgeAudit(
        archive_path=str(path.resolve()),
        archive_sha256=archive_sha256,
        repository_commit=_repository_commit(root),
        repository_url=OPEN_SURGE_REPOSITORY_URL,
        commit_url=(
            f"{OPEN_SURGE_REPOSITORY_URL}/tree/{_repository_commit(root)}"
            if _repository_commit(root)
            else None
        ),
        root_prefix=root,
        counts=counts,
        scripts=tuple(scripts),
        sprites=tuple(sprites),
        source_sheets=tuple(sheets),
        asset_credits=credits,
        entity_classes=tuple(
            EntityClassCount(key, value) for key, value in sorted(entity_counter.items())
        ),
        actions=tuple(
            ActionCount(action, action_counter[action], action_frames[action])
            for action in sorted(action_counter)
        ),
        evidence_documents=evidence_documents,
        issues=tuple(issues),
    )


def audit_known_open_surge_archive(archive_path: str | Path) -> OpenSurgeAudit:
    """Audit the exact pinned archive and reject any payload/root substitution."""

    path = Path(archive_path)
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != EXPECTED_OPEN_SURGE_ARCHIVE_SHA256:
        raise OpenSurgeArchiveError(
            "archive digest mismatch: expected "
            f"{EXPECTED_OPEN_SURGE_ARCHIVE_SHA256}, got {actual_sha256}"
        )
    audit = audit_open_surge_archive(path)
    if audit.repository_commit != OPEN_SURGE_COMMIT:
        raise OpenSurgeArchiveError(
            f"archive root commit mismatch: expected {OPEN_SURGE_COMMIT}, "
            f"got {audit.repository_commit}"
        )
    if audit.root_prefix != _EXPECTED_ROOT:
        raise OpenSurgeArchiveError(
            f"archive root mismatch: expected {_EXPECTED_ROOT!r}, got {audit.root_prefix!r}"
        )
    return audit
