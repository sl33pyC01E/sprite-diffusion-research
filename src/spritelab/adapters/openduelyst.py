"""Read-only, evidence-preserving audit adapter for the pinned OpenDuelyst tree.

OpenDuelyst does not describe its animations as regular grids.  The executable
``app/data/resources.js`` file names a TexturePacker plist and frame prefix for
each animation.  At runtime, ``UtilsResources.getFrameKeys`` selects matching
plist keys and sorts them by the final number in each key; the package manager
then multiplies the declared delay by ``0.8``.  This module reproduces those
semantics and retains the declarations that justify them.

The adapter never extracts archive members and never writes a database or data
artifact.  Classification is intentionally conservative: entity/action links
come from ``setBaseAnimResource`` calls, while direction and looping remain
unknown or role-dependent where the source does not establish one answer.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import plistlib
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cmp_to_key
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

EXPECTED_OPENDUELYST_ARCHIVE_SHA256 = (
    "9d907a2d299b0f1598984192e3d4832aeb770e75fa2507370ff8e66428282f8e"
)
OPENDUELYST_COMMIT = "2843f2400854136598631288c2e8dfb8f5173de7"
OPENDUELYST_REPOSITORY_URL = "https://github.com/open-duelyst/duelyst"
OPENDUELYST_COMMIT_URL = f"{OPENDUELYST_REPOSITORY_URL}/tree/{OPENDUELYST_COMMIT}"

_EXPECTED_ROOT = f"duelyst-{OPENDUELYST_COMMIT}"
_RESOURCES_PATH = "app/data/resources.js"
_CARD_LOOKUP_PATH = "app/sdk/cards/cardsLookup.coffee"
_FACTORY_PREFIX = "app/sdk/cards/factory/"
_LOCALIZATION_PREFIX = "app/localization/locales/en/"
_RUNTIME_DELAY_MULTIPLIER = 0.8

_LOOP_ROLES = frozenset({"idle", "breathing", "walk", "castLoop", "active", "occupied"})
_ONE_SHOT_ROLES = frozenset(
    {"attack", "damage", "death", "castStart", "castEnd", "cast", "apply", "depleted"}
)

Numeric = int | float


class OpenDuelystArchiveError(ValueError):
    """Raised when an archive cannot be audited as the expected repository tree."""


class OpenDuelystParseError(ValueError):
    """Raised when source animation evidence is malformed or contradictory."""


@dataclass(frozen=True)
class RawExpression:
    name: str
    expression: str
    line_number: int


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int
    raw: str

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True)
class Size:
    width: int
    height: int
    raw: str


@dataclass(frozen=True)
class NumericPoint:
    x: float
    y: float
    raw: str


@dataclass(frozen=True)
class AtlasFrame:
    key: str
    declaration_index: int
    frame: Rectangle
    offset: NumericPoint
    rotated: bool
    source_color_rect: Rectangle
    source_size: Size
    within_image_bounds: bool | None
    is_trimmed: bool


@dataclass(frozen=True)
class AtlasSheet:
    relative_path: str
    member_path: str
    image_relative_path: str | None
    image_member_path: str | None
    image_width: int | None
    image_height: int | None
    image_mode: str | None
    image_format: str | None
    image_sha256: str | None
    metadata_format: int | None
    metadata_size: Size | None
    metadata_texture_file_name: str | None
    metadata_real_texture_file_name: str | None
    frames: tuple[AtlasFrame, ...]
    descriptor_reference_count: int
    duplicate_frame_keys: tuple[str, ...]
    metadata_matches_image_size: bool | None

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class ResourceAnimationDeclaration:
    alias: str
    name: str
    frame_prefix: str
    frame_delay: float
    frame_delay_expression: str
    image_path: str
    plist_path: str
    raw_fields: tuple[RawExpression, ...]
    evidence_relative_path: str
    evidence_member_path: str
    line_number: int

    @property
    def effective_frame_delay_seconds(self) -> float:
        return self.frame_delay * _RUNTIME_DELAY_MULTIPLIER


@dataclass(frozen=True)
class EntityAnimationField:
    role: str
    expression: str
    resource_alias: str | None
    numeric_value: float | None
    line_number: int


@dataclass(frozen=True)
class EntityAnimationReference:
    role: str
    resource_alias: str
    expression: str
    evidence_relative_path: str
    evidence_member_path: str
    line_number: int


@dataclass(frozen=True)
class EntityMapping:
    identifier_expression: str | None
    identifier_token: str | None
    card_id: int | None
    card_kind: str | None
    card_name_expression: str | None
    localization_key: str | None
    display_name: str | None
    faction_expression: str | None
    race_expression: str | None
    animation_fields: tuple[EntityAnimationField, ...]
    animation_references: tuple[EntityAnimationReference, ...]
    evidence_relative_path: str
    evidence_member_path: str
    line_number: int


@dataclass(frozen=True)
class SequenceFrame:
    occurrence_index: int
    atlas_declaration_index: int
    key: str
    final_numeric_token: int | None
    frame: Rectangle
    offset: NumericPoint
    rotated: bool
    source_color_rect: Rectangle
    source_size: Size
    within_image_bounds: bool | None
    is_trimmed: bool


@dataclass(frozen=True)
class AnimationSequence:
    resource_alias: str
    runtime_name: str
    category: str
    plist_path: str
    image_path: str
    frame_prefix: str
    declared_frame_delay_seconds: float
    declared_frame_delay_expression: str
    runtime_delay_multiplier: float
    effective_frame_delay_seconds: float
    total_duration_seconds: float
    frames: tuple[SequenceFrame, ...]
    source_roles: tuple[str, ...]
    entity_mapping_indices: tuple[int, ...]
    normalized_action: str | None
    normalized_action_basis: str
    loop_mode: Literal["loop", "one_shot", "role_dependent", "unknown"]
    loop_basis: str
    direction: str | None
    direction_semantics: str
    ambiguity_reasons: tuple[str, ...]
    evidence_relative_path: str
    evidence_member_path: str
    line_number: int

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True)
class GifFrame:
    frame_index: int
    duration_milliseconds: int


@dataclass(frozen=True)
class GifAnimation:
    relative_path: str
    member_path: str
    identity_hint: str
    action_hint: str | None
    width: int
    height: int
    frame_count: int
    frames: tuple[GifFrame, ...]
    loop_value: int | None
    has_transparency: bool
    sha256: str
    size_bytes: int
    timing_authority: str


@dataclass(frozen=True)
class DuplicateGroup:
    kind: str
    keys: tuple[str, ...]
    evidence: str


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
class SourceCodeEvidence:
    relative_path: str
    member_path: str
    sha256: str
    line_numbers: tuple[int, ...]
    establishes: str


@dataclass(frozen=True)
class EmbeddedMetadataSummary:
    png_file_count: int
    readable_png_count: int
    unique_png_sha256_count: int
    duplicate_png_sha256_group_count: int
    duplicate_png_path_excess_count: int
    png_with_xmp_count: int
    png_with_comment_count: int
    gimp_comment_count: int
    png_with_asset_attribution_field_count: int
    attribution_field_names: tuple[str, ...]


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    related_names: tuple[str, ...]


@dataclass(frozen=True)
class OpenDuelystCounts:
    zip_member_count: int
    file_member_count: int
    directory_member_count: int
    archive_png_file_count: int
    unique_png_payload_count: int
    duplicate_png_payload_group_count: int
    archive_gif_file_count: int
    archive_plist_file_count: int
    archive_coffeescript_file_count: int
    resource_entry_count: int
    animation_descriptor_count: int
    unique_descriptor_plist_count: int
    unique_descriptor_image_count: int
    unique_descriptor_plist_prefix_count: int
    texture_atlas_count: int
    atlas_frame_count: int
    rotated_atlas_frame_count: int
    trimmed_atlas_frame_count: int
    nonzero_offset_atlas_frame_count: int
    descriptor_frame_occurrence_count: int
    descriptor_unique_frame_count: int
    resolved_animation_descriptor_count: int
    empty_animation_descriptor_count: int
    atlas_frame_unmatched_by_descriptor_count: int
    unreferenced_atlas_count: int
    unreferenced_atlas_frame_count: int
    entity_mapping_count: int
    entity_animation_reference_count: int
    unique_referenced_resource_alias_count: int
    mapped_resource_alias_multiple_role_count: int
    resource_alias_name_mismatch_count: int
    runtime_name_collision_count: int
    shared_physical_frame_key_count: int
    exact_timeline_alias_group_count: int
    gif_animation_count: int
    unique_gif_payload_count: int
    duplicate_gif_payload_group_count: int
    gif_frame_count: int
    evidence_document_count: int
    plist_parse_error_count: int


@dataclass(frozen=True)
class OpenDuelystAudit:
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    repository_commit: str
    repository_url: str
    commit_url: str
    root_prefix: str
    counts: OpenDuelystCounts
    declarations: tuple[ResourceAnimationDeclaration, ...]
    atlases: tuple[AtlasSheet, ...]
    sequences: tuple[AnimationSequence, ...]
    entity_mappings: tuple[EntityMapping, ...]
    gifs: tuple[GifAnimation, ...]
    duplicate_groups: tuple[DuplicateGroup, ...]
    evidence_documents: tuple[EvidenceDocument, ...]
    source_code_evidence: tuple[SourceCodeEvidence, ...]
    embedded_metadata: EmbeddedMetadataSummary
    issues: tuple[AuditIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively serialisable representation without changing evidence."""

        return asdict(self)


def _line_number(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_text(payload: bytes, relative_path: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise OpenDuelystParseError(f"{relative_path}: expected UTF-8 text") from error


def _skip_space_and_comments(source: str, position: int, end: int) -> int:
    while position < end:
        if source[position].isspace() or source[position] == ",":
            position += 1
            continue
        if source.startswith("//", position):
            newline = source.find("\n", position + 2, end)
            position = end if newline < 0 else newline + 1
            continue
        if source.startswith("/*", position):
            close = source.find("*/", position + 2, end)
            if close < 0:
                raise OpenDuelystParseError("unterminated JavaScript block comment")
            position = close + 2
            continue
        break
    return position


def _scan_string(source: str, position: int, end: int) -> int:
    quote = source[position]
    position += 1
    while position < end:
        if source[position] == "\\":
            position += 2
        elif source[position] == quote:
            return position + 1
        else:
            position += 1
    raise OpenDuelystParseError("unterminated JavaScript string literal")


def _find_matching(source: str, opening: int, open_char: str, close_char: str) -> int:
    depth = 0
    position = opening
    while position < len(source):
        char = source[position]
        if char in "'\"`":
            position = _scan_string(source, position, len(source))
            continue
        if source.startswith("//", position):
            newline = source.find("\n", position + 2)
            position = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", position):
            close = source.find("*/", position + 2)
            if close < 0:
                raise OpenDuelystParseError("unterminated block comment")
            position = close + 2
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return position
        position += 1
    raise OpenDuelystParseError(f"unterminated {open_char}{close_char} block")


def _scan_expression_end(source: str, start: int, end: int) -> int:
    depths = {"{": 0, "[": 0, "(": 0}
    closing = {"}": "{", "]": "[", ")": "("}
    position = start
    while position < end:
        char = source[position]
        if char in "'\"`":
            position = _scan_string(source, position, end)
            continue
        if source.startswith("//", position):
            newline = source.find("\n", position + 2, end)
            position = end if newline < 0 else newline + 1
            continue
        if source.startswith("/*", position):
            close = source.find("*/", position + 2, end)
            if close < 0:
                raise OpenDuelystParseError("unterminated block comment")
            position = close + 2
            continue
        if char in depths:
            depths[char] += 1
        elif char in closing:
            opener = closing[char]
            if depths[opener] == 0:
                return position
            depths[opener] -= 1
        elif char == "," and all(value == 0 for value in depths.values()):
            return position
        position += 1
    return position


def _parse_object_properties(
    source: str, opening: int, closing: int
) -> tuple[tuple[str, str, int], ...]:
    properties: list[tuple[str, str, int]] = []
    position = opening + 1
    while True:
        position = _skip_space_and_comments(source, position, closing)
        if position >= closing:
            break
        key_start = position
        if source[position] in "'\"":
            key_end = _scan_string(source, position, closing)
            key = _parse_js_string(source[position:key_end])
            position = key_end
        else:
            match = re.match(r"[A-Za-z_$][\w$]*", source[position:closing])
            if match is None:
                raise OpenDuelystParseError(
                    f"unexpected object key at line {_line_number(source, position)}"
                )
            key = match.group(0)
            position += len(key)
        position = _skip_space_and_comments(source, position, closing)
        if position >= closing or source[position] != ":":
            raise OpenDuelystParseError(
                f"missing ':' after {key!r} at line {_line_number(source, key_start)}"
            )
        value_start = _skip_space_and_comments(source, position + 1, closing)
        value_end = _scan_expression_end(source, value_start, closing)
        expression = source[value_start:value_end].strip()
        properties.append((key, expression, _line_number(source, key_start)))
        position = value_end + (value_end < closing and source[value_end] == ",")
    return tuple(properties)


def _parse_js_string(expression: str) -> str:
    expression = expression.strip()
    if len(expression) < 2 or expression[0] not in "'\"" or expression[-1] != expression[0]:
        raise OpenDuelystParseError(f"expected quoted JavaScript string, got {expression!r}")
    try:
        value = ast.literal_eval(expression)
    except (SyntaxError, ValueError) as error:
        raise OpenDuelystParseError(f"invalid string literal {expression!r}") from error
    if not isinstance(value, str):
        raise OpenDuelystParseError(f"expected string literal, got {expression!r}")
    return value


def _parse_number(expression: str) -> float:
    expression = expression.strip()
    if re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", expression) is None:
        raise OpenDuelystParseError(f"expected numeric literal, got {expression!r}")
    return float(expression)


def parse_resource_descriptors(
    source: str,
    *,
    relative_path: str = _RESOURCES_PATH,
    member_path: str | None = None,
) -> tuple[ResourceAnimationDeclaration, ...]:
    """Parse only source-declared animation descriptors from ``resources.js``.

    Entries lacking any of ``name``, ``framePrefix``, ``frameDelay``, ``img`` or
    ``plist`` are ordinary resources and are intentionally not promoted into
    animation sequences.
    """

    match = re.search(r"\b(?:const|let|var)\s+RSX\s*=\s*\{", source)
    if match is None:
        raise OpenDuelystParseError(f"{relative_path}: could not locate the RSX object")
    opening = source.find("{", match.start())
    closing = _find_matching(source, opening, "{", "}")
    top_properties = _parse_object_properties(source, opening, closing)
    aliases: set[str] = set()
    declarations: list[ResourceAnimationDeclaration] = []
    required = {"name", "framePrefix", "frameDelay", "img", "plist"}
    for alias, expression, line_number in top_properties:
        if alias in aliases:
            raise OpenDuelystParseError(
                f"{relative_path}:{line_number}: duplicate RSX alias {alias}"
            )
        aliases.add(alias)
        stripped = expression.strip()
        if not stripped.startswith("{"):
            continue
        object_closing = _find_matching(stripped, 0, "{", "}")
        if stripped[object_closing + 1 :].strip():
            continue
        raw_properties = _parse_object_properties(stripped, 0, object_closing)
        by_name: dict[str, tuple[str, int]] = {}
        raw_fields: list[RawExpression] = []
        for field_name, field_expression, relative_line in raw_properties:
            absolute_line = line_number + relative_line - 1
            if field_name in by_name:
                raise OpenDuelystParseError(
                    f"{relative_path}:{absolute_line}: duplicate {field_name!r} in {alias}"
                )
            by_name[field_name] = (field_expression, absolute_line)
            raw_fields.append(RawExpression(field_name, field_expression, absolute_line))
        if not required.issubset(by_name):
            continue
        name = _parse_js_string(by_name["name"][0])
        prefix = _parse_js_string(by_name["framePrefix"][0])
        delay_expression = by_name["frameDelay"][0]
        delay = _parse_number(delay_expression)
        if delay <= 0:
            raise OpenDuelystParseError(
                f"{relative_path}:{line_number}: non-positive frameDelay for {alias}"
            )
        declarations.append(
            ResourceAnimationDeclaration(
                alias=alias,
                name=name,
                frame_prefix=prefix,
                frame_delay=delay,
                frame_delay_expression=delay_expression.strip(),
                image_path=_parse_js_string(by_name["img"][0]),
                plist_path=_parse_js_string(by_name["plist"][0]),
                raw_fields=tuple(raw_fields),
                evidence_relative_path=relative_path,
                evidence_member_path=member_path or relative_path,
                line_number=line_number,
            )
        )
    return tuple(declarations)


_RECT_PATTERN = re.compile(
    r"^\{\s*\{\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\}\s*,\s*"
    r"\{\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\}\s*\}$"
)
_SIZE_PATTERN = re.compile(r"^\{\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\}$")
_POINT_PATTERN = re.compile(
    r"^\{\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}$"
)


def _parse_rectangle(raw: Any, *, field: str, relative_path: str) -> Rectangle:
    if not isinstance(raw, str) or (match := _RECT_PATTERN.fullmatch(raw)) is None:
        raise OpenDuelystParseError(f"{relative_path}: malformed {field}: {raw!r}")
    x, y, width, height = (int(value) for value in match.groups())
    if width < 0 or height < 0:
        raise OpenDuelystParseError(f"{relative_path}: negative {field} size: {raw!r}")
    return Rectangle(x, y, width, height, raw)


def _parse_size(raw: Any, *, field: str, relative_path: str) -> Size:
    if not isinstance(raw, str) or (match := _SIZE_PATTERN.fullmatch(raw)) is None:
        raise OpenDuelystParseError(f"{relative_path}: malformed {field}: {raw!r}")
    width, height = (int(value) for value in match.groups())
    if width < 0 or height < 0:
        raise OpenDuelystParseError(f"{relative_path}: negative {field}: {raw!r}")
    return Size(width, height, raw)


def _parse_point(raw: Any, *, field: str, relative_path: str) -> NumericPoint:
    if not isinstance(raw, str) or (match := _POINT_PATTERN.fullmatch(raw)) is None:
        raise OpenDuelystParseError(f"{relative_path}: malformed {field}: {raw!r}")
    return NumericPoint(float(match.group(1)), float(match.group(2)), raw)


def _xml_frame_key_order(payload: bytes) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read XML dictionary order separately from plistlib and detect duplicate keys."""

    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return (), ()
    document_dict = root.find("dict")
    if document_dict is None:
        return (), ()
    children = list(document_dict)
    for index in range(0, len(children) - 1, 2):
        if children[index].tag == "key" and children[index].text == "frames":
            frames_dict = children[index + 1]
            if frames_dict.tag != "dict":
                return (), ()
            keys = [
                child.text or ""
                for child_index, child in enumerate(frames_dict)
                if child_index % 2 == 0 and child.tag == "key"
            ]
            counts = Counter(keys)
            duplicates = tuple(key for key in keys if counts[key] > 1)
            return tuple(keys), tuple(dict.fromkeys(duplicates))
    return (), ()


def parse_texture_packer_plist(
    payload: bytes,
    *,
    relative_path: str,
    member_path: str | None = None,
    image_relative_path: str | None = None,
    image_member_path: str | None = None,
    image_size: tuple[int, int] | None = None,
    image_mode: str | None = None,
    image_format: str | None = None,
    image_sha256: str | None = None,
    descriptor_reference_count: int = 0,
) -> AtlasSheet:
    """Parse one TexturePacker format-2 plist without inventing sequences."""

    try:
        document = plistlib.loads(payload)
    except Exception as error:
        raise OpenDuelystParseError(f"{relative_path}: invalid plist: {error}") from error
    if not isinstance(document, dict) or not isinstance(document.get("frames"), dict):
        raise OpenDuelystParseError(f"{relative_path}: plist has no frame dictionary")
    frames_document: Mapping[str, Any] = document["frames"]
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise OpenDuelystParseError(f"{relative_path}: plist has no metadata dictionary")
    xml_order, duplicate_keys = _xml_frame_key_order(payload)
    declaration_keys = xml_order or tuple(frames_document)
    if set(declaration_keys) != set(frames_document):
        raise OpenDuelystParseError(f"{relative_path}: XML/plistlib frame key disagreement")
    image_width, image_height = image_size or (None, None)
    frames: list[AtlasFrame] = []
    expected_fields = {"frame", "offset", "rotated", "sourceColorRect", "sourceSize"}
    for declaration_index, key in enumerate(declaration_keys):
        raw_frame = frames_document[key]
        if not isinstance(key, str) or not isinstance(raw_frame, dict):
            raise OpenDuelystParseError(f"{relative_path}: invalid frame entry {key!r}")
        missing = expected_fields.difference(raw_frame)
        if missing:
            raise OpenDuelystParseError(
                f"{relative_path}: frame {key!r} missing {sorted(missing)!r}"
            )
        frame = _parse_rectangle(raw_frame["frame"], field="frame", relative_path=relative_path)
        offset = _parse_point(raw_frame["offset"], field="offset", relative_path=relative_path)
        source_rect = _parse_rectangle(
            raw_frame["sourceColorRect"], field="sourceColorRect", relative_path=relative_path
        )
        source_size = _parse_size(
            raw_frame["sourceSize"], field="sourceSize", relative_path=relative_path
        )
        rotated = raw_frame["rotated"]
        if not isinstance(rotated, bool):
            raise OpenDuelystParseError(
                f"{relative_path}: frame {key!r} has non-boolean rotated value"
            )
        in_bounds = None
        if image_width is not None and image_height is not None:
            in_bounds = (
                frame.x >= 0
                and frame.y >= 0
                and frame.right <= image_width
                and frame.bottom <= image_height
            )
        is_trimmed = not (
            source_rect.x == 0
            and source_rect.y == 0
            and source_rect.width == source_size.width
            and source_rect.height == source_size.height
        )
        frames.append(
            AtlasFrame(
                key=key,
                declaration_index=declaration_index,
                frame=frame,
                offset=offset,
                rotated=rotated,
                source_color_rect=source_rect,
                source_size=source_size,
                within_image_bounds=in_bounds,
                is_trimmed=is_trimmed,
            )
        )
    metadata_format = metadata.get("format")
    if metadata_format is not None and not isinstance(metadata_format, int):
        raise OpenDuelystParseError(f"{relative_path}: non-integer metadata format")
    metadata_size = None
    if "size" in metadata:
        metadata_size = _parse_size(
            metadata["size"], field="metadata.size", relative_path=relative_path
        )
    metadata_matches = None
    if metadata_size is not None and image_width is not None and image_height is not None:
        metadata_matches = (metadata_size.width, metadata_size.height) == (
            image_width,
            image_height,
        )
    return AtlasSheet(
        relative_path=relative_path,
        member_path=member_path or relative_path,
        image_relative_path=image_relative_path,
        image_member_path=image_member_path,
        image_width=image_width,
        image_height=image_height,
        image_mode=image_mode,
        image_format=image_format,
        image_sha256=image_sha256,
        metadata_format=metadata_format,
        metadata_size=metadata_size,
        metadata_texture_file_name=metadata.get("textureFileName"),
        metadata_real_texture_file_name=metadata.get("realTextureFileName"),
        frames=tuple(frames),
        descriptor_reference_count=descriptor_reference_count,
        duplicate_frame_keys=duplicate_keys,
        metadata_matches_image_size=metadata_matches,
    )


def _final_numeric_token(key: str) -> int | None:
    matches = re.findall(r"\d+", key)
    return int(matches[-1]) if matches else None


def runtime_frame_keys(frame_keys: Sequence[str], frame_prefix: str) -> tuple[str, ...]:
    """Reproduce the snapshot's JS prefix selection and stable numeric ordering."""

    # Source builds a regexp directly from framePrefix.  Pinned prefixes contain
    # no regexp operators, but escaping here makes the same intended prefix rule
    # safe for callers supplying independent fixtures.
    pattern = re.compile(rf"^{re.escape(frame_prefix)}(?=[0-9.\x08])")
    selected = [key for key in frame_keys if pattern.search(key)]

    def compare(left: str, right: str) -> int:
        left_number = _final_numeric_token(left)
        right_number = _final_numeric_token(right)
        if left_number is None or right_number is None:
            return 0
        return (left_number > right_number) - (left_number < right_number)

    return tuple(sorted(selected, key=cmp_to_key(compare)))


def _sequence_category(plist_path: str) -> str:
    parts = PurePosixPath(plist_path).parts
    if "units" in parts:
        return "unit"
    if "fx" in parts:
        return "effect"
    if "icons" in parts:
        return "icon_animation"
    if "tiles" in parts:
        return "tile"
    if "runes" in parts:
        return "rune"
    if "arena" in parts:
        return "arena_effect"
    return "other"


def resolve_animation_sequence(
    declaration: ResourceAnimationDeclaration,
    atlas: AtlasSheet,
    *,
    source_roles: Iterable[str] = (),
    entity_mapping_indices: Iterable[int] = (),
) -> AnimationSequence:
    """Resolve a declaration against its exact plist and attach source-backed roles."""

    if atlas.relative_path != declaration.plist_path:
        raise OpenDuelystParseError(
            f"{declaration.alias}: descriptor plist {declaration.plist_path!r} does not match "
            f"atlas {atlas.relative_path!r}"
        )
    frames_by_key = {frame.key: frame for frame in atlas.frames}
    ordered_keys = runtime_frame_keys(tuple(frames_by_key), declaration.frame_prefix)
    sequence_frames: list[SequenceFrame] = []
    for occurrence_index, key in enumerate(ordered_keys):
        frame = frames_by_key[key]
        sequence_frames.append(
            SequenceFrame(
                occurrence_index=occurrence_index,
                atlas_declaration_index=frame.declaration_index,
                key=key,
                final_numeric_token=_final_numeric_token(key),
                frame=frame.frame,
                offset=frame.offset,
                rotated=frame.rotated,
                source_color_rect=frame.source_color_rect,
                source_size=frame.source_size,
                within_image_bounds=frame.within_image_bounds,
                is_trimmed=frame.is_trimmed,
            )
        )
    roles = tuple(dict.fromkeys(source_roles))
    loop_roles = set(roles).intersection(_LOOP_ROLES)
    one_shot_roles = set(roles).intersection(_ONE_SHOT_ROLES)
    unknown_roles = set(roles).difference(_LOOP_ROLES | _ONE_SHOT_ROLES)
    ambiguity: list[str] = []
    if not sequence_frames:
        ambiguity.append("frame_prefix_matches_no_plist_key")
    if len(roles) == 1:
        normalized_action = roles[0]
        normalized_action_basis = "single_exact_setBaseAnimResource_role"
    elif roles:
        normalized_action = None
        normalized_action_basis = "multiple_source_roles_preserved"
        ambiguity.append("multiple_source_roles")
    else:
        normalized_action = None
        normalized_action_basis = "no_setBaseAnimResource_role"
    if loop_roles and not one_shot_roles and not unknown_roles:
        loop_mode: Literal["loop", "one_shot", "role_dependent", "unknown"] = "loop"
        loop_basis = "all_exact_source_roles_are_repeated_by_runtime_callers"
    elif one_shot_roles and not loop_roles and not unknown_roles:
        loop_mode = "one_shot"
        loop_basis = "all_exact_source_roles_are_played_once_by_runtime_callers"
    elif loop_roles or one_shot_roles:
        loop_mode = "role_dependent"
        loop_basis = "source_roles_have_conflicting_or_unresolved_runtime_call_semantics"
        ambiguity.append("loop_depends_on_entity_role")
    else:
        loop_mode = "unknown"
        loop_basis = "descriptor_and_plist_do_not_declare_looping"
    effective_delay = declaration.effective_frame_delay_seconds
    return AnimationSequence(
        resource_alias=declaration.alias,
        runtime_name=declaration.name,
        category=_sequence_category(declaration.plist_path),
        plist_path=declaration.plist_path,
        image_path=declaration.image_path,
        frame_prefix=declaration.frame_prefix,
        declared_frame_delay_seconds=declaration.frame_delay,
        declared_frame_delay_expression=declaration.frame_delay_expression,
        runtime_delay_multiplier=_RUNTIME_DELAY_MULTIPLIER,
        effective_frame_delay_seconds=effective_delay,
        total_duration_seconds=len(sequence_frames) * effective_delay,
        frames=tuple(sequence_frames),
        source_roles=roles,
        entity_mapping_indices=tuple(dict.fromkeys(entity_mapping_indices)),
        normalized_action=normalized_action,
        normalized_action_basis=normalized_action_basis,
        loop_mode=loop_mode,
        loop_basis=loop_basis,
        direction=None,
        direction_semantics=(
            "single_source_track; unit rendering horizontally flips at runtime by owner/target; "
            "source-facing direction is not labelled"
        ),
        ambiguity_reasons=tuple(ambiguity),
        evidence_relative_path=declaration.evidence_relative_path,
        evidence_member_path=declaration.evidence_member_path,
        line_number=declaration.line_number,
    )


def parse_card_lookup(source: str) -> dict[str, int]:
    """Parse exact ``Cards.Group.Member`` integer definitions."""

    result: dict[str, int] = {}
    current_group: str | None = None
    group_indent = -1
    for line_number, line in enumerate(source.splitlines(), 1):
        group_match = re.match(r"^(\s*)@(\w+)\s*:\s*\{\s*$", line)
        if group_match:
            current_group = group_match.group(2)
            group_indent = len(group_match.group(1))
            continue
        if current_group is None:
            continue
        stripped = line.strip()
        if stripped == "}":
            current_group = None
            continue
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= group_indent:
            current_group = None
            continue
        value_match = re.match(r"^\s*(\w+)\s*:\s*(-?\d+)\s*,?\s*$", line)
        if value_match is None:
            continue
        key = f"Cards.{current_group}.{value_match.group(1)}"
        if key in result:
            raise OpenDuelystParseError(f"cardsLookup:{line_number}: duplicate {key}")
        result[key] = int(value_match.group(2))
    return result


def _coffee_call_block(lines: Sequence[str], call_index: int) -> tuple[int, int]:
    """Return inclusive line indices spanning a parenthesized CoffeeScript call."""

    depth = 0
    seen_open = False
    quote: str | None = None
    escaped = False
    for index in range(call_index, len(lines)):
        for char in lines[index]:
            if quote is not None:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in "'\"":
                quote = char
            elif char == "(":
                seen_open = True
                depth += 1
            elif char == ")":
                depth -= 1
                if seen_open and depth == 0:
                    return call_index, index
    raise OpenDuelystParseError(f"unterminated setBaseAnimResource call at line {call_index + 1}")


def _nearest_assignment(
    lines: Sequence[str], before: int, pattern: str, *, lower_bound: int
) -> tuple[str | None, int | None]:
    compiled = re.compile(pattern)
    for index in range(before - 1, lower_bound - 1, -1):
        if match := compiled.search(lines[index]):
            return match.group(1).strip(), index + 1
    return None, None


def parse_entity_animation_mappings(
    source: str,
    *,
    relative_path: str,
    member_path: str | None = None,
    card_ids: Mapping[str, int] | None = None,
    localization: Mapping[str, str] | None = None,
) -> tuple[EntityMapping, ...]:
    """Parse every ``setBaseAnimResource`` block and its enclosing card evidence."""

    lines = source.splitlines()
    mappings: list[EntityMapping] = []
    for call_index, line in enumerate(lines):
        if "setBaseAnimResource" not in line:
            continue
        start, end = _coffee_call_block(lines, call_index)
        call_indent = len(lines[start]) - len(lines[start].lstrip())
        condition_index = -1
        condition_expression: str | None = None
        for index in range(start - 1, -1, -1):
            condition_match = re.match(
                r"^\s*(?:else\s+)?if\s+\(identifier\s*==\s*(.*?)\)\s*$", lines[index]
            )
            if condition_match:
                condition_index = index
                condition_expression = condition_match.group(1).strip()
                break
            indent = len(lines[index]) - len(lines[index].lstrip())
            if (
                lines[index].strip()
                and indent < call_indent
                and "cardForIdentifier" in lines[index]
            ):
                break
        lower_bound = condition_index if condition_index >= 0 else 0
        kind_expression, _ = _nearest_assignment(
            lines,
            start,
            r"\bcard\s*=\s*new\s+([\w.$]+)",
            lower_bound=lower_bound,
        )
        name_expression, _ = _nearest_assignment(
            lines,
            start,
            r"\bcard\.name\s*=\s*(.+?)\s*$",
            lower_bound=lower_bound,
        )
        faction_expression, _ = _nearest_assignment(
            lines,
            start,
            r"\bcard\.factionId\s*=\s*(.+?)\s*$",
            lower_bound=lower_bound,
        )
        race_expression, _ = _nearest_assignment(
            lines,
            start,
            r"\bcard\.raceId\s*=\s*(.+?)\s*$",
            lower_bound=lower_bound,
        )
        localization_key = None
        display_name = None
        if name_expression and (
            localization_match := re.fullmatch(
                r"i18next\.t\(\s*['\"]([^'\"]+)['\"]\s*\)", name_expression
            )
        ):
            localization_key = localization_match.group(1)
            lookup_key = (
                localization_key if "." in localization_key else f"cards.{localization_key}"
            )
            display_name = localization.get(lookup_key) if localization else None
        elif name_expression and len(name_expression) >= 2 and name_expression[0] in "'\"":
            with contextlib.suppress(OpenDuelystParseError):
                display_name = _parse_js_string(name_expression)
        fields: list[EntityAnimationField] = []
        references: list[EntityAnimationReference] = []
        for field_index in range(start + 1, end):
            field_match = re.match(r"^\s*([\w$]+)\s*:\s*(.+?)\s*,?\s*$", lines[field_index])
            if field_match is None:
                continue
            role, expression = field_match.groups()
            resource_match = re.fullmatch(r"RSX\.([A-Za-z_$][\w$]*)\s*\.\s*name", expression)
            resource_alias = resource_match.group(1) if resource_match else None
            numeric_value = None
            if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", expression):
                numeric_value = float(expression)
            fields.append(
                EntityAnimationField(
                    role=role,
                    expression=expression,
                    resource_alias=resource_alias,
                    numeric_value=numeric_value,
                    line_number=field_index + 1,
                )
            )
            if resource_alias is not None:
                references.append(
                    EntityAnimationReference(
                        role=role,
                        resource_alias=resource_alias,
                        expression=expression,
                        evidence_relative_path=relative_path,
                        evidence_member_path=member_path or relative_path,
                        line_number=field_index + 1,
                    )
                )
        if not fields:
            raise OpenDuelystParseError(
                f"{relative_path}:{start + 1}: empty setBaseAnimResource declaration"
            )
        identifier_token = (
            condition_expression
            if condition_expression and re.fullmatch(r"[\w.$]+", condition_expression)
            else None
        )
        mappings.append(
            EntityMapping(
                identifier_expression=condition_expression,
                identifier_token=identifier_token,
                card_id=card_ids.get(identifier_token) if card_ids and identifier_token else None,
                card_kind=kind_expression,
                card_name_expression=name_expression,
                localization_key=localization_key,
                display_name=display_name,
                faction_expression=faction_expression,
                race_expression=race_expression,
                animation_fields=tuple(fields),
                animation_references=tuple(references),
                evidence_relative_path=relative_path,
                evidence_member_path=member_path or relative_path,
                line_number=start + 1,
            )
        )
    return tuple(mappings)


def _parse_english_localization(archive: ZipFile, root: str) -> dict[str, str]:
    localization: dict[str, str] = {}
    for info in archive.infolist():
        prefix = f"{root}/{_LOCALIZATION_PREFIX}"
        if (
            info.is_dir()
            or not info.filename.startswith(prefix)
            or not info.filename.endswith(".json")
        ):
            continue
        relative = info.filename[len(prefix) :]
        if "/" in relative:
            continue
        namespace = PurePosixPath(relative).stem
        try:
            document = json.loads(_decode_text(archive.read(info), info.filename))
        except (json.JSONDecodeError, OpenDuelystParseError):
            continue
        if not isinstance(document, dict):
            continue
        for key, value in document.items():
            if isinstance(value, str):
                localization[f"{namespace}.{key}"] = value
    return localization


def _image_info(payload: bytes) -> tuple[int, int, str, str | None]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            return image.width, image.height, image.mode, image.format
    except (OSError, UnidentifiedImageError) as error:
        raise OpenDuelystParseError(f"unreadable atlas image: {error}") from error


def _gif_audit(payload: bytes, relative_path: str, member_path: str) -> GifAnimation:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            frame_count = getattr(image, "n_frames", 1)
            frames: list[GifFrame] = []
            has_transparency = image.info.get("transparency") is not None
            loop_value = image.info.get("loop")
            width, height = image.size
            for frame_index in range(frame_count):
                image.seek(frame_index)
                duration = image.info.get("duration", 0)
                frames.append(GifFrame(frame_index, int(duration)))
                has_transparency = has_transparency or image.info.get("transparency") is not None
    except (OSError, UnidentifiedImageError) as error:
        raise OpenDuelystParseError(f"{relative_path}: unreadable GIF: {error}") from error
    stem = PurePosixPath(relative_path).stem
    hint_match = re.match(
        r"^(.*?)(?:[_-](attack|breathing|death|hit|idle|run|walk|cast|active))$",
        stem,
        re.IGNORECASE,
    )
    identity_hint = hint_match.group(1) if hint_match else stem
    action_hint = hint_match.group(2).lower() if hint_match else None
    return GifAnimation(
        relative_path=relative_path,
        member_path=member_path,
        identity_hint=identity_hint,
        action_hint=action_hint,
        width=width,
        height=height,
        frame_count=frame_count,
        frames=tuple(frames),
        loop_value=int(loop_value) if loop_value is not None else None,
        has_transparency=has_transparency,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        timing_authority=(
            "encoded GIF preview timing only; does not override resources.js/plist runtime timing"
        ),
    )


_EVIDENCE_PATHS: Mapping[str, tuple[str, str]] = {
    "LICENSE": (
        "repository_project",
        "Root CC0-1.0 declaration; repository-level evidence, not per-asset authorship.",
    ),
    "COPYING": (
        "repository_project",
        "Root CC0-1.0 legal text; it does not clear third-party publicity, "
        "privacy, or other rights.",
    ),
    "README.md": (
        "repository_project",
        "States OpenDuelyst is CC0-1.0; no per-asset creator or provenance mapping.",
    ),
    "package.json": (
        "repository_project",
        "Root package manifest declares CC0-1.0.",
    ),
    "desktop/package.json": (
        "desktop_subproject",
        "Desktop package manifest declares CC0-1.0.",
    ),
    "app/vendor/cocos2d-html5/AUTHORS.txt": (
        "app/vendor/cocos2d-html5",
        "Vendored engine author list; not evidence for OpenDuelyst art authorship.",
    ),
    "app/vendor/cocos2d-html5/licenses/LICENSE_cocos2d-html5.txt": (
        "app/vendor/cocos2d-html5",
        "Vendored cocos2d-html5 MIT license.",
    ),
    "app/vendor/cocos2d-html5/licenses/LICENSE_cocos2d-x.txt": (
        "app/vendor/cocos2d-html5",
        "Vendored cocos2d-x MIT license.",
    ),
    "app/vendor/cocos2d-html5/licenses/LICENSE_zlib.js.txt": (
        "app/vendor/cocos2d-html5",
        "Vendored zlib.js MIT license.",
    ),
    "packages/backfire/LICENSE": (
        "packages/backfire",
        "Vendored/local package MIT license, scoped to backfire.",
    ),
    "packages/warlock/LICENSE": (
        "packages/warlock",
        "Vendored/local package MIT license, scoped to warlock.",
    ),
}


def _license_identifiers(relative_path: str, payload: bytes) -> tuple[str, ...]:
    text = payload.decode("utf-8", errors="replace")
    identifiers: list[str] = []
    if (
        "Creative Commons Zero v1.0 Universal" in text
        or "CC0 1.0 Universal" in text
        or re.search(r'"license"\s*:\s*"CC0-1\.0"', text)
    ):
        identifiers.append("CC0-1.0")
    if "MIT License" in text or "Permission is hereby granted, free of charge" in text:
        identifiers.append("MIT")
    if relative_path.endswith("AUTHORS.txt"):
        identifiers.append("AUTHORS-NOT-A-LICENSE")
    return tuple(identifiers)


def _evidence_documents(archive: ZipFile, root: str) -> tuple[EvidenceDocument, ...]:
    documents: list[EvidenceDocument] = []
    names = {info.filename: info for info in archive.infolist() if not info.is_dir()}
    for relative_path, (scope, notes) in _EVIDENCE_PATHS.items():
        member_path = f"{root}/{relative_path}"
        info = names.get(member_path)
        if info is None:
            continue
        payload = archive.read(info)
        documents.append(
            EvidenceDocument(
                relative_path=relative_path,
                member_path=member_path,
                sha256=_sha256_bytes(payload),
                size_bytes=len(payload),
                detected_license_identifiers=_license_identifiers(relative_path, payload),
                scope=scope,
                notes=notes,
            )
        )
    return tuple(documents)


_CODE_EVIDENCE_SPECS: Mapping[str, tuple[tuple[int, ...], str]] = {
    "app/data/resources.js": (
        (),
        "Declares animation runtime name, plist, image, frame prefix and source delay.",
    ),
    "app/ui/managers/package_manager.js": (
        (866, 869, 875, 883, 884),
        "Multiplies frameDelay by 0.8, obtains ordered keys, and caches a cc.Animation by name.",
    ),
    "app/common/utils/utils_resources.js": (
        (151, 162, 165, 168, 175, 176, 178, 180),
        "Selects prefix-matching plist keys and stable-sorts on the final numeric token.",
    ),
    "app/common/utils/utils_engine.js": (
        (644, 646, 649, 653),
        "Applies repeatForever only when the animation caller requests looping.",
    ),
    "app/view/nodes/EntityNode.js": (
        (2034, 2042),
        "Flips entity rendering horizontally at runtime based on owner/target state.",
    ),
}


def _source_code_evidence(archive: ZipFile, root: str) -> tuple[SourceCodeEvidence, ...]:
    names = {info.filename: info for info in archive.infolist() if not info.is_dir()}
    evidence: list[SourceCodeEvidence] = []
    for relative_path, (line_numbers, establishes) in _CODE_EVIDENCE_SPECS.items():
        member_path = f"{root}/{relative_path}"
        info = names.get(member_path)
        if info is None:
            # EntityNode moved between snapshots; locate exact suffix without
            # converting the absence into guessed semantics.
            suffix = "/" + PurePosixPath(relative_path).name
            matches = [candidate for candidate in names if candidate.endswith(suffix)]
            if len(matches) != 1:
                continue
            member_path = matches[0]
            relative_path = member_path[len(root) + 1 :]
            info = names[member_path]
        payload = archive.read(info)
        evidence.append(
            SourceCodeEvidence(
                relative_path=relative_path,
                member_path=member_path,
                sha256=_sha256_bytes(payload),
                line_numbers=line_numbers,
                establishes=establishes,
            )
        )
    return tuple(evidence)


_ATTRIBUTION_METADATA_KEYS = frozenset(
    {"author", "artist", "copyright", "license", "credit", "dc:creator", "dc:rights"}
)
_ATTRIBUTION_XML_PATTERN = re.compile(
    r"<(dc:creator|dc:rights|xmprights:usageterms|photoshop:credit|"
    r"photoshop:authorsposition)\b",
    re.IGNORECASE,
)


def _embedded_png_metadata(
    archive: ZipFile, png_infos: Sequence[ZipInfo]
) -> tuple[EmbeddedMetadataSummary, dict[str, tuple[str, ...]]]:
    readable = 0
    with_xmp = 0
    with_comment = 0
    gimp_comment = 0
    with_attribution = 0
    found_fields: set[str] = set()
    hash_paths: defaultdict[str, list[str]] = defaultdict(list)
    root = _validate_archive_members(tuple(archive.infolist()))
    for info in png_infos:
        payload = archive.read(info)
        hash_paths[_sha256_bytes(payload)].append(info.filename[len(root) + 1 :])
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                readable += 1
                metadata = {str(key): value for key, value in image.info.items()}
        except (OSError, UnidentifiedImageError):
            continue
        lower_keys = {key.lower() for key in metadata}
        if "xmp" in lower_keys or any("xml" in key for key in lower_keys):
            with_xmp += 1
        comment_values = [
            value for key, value in metadata.items() if key.lower() in {"comment", "description"}
        ]
        if comment_values:
            with_comment += 1
            if any("created with gimp" in str(value).lower() for value in comment_values):
                gimp_comment += 1
        searchable = "\n".join(
            [
                *metadata,
                *(str(value) for value in metadata.values() if isinstance(value, (str, bytes))),
            ]
        ).lower()
        member_fields = {
            key.lower() for key in metadata if key.lower() in _ATTRIBUTION_METADATA_KEYS
        }
        member_fields.update(
            match.lower() for match in _ATTRIBUTION_XML_PATTERN.findall(searchable)
        )
        if member_fields:
            with_attribution += 1
            found_fields.update(member_fields)
    duplicate_hash_paths = {
        digest: tuple(paths) for digest, paths in hash_paths.items() if len(paths) > 1
    }
    summary = EmbeddedMetadataSummary(
        png_file_count=len(png_infos),
        readable_png_count=readable,
        unique_png_sha256_count=len(hash_paths),
        duplicate_png_sha256_group_count=len(duplicate_hash_paths),
        duplicate_png_path_excess_count=sum(
            len(paths) - 1 for paths in duplicate_hash_paths.values()
        ),
        png_with_xmp_count=with_xmp,
        png_with_comment_count=with_comment,
        gimp_comment_count=gimp_comment,
        png_with_asset_attribution_field_count=with_attribution,
        attribution_field_names=tuple(sorted(found_fields)),
    )
    return summary, duplicate_hash_paths


def _validate_archive_members(infos: Sequence[ZipInfo]) -> str:
    if not infos:
        raise OpenDuelystArchiveError("archive is empty")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        raise OpenDuelystArchiveError(
            f"archive contains duplicate member names: {duplicates[:3]!r}"
        )
    roots: set[str] = set()
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise OpenDuelystArchiveError(f"unsafe archive member path: {name!r}")
        roots.add(path.parts[0])
    if len(roots) != 1:
        raise OpenDuelystArchiveError(f"archive has multiple roots: {sorted(roots)!r}")
    return roots.pop()


def _resource_entry_count(source: str) -> int:
    match = re.search(r"\b(?:const|let|var)\s+RSX\s*=\s*\{", source)
    if match is None:
        return 0
    opening = source.find("{", match.start())
    closing = _find_matching(source, opening, "{", "}")
    return len(_parse_object_properties(source, opening, closing))


def _resolve_atlas_image_path(
    relative_plist_path: str, metadata: Mapping[str, Any], all_relative_paths: set[str]
) -> str | None:
    candidates: list[str] = []
    parent = PurePosixPath(relative_plist_path).parent
    for key in ("realTextureFileName", "textureFileName"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            candidates.append(str(parent / PurePosixPath(value).name))
    stem = str(PurePosixPath(relative_plist_path).with_suffix(""))
    candidates.extend(f"{stem}{extension}" for extension in (".png", ".jpg", ".jpeg"))
    return next((candidate for candidate in candidates if candidate in all_relative_paths), None)


def _resource_relative_path(repository_relative_path: str) -> str:
    """Convert ``app/resources/...`` members to the path space used by RSX."""

    if repository_relative_path.startswith("app/"):
        return repository_relative_path[4:]
    return repository_relative_path


def audit_openduelyst_archive(path: str | Path) -> OpenDuelystAudit:
    """Audit an OpenDuelyst repository ZIP in place, without extraction or writes."""

    archive_path = Path(path)
    if not archive_path.is_file():
        raise OpenDuelystArchiveError(f"archive does not exist: {archive_path}")
    archive_sha256 = _sha256_file(archive_path)
    try:
        with ZipFile(archive_path) as archive:
            infos = tuple(archive.infolist())
            root = _validate_archive_members(infos)
            files = tuple(info for info in infos if not info.is_dir())
            by_relative: dict[str, ZipInfo] = {
                info.filename[len(root) + 1 :]: info
                for info in files
                if info.filename.startswith(f"{root}/")
            }
            by_resource_path = {
                _resource_relative_path(relative_path): info
                for relative_path, info in by_relative.items()
                if relative_path.startswith("app/")
            }
            if _RESOURCES_PATH not in by_relative:
                raise OpenDuelystArchiveError(f"archive has no {_RESOURCES_PATH}")
            resources_info = by_relative[_RESOURCES_PATH]
            resources_source = _decode_text(archive.read(resources_info), _RESOURCES_PATH)
            declarations = parse_resource_descriptors(
                resources_source,
                relative_path=_RESOURCES_PATH,
                member_path=resources_info.filename,
            )
            declaration_plists = Counter(declaration.plist_path for declaration in declarations)

            all_relative_paths = set(by_resource_path)
            atlas_list: list[AtlasSheet] = []
            plist_parse_errors: list[str] = []
            for repository_relative_path, info in by_relative.items():
                if not repository_relative_path.lower().endswith(".plist"):
                    continue
                relative_path = _resource_relative_path(repository_relative_path)
                payload = archive.read(info)
                try:
                    document = plistlib.loads(payload)
                except Exception:
                    # Preserve the failed path in the audit.  In the pinned tree
                    # these are particle configuration plists, not atlases.
                    plist_parse_errors.append(repository_relative_path)
                    continue
                if not isinstance(document, dict) or not isinstance(document.get("frames"), dict):
                    continue
                metadata = document.get("metadata")
                if not isinstance(metadata, dict):
                    raise OpenDuelystParseError(f"{relative_path}: atlas has no metadata")
                image_relative_path = _resolve_atlas_image_path(
                    relative_path, metadata, all_relative_paths
                )
                image_member_path = None
                image_size = None
                image_mode = None
                image_format = None
                image_sha256 = None
                if image_relative_path is not None:
                    image_info = by_resource_path[image_relative_path]
                    image_payload = archive.read(image_info)
                    width, height, image_mode, image_format = _image_info(image_payload)
                    image_size = (width, height)
                    image_member_path = image_info.filename
                    image_sha256 = _sha256_bytes(image_payload)
                atlas_list.append(
                    parse_texture_packer_plist(
                        payload,
                        relative_path=relative_path,
                        member_path=info.filename,
                        image_relative_path=image_relative_path,
                        image_member_path=image_member_path,
                        image_size=image_size,
                        image_mode=image_mode,
                        image_format=image_format,
                        image_sha256=image_sha256,
                        descriptor_reference_count=declaration_plists[relative_path],
                    )
                )
            atlases = tuple(sorted(atlas_list, key=lambda atlas: atlas.relative_path))
            atlas_by_path = {atlas.relative_path: atlas for atlas in atlases}

            lookup: Mapping[str, int] = {}
            if lookup_info := by_relative.get(_CARD_LOOKUP_PATH):
                lookup_source = _decode_text(archive.read(lookup_info), _CARD_LOOKUP_PATH)
                lookup = parse_card_lookup(lookup_source)
            localization = _parse_english_localization(archive, root)
            mapping_list: list[EntityMapping] = []
            for relative_path, info in sorted(by_relative.items()):
                if relative_path.startswith(_FACTORY_PREFIX) and relative_path.endswith(".coffee"):
                    mapping_list.extend(
                        parse_entity_animation_mappings(
                            _decode_text(archive.read(info), relative_path),
                            relative_path=relative_path,
                            member_path=info.filename,
                            card_ids=lookup,
                            localization=localization,
                        )
                    )
            entity_mappings = tuple(mapping_list)
            roles_by_alias: defaultdict[str, list[str]] = defaultdict(list)
            mapping_indices_by_alias: defaultdict[str, list[int]] = defaultdict(list)
            for mapping_index, mapping in enumerate(entity_mappings):
                for reference in mapping.animation_references:
                    roles_by_alias[reference.resource_alias].append(reference.role)
                    mapping_indices_by_alias[reference.resource_alias].append(mapping_index)

            sequence_list: list[AnimationSequence] = []
            missing_descriptor_plists: list[str] = []
            for declaration in declarations:
                atlas = atlas_by_path.get(declaration.plist_path)
                if atlas is None:
                    missing_descriptor_plists.append(declaration.alias)
                    continue
                sequence_list.append(
                    resolve_animation_sequence(
                        declaration,
                        atlas,
                        source_roles=roles_by_alias[declaration.alias],
                        entity_mapping_indices=mapping_indices_by_alias[declaration.alias],
                    )
                )
            sequences = tuple(sequence_list)

            gif_list: list[GifAnimation] = []
            for relative_path, info in sorted(by_relative.items()):
                if relative_path.lower().endswith(".gif"):
                    gif_list.append(_gif_audit(archive.read(info), relative_path, info.filename))
            gifs = tuple(gif_list)

            evidence_documents = _evidence_documents(archive, root)
            source_code_evidence = _source_code_evidence(archive, root)
            png_infos = tuple(
                info
                for relative_path, info in by_relative.items()
                if relative_path.lower().endswith(".png")
            )
            embedded_metadata, duplicate_png_hash_paths = _embedded_png_metadata(archive, png_infos)
    except (BadZipFile, OSError) as error:
        raise OpenDuelystArchiveError(f"cannot read ZIP {archive_path}: {error}") from error

    frame_owners: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for sequence in sequences:
        for frame in sequence.frames:
            frame_owners[(sequence.plist_path, frame.key)].append(sequence.resource_alias)
    shared_physical = {
        key: tuple(dict.fromkeys(owners))
        for key, owners in frame_owners.items()
        if len(set(owners)) > 1
    }
    duplicate_groups: list[DuplicateGroup] = []
    for (plist_path, frame_key), owners in sorted(shared_physical.items()):
        duplicate_groups.append(
            DuplicateGroup(
                kind="shared_physical_frame",
                keys=owners,
                evidence=f"{plist_path}:{frame_key}",
            )
        )
    timeline_groups: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)
    for sequence in sequences:
        if not sequence.frames:
            continue
        signature = (
            sequence.plist_path,
            tuple(frame.key for frame in sequence.frames),
            sequence.declared_frame_delay_expression,
        )
        timeline_groups[signature].append(sequence.resource_alias)
    exact_timeline_groups = [
        tuple(aliases) for aliases in timeline_groups.values() if len(aliases) > 1
    ]
    for aliases in exact_timeline_groups:
        duplicate_groups.append(
            DuplicateGroup(
                kind="exact_nonempty_timeline_alias",
                keys=aliases,
                evidence="same plist, ordered physical frame keys, and declared delay literal",
            )
        )
    gif_hash_groups: defaultdict[str, list[str]] = defaultdict(list)
    for gif in gifs:
        gif_hash_groups[gif.sha256].append(gif.relative_path)
    for paths in gif_hash_groups.values():
        if len(paths) > 1:
            duplicate_groups.append(
                DuplicateGroup(
                    kind="byte_identical_gif",
                    keys=tuple(paths),
                    evidence="identical SHA-256; paths retained as aliases, not merged",
                )
            )
    for digest, paths in sorted(duplicate_png_hash_paths.items()):
        duplicate_groups.append(
            DuplicateGroup(
                kind="byte_identical_png",
                keys=paths,
                evidence=f"identical SHA-256 {digest}; path aliases are retained",
            )
        )

    runtime_names: defaultdict[str, list[str]] = defaultdict(list)
    for declaration in declarations:
        runtime_names[declaration.name].append(declaration.alias)
    runtime_collisions = {
        name: tuple(aliases) for name, aliases in runtime_names.items() if len(aliases) > 1
    }
    alias_name_mismatches = [
        declaration for declaration in declarations if declaration.alias != declaration.name
    ]
    described_keys = set(frame_owners)
    unreferenced_atlases = tuple(
        atlas for atlas in atlases if atlas.descriptor_reference_count == 0
    )
    referenced_atlas_keys = {
        (atlas.relative_path, frame.key)
        for atlas in atlases
        if atlas.descriptor_reference_count > 0
        for frame in atlas.frames
    }
    empty_sequences = tuple(sequence for sequence in sequences if not sequence.frames)
    multi_role_aliases = {alias for alias, roles in roles_by_alias.items() if len(set(roles)) > 1}
    issues: list[AuditIssue] = []
    if empty_sequences:
        issues.append(
            AuditIssue(
                "warning",
                "EMPTY_DESCRIPTOR_PREFIX",
                "Source descriptors whose exact prefix matches no plist key must be quarantined.",
                tuple(sequence.resource_alias for sequence in empty_sequences),
            )
        )
    if runtime_collisions:
        issues.append(
            AuditIssue(
                "warning",
                "RUNTIME_NAME_COLLISION",
                "Different RSX aliases share a cc.animationCache name and may overwrite "
                "one another.",
                tuple(
                    f"{name}: {', '.join(aliases)}"
                    for name, aliases in sorted(runtime_collisions.items())
                ),
            )
        )
    if alias_name_mismatches:
        issues.append(
            AuditIssue(
                "notice",
                "RSX_ALIAS_NAME_MISMATCH",
                "RSX lookup aliases and runtime cache names differ; both identities are retained.",
                tuple(
                    f"{declaration.alias}->{declaration.name}"
                    for declaration in alias_name_mismatches
                ),
            )
        )
    metadata_mismatches = [
        atlas.relative_path for atlas in atlases if atlas.metadata_matches_image_size is False
    ]
    if metadata_mismatches:
        issues.append(
            AuditIssue(
                "notice",
                "ATLAS_METADATA_IMAGE_SIZE_MISMATCH",
                "TexturePacker metadata size differs from the encoded image size; "
                "rectangles remain in bounds.",
                tuple(metadata_mismatches),
            )
        )
    if plist_parse_errors:
        issues.append(
            AuditIssue(
                "notice",
                "NON_ATLAS_PLIST_PARSE_ERROR",
                "Malformed non-atlas particle configuration plists were retained as paths, "
                "not guessed as atlases.",
                tuple(plist_parse_errors),
            )
        )
    if missing_descriptor_plists:
        issues.append(
            AuditIssue(
                "error",
                "DESCRIPTOR_PLIST_MISSING",
                "Animation descriptors reference an absent or unreadable atlas plist.",
                tuple(missing_descriptor_plists),
            )
        )
    issues.append(
        AuditIssue(
            "notice",
            "REPOSITORY_LICENSE_SCOPE_ONLY",
            "Root files make a project/repository CC0-1.0 claim, but no per-asset creator, "
            "license, or provenance manifest was found; vendor licenses remain subtree-scoped.",
            tuple(document.relative_path for document in evidence_documents),
        )
    )

    suffix_counts = Counter(PurePosixPath(info.filename).suffix.lower() for info in files)
    counts = OpenDuelystCounts(
        zip_member_count=len(infos),
        file_member_count=len(files),
        directory_member_count=len(infos) - len(files),
        archive_png_file_count=suffix_counts[".png"],
        unique_png_payload_count=embedded_metadata.unique_png_sha256_count,
        duplicate_png_payload_group_count=(embedded_metadata.duplicate_png_sha256_group_count),
        archive_gif_file_count=suffix_counts[".gif"],
        archive_plist_file_count=suffix_counts[".plist"],
        archive_coffeescript_file_count=suffix_counts[".coffee"],
        resource_entry_count=_resource_entry_count(resources_source),
        animation_descriptor_count=len(declarations),
        unique_descriptor_plist_count=len({item.plist_path for item in declarations}),
        unique_descriptor_image_count=len({item.image_path for item in declarations}),
        unique_descriptor_plist_prefix_count=len(
            {(item.plist_path, item.frame_prefix) for item in declarations}
        ),
        texture_atlas_count=len(atlases),
        atlas_frame_count=sum(len(atlas.frames) for atlas in atlases),
        rotated_atlas_frame_count=sum(frame.rotated for atlas in atlases for frame in atlas.frames),
        trimmed_atlas_frame_count=sum(
            frame.is_trimmed for atlas in atlases for frame in atlas.frames
        ),
        nonzero_offset_atlas_frame_count=sum(
            frame.offset.x != 0 or frame.offset.y != 0
            for atlas in atlases
            for frame in atlas.frames
        ),
        descriptor_frame_occurrence_count=sum(len(sequence.frames) for sequence in sequences),
        descriptor_unique_frame_count=len(described_keys),
        resolved_animation_descriptor_count=sum(bool(sequence.frames) for sequence in sequences),
        empty_animation_descriptor_count=len(empty_sequences),
        atlas_frame_unmatched_by_descriptor_count=len(referenced_atlas_keys - described_keys),
        unreferenced_atlas_count=len(unreferenced_atlases),
        unreferenced_atlas_frame_count=sum(len(atlas.frames) for atlas in unreferenced_atlases),
        entity_mapping_count=len(entity_mappings),
        entity_animation_reference_count=sum(
            len(mapping.animation_references) for mapping in entity_mappings
        ),
        unique_referenced_resource_alias_count=sum(
            bool(roles) for roles in roles_by_alias.values()
        ),
        mapped_resource_alias_multiple_role_count=len(multi_role_aliases),
        resource_alias_name_mismatch_count=len(alias_name_mismatches),
        runtime_name_collision_count=len(runtime_collisions),
        shared_physical_frame_key_count=len(shared_physical),
        exact_timeline_alias_group_count=len(exact_timeline_groups),
        gif_animation_count=len(gifs),
        unique_gif_payload_count=len(gif_hash_groups),
        duplicate_gif_payload_group_count=sum(len(paths) > 1 for paths in gif_hash_groups.values()),
        gif_frame_count=sum(gif.frame_count for gif in gifs),
        evidence_document_count=len(evidence_documents),
        plist_parse_error_count=len(plist_parse_errors),
    )
    return OpenDuelystAudit(
        archive_path=str(archive_path.resolve()),
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_path.stat().st_size,
        repository_commit=OPENDUELYST_COMMIT,
        repository_url=OPENDUELYST_REPOSITORY_URL,
        commit_url=OPENDUELYST_COMMIT_URL,
        root_prefix=root,
        counts=counts,
        declarations=declarations,
        atlases=atlases,
        sequences=sequences,
        entity_mappings=entity_mappings,
        gifs=gifs,
        duplicate_groups=tuple(duplicate_groups),
        evidence_documents=evidence_documents,
        source_code_evidence=source_code_evidence,
        embedded_metadata=embedded_metadata,
        issues=tuple(issues),
    )


def audit_known_openduelyst_archive(path: str | Path) -> OpenDuelystAudit:
    """Audit only the exact pinned CAS payload and reject all other snapshots."""

    archive_path = Path(path)
    if not archive_path.is_file():
        raise OpenDuelystArchiveError(f"archive does not exist: {archive_path}")
    digest = _sha256_file(archive_path)
    if digest != EXPECTED_OPENDUELYST_ARCHIVE_SHA256:
        raise OpenDuelystArchiveError(
            "OpenDuelyst archive digest mismatch: "
            f"expected {EXPECTED_OPENDUELYST_ARCHIVE_SHA256}, got {digest}"
        )
    audit = audit_openduelyst_archive(archive_path)
    if audit.root_prefix != _EXPECTED_ROOT:
        raise OpenDuelystArchiveError(
            f"OpenDuelyst root mismatch: expected {_EXPECTED_ROOT!r}, got {audit.root_prefix!r}"
        )
    return audit


# Spelling-compatible aliases for callers that visually separate the project name.
audit_open_duelyst_archive = audit_openduelyst_archive
audit_known_open_duelyst_archive = audit_known_openduelyst_archive
