"""Read-only, exact audit adapter for the pinned TMWA legacy client-data ZIP.

The adapter deliberately stops at evidence.  It never extracts files, recolors
palette expressions, composites runtime layers, or writes to the provenance
database.  ManaPlus semantics used below are pinned to immutable source files
and are represented explicitly so a later projection can reject every track it
cannot reproduce exactly.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import unicodedata
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree
from xml.parsers import expat

from PIL import Image

SOURCE_ID = "tmwa_client_data"
TMWA_CLIENT_DATA_COMMIT = "3e63a6f033b6406fe855dba14dbead3db28671fd"
EXPECTED_TMWA_ARCHIVE_SHA256 = "7b7a8c61eca284b35ddedaa1af4210a1694933e36ebdc4299bd73395a6532152"
EXPECTED_TMWA_ARCHIVE_BYTES = 65_557_370
EXPECTED_TMWA_ARCHIVE_ROOT = f"tmwa-client-data-{TMWA_CLIENT_DATA_COMMIT}"
EXPECTED_ZIP_MEMBER_COUNT = 5_082
EXPECTED_REGULAR_FILE_COUNT = 4_912
EXPECTED_EXPANDED_MEMBER_BYTES = 193_431_213
EXPECTED_COMPRESSED_MEMBER_BYTES = 64_110_574
EXPECTED_INVENTORY_SHA256 = "53bdb96c7165b55e6da670b1234228bd21797fecb2ef049b91a227ef2b9aa4c7"

MANAPLUS_ENGINE_COMMIT = "986a3bff49af01f6abd13c1d3b9d41cf50c557ce"
SPRITE_ROOT = "graphics/sprites/"
RIGHTS_SCOPE_CAVEAT = (
    "license.md is an asset-path claim that explicitly warns it may be incomplete "
    "or incorrect. COPYING is preserved as repository evidence, not promoted to a "
    "per-file grant. Missing, contradictory, unknown-contributor, and absent path "
    "claims remain quarantined."
)


class TmwaArchiveError(ValueError):
    """Raised when archive identity or ZIP safety invariants do not hold."""


class TmwaParseError(ValueError):
    """Raised when source declarations cannot be parsed literally."""


@dataclass(frozen=True)
class SourceLocation:
    member_path: str
    logical_path: str
    line_number: int


@dataclass(frozen=True)
class EngineSourceEvidence:
    relative_path: str
    immutable_url: str
    git_blob_sha1: str | None
    sha256: str
    size_bytes: int
    semantics: tuple[str, ...]


ENGINE_SOURCE_EVIDENCE = (
    EngineSourceEvidence(
        relative_path="src/resources/sprite/spritedef.cpp",
        immutable_url=(
            "https://github.com/ManaPlus/ManaPlus/blob/"
            f"{MANAPLUS_ENGINE_COMMIT}/src/resources/sprite/spritedef.cpp"
        ),
        git_blob_sha1="2ccabd5aa477c74940d0c1e4f8f693a20c78ae2b",
        sha256="ea9852b6f17f8d2c30ecee333f7493d4945dcea7f5daeb2b80c750f427dd3407",
        size_bytes=22_433,
        semantics=(
            "sprite includes are rooted at graphics/sprites and processed once",
            "imageset names are first-definition-wins; action names overwrite",
            "frame indices receive the declaring sprite variant offset",
            "sequence endpoints are inclusive in ascending or descending order",
            "frame delays and offsets are read per command",
            "end, jump, label, and goto are runtime timeline commands",
            "included sprites do not inherit the outer palette argument",
        ),
    ),
    EngineSourceEvidence(
        relative_path="src/resources/imageset.cpp",
        immutable_url=(
            "https://github.com/ManaPlus/ManaPlus/blob/"
            f"{MANAPLUS_ENGINE_COMMIT}/src/resources/imageset.cpp"
        ),
        git_blob_sha1="bb7f1bc621959f2eba2a8f80ec351fe44f7beb48",
        sha256="efc8d3d7e7f5ac118ef6b97d05eacb628bc910b8ff8ccef18a7d893104175dac",
        size_bytes=2_318,
        semantics=(
            "cells are row-major with x as the inner iteration",
            "only complete width-by-height cells are admitted",
        ),
    ),
    EngineSourceEvidence(
        relative_path="src/resources/dye/dye.cpp",
        immutable_url=(
            "https://github.com/ManaPlus/ManaPlus/blob/"
            f"{MANAPLUS_ENGINE_COMMIT}/src/resources/dye/dye.cpp"
        ),
        git_blob_sha1="37a8d5950f149e34c67bed87a9d43d17cdb560e1",
        sha256="88868ec94f682bb52e91ae2cad4c284f3aefad9712f8751c8da9ed51bc0403dc",
        size_bytes=8_237,
        semantics=(
            "R,G,Y,B,M,C,W,S,A channel expressions describe pixel transforms",
            "one-letter channel placeholders consume external semicolon palettes",
            "explicit channel palettes are retained by instantiation",
        ),
    ),
    EngineSourceEvidence(
        relative_path="src/resources/animation/animation.cpp",
        immutable_url=(
            "https://github.com/ManaPlus/ManaPlus/blob/"
            f"{MANAPLUS_ENGINE_COMMIT}/src/resources/animation/animation.cpp"
        ),
        git_blob_sha1="f438744ccc74d2738f3004dc4636b2b1c3450a9c",
        sha256="defecd8def93ee2ae35004b0738b9933309b44fd063b5fc19176c4325781a085",
        size_bytes=3_011,
        semantics=("animation records preserve delay, offset, probability, and command type",),
    ),
    EngineSourceEvidence(
        relative_path="src/resources/sprite/animatedsprite.cpp",
        immutable_url=(
            "https://github.com/ManaPlus/ManaPlus/blob/"
            f"{MANAPLUS_ENGINE_COMMIT}/src/resources/sprite/animatedsprite.cpp"
        ),
        git_blob_sha1="a34aba041d74c98ae71022b839c102c281d3ad07",
        sha256="2742c23d2490d6c9bbf77354e532d48734adc330bbfb1bede516f23deb4cc5b9",
        size_bytes=12_925,
        semantics=(
            "positive delays drive automatic advancement; zero delay holds",
            "ordinary tracks wrap, while end returns the actor to stand",
            "jump, label, and goto alter runtime control flow probabilistically",
        ),
    ),
    EngineSourceEvidence(
        relative_path="src/resources/action.cpp",
        immutable_url=(
            "https://github.com/ManaPlus/ManaPlus/blob/"
            f"{MANAPLUS_ENGINE_COMMIT}/src/resources/action.cpp"
        ),
        git_blob_sha1="b2909b442b2f1bdaa1bbbc4aae042d42fa76b436",
        sha256="b2f8ebddb1105bf31a9b7bf7e3e85cb4bac9af9dc2b9585c2f6669ba699fe8d1",
        size_bytes=2_969,
        semantics=(
            "authored direction lookup may fall back at runtime",
            "the audit retains authored tracks and never synthesizes a fallback track",
        ),
    ),
    EngineSourceEvidence(
        relative_path="src/defaults.cpp",
        immutable_url=(
            f"https://github.com/ManaPlus/ManaPlus/blob/{MANAPLUS_ENGINE_COMMIT}/src/defaults.cpp"
        ),
        git_blob_sha1="5fa3a6b7b48f5abb381da49b806ab725e72488b1",
        sha256="0fee23b274d52e2e6bc7eaa5277963f83b599f4b355b31d8bef363393cf8fea7",
        size_bytes=27_184,
        semantics=("fixDeadAnimation has a default value of true",),
    ),
    EngineSourceEvidence(
        relative_path="src/progs/manaplus/client.cpp",
        immutable_url=(
            "https://github.com/ManaPlus/ManaPlus/blob/"
            f"{MANAPLUS_ENGINE_COMMIT}/src/progs/manaplus/client.cpp"
        ),
        git_blob_sha1="a632e6ccf307df2e90db4249193b339c18dbf30e",
        sha256="708f06757b113aa827c5a502602a01d98bf5799570bae8717ab7797909639d77",
        size_bytes=64_668,
        semantics=("fixDeadAnimation is read from the server feature database",),
    ),
)


ACTION_MAP: dict[str, str] = {
    "stand": "idle",
    "walk": "walk",
    "attack": "attack",
    "attack_bow": "attack_ranged",
    "attack_distance": "attack_ranged",
    "attack_magic": "cast",
    "attack_wand": "cast",
    "attack_chop": "attack_melee",
    "attack_chop_long": "attack_melee",
    "attack_chop_old": "attack_melee",
    "attack_dagger_stab": "attack_melee",
    "attack_scythe": "attack_melee",
    "attack_spear": "attack_melee",
    "attack_stab_long": "attack_melee",
    "attack_sword_stab": "attack_melee",
    "attack_2hand": "attack_melee",
    "dead": "death",
    "spawn": "spawn",
    "cast": "cast",
    "hurt": "hurt",
}

DIRECTION_MAP: dict[str, str] = {
    "default": "none",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "upleft": "up_left",
    "upright": "up_right",
    "downleft": "down_left",
    "downright": "down_right",
}


@dataclass(frozen=True)
class ArchiveMemberEvidence:
    ordinal: int
    member_path: str
    normalized_path: str
    logical_path: str
    member_kind: str
    size_bytes: int
    compressed_bytes: int
    crc32: int
    compression_method: int
    modified_at: tuple[int, int, int, int, int, int]
    content_sha256: str | None


@dataclass(frozen=True)
class XmlCommentClaim:
    location: SourceLocation
    verbatim: str


@dataclass(frozen=True)
class SourceImageAudit:
    member_path: str
    logical_path: str
    sha256: str
    size_bytes: int
    width: int
    height: int
    mode: str
    media_format: str
    has_alpha: bool


@dataclass(frozen=True)
class ImageSetDeclaration:
    name: str
    source_literal: str
    image_logical_path: str
    palette_expression: str | None
    cell_width: int
    cell_height: int
    offset_x: int
    offset_y: int
    column_count: int | None
    row_count: int | None
    complete_cell_count: int | None
    remainder_x: int | None
    remainder_y: int | None
    image: SourceImageAudit | None
    location: SourceLocation


@dataclass(frozen=True)
class IncludeDeclaration:
    include_literal: str
    target_logical_path: str
    resolved: bool
    location: SourceLocation


@dataclass(frozen=True)
class TimelineCommand:
    tag: str
    attributes: tuple[tuple[str, str], ...]
    declared_indices: tuple[int, ...]
    delay_literal: str | None
    effective_delay_ms: int | None
    offset_x_literal: str | None
    offset_y_literal: str | None
    effective_offset_x: int | None
    effective_offset_y: int | None
    location: SourceLocation


@dataclass(frozen=True)
class AnimationDeclaration:
    animation_ordinal: int
    direction_literal: str
    normalized_direction: str | None
    commands: tuple[TimelineCommand, ...]
    location: SourceLocation


@dataclass(frozen=True)
class ActionDeclaration:
    action_ordinal: int
    source_action: str
    normalized_action: str | None
    normalized_action_basis: str
    imageset_name: str
    animations: tuple[AnimationDeclaration, ...]
    location: SourceLocation


@dataclass(frozen=True)
class SpriteDocumentAudit:
    member_path: str
    logical_path: str
    sha256: str
    size_bytes: int
    family: str
    variant_count: int
    variant_offset: int
    imagesets: tuple[ImageSetDeclaration, ...]
    includes: tuple[IncludeDeclaration, ...]
    actions: tuple[ActionDeclaration, ...]
    comments: tuple[XmlCommentClaim, ...]


@dataclass(frozen=True)
class FrameRectangle:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class ResolvedFrame:
    ordinal: int
    source_frame_index: int
    unshifted_frame_index: int
    variant_index: int
    declaring_variant_count: int
    declaring_variant_offset: int
    declared_duration_ms: int
    duration_ms: int
    duration_adjustment_basis: str | None
    source_image: SourceImageAudit
    imageset_name: str
    imageset_source_literal: str
    palette_expression: str | None
    rectangle: FrameRectangle
    xml_offset_x: int
    xml_offset_y: int
    engine_offset_x: int
    engine_offset_y: int
    location: SourceLocation


@dataclass(frozen=True)
class EffectiveTrack:
    definition_logical_path: str
    definition_member_path: str
    definition_family: str
    variant_index: int
    source_action: str
    normalized_action: str | None
    normalized_action_basis: str
    action_ordinal: int
    action_location: SourceLocation
    direction_literal: str
    normalized_direction: str | None
    animation_ordinal: int
    animation_location: SourceLocation
    source_documents: tuple[str, ...]
    commands: tuple[TimelineCommand, ...]
    frames: tuple[ResolvedFrame, ...]
    declared_frame_count: int
    loop_mode: str
    issues: tuple[str, ...]

    @property
    def control_flow_present(self) -> bool:
        return any(command.tag in {"jump", "goto", "label"} for command in self.commands)


@dataclass(frozen=True)
class EntityClassification:
    entity_class: str
    entity_subclass: str
    basis: str
    quadruped_cue: bool | None


@dataclass(frozen=True)
class SemanticBinding:
    corpus: str
    entity_external_id: str
    entity_name: str | None
    entity_type_literal: str | None
    entity_location: SourceLocation
    layer_ordinal: int
    layer_count: int
    layer_role: str
    sprite_literal: str
    definition_logical_path: str
    palette_expression: str | None
    attributes: tuple[tuple[str, str], ...]
    definition_resolved: bool
    classification: EntityClassification
    location: SourceLocation


@dataclass(frozen=True)
class SemanticIncludeIssue:
    corpus: str
    source_logical_path: str
    include_literal: str
    target_logical_path: str
    reason: str
    location: SourceLocation


@dataclass(frozen=True)
class SemanticCorpusAudit:
    corpus: str
    root_logical_path: str
    document_count: int
    entity_count: int
    sprite_layer_reference_count: int
    unique_definition_path_count: int
    zero_layer_entity_count: int
    single_layer_entity_count: int
    multi_layer_entity_count: int
    palette_reference_count: int
    resolved_reference_count: int
    unresolved_reference_count: int
    include_issues: tuple[SemanticIncludeIssue, ...]


@dataclass(frozen=True)
class RightsClaim:
    claim_kind: str
    scope_path: str
    artists_raw: str | None
    licenses_raw: str | None
    unknown_contributor: bool
    location: SourceLocation
    verbatim: str


@dataclass(frozen=True)
class EvidenceDocument:
    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int
    verbatim: str
    scope: str


@dataclass(frozen=True)
class ImageRightsAssessment:
    image_logical_path: str
    status: str
    table_claims: tuple[RightsClaim, ...]
    missing_claims: tuple[RightsClaim, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RightsAudit:
    documents: tuple[EvidenceDocument, ...]
    claims: tuple[RightsClaim, ...]
    image_assessments: tuple[ImageRightsAssessment, ...]
    table_claim_count: int
    unique_table_path_count: int
    missing_claim_count: int
    unique_missing_path_count: int
    contradictory_path_count: int
    inconsistent_duplicate_path_count: int


@dataclass(frozen=True)
class TmwaAuditCounts:
    zip_member_count: int
    non_directory_member_count: int
    regular_file_member_count: int
    directory_member_count: int
    symlink_member_count: int
    expanded_member_bytes: int
    compressed_member_bytes: int
    xml_member_count: int
    png_member_count: int
    inspected_png_count: int
    sprite_document_count: int
    physical_imageset_count: int
    physical_include_count: int
    physical_action_count: int
    physical_animation_count: int
    physical_frame_command_count: int
    physical_sequence_command_count: int
    physical_end_command_count: int
    physical_jump_command_count: int
    physical_label_command_count: int
    physical_goto_command_count: int
    effective_track_count: int
    effective_resolved_frame_count: int
    xml_comment_count: int
    relevant_extracted_record_count: int


@dataclass(frozen=True)
class TmwaArchiveAudit:
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    archive_root: str
    repository_commit: str
    engine_semantics_commit: str
    fix_dead_animation: bool
    fix_dead_animation_basis: str
    inventory_sha256: str
    members: tuple[ArchiveMemberEvidence, ...]
    images: tuple[SourceImageAudit, ...]
    sprite_documents: tuple[SpriteDocumentAudit, ...]
    effective_tracks: tuple[EffectiveTrack, ...]
    semantic_corpora: tuple[SemanticCorpusAudit, ...]
    semantic_bindings: tuple[SemanticBinding, ...]
    xml_comments: tuple[XmlCommentClaim, ...]
    rights: RightsAudit
    engine_evidence: tuple[EngineSourceEvidence, ...]
    counts: TmwaAuditCounts

    @property
    def image_by_logical_path(self) -> dict[str, SourceImageAudit]:
        return {item.logical_path: item for item in self.images}

    @property
    def rights_by_image_path(self) -> dict[str, ImageRightsAssessment]:
        return {item.image_logical_path: item for item in self.rights.image_assessments}

    @property
    def binding_by_definition_path(self) -> dict[str, tuple[SemanticBinding, ...]]:
        grouped: defaultdict[str, list[SemanticBinding]] = defaultdict(list)
        for item in self.semantic_bindings:
            grouped[item.definition_logical_path].append(item)
        return {path: tuple(items) for path, items in grouped.items()}


@dataclass(frozen=True)
class _ParsedXml:
    root: ElementTree.Element
    lines: dict[int, int]
    comments: tuple[tuple[int, str], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_kind(info: zipfile.ZipInfo) -> str:
    if info.is_dir():
        return "directory"
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        return "symlink"
    return "file"


def _safe_logical_path(member_path: str, archive_root: str) -> str:
    if (
        not member_path
        or "\x00" in member_path
        or "\\" in member_path
        or member_path.startswith("/")
        or re.match(r"^[A-Za-z]:", member_path)
    ):
        raise TmwaArchiveError(f"Unsafe ZIP member path: {member_path!r}")
    pure = PurePosixPath(member_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise TmwaArchiveError(f"Unsafe ZIP member path: {member_path!r}")
    if not pure.parts or pure.parts[0] != archive_root:
        raise TmwaArchiveError(f"ZIP member is outside the pinned root: {member_path!r}")
    return PurePosixPath(*pure.parts[1:]).as_posix() if len(pure.parts) > 1 else ""


def _normalized_member_path(member_path: str) -> str:
    portable = unicodedata.normalize("NFC", member_path.replace("\\", "/"))
    parts = [part for part in portable.split("/") if part not in {"", "."}]
    return "/".join(parts)


def _parse_xml(data: bytes, member_path: str, logical_path: str) -> _ParsedXml:
    starts: list[tuple[str, int]] = []
    comments: list[tuple[int, str]] = []
    parser = expat.ParserCreate()

    def start(name: str, _attributes: dict[str, str]) -> None:
        starts.append((name, parser.CurrentLineNumber))

    def comment(text: str) -> None:
        comments.append((parser.CurrentLineNumber, text))

    parser.StartElementHandler = start
    parser.CommentHandler = comment
    try:
        parser.Parse(data, True)
        root = ElementTree.fromstring(data)
    except (expat.ExpatError, ElementTree.ParseError) as exc:
        raise TmwaParseError(f"Invalid XML in {logical_path}: {exc}") from exc
    elements = list(root.iter())
    if len(elements) != len(starts):
        raise TmwaParseError(f"Element/line accounting differs in {logical_path}")
    lines: dict[int, int] = {}
    for element, (name, line_number) in zip(elements, starts, strict=True):
        if element.tag != name:
            raise TmwaParseError(
                f"Element/line order differs in {logical_path}: {element.tag!r} != {name!r}"
            )
        lines[id(element)] = line_number
    return _ParsedXml(root=root, lines=lines, comments=tuple(comments))


def _location(
    member_path: str,
    logical_path: str,
    parsed: _ParsedXml,
    element: ElementTree.Element,
) -> SourceLocation:
    return SourceLocation(member_path, logical_path, parsed.lines[id(element)])


def _integer(value: str | None, *, default: int, context: str) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise TmwaParseError(f"Expected an integer for {context}, got {value!r}") from exc


def _palette_parts(literal: str) -> tuple[str, str | None]:
    base, separator, expression = literal.partition("|")
    return base, expression if separator else None


def _frame_indices(tag: str, attributes: dict[str, str], context: str) -> tuple[int, ...]:
    if tag == "frame":
        return (_integer(attributes.get("index"), default=0, context=f"{context} index"),)
    if tag != "sequence":
        return ()
    start = _integer(attributes.get("start"), default=0, context=f"{context} start")
    end = _integer(attributes.get("end"), default=0, context=f"{context} end")
    if "repeat" in attributes or "value" in attributes:
        # Those engine forms are retained in attributes, but deliberately not
        # guessed here; the pinned archive does not use either form.
        return ()
    step = 1 if end >= start else -1
    return tuple(range(start, end + step, step))


def _timeline_command(
    element: ElementTree.Element,
    *,
    member_path: str,
    logical_path: str,
    parsed: _ParsedXml,
) -> TimelineCommand:
    attributes = {str(key): str(value) for key, value in element.attrib.items()}
    indices = _frame_indices(element.tag, attributes, f"{logical_path}:{element.tag}")
    delay_literal = attributes.get("delay") if element.tag in {"frame", "sequence"} else None
    effective_delay = (
        min(100_000, max(0, _integer(delay_literal, default=0, context="frame delay")))
        if element.tag in {"frame", "sequence"}
        else None
    )
    offset_x_literal = attributes.get("offsetX") if indices else None
    offset_y_literal = attributes.get("offsetY") if indices else None
    return TimelineCommand(
        tag=element.tag,
        attributes=tuple(sorted(attributes.items())),
        declared_indices=indices,
        delay_literal=delay_literal,
        effective_delay_ms=effective_delay,
        offset_x_literal=offset_x_literal,
        offset_y_literal=offset_y_literal,
        effective_offset_x=(
            _integer(offset_x_literal, default=0, context="frame offsetX") if indices else None
        ),
        effective_offset_y=(
            _integer(offset_y_literal, default=0, context="frame offsetY") if indices else None
        ),
        location=_location(member_path, logical_path, parsed, element),
    )


def _family(logical_path: str) -> str:
    relative = logical_path.removeprefix(SPRITE_ROOT)
    parts = PurePosixPath(relative).parts
    return parts[0] if len(parts) > 1 else "root"


def _inventory_digest(members: Iterable[ArchiveMemberEvidence]) -> str:
    payload = [
        {
            "ordinal": item.ordinal,
            "path": item.normalized_path,
            "directory": item.member_kind == "directory",
            "symlink": item.member_kind == "symlink",
            "compressed_bytes": item.compressed_bytes,
            "uncompressed_bytes": item.size_bytes,
            "crc32": item.crc32,
            "compression_method": item.compression_method,
            "modified_at": item.modified_at,
        }
        for item in members
    ]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _audit_png(
    data: bytes,
    *,
    member_path: str,
    logical_path: str,
) -> SourceImageAudit:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            media_format = str(image.format)
            mode = image.mode
            width, height = image.size
    except Exception as exc:  # Pillow raises format-specific subclasses.
        raise TmwaParseError(f"Invalid PNG {logical_path}: {exc}") from exc
    if media_format != "PNG":
        raise TmwaParseError(f"PNG-named member is {media_format!r}: {logical_path}")
    return SourceImageAudit(
        member_path=member_path,
        logical_path=logical_path,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        width=width,
        height=height,
        mode=mode,
        media_format=media_format,
        has_alpha="A" in mode,
    )


def _imageset_declaration(
    element: ElementTree.Element,
    *,
    member_path: str,
    logical_path: str,
    parsed: _ParsedXml,
    images: dict[str, SourceImageAudit],
) -> ImageSetDeclaration:
    source_literal = element.attrib["src"]
    image_path, palette_expression = _palette_parts(source_literal)
    cell_width = _integer(
        element.attrib.get("width"), default=0, context=f"{logical_path} imageset width"
    )
    cell_height = _integer(
        element.attrib.get("height"), default=0, context=f"{logical_path} imageset height"
    )
    if cell_width <= 0 or cell_height <= 0:
        raise TmwaParseError(f"Non-positive imageset cell in {logical_path}")
    image = images.get(image_path)
    if image is None:
        columns = rows = count = remainder_x = remainder_y = None
    else:
        columns = image.width // cell_width
        rows = image.height // cell_height
        count = columns * rows
        remainder_x = image.width % cell_width
        remainder_y = image.height % cell_height
    return ImageSetDeclaration(
        name=element.attrib["name"],
        source_literal=source_literal,
        image_logical_path=image_path,
        palette_expression=palette_expression,
        cell_width=cell_width,
        cell_height=cell_height,
        offset_x=_integer(element.attrib.get("offsetX"), default=0, context="imageset offsetX"),
        offset_y=_integer(element.attrib.get("offsetY"), default=0, context="imageset offsetY"),
        column_count=columns,
        row_count=rows,
        complete_cell_count=count,
        remainder_x=remainder_x,
        remainder_y=remainder_y,
        image=image,
        location=_location(member_path, logical_path, parsed, element),
    )


def _sprite_document(
    *,
    data: bytes,
    member_path: str,
    logical_path: str,
    parsed: _ParsedXml,
    images: dict[str, SourceImageAudit],
    sprite_paths: frozenset[str],
) -> SpriteDocumentAudit:
    if parsed.root.tag != "sprite":
        raise TmwaParseError(f"Expected <sprite> root in {logical_path}")
    imagesets: list[ImageSetDeclaration] = []
    includes: list[IncludeDeclaration] = []
    actions: list[ActionDeclaration] = []
    action_ordinal = 0
    for child in parsed.root:
        if child.tag == "imageset":
            imagesets.append(
                _imageset_declaration(
                    child,
                    member_path=member_path,
                    logical_path=logical_path,
                    parsed=parsed,
                    images=images,
                )
            )
            continue
        if child.tag == "include":
            include_literal = child.attrib["file"]
            target = SPRITE_ROOT + PurePosixPath(include_literal).as_posix().lstrip("/")
            includes.append(
                IncludeDeclaration(
                    include_literal=include_literal,
                    target_logical_path=target,
                    resolved=target in sprite_paths,
                    location=_location(member_path, logical_path, parsed, child),
                )
            )
            continue
        if child.tag != "action":
            continue
        source_action = child.attrib["name"]
        animations: list[AnimationDeclaration] = []
        for animation_ordinal, animation in enumerate(child.findall("animation")):
            direction_literal = animation.attrib.get("direction", "default") or "default"
            commands = tuple(
                _timeline_command(
                    command,
                    member_path=member_path,
                    logical_path=logical_path,
                    parsed=parsed,
                )
                for command in animation
            )
            animations.append(
                AnimationDeclaration(
                    animation_ordinal=animation_ordinal,
                    direction_literal=direction_literal,
                    normalized_direction=DIRECTION_MAP.get(direction_literal),
                    commands=commands,
                    location=_location(member_path, logical_path, parsed, animation),
                )
            )
        actions.append(
            ActionDeclaration(
                action_ordinal=action_ordinal,
                source_action=source_action,
                normalized_action=ACTION_MAP.get(source_action),
                normalized_action_basis=(
                    "exact_reviewed_tmwa_action_literal"
                    if source_action in ACTION_MAP
                    else "unmapped_literal_retained"
                ),
                imageset_name=child.attrib["imageset"],
                animations=tuple(animations),
                location=_location(member_path, logical_path, parsed, child),
            )
        )
        action_ordinal += 1
    comments = tuple(
        XmlCommentClaim(
            location=SourceLocation(member_path, logical_path, line_number),
            verbatim=f"<!--{text}-->",
        )
        for line_number, text in parsed.comments
    )
    return SpriteDocumentAudit(
        member_path=member_path,
        logical_path=logical_path,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        family=_family(logical_path),
        variant_count=_integer(
            parsed.root.attrib.get("variants"), default=0, context=f"{logical_path} variants"
        ),
        variant_offset=_integer(
            parsed.root.attrib.get("variant_offset"),
            default=0,
            context=f"{logical_path} variant_offset",
        ),
        imagesets=tuple(imagesets),
        includes=tuple(includes),
        actions=tuple(actions),
        comments=comments,
    )


def _command_attributes(command: TimelineCommand) -> dict[str, str]:
    return dict(command.attributes)


def _resolve_animation(
    *,
    definition: SpriteDocumentAudit,
    declaring_document: SpriteDocumentAudit,
    action: ActionDeclaration,
    animation: AnimationDeclaration,
    imagesets: dict[str, ImageSetDeclaration],
    source_documents: tuple[str, ...],
    variant_index: int,
    fix_dead_animation: bool,
) -> EffectiveTrack:
    frames: list[ResolvedFrame] = []
    issues: set[str] = set()
    declared_frame_count = 0
    timeline_tags = [command.tag for command in animation.commands]
    supported_tags = {"frame", "sequence", "end", "jump", "goto", "label"}
    if any(command.tag not in supported_tags for command in animation.commands):
        issues.add("unsupported_timeline_command")
    if any(
        command.tag == "sequence"
        and any(key in {"repeat", "value"} for key, _value in command.attributes)
        for command in animation.commands
    ):
        issues.add("unsupported_sequence_repeat_or_value")
    for command in animation.commands:
        if command.tag not in {"frame", "sequence"}:
            continue
        declared_frame_count += len(command.declared_indices)
        attributes = _command_attributes(command)
        imageset_name = attributes.get("imageset", action.imageset_name)
        imageset = imagesets.get(imageset_name)
        if imageset is None:
            issues.add("imageset_name_unresolved")
            continue
        if imageset.image is None or imageset.complete_cell_count is None:
            issues.add("source_image_missing")
            continue
        for unshifted_index in command.declared_indices:
            source_index = unshifted_index + declaring_document.variant_offset * variant_index
            if source_index < 0 or source_index >= imageset.complete_cell_count:
                issues.add("frame_index_out_of_complete_grid_bounds")
                continue
            if imageset.column_count is None or imageset.column_count <= 0:
                issues.add("imageset_has_no_complete_columns")
                continue
            column = source_index % imageset.column_count
            row = source_index // imageset.column_count
            xml_offset_x = command.effective_offset_x or 0
            xml_offset_y = command.effective_offset_y or 0
            frames.append(
                ResolvedFrame(
                    ordinal=len(frames),
                    source_frame_index=source_index,
                    unshifted_frame_index=unshifted_index,
                    variant_index=variant_index,
                    declaring_variant_count=declaring_document.variant_count,
                    declaring_variant_offset=declaring_document.variant_offset,
                    declared_duration_ms=command.effective_delay_ms or 0,
                    duration_ms=command.effective_delay_ms or 0,
                    duration_adjustment_basis=None,
                    source_image=imageset.image,
                    imageset_name=imageset_name,
                    imageset_source_literal=imageset.source_literal,
                    palette_expression=imageset.palette_expression,
                    rectangle=FrameRectangle(
                        x=column * imageset.cell_width,
                        y=row * imageset.cell_height,
                        width=imageset.cell_width,
                        height=imageset.cell_height,
                    ),
                    xml_offset_x=xml_offset_x,
                    xml_offset_y=xml_offset_y,
                    engine_offset_x=(
                        xml_offset_x + imageset.offset_x - imageset.cell_width // 2 + 16
                    ),
                    engine_offset_y=(xml_offset_y + imageset.offset_y - imageset.cell_height + 32),
                    location=command.location,
                )
            )
    if fix_dead_animation and action.source_action == "dead" and frames:
        frames[-1] = replace(
            frames[-1],
            duration_ms=0,
            duration_adjustment_basis=(
                "manaplus_fixDeadAnimation_true_forces_final_dead_frame_delay_zero"
            ),
        )
    if not frames:
        issues.add("no_resolved_frames")
    if "jump" in timeline_tags or "goto" in timeline_tags or "label" in timeline_tags:
        loop_mode = "runtime_control_flow_unresolved"
    elif len(frames) == 1 and frames[0].duration_ms == 0:
        loop_mode = "hold"
    elif any(frame.duration_ms == 0 for frame in frames):
        loop_mode = "zero_delay_hold_inside_multi_frame_track"
        issues.add("multi_frame_track_contains_zero_delay_hold")
    elif "end" in timeline_tags:
        end_positions = [index for index, tag in enumerate(timeline_tags) if tag == "end"]
        if len(end_positions) == 1 and end_positions[0] == len(timeline_tags) - 1:
            loop_mode = "one_shot_return_to_stand"
        else:
            loop_mode = "malformed_or_nonterminal_end"
            issues.add("end_command_not_single_terminal_command")
    else:
        loop_mode = "loop"
    if any(
        command.tag in {"frame", "sequence"}
        and _command_attributes(command).get("rand", "100") != "100"
        for command in animation.commands
    ):
        issues.add("probabilistic_frame_gate_unresolved")
    return EffectiveTrack(
        definition_logical_path=definition.logical_path,
        definition_member_path=definition.member_path,
        definition_family=definition.family,
        variant_index=variant_index,
        source_action=action.source_action,
        normalized_action=action.normalized_action,
        normalized_action_basis=action.normalized_action_basis,
        action_ordinal=action.action_ordinal,
        action_location=action.location,
        direction_literal=animation.direction_literal,
        normalized_direction=animation.normalized_direction,
        animation_ordinal=animation.animation_ordinal,
        animation_location=animation.location,
        source_documents=source_documents,
        commands=animation.commands,
        frames=tuple(frames),
        declared_frame_count=declared_frame_count,
        loop_mode=loop_mode,
        issues=tuple(sorted(issues)),
    )


def _effective_definition_state(
    definition: SpriteDocumentAudit,
    by_path: dict[str, SpriteDocumentAudit],
) -> tuple[
    dict[str, ImageSetDeclaration],
    dict[str, tuple[SpriteDocumentAudit, ActionDeclaration]],
    tuple[str, ...],
]:
    imagesets: dict[str, ImageSetDeclaration] = {}
    actions: dict[str, tuple[SpriteDocumentAudit, ActionDeclaration]] = {}
    processed: set[str] = set()
    source_documents: list[str] = []

    def load(document: SpriteDocumentAudit) -> None:
        if document.logical_path in processed:
            return
        processed.add(document.logical_path)
        source_documents.append(document.member_path)
        imageset_by_line = {item.location.line_number: item for item in document.imagesets}
        action_by_line = {item.location.line_number: item for item in document.actions}
        include_by_line = {item.location.line_number: item for item in document.includes}
        events: list[tuple[int, str]] = []
        events.extend((line, "imageset") for line in imageset_by_line)
        events.extend((line, "action") for line in action_by_line)
        events.extend((line, "include") for line in include_by_line)
        for line_number, kind in sorted(events):
            if kind == "imageset":
                declaration = imageset_by_line[line_number]
                imagesets.setdefault(declaration.name, declaration)
            elif kind == "action":
                action = action_by_line[line_number]
                if action.imageset_name in imagesets:
                    actions[action.source_action] = (document, action)
            else:
                include = include_by_line[line_number]
                target = by_path.get(include.target_logical_path)
                if target is not None:
                    load(target)

    load(definition)
    return imagesets, actions, tuple(source_documents)


def _effective_tracks(
    documents: tuple[SpriteDocumentAudit, ...],
    *,
    variant_index: int = 0,
    fix_dead_animation: bool = True,
) -> tuple[EffectiveTrack, ...]:
    by_path = {item.logical_path: item for item in documents}
    result: list[EffectiveTrack] = []
    for definition in documents:
        imagesets, actions, source_document_tuple = _effective_definition_state(definition, by_path)
        for _source_action, (declaring_document, action) in sorted(actions.items()):
            tracks: dict[str, AnimationDeclaration] = {}
            for animation in action.animations:
                tracks[animation.direction_literal] = animation
            for _direction, animation in sorted(tracks.items()):
                result.append(
                    _resolve_animation(
                        definition=definition,
                        declaring_document=declaring_document,
                        action=action,
                        animation=animation,
                        imagesets=imagesets,
                        source_documents=source_document_tuple,
                        variant_index=variant_index,
                        fix_dead_animation=fix_dead_animation,
                    )
                )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.definition_logical_path,
                item.source_action,
                item.direction_literal,
                item.action_location.member_path,
                item.action_location.line_number,
            ),
        )
    )


def _classify_entity(
    *,
    corpus: str,
    entity_name: str | None,
    entity_type: str | None,
    definition_path: str,
) -> EntityClassification:
    if corpus == "items":
        subtype = entity_type or "item"
        return EntityClassification("object", subtype, "source_items_type_literal", None)
    if corpus in {"npcs", "avatars"}:
        return EntityClassification("humanoid", "npc_or_avatar", "source_corpus_role", None)
    if corpus in {"effects", "emotes"}:
        return EntityClassification("effect", corpus.rstrip("s"), "source_corpus_role", None)
    text = " ".join(filter(None, (entity_name, entity_type, definition_path))).casefold()
    token_text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = frozenset(token_text.split())
    undead = {"skeleton", "skull", "zombie", "ghost", "wraith", "undead", "rotter"}
    objects = {
        "beehive",
        "clover",
        "flower",
        "gift",
        "grass",
        "hive",
        "patch",
        "snowflower",
        "spelt",
        "tree",
        "woodling",
    }
    humanoids = {"goblin", "sasquatch", "tengu", "terranite", "yeti"}
    animals = {
        "archant",
        "bat",
        "bee",
        "bird",
        "cat",
        "croc",
        "duck",
        "frog",
        "junglefowl",
        "maggot",
        "moubi",
        "mouboo",
        "penguin",
        "piou",
        "serqet",
        "snake",
        "spider",
        "tortuga",
    }
    quadrupeds = {"cat", "croc", "frog", "moubi", "mouboo", "tortuga"}
    if tokens & undead:
        return EntityClassification("monster", "undead", "reviewed_source_name_token", None)
    if tokens & objects:
        return EntityClassification(
            "object",
            "animated_object_or_plant",
            "reviewed_source_name_token",
            None,
        )
    if tokens & humanoids:
        return EntityClassification(
            "humanoid", "fantasy_humanoid", "reviewed_source_name_token", None
        )
    if tokens & animals:
        return EntityClassification(
            "animal",
            "animal",
            "reviewed_source_name_token",
            True if tokens & quadrupeds else None,
        )
    if corpus == "monsters":
        return EntityClassification("monster", "creature", "source_monster_corpus_role", None)
    return EntityClassification("unknown", corpus.rstrip("s") or "unknown", "unmapped", None)


def _definition_path(sprite_literal: str) -> str:
    base, _palette = _palette_parts(sprite_literal)
    normalized = PurePosixPath(base).as_posix().lstrip("/")
    return normalized if normalized.startswith(SPRITE_ROOT) else SPRITE_ROOT + normalized


def _layer_role(corpus: str, layer_count: int, definition_path: str) -> str:
    if corpus == "items":
        if "/equipment/" in definition_path:
            return "modular_equipment_layer"
        if "/hairstyles/" in definition_path:
            return "modular_hair_layer"
        if "/races/" in definition_path or "/model/" in definition_path:
            return "modular_race_or_body_layer"
        return "item_sprite_reference"
    if layer_count == 1:
        return "complete_single_layer_entity"
    if "/equipment/" in definition_path:
        return "modular_equipment_layer"
    if "/hairstyles/" in definition_path:
        return "modular_hair_layer"
    if "/races/" in definition_path or "/model/" in definition_path:
        return "modular_race_or_body_layer"
    return "explicit_multi_layer_runtime_composite"


def _audit_semantic_corpus(
    *,
    corpus: str,
    root_logical_path: str,
    entity_tag: str,
    parsed_xml: dict[str, _ParsedXml],
    member_paths: dict[str, str],
    sprite_paths: frozenset[str],
) -> tuple[SemanticCorpusAudit, tuple[SemanticBinding, ...]]:
    processed: set[str] = set()
    visiting: set[str] = set()
    issues: list[SemanticIncludeIssue] = []
    entities: list[tuple[str, _ParsedXml, ElementTree.Element, int]] = []

    def visit(logical_path: str) -> None:
        if logical_path in processed:
            return
        parsed = parsed_xml.get(logical_path)
        if parsed is None:
            return
        processed.add(logical_path)
        visiting.add(logical_path)
        member_path = member_paths[logical_path]
        entity_ordinal = 0
        for child in parsed.root:
            if child.tag == entity_tag:
                entities.append((logical_path, parsed, child, entity_ordinal))
                entity_ordinal += 1
                continue
            if child.tag != "include":
                continue
            include_literal = child.attrib.get("name") or child.attrib.get("file") or ""
            target = PurePosixPath(include_literal).as_posix().lstrip("/")
            location = _location(member_path, logical_path, parsed, child)
            if target not in parsed_xml:
                issues.append(
                    SemanticIncludeIssue(
                        corpus=corpus,
                        source_logical_path=logical_path,
                        include_literal=include_literal,
                        target_logical_path=target,
                        reason="included_document_unavailable",
                        location=location,
                    )
                )
            elif target in visiting:
                issues.append(
                    SemanticIncludeIssue(
                        corpus=corpus,
                        source_logical_path=logical_path,
                        include_literal=include_literal,
                        target_logical_path=target,
                        reason="include_cycle_suppressed",
                        location=location,
                    )
                )
            else:
                visit(target)
        visiting.remove(logical_path)

    visit(root_logical_path)
    bindings: list[SemanticBinding] = []
    zero = single = multiple = 0
    palette_count = resolved_count = unresolved_count = 0
    for logical_path, parsed, entity, entity_ordinal in entities:
        member_path = member_paths[logical_path]
        sprites = entity.findall("sprite")
        if not sprites:
            zero += 1
        elif len(sprites) == 1:
            single += 1
        else:
            multiple += 1
        external_id = entity.attrib.get("id") or f"{logical_path}#{entity_ordinal}"
        name = entity.attrib.get("name")
        entity_type = entity.attrib.get("type")
        entity_location = _location(member_path, logical_path, parsed, entity)
        for layer_ordinal, sprite in enumerate(sprites):
            literal = (sprite.text or "").strip()
            definition_path = _definition_path(literal)
            _base, palette = _palette_parts(literal)
            if palette is not None:
                palette_count += 1
            resolved = definition_path in sprite_paths
            resolved_count += int(resolved)
            unresolved_count += int(not resolved)
            bindings.append(
                SemanticBinding(
                    corpus=corpus,
                    entity_external_id=str(external_id),
                    entity_name=name,
                    entity_type_literal=entity_type,
                    entity_location=entity_location,
                    layer_ordinal=layer_ordinal,
                    layer_count=len(sprites),
                    layer_role=_layer_role(corpus, len(sprites), definition_path),
                    sprite_literal=literal,
                    definition_logical_path=definition_path,
                    palette_expression=palette,
                    attributes=tuple(sorted((str(k), str(v)) for k, v in sprite.attrib.items())),
                    definition_resolved=resolved,
                    classification=_classify_entity(
                        corpus=corpus,
                        entity_name=name,
                        entity_type=entity_type,
                        definition_path=definition_path,
                    ),
                    location=_location(member_path, logical_path, parsed, sprite),
                )
            )
    corpus_audit = SemanticCorpusAudit(
        corpus=corpus,
        root_logical_path=root_logical_path,
        document_count=len(processed),
        entity_count=len(entities),
        sprite_layer_reference_count=len(bindings),
        unique_definition_path_count=len({item.definition_logical_path for item in bindings}),
        zero_layer_entity_count=zero,
        single_layer_entity_count=single,
        multi_layer_entity_count=multiple,
        palette_reference_count=palette_count,
        resolved_reference_count=resolved_count,
        unresolved_reference_count=unresolved_count,
        include_issues=tuple(
            sorted(
                issues,
                key=lambda item: (
                    item.source_logical_path,
                    item.location.line_number,
                    item.target_logical_path,
                ),
            )
        ),
    )
    return corpus_audit, tuple(
        sorted(
            bindings,
            key=lambda item: (
                item.corpus,
                item.entity_location.logical_path,
                item.entity_location.line_number,
                item.layer_ordinal,
            ),
        )
    )


def _evidence_document(
    logical_path: str,
    member_path: str,
    data: bytes,
    scope: str,
) -> EvidenceDocument:
    return EvidenceDocument(
        logical_path=logical_path,
        member_path=member_path,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        verbatim=data.decode("utf-8", "strict"),
        scope=scope,
    )


def _rights_audit(
    *,
    data_by_logical_path: dict[str, bytes],
    member_paths: dict[str, str],
    xml_comments: tuple[XmlCommentClaim, ...],
    images: tuple[SourceImageAudit, ...],
) -> RightsAudit:
    documents = (
        _evidence_document(
            "license.md",
            member_paths["license.md"],
            data_by_logical_path["license.md"],
            "asset_path_table_with_repository_self-warning",
        ),
        _evidence_document(
            "license-missing",
            member_paths["license-missing"],
            data_by_logical_path["license-missing"],
            "explicit_missing_license_path_list",
        ),
        _evidence_document(
            "COPYING",
            member_paths["COPYING"],
            data_by_logical_path["COPYING"],
            "repository_license_text_not_automatically_per_asset",
        ),
    )
    table_claims: list[RightsClaim] = []
    license_member = member_paths["license.md"]
    for line_number, line in enumerate(documents[0].verbatim.splitlines(), 1):
        if not line.startswith("`") or line.count("|") < 2:
            continue
        left, _separator, remainder = line.partition("|")
        middle, _separator, right = remainder.rpartition("|")
        scope_path = left.strip().strip("`")
        artists = middle.strip()
        licenses = right.strip()
        table_claims.append(
            RightsClaim(
                claim_kind="license_table_path_claim",
                scope_path=scope_path,
                artists_raw=artists,
                licenses_raw=licenses,
                unknown_contributor=":grey_question:" in artists,
                location=SourceLocation(license_member, "license.md", line_number),
                verbatim=line,
            )
        )
    missing_claims: list[RightsClaim] = []
    missing_member = member_paths["license-missing"]
    prefix = "Missing license for "
    for line_number, line in enumerate(documents[1].verbatim.splitlines(), 1):
        if not line.startswith(prefix):
            continue
        missing_claims.append(
            RightsClaim(
                claim_kind="explicit_missing_license_claim",
                scope_path=line[len(prefix) :],
                artists_raw=None,
                licenses_raw=None,
                unknown_contributor=True,
                location=SourceLocation(missing_member, "license-missing", line_number),
                verbatim=line,
            )
        )
    comment_claims = tuple(
        RightsClaim(
            claim_kind="embedded_xml_comment_verbatim",
            scope_path=comment.location.logical_path,
            artists_raw=None,
            licenses_raw=None,
            unknown_contributor=False,
            location=comment.location,
            verbatim=comment.verbatim,
        )
        for comment in xml_comments
    )
    table_by_path: defaultdict[str, list[RightsClaim]] = defaultdict(list)
    missing_by_path: defaultdict[str, list[RightsClaim]] = defaultdict(list)
    for claim in table_claims:
        table_by_path[claim.scope_path].append(claim)
    for claim in missing_claims:
        missing_by_path[claim.scope_path].append(claim)
    image_assessments: list[ImageRightsAssessment] = []
    for image in images:
        table = tuple(table_by_path.get(image.logical_path, ()))
        missing = tuple(missing_by_path.get(image.logical_path, ()))
        reasons: list[str] = []
        signatures = {(claim.artists_raw, claim.licenses_raw) for claim in table}
        if table and missing:
            reasons.append("path_has_documented_and_missing_license_claims")
        if len(signatures) > 1:
            reasons.append("duplicate_table_rows_disagree")
        if not table:
            reasons.append("no_asset_path_license_table_claim")
        if missing:
            reasons.append("explicitly_listed_in_license_missing")
        if any(claim.unknown_contributor for claim in table):
            reasons.append("unknown_contributor_marker")
        if any(not claim.licenses_raw or claim.licenses_raw == "???" for claim in table):
            reasons.append("unknown_or_empty_license_literal")
        if "path_has_documented_and_missing_license_claims" in reasons or (
            "duplicate_table_rows_disagree" in reasons
        ):
            status = "contradictory"
        elif "explicitly_listed_in_license_missing" in reasons:
            status = "license_missing"
        elif "no_asset_path_license_table_claim" in reasons:
            status = "unclaimed"
        elif any(reason.startswith("unknown_") for reason in reasons):
            status = "unresolved_contributor_or_license"
        else:
            status = "documented_path_claim"
        image_assessments.append(
            ImageRightsAssessment(
                image_logical_path=image.logical_path,
                status=status,
                table_claims=table,
                missing_claims=missing,
                reasons=tuple(reasons),
            )
        )
    table_paths = set(table_by_path)
    missing_paths = set(missing_by_path)
    inconsistent = sum(
        len({(claim.artists_raw, claim.licenses_raw) for claim in claims}) > 1
        for claims in table_by_path.values()
    )
    return RightsAudit(
        documents=documents,
        claims=tuple(table_claims) + tuple(missing_claims) + comment_claims,
        image_assessments=tuple(image_assessments),
        table_claim_count=len(table_claims),
        unique_table_path_count=len(table_paths),
        missing_claim_count=len(missing_claims),
        unique_missing_path_count=len(missing_paths),
        contradictory_path_count=len(table_paths & missing_paths),
        inconsistent_duplicate_path_count=inconsistent,
    )


def _fix_dead_animation_setting(parsed_xml: dict[str, _ParsedXml]) -> tuple[bool, str]:
    """Resolve the server feature over the pinned ManaPlus default of true."""

    value = True
    features = parsed_xml.get("features.xml")
    if features is None:
        return value, "manaplus_default_true_features_document_absent"
    basis = "manaplus_default_true_features_xml_has_no_override"
    for option in features.root.findall("option"):
        if option.attrib.get("name") != "fixDeadAnimation":
            continue
        literal = option.attrib.get("value", "").strip().casefold()
        if literal in {"true", "1", "yes", "on"}:
            value = True
        elif literal in {"false", "0", "no", "off"}:
            value = False
        else:
            raise TmwaParseError(
                f"Unknown features.xml fixDeadAnimation boolean literal: {literal!r}"
            )
        basis = f"features_xml_explicit_override:{literal}"
    return value, basis


def audit_tmwa_archive(archive_path: str | Path) -> TmwaArchiveAudit:
    """Audit a TMWA client-data ZIP without extracting or writing any file."""

    path = Path(archive_path)
    archive_size = path.stat().st_size
    archive_sha256 = _sha256_file(path)
    members: list[ArchiveMemberEvidence] = []
    data_by_logical_path: dict[str, bytes] = {}
    member_paths: dict[str, str] = {}
    parsed_xml: dict[str, _ParsedXml] = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) > 100_000:
            raise TmwaArchiveError(f"ZIP member count exceeds audit ceiling: {len(infos)}")
        if sum(info.file_size for info in infos) > 4 * 1024**3:
            raise TmwaArchiveError("ZIP expanded byte count exceeds audit ceiling")
        roots = {PurePosixPath(info.filename).parts[0] for info in infos if info.filename}
        if len(roots) != 1:
            raise TmwaArchiveError(f"Expected one archive root, found {sorted(roots)!r}")
        archive_root = next(iter(roots))
        seen_member_paths: set[str] = set()
        seen_normalized_paths: set[str] = set()
        seen_casefolded_paths: set[str] = set()
        seen_logical_paths: set[str] = set()
        for ordinal, info in enumerate(infos):
            if info.filename in seen_member_paths:
                raise TmwaArchiveError(f"Duplicate ZIP member: {info.filename!r}")
            seen_member_paths.add(info.filename)
            logical_path = _safe_logical_path(info.filename, archive_root)
            normalized_path = _normalized_member_path(info.filename)
            if normalized_path in seen_normalized_paths:
                raise TmwaArchiveError(f"Duplicate normalized ZIP path: {normalized_path!r}")
            folded_path = normalized_path.casefold()
            if folded_path in seen_casefolded_paths:
                raise TmwaArchiveError(f"Case-colliding ZIP path: {normalized_path!r}")
            seen_normalized_paths.add(normalized_path)
            seen_casefolded_paths.add(folded_path)
            if logical_path in seen_logical_paths:
                raise TmwaArchiveError(f"Duplicate normalized ZIP path: {logical_path!r}")
            seen_logical_paths.add(logical_path)
            if info.flag_bits & 0x1:
                raise TmwaArchiveError(f"Encrypted ZIP member is unsupported: {info.filename!r}")
            if info.file_size > 1024**3:
                raise TmwaArchiveError(f"ZIP member exceeds audit ceiling: {info.filename!r}")
            kind = _member_kind(info)
            if kind == "directory" and info.file_size != 0:
                raise TmwaArchiveError(f"ZIP directory carries data: {info.filename!r}")
            ratio = info.file_size / max(1, info.compress_size)
            if ratio > 1_000:
                raise TmwaArchiveError(f"ZIP member compression ratio is unsafe: {info.filename!r}")
            suffix = PurePosixPath(logical_path).suffix.casefold()
            relevant = kind == "file" and (
                suffix in {".xml", ".png", ".txt", ".md"}
                or logical_path in {"COPYING", "license-missing"}
            )
            content_sha256: str | None = None
            if relevant:
                try:
                    content = archive.read(info)
                except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                    raise TmwaArchiveError(f"Could not validate {info.filename}: {exc}") from exc
                if len(content) != info.file_size:
                    raise TmwaArchiveError(f"ZIP member size differs for {info.filename!r}")
                content_sha256 = hashlib.sha256(content).hexdigest()
                data_by_logical_path[logical_path] = content
            members.append(
                ArchiveMemberEvidence(
                    ordinal=ordinal,
                    member_path=info.filename,
                    normalized_path=normalized_path,
                    logical_path=logical_path,
                    member_kind=kind,
                    size_bytes=info.file_size,
                    compressed_bytes=info.compress_size,
                    crc32=info.CRC,
                    compression_method=info.compress_type,
                    modified_at=info.date_time,
                    content_sha256=content_sha256,
                )
            )
            member_paths[logical_path] = info.filename

    inventory_sha256 = _inventory_digest(members)
    images = tuple(
        _audit_png(
            data,
            member_path=member_paths[logical_path],
            logical_path=logical_path,
        )
        for logical_path, data in sorted(data_by_logical_path.items())
        if logical_path.casefold().endswith(".png")
    )
    image_by_path = {item.logical_path: item for item in images}
    xml_comments: list[XmlCommentClaim] = []
    for logical_path, data in sorted(data_by_logical_path.items()):
        if not logical_path.casefold().endswith(".xml"):
            continue
        member_path = member_paths[logical_path]
        parsed = _parse_xml(data, member_path, logical_path)
        parsed_xml[logical_path] = parsed
        xml_comments.extend(
            XmlCommentClaim(
                location=SourceLocation(member_path, logical_path, line_number),
                verbatim=f"<!--{text}-->",
            )
            for line_number, text in parsed.comments
        )
    sprite_paths = frozenset(
        path for path in parsed_xml if path.startswith(SPRITE_ROOT) and path.endswith(".xml")
    )
    sprite_documents = tuple(
        _sprite_document(
            data=data_by_logical_path[logical_path],
            member_path=member_paths[logical_path],
            logical_path=logical_path,
            parsed=parsed_xml[logical_path],
            images=image_by_path,
            sprite_paths=sprite_paths,
        )
        for logical_path in sorted(sprite_paths)
    )
    fix_dead_animation, fix_dead_animation_basis = _fix_dead_animation_setting(parsed_xml)
    tracks = _effective_tracks(
        sprite_documents,
        fix_dead_animation=fix_dead_animation,
    )
    corpus_specs = (
        ("monsters", "monsters.xml", "monster"),
        ("npcs", "npcs.xml", "npc"),
        ("items", "items.xml", "item"),
        ("avatars", "avatars.xml", "avatar"),
        ("pets", "pets.xml", "pet"),
        ("horses", "horses.xml", "horse"),
        ("mercenaries", "mercenaries.xml", "mercenary"),
        ("homunculuses", "homunculuses.xml", "homunculus"),
        ("elementals", "elementals.xml", "elemental"),
        ("effects", "effects.xml", "effect"),
        ("emotes", "emotes.xml", "emote"),
    )
    semantic_corpora: list[SemanticCorpusAudit] = []
    semantic_bindings: list[SemanticBinding] = []
    for corpus, root_path, entity_tag in corpus_specs:
        if root_path not in parsed_xml:
            continue
        corpus_audit, bindings = _audit_semantic_corpus(
            corpus=corpus,
            root_logical_path=root_path,
            entity_tag=entity_tag,
            parsed_xml=parsed_xml,
            member_paths=member_paths,
            sprite_paths=sprite_paths,
        )
        semantic_corpora.append(corpus_audit)
        semantic_bindings.extend(bindings)
    comment_tuple = tuple(
        sorted(
            xml_comments,
            key=lambda item: (item.location.logical_path, item.location.line_number, item.verbatim),
        )
    )
    rights = _rights_audit(
        data_by_logical_path=data_by_logical_path,
        member_paths=member_paths,
        xml_comments=comment_tuple,
        images=images,
    )
    tags = Counter(
        command.tag
        for document in sprite_documents
        for action in document.actions
        for animation in action.animations
        for command in animation.commands
    )
    counts = TmwaAuditCounts(
        zip_member_count=len(members),
        non_directory_member_count=sum(item.member_kind != "directory" for item in members),
        regular_file_member_count=sum(item.member_kind == "file" for item in members),
        directory_member_count=sum(item.member_kind == "directory" for item in members),
        symlink_member_count=sum(item.member_kind == "symlink" for item in members),
        expanded_member_bytes=sum(item.size_bytes for item in members),
        compressed_member_bytes=sum(item.compressed_bytes for item in members),
        xml_member_count=len(parsed_xml),
        png_member_count=len(images),
        inspected_png_count=len(images),
        sprite_document_count=len(sprite_documents),
        physical_imageset_count=sum(len(item.imagesets) for item in sprite_documents),
        physical_include_count=sum(len(item.includes) for item in sprite_documents),
        physical_action_count=sum(len(item.actions) for item in sprite_documents),
        physical_animation_count=sum(
            len(action.animations) for item in sprite_documents for action in item.actions
        ),
        physical_frame_command_count=tags["frame"],
        physical_sequence_command_count=tags["sequence"],
        physical_end_command_count=tags["end"],
        physical_jump_command_count=tags["jump"],
        physical_label_command_count=tags["label"],
        physical_goto_command_count=tags["goto"],
        effective_track_count=len(tracks),
        effective_resolved_frame_count=sum(len(item.frames) for item in tracks),
        xml_comment_count=len(comment_tuple),
        relevant_extracted_record_count=len(data_by_logical_path),
    )
    commit_match = re.search(r"([0-9a-f]{40})$", archive_root)
    repository_commit = commit_match.group(1) if commit_match else "unknown"
    return TmwaArchiveAudit(
        archive_path=str(path.resolve()),
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_size,
        archive_root=archive_root,
        repository_commit=repository_commit,
        engine_semantics_commit=MANAPLUS_ENGINE_COMMIT,
        fix_dead_animation=fix_dead_animation,
        fix_dead_animation_basis=fix_dead_animation_basis,
        inventory_sha256=inventory_sha256,
        members=tuple(members),
        images=images,
        sprite_documents=sprite_documents,
        effective_tracks=tracks,
        semantic_corpora=tuple(semantic_corpora),
        semantic_bindings=tuple(semantic_bindings),
        xml_comments=comment_tuple,
        rights=rights,
        engine_evidence=ENGINE_SOURCE_EVIDENCE,
        counts=counts,
    )


def _known_count_failures(audit: TmwaArchiveAudit) -> tuple[str, ...]:
    expected = {
        "zip_member_count": EXPECTED_ZIP_MEMBER_COUNT,
        "regular_file_member_count": EXPECTED_REGULAR_FILE_COUNT,
        "expanded_member_bytes": EXPECTED_EXPANDED_MEMBER_BYTES,
        "compressed_member_bytes": EXPECTED_COMPRESSED_MEMBER_BYTES,
        "xml_member_count": 2_521,
        "png_member_count": 1_636,
        "inspected_png_count": 1_636,
        "sprite_document_count": 756,
        "physical_imageset_count": 762,
        "physical_include_count": 209,
        "physical_action_count": 3_501,
        "physical_animation_count": 12_744,
        "physical_frame_command_count": 31_395,
        "physical_sequence_command_count": 1_601,
        "physical_end_command_count": 7_062,
        "physical_jump_command_count": 6,
        "physical_label_command_count": 3,
        "physical_goto_command_count": 5,
        "xml_comment_count": 587,
        "relevant_extracted_record_count": 4_169,
    }
    return tuple(
        f"{name}={getattr(audit.counts, name)} (expected {value})"
        for name, value in expected.items()
        if getattr(audit.counts, name) != value
    )


def audit_known_tmwa_archive(archive_path: str | Path) -> TmwaArchiveAudit:
    """Audit only the exact acquired TMWA CAS object and enforce regressions."""

    path = Path(archive_path)
    if path.stat().st_size != EXPECTED_TMWA_ARCHIVE_BYTES:
        raise TmwaArchiveError("Refusing a TMWA archive with an unexpected byte length")
    if _sha256_file(path) != EXPECTED_TMWA_ARCHIVE_SHA256:
        raise TmwaArchiveError("Refusing a TMWA archive outside the exact acquired CAS pin")
    audit = audit_tmwa_archive(path)
    identity_failures: list[str] = []
    if audit.archive_sha256 != EXPECTED_TMWA_ARCHIVE_SHA256:
        identity_failures.append("archive SHA-256")
    if audit.archive_root != EXPECTED_TMWA_ARCHIVE_ROOT:
        identity_failures.append("archive root")
    if audit.repository_commit != TMWA_CLIENT_DATA_COMMIT:
        identity_failures.append("repository commit")
    if audit.inventory_sha256 != EXPECTED_INVENTORY_SHA256:
        identity_failures.append("central-directory inventory SHA-256")
    count_failures = _known_count_failures(audit)
    corpus = {item.corpus: item for item in audit.semantic_corpora}
    monsters = corpus.get("monsters")
    if monsters is None:
        identity_failures.append("monsters corpus")
    elif (
        monsters.entity_count,
        monsters.sprite_layer_reference_count,
        monsters.unique_definition_path_count,
        monsters.single_layer_entity_count,
        monsters.multi_layer_entity_count,
    ) != (233, 442, 224, 167, 66):
        identity_failures.append("monsters semantic counts")
    if monsters is not None and not any(
        issue.target_logical_path == "mods/monsters.xml"
        and issue.reason == "included_document_unavailable"
        for issue in monsters.include_issues
    ):
        identity_failures.append("unavailable mods/monsters.xml include")
    if any(image.mode != "RGBA" or not image.has_alpha for image in audit.images):
        identity_failures.append("all-PNG RGBA inspection invariant")
    if not audit.fix_dead_animation or audit.fix_dead_animation_basis != (
        "manaplus_default_true_features_xml_has_no_override"
    ):
        identity_failures.append("fixDeadAnimation feature resolution")
    if audit.rights.table_claim_count != 1_534:
        identity_failures.append("license.md path-claim count")
    if audit.rights.missing_claim_count != 309:
        identity_failures.append("license-missing claim count")
    if audit.rights.contradictory_path_count != 4:
        identity_failures.append("rights contradiction count")
    if audit.rights.inconsistent_duplicate_path_count != 1:
        identity_failures.append("inconsistent duplicate rights count")
    if identity_failures or count_failures:
        details = "; ".join((*identity_failures, *count_failures))
        raise TmwaArchiveError(f"Pinned TMWA regression mismatch: {details}")
    return audit


__all__ = [
    "ACTION_MAP",
    "DIRECTION_MAP",
    "ENGINE_SOURCE_EVIDENCE",
    "EXPECTED_INVENTORY_SHA256",
    "EXPECTED_TMWA_ARCHIVE_ROOT",
    "EXPECTED_TMWA_ARCHIVE_SHA256",
    "MANAPLUS_ENGINE_COMMIT",
    "RIGHTS_SCOPE_CAVEAT",
    "SOURCE_ID",
    "TMWA_CLIENT_DATA_COMMIT",
    "ActionDeclaration",
    "ArchiveMemberEvidence",
    "EffectiveTrack",
    "EntityClassification",
    "EvidenceDocument",
    "FrameRectangle",
    "ImageRightsAssessment",
    "ImageSetDeclaration",
    "ResolvedFrame",
    "RightsAudit",
    "RightsClaim",
    "SemanticBinding",
    "SemanticCorpusAudit",
    "SemanticIncludeIssue",
    "SourceImageAudit",
    "SourceLocation",
    "SpriteDocumentAudit",
    "TimelineCommand",
    "TmwaArchiveAudit",
    "TmwaArchiveError",
    "TmwaAuditCounts",
    "TmwaParseError",
    "XmlCommentClaim",
    "audit_known_tmwa_archive",
    "audit_tmwa_archive",
]
