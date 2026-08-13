"""Exact-snapshot audit for Flare: Empyrean Campaign animation data.

The game repository stores animation geometry in INI-like ``.txt`` files.  This
module follows the paired Flare engine parser instead of guessing sprite-sheet
grids: compressed ``frame`` records carry explicit rectangles, offsets, frame
indices, and directions; the one grid-style parent timeline in the snapshot is
retained without geometry because it declares neither an image nor a cell size.

The archive audit is read-only.  It understands the runtime mod stack
``fantasycore, empyrean_campaign``, follows ``INCLUDE`` directives with source
locations intact, inspects PNG headers in memory, and never extracts members or
writes to the corpus database.
"""

from __future__ import annotations

import hashlib
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

EXPECTED_FLARE_ARCHIVE_SHA256 = "9c8e58bb704f55174928990a1a375c213ffb2847d7b928d6117a84db3e1215cc"
FLARE_GAME_COMMIT = "af6eee6d339ac98011864bfe89da837fe7769c28"
FLARE_ENGINE_COMMIT = "cf4d42f09442c2d8d08b6f1bdf9b6043e73a4443"
FLARE_GAME_REPOSITORY_URL = "https://github.com/flareteam/flare-game"
FLARE_ENGINE_REPOSITORY_URL = "https://github.com/flareteam/flare-engine"
FLARE_GAME_COMMIT_URL = f"{FLARE_GAME_REPOSITORY_URL}/tree/{FLARE_GAME_COMMIT}"
FLARE_ENGINE_COMMIT_URL = f"{FLARE_ENGINE_REPOSITORY_URL}/tree/{FLARE_ENGINE_COMMIT}"

FLARE_ACTIVE_MODS = ("fantasycore", "empyrean_campaign")
FLARE_DIRECTION_NAMES = (
    "southwest",
    "west",
    "northwest",
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
)
FLARE_DIRECTION_TOKENS = ("SW", "W", "NW", "N", "NE", "E", "SE", "S")
FLARE_DEFAULT_ENGINE_FPS = 60

_EXPECTED_ROOT = f"flare-game-{FLARE_GAME_COMMIT}"
_ANIMATION_TYPES = frozenset({"play_once", "back_forth", "looped"})
_ACTION_MAP: Mapping[str, str] = {
    "stance": "idle",
    "run": "run",
    "run_alt": "run",
    "swing": "attack",
    "dash_attack": "attack",
    "shield_bash": "attack",
    "shoot": "shoot",
    "cast": "cast",
    "cast_alt": "cast",
    "die": "death",
    "critdie": "death",
    "hit": "hurt",
    "block": "block",
    "spawn": "spawn",
}
_EVIDENCE_PATHS = (
    "LICENSE.txt",
    "README",
    "CREDITS.txt",
    "CONTRIBUTING.md",
    "distribution/org.flarerpg.Flare.appdata.xml",
    "mods/fantasycore/cutscenes/credits.txt",
    "mods/fantasycore/cutscenes/credits_fantasycore.txt",
    "mods/empyrean_campaign/cutscenes/credits.txt",
    "mods/empyrean_campaign/cutscenes/credits_empyrean.txt",
)
_ATTRIBUTION_METADATA_TERMS = (
    "artist",
    "author",
    "copyright",
    "credit",
    "creator",
    "description",
    "license",
    "rights",
    "source",
)
_DURATION_RE = re.compile(r"^(?P<value>[0-9]+)(?P<suffix>ms|s)?$")


class FlareArchiveError(ValueError):
    """Raised when a ZIP cannot be audited as a safe Flare repository tree."""


class FlareParseError(ValueError):
    """Raised when a Flare animation declaration is structurally invalid."""


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
class SourceLocation:
    logical_path: str
    member_path: str | None
    line_number: int


@dataclass(frozen=True)
class IncludeDirective:
    included_path: str
    location: SourceLocation


@dataclass(frozen=True)
class ImageBinding:
    image_id: str
    logical_path: str
    location: SourceLocation


@dataclass(frozen=True)
class FrameRecord:
    """One effective compressed frame assignment.

    ``index`` is temporal order and ``direction`` uses the engine's exact
    ``SW,W,NW,N,NE,E,SE,S`` integer mapping.  A repeated assignment to the same
    pair remains in ``raw_frames``; the last assignment is effective at runtime.
    """

    index: int
    direction: int
    direction_name: str
    rectangle: Rectangle
    offset: Point
    image_id: str
    image_path: str | None
    within_image_bounds: bool | None
    location: SourceLocation


@dataclass(frozen=True)
class FrameSlot:
    index: int
    direction: int
    direction_name: str
    frame: FrameRecord | None
    explicit: bool
    fallback_from_direction: int | None


@dataclass(frozen=True)
class DirectionTrack:
    direction: int
    direction_token: str
    direction_name: str
    frames: tuple[FrameSlot, ...]

    @property
    def explicit_frame_count(self) -> int:
        return sum(slot.explicit for slot in self.frames)

    @property
    def fallback_frame_count(self) -> int:
        return sum(slot.fallback_from_direction is not None for slot in self.frames)

    @property
    def complete(self) -> bool:
        return all(slot.frame is not None for slot in self.frames)


@dataclass(frozen=True)
class TickSchedule:
    tick_rate: int
    tick_count: int
    frame_indices: tuple[int, ...]
    per_frame_tick_counts: tuple[int, ...]
    effective_duration_milliseconds: float


@dataclass(frozen=True)
class AnimationAction:
    source_action: str
    normalized_action: str | None
    normalized_action_basis: str
    declared_frame_count: int
    duration_literal: str
    duration_milliseconds: int
    nominal_fps: float
    animation_type: str
    loop_mode: str
    position: int | None
    active_frames: tuple[int, ...] | Literal["all"] | None
    active_sub_frame: str | None
    section_location: SourceLocation
    layout_mode: str
    raw_frames: tuple[FrameRecord, ...]
    direction_tracks: tuple[DirectionTrack, ...]
    default_tick_schedule: TickSchedule

    @property
    def has_exact_geometry(self) -> bool:
        return any(
            slot.frame is not None for track in self.direction_tracks for slot in track.frames
        )

    @property
    def source_frame_order(self) -> tuple[int, ...]:
        return tuple(range(self.declared_frame_count))


@dataclass(frozen=True)
class AnimationDefinition:
    logical_path: str
    member_path: str | None
    source_mod: str | None
    sha256: str | None
    size_bytes: int | None
    includes: tuple[IncludeDirective, ...]
    source_documents: tuple[str, ...]
    image_bindings: tuple[ImageBinding, ...]
    render_size: Point | None
    render_offset: Point
    blend_mode: str
    alpha_mod: int
    color_mod: tuple[int, int, int]
    entity_family: str
    identity: str
    body_variant: str | None
    attachment_id: str | None
    actions: tuple[AnimationAction, ...]
    unknown_keys: tuple[tuple[str, str, SourceLocation], ...]


@dataclass(frozen=True)
class PngMetadataSummary:
    png_file_count: int
    readable_png_count: int
    metadata_field_counts: tuple[tuple[str, int], ...]
    png_with_text_count: int
    png_with_comment_count: int
    gimp_comment_count: int
    png_with_source_file_field_count: int
    png_with_attribution_field_count: int
    png_with_software_field_count: int
    unreadable_member_paths: tuple[str, ...]
    attribution_fields: tuple[tuple[str, str], ...]
    source_file_fields: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SourceImageAudit:
    logical_path: str
    member_path: str
    source_mod: str
    width: int
    height: int
    image_mode: str
    image_format: str | None
    has_transparency: bool
    size_bytes: int
    compressed_size_bytes: int
    crc32: str
    sha256: str
    definition_reference_count: int


@dataclass(frozen=True)
class EntityBinding:
    entity_kind: str
    definition_path: str
    member_path: str
    source_mod: str
    display_name: str | None
    humanoid: bool | None
    categories: tuple[str, ...]
    animation_paths: tuple[str, ...]
    animation_locations: tuple[SourceLocation, ...]
    is_template: bool


@dataclass(frozen=True)
class AnimationUsage:
    usage_kind: str
    owner_id: str | None
    owner_name: str | None
    animation_path: str
    location: SourceLocation


@dataclass(frozen=True)
class AttachmentBinding:
    item_id: str
    item_name: str | None
    layer_slot: str
    gfx_id: str
    candidate_animation_paths: tuple[str, ...]
    location: SourceLocation


@dataclass(frozen=True)
class HeroLayerOrder:
    direction: int
    direction_token: str
    direction_name: str
    layers_back_to_front: tuple[str, ...]
    location: SourceLocation


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
class SymlinkEvidence:
    relative_path: str
    member_path: str
    target: str
    unix_mode: str


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    related_paths: tuple[str, ...]


@dataclass(frozen=True)
class FlareArchiveCounts:
    zip_member_count: int
    regular_file_member_count: int
    directory_member_count: int
    symlink_member_count: int
    expanded_member_bytes: int
    expanded_regular_file_bytes: int
    archive_png_file_count: int
    active_mod_png_file_count: int
    animation_definition_file_count: int
    fantasycore_animation_definition_count: int
    empyrean_animation_definition_count: int
    included_animation_definition_count: int
    physical_action_declaration_count: int
    physical_explicit_frame_record_count: int
    action_count: int
    exact_geometry_action_count: int
    geometry_missing_action_count: int
    direction_track_count: int
    explicit_direction_track_count: int
    fallback_only_direction_track_count: int
    unresolved_direction_track_count: int
    explicit_frame_record_count: int
    effective_frame_slot_count: int
    explicit_frame_slot_count: int
    direction_zero_fallback_slot_count: int
    unresolved_frame_slot_count: int
    complete_eight_direction_action_count: int
    play_once_action_count: int
    looped_action_count: int
    back_forth_action_count: int
    referenced_source_image_count: int
    missing_source_image_count: int
    out_of_bounds_frame_record_count: int
    entity_binding_count: int
    concrete_entity_binding_count: int
    template_entity_binding_count: int
    enemy_binding_count: int
    npc_binding_count: int
    explicit_humanoid_binding_count: int
    animation_usage_count: int
    attachment_binding_count: int
    avatar_attachment_definition_count: int
    attachment_parent_mismatch_count: int
    hero_layer_direction_count: int
    evidence_document_count: int


@dataclass(frozen=True)
class FlareArchiveAudit:
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    repository_commit: str | None
    repository_url: str
    commit_url: str | None
    engine_semantics_commit: str
    engine_semantics_url: str
    root_prefix: str
    active_mods: tuple[str, ...]
    counts: FlareArchiveCounts
    definitions: tuple[AnimationDefinition, ...]
    source_images: tuple[SourceImageAudit, ...]
    entities: tuple[EntityBinding, ...]
    usages: tuple[AnimationUsage, ...]
    attachments: tuple[AttachmentBinding, ...]
    hero_layers: tuple[HeroLayerOrder, ...]
    evidence_documents: tuple[EvidenceDocument, ...]
    symlinks: tuple[SymlinkEvidence, ...]
    png_metadata: PngMetadataSummary
    definition_family_counts: tuple[tuple[str, int], ...]
    body_variant_counts: tuple[tuple[str, int], ...]
    usage_counts: tuple[tuple[str, int], ...]
    action_counts: tuple[tuple[str, int], ...]
    direction_explicit_frame_counts: tuple[tuple[str, int], ...]
    issues: tuple[AuditIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Event:
    section: str
    key: str
    value: str
    location: SourceLocation
    new_section: bool


@dataclass(frozen=True)
class _PhysicalFile:
    logical_path: str
    member_path: str
    source_mod: str
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class _PngInfo:
    member_path: str
    relative_path: str
    logical_path: str | None
    source_mod: str | None
    width: int
    height: int
    mode: str
    image_format: str | None
    has_transparency: bool
    metadata: tuple[tuple[str, str], ...]
    size_bytes: int
    compressed_size_bytes: int
    crc32: str
    sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_path(path: str, *, label: str = "path") -> str:
    normalized = path.replace("\\", "/").strip()
    parsed = PurePosixPath(normalized)
    if not normalized or parsed.is_absolute() or ".." in parsed.parts:
        raise FlareParseError(f"{label} must be a safe relative path: {path!r}")
    return str(parsed)


def parse_duration_milliseconds(literal: str) -> int:
    """Parse the exact Flare duration vocabulary without applying an FPS."""

    match = _DURATION_RE.fullmatch(literal.strip())
    if not match:
        raise FlareParseError(f"invalid duration literal: {literal!r}")
    value = int(match.group("value"))
    return value * 1000 if match.group("suffix") == "s" else value


def engine_tick_schedule(
    frame_count: int,
    duration_literal: str,
    *,
    tick_rate: int = FLARE_DEFAULT_ENGINE_FPS,
) -> TickSchedule:
    """Reproduce ``Parse::toDuration`` and ``Animation::setup`` at one tick rate.

    Flare stores a duration for the whole forward action, not a per-frame delay.
    Milliseconds are rounded to the nearest engine tick (halves upward), then the
    frame indices are distributed either evenly or with the engine's Bresenham
    branch.  ``back_forth`` playback direction is a separate loop property and
    is intentionally not folded into this forward schedule.
    """

    if isinstance(frame_count, bool) or frame_count < 0:
        raise FlareParseError("frame_count must be a non-negative integer")
    if isinstance(tick_rate, bool) or tick_rate <= 0:
        raise FlareParseError("tick_rate must be a positive integer")
    literal = duration_literal.strip()
    parse_duration_milliseconds(literal)
    match = _DURATION_RE.fullmatch(literal)
    assert match is not None
    source_value = int(match.group("value"))
    if source_value == 0:
        tick_count = 0
    elif match.group("suffix") == "s":
        tick_count = source_value * tick_rate
    else:
        tick_count = (source_value * tick_rate * 2 + 1000) // 2000
        tick_count = max(tick_count, 1)

    indices: list[int] = []
    if frame_count > 0 and tick_count > 0 and tick_count % frame_count == 0:
        for frame_index in range(frame_count):
            indices.extend([frame_index] * (tick_count // frame_count))
    elif frame_count > 0 and tick_count > 0:
        # Literal translation of Animation::setup's Bresenham branch.
        x1 = tick_count - 1
        y1 = frame_count - 1
        dx = x1
        dy = y1
        decision = 2 * dy - dx
        indices.append(0)
        x = 1
        y = 0
        while x <= x1:
            if decision > 0:
                y += 1
                decision += (2 * dy) - (2 * dx)
            else:
                decision += 2 * dy
            indices.append(y)
            x += 1

    per_frame = tuple(indices.count(index) for index in range(frame_count))
    return TickSchedule(
        tick_rate=tick_rate,
        tick_count=tick_count,
        frame_indices=tuple(indices),
        per_frame_tick_counts=per_frame,
        effective_duration_milliseconds=(len(indices) * 1000 / tick_rate),
    )


def direction_index(value: str) -> int:
    """Parse Flare's exact named/numeric direction mapping."""

    token = value.strip()
    if token in FLARE_DIRECTION_TOKENS:
        return FLARE_DIRECTION_TOKENS.index(token)
    try:
        result = int(token)
    except ValueError as exc:
        raise FlareParseError(f"invalid direction: {value!r}") from exc
    if not 0 <= result < 8:
        raise FlareParseError(f"direction is outside 0..7: {value!r}")
    return result


def _decode_text(payload: bytes, logical_path: str) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise FlareParseError(f"{logical_path} is not UTF-8 text") from exc


def _direct_events(
    payload: str,
    *,
    logical_path: str,
    member_path: str | None,
    inherited_section: str = "",
) -> tuple[tuple[_Event | IncludeDirective, ...], tuple[IncludeDirective, ...]]:
    section = inherited_section
    pending_section = False
    records: list[_Event | IncludeDirective] = []
    includes: list[IncludeDirective] = []
    for line_number, raw_line in enumerate(payload.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "APPEND":
            continue
        location = SourceLocation(logical_path, member_path, line_number)
        if line.startswith("["):
            closing = line.find("]")
            if closing < 0:
                raise FlareParseError(f"{logical_path}:{line_number}: unterminated section")
            section = line[1:closing]
            pending_section = True
            continue
        first_space = line.find(" ")
        if first_space >= 0 and line[:first_space] == "INCLUDE":
            included = _normalize_path(line[first_space + 1 :], label="INCLUDE path")
            directive = IncludeDirective(included, location)
            records.append(directive)
            includes.append(directive)
            continue
        if "=" not in line:
            raise FlareParseError(
                f"{logical_path}:{line_number}: expected key=value, section, or INCLUDE"
            )
        key, value = (part.strip() for part in line.split("=", 1))
        if not key:
            raise FlareParseError(f"{logical_path}:{line_number}: empty key")
        records.append(_Event(section, key, value, location, pending_section))
        pending_section = False
    return tuple(records), tuple(includes)


def _mapping_resolver(files: Mapping[str, str | bytes]) -> dict[str, _PhysicalFile]:
    resolved: dict[str, _PhysicalFile] = {}
    for raw_path, raw_payload in files.items():
        path = _normalize_path(raw_path)
        payload = raw_payload.encode("utf-8") if isinstance(raw_payload, str) else raw_payload
        if not isinstance(payload, bytes):
            raise TypeError("animation source mapping values must be str or bytes")
        resolved[path] = _PhysicalFile(path, path, "mapping", payload, _sha256_bytes(payload))
    return resolved


def _physical_sequence(
    value: _PhysicalFile | Sequence[_PhysicalFile],
) -> tuple[_PhysicalFile, ...]:
    if isinstance(value, _PhysicalFile):
        return (value,)
    return tuple(value)


def _expand_events(
    logical_path: str,
    files: Mapping[str, _PhysicalFile | Sequence[_PhysicalFile]],
    *,
    inherited_section: str = "",
    stack: tuple[str, ...] = (),
) -> tuple[tuple[_Event, ...], tuple[IncludeDirective, ...], tuple[str, ...]]:
    path = _normalize_path(logical_path)
    if path in stack:
        cycle = " -> ".join((*stack, path))
        raise FlareParseError(f"recursive INCLUDE chain: {cycle}")
    try:
        physical_files = _physical_sequence(files[path])
    except KeyError as exc:
        raise FlareParseError(f"included file is missing: {path}") from exc
    events: list[_Event] = []
    includes: list[IncludeDirective] = []
    documents: list[str] = []
    current_section = inherited_section
    for physical_index, physical in enumerate(physical_files):
        documents.append(physical.member_path)
        text = _decode_text(physical.payload, path)
        direct, direct_includes = _direct_events(
            text,
            logical_path=path,
            member_path=physical.member_path,
            inherited_section=current_section,
        )
        includes.extend(direct_includes)
        first_event_in_file = True
        for record in direct:
            if isinstance(record, IncludeDirective):
                child_events, child_includes, child_documents = _expand_events(
                    record.included_path,
                    files,
                    inherited_section=current_section,
                    stack=(*stack, path),
                )
                if (first_event_in_file and physical_index > 0) and child_events:
                    child_events = (replace(child_events[0], new_section=True), *child_events[1:])
                events.extend(child_events)
                includes.extend(child_includes)
                documents.extend(child_documents)
                first_event_in_file = False
                continue
            current_section = record.section
            events.append(
                replace(
                    record,
                    new_section=(
                        record.new_section or (first_event_in_file and physical_index > 0)
                    ),
                )
            )
            first_event_in_file = False
    return tuple(events), tuple(includes), tuple(dict.fromkeys(documents))


def _split_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(","))


def _parse_int(value: str, *, location: SourceLocation, field: str) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise FlareParseError(
            f"{location.logical_path}:{location.line_number}: {field} must be an integer"
        ) from exc


def _parse_int_pair(value: str, *, location: SourceLocation, field: str) -> Point:
    parts = _split_csv(value)
    if len(parts) != 2:
        raise FlareParseError(
            f"{location.logical_path}:{location.line_number}: {field} needs two integers"
        )
    return Point(
        _parse_int(parts[0], location=location, field=field),
        _parse_int(parts[1], location=location, field=field),
    )


def _parse_color(value: str, *, location: SourceLocation) -> tuple[int, int, int]:
    parts = _split_csv(value)
    if len(parts) != 3:
        raise FlareParseError(
            f"{location.logical_path}:{location.line_number}: color_mod needs three integers"
        )
    color = tuple(_parse_int(part, location=location, field="color_mod") for part in parts)
    if any(not 0 <= channel <= 255 for channel in color):
        raise FlareParseError(
            f"{location.logical_path}:{location.line_number}: color_mod is outside 0..255"
        )
    return color  # type: ignore[return-value]


def _definition_identity(logical_path: str) -> tuple[str, str, str | None, str | None]:
    path = PurePosixPath(logical_path)
    parts = path.parts
    if len(parts) < 2 or parts[0] != "animations":
        return "unknown", path.stem, None, None
    if len(parts) == 2:
        return "hero_parent", path.stem, None, None
    family = parts[1]
    identity = "/".join((*parts[1:-1], path.stem))
    if family != "avatar" or len(parts) < 4:
        singular = {
            "enemies": "enemy",
            "loot": "loot",
            "npcs": "npc",
            "powers": "power",
        }.get(family, family)
        return singular, identity, None, None
    return "avatar_attachment", identity, parts[2], path.stem


def _parse_image_binding(event: _Event) -> ImageBinding:
    parts = _split_csv(event.value)
    if not parts or not parts[0]:
        raise FlareParseError(
            f"{event.location.logical_path}:{event.location.line_number}: image path is empty"
        )
    if len(parts) > 2:
        raise FlareParseError(
            f"{event.location.logical_path}:{event.location.line_number}: image has too many fields"
        )
    return ImageBinding(
        image_id=parts[1] if len(parts) == 2 else "",
        logical_path=_normalize_path(parts[0], label="image path"),
        location=event.location,
    )


def _selected_image_path(
    image_id: str,
    bindings: Sequence[ImageBinding],
) -> str | None:
    effective: dict[str, str] = {}
    first_key: str | None = None
    for binding in bindings:
        if first_key is None:
            first_key = binding.image_id
        effective[binding.image_id] = binding.logical_path
    if image_id in effective:
        return effective[image_id]
    if first_key is not None:
        return effective[first_key]
    return None


def _parse_frame(
    event: _Event,
    *,
    image_bindings: Sequence[ImageBinding],
    image_sizes: Mapping[str, tuple[int, int]],
) -> FrameRecord:
    parts = _split_csv(event.value)
    if len(parts) not in {8, 9}:
        raise FlareParseError(
            f"{event.location.logical_path}:{event.location.line_number}: frame needs 8 or 9 fields"
        )
    index = _parse_int(parts[0], location=event.location, field="frame index")
    direction = direction_index(parts[1])
    x, y, width, height, offset_x, offset_y = (
        _parse_int(part, location=event.location, field="frame") for part in parts[2:8]
    )
    if index < 0:
        raise FlareParseError(
            f"{event.location.logical_path}:{event.location.line_number}: negative frame index"
        )
    if width <= 0 or height <= 0:
        raise FlareParseError(
            f"{event.location.logical_path}:{event.location.line_number}: non-positive frame size"
        )
    image_id = parts[8] if len(parts) == 9 else ""
    image_path = _selected_image_path(image_id, image_bindings)
    bounds: bool | None = None
    if image_path in image_sizes:
        image_width, image_height = image_sizes[image_path]
        bounds = x >= 0 and y >= 0 and x + width <= image_width and y + height <= image_height
    return FrameRecord(
        index=index,
        direction=direction,
        direction_name=FLARE_DIRECTION_NAMES[direction],
        rectangle=Rectangle(x, y, width, height),
        offset=Point(offset_x, offset_y),
        image_id=image_id,
        image_path=image_path,
        within_image_bounds=bounds,
        location=event.location,
    )


def _frame_tracks(
    declared_frames: int,
    raw_frames: Sequence[FrameRecord],
) -> tuple[DirectionTrack, ...]:
    effective: dict[tuple[int, int], FrameRecord] = {}
    for frame in raw_frames:
        if frame.index >= declared_frames:
            raise FlareParseError(
                f"{frame.location.logical_path}:{frame.location.line_number}: frame index "
                f"{frame.index} is outside declared count {declared_frames}"
            )
        effective[(frame.index, frame.direction)] = frame
    tracks: list[DirectionTrack] = []
    for direction in range(8):
        slots: list[FrameSlot] = []
        for index in range(declared_frames):
            explicit = effective.get((index, direction))
            fallback = None
            frame = explicit
            if frame is None and direction != 0:
                frame = effective.get((index, 0))
                if frame is not None:
                    fallback = 0
            slots.append(
                FrameSlot(
                    index=index,
                    direction=direction,
                    direction_name=FLARE_DIRECTION_NAMES[direction],
                    frame=frame,
                    explicit=explicit is not None,
                    fallback_from_direction=fallback,
                )
            )
        tracks.append(
            DirectionTrack(
                direction=direction,
                direction_token=FLARE_DIRECTION_TOKENS[direction],
                direction_name=FLARE_DIRECTION_NAMES[direction],
                frames=tuple(slots),
            )
        )
    return tuple(tracks)


def _uncompressed_tracks(
    declared_frames: int,
    position: int | None,
    render_size: Point | None,
    render_offset: Point,
    image_bindings: Sequence[ImageBinding],
    image_sizes: Mapping[str, tuple[int, int]],
    location: SourceLocation,
    image_id: str = "",
) -> tuple[DirectionTrack, ...]:
    if position is None or render_size is None or render_size.x <= 0 or render_size.y <= 0:
        return _frame_tracks(declared_frames, ())
    image_path = _selected_image_path(image_id, image_bindings)
    if image_path is None:
        return _frame_tracks(declared_frames, ())
    records: list[FrameRecord] = []
    for index in range(declared_frames):
        for direction in range(8):
            rectangle = Rectangle(
                render_size.x * (position + index),
                render_size.y * direction,
                render_size.x,
                render_size.y,
            )
            bounds: bool | None = None
            if image_path in image_sizes:
                width, height = image_sizes[image_path]
                bounds = rectangle.right <= width and rectangle.bottom <= height
            records.append(
                FrameRecord(
                    index=index,
                    direction=direction,
                    direction_name=FLARE_DIRECTION_NAMES[direction],
                    rectangle=rectangle,
                    offset=render_offset,
                    image_id=image_id,
                    image_path=image_path,
                    within_image_bounds=bounds,
                    location=location,
                )
            )
    return _frame_tracks(declared_frames, records)


def _parse_active_frames(
    value: str,
    *,
    location: SourceLocation,
) -> tuple[int, ...] | Literal["all"]:
    if value == "all":
        return "all"
    parsed = tuple(
        _parse_int(part, location=location, field="active_frame") for part in _split_csv(value)
    )
    if any(index < 0 for index in parsed):
        raise FlareParseError(
            f"{location.logical_path}:{location.line_number}: negative active frame"
        )
    return tuple(sorted(set(parsed)))


def _build_animation_definition(
    logical_path: str,
    events: Sequence[_Event],
    includes: Sequence[IncludeDirective],
    source_documents: Sequence[str],
    *,
    physical: _PhysicalFile | None,
    image_sizes: Mapping[str, tuple[int, int]],
) -> AnimationDefinition:
    image_bindings = tuple(
        _parse_image_binding(event)
        for event in events
        if not event.section and event.key == "image"
    )
    render_size: Point | None = None
    render_offset = Point(0, 0)
    blend_mode = "normal"
    alpha_mod = 255
    color_mod = (255, 255, 255)
    unknown: list[tuple[str, str, SourceLocation]] = []

    section_events: dict[str, list[_Event]] = defaultdict(list)
    section_order: list[str] = []
    for event in events:
        if not event.section:
            if event.key == "image":
                continue
            if event.key == "render_size":
                render_size = _parse_int_pair(
                    event.value, location=event.location, field="render_size"
                )
            elif event.key == "render_offset":
                render_offset = _parse_int_pair(
                    event.value, location=event.location, field="render_offset"
                )
            elif event.key == "blend_mode":
                if event.value not in {"normal", "add"}:
                    raise FlareParseError(
                        f"{event.location.logical_path}:{event.location.line_number}: "
                        f"unknown blend mode {event.value!r}"
                    )
                blend_mode = event.value
            elif event.key == "alpha_mod":
                alpha_mod = _parse_int(event.value, location=event.location, field="alpha_mod")
                if not 0 <= alpha_mod <= 255:
                    raise FlareParseError(
                        f"{event.location.logical_path}:{event.location.line_number}: "
                        "alpha_mod is outside 0..255"
                    )
            elif event.key == "color_mod":
                color_mod = _parse_color(event.value, location=event.location)
            else:
                unknown.append((event.key, event.value, event.location))
            continue
        if event.section not in section_events:
            section_order.append(event.section)
        section_events[event.section].append(event)

    actions: list[AnimationAction] = []
    for section in section_order:
        action_events = section_events[section]
        if not action_events:
            continue
        values: dict[str, _Event] = {}
        raw_frame_events: list[_Event] = []
        for event in action_events:
            if event.key == "frame":
                raw_frame_events.append(event)
            elif event.key in {
                "position",
                "frames",
                "duration",
                "type",
                "active_frame",
                "active_sub_frame",
                "image",
            }:
                values[event.key] = event
            else:
                unknown.append((f"{section}.{event.key}", event.value, event.location))
        required = {"frames", "duration", "type"}
        missing = required.difference(values)
        if missing:
            location = action_events[0].location
            raise FlareParseError(
                f"{location.logical_path}:{location.line_number}: section [{section}] "
                f"is missing {', '.join(sorted(missing))}"
            )
        frames_event = values["frames"]
        declared_frames = _parse_int(
            frames_event.value, location=frames_event.location, field="frames"
        )
        if declared_frames <= 0:
            raise FlareParseError(
                f"{frames_event.location.logical_path}:{frames_event.location.line_number}: "
                "frames must be positive"
            )
        duration_event = values["duration"]
        duration_literal = duration_event.value
        duration_ms = parse_duration_milliseconds(duration_literal)
        if duration_ms <= 0:
            raise FlareParseError(
                f"{duration_event.location.logical_path}:{duration_event.location.line_number}: "
                "animation duration must be positive"
            )
        type_event = values["type"]
        animation_type = type_event.value
        if animation_type not in _ANIMATION_TYPES:
            raise FlareParseError(
                f"{type_event.location.logical_path}:{type_event.location.line_number}: "
                f"unknown animation type {animation_type!r}"
            )
        position = None
        if "position" in values:
            position_event = values["position"]
            position = _parse_int(
                position_event.value, location=position_event.location, field="position"
            )
            if position < 0:
                raise FlareParseError(
                    f"{position_event.location.logical_path}:"
                    f"{position_event.location.line_number}: "
                    "position must be non-negative"
                )
        active_frames: tuple[int, ...] | Literal["all"] | None = None
        if "active_frame" in values:
            event = values["active_frame"]
            active_frames = _parse_active_frames(event.value, location=event.location)
        active_sub_frame = None
        if "active_sub_frame" in values:
            event = values["active_sub_frame"]
            if event.value not in {"start", "end", "all"}:
                raise FlareParseError(
                    f"{event.location.logical_path}:{event.location.line_number}: "
                    f"unknown active_sub_frame {event.value!r}"
                )
            active_sub_frame = event.value

        raw_frames = tuple(
            _parse_frame(
                event,
                image_bindings=image_bindings,
                image_sizes=image_sizes,
            )
            for event in raw_frame_events
        )
        if raw_frames:
            layout_mode = "compressed_explicit_rectangles"
            tracks = _frame_tracks(declared_frames, raw_frames)
        else:
            action_image_id = values["image"].value if "image" in values else ""
            tracks = _uncompressed_tracks(
                declared_frames,
                position,
                render_size,
                render_offset,
                image_bindings,
                image_sizes,
                action_events[0].location,
                action_image_id,
            )
            if position is not None and render_size is not None and image_bindings:
                layout_mode = "uncompressed_declared_grid"
            else:
                layout_mode = "uncompressed_missing_geometry"

        loop_mode = {
            "play_once": "one_shot_hold_last",
            "looped": "loop",
            "back_forth": "ping_pong_loop",
        }[animation_type]
        normalized = _ACTION_MAP.get(section)
        actions.append(
            AnimationAction(
                source_action=section,
                normalized_action=normalized,
                normalized_action_basis=(
                    "exact_flare_state_mapping"
                    if normalized is not None
                    else "unmapped_source_state"
                ),
                declared_frame_count=declared_frames,
                duration_literal=duration_literal,
                duration_milliseconds=duration_ms,
                nominal_fps=(declared_frames * 1000 / duration_ms),
                animation_type=animation_type,
                loop_mode=loop_mode,
                position=position,
                active_frames=active_frames,
                active_sub_frame=active_sub_frame,
                section_location=action_events[0].location,
                layout_mode=layout_mode,
                raw_frames=raw_frames,
                direction_tracks=tracks,
                default_tick_schedule=engine_tick_schedule(declared_frames, duration_literal),
            )
        )

    family, identity, body_variant, attachment_id = _definition_identity(logical_path)
    return AnimationDefinition(
        logical_path=logical_path,
        member_path=physical.member_path if physical else None,
        source_mod=physical.source_mod if physical else None,
        sha256=physical.sha256 if physical else None,
        size_bytes=len(physical.payload) if physical else None,
        includes=tuple(includes),
        source_documents=tuple(dict.fromkeys(source_documents)),
        image_bindings=image_bindings,
        render_size=render_size,
        render_offset=render_offset,
        blend_mode=blend_mode,
        alpha_mod=alpha_mod,
        color_mod=color_mod,
        entity_family=family,
        identity=identity,
        body_variant=body_variant,
        attachment_id=attachment_id,
        actions=tuple(actions),
        unknown_keys=tuple(unknown),
    )


def parse_animation_definition(
    payload: str | bytes,
    *,
    source_path: str = "animations/fixture.txt",
    image_sizes: Mapping[str, tuple[int, int]] | None = None,
) -> AnimationDefinition:
    """Parse one physical animation file without resolving its ``INCLUDE`` paths.

    Includes are retained as evidence.  Use :func:`resolve_animation_definition`
    when the effective inherited animation is required.
    """

    logical_path = _normalize_path(source_path)
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if not isinstance(raw, bytes):
        raise TypeError("payload must be str or bytes")
    physical = _PhysicalFile(logical_path, logical_path, "payload", raw, _sha256_bytes(raw))
    direct, includes = _direct_events(
        _decode_text(raw, logical_path),
        logical_path=logical_path,
        member_path=logical_path,
    )
    events = tuple(record for record in direct if isinstance(record, _Event))
    return _build_animation_definition(
        logical_path,
        events,
        includes,
        (logical_path,),
        physical=physical,
        image_sizes=image_sizes or {},
    )


def resolve_animation_definition(
    logical_path: str,
    files: Mapping[str, str | bytes],
    *,
    image_sizes: Mapping[str, tuple[int, int]] | None = None,
) -> AnimationDefinition:
    """Resolve one animation against an in-memory logical-path source mapping."""

    path = _normalize_path(logical_path)
    resolved = _mapping_resolver(files)
    events, includes, documents = _expand_events(path, resolved)
    return _build_animation_definition(
        path,
        events,
        includes,
        documents,
        physical=resolved[path],
        image_sizes=image_sizes or {},
    )


def _repository_commit_from_root(root: str) -> str | None:
    prefix = "flare-game-"
    if not root.startswith(prefix):
        return None
    commit = root[len(prefix) :]
    return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None


def _validate_archive_members(
    infos: Sequence[ZipInfo],
) -> tuple[str, dict[str, ZipInfo], tuple[SymlinkEvidence, ...]]:
    if not infos:
        raise FlareArchiveError("archive is empty")
    roots: set[str] = set()
    relative: dict[str, ZipInfo] = {}
    symlinks: list[SymlinkEvidence] = []
    for info in infos:
        name = info.filename.replace("\\", "/")
        parsed = PurePosixPath(name)
        if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
            raise FlareArchiveError(f"unsafe ZIP member path: {info.filename!r}")
        roots.add(parsed.parts[0])
        if info.flag_bits & 0x1:
            raise FlareArchiveError(f"encrypted ZIP member is unsupported: {name}")
    if len(roots) != 1:
        raise FlareArchiveError("archive must contain exactly one repository root")
    root = next(iter(roots))
    prefix = f"{root}/"
    for info in infos:
        name = info.filename.replace("\\", "/")
        if name in (root, prefix):
            continue
        if not name.startswith(prefix):
            raise FlareArchiveError(f"member escapes repository root: {name}")
        path = name[len(prefix) :].rstrip("/")
        if path in relative:
            raise FlareArchiveError(f"duplicate ZIP member: {path}")
        relative[path] = info
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if kind == stat.S_IFLNK:
            symlinks.append(
                SymlinkEvidence(
                    relative_path=path,
                    member_path=name,
                    target="",  # populated without extraction by the caller
                    unix_mode=oct(mode),
                )
            )
        elif not info.is_dir() and kind not in {0, stat.S_IFREG}:
            raise FlareArchiveError(f"unsupported special ZIP member: {path} ({oct(mode)})")
    return root, relative, tuple(symlinks)


def _is_append_file(payload: bytes, logical_path: str) -> bool:
    text = _decode_text(payload, logical_path)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        return line == "APPEND"
    return False


def _active_resource_files(
    archive: ZipFile,
    root: str,
    relative_infos: Mapping[str, ZipInfo],
) -> tuple[
    dict[str, tuple[_PhysicalFile, ...]],
    dict[str, tuple[tuple[str, ZipInfo], ...]],
]:
    text_groups: defaultdict[str, list[_PhysicalFile]] = defaultdict(list)
    resource_groups: defaultdict[str, list[tuple[str, ZipInfo]]] = defaultdict(list)
    mod_rank = {name: index for index, name in enumerate(FLARE_ACTIVE_MODS)}
    for relative_path, info in relative_infos.items():
        if info.is_dir():
            continue
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_IFMT(mode) == stat.S_IFLNK:
            continue
        parts = PurePosixPath(relative_path).parts
        if len(parts) < 3 or parts[0] != "mods" or parts[1] not in mod_rank:
            continue
        source_mod = parts[1]
        logical_path = "/".join(parts[2:])
        resource_groups[logical_path].append((source_mod, info))
        if logical_path.endswith(".txt"):
            payload = archive.read(info)
            text_groups[logical_path].append(
                _PhysicalFile(
                    logical_path=logical_path,
                    member_path=f"{root}/{relative_path}",
                    source_mod=source_mod,
                    payload=payload,
                    sha256=_sha256_bytes(payload),
                )
            )

    selected_text: dict[str, tuple[_PhysicalFile, ...]] = {}
    for logical_path, candidates in text_groups.items():
        ordered = sorted(candidates, key=lambda item: mod_rank[item.source_mod])
        start = 0
        for index in range(len(ordered) - 1, -1, -1):
            if not _is_append_file(ordered[index].payload, logical_path):
                start = index
                break
        selected_text[logical_path] = tuple(ordered[start:])
    ordered_resources = {
        path: tuple(sorted(items, key=lambda item: mod_rank[item[0]]))
        for path, items in resource_groups.items()
    }
    return selected_text, ordered_resources


def _image_has_transparency(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA", "PA"} or (
        image.mode == "P" and "transparency" in image.info
    )


def _inspect_pngs(
    archive: ZipFile,
    root: str,
    relative_infos: Mapping[str, ZipInfo],
) -> tuple[tuple[_PngInfo, ...], PngMetadataSummary]:
    pngs: list[_PngInfo] = []
    unreadable: list[str] = []
    attribution_fields: list[tuple[str, str]] = []
    source_file_fields: list[tuple[str, str]] = []
    metadata_field_counts: Counter[str] = Counter()
    with_text = 0
    with_comment = 0
    gimp_comments = 0
    with_source_file = 0
    with_attribution = 0
    with_software = 0
    for relative_path, info in sorted(relative_infos.items()):
        if info.is_dir() or not relative_path.casefold().endswith(".png"):
            continue
        member_path = f"{root}/{relative_path}"
        payload = archive.read(info)
        try:
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
                mode = image.mode
                image_format = image.format
                has_transparency = _image_has_transparency(image)
                metadata_field_counts.update(str(key) for key in image.info)
                metadata = tuple(
                    sorted(
                        (str(key), str(value)[:512])
                        for key, value in image.info.items()
                        if isinstance(value, (str, int, float))
                    )
                )
        except (OSError, UnidentifiedImageError, ValueError):
            unreadable.append(member_path)
            continue
        textual = tuple((key, value) for key, value in metadata if value)
        if textual:
            with_text += 1
        comments = tuple(value for key, value in textual if key.casefold() == "comment")
        if comments:
            with_comment += 1
            if any("created with gimp" in value.casefold() for value in comments):
                gimp_comments += 1
        source_files = tuple(value for key, value in textual if key.casefold() == "file")
        if source_files:
            with_source_file += 1
            source_file_fields.extend((member_path, value) for value in source_files)
        attribution_keys = tuple(
            key
            for key, _ in textual
            if any(term in key.casefold() for term in _ATTRIBUTION_METADATA_TERMS)
        )
        if attribution_keys:
            with_attribution += 1
            attribution_fields.extend((member_path, key) for key in attribution_keys)
        if any(
            term in key.casefold() for key, _ in textual for term in ("software", "tool", "program")
        ):
            with_software += 1
        parts = PurePosixPath(relative_path).parts
        logical_path = None
        source_mod = None
        if len(parts) >= 3 and parts[0] == "mods" and parts[1] in FLARE_ACTIVE_MODS:
            source_mod = parts[1]
            logical_path = "/".join(parts[2:])
        pngs.append(
            _PngInfo(
                member_path=member_path,
                relative_path=relative_path,
                logical_path=logical_path,
                source_mod=source_mod,
                width=width,
                height=height,
                mode=mode,
                image_format=image_format,
                has_transparency=has_transparency,
                metadata=metadata,
                size_bytes=info.file_size,
                compressed_size_bytes=info.compress_size,
                crc32=f"{info.CRC:08x}",
                sha256=_sha256_bytes(payload),
            )
        )
    summary = PngMetadataSummary(
        png_file_count=sum(
            not info.is_dir() and path.casefold().endswith(".png")
            for path, info in relative_infos.items()
        ),
        readable_png_count=len(pngs),
        metadata_field_counts=tuple(sorted(metadata_field_counts.items())),
        png_with_text_count=with_text,
        png_with_comment_count=with_comment,
        gimp_comment_count=gimp_comments,
        png_with_source_file_field_count=with_source_file,
        png_with_attribution_field_count=with_attribution,
        png_with_software_field_count=with_software,
        unreadable_member_paths=tuple(unreadable),
        attribution_fields=tuple(attribution_fields),
        source_file_fields=tuple(source_file_fields),
    )
    return tuple(pngs), summary


def _effective_image_paths(bindings: Sequence[ImageBinding]) -> tuple[str, ...]:
    effective: dict[str, str] = {}
    for binding in bindings:
        effective[binding.image_id] = binding.logical_path
    return tuple(dict.fromkeys(effective.values()))


def _parse_bool(value: str, *, location: SourceLocation, field: str) -> bool:
    normalized = value.casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise FlareParseError(
        f"{location.logical_path}:{location.line_number}: {field} must be true or false"
    )


def _entity_bindings(
    text_files: Mapping[str, Sequence[_PhysicalFile]],
) -> tuple[EntityBinding, ...]:
    entities: list[EntityBinding] = []
    for logical_path in sorted(text_files):
        if not logical_path.startswith(("enemies/", "npcs/")):
            continue
        events, _, _ = _expand_events(logical_path, text_files)
        root_events = [event for event in events if not event.section]
        animation_event = next(
            (event for event in reversed(root_events) if event.key == "animations"), None
        )
        if animation_event is None:
            continue
        name_event = next((event for event in reversed(root_events) if event.key == "name"), None)
        humanoid_event = next(
            (event for event in reversed(root_events) if event.key == "humanoid"), None
        )
        categories_event = next(
            (event for event in reversed(root_events) if event.key == "categories"), None
        )
        selected = text_files[logical_path][-1]
        parts = PurePosixPath(logical_path).parts
        template_tokens = {"base", "docs", "xp_scaling"}
        entities.append(
            EntityBinding(
                entity_kind="enemy" if logical_path.startswith("enemies/") else "npc",
                definition_path=logical_path,
                member_path=selected.member_path,
                source_mod=selected.source_mod,
                display_name=name_event.value if name_event else None,
                humanoid=(
                    _parse_bool(
                        humanoid_event.value,
                        location=humanoid_event.location,
                        field="humanoid",
                    )
                    if humanoid_event
                    else None
                ),
                categories=(
                    tuple(part for part in _split_csv(categories_event.value) if part)
                    if categories_event
                    else ()
                ),
                animation_paths=(_normalize_path(animation_event.value),),
                animation_locations=(animation_event.location,),
                is_template=bool(template_tokens.intersection(parts)),
            )
        )
    return tuple(entities)


def _section_records(events: Sequence[_Event], section: str) -> tuple[tuple[_Event, ...], ...]:
    records: list[list[_Event]] = []
    current: list[_Event] | None = None
    for event in events:
        if event.section != section:
            continue
        if event.key == "id" or current is None:
            current = []
            records.append(current)
        current.append(event)
    return tuple(tuple(record) for record in records if record)


def _last_event(record: Sequence[_Event], key: str) -> _Event | None:
    return next((event for event in reversed(record) if event.key == key), None)


def _item_bindings_and_usages(
    text_files: Mapping[str, Sequence[_PhysicalFile]],
    definition_paths: frozenset[str],
) -> tuple[tuple[AttachmentBinding, ...], tuple[AnimationUsage, ...]]:
    if "items/items.txt" not in text_files:
        return (), ()
    events, _, _ = _expand_events("items/items.txt", text_files)
    attachments: list[AttachmentBinding] = []
    usages: list[AnimationUsage] = []
    for record in _section_records(events, "item"):
        id_event = _last_event(record, "id")
        if id_event is None:
            continue
        name_event = _last_event(record, "name")
        item_type_event = _last_event(record, "item_type")
        gfx_event = _last_event(record, "gfx")
        if item_type_event and gfx_event:
            gfx_id = gfx_event.value
            candidates = tuple(
                sorted(
                    path
                    for path in definition_paths
                    if path.startswith("animations/avatar/") and PurePosixPath(path).stem == gfx_id
                )
            )
            attachments.append(
                AttachmentBinding(
                    item_id=id_event.value,
                    item_name=name_event.value if name_event else None,
                    layer_slot=item_type_event.value,
                    gfx_id=gfx_id,
                    candidate_animation_paths=candidates,
                    location=gfx_event.location,
                )
            )
        for event in record:
            if event.key != "loot_animation":
                continue
            animation_path = _normalize_path(_split_csv(event.value)[0])
            usages.append(
                AnimationUsage(
                    usage_kind="item_loot",
                    owner_id=id_event.value,
                    owner_name=name_event.value if name_event else None,
                    animation_path=animation_path,
                    location=event.location,
                )
            )
    return tuple(attachments), tuple(usages)


def _record_animation_usages(
    logical_path: str,
    section: str,
    usage_kind: str,
    text_files: Mapping[str, Sequence[_PhysicalFile]],
) -> tuple[AnimationUsage, ...]:
    if logical_path not in text_files:
        return ()
    events, _, _ = _expand_events(logical_path, text_files)
    usages: list[AnimationUsage] = []
    for record in _section_records(events, section):
        id_event = _last_event(record, "id")
        name_event = _last_event(record, "name")
        for event in record:
            if event.key != "animation":
                continue
            usages.append(
                AnimationUsage(
                    usage_kind=usage_kind,
                    owner_id=id_event.value if id_event else None,
                    owner_name=name_event.value if name_event else None,
                    animation_path=_normalize_path(event.value),
                    location=event.location,
                )
            )
    return tuple(usages)


def _hero_layers(
    text_files: Mapping[str, Sequence[_PhysicalFile]],
) -> tuple[HeroLayerOrder, ...]:
    logical_path = "engine/hero_layers.txt"
    if logical_path not in text_files:
        return ()
    events, _, _ = _expand_events(logical_path, text_files)
    layers: list[HeroLayerOrder] = []
    for event in events:
        if event.key != "layer":
            continue
        parts = _split_csv(event.value)
        if len(parts) < 2:
            raise FlareParseError(
                f"{event.location.logical_path}:{event.location.line_number}: layer is incomplete"
            )
        direction = direction_index(parts[0])
        layers.append(
            HeroLayerOrder(
                direction=direction,
                direction_token=FLARE_DIRECTION_TOKENS[direction],
                direction_name=FLARE_DIRECTION_NAMES[direction],
                layers_back_to_front=tuple(parts[1:]),
                location=event.location,
            )
        )
    return tuple(sorted(layers, key=lambda item: item.direction))


def _license_identifiers(relative_path: str, payload: bytes) -> tuple[str, ...]:
    text = payload.decode("utf-8-sig", errors="replace").casefold()
    identifiers: list[str] = []
    if "attribution-sharealike 3.0" in text or "cc-by-sa 3.0" in text:
        identifiers.append("CC-BY-SA-3.0")
    if "cc-by-sa 3.0 or later" in text or "later versions are permitted" in text:
        identifiers.append("CC-BY-SA-3.0-or-later")
    if "gpl version 3 or later" in text or "gpl-3.0-or-later" in text:
        identifiers.append("GPL-3.0-or-later")
    if "sil open font license, version 1.1" in text:
        identifiers.append("OFL-1.1")
    if "<metadata_license>cc0-1.0</metadata_license>" in text:
        identifiers.append("CC0-1.0")
    return tuple(dict.fromkeys(identifiers))


def _evidence_documents(
    archive: ZipFile,
    root: str,
    relative_infos: Mapping[str, ZipInfo],
) -> tuple[EvidenceDocument, ...]:
    scopes: Mapping[str, tuple[str, str]] = {
        "LICENSE.txt": (
            "repository_art_and_data",
            "Full CC BY-SA 3.0 Unported legal text; no per-file authorship mapping.",
        ),
        "README": (
            "repository_project_with_named_font_exceptions",
            "Art/data CC BY-SA 3.0 with later versions permitted; names four OFL font families.",
        ),
        "CREDITS.txt": (
            "repository_contributor_categories",
            "General contributors and external-art acknowledgements; links to mutable "
            "online credits.",
        ),
        "CONTRIBUTING.md": (
            "repository_contributions",
            "Contribution terms say art/data are CC BY-SA 3.0 or later unless otherwise noted.",
        ),
        "distribution/org.flarerpg.Flare.appdata.xml": (
            "distribution_metadata",
            "Separates engine GPL, campaign CC BY-SA, and AppStream metadata CC0 claims.",
        ),
        "mods/fantasycore/cutscenes/credits.txt": (
            "mod_fantasycore_credit_aggregator",
            "Runtime include list, not a per-asset author map.",
        ),
        "mods/fantasycore/cutscenes/credits_fantasycore.txt": (
            "mod_fantasycore_contributor_categories",
            "Names visual artists and other contributors at mod scope only.",
        ),
        "mods/empyrean_campaign/cutscenes/credits.txt": (
            "mod_empyrean_campaign_credit_aggregator",
            "Runtime include list, not a per-asset author map.",
        ),
        "mods/empyrean_campaign/cutscenes/credits_empyrean.txt": (
            "mod_empyrean_campaign_contributor_categories",
            "Names visual artists and other contributors at mod scope only.",
        ),
    }
    documents: list[EvidenceDocument] = []
    for relative_path in _EVIDENCE_PATHS:
        info = relative_infos.get(relative_path)
        if info is None or info.is_dir():
            continue
        payload = archive.read(info)
        scope, notes = scopes[relative_path]
        documents.append(
            EvidenceDocument(
                relative_path=relative_path,
                member_path=f"{root}/{relative_path}",
                sha256=_sha256_bytes(payload),
                size_bytes=len(payload),
                detected_license_identifiers=_license_identifiers(relative_path, payload),
                scope=scope,
                notes=notes,
            )
        )
    return tuple(documents)


def audit_flare_archive(archive_path: str | Path) -> FlareArchiveAudit:
    """Audit a structurally compatible Flare game ZIP without extracting it."""

    path = Path(archive_path)
    if not path.is_file():
        raise FlareArchiveError(f"archive does not exist: {path}")
    archive_sha256 = _sha256_file(path)
    issues: list[AuditIssue] = []
    try:
        archive = ZipFile(path)
    except BadZipFile as exc:
        raise FlareArchiveError(f"not a readable ZIP archive: {path}") from exc
    with archive:
        infos = archive.infolist()
        root, relative_infos, raw_symlinks = _validate_archive_members(infos)
        symlinks: list[SymlinkEvidence] = []
        for link in raw_symlinks:
            info = relative_infos[link.relative_path]
            try:
                target = archive.read(info).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise FlareArchiveError(
                    f"symlink target is not UTF-8: {link.relative_path}"
                ) from exc
            symlinks.append(replace(link, target=target))

        text_files, resource_files = _active_resource_files(archive, root, relative_infos)
        png_infos, png_metadata = _inspect_pngs(archive, root, relative_infos)
        png_by_member = {info.member_path: info for info in png_infos}
        selected_pngs: dict[str, _PngInfo] = {}
        for logical_path, candidates in resource_files.items():
            source_mod, info = candidates[-1]
            selected = png_by_member.get(info.filename.replace("\\", "/"))
            if selected is not None:
                # Keep the ModManager-selected source even when selected metadata
                # was derived independently from the path.
                selected_pngs[logical_path] = replace(selected, source_mod=source_mod)
        image_sizes = {
            logical_path: (png.width, png.height) for logical_path, png in selected_pngs.items()
        }

        definitions: list[AnimationDefinition] = []
        physical_actions: list[AnimationAction] = []
        definition_paths = sorted(
            logical_path
            for logical_path in text_files
            if logical_path.startswith("animations/") and logical_path.endswith(".txt")
        )
        for logical_path in definition_paths:
            try:
                physical = text_files[logical_path][-1]
                direct_records, direct_includes = _direct_events(
                    _decode_text(physical.payload, logical_path),
                    logical_path=logical_path,
                    member_path=physical.member_path,
                )
                direct_definition = _build_animation_definition(
                    logical_path,
                    tuple(record for record in direct_records if isinstance(record, _Event)),
                    direct_includes,
                    (physical.member_path,),
                    physical=physical,
                    image_sizes=image_sizes,
                )
                physical_actions.extend(direct_definition.actions)
                events, includes, documents = _expand_events(logical_path, text_files)
                definition = _build_animation_definition(
                    logical_path,
                    events,
                    includes,
                    documents,
                    physical=text_files[logical_path][-1],
                    image_sizes=image_sizes,
                )
            except FlareParseError as exc:
                issues.append(
                    AuditIssue(
                        severity="error",
                        code="animation_parse_error",
                        message=str(exc),
                        related_paths=(logical_path,),
                    )
                )
                continue
            definitions.append(definition)

        definition_path_set = frozenset(item.logical_path for item in definitions)
        parent_definition = next(
            (item for item in definitions if item.logical_path == "animations/hero.txt"),
            None,
        )
        parent_frame_counts = (
            {
                action.source_action: action.declared_frame_count
                for action in parent_definition.actions
            }
            if parent_definition
            else {}
        )
        attachment_parent_mismatches: list[str] = []
        for definition in definitions:
            if definition.entity_family != "avatar_attachment":
                continue
            frame_counts = {
                action.source_action: action.declared_frame_count for action in definition.actions
            }
            if frame_counts != parent_frame_counts:
                attachment_parent_mismatches.append(definition.logical_path)
        if attachment_parent_mismatches:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="attachment_parent_timeline_mismatch",
                    message=(
                        "Avatar attachment action/frame counts differ from animations/hero.txt; "
                        "the engine would coerce declared frame counts when setting the parent."
                    ),
                    related_paths=tuple(sorted(attachment_parent_mismatches)),
                )
            )
        image_reference_counts: Counter[str] = Counter()
        for definition in definitions:
            for image_path in set(_effective_image_paths(definition.image_bindings)):
                image_reference_counts[image_path] += 1
        missing_images = tuple(
            sorted(path for path in image_reference_counts if path not in selected_pngs)
        )
        if missing_images:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="missing_animation_images",
                    message="Animation image paths do not resolve in the active mod stack.",
                    related_paths=missing_images,
                )
            )
        source_images = tuple(
            SourceImageAudit(
                logical_path=logical_path,
                member_path=selected_pngs[logical_path].member_path,
                source_mod=selected_pngs[logical_path].source_mod or "",
                width=selected_pngs[logical_path].width,
                height=selected_pngs[logical_path].height,
                image_mode=selected_pngs[logical_path].mode,
                image_format=selected_pngs[logical_path].image_format,
                has_transparency=selected_pngs[logical_path].has_transparency,
                size_bytes=selected_pngs[logical_path].size_bytes,
                compressed_size_bytes=selected_pngs[logical_path].compressed_size_bytes,
                crc32=selected_pngs[logical_path].crc32,
                sha256=selected_pngs[logical_path].sha256,
                definition_reference_count=image_reference_counts[logical_path],
            )
            for logical_path in sorted(image_reference_counts)
            if logical_path in selected_pngs
        )

        entities = _entity_bindings(text_files)
        entity_usages: list[AnimationUsage] = []
        for entity in entities:
            for index, animation_path in enumerate(entity.animation_paths):
                location = entity.animation_locations[
                    min(index, len(entity.animation_locations) - 1)
                ]
                entity_usages.append(
                    AnimationUsage(
                        usage_kind=entity.entity_kind,
                        owner_id=entity.definition_path,
                        owner_name=entity.display_name,
                        animation_path=animation_path,
                        location=location,
                    )
                )
        attachments, item_usages = _item_bindings_and_usages(text_files, definition_path_set)
        power_usages = _record_animation_usages("powers/powers.txt", "power", "power", text_files)
        effect_usages = _record_animation_usages(
            "powers/effects.txt", "effect", "effect", text_files
        )
        usages = tuple((*entity_usages, *item_usages, *power_usages, *effect_usages))
        hero_layers = _hero_layers(text_files)
        evidence_documents = _evidence_documents(archive, root, relative_infos)

        unresolved_usage_paths = tuple(
            sorted(
                {
                    usage.animation_path
                    for usage in usages
                    if usage.animation_path not in definition_path_set
                }
            )
        )
        if unresolved_usage_paths:
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="usage_outside_snapshot_animation_set",
                    message=(
                        "Usages reference animation definitions not supplied by the two game mods; "
                        "some may belong to the separately distributed engine default mod."
                    ),
                    related_paths=unresolved_usage_paths,
                )
            )
        empty_attachment_candidates = tuple(
            sorted({item.gfx_id for item in attachments if not item.candidate_animation_paths})
        )
        if empty_attachment_candidates:
            issues.append(
                AuditIssue(
                    severity="warning",
                    code="attachment_without_archived_body_variant",
                    message="Item gfx identifiers have no matching archived avatar definition.",
                    related_paths=empty_attachment_candidates,
                )
            )
        geometry_missing_paths = tuple(
            sorted(
                {
                    definition.logical_path
                    for definition in definitions
                    for action in definition.actions
                    if not action.has_exact_geometry
                }
            )
        )
        if geometry_missing_paths:
            issues.append(
                AuditIssue(
                    severity="info",
                    code="timeline_without_sheet_geometry",
                    message=(
                        "Timeline definitions without both an image and declared geometry are "
                        "retained but never projected as guessed grids."
                    ),
                    related_paths=geometry_missing_paths,
                )
            )
        out_of_bounds_paths = tuple(
            sorted(
                {
                    frame.location.logical_path
                    for definition in definitions
                    for action in definition.actions
                    for frame in action.raw_frames
                    if frame.within_image_bounds is False
                }
            )
        )
        if out_of_bounds_paths:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="frame_rectangle_out_of_bounds",
                    message="Explicit frame rectangles exceed their selected PNG bounds.",
                    related_paths=out_of_bounds_paths,
                )
            )
        if png_metadata.unreadable_member_paths:
            issues.append(
                AuditIssue(
                    severity="error",
                    code="unreadable_png",
                    message="One or more PNG members failed Pillow verification.",
                    related_paths=png_metadata.unreadable_member_paths,
                )
            )
        issues.append(
            AuditIssue(
                severity="info",
                code="no_per_asset_credit_manifest",
                message=(
                    "The snapshot has repository- and mod-scoped credit evidence but no immutable "
                    "per-file artist/license manifest; the linked wiki is external and mutable."
                ),
                related_paths=("CREDITS.txt",),
            )
        )

        actions = [action for definition in definitions for action in definition.actions]
        tracks = [track for action in actions for track in action.direction_tracks]
        slots = [slot for track in tracks for slot in track.frames]
        raw_frames = [frame for action in actions for frame in action.raw_frames]
        regular_infos = [
            info
            for info in infos
            if not info.is_dir()
            and stat.S_IFMT((info.external_attr >> 16) & 0xFFFF) != stat.S_IFLNK
        ]
        active_mod_png_count = sum(png.logical_path is not None for png in png_infos)
        counts = FlareArchiveCounts(
            zip_member_count=len(infos),
            regular_file_member_count=len(regular_infos),
            directory_member_count=sum(info.is_dir() for info in infos),
            symlink_member_count=len(symlinks),
            expanded_member_bytes=sum(info.file_size for info in infos if not info.is_dir()),
            expanded_regular_file_bytes=sum(info.file_size for info in regular_infos),
            archive_png_file_count=png_metadata.png_file_count,
            active_mod_png_file_count=active_mod_png_count,
            animation_definition_file_count=len(definition_paths),
            fantasycore_animation_definition_count=sum(
                definition.source_mod == "fantasycore" for definition in definitions
            ),
            empyrean_animation_definition_count=sum(
                definition.source_mod == "empyrean_campaign" for definition in definitions
            ),
            included_animation_definition_count=sum(bool(item.includes) for item in definitions),
            physical_action_declaration_count=len(physical_actions),
            physical_explicit_frame_record_count=sum(
                len(action.raw_frames) for action in physical_actions
            ),
            action_count=len(actions),
            exact_geometry_action_count=sum(action.has_exact_geometry for action in actions),
            geometry_missing_action_count=sum(not action.has_exact_geometry for action in actions),
            direction_track_count=len(tracks),
            explicit_direction_track_count=sum(track.explicit_frame_count > 0 for track in tracks),
            fallback_only_direction_track_count=sum(
                track.explicit_frame_count == 0 and track.fallback_frame_count > 0
                for track in tracks
            ),
            unresolved_direction_track_count=sum(not track.complete for track in tracks),
            explicit_frame_record_count=len(raw_frames),
            effective_frame_slot_count=len(slots),
            explicit_frame_slot_count=sum(slot.explicit for slot in slots),
            direction_zero_fallback_slot_count=sum(
                slot.fallback_from_direction is not None for slot in slots
            ),
            unresolved_frame_slot_count=sum(slot.frame is None for slot in slots),
            complete_eight_direction_action_count=sum(
                all(track.complete for track in action.direction_tracks) for action in actions
            ),
            play_once_action_count=sum(action.animation_type == "play_once" for action in actions),
            looped_action_count=sum(action.animation_type == "looped" for action in actions),
            back_forth_action_count=sum(
                action.animation_type == "back_forth" for action in actions
            ),
            referenced_source_image_count=len(source_images),
            missing_source_image_count=len(missing_images),
            out_of_bounds_frame_record_count=sum(
                frame.within_image_bounds is False for frame in raw_frames
            ),
            entity_binding_count=len(entities),
            concrete_entity_binding_count=sum(not item.is_template for item in entities),
            template_entity_binding_count=sum(item.is_template for item in entities),
            enemy_binding_count=sum(item.entity_kind == "enemy" for item in entities),
            npc_binding_count=sum(item.entity_kind == "npc" for item in entities),
            explicit_humanoid_binding_count=sum(item.humanoid is True for item in entities),
            animation_usage_count=len(usages),
            attachment_binding_count=len(attachments),
            avatar_attachment_definition_count=sum(
                item.entity_family == "avatar_attachment" for item in definitions
            ),
            attachment_parent_mismatch_count=len(attachment_parent_mismatches),
            hero_layer_direction_count=len(hero_layers),
            evidence_document_count=len(evidence_documents),
        )
        action_counts = tuple(sorted(Counter(action.source_action for action in actions).items()))
        direction_counts = tuple(
            (
                FLARE_DIRECTION_NAMES[direction],
                sum(frame.direction == direction for frame in raw_frames),
            )
            for direction in range(8)
        )
        definition_family_counts = tuple(
            sorted(Counter(item.entity_family for item in definitions).items())
        )
        body_variant_counts = tuple(
            sorted(
                Counter(
                    item.body_variant for item in definitions if item.body_variant is not None
                ).items()
            )
        )
        usage_counts = tuple(sorted(Counter(item.usage_kind for item in usages).items()))

    repository_commit = _repository_commit_from_root(root)
    return FlareArchiveAudit(
        archive_path=str(path.resolve()),
        archive_sha256=archive_sha256,
        archive_size_bytes=path.stat().st_size,
        repository_commit=repository_commit,
        repository_url=FLARE_GAME_REPOSITORY_URL,
        commit_url=(
            f"{FLARE_GAME_REPOSITORY_URL}/tree/{repository_commit}" if repository_commit else None
        ),
        engine_semantics_commit=FLARE_ENGINE_COMMIT,
        engine_semantics_url=FLARE_ENGINE_COMMIT_URL,
        root_prefix=root,
        active_mods=FLARE_ACTIVE_MODS,
        counts=counts,
        definitions=tuple(definitions),
        source_images=source_images,
        entities=entities,
        usages=usages,
        attachments=attachments,
        hero_layers=hero_layers,
        evidence_documents=evidence_documents,
        symlinks=tuple(symlinks),
        png_metadata=png_metadata,
        definition_family_counts=definition_family_counts,
        body_variant_counts=body_variant_counts,
        usage_counts=usage_counts,
        action_counts=action_counts,
        direction_explicit_frame_counts=direction_counts,
        issues=tuple(issues),
    )


def audit_known_flare_archive(archive_path: str | Path) -> FlareArchiveAudit:
    """Audit and require the exact pinned Flare game CAS snapshot."""

    path = Path(archive_path)
    if not path.is_file():
        raise FlareArchiveError(f"archive does not exist: {path}")
    digest = _sha256_file(path)
    if digest != EXPECTED_FLARE_ARCHIVE_SHA256:
        raise FlareArchiveError(
            "archive SHA-256 does not match the pinned Flare snapshot: "
            f"expected {EXPECTED_FLARE_ARCHIVE_SHA256}, got {digest}"
        )
    audit = audit_flare_archive(path)
    if audit.root_prefix != _EXPECTED_ROOT or audit.repository_commit != FLARE_GAME_COMMIT:
        raise FlareArchiveError(
            f"archive root must be {_EXPECTED_ROOT!r}, got {audit.root_prefix!r}"
        )
    return audit


audit_flare_empyrean_archive = audit_flare_archive
audit_known_flare_empyrean_archive = audit_known_flare_archive
