"""Read-only audit of commit-pinned /tg/station DMI mob sprites.

Dream Maker icon (DMI) files are PNG atlases whose ``zTXt Description``
metadata declares states, directions, temporal frames, delays, movement
variants, loop counts, rewind behaviour, and hotspots.  This module parses the
literal metadata and follows the atlas ordering in /tg/station's pinned
``tools/dmi`` implementation: state order, then temporal frame, then direction,
with cells laid out row-major in the PNG.

The adapter never extracts the repository archive, executes DM/Python from the
archive, or writes to the corpus database.  Complete-entity candidates are
kept separate from clothing, body parts, overlays, effects, and UI assets.
Ambiguous or malformed records remain hash-addressed quarantine evidence.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import struct
import zlib
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

SOURCE_ID = "tgstation_dmi_mobs"
TGSTATION_COMMIT = "ca8b5ffda1ec44e0fa04a32801dc5f6e3b77f0e9"
TGSTATION_REPOSITORY_URL = "https://github.com/tgstation/tgstation"
TGSTATION_COMMIT_URL = f"{TGSTATION_REPOSITORY_URL}/tree/{TGSTATION_COMMIT}"
TGSTATION_ARCHIVE_URL = f"https://codeload.github.com/tgstation/tgstation/zip/{TGSTATION_COMMIT}"
EXPECTED_TGSTATION_ARCHIVE_SHA256 = (
    "6f37531d28b8e48ca9399daccdbeef3683e9561eca0cf2272c4bad11c5a2a07c"
)
EXPECTED_TGSTATION_ARCHIVE_BYTES = 193_871_729
EXPECTED_TGSTATION_ARCHIVE_ROOT = f"tgstation-{TGSTATION_COMMIT}"

EXPECTED_TGSTATION_INVENTORY_SHA256 = (
    "ad3e1356ccc701577b3a1c612b2487f4a12544082f55cdc27bcce5e93636d678"
)
EXPECTED_TGSTATION_AUDIT_RECORD_SHA256 = (
    "22cb2cc6bc828c082728287d2c47702834cd23667f01246ec5fc50af60fe2249"
)

TGSTATION_ARCHIVE_ETAG = '"ade05c6567faddf6417104e00879fa697ec07189d70933a903142c59c71f9de1"'
BYOND_ICON_REFERENCE_URL = "https://www.byond.com/docs/ref/info.html#/icon"

_MOB_DMI_RE = re.compile(r"^icons/mob(?:/.*)?\.dmi$", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_DIRECTION_NAMES = (
    "south",
    "north",
    "east",
    "west",
    "southeast",
    "southwest",
    "northeast",
    "northwest",
)
_MAX_ARCHIVE_MEMBERS = 25_000
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024**3
_MAX_MEMBER_BYTES = 64 * 1024**2
_MAX_DESCRIPTION_BYTES = 8 * 1024**2

PackRole = Literal[
    "complete_entity_candidate",
    "modular_component_pack",
    "effect_pack",
    "icon_or_ui_pack",
    "ambiguous_pack",
]
StateRole = Literal[
    "complete_entity_candidate",
    "modular_component",
    "effect_or_overlay",
    "icon_or_ui",
    "ambiguous",
]


class TgstationArchiveError(ValueError):
    """Raised when the ZIP is unsafe or does not match snapshot invariants."""


class DmiMetadataError(ValueError):
    """Raised when a DMI cannot be interpreted without guessing."""


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int
    purpose: str


@dataclass(frozen=True, slots=True)
class AcquisitionEvidence:
    requested_url: str
    final_url: str
    expected_sha256: str
    observed_size_bytes: int
    observed_etag: str
    retrieval_method: str


@dataclass(frozen=True, slots=True)
class PngTextEvidence:
    chunk_type: str
    keyword: str
    compression_method: int | None
    decoded_encoding: str
    decoded_size_bytes: int
    decoded_sha256: str
    raw_chunk_sha256: str
    crc32: str


@dataclass(frozen=True, slots=True)
class HotspotRecord:
    x: int
    y: int
    first_frame_one_based: int


@dataclass(frozen=True, slots=True)
class FrameCell:
    source_cell_index: int
    state_cell_index: int
    temporal_frame_index: int
    direction_index: int
    direction: str
    left: int
    top: int
    right: int
    bottom: int
    duration_milliseconds: int | None
    rgba_sha256: str


@dataclass(frozen=True, slots=True)
class DmiState:
    declaration_index: int
    name: str
    name_occurrence_index: int
    runtime_key_occurrence_index: int
    entity_cue: str
    entity_class: str
    entity_class_basis: str
    normalized_action: str | None
    normalized_action_basis: str
    role: StateRole
    role_basis: str
    direction_count: int
    direction_names: tuple[str, ...]
    temporal_frame_count: int
    delay_decisecond_literals: tuple[str, ...]
    durations_milliseconds: tuple[int, ...]
    delays_declared: bool
    loop_count: int
    rewind: bool
    movement: bool
    playback_semantics: str
    hotspots: tuple[HotspotRecord, ...]
    source_cell_start: int
    source_cell_count: int
    frames: tuple[FrameCell, ...]
    source_sequence_sha256: str
    timed_sequence_sha256: str
    is_temporally_animated: bool
    eligible_complete_entity_sequence: bool
    eligible_animated_action_sequence: bool
    quarantine_reasons: tuple[str, ...]
    selection_exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DmiPack:
    logical_path: str
    member_path: str
    blob_url: str
    history_url: str
    lineage_key: str
    asset_deduplication_key: str
    sha256: str
    size_bytes: int
    detected_format: str
    image_mode: str
    has_alpha: bool
    image_width: int
    image_height: int
    frame_width: int
    frame_height: int
    grid_columns: int
    grid_rows: int
    grid_capacity: int
    declared_source_cells: int
    unused_source_cells: int
    metadata: PngTextEvidence
    description_verbatim: str
    pack_role: PackRole
    pack_role_basis: str
    entity_class: str
    entity_class_basis: str
    states: tuple[DmiState, ...]
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MalformedDmiRecord:
    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int
    error_type: str
    error: str


@dataclass(frozen=True, slots=True)
class StateReference:
    logical_path: str
    declaration_index: int
    name: str
    name_occurrence_index: int


@dataclass(frozen=True, slots=True)
class DuplicateDmiGroup:
    sha256: str
    logical_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DuplicateStateGroup:
    timed_sequence_sha256: str
    references: tuple[StateReference, ...]


@dataclass(frozen=True, slots=True)
class EntityActionSet:
    entity_key: str
    logical_path: str
    entity_cue: str
    entity_class: str
    state_references: tuple[StateReference, ...]
    actions: tuple[str, ...]
    complete_sequence_count: int
    animated_action_sequence_count: int
    steerable: bool
    has_animated_action: bool


@dataclass(frozen=True, slots=True)
class RightsAudit:
    asset_license_expression: str
    asset_license_scope: str
    asset_license_basis: str
    code_license_expression: str
    root_license: EvidenceDocument
    historical_code_license: EvidenceDocument
    readme: EvidenceDocument
    path_local_rights_documents: tuple[EvidenceDocument, ...]
    per_file_author_manifest_present: bool
    attribution_policy: tuple[str, ...]
    caveat: str


@dataclass(frozen=True, slots=True)
class EngineSemanticsAudit:
    implementation: EvidenceDocument
    immutable_url: str
    official_reference_url: str
    semantics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AuditIssue:
    code: str
    count: int
    detail: str


@dataclass(frozen=True, slots=True)
class TgstationArchiveCounts:
    archive_members: int
    archive_files: int
    archive_directories: int
    archive_symlinks: int
    archive_compressed_bytes: int
    archive_uncompressed_bytes: int
    mob_dmi_files: int
    parsed_dmi_files: int
    malformed_dmi_files: int
    dmi_states: int
    declared_source_cells: int
    temporally_animated_states: int
    directional_states: int
    movement_states: int
    rewind_states: int
    finite_loop_states: int
    delay_declared_states: int
    delay_count_mismatch_states: int
    invalid_hotspot_states: int
    duplicate_runtime_key_excess: int
    exact_capacity_dmis: int
    surplus_capacity_dmis: int
    unused_source_cells: int
    complete_entity_candidate_states: int
    eligible_complete_entity_sequences: int
    eligible_action_sequences: int
    eligible_animated_action_sequences: int
    entity_action_sets: int
    steerable_entity_action_sets: int
    steerable_entity_action_sets_with_animation: int
    duplicate_dmi_hash_groups: int
    duplicate_dmi_hash_excess: int
    duplicate_complete_state_groups: int
    duplicate_complete_state_excess: int
    pack_role_counts: tuple[tuple[str, int], ...]
    state_role_counts: tuple[tuple[str, int], ...]
    entity_class_counts: tuple[tuple[str, int], ...]
    action_counts: tuple[tuple[str, int], ...]
    direction_count_counts: tuple[tuple[str, int], ...]
    loop_count_counts: tuple[tuple[str, int], ...]
    image_mode_counts: tuple[tuple[str, int], ...]
    frame_size_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class TgstationArchiveAudit:
    source_id: str
    archive_sha256: str
    archive_size_bytes: int
    inventory_sha256: str
    repository_url: str
    commit: str
    commit_url: str
    archive_url: str
    archive_root: str
    counts: TgstationArchiveCounts
    packs: tuple[DmiPack, ...]
    malformed_dmis: tuple[MalformedDmiRecord, ...]
    entity_action_sets: tuple[EntityActionSet, ...]
    duplicate_dmi_groups: tuple[DuplicateDmiGroup, ...]
    duplicate_state_groups: tuple[DuplicateStateGroup, ...]
    rights: RightsAudit
    engine_semantics: EngineSemanticsAudit
    acquisition_evidence: AcquisitionEvidence
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


@dataclass(slots=True)
class _RawState:
    declaration_index: int
    name: str
    dirs: int = 1
    frames: int = 1
    delay_literals: tuple[str, ...] = ()
    loop_count: int = 0
    rewind: bool = False
    movement: bool = False
    hotspots: tuple[HotspotRecord, ...] = ()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _counter_tuple(counter: Counter[Any]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(key), value) for key, value in counter.items()))


def _normalize_member_name(name: str) -> str:
    if not name or "\x00" in name or "\\" in name:
        raise TgstationArchiveError(f"unsafe archive member path: {name!r}")
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or re.match(r"^[A-Za-z]:", name):
        raise TgstationArchiveError(f"unsafe archive member path: {name!r}")
    normalized = pure.as_posix()
    if name.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    return normalized


def _entry_kind(info: ZipInfo) -> tuple[Literal["file", "directory", "symlink"], int]:
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if info.is_dir():
        if file_type not in (0, stat.S_IFDIR):
            raise TgstationArchiveError(f"directory has incompatible mode: {info.filename!r}")
        return "directory", mode
    if file_type == stat.S_IFLNK:
        return "symlink", mode
    if file_type in (0, stat.S_IFREG):
        return "file", mode
    raise TgstationArchiveError(f"unsupported special ZIP member: {info.filename!r}")


def _validate_archive_members(
    archive: ZipFile,
) -> tuple[str, tuple[_ArchiveEntry, ...], str]:
    infos = archive.infolist()
    if not infos:
        raise TgstationArchiveError("archive is empty")
    if len(infos) > _MAX_ARCHIVE_MEMBERS:
        raise TgstationArchiveError(f"archive has too many members: {len(infos)}")
    if sum(info.file_size for info in infos) > _MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise TgstationArchiveError("archive uncompressed size exceeds the audit limit")

    seen: set[str] = set()
    folded: dict[str, str] = {}
    roots: set[str] = set()
    entries: list[_ArchiveEntry] = []
    inventory_rows: list[dict[str, Any]] = []
    for info in infos:
        member_path = _normalize_member_name(info.filename)
        collision_key = member_path.rstrip("/")
        if collision_key in seen:
            raise TgstationArchiveError(f"duplicate archive member: {member_path!r}")
        seen.add(collision_key)
        prior = folded.get(collision_key.casefold())
        if prior is not None and prior != collision_key:
            raise TgstationArchiveError(
                f"case-colliding archive members: {prior!r}, {collision_key!r}"
            )
        folded[collision_key.casefold()] = collision_key
        if info.flag_bits & 0x1:
            raise TgstationArchiveError(f"encrypted ZIP member is not accepted: {member_path!r}")
        if info.file_size > _MAX_MEMBER_BYTES:
            raise TgstationArchiveError(f"ZIP member exceeds audit limit: {member_path!r}")
        parts = PurePosixPath(collision_key).parts
        if not parts:
            raise TgstationArchiveError(f"member has no archive root: {member_path!r}")
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
        raise TgstationArchiveError(f"archive must have one root, found {sorted(roots)!r}")
    return next(iter(roots)), tuple(entries), _canonical_hash(inventory_rows)


def _decode_ztxt_description(payload: bytes) -> tuple[str, PngTextEvidence]:
    if not payload.startswith(_PNG_SIGNATURE):
        raise DmiMetadataError("DMI payload is not a PNG")
    offset = len(_PNG_SIGNATURE)
    descriptions: list[tuple[str, PngTextEvidence]] = []
    found_iend = False
    while offset < len(payload):
        if offset + 12 > len(payload):
            raise DmiMetadataError("truncated PNG chunk header")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunk_type = payload[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(payload):
            raise DmiMetadataError("truncated PNG chunk payload")
        chunk = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", payload[offset + 8 + length : end])[0]
        observed_crc = zlib.crc32(chunk_type + chunk) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            raise DmiMetadataError(f"PNG CRC mismatch for {chunk_type!r}")
        if chunk_type == b"zTXt":
            try:
                keyword_bytes, compressed = chunk.split(b"\x00", 1)
            except ValueError as error:
                raise DmiMetadataError("zTXt chunk has no keyword separator") from error
            if not compressed:
                raise DmiMetadataError("zTXt chunk has no compression method")
            method = compressed[0]
            if method != 0:
                raise DmiMetadataError(f"unsupported zTXt compression method: {method}")
            keyword = keyword_bytes.decode("latin-1")
            decompressor = zlib.decompressobj()
            decoded_bytes = decompressor.decompress(compressed[1:], _MAX_DESCRIPTION_BYTES + 1)
            decoded_bytes += decompressor.flush()
            if len(decoded_bytes) > _MAX_DESCRIPTION_BYTES:
                raise DmiMetadataError("DMI Description exceeds decompression limit")
            if not decompressor.eof or decompressor.unused_data:
                raise DmiMetadataError("invalid zTXt compressed stream")
            if keyword == "Description":
                text = decoded_bytes.decode("latin-1")
                descriptions.append(
                    (
                        text,
                        PngTextEvidence(
                            chunk_type="zTXt",
                            keyword=keyword,
                            compression_method=method,
                            decoded_encoding="latin-1",
                            decoded_size_bytes=len(decoded_bytes),
                            decoded_sha256=_sha256_bytes(decoded_bytes),
                            raw_chunk_sha256=_sha256_bytes(chunk),
                            crc32=f"{expected_crc:08x}",
                        ),
                    )
                )
        if chunk_type == b"IEND":
            found_iend = True
            if end != len(payload):
                raise DmiMetadataError("bytes follow PNG IEND")
        offset = end
    if not found_iend:
        raise DmiMetadataError("PNG has no IEND")
    if len(descriptions) != 1:
        raise DmiMetadataError(
            f"expected exactly one zTXt Description chunk, found {len(descriptions)}"
        )
    return descriptions[0]


def _unescape_dmi_state(value: str) -> str:
    if len(value) < 2 or not value.startswith('"') or not value.endswith('"'):
        raise DmiMetadataError(f"state name is not double quoted: {value!r}")
    # This mirrors the exact two substitutions in the pinned tools/dmi parser.
    return value[1:-1].replace(r"\"", '"').replace(r"\\", "\\")


def _parse_positive_int(value: str, *, key: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise DmiMetadataError(f"{key} is not an integer: {value!r}") from error
    if parsed <= 0:
        raise DmiMetadataError(f"{key} must be positive: {value!r}")
    return parsed


def _parse_bool(value: str, *, key: str) -> bool:
    if value not in {"0", "1"}:
        raise DmiMetadataError(f"{key} must be 0 or 1: {value!r}")
    return value == "1"


def _parse_delay_milliseconds(literal: str) -> int:
    try:
        value = Decimal(literal)
    except InvalidOperation as error:
        raise DmiMetadataError(f"delay is not numeric: {literal!r}") from error
    if not value.is_finite() or value <= 0:
        raise DmiMetadataError(f"delay must be finite and positive: {literal!r}")
    milliseconds = value * 100
    if milliseconds != milliseconds.to_integral_value():
        raise DmiMetadataError(f"delay is not an integral millisecond value: {literal!r}")
    return int(milliseconds)


def _parse_description(description: str) -> tuple[int, int, tuple[_RawState, ...]]:
    lines = description.splitlines()
    if len(lines) < 3 or lines[0] != "# BEGIN DMI":
        raise DmiMetadataError("Description does not begin with '# BEGIN DMI'")
    if lines[1] != "version = 4.0":
        raise DmiMetadataError(f"unsupported DMI version header: {lines[1]!r}")
    width = 32
    height = 32
    states: list[_RawState] = []
    current: _RawState | None = None
    ended = False
    for line_number, line in enumerate(lines[2:], start=3):
        if line == "# END DMI":
            ended = True
            if any(item.strip() for item in lines[line_number:]):
                raise DmiMetadataError("nonempty lines follow '# END DMI'")
            break
        try:
            key, value = line.lstrip().split(" = ", 1)
        except ValueError as error:
            raise DmiMetadataError(f"malformed metadata line {line_number}: {line!r}") from error
        if key == "width":
            if current is not None:
                raise DmiMetadataError("width appears after the first state")
            width = _parse_positive_int(value, key="width")
        elif key == "height":
            if current is not None:
                raise DmiMetadataError("height appears after the first state")
            height = _parse_positive_int(value, key="height")
        elif key == "state":
            current = _RawState(len(states), _unescape_dmi_state(value))
            states.append(current)
        elif current is None:
            raise DmiMetadataError(f"{key!r} appears before the first state")
        elif key == "dirs":
            current.dirs = _parse_positive_int(value, key="dirs")
        elif key == "frames":
            current.frames = _parse_positive_int(value, key="frames")
        elif key == "delay":
            literals = tuple(item.strip() for item in value.split(","))
            if not literals or any(not item for item in literals):
                raise DmiMetadataError(f"empty delay entry on line {line_number}")
            for literal in literals:
                _parse_delay_milliseconds(literal)
            current.delay_literals = literals
        elif key == "loop":
            try:
                loop_count = int(value)
            except ValueError as error:
                raise DmiMetadataError(f"loop is not an integer: {value!r}") from error
            if loop_count < 0:
                raise DmiMetadataError(f"loop must be nonnegative: {value!r}")
            current.loop_count = loop_count
        elif key == "rewind":
            current.rewind = _parse_bool(value, key="rewind")
        elif key == "movement":
            current.movement = _parse_bool(value, key="movement")
        elif key == "hotspot":
            try:
                x, y, frame = (int(item.strip()) for item in value.split(","))
            except (TypeError, ValueError) as error:
                raise DmiMetadataError(f"invalid hotspot: {value!r}") from error
            current.hotspots = (*current.hotspots, HotspotRecord(x, y, frame))
        else:
            raise DmiMetadataError(f"unsupported DMI metadata key: {key!r}")
    if not ended:
        raise DmiMetadataError("Description has no '# END DMI'")
    if not states:
        raise DmiMetadataError("Description declares no states")
    return width, height, tuple(states)


def _classify_pack_role(logical_path: str) -> tuple[PackRole, str]:
    parts = PurePosixPath(logical_path).parts
    relative = parts[2:]
    top = relative[0].casefold() if len(relative) > 1 else "<root>"
    filename = parts[-1].casefold()
    mapped: dict[str, tuple[PackRole, str]] = {
        "actions": ("icon_or_ui_pack", "mob_actions_ui_subtree"),
        "augmentation": ("modular_component_pack", "augmentation_layer_subtree"),
        "clothing": ("modular_component_pack", "runtime_clothing_layer_subtree"),
        "effects": ("effect_pack", "mob_effects_subtree"),
        "human": ("modular_component_pack", "runtime_humanoid_composition_subtree"),
        "huds": ("icon_or_ui_pack", "hud_subtree"),
        "inhands": ("modular_component_pack", "runtime_inhand_layer_subtree"),
        "large-worn-icons": ("modular_component_pack", "worn_icon_layer_subtree"),
        "telegraphing": ("effect_pack", "telegraph_effect_subtree"),
    }
    if top in mapped:
        return mapped[top]
    if top == "simple":
        if filename in {"corgi_back.dmi", "corgi_head.dmi"}:
            return "modular_component_pack", "explicit_separate_pet_body_layer_pack"
        if filename in {"bileworm_jump.dmi", "nest.dmi", "tendril.dmi"}:
            return "effect_pack", "explicit_jump_nest_or_tendril_effect_pack"
        return "complete_entity_candidate", "simple_mob_subtree"
    if top == "nonhuman-player":
        if filename in {"alienleap.dmi", "blob.dmi"}:
            return "effect_pack", "explicit_leap_or_runtime_blob_effect_pack"
        return "complete_entity_candidate", "nonhuman_player_mob_subtree"
    if top == "silicon":
        if filename in {"aibot_faces.dmi", "robot_items.dmi"}:
            return "modular_component_pack", "silicon_face_or_item_layer_pack"
        if filename == "ai.dmi":
            return "ambiguous_pack", "large_ai_screen_and_chassis_mixed_pack"
        return "complete_entity_candidate", "silicon_mob_subtree"
    if top == "rideables":
        if filename in {"mech_construct.dmi", "mech_construction.dmi"}:
            return "modular_component_pack", "mecha_construction_component_pack"
        return "complete_entity_candidate", "rideable_whole_entity_pack"
    root_roles: dict[str, tuple[PackRole, str]] = {
        "butts.dmi": ("modular_component_pack", "explicit_body_part_pack"),
        "cows.dmi": ("complete_entity_candidate", "root_whole_animal_pack"),
        "dust_animation.dmi": ("effect_pack", "explicit_dust_animation_pack"),
        "eyemob.dmi": ("complete_entity_candidate", "root_whole_mob_pack"),
        "gondolapod.dmi": ("complete_entity_candidate", "root_whole_mob_pack"),
        "landmarks.dmi": ("icon_or_ui_pack", "map_landmark_icon_pack"),
        "leg_masks.dmi": ("modular_component_pack", "explicit_leg_mask_pack"),
        "shells.dmi": ("ambiguous_pack", "shell_component_or_whole_entity_ambiguous"),
        "spacevines.dmi": ("effect_pack", "runtime_environmental_growth_pack"),
        "vatgrowing.dmi": ("effect_pack", "runtime_vat_growth_effect_pack"),
    }
    if top == "<root>" and filename in root_roles:
        return root_roles[filename]
    return "ambiguous_pack", "path_not_in_conservative_complete_entity_allowlist"


def _entity_class(logical_path: str, pack_role: PackRole) -> tuple[str, str]:
    path = logical_path.casefold()
    filename = PurePosixPath(path).name
    if "/human/" in path:
        return "humanoid", "human_subtree"
    if "/silicon/" in path or "/rideables/" in path or "hivebot" in filename:
        return "robot", "silicon_rideable_or_hivebot_path"
    if "/nonhuman-player/" in path:
        return "monster", "nonhuman_player_subtree"
    if "/simple/" in path:
        if filename in {"simple_human.dmi", "tourists.dmi"}:
            return "humanoid", "explicit_simple_human_pack"
        animal_tokens = {
            "animal",
            "arachnoid",
            "bargorilla",
            "bees",
            "cargorillia",
            "carp",
            "cows",
            "gorilla",
            "penguins",
            "pets",
            "rabbit",
            "sheep",
            "smspider",
            "turtle_trees",
        }
        if any(token in filename for token in animal_tokens):
            return "animal", "explicit_animal_filename_token"
        monster_tokens = {
            "demon",
            "eldritch",
            "icemoon",
            "lavaland",
            "thething",
            "voidwalker",
            "slimes",
            "clown_mobs",
        }
        if any(token in path for token in monster_tokens):
            return "monster", "explicit_monster_path_token"
        if filename in {"mad_piano.dmi", "meteor_heart.dmi"}:
            return "object", "explicit_animated_object_pack"
        return "creature", "simple_mob_without_narrower_class_evidence"
    if filename == "cows.dmi":
        return "animal", "explicit_cow_pack"
    if filename in {"eyemob.dmi", "gondolapod.dmi"}:
        return "monster", "explicit_nonhuman_root_mob_pack"
    if pack_role == "complete_entity_candidate":
        return "creature", "complete_entity_path_without_narrower_class_evidence"
    return "unknown", "noncomplete_or_ambiguous_pack"


def _classify_state_role(pack_role: PackRole, state_name: str) -> tuple[StateRole, str]:
    mapped: dict[PackRole, tuple[StateRole, str]] = {
        "modular_component_pack": ("modular_component", "pack_level_component_classification"),
        "effect_pack": ("effect_or_overlay", "pack_level_effect_classification"),
        "icon_or_ui_pack": ("icon_or_ui", "pack_level_icon_or_ui_classification"),
        "ambiguous_pack": ("ambiguous", "pack_level_ambiguity"),
    }
    if pack_role in mapped:
        return mapped[pack_role]
    stripped = state_name.casefold().strip()
    tokens = set(_TOKEN_RE.findall(stripped))
    if not stripped or stripped in {"blank", "error"}:
        return "ambiguous", "empty_blank_or_error_state_name"
    if re.search(r"(?:^|_)(?:e|e_r|l|bloom)$", stripped) or tokens & {
        "emissive",
        "overlay",
        "glow",
        "glowmask",
        "behind",
        "front",
        "adj",
    }:
        return "effect_or_overlay", "explicit_render_layer_suffix_or_token"
    if "front half" in stripped or "back half" in stripped:
        return "modular_component", "explicit_split_body_half_state"
    if tokens & {"eyes", "eye", "mouth", "face", "mask"} or stripped.endswith("_base"):
        return "modular_component", "explicit_face_eye_mask_or_base_layer_token"
    if (
        tokens
        & {
            "gib",
            "gibs",
            "splat",
            "beam",
            "cloud",
            "projectile",
            "telegraph",
            "tentacle",
            "trail",
            "aura",
            "acid",
        }
        or "spit" in stripped
    ):
        return "effect_or_overlay", "explicit_effect_payload_token"
    return "complete_entity_candidate", "whole_entity_pack_without_layer_or_effect_token"


_ACTION_SUFFIXES = frozenset(
    {
        "idle",
        "stand",
        "standing",
        "walk",
        "walking",
        "crawl",
        "run",
        "running",
        "fly",
        "flying",
        "swim",
        "swimming",
        "jump",
        "leap",
        "move",
        "moving",
        "dead",
        "death",
        "dying",
        "husked",
        "attack",
        "attacking",
        "firing",
        "fire",
        "pounce",
        "slash",
        "stab",
        "hurt",
        "hit",
        "critical",
        "stun",
        "stunned",
        "unconscious",
        "sleep",
        "sleeping",
        "sit",
        "alert",
        "spawn",
        "hatch",
        "hatched",
        "transform",
        "opening",
        "open",
        "broken",
        "dance",
        "roar",
        "cry",
        "wiggle",
    }
)


def _entity_cue(state_name: str) -> str:
    tokens = _TOKEN_RE.findall(state_name.casefold())
    strong_action_markers = {
        "alert",
        "attack",
        "attacking",
        "broken",
        "crawl",
        "cry",
        "dance",
        "dead",
        "death",
        "dying",
        "firing",
        "fly",
        "flying",
        "hatch",
        "hatched",
        "hit",
        "hurt",
        "husked",
        "idle",
        "jump",
        "leap",
        "move",
        "moving",
        "opening",
        "pounce",
        "roar",
        "run",
        "running",
        "sit",
        "slash",
        "sleep",
        "sleeping",
        "spawn",
        "stab",
        "stand",
        "standing",
        "stun",
        "stunned",
        "swim",
        "swimming",
        "transform",
        "unconscious",
        "walk",
        "walking",
        "wiggle",
    }
    marker_indices = [index for index, token in enumerate(tokens) if token in strong_action_markers]
    if marker_indices and marker_indices[0] > 0:
        tokens = tokens[: marker_indices[0]]
    while tokens and tokens[-1] in _ACTION_SUFFIXES:
        tokens.pop()
    if tokens and tokens[-1] in {"e", "l", "bloom"}:
        tokens.pop()
    cue = "-".join(tokens).strip("-")
    return cue or "unnamed"


def _normalize_action(
    raw: _RawState,
    *,
    paired_movement_names: frozenset[str],
    state_role: StateRole,
) -> tuple[str | None, str]:
    if raw.movement:
        return "walk", "dmi_movement_flag"
    tokens = _TOKEN_RE.findall(raw.name.casefold())
    token_set = set(tokens)
    if raw.name in paired_movement_names:
        return "idle", "same_state_name_has_movement_variant"
    checks: tuple[tuple[str, frozenset[str]], ...] = (
        ("death", frozenset({"dead", "death", "dying", "husked", "remains"})),
        ("attack", frozenset({"attack", "attacking", "firing", "fire", "pounce", "slash", "stab"})),
        ("run", frozenset({"run", "running"})),
        ("walk", frozenset({"walk", "walking", "move", "moving", "crawl"})),
        ("fly", frozenset({"fly", "flying"})),
        ("swim", frozenset({"swim", "swimming"})),
        ("jump", frozenset({"jump", "leap"})),
        ("idle", frozenset({"idle", "stand", "standing", "sit"})),
        ("sleep", frozenset({"sleep", "sleeping"})),
        ("hurt", frozenset({"hurt", "hit", "critical"})),
        ("stun", frozenset({"stun", "stunned", "unconscious"})),
        ("spawn", frozenset({"spawn", "hatch", "hatched", "opening"})),
        ("transform", frozenset({"transform"})),
        ("emote", frozenset({"dance", "roar", "alert", "cry", "wiggle"})),
    )
    for action, action_tokens in checks:
        if token_set & action_tokens:
            return action, "explicit_state_name_token"
    if state_role == "complete_entity_candidate" and raw.dirs in {4, 8}:
        return "idle", "complete_directional_base_state"
    return None, "no_conservative_action_evidence"


def _playback_semantics(raw: _RawState) -> str:
    if raw.loop_count == 0 and raw.rewind:
        return "unbounded_rewind_cycle"
    if raw.loop_count == 0:
        return "unbounded_forward_cycle"
    if raw.rewind:
        return "finite_declared_loop_count_with_rewind"
    if raw.loop_count == 1:
        return "one_shot_forward"
    return "finite_declared_loop_count_forward"


def _state_hash_payload(
    *,
    frame_width: int,
    frame_height: int,
    raw: _RawState,
    frame_hashes: Sequence[str],
    durations: Sequence[int],
    include_timing: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "directions": raw.dirs,
        "temporal_frames": raw.frames,
        "rgba_frame_sha256": list(frame_hashes),
    }
    if include_timing:
        payload.update(
            {
                "durations_milliseconds": list(durations),
                "loop_count": raw.loop_count,
                "rewind": raw.rewind,
                "movement": raw.movement,
            }
        )
    return payload


def _parse_dmi(logical_path: str, member_path: str, payload: bytes) -> DmiPack:
    description, metadata = _decode_ztxt_description(payload)
    frame_width, frame_height, raw_states = _parse_description(description)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            detected_format = image.format or "unknown"
            image.load()
            image_mode = image.mode
            image_width, image_height = image.size
            pil_description = image.info.get("Description")
            has_alpha = image_mode in {"RGBA", "LA"} or "transparency" in image.info
            rgba = image.convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise DmiMetadataError(f"Pillow could not decode DMI PNG: {error}") from error
    if detected_format != "PNG":
        raise DmiMetadataError(f"DMI detected as {detected_format!r}, expected PNG")
    if pil_description != description:
        raise DmiMetadataError("Pillow Description differs from literal zTXt decoding")
    if image_width % frame_width or image_height % frame_height:
        raise DmiMetadataError(
            f"atlas {image_width}x{image_height} is not divisible by {frame_width}x{frame_height}"
        )
    columns = image_width // frame_width
    rows = image_height // frame_height
    capacity = columns * rows
    declared_cells = sum(state.dirs * state.frames for state in raw_states)
    if declared_cells > capacity:
        raise DmiMetadataError(f"declared {declared_cells} cells exceed atlas capacity {capacity}")

    pack_role, pack_role_basis = _classify_pack_role(logical_path)
    entity_class, entity_class_basis = _entity_class(logical_path, pack_role)
    runtime_counts = Counter((state.name, state.movement) for state in raw_states)
    name_seen: Counter[str] = Counter()
    runtime_seen: Counter[tuple[str, bool]] = Counter()
    moving_names = {state.name for state in raw_states if state.movement}
    still_names = {state.name for state in raw_states if not state.movement}
    paired_names = frozenset(moving_names & still_names)

    states: list[DmiState] = []
    source_cell = 0
    for raw in raw_states:
        name_seen[raw.name] += 1
        runtime_key = (raw.name, raw.movement)
        runtime_seen[runtime_key] += 1
        role, role_basis = _classify_state_role(pack_role, raw.name)
        action, action_basis = _normalize_action(
            raw, paired_movement_names=paired_names, state_role=role
        )
        direction_names = _DIRECTION_NAMES[: raw.dirs] if raw.dirs <= 8 else ()
        issues: list[str] = []
        if raw.dirs not in {1, 4, 8}:
            issues.append("unsupported_direction_count")
        if raw.delay_literals and len(raw.delay_literals) != raw.frames:
            issues.append("delay_count_mismatch")
            durations: tuple[int, ...] = ()
        elif raw.delay_literals:
            durations = tuple(_parse_delay_milliseconds(item) for item in raw.delay_literals)
        else:
            durations = (100,) * raw.frames
        if any(
            hotspot.first_frame_one_based < 1 or hotspot.first_frame_one_based > raw.frames
            for hotspot in raw.hotspots
        ):
            issues.append("hotspot_frame_out_of_range")
        if runtime_counts[runtime_key] > 1:
            issues.append("duplicate_name_and_movement_runtime_key")
        if role != "complete_entity_candidate":
            issues.append("not_a_complete_entity_state")
        if entity_class == "unknown":
            issues.append("entity_class_not_supported_by_path_evidence")

        frame_records: list[FrameCell] = []
        frame_hashes: list[str] = []
        for temporal_index in range(raw.frames):
            for direction_index in range(raw.dirs):
                atlas_index = source_cell + temporal_index * raw.dirs + direction_index
                left = (atlas_index % columns) * frame_width
                top = (atlas_index // columns) * frame_height
                crop = rgba.crop((left, top, left + frame_width, top + frame_height))
                pixel_payload = struct.pack(">II", frame_width, frame_height) + crop.tobytes()
                pixel_hash = _sha256_bytes(pixel_payload)
                frame_hashes.append(pixel_hash)
                duration = durations[temporal_index] if durations else None
                frame_records.append(
                    FrameCell(
                        source_cell_index=atlas_index,
                        state_cell_index=temporal_index * raw.dirs + direction_index,
                        temporal_frame_index=temporal_index,
                        direction_index=direction_index,
                        direction=(
                            direction_names[direction_index]
                            if direction_index < len(direction_names)
                            else f"unsupported_{direction_index}"
                        ),
                        left=left,
                        top=top,
                        right=left + frame_width,
                        bottom=top + frame_height,
                        duration_milliseconds=duration,
                        rgba_sha256=pixel_hash,
                    )
                )
        source_sequence_hash = _canonical_hash(
            _state_hash_payload(
                frame_width=frame_width,
                frame_height=frame_height,
                raw=raw,
                frame_hashes=frame_hashes,
                durations=durations,
                include_timing=False,
            )
        )
        timed_sequence_hash = _canonical_hash(
            _state_hash_payload(
                frame_width=frame_width,
                frame_height=frame_height,
                raw=raw,
                frame_hashes=frame_hashes,
                durations=durations,
                include_timing=True,
            )
        )
        quarantine = tuple(sorted(set(issues)))
        eligible_complete = not quarantine
        eligible_animated_action = eligible_complete and raw.frames > 1 and action is not None
        exclusions: list[str] = []
        if action is None:
            exclusions.append("no_conservative_action_label")
        if raw.frames <= 1:
            exclusions.append("not_temporally_animated")
        exclusions.extend(quarantine)
        states.append(
            DmiState(
                declaration_index=raw.declaration_index,
                name=raw.name,
                name_occurrence_index=name_seen[raw.name] - 1,
                runtime_key_occurrence_index=runtime_seen[runtime_key] - 1,
                entity_cue=_entity_cue(raw.name),
                entity_class=entity_class,
                entity_class_basis=entity_class_basis,
                normalized_action=action,
                normalized_action_basis=action_basis,
                role=role,
                role_basis=role_basis,
                direction_count=raw.dirs,
                direction_names=direction_names,
                temporal_frame_count=raw.frames,
                delay_decisecond_literals=raw.delay_literals,
                durations_milliseconds=durations,
                delays_declared=bool(raw.delay_literals),
                loop_count=raw.loop_count,
                rewind=raw.rewind,
                movement=raw.movement,
                playback_semantics=_playback_semantics(raw),
                hotspots=raw.hotspots,
                source_cell_start=source_cell,
                source_cell_count=raw.dirs * raw.frames,
                frames=tuple(frame_records),
                source_sequence_sha256=source_sequence_hash,
                timed_sequence_sha256=timed_sequence_hash,
                is_temporally_animated=raw.frames > 1,
                eligible_complete_entity_sequence=eligible_complete,
                eligible_animated_action_sequence=eligible_animated_action,
                quarantine_reasons=quarantine,
                selection_exclusion_reasons=tuple(sorted(set(exclusions))),
            )
        )
        source_cell += raw.dirs * raw.frames

    pack_issues: list[str] = []
    if pack_role != "complete_entity_candidate":
        pack_issues.append("not_a_complete_entity_pack")
    if declared_cells < capacity:
        pack_issues.append("atlas_has_unused_cells")
    if image_mode not in {"RGBA", "P"}:
        pack_issues.append("unreviewed_png_color_mode")
    if not has_alpha:
        pack_issues.append("png_has_no_alpha_channel_or_transparency")
    blob_url = f"{TGSTATION_REPOSITORY_URL}/blob/{TGSTATION_COMMIT}/{logical_path}"
    history_url = f"{TGSTATION_REPOSITORY_URL}/commits/{TGSTATION_COMMIT}/{logical_path}"
    return DmiPack(
        logical_path=logical_path,
        member_path=member_path,
        blob_url=blob_url,
        history_url=history_url,
        lineage_key=f"github:tgstation/tgstation@{TGSTATION_COMMIT}",
        asset_deduplication_key=(f"github:tgstation/tgstation@{TGSTATION_COMMIT}:{logical_path}"),
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        detected_format=detected_format,
        image_mode=image_mode,
        has_alpha=has_alpha,
        image_width=image_width,
        image_height=image_height,
        frame_width=frame_width,
        frame_height=frame_height,
        grid_columns=columns,
        grid_rows=rows,
        grid_capacity=capacity,
        declared_source_cells=declared_cells,
        unused_source_cells=capacity - declared_cells,
        metadata=metadata,
        description_verbatim=description,
        pack_role=pack_role,
        pack_role_basis=pack_role_basis,
        entity_class=entity_class,
        entity_class_basis=entity_class_basis,
        states=tuple(states),
        quarantine_reasons=tuple(sorted(set(pack_issues))),
    )


def _evidence_document(
    archive: ZipFile,
    by_path: dict[str, _ArchiveEntry],
    logical_path: str,
    purpose: str,
) -> EvidenceDocument:
    entry = by_path.get(logical_path)
    if entry is None or entry.kind != "file":
        raise TgstationArchiveError(f"required evidence is absent: {logical_path}")
    payload = archive.read(entry.info)
    return EvidenceDocument(
        logical_path=logical_path,
        member_path=entry.member_path,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        purpose=purpose,
    )


def _state_reference(pack: DmiPack, state: DmiState) -> StateReference:
    return StateReference(
        logical_path=pack.logical_path,
        declaration_index=state.declaration_index,
        name=state.name,
        name_occurrence_index=state.name_occurrence_index,
    )


def _entity_action_sets(packs: Sequence[DmiPack]) -> tuple[EntityActionSet, ...]:
    grouped: defaultdict[tuple[str, str], list[tuple[DmiPack, DmiState]]] = defaultdict(list)
    for pack in packs:
        for state in pack.states:
            if state.eligible_complete_entity_sequence:
                grouped[(pack.logical_path, state.entity_cue)].append((pack, state))
    result: list[EntityActionSet] = []
    for (logical_path, cue), rows in sorted(grouped.items()):
        actions = tuple(
            sorted({state.normalized_action for _, state in rows if state.normalized_action})
        )
        references = tuple(_state_reference(pack, state) for pack, state in rows)
        result.append(
            EntityActionSet(
                entity_key=(f"github:tgstation/tgstation@{TGSTATION_COMMIT}:{logical_path}#{cue}"),
                logical_path=logical_path,
                entity_cue=cue,
                entity_class=rows[0][1].entity_class,
                state_references=references,
                actions=actions,
                complete_sequence_count=len(rows),
                animated_action_sequence_count=sum(
                    state.eligible_animated_action_sequence for _, state in rows
                ),
                steerable=len(actions) >= 2,
                has_animated_action=any(
                    state.eligible_animated_action_sequence for _, state in rows
                ),
            )
        )
    return tuple(result)


def _audit_payload_without_hash(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    inventory_sha256: str,
    archive_root: str,
    counts: TgstationArchiveCounts,
    packs: tuple[DmiPack, ...],
    malformed_dmis: tuple[MalformedDmiRecord, ...],
    entity_action_sets: tuple[EntityActionSet, ...],
    duplicate_dmi_groups: tuple[DuplicateDmiGroup, ...],
    duplicate_state_groups: tuple[DuplicateStateGroup, ...],
    rights: RightsAudit,
    engine_semantics: EngineSemanticsAudit,
    acquisition_evidence: AcquisitionEvidence,
    issues: tuple[AuditIssue, ...],
) -> dict[str, Any]:
    projection_policy = (
        "Only states classified as complete entities with no quarantine reasons qualify.",
        "Clothing, humanoid body parts, in-hands, overlays, effects, UI, and ambiguous "
        "packs stay separate.",
        "State declarations remain distinct; duplicate names and movement variants are never "
        "collapsed.",
        "Atlas cells follow pinned /tg/station ordering: state, temporal frame, direction, "
        "row-major cell.",
        "Raw delays, loop counts, rewind, movement, hotspots, and verbatim zTXt Description "
        "travel with exports.",
        "Timing-list mismatches and unsupported direction counts are evidence, never guessed "
        "sequences.",
        "Pixel and timed-sequence hashes support leakage-safe deduplication without losing "
        "upstream lineage.",
        "CC BY-SA 3.0 asset evidence, exact commit/path links, and the no-per-file-author caveat "
        "travel with exports.",
        "No archive code is executed and the repository ZIP is never extracted by this adapter.",
    )
    return {
        "source_id": SOURCE_ID,
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "inventory_sha256": inventory_sha256,
        "repository_url": TGSTATION_REPOSITORY_URL,
        "commit": TGSTATION_COMMIT,
        "commit_url": TGSTATION_COMMIT_URL,
        "archive_url": TGSTATION_ARCHIVE_URL,
        "archive_root": archive_root,
        "counts": asdict(counts),
        "packs": [asdict(pack) for pack in packs],
        "malformed_dmis": [asdict(item) for item in malformed_dmis],
        "entity_action_sets": [asdict(item) for item in entity_action_sets],
        "duplicate_dmi_groups": [asdict(item) for item in duplicate_dmi_groups],
        "duplicate_state_groups": [asdict(item) for item in duplicate_state_groups],
        "rights": asdict(rights),
        "engine_semantics": asdict(engine_semantics),
        "acquisition_evidence": asdict(acquisition_evidence),
        "issues": [asdict(item) for item in issues],
        "projection_policy": list(projection_policy),
    }


def audit_tgstation_archive(
    archive_path: Path, *, archive_sha256: str | None = None
) -> TgstationArchiveAudit:
    """Audit all ``icons/mob/**/*.dmi`` files without extraction or DB writes."""

    archive_path = Path(archive_path)
    digest = archive_sha256 or _sha256_file(archive_path)
    try:
        with ZipFile(archive_path) as archive:
            root, entries, inventory_sha256 = _validate_archive_members(archive)
            by_path = {entry.logical_path: entry for entry in entries if entry.logical_path}
            dmi_entries = tuple(
                sorted(
                    (
                        entry
                        for entry in entries
                        if entry.kind == "file" and _MOB_DMI_RE.fullmatch(entry.logical_path)
                    ),
                    key=lambda entry: entry.logical_path.encode("utf-8"),
                )
            )
            if not dmi_entries:
                raise TgstationArchiveError("no icons/mob DMI files found")

            packs: list[DmiPack] = []
            malformed: list[MalformedDmiRecord] = []
            for entry in dmi_entries:
                payload = archive.read(entry.info)
                try:
                    packs.append(_parse_dmi(entry.logical_path, entry.member_path, payload))
                except (DmiMetadataError, OSError, ValueError) as error:
                    malformed.append(
                        MalformedDmiRecord(
                            logical_path=entry.logical_path,
                            member_path=entry.member_path,
                            sha256=_sha256_bytes(payload),
                            size_bytes=len(payload),
                            error_type=type(error).__name__,
                            error=str(error),
                        )
                    )

            pack_tuple = tuple(packs)
            malformed_tuple = tuple(malformed)
            action_sets = _entity_action_sets(pack_tuple)
            dmi_by_hash: defaultdict[str, list[str]] = defaultdict(list)
            for pack in pack_tuple:
                dmi_by_hash[pack.sha256].append(pack.logical_path)
            duplicate_dmis = tuple(
                DuplicateDmiGroup(sha256=sha, logical_paths=tuple(paths))
                for sha, paths in sorted(dmi_by_hash.items())
                if len(paths) > 1
            )
            state_by_hash: defaultdict[str, list[StateReference]] = defaultdict(list)
            for pack in pack_tuple:
                for state in pack.states:
                    if state.eligible_complete_entity_sequence:
                        state_by_hash[state.timed_sequence_sha256].append(
                            _state_reference(pack, state)
                        )
            duplicate_states = tuple(
                DuplicateStateGroup(timed_sequence_sha256=sha, references=tuple(refs))
                for sha, refs in sorted(state_by_hash.items())
                if len(refs) > 1
            )

            states = [state for pack in pack_tuple for state in pack.states]
            pack_roles = Counter(pack.pack_role for pack in pack_tuple)
            state_roles = Counter(state.role for state in states)
            eligible = [state for state in states if state.eligible_complete_entity_sequence]
            classes = Counter(state.entity_class for state in eligible)
            actions = Counter(
                state.normalized_action for state in eligible if state.normalized_action is not None
            )
            counts = TgstationArchiveCounts(
                archive_members=len(entries),
                archive_files=sum(entry.kind == "file" for entry in entries),
                archive_directories=sum(entry.kind == "directory" for entry in entries),
                archive_symlinks=sum(entry.kind == "symlink" for entry in entries),
                archive_compressed_bytes=sum(entry.info.compress_size for entry in entries),
                archive_uncompressed_bytes=sum(entry.info.file_size for entry in entries),
                mob_dmi_files=len(dmi_entries),
                parsed_dmi_files=len(pack_tuple),
                malformed_dmi_files=len(malformed_tuple),
                dmi_states=len(states),
                declared_source_cells=sum(pack.declared_source_cells for pack in pack_tuple),
                temporally_animated_states=sum(state.is_temporally_animated for state in states),
                directional_states=sum(state.direction_count > 1 for state in states),
                movement_states=sum(state.movement for state in states),
                rewind_states=sum(state.rewind for state in states),
                finite_loop_states=sum(state.loop_count > 0 for state in states),
                delay_declared_states=sum(state.delays_declared for state in states),
                delay_count_mismatch_states=sum(
                    "delay_count_mismatch" in state.quarantine_reasons for state in states
                ),
                invalid_hotspot_states=sum(
                    "hotspot_frame_out_of_range" in state.quarantine_reasons for state in states
                ),
                duplicate_runtime_key_excess=sum(
                    state.runtime_key_occurrence_index > 0 for state in states
                ),
                exact_capacity_dmis=sum(pack.unused_source_cells == 0 for pack in pack_tuple),
                surplus_capacity_dmis=sum(pack.unused_source_cells > 0 for pack in pack_tuple),
                unused_source_cells=sum(pack.unused_source_cells for pack in pack_tuple),
                complete_entity_candidate_states=state_roles["complete_entity_candidate"],
                eligible_complete_entity_sequences=len(eligible),
                eligible_action_sequences=sum(
                    state.normalized_action is not None for state in eligible
                ),
                eligible_animated_action_sequences=sum(
                    state.eligible_animated_action_sequence for state in states
                ),
                entity_action_sets=len(action_sets),
                steerable_entity_action_sets=sum(item.steerable for item in action_sets),
                steerable_entity_action_sets_with_animation=sum(
                    item.steerable and item.has_animated_action for item in action_sets
                ),
                duplicate_dmi_hash_groups=len(duplicate_dmis),
                duplicate_dmi_hash_excess=sum(
                    len(group.logical_paths) - 1 for group in duplicate_dmis
                ),
                duplicate_complete_state_groups=len(duplicate_states),
                duplicate_complete_state_excess=sum(
                    len(group.references) - 1 for group in duplicate_states
                ),
                pack_role_counts=_counter_tuple(pack_roles),
                state_role_counts=_counter_tuple(state_roles),
                entity_class_counts=_counter_tuple(classes),
                action_counts=_counter_tuple(actions),
                direction_count_counts=_counter_tuple(
                    Counter(state.direction_count for state in states)
                ),
                loop_count_counts=_counter_tuple(Counter(state.loop_count for state in states)),
                image_mode_counts=_counter_tuple(Counter(pack.image_mode for pack in pack_tuple)),
                frame_size_counts=_counter_tuple(
                    Counter(f"{pack.frame_width}x{pack.frame_height}" for pack in pack_tuple)
                ),
            )

            readme = _evidence_document(
                archive, by_path, "README.md", "asset_license_and_project_description"
            )
            rights_filenames = {
                "authors",
                "authors.txt",
                "copying",
                "copyright",
                "copyright.txt",
                "credits",
                "credits.md",
                "credits.txt",
                "license",
                "license.md",
                "license.txt",
                "readme",
                "readme.md",
                "readme.txt",
            }
            path_local_rights = tuple(
                _evidence_document(
                    archive,
                    by_path,
                    path,
                    "path_local_icon_rights_or_credit_document",
                )
                for path in sorted(by_path)
                if path.casefold().startswith("icons/mob/")
                and PurePosixPath(path).name.casefold() in rights_filenames
                and by_path[path].kind == "file"
            )
            rights = RightsAudit(
                asset_license_expression="CC-BY-SA-3.0",
                asset_license_scope="all assets including icons unless otherwise indicated",
                asset_license_basis=(
                    "Pinned README.md explicitly states that all assets including icons are "
                    "Creative Commons 3.0 BY-SA unless otherwise indicated."
                ),
                code_license_expression="AGPL-3.0 after the documented 2014 cutoff",
                root_license=_evidence_document(
                    archive, by_path, "LICENSE", "current_repository_code_license"
                ),
                historical_code_license=_evidence_document(
                    archive, by_path, "GPLv3.txt", "historical_repository_code_license"
                ),
                readme=readme,
                path_local_rights_documents=path_local_rights,
                per_file_author_manifest_present=bool(path_local_rights),
                attribution_policy=(
                    "Preserve repository, exact commit, DMI path, blob URL, and history URL.",
                    "Preserve README asset-license evidence with every projected or exported "
                    "asset.",
                    "Use commit history for contributor attribution; do not invent per-file "
                    "authors.",
                    "Treat any later-discovered per-file exception as narrower than the README "
                    "default.",
                ),
                caveat=(
                    "Path-local rights documents are preserved and take precedence over the "
                    "README default where applicable; git history is not converted into a "
                    "single-author claim."
                    if path_local_rights
                    else "The pinned tree has no per-DMI author/license manifest under icons/mob. "
                    "The README default says 'unless otherwise indicated'; this audit found no "
                    "narrower path-local license document and does not convert git history into a "
                    "single-author claim."
                ),
            )
            engine_document = _evidence_document(
                archive, by_path, "tools/dmi/__init__.py", "pinned_dmi_loader_semantics"
            )
            engine = EngineSemanticsAudit(
                implementation=engine_document,
                immutable_url=(
                    f"{TGSTATION_REPOSITORY_URL}/blob/{TGSTATION_COMMIT}/tools/dmi/__init__.py"
                ),
                official_reference_url=BYOND_ICON_REFERENCE_URL,
                semantics=(
                    "DMI version is 4.0 and default cells are 32x32.",
                    "Direction order is south, north, east, west, southeast, southwest, "
                    "northeast, northwest.",
                    "Atlas traversal is state declaration, temporal frame, direction, then "
                    "row-major cell.",
                    "Absent frame delays default to one decisecond; declared delays are "
                    "preserved exactly.",
                    "Loop zero is unlimited, loop one is one-shot, and rewind/movement are "
                    "explicit flags.",
                    "Hotspot changes are declared with one-based temporal frame positions.",
                ),
            )
            acquisition = AcquisitionEvidence(
                requested_url=TGSTATION_ARCHIVE_URL,
                final_url=TGSTATION_ARCHIVE_URL,
                expected_sha256=EXPECTED_TGSTATION_ARCHIVE_SHA256,
                observed_size_bytes=archive_path.stat().st_size,
                observed_etag=TGSTATION_ARCHIVE_ETAG,
                retrieval_method="guarded_resumable_http_into_immutable_sha256_cas",
            )
            issues = (
                AuditIssue(
                    "malformed_dmi",
                    counts.malformed_dmi_files,
                    "Unreadable or structurally ambiguous DMI files remain hash-addressed "
                    "quarantine records.",
                ),
                AuditIssue(
                    "delay_count_mismatch",
                    counts.delay_count_mismatch_states,
                    "Delay lists that do not match temporal frame counts are retained without "
                    "guessed timing.",
                ),
                AuditIssue(
                    "invalid_hotspot_frame",
                    counts.invalid_hotspot_states,
                    "Hotspot frame positions outside the declared temporal range are quarantined.",
                ),
                AuditIssue(
                    "duplicate_runtime_key",
                    counts.duplicate_runtime_key_excess,
                    "Repeated name-plus-movement keys remain separate and are not assigned "
                    "guessed precedence.",
                ),
                AuditIssue(
                    "unused_atlas_cell",
                    counts.unused_source_cells,
                    "Trailing PNG grid cells not claimed by metadata are inventory evidence only.",
                ),
                AuditIssue(
                    "duplicate_complete_state_payload",
                    counts.duplicate_complete_state_excess,
                    "Identical timed RGBA sequences require identity-safe split and "
                    "deduplication policy.",
                ),
            )
            payload = _audit_payload_without_hash(
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                inventory_sha256=inventory_sha256,
                archive_root=root,
                counts=counts,
                packs=pack_tuple,
                malformed_dmis=malformed_tuple,
                entity_action_sets=action_sets,
                duplicate_dmi_groups=duplicate_dmis,
                duplicate_state_groups=duplicate_states,
                rights=rights,
                engine_semantics=engine,
                acquisition_evidence=acquisition,
                issues=issues,
            )
            audit_hash = _canonical_hash(payload)
            return TgstationArchiveAudit(
                source_id=SOURCE_ID,
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                inventory_sha256=inventory_sha256,
                repository_url=TGSTATION_REPOSITORY_URL,
                commit=TGSTATION_COMMIT,
                commit_url=TGSTATION_COMMIT_URL,
                archive_url=TGSTATION_ARCHIVE_URL,
                archive_root=root,
                counts=counts,
                packs=pack_tuple,
                malformed_dmis=malformed_tuple,
                entity_action_sets=action_sets,
                duplicate_dmi_groups=duplicate_dmis,
                duplicate_state_groups=duplicate_states,
                rights=rights,
                engine_semantics=engine,
                acquisition_evidence=acquisition,
                issues=issues,
                projection_policy=tuple(payload["projection_policy"]),
                audit_record_sha256=audit_hash,
            )
    except BadZipFile as error:
        raise TgstationArchiveError(f"not a valid ZIP archive: {archive_path}") from error


def audit_known_tgstation_archive(archive_path: Path) -> TgstationArchiveAudit:
    """Hash-check and audit the exact pinned /tg/station repository snapshot."""

    archive_path = Path(archive_path)
    size = archive_path.stat().st_size
    if size != EXPECTED_TGSTATION_ARCHIVE_BYTES:
        raise TgstationArchiveError(
            "/tg/station archive size mismatch: expected "
            f"{EXPECTED_TGSTATION_ARCHIVE_BYTES}, got {size}"
        )
    digest = _sha256_file(archive_path)
    if digest != EXPECTED_TGSTATION_ARCHIVE_SHA256:
        raise TgstationArchiveError(
            "/tg/station archive SHA-256 mismatch: "
            f"expected {EXPECTED_TGSTATION_ARCHIVE_SHA256}, got {digest}"
        )
    audit = audit_tgstation_archive(archive_path, archive_sha256=digest)
    if audit.archive_root != EXPECTED_TGSTATION_ARCHIVE_ROOT:
        raise TgstationArchiveError(
            f"/tg/station archive root mismatch: expected {EXPECTED_TGSTATION_ARCHIVE_ROOT!r}, "
            f"got {audit.archive_root!r}"
        )
    if (
        EXPECTED_TGSTATION_INVENTORY_SHA256
        and audit.inventory_sha256 != EXPECTED_TGSTATION_INVENTORY_SHA256
    ):
        raise TgstationArchiveError(
            "/tg/station inventory SHA-256 mismatch: expected "
            f"{EXPECTED_TGSTATION_INVENTORY_SHA256}, got {audit.inventory_sha256}"
        )
    if (
        EXPECTED_TGSTATION_AUDIT_RECORD_SHA256
        and audit.audit_record_sha256 != EXPECTED_TGSTATION_AUDIT_RECORD_SHA256
    ):
        raise TgstationArchiveError(
            "/tg/station audit-record SHA-256 mismatch: expected "
            f"{EXPECTED_TGSTATION_AUDIT_RECORD_SHA256}, got {audit.audit_record_sha256}"
        )
    return audit


def known_tgstation_cas_path(raw_root: Path) -> Path:
    """Return the immutable CAS path for the pinned archive digest."""

    digest = EXPECTED_TGSTATION_ARCHIVE_SHA256
    return Path(raw_root) / "objects" / "sha256" / digest[:2] / digest[2:4] / digest
