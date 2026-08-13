"""Read-only audit of commit-pinned Space Station 14 RSI mob sprites.

An RSI is a directory containing ``meta.json`` and one image per named state.
This adapter follows the paired RobustToolbox loader: source cells are read in
row-major order, direction runs are concatenated South/North/East/West and then
the four diagonals, omitted delays mean one one-second frame per direction, and
different directional timings are folded onto a common millisecond timeline.

The audit never extracts the repository archive and never writes to the corpus
database.  It retains every per-RSI license/copyright field and source URL,
quarantines non-commercial packs, and keeps modular layers separate from
complete-entity candidates.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import stat
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile, ZipInfo

from PIL import Image, UnidentifiedImageError

EXPECTED_SS14_ARCHIVE_SHA256 = "125ca78d04a4f522e04597bf49d49fdb67a8cd2c2d079be13a2b3edb5591c444"
SS14_COMMIT = "c724191e0407f2868780de6d308183477701538e"
SS14_REPOSITORY_URL = "https://github.com/space-wizards/space-station-14"
SS14_COMMIT_URL = f"{SS14_REPOSITORY_URL}/tree/{SS14_COMMIT}"
SS14_ARCHIVE_URL = f"https://codeload.github.com/space-wizards/space-station-14/zip/{SS14_COMMIT}"
ROBUST_TOOLBOX_COMMIT = "15297f18f697d3a60cc1c764614fce85d234a395"
ROBUST_TOOLBOX_REPOSITORY_URL = "https://github.com/space-wizards/RobustToolbox"
ROBUST_TOOLBOX_ARCHIVE_URL = (
    f"https://codeload.github.com/space-wizards/RobustToolbox/zip/{ROBUST_TOOLBOX_COMMIT}"
)
EXPECTED_ROBUST_TOOLBOX_ARCHIVE_SHA256 = (
    "eb42a1fa7e6ca3fa5e11df5c9ca89b1fc609078973278959c02a22f600c9ed82"
)

_EXPECTED_ROOT = f"space-station-14-{SS14_COMMIT}"
_MOBS_PREFIX = "Resources/Textures/Mobs/"
_META_SUFFIX = ".rsi/meta.json"
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
_KNOWN_LICENSES = frozenset(
    {
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
        "CC-BY-NC-3.0",
        "CC-BY-NC-4.0",
        "CC-BY-NC-SA-3.0",
        "CC-BY-NC-SA-4.0",
        "CC0-1.0",
    }
)
_CLASSIFICATION_EVIDENCE_PATHS = (
    "Resources/Prototypes/AppearanceCustomization/station_ai.yml",
    "Resources/Prototypes/Entities/Mobs/Player/silicon.yml",
    "Resources/Prototypes/Entities/Mobs/Cyborgs/base_borg_chassis.yml",
    "Resources/Prototypes/Entities/Mobs/Cyborgs/borg_chassis.yml",
    "Resources/Prototypes/Entities/Mobs/Cyborgs/xenoborgs.yml",
    "Resources/Prototypes/Entities/Mobs/NPCs/animals.yml",
    "Resources/Prototypes/Entities/Mobs/NPCs/pets.yml",
)
_URL_RE = re.compile(r"https?://[^\s|<>\"']+", re.IGNORECASE)
_COMMIT_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", re.IGNORECASE)
_TOKEN_RE = re.compile(r"[a-z0-9]+")

StateRole = Literal[
    "complete_entity_candidate",
    "modular_component",
    "effect_or_overlay",
    "icon_or_item_view",
    "ambiguous",
]


class Ss14ArchiveError(ValueError):
    """Raised when an archive is not a safe SS14 repository snapshot."""


class RsiMetadataError(ValueError):
    """Raised when RSI metadata contradicts the pinned loader contract."""


@dataclass(frozen=True, slots=True)
class EvidenceDocument:
    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int
    purpose: str


@dataclass(frozen=True, slots=True)
class UpstreamReference:
    url: str
    host: str
    repository: str | None
    reference_kind: str
    revision: str | None
    revision_is_immutable: bool
    asset_path: str | None
    lineage_key: str | None
    asset_deduplication_key: str | None
    tgstation_family: bool


@dataclass(frozen=True, slots=True)
class FrameCell:
    source_cell_index: int
    direction_index: int
    direction: str
    frame_index_in_direction: int
    delay_seconds: float
    left: int
    top: int
    right: int
    bottom: int


@dataclass(frozen=True, slots=True)
class StateImageEvidence:
    logical_path: str
    member_path: str
    sha256: str
    size_bytes: int
    declared_extension: str
    detected_format: str
    image_mode: str
    width: int
    height: int
    grid_columns: int
    grid_rows: int
    grid_capacity: int
    unused_cell_count: int


@dataclass(frozen=True, slots=True)
class RsiState:
    name: str
    entity_cue: str
    normalized_action: str | None
    normalized_action_basis: str
    role: StateRole
    role_basis: str
    direction_count: int
    direction_names: tuple[str, ...]
    delays_declared: bool
    source_delays_seconds: tuple[tuple[float, ...], ...]
    engine_delays_seconds: tuple[float, ...]
    engine_source_cell_indices: tuple[tuple[int, ...], ...]
    source_frames: tuple[FrameCell, ...]
    image: StateImageEvidence | None
    expected_source_cell_count: int
    is_animated: bool
    loop_semantics: str
    eligible_complete_entity_sequence: bool
    eligible_animated_action_sequence: bool
    quarantine_reasons: tuple[str, ...]
    selection_exclusion_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RsiPack:
    logical_path: str
    category: str
    entity_class_candidates: tuple[str, ...]
    entity_class_basis: str
    version: int
    frame_width: int
    frame_height: int
    license_expression: str
    copyright: str
    rights_status: str
    metadata_evidence: EvidenceDocument
    upstream_references: tuple[UpstreamReference, ...]
    load_srgb: bool
    meta_atlas: bool
    rsic: bool
    states: tuple[RsiState, ...]
    declared_state_names: tuple[str, ...]
    extra_image_members: tuple[str, ...]
    role_summary: str
    quarantine_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RightsAudit:
    repository_license_expression: str
    repository_license_scope: str
    root_license: EvidenceDocument
    rsi_schema: EvidenceDocument | None
    per_pack_rights_required: bool
    noncommercial_policy: str


@dataclass(frozen=True, slots=True)
class AuditIssue:
    code: str
    count: int
    detail: str


@dataclass(frozen=True, slots=True)
class Ss14ArchiveCounts:
    archive_members: int
    archive_files: int
    mob_rsi_packs: int
    rsi_directories_without_meta: int
    states: int
    expected_source_cells: int
    decoded_source_cells: int
    engine_timeline_occurrences: int
    animated_states: int
    directional_states: int
    directional_animated_states: int
    normalized_action_states: int
    eligible_complete_entity_sequences: int
    eligible_animated_action_sequences: int
    noncommercial_packs: int
    noncommercial_states: int
    tgstation_family_packs: int
    tgstation_immutable_revision_packs: int
    exact_capacity_images: int
    surplus_capacity_images: int
    unused_source_cells: int
    missing_images: int
    invalid_or_short_images: int
    duplicate_image_hash_groups: int
    duplicate_image_hash_excess: int
    undeclared_state_images: int
    srgb_false_packs: int
    meta_atlas_false_packs: int
    category_counts: tuple[tuple[str, int], ...]
    license_counts: tuple[tuple[str, int], ...]
    action_counts: tuple[tuple[str, int], ...]
    role_counts: tuple[tuple[str, int], ...]
    image_format_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class Ss14ArchiveAudit:
    archive_sha256: str
    archive_size_bytes: int
    repository_url: str
    commit: str
    commit_url: str
    archive_url: str
    archive_root: str
    robust_toolbox_commit: str
    engine_evidence_urls: tuple[str, ...]
    counts: Ss14ArchiveCounts
    packs: tuple[RsiPack, ...]
    rights: RightsAudit
    classification_evidence: tuple[EvidenceDocument, ...]
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
        raise Ss14ArchiveError(f"unsafe archive member path: {name!r}")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise Ss14ArchiveError(f"unsafe archive member path: {name!r}")
    return pure.as_posix()


def _validate_members(infos: Sequence[ZipInfo]) -> tuple[str, tuple[_ArchiveMember, ...]]:
    seen: set[str] = set()
    roots: set[str] = set()
    files: list[_ArchiveMember] = []
    prepared: list[tuple[str, ZipInfo]] = []
    for info in infos:
        name = _normalize_member_name(info.filename)
        if name in seen:
            raise Ss14ArchiveError(f"duplicate archive member: {name}")
        seen.add(name)
        roots.add(PurePosixPath(name).parts[0])
        if info.flag_bits & 0x1:
            raise Ss14ArchiveError(f"encrypted archive member: {name}")
        mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode)
        if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise Ss14ArchiveError(f"non-regular archive member: {name}")
        prepared.append((name, info))
    if len(roots) != 1:
        raise Ss14ArchiveError(f"expected one archive root, found {sorted(roots)!r}")
    root = next(iter(roots))
    prefix = root + "/"
    for name, info in prepared:
        if info.is_dir() or name == root:
            continue
        if not name.startswith(prefix):
            raise Ss14ArchiveError(f"member escaped archive root: {name}")
        files.append(_ArchiveMember(name[len(prefix) :], name, info))
    return root, tuple(files)


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fold_direction_delays(
    delays: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[tuple[int, ...], ...]]:
    """Mirror RobustToolbox ``RSIResource.FoldDelays``.

    Returned indices address source cells in direction-major order.  Multi-
    direction inputs use the engine's 1 ms fixed-point truncation.
    """

    if not delays:
        raise RsiMetadataError("an RSI state must have at least one direction")
    prepared: list[list[float]] = []
    for row in delays:
        if not row:
            prepared.append([1.0])
            continue
        values: list[float] = []
        for raw in row:
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise RsiMetadataError(f"delay is not numeric: {raw!r}")
            value = _float32(float(raw))
            if not math.isfinite(value) or value <= 0:
                raise RsiMetadataError(f"delay must be finite and positive: {raw!r}")
            values.append(value)
        prepared.append(values)
    if len(prepared) == 1:
        values = tuple(prepared[0])
        return values, (tuple(range(len(values))),)

    fixed = [[int(value * 1000) for value in row] for row in prepared]
    if any(value <= 0 for row in fixed for value in row):
        raise RsiMetadataError("multi-direction delays must survive 1 ms engine quantization")
    totals = [sum(row) for row in fixed]
    maximum = max(totals)
    for row, total in zip(fixed, totals, strict=True):
        row[-1] += maximum - total

    base_offsets: list[int] = []
    running = 0
    for row in fixed:
        base_offsets.append(running)
        running += len(row)
    positions = [0] * len(fixed)
    common: list[int] = []
    indices: list[list[int]] = [[] for _ in fixed]
    while True:
        interval = min(row[position] for row, position in zip(fixed, positions, strict=True))
        common.append(interval)
        for direction, base in enumerate(base_offsets):
            indices[direction].append(base + positions[direction])
        for direction, row in enumerate(fixed):
            position = positions[direction]
            row[position] -= interval
            if row[position] == 0:
                positions[direction] += 1
            if positions[direction] == len(row):
                return (
                    tuple(value / 1000 for value in common),
                    tuple(tuple(row_indices) for row_indices in indices),
                )


def normalize_state_action(name: str) -> tuple[str | None, str]:
    """Map explicit state-name tokens into conservative source action cues.

    These cues are audit evidence, not necessarily members of the configured
    conditioning taxonomy.  The DB projection resolves that boundary and
    quarantines noncanonical cues without silently relabelling them.
    """

    tokens = _TOKEN_RE.findall(name.casefold())
    checks: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("death", ("dead", "death", "die", "dying")),
        ("attack", ("attack", "attacking", "preattack")),
        ("run", ("run", "running")),
        ("walk", ("walk", "walking")),
        ("move", ("move", "moving")),
        ("idle", ("idle", "idling", "standing", "waiting")),
        ("spawn", ("spawn", "spawning")),
        ("sleep", ("sleep", "sleeping")),
        ("rest", ("rest", "resting")),
        ("sit", ("sit", "sitting")),
        ("hurt", ("hurt", "hit", "crit", "critical")),
        ("stun", ("stun", "stunned")),
        ("dance", ("dance", "dancing")),
        ("jump", ("jump", "jumping")),
        ("fly", ("fly", "flying")),
        ("swim", ("swim", "swimming")),
        ("emote", ("emote", "wiggle", "roar")),
    )
    for action, words in checks:
        for token in tokens:
            if token in words or (action == "death" and token.startswith("dead")):
                return action, "explicit_state_name_token"
    return None, "state_name_unmapped"


def _entity_cue(rsi_path: str, state_name: str) -> str:
    pack = PurePosixPath(rsi_path).name.removesuffix(".rsi")
    parts = _TOKEN_RE.findall(state_name.casefold())
    removable = {
        "attack",
        "attacking",
        "preattack",
        "run",
        "running",
        "walk",
        "walking",
        "move",
        "moving",
        "idle",
        "idling",
        "standing",
        "waiting",
        "spawn",
        "spawning",
        "sleep",
        "sleeping",
        "rest",
        "resting",
        "sit",
        "sitting",
        "hurt",
        "hit",
        "crit",
        "critical",
        "stun",
        "stunned",
        "dance",
        "dancing",
        "jump",
        "jumping",
        "fly",
        "flying",
        "swim",
        "swimming",
        "emote",
        "wiggle",
        "roar",
        "icon",
        "preview",
    }
    kept = [
        token
        for token in parts
        if token not in removable and not token.startswith("dead") and not token.isdigit()
    ]
    return "-".join(kept) if kept else re.sub(r"[^a-z0-9]+", "-", pack.casefold()).strip("-")


def classify_state_role(rsi_path: str, state_name: str) -> tuple[StateRole, str]:
    """Conservatively separate whole-entity candidates from compositing layers."""

    parts = PurePosixPath(rsi_path).parts
    try:
        category = parts[parts.index("Mobs") + 1]
    except (ValueError, IndexError):
        return "ambiguous", "path_outside_mobs_taxonomy"
    if category == "Customization":
        return "modular_component", "customization_subtree"
    if category == "Species":
        return "modular_component", "species_parts_organs_or_displacement_subtree"
    if category == "Effects":
        return "effect_or_overlay", "effects_subtree"

    tokens = set(_TOKEN_RE.findall(state_name.casefold()))
    path_tokens = set(_TOKEN_RE.findall(rsi_path.casefold()))
    rsi_stem = PurePosixPath(rsi_path).name.removesuffix(".rsi").casefold()
    if path_tokens & {"displacement", "displacements"}:
        return "modular_component", "displacement_pack_path"
    if path_tokens & {"crack", "cracks"}:
        return "effect_or_overlay", "damage_overlay_pack_path"
    if rsi_stem == "station_ai":
        return "modular_component", "runtime_composites_station_ai_base_and_icon_layers"
    if rsi_stem == "chassis" and re.search(r"_(?:e(?:_r)?|l|rad|crystal)$", state_name.casefold()):
        return "effect_or_overlay", "cyborg_emissive_light_or_module_overlay_suffix"
    if tokens & {"icon", "preview", "inhand", "equipped"}:
        return "icon_or_item_view", "explicit_icon_inhand_or_equipped_token"
    if tokens & {
        "effect",
        "overlay",
        "outline",
        "splat",
        "gib",
        "stunned",
        "damage",
        "glowmask",
    } or any(token.endswith("overlay") for token in tokens):
        return "effect_or_overlay", "explicit_effect_or_overlay_token"
    if tokens & {"extract"}:
        return "icon_or_item_view", "explicit_extract_or_item_token"
    if tokens & {"screen", "frame", "displacement"}:
        return "modular_component", "explicit_screen_frame_or_displacement_token"
    if tokens & {
        "base",
        "eye",
        "eyes",
        "mouth",
        "glow",
        "flare",
        "unshaded",
        "mask",
        "head",
        "arm",
        "hand",
        "leg",
        "foot",
        "torso",
    }:
        return "modular_component", "explicit_layer_or_body_part_token"
    return "complete_entity_candidate", "mob_subtree_without_component_or_view_token"


def _entity_classes(category: str) -> tuple[tuple[str, ...], str]:
    mapping: Mapping[str, tuple[str, ...]] = {
        "Aliens": ("creature",),
        "Animals": ("animal",),
        "Customization": ("humanoid",),
        "Demons": ("monster",),
        "Effects": (),
        "Elemental": ("creature",),
        "Ghosts": ("creature",),
        "Pets": ("animal",),
        "Silicon": ("robot",),
        "Species": ("humanoid",),
    }
    return mapping.get(category, ()), "repository_mobs_category_only"


def extract_upstream_references(copyright_text: str) -> tuple[UpstreamReference, ...]:
    """Parse source URLs without discarding the original copyright text."""

    urls = [match.group(0).rstrip(".,;:)]}") for match in _URL_RE.finditer(copyright_text)]
    free_commits = tuple(
        dict.fromkeys(match.casefold() for match in _COMMIT_RE.findall(copyright_text))
    )
    records: list[UpstreamReference] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold()
        repository: str | None = None
        kind = "web"
        revision: str | None = None
        asset_path: str | None = None
        immutable = False
        lineage_key: str | None = None
        asset_key: str | None = None
        tgstation = False
        if host in {"github.com", "www.github.com"}:
            components = [part for part in parsed.path.split("/") if part]
            if len(components) >= 2:
                repository = f"{components[0]}/{components[1].removesuffix('.git')}"
                tgstation = "tgstation" in repository.casefold()
                if len(components) >= 4 and components[2].casefold() in {"commit", "blob"}:
                    kind = components[2].casefold()
                    revision = components[3]
                    asset_path = "/".join(components[4:]) or None
                elif len(components) >= 4 and components[2].casefold() == "pull":
                    kind = "pull_request"
                elif len(components) == 2:
                    kind = "repository"
                    if tgstation and len(free_commits) == 1:
                        revision = free_commits[0]
                else:
                    kind = "github_path"
                immutable = bool(re.fullmatch(r"[0-9a-f]{40}", revision or "", re.I))
                revision_key = revision.casefold() if revision else "unversioned"
                lineage_key = f"github:{repository.casefold()}@{revision_key}"
                if immutable and asset_path:
                    asset_key = f"{lineage_key}:{asset_path.casefold()}"
        records.append(
            UpstreamReference(
                url=url,
                host=host,
                repository=repository,
                reference_kind=kind,
                revision=revision,
                revision_is_immutable=immutable,
                asset_path=asset_path,
                lineage_key=lineage_key,
                asset_deduplication_key=asset_key,
                tgstation_family=tgstation,
            )
        )
    return tuple(records)


def _rights_status(license_expression: str) -> tuple[str, tuple[str, ...]]:
    if re.search(r"(?:^|-)NC(?:-|$)", license_expression, re.IGNORECASE):
        return "quarantine_noncommercial", ("noncommercial_asset_license",)
    if license_expression not in _KNOWN_LICENSES:
        return "quarantine_unrecognized_license", ("unrecognized_asset_license",)
    if license_expression == "CC0-1.0":
        return "candidate_cc0", ()
    return "candidate_attribution_and_sharealike_review", ()


def _evidence(
    archive: ZipFile,
    member: _ArchiveMember,
    *,
    purpose: str,
) -> EvidenceDocument:
    payload = archive.read(member.member_path)
    return EvidenceDocument(
        logical_path=member.logical_path,
        member_path=member.member_path,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        purpose=purpose,
    )


def _positive_integer(value: object, *, field: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RsiMetadataError(f"{context}: {field} must be a positive integer")
    return value


def _optional_boolean(
    mapping: Mapping[str, Any],
    key: str,
    *,
    default: bool,
    context: str,
) -> bool:
    value = mapping.get(key, default)
    if not isinstance(value, bool):
        raise RsiMetadataError(f"{context}: {key} must be a boolean")
    return value


def _state_delays(
    raw: Mapping[str, Any],
    *,
    directions: int,
    context: str,
) -> tuple[bool, tuple[tuple[float, ...], ...]]:
    declared = "delays" in raw and raw["delays"] is not None
    if not declared:
        return False, tuple((1.0,) for _ in range(directions))
    delay_rows = raw["delays"]
    if not isinstance(delay_rows, list) or len(delay_rows) != directions:
        raise RsiMetadataError(
            f"{context}: delays must contain exactly {directions} direction rows"
        )
    rows: list[tuple[float, ...]] = []
    for delay_row in delay_rows:
        if not isinstance(delay_row, list):
            raise RsiMetadataError(f"{context}: each delay row must be an array")
        if not delay_row:
            rows.append((1.0,))
            continue
        values: list[float] = []
        for raw_value in delay_row:
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise RsiMetadataError(f"{context}: delay is not numeric: {raw_value!r}")
            value = _float32(float(raw_value))
            if not math.isfinite(value) or value <= 0:
                raise RsiMetadataError(f"{context}: delay must be finite and positive")
            values.append(value)
        rows.append(tuple(values))
    return True, tuple(rows)


def _read_state_image(
    archive: ZipFile,
    member: _ArchiveMember,
    *,
    frame_width: int,
    frame_height: int,
    expected_cells: int,
) -> tuple[StateImageEvidence | None, tuple[str, ...]]:
    payload = archive.read(member.member_path)
    try:
        with Image.open(io.BytesIO(payload)) as image:
            detected_format = image.format or "unknown"
            image_mode = image.mode
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return None, (f"unreadable_state_image:{type(error).__name__}",)
    if width % frame_width or height % frame_height:
        return None, ("state_image_not_multiple_of_frame_size",)
    columns = width // frame_width
    rows = height // frame_height
    capacity = columns * rows
    evidence = StateImageEvidence(
        logical_path=member.logical_path,
        member_path=member.member_path,
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        declared_extension=PurePosixPath(member.logical_path).suffix.casefold(),
        detected_format=detected_format,
        image_mode=image_mode,
        width=width,
        height=height,
        grid_columns=columns,
        grid_rows=rows,
        grid_capacity=capacity,
        unused_cell_count=max(0, capacity - expected_cells),
    )
    if capacity < expected_cells:
        return evidence, ("state_image_has_too_few_cells",)
    return evidence, ()


def _role_summary(states: Sequence[RsiState]) -> str:
    roles = {state.role for state in states}
    if roles == {"modular_component"}:
        return "modular_component_pack"
    if roles == {"effect_or_overlay"}:
        return "effect_or_overlay_pack"
    if roles <= {"complete_entity_candidate", "icon_or_item_view"} and (
        "complete_entity_candidate" in roles
    ):
        return "complete_entity_pack_candidate"
    if len(roles) > 1:
        return "mixed_roles"
    return next(iter(roles), "empty_pack")


def _parse_pack(
    archive: ZipFile,
    meta_member: _ArchiveMember,
    by_logical: Mapping[str, _ArchiveMember],
) -> RsiPack:
    metadata_payload = archive.read(meta_member.member_path)
    try:
        metadata = json.loads(metadata_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RsiMetadataError(f"{meta_member.logical_path}: invalid JSON") from error
    if not isinstance(metadata, dict):
        raise RsiMetadataError(f"{meta_member.logical_path}: metadata must be an object")
    context = meta_member.logical_path
    version = _positive_integer(metadata.get("version"), field="version", context=context)
    if version != 1:
        raise RsiMetadataError(f"{context}: unsupported RSI version {version}")
    size = metadata.get("size")
    if not isinstance(size, dict):
        raise RsiMetadataError(f"{context}: size must be an object")
    frame_width = _positive_integer(size.get("x"), field="size.x", context=context)
    frame_height = _positive_integer(size.get("y"), field="size.y", context=context)
    license_expression = metadata.get("license")
    copyright_text = metadata.get("copyright")
    if not isinstance(license_expression, str) or not license_expression.strip():
        raise RsiMetadataError(f"{context}: license must be a non-empty string")
    if not isinstance(copyright_text, str) or not copyright_text.strip():
        raise RsiMetadataError(f"{context}: copyright must be a non-empty string")
    raw_states = metadata.get("states")
    if not isinstance(raw_states, list):
        raise RsiMetadataError(f"{context}: states must be an array")
    names: list[str] = []
    for raw_state in raw_states:
        if not isinstance(raw_state, dict):
            raise RsiMetadataError(f"{context}: state must be an object")
        name = raw_state.get("name")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise RsiMetadataError(f"{context}: unsafe or empty state name {name!r}")
        names.append(name)
    if len(names) != len(set(names)):
        raise RsiMetadataError(f"{context}: duplicate state names")

    rsi_path = meta_member.logical_path.removesuffix("/meta.json")
    path_parts = PurePosixPath(rsi_path).parts
    category = path_parts[3] if len(path_parts) > 3 else "unknown"
    entity_classes, entity_basis = _entity_classes(category)
    rights_status, pack_quarantine = _rights_status(license_expression)
    states: list[RsiState] = []
    for raw_state, state_name in zip(raw_states, names, strict=True):
        state_context = f"{context}:{state_name}"
        direction_count = raw_state.get("directions", 1)
        if (
            isinstance(direction_count, bool)
            or not isinstance(direction_count, int)
            or direction_count not in {1, 4, 8}
        ):
            raise RsiMetadataError(f"{state_context}: directions must be 1, 4, or 8")
        delays_declared, source_delays = _state_delays(
            raw_state,
            directions=direction_count,
            context=state_context,
        )
        engine_delays, engine_indices = fold_direction_delays(source_delays)
        expected_cells = sum(len(row) for row in source_delays)
        image_path = f"{rsi_path}/{state_name}.png"
        image_member = by_logical.get(image_path)
        image: StateImageEvidence | None = None
        image_quarantine: tuple[str, ...]
        if image_member is None:
            image_quarantine = ("missing_state_image",)
        else:
            image, image_quarantine = _read_state_image(
                archive,
                image_member,
                frame_width=frame_width,
                frame_height=frame_height,
                expected_cells=expected_cells,
            )
        frames: list[FrameCell] = []
        source_cell = 0
        capacity = image.grid_capacity if image is not None else 0
        columns = image.grid_columns if image is not None else 1
        for direction_index, row in enumerate(source_delays):
            for frame_index, delay in enumerate(row):
                if source_cell < capacity:
                    column = source_cell % columns
                    grid_row = source_cell // columns
                    left = column * frame_width
                    top = grid_row * frame_height
                    frames.append(
                        FrameCell(
                            source_cell_index=source_cell,
                            direction_index=direction_index,
                            direction=_DIRECTION_NAMES[direction_index],
                            frame_index_in_direction=frame_index,
                            delay_seconds=delay,
                            left=left,
                            top=top,
                            right=left + frame_width,
                            bottom=top + frame_height,
                        )
                    )
                source_cell += 1
        action, action_basis = normalize_state_action(state_name)
        role, role_basis = classify_state_role(rsi_path, state_name)
        quarantine = tuple(dict.fromkeys((*pack_quarantine, *image_quarantine)))
        animated = len(engine_delays) > 1
        exclusions: list[str] = []
        if role != "complete_entity_candidate":
            exclusions.append(f"state_role:{role}")
        if quarantine:
            exclusions.extend(quarantine)
        if action is None:
            exclusions.append("unmapped_action")
        if not animated:
            exclusions.append("single_engine_timeline_frame")
        geometry_safe = image is not None and image.grid_capacity >= expected_cells
        complete_eligible = (
            role == "complete_entity_candidate" and not pack_quarantine and geometry_safe
        )
        animated_action_eligible = complete_eligible and animated and action is not None
        states.append(
            RsiState(
                name=state_name,
                entity_cue=_entity_cue(rsi_path, state_name),
                normalized_action=action,
                normalized_action_basis=action_basis,
                role=role,
                role_basis=role_basis,
                direction_count=direction_count,
                direction_names=_DIRECTION_NAMES[:direction_count],
                delays_declared=delays_declared,
                source_delays_seconds=source_delays,
                engine_delays_seconds=engine_delays,
                engine_source_cell_indices=engine_indices,
                source_frames=tuple(frames),
                image=image,
                expected_source_cell_count=expected_cells,
                is_animated=animated,
                loop_semantics="not_encoded_in_rsi_caller_controls_playback",
                eligible_complete_entity_sequence=complete_eligible,
                eligible_animated_action_sequence=animated_action_eligible,
                quarantine_reasons=quarantine,
                selection_exclusion_reasons=tuple(dict.fromkeys(exclusions)),
            )
        )
    state_tuple = tuple(states)
    direct_images = {
        path
        for path in by_logical
        if path.startswith(rsi_path + "/")
        and "/" not in path[len(rsi_path) + 1 :]
        and path.casefold().endswith(".png")
    }
    declared_images = {f"{rsi_path}/{name}.png" for name in names}
    load = metadata.get("load")
    if load is not None and not isinstance(load, dict):
        raise RsiMetadataError(f"{context}: load must be an object")
    load_srgb = (
        True
        if load is None
        else _optional_boolean(load, "srgb", default=True, context=f"{context}:load")
    )
    return RsiPack(
        logical_path=rsi_path,
        category=category,
        entity_class_candidates=entity_classes,
        entity_class_basis=entity_basis,
        version=version,
        frame_width=frame_width,
        frame_height=frame_height,
        license_expression=license_expression,
        copyright=copyright_text,
        rights_status=rights_status,
        metadata_evidence=EvidenceDocument(
            logical_path=meta_member.logical_path,
            member_path=meta_member.member_path,
            sha256=_sha256_bytes(metadata_payload),
            size_bytes=len(metadata_payload),
            purpose="per_rsi_license_copyright_geometry_and_timing",
        ),
        upstream_references=extract_upstream_references(copyright_text),
        load_srgb=load_srgb,
        meta_atlas=_optional_boolean(
            metadata,
            "metaAtlas",
            default=True,
            context=context,
        ),
        rsic=_optional_boolean(metadata, "rsic", default=True, context=context),
        states=state_tuple,
        declared_state_names=tuple(names),
        extra_image_members=tuple(sorted(direct_images - declared_images)),
        role_summary=_role_summary(state_tuple),
        quarantine_reasons=pack_quarantine,
    )


def _sorted_counts(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counter.items()))


def _audit_payload_without_hash(
    *,
    archive_sha256: str,
    archive_size_bytes: int,
    archive_root: str,
    counts: Ss14ArchiveCounts,
    packs: tuple[RsiPack, ...],
    rights: RightsAudit,
    classification_evidence: tuple[EvidenceDocument, ...],
    issues: tuple[AuditIssue, ...],
) -> dict[str, Any]:
    return {
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive_size_bytes,
        "repository_url": SS14_REPOSITORY_URL,
        "commit": SS14_COMMIT,
        "commit_url": SS14_COMMIT_URL,
        "archive_url": SS14_ARCHIVE_URL,
        "archive_root": archive_root,
        "robust_toolbox_commit": ROBUST_TOOLBOX_COMMIT,
        "engine_evidence_urls": [
            f"{ROBUST_TOOLBOX_REPOSITORY_URL}/blob/{ROBUST_TOOLBOX_COMMIT}/"
            "Robust.Shared/Resources/RsiLoading.cs",
            f"{ROBUST_TOOLBOX_REPOSITORY_URL}/blob/{ROBUST_TOOLBOX_COMMIT}/"
            "Robust.Client/ResourceManagement/ResourceTypes/RSIResource.cs",
            f"{ROBUST_TOOLBOX_REPOSITORY_URL}/blob/{ROBUST_TOOLBOX_COMMIT}/"
            "Robust.Shared/Graphics/RSI/RsiDirection.cs",
        ],
        "counts": asdict(counts),
        "packs": [asdict(pack) for pack in packs],
        "rights": asdict(rights),
        "classification_evidence": [asdict(document) for document in classification_evidence],
        "issues": [asdict(issue) for issue in issues],
        "projection_policy": [
            "retain every pack and state in the audit inventory",
            "quarantine every license expression containing the SPDX NC component",
            "project complete-entity candidates separately from customization, species-part, "
            "effect, overlay, icon, in-hand, and equipment layers",
            "require a decoded immutable image with enough cells before frame projection",
            "preserve source direction timing and the exact folded RobustToolbox timeline",
            "treat filename action mappings as conservative hints, never ground-truth behavior",
            "leave loop versus one-shot semantics unspecified because RSI metadata does not "
            "encode it",
            "retain upstream URLs, immutable revisions, lineage keys, and pixel hashes for "
            "deduplication",
        ],
    }


def audit_ss14_archive(
    archive_path: Path,
    *,
    archive_sha256: str | None = None,
) -> Ss14ArchiveAudit:
    """Audit the ``Resources/Textures/Mobs`` RSI slice without extraction."""

    archive_path = Path(archive_path)
    digest = archive_sha256 or _sha256_file(archive_path)
    try:
        with ZipFile(archive_path) as archive:
            root, members = _validate_members(archive.infolist())
            by_logical = {member.logical_path: member for member in members}
            meta_members = tuple(
                sorted(
                    (
                        member
                        for member in members
                        if member.logical_path.startswith(_MOBS_PREFIX)
                        and member.logical_path.endswith(_META_SUFFIX)
                    ),
                    key=lambda member: member.logical_path,
                )
            )
            if not meta_members:
                raise Ss14ArchiveError("archive contains no Mobs RSI metadata")
            rsi_directories = {
                logical_path.partition(".rsi/")[0] + ".rsi"
                for logical_path in by_logical
                if logical_path.startswith(_MOBS_PREFIX) and ".rsi/" in logical_path
            }
            metadata_rsi_directories = {
                member.logical_path.removesuffix("/meta.json") for member in meta_members
            }
            packs = tuple(_parse_pack(archive, member, by_logical) for member in meta_members)

            license_member = by_logical.get("LICENSE.TXT")
            if license_member is None:
                raise Ss14ArchiveError("archive lacks root LICENSE.TXT")
            schema_member = by_logical.get(".github/rsi-schema.json")
            rights = RightsAudit(
                repository_license_expression="MIT",
                repository_license_scope="code_and_repository_default_not_per_RSI_art_override",
                root_license=_evidence(
                    archive,
                    license_member,
                    purpose="repository_root_license",
                ),
                rsi_schema=(
                    _evidence(archive, schema_member, purpose="repository_RSI_schema")
                    if schema_member is not None
                    else None
                ),
                per_pack_rights_required=True,
                noncommercial_policy="inventory_only_quarantine_from_training_projection",
            )
            classification_evidence = tuple(
                _evidence(
                    archive,
                    by_logical[path],
                    purpose="complete_entity_versus_runtime_component_role",
                )
                for path in _CLASSIFICATION_EVIDENCE_PATHS
                if path in by_logical
            )
            states = tuple(state for pack in packs for state in pack.states)
            images = tuple(state.image for state in states if state.image is not None)
            category_counts = Counter(pack.category for pack in packs)
            license_counts = Counter(pack.license_expression for pack in packs)
            action_counts = Counter(state.normalized_action or "unknown" for state in states)
            role_counts = Counter(state.role for state in states)
            format_counts = Counter(image.detected_format for image in images)
            image_hash_counts = Counter(image.sha256 for image in images)
            tg_packs = tuple(
                pack
                for pack in packs
                if any(reference.tgstation_family for reference in pack.upstream_references)
            )
            counts = Ss14ArchiveCounts(
                archive_members=len(archive.infolist()),
                archive_files=len(members),
                mob_rsi_packs=len(packs),
                rsi_directories_without_meta=len(rsi_directories - metadata_rsi_directories),
                states=len(states),
                expected_source_cells=sum(state.expected_source_cell_count for state in states),
                decoded_source_cells=sum(len(state.source_frames) for state in states),
                engine_timeline_occurrences=sum(
                    len(state.engine_delays_seconds) * state.direction_count for state in states
                ),
                animated_states=sum(state.is_animated for state in states),
                directional_states=sum(state.direction_count > 1 for state in states),
                directional_animated_states=sum(
                    state.direction_count > 1 and state.is_animated for state in states
                ),
                normalized_action_states=sum(
                    state.normalized_action is not None for state in states
                ),
                eligible_complete_entity_sequences=sum(
                    state.eligible_complete_entity_sequence for state in states
                ),
                eligible_animated_action_sequences=sum(
                    state.eligible_animated_action_sequence for state in states
                ),
                noncommercial_packs=sum(
                    "noncommercial_asset_license" in pack.quarantine_reasons for pack in packs
                ),
                noncommercial_states=sum(
                    "noncommercial_asset_license" in state.quarantine_reasons for state in states
                ),
                tgstation_family_packs=len(tg_packs),
                tgstation_immutable_revision_packs=sum(
                    any(
                        reference.tgstation_family and reference.revision_is_immutable
                        for reference in pack.upstream_references
                    )
                    for pack in tg_packs
                ),
                exact_capacity_images=sum(
                    state.image is not None
                    and state.image.grid_capacity == state.expected_source_cell_count
                    for state in states
                ),
                surplus_capacity_images=sum(image.unused_cell_count > 0 for image in images),
                unused_source_cells=sum(image.unused_cell_count for image in images),
                missing_images=sum(
                    "missing_state_image" in state.quarantine_reasons for state in states
                ),
                invalid_or_short_images=sum(
                    any(
                        reason.startswith("unreadable_state_image")
                        or reason
                        in {
                            "state_image_not_multiple_of_frame_size",
                            "state_image_has_too_few_cells",
                        }
                        for reason in state.quarantine_reasons
                    )
                    for state in states
                ),
                duplicate_image_hash_groups=sum(value > 1 for value in image_hash_counts.values()),
                duplicate_image_hash_excess=sum(
                    value - 1 for value in image_hash_counts.values() if value > 1
                ),
                undeclared_state_images=sum(len(pack.extra_image_members) for pack in packs),
                srgb_false_packs=sum(not pack.load_srgb for pack in packs),
                meta_atlas_false_packs=sum(not pack.meta_atlas for pack in packs),
                category_counts=_sorted_counts(category_counts),
                license_counts=_sorted_counts(license_counts),
                action_counts=_sorted_counts(action_counts),
                role_counts=_sorted_counts(role_counts),
                image_format_counts=_sorted_counts(format_counts),
            )
            mutable_tg = sum(
                any(
                    reference.tgstation_family and not reference.revision_is_immutable
                    for reference in pack.upstream_references
                )
                for pack in packs
            )
            issues = (
                AuditIssue(
                    "rsi_directory_without_meta",
                    counts.rsi_directories_without_meta,
                    "RSI-looking directories without metadata are not decoded as packs.",
                ),
                AuditIssue(
                    "undeclared_state_image",
                    counts.undeclared_state_images,
                    "Direct .png members not named by metadata are retained as pack extras only.",
                ),
                AuditIssue(
                    "noncommercial_pack_quarantine",
                    counts.noncommercial_packs,
                    "Packs whose per-RSI SPDX expression contains NC remain inventory-only.",
                ),
                AuditIssue(
                    "surplus_image_cells",
                    counts.surplus_capacity_images,
                    "The engine consumes only the metadata-declared prefix; surplus grid cells "
                    "are recorded but not projected.",
                ),
                AuditIssue(
                    "non_png_payload_with_png_name",
                    sum(image.detected_format != "PNG" for image in images),
                    "ImageSharp detects payload format instead of trusting the .png filename.",
                ),
                AuditIssue(
                    "mutable_or_unversioned_tgstation_reference",
                    mutable_tg,
                    "These lineage links aid review but are not immutable asset-level dedup keys.",
                ),
                AuditIssue(
                    "unmapped_state_action",
                    action_counts["unknown"],
                    "State names outside the explicit vocabulary retain no guessed action label.",
                ),
                AuditIssue(
                    "duplicate_state_image_payload",
                    counts.duplicate_image_hash_excess,
                    "Exact image-byte duplicates require identity-aware downstream deduplication.",
                ),
            )
            payload = _audit_payload_without_hash(
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                archive_root=root,
                counts=counts,
                packs=packs,
                rights=rights,
                classification_evidence=classification_evidence,
                issues=issues,
            )
            record_hash = _sha256_bytes(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            return Ss14ArchiveAudit(
                archive_sha256=digest,
                archive_size_bytes=archive_path.stat().st_size,
                repository_url=SS14_REPOSITORY_URL,
                commit=SS14_COMMIT,
                commit_url=SS14_COMMIT_URL,
                archive_url=SS14_ARCHIVE_URL,
                archive_root=root,
                robust_toolbox_commit=ROBUST_TOOLBOX_COMMIT,
                engine_evidence_urls=tuple(payload["engine_evidence_urls"]),
                counts=counts,
                packs=packs,
                rights=rights,
                classification_evidence=classification_evidence,
                issues=issues,
                projection_policy=tuple(payload["projection_policy"]),
                audit_record_sha256=record_hash,
            )
    except BadZipFile as error:
        raise Ss14ArchiveError(f"not a valid ZIP archive: {archive_path}") from error


def audit_known_ss14_archive(archive_path: Path) -> Ss14ArchiveAudit:
    """Hash-check and audit the exact pinned Space Station 14 snapshot."""

    archive_path = Path(archive_path)
    digest = _sha256_file(archive_path)
    if digest != EXPECTED_SS14_ARCHIVE_SHA256:
        raise Ss14ArchiveError(
            f"SS14 archive SHA-256 mismatch: expected {EXPECTED_SS14_ARCHIVE_SHA256}, got {digest}"
        )
    audit = audit_ss14_archive(archive_path, archive_sha256=digest)
    if audit.archive_root != _EXPECTED_ROOT:
        raise Ss14ArchiveError(
            f"SS14 archive root mismatch: expected {_EXPECTED_ROOT!r}, got {audit.archive_root!r}"
        )
    return audit


def known_ss14_cas_path(raw_root: Path) -> Path:
    digest = EXPECTED_SS14_ARCHIVE_SHA256
    return Path(raw_root) / "objects" / "sha256" / digest[:2] / digest[2:4] / digest


def known_robust_toolbox_cas_path(raw_root: Path) -> Path:
    digest = EXPECTED_ROBUST_TOOLBOX_ARCHIVE_SHA256
    return Path(raw_root) / "objects" / "sha256" / digest[:2] / digest[2:4] / digest


__all__ = [
    "EXPECTED_SS14_ARCHIVE_SHA256",
    "EXPECTED_ROBUST_TOOLBOX_ARCHIVE_SHA256",
    "ROBUST_TOOLBOX_ARCHIVE_URL",
    "ROBUST_TOOLBOX_COMMIT",
    "SS14_ARCHIVE_URL",
    "SS14_COMMIT",
    "SS14_COMMIT_URL",
    "SS14_REPOSITORY_URL",
    "AuditIssue",
    "EvidenceDocument",
    "FrameCell",
    "RightsAudit",
    "RsiMetadataError",
    "RsiPack",
    "RsiState",
    "Ss14ArchiveAudit",
    "Ss14ArchiveCounts",
    "Ss14ArchiveError",
    "StateImageEvidence",
    "UpstreamReference",
    "audit_known_ss14_archive",
    "audit_ss14_archive",
    "classify_state_role",
    "extract_upstream_references",
    "fold_direction_delays",
    "known_ss14_cas_path",
    "known_robust_toolbox_cas_path",
    "normalize_state_action",
]
