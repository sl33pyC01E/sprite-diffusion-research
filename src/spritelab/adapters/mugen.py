"""Pure, non-executing interpretation of M.U.G.E.N character packages.

M.U.G.E.N characters mix declarative media files (DEF, AIR, SFF, ACT) with
runtime program files (CMD/CNS and, in some community packages, native code).
This module intentionally interprets only the declarative identity and
animation layers.  It never imports or executes character logic.
"""

from __future__ import annotations

import configparser
import hashlib
import io
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from PIL import Image

from spritelab.archive import ArchiveLimits, ZipManifest, ZipMember, inspect_zip

_ACTION_HEADER = re.compile(r"^\[\s*begin\s+action\s+(-?\d+)\s*\]$", re.IGNORECASE)
_COLLISION = re.compile(r"^clsn([12])(?:default)?\s*:\s*(\d+)\s*$", re.IGNORECASE)
_INTEGER = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

# Exact standard/reserved actions from Elecbyte's AIR standard.  Broad numeric
# attack ranges are handled separately and clearly identified as range-based.
_EXACT_ACTIONS: dict[int, tuple[str, str]] = {
    0: ("idle", "standing"),
    5: ("idle", "stand_turning"),
    6: ("idle", "crouch_turning"),
    10: ("idle", "stand_to_crouch"),
    11: ("idle", "crouching"),
    12: ("idle", "crouch_to_stand"),
    20: ("walk", "walking_forwards"),
    21: ("walk", "walking_backwards"),
    40: ("jump", "jump_start"),
    41: ("jump", "jump_neutral"),
    42: ("jump", "jump_forwards"),
    43: ("jump", "jump_backwards"),
    44: ("jump", "jump_land"),
    100: ("run", "run_forwards"),
    105: ("jump", "hop_backwards"),
    120: ("block", "guard_start_standing"),
    121: ("block", "guard_start_crouching"),
    122: ("block", "guard_start_air"),
    130: ("block", "guard_standing"),
    131: ("block", "guard_crouching"),
    132: ("block", "guard_air"),
    140: ("block", "guard_end_standing"),
    141: ("block", "guard_end_crouching"),
    142: ("block", "guard_end_air"),
    170: ("hurt", "lose"),
    175: ("hurt", "time_over"),
    180: ("emote", "win"),
    190: ("spawn", "intro"),
    5040: ("hurt", "air_recover"),
    5050: ("hurt", "air_fall"),
    5070: ("hurt", "tripped"),
    5080: ("hurt", "lie_down_hit"),
    5100: ("hurt", "hit_ground_from_fall"),
    5110: ("death", "lie_down"),
    5120: ("idle", "get_up"),
    5140: ("death", "lie_dead_first_rounds"),
    5150: ("death", "lie_dead_final_round"),
    5300: ("hurt", "dizzy"),
}

_EXECUTABLE_EXTENSIONS = frozenset(
    {".bat", ".cmd.exe", ".com", ".dll", ".exe", ".jar", ".lnk", ".msi", ".ps1", ".scr"}
)
_RUNTIME_LOGIC_EXTENSIONS = frozenset({".cmd", ".cns", ".st", ".zss"})
_DECLARATIVE_EXTENSIONS = frozenset({".act", ".air", ".def", ".sff"})


@dataclass(frozen=True)
class MugenCharacterDefinition:
    """Identity and referenced files from one character DEF."""

    name: str | None
    display_name: str | None
    author: str | None
    version_date: str | None
    mugen_version: str | None
    local_coord: tuple[int, int] | None
    palette_defaults: tuple[int, ...]
    files: tuple[tuple[str, str], ...]
    source_comments: tuple[str, ...]

    def file(self, key: str) -> str | None:
        wanted = key.casefold()
        return next((value for name, value in self.files if name == wanted), None)


@dataclass(frozen=True)
class MugenActionLabel:
    normalized_action: str | None
    source_meaning: str | None
    method: str


@dataclass(frozen=True)
class MugenAirElement:
    sprite_group: int
    sprite_image: int
    x_offset: float
    y_offset: float
    duration_ticks: int
    duration_seconds: float | None
    horizontal_flip: bool
    vertical_flip: bool
    optional_tokens: tuple[str, ...]
    source_line: int


@dataclass(frozen=True)
class MugenAirAction:
    action_number: int
    label: MugenActionLabel
    elements: tuple[MugenAirElement, ...]
    loop_start_index: int | None
    loop_mode: str
    finite_duration_ticks: int
    collision_1_declarations: int
    collision_2_declarations: int
    source_comments: tuple[str, ...]


@dataclass(frozen=True)
class MugenSffHeader:
    signature: str
    version_bytes: tuple[int, int, int, int]
    format_family: str
    file_bytes: int
    sha256: str


@dataclass(frozen=True)
class MugenSffV1Sprite:
    """One fully decoded SFF v1 sprite with exact palette-index lineage."""

    archive_index: int
    group_number: int
    image_number: int
    axis_x: int
    axis_y: int
    width: int
    height: int
    linked_sprite_index: int | None
    palette_reuse: bool
    indices: bytes
    palette_rgb: bytes
    rgba: bytes
    indices_sha256: str
    palette_sha256: str
    rgba_sha256: str


@dataclass(frozen=True)
class MugenCharacterArchiveAudit:
    archive_sha256: str
    archive_bytes: int
    inventory_sha256: str
    member_count: int
    definition_member: str
    definition_members: tuple[str, ...]
    definition_variants: tuple[MugenCharacterDefinition, ...]
    air_member: str
    sff_member: str
    definition: MugenCharacterDefinition
    actions: tuple[MugenAirAction, ...]
    sff_header: MugenSffHeader
    executable_members: tuple[str, ...]
    runtime_logic_members: tuple[str, ...]
    declarative_members: tuple[str, ...]
    unclassified_members: tuple[str, ...]


@dataclass(frozen=True)
class MugenActionFrame:
    ordinal: int
    sprite_group: int
    sprite_image: int
    duration_ticks: int
    source_line: int
    source_rgba_sha256: str
    world_left: int
    world_top: int
    width: int
    height: int
    horizontal_flip: bool
    vertical_flip: bool
    rgba: bytes
    rgba_sha256: str


@dataclass(frozen=True)
class MugenActionMaterialization:
    action_number: int
    normalized_action: str | None
    source_meaning: str | None
    loop_mode: str
    loop_start_index: int | None
    canvas_world_left: int
    canvas_world_top: int
    canvas_width: int
    canvas_height: int
    frames: tuple[MugenActionFrame, ...]


@dataclass(frozen=True)
class MugenActionExclusion:
    action_number: int
    reason: Literal[
        "empty_action",
        "missing_sprite",
        "ambiguous_sprite_key",
        "non_integral_offset",
        "unsupported_air_transform",
    ]
    detail: str


@dataclass(frozen=True)
class MugenActionPlan:
    admitted: tuple[MugenActionMaterialization, ...]
    excluded: tuple[MugenActionExclusion, ...]


def decode_mugen_text(payload: bytes) -> str:
    """Decode common legacy M.U.G.E.N text without changing its content."""

    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16")
    if payload.startswith(b"\xef\xbb\xbf"):
        return payload.decode("utf-8-sig")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return payload.decode("cp1252")
        except UnicodeDecodeError:
            # Legacy Japanese/Asian packs often contain undefined Windows-1252
            # control bytes in comments. Latin-1 is the only lossless 1:1 text
            # fallback; semantic claims remain verbatim and visibly unnormalized.
            return payload.decode("latin-1")


def parse_character_def(payload: bytes | str) -> MugenCharacterDefinition:
    """Parse the conservative Info/Files subset of a character DEF."""

    text = decode_mugen_text(payload) if isinstance(payload, bytes) else payload
    comments = tuple(
        stripped[1:].strip()
        for line in text.splitlines()
        if (stripped := line.strip()).startswith(";") and stripped[1:].strip()
    )
    parser = configparser.RawConfigParser(
        interpolation=None,
        strict=False,
        delimiters=("=",),
        comment_prefixes=(";",),
        inline_comment_prefixes=(";",),
    )
    parser.optionxform = str.casefold
    lines = text.splitlines()
    first_section = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("[")), None
    )
    parse_text = "\n".join(lines[first_section:]) if first_section is not None else text
    try:
        parser.read_string(parse_text)
    except configparser.Error as exc:
        raise ValueError(f"invalid M.U.G.E.N DEF: {exc}") from exc

    info = _section(parser, "info")
    files = _section(parser, "files")
    file_rows = tuple(sorted((key, _unquote(value.strip())) for key, value in files.items()))
    return MugenCharacterDefinition(
        name=_optional_value(info, "name"),
        display_name=_optional_value(info, "displayname"),
        author=_optional_value(info, "author"),
        version_date=_optional_value(info, "versiondate"),
        mugen_version=_optional_value(info, "mugenversion"),
        local_coord=_pair_of_ints(_optional_value(info, "localcoord")),
        palette_defaults=_int_tuple(_optional_value(info, "pal.defaults")),
        files=file_rows,
        source_comments=comments,
    )


def parse_air(
    payload: bytes | str,
    *,
    reject_duplicate_actions: bool = True,
) -> tuple[MugenAirAction, ...]:
    """Parse ordered sprite references, timing, transforms, and loop points."""

    text = decode_mugen_text(payload) if isinstance(payload, bytes) else payload
    actions: list[MugenAirAction] = []
    number: int | None = None
    elements: list[MugenAirElement] = []
    loop_start: int | None = None
    collision_1 = 0
    collision_2 = 0
    comments: list[str] = []
    pending_comments: list[str] = []

    def finish() -> None:
        nonlocal number, elements, loop_start, collision_1, collision_2, comments
        if number is None:
            return
        if not elements:
            loop_mode = "empty"
        elif elements[-1].duration_ticks == -1:
            loop_mode = "terminal_hold"
        elif loop_start is not None:
            loop_mode = "intro_then_loop"
        else:
            loop_mode = "loop"
        actions.append(
            MugenAirAction(
                action_number=number,
                label=label_action_number(number),
                elements=tuple(elements),
                loop_start_index=loop_start,
                loop_mode=loop_mode,
                finite_duration_ticks=sum(max(0, row.duration_ticks) for row in elements),
                collision_1_declarations=collision_1,
                collision_2_declarations=collision_2,
                source_comments=tuple(comments),
            )
        )
        number = None
        elements = []
        loop_start = None
        collision_1 = collision_2 = 0
        comments = []

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        content, comment = _split_comment(raw_line)
        stripped = content.strip()
        if comment:
            pending_comments.append(comment)
        if not stripped:
            continue
        header = _ACTION_HEADER.match(stripped)
        if header:
            finish()
            number = int(header.group(1))
            comments = pending_comments
            pending_comments = []
            continue
        if number is None:
            continue
        if stripped.casefold() == "loopstart":
            if loop_start is not None:
                raise ValueError(f"action {number} has multiple Loopstart lines")
            loop_start = len(elements)
            continue
        collision = _COLLISION.match(stripped)
        if collision:
            count = int(collision.group(2))
            if collision.group(1) == "1":
                collision_1 += count
            else:
                collision_2 += count
            continue
        if stripped.casefold().startswith(("clsn1[", "clsn2[", "interpolate ")):
            continue
        if "," not in stripped:
            continue
        fields = tuple(value.strip() for value in stripped.split(","))
        if len(fields) < 5 or not all(_INTEGER.match(value) for value in fields[:2]):
            continue
        if not all(_FLOAT.match(value) for value in fields[2:4]) or not _INTEGER.match(fields[4]):
            raise ValueError(f"invalid AIR element at line {line_number}: {raw_line!r}")
        duration = int(fields[4])
        if duration < -1:
            raise ValueError(f"invalid AIR duration at line {line_number}: {duration}")
        optional = tuple(value for value in fields[5:] if value)
        flags = optional[0].casefold() if optional else ""
        elements.append(
            MugenAirElement(
                sprite_group=int(fields[0]),
                sprite_image=int(fields[1]),
                x_offset=float(fields[2]),
                y_offset=float(fields[3]),
                duration_ticks=duration,
                duration_seconds=None if duration == -1 else duration / 60.0,
                horizontal_flip="h" in flags,
                vertical_flip="v" in flags,
                optional_tokens=optional,
                source_line=line_number,
            )
        )
    finish()
    numbers = [action.action_number for action in actions]
    if reject_duplicate_actions and len(set(numbers)) != len(numbers):
        raise ValueError("AIR contains duplicate action numbers")
    return tuple(actions)


def label_action_number(number: int) -> MugenActionLabel:
    """Return a conservative standard/range label without reading runtime CNS."""

    if number in _EXACT_ACTIONS:
        action, meaning = _EXACT_ACTIONS[number]
        return MugenActionLabel(action, meaning, "elecbyte_standard_exact")
    if 181 <= number <= 189:
        return MugenActionLabel("emote", "alternate_win", "elecbyte_standard_range")
    if 191 <= number <= 199:
        return MugenActionLabel("spawn", "alternate_intro", "elecbyte_standard_range")
    if 200 <= number <= 799:
        return MugenActionLabel("attack", "recommended_attack_range", "elecbyte_recommended_range")
    if 1000 <= number <= 4999:
        kind = "special_attack" if number < 3000 else "hyper_attack"
        return MugenActionLabel("attack", kind, "elecbyte_recommended_range")
    return MugenActionLabel(None, None, "unmapped_numeric_action")


def inspect_sff_header(payload: bytes) -> MugenSffHeader:
    """Inspect stable SFF identity/version bytes without decoding sprites."""

    if len(payload) < 16:
        raise ValueError("SFF payload is shorter than its header")
    signature_bytes = payload[:12]
    if not signature_bytes.startswith(b"ElecbyteSpr"):
        raise ValueError(f"unsupported SFF signature: {signature_bytes!r}")
    version = tuple(payload[12:16])
    # Version bytes are stored least-significant component first in SFF files.
    family = "sff_v2" if version[3] >= 2 else "sff_v1"
    return MugenSffHeader(
        signature=signature_bytes.rstrip(b"\x00").decode("ascii"),
        version_bytes=version,
        format_family=family,
        file_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def decode_sff_v1(payload: bytes) -> tuple[MugenSffV1Sprite, ...]:
    """Decode SFF v1 PCX sprites and links without invoking external tools.

    Palette index zero is converted to alpha zero according to Elecbyte's sprite
    masking standard. All other palette entries are opaque. The original indexed
    pixels and palette are retained beside RGBA so the conversion is auditable.
    """

    header = inspect_sff_header(payload)
    if header.format_family != "sff_v1":
        raise ValueError("decode_sff_v1 requires an SFF v1 payload")
    if len(payload) < 512:
        raise ValueError("SFF v1 payload is shorter than its 512-byte header")
    _, image_total, first_offset, subheader_bytes = struct.unpack_from("<4I", payload, 16)
    shared_palette = bool(payload[32])
    if subheader_bytes < 32:
        raise ValueError(f"invalid SFF v1 subheader size: {subheader_bytes}")
    if first_offset < 512 or first_offset >= len(payload):
        raise ValueError(f"invalid SFF v1 first subfile offset: {first_offset}")

    sprites: list[MugenSffV1Sprite] = []
    offset = first_offset
    previous_palette: bytes | None = None
    seen_offsets: set[int] = set()
    for archive_index in range(image_total):
        if offset in seen_offsets:
            raise ValueError(f"SFF v1 subfile chain cycles at offset {offset}")
        seen_offsets.add(offset)
        if offset < 0 or offset + 32 > len(payload):
            raise ValueError(f"SFF v1 subfile header {archive_index} is out of bounds")
        next_offset, data_bytes, axis_x, axis_y, group, image, link, reuse = struct.unpack_from(
            "<IIhhhhHB", payload, offset
        )
        record_end = next_offset if next_offset not in {0, len(payload)} else len(payload)
        actual_subheader_bytes = record_end - offset - data_bytes
        # Several real legacy writers incorrectly place 512 in the declared
        # subheader-size field while still emitting the standard 32-byte record.
        # Deriving it from the linked-list boundary is exact and fail-closed.
        if actual_subheader_bytes < 32 or actual_subheader_bytes > subheader_bytes:
            raise ValueError(
                f"SFF v1 sprite {archive_index} has invalid derived subheader size "
                f"{actual_subheader_bytes}"
            )
        data_start = offset + actual_subheader_bytes
        data_end = data_start + data_bytes
        if data_end > len(payload):
            raise ValueError(f"SFF v1 sprite {archive_index} data is out of bounds")

        linked_index: int | None = None
        if data_bytes == 0:
            linked_index = link
            if linked_index >= len(sprites):
                raise ValueError(
                    f"SFF v1 sprite {archive_index} links to unavailable index {linked_index}"
                )
            source = sprites[linked_index]
            width, height = source.width, source.height
            indices, palette = source.indices, source.palette_rgb
        else:
            indices, palette, width, height = _decode_sff_v1_pcx(
                payload[data_start:data_end],
                previous_palette=previous_palette,
                palette_reuse=bool(reuse) or shared_palette,
            )
        previous_palette = palette
        rgba = _indexed_rgba(indices, palette)
        sprites.append(
            MugenSffV1Sprite(
                archive_index=archive_index,
                group_number=group,
                image_number=image,
                axis_x=axis_x,
                axis_y=axis_y,
                width=width,
                height=height,
                linked_sprite_index=linked_index,
                palette_reuse=bool(reuse),
                indices=indices,
                palette_rgb=palette,
                rgba=rgba,
                indices_sha256=hashlib.sha256(indices).hexdigest(),
                palette_sha256=hashlib.sha256(palette).hexdigest(),
                rgba_sha256=hashlib.sha256(rgba).hexdigest(),
            )
        )
        if archive_index + 1 < image_total:
            if next_offset == 0:
                raise ValueError(
                    f"SFF v1 chain ended after {archive_index + 1} of {image_total} sprites"
                )
            offset = next_offset
        elif next_offset not in {0, len(payload)}:
            raise ValueError("SFF v1 final subfile does not terminate at zero or EOF")
    return tuple(sprites)


def audit_character_zip(
    archive_payload: bytes,
    *,
    limits: ArchiveLimits | None = None,
) -> MugenCharacterArchiveAudit:
    """Audit one ZIP character pack entirely in memory and without execution."""

    manifest = inspect_zip(io.BytesIO(archive_payload), limits=limits or ArchiveLimits())
    regular = tuple(member for member in manifest.members if member.is_regular_file)
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        infos = archive.infolist()
        definitions: list[tuple[ZipMember, MugenCharacterDefinition]] = []
        for member in regular:
            if member.extension != ".def":
                continue
            try:
                candidate = parse_character_def(archive.read(infos[member.archive_index]))
            except ValueError:
                continue
            if candidate.file("anim") and candidate.file("sprite"):
                definitions.append((member, candidate))
        if not definitions:
            raise ValueError(
                "expected at least one character DEF with sprite+anim references, found 0"
            )
        definitions.sort(key=lambda row: row[0].normalized_name.encode("utf-8"))
        resolved_variants = [
            (
                member,
                definition,
                _resolve_reference(manifest, member.normalized_name, definition.file("anim")),
                _resolve_reference(manifest, member.normalized_name, definition.file("sprite")),
            )
            for member, definition in definitions
        ]
        air_paths = {row[2].normalized_name.casefold() for row in resolved_variants}
        sff_paths = {row[3].normalized_name.casefold() for row in resolved_variants}
        if len(air_paths) != 1 or len(sff_paths) != 1:
            raise ValueError("character DEF variants do not share one AIR/SFF media pair")
        definition_member, definition, air_member, sff_member = resolved_variants[0]
        actions = parse_air(
            archive.read(infos[air_member.archive_index]), reject_duplicate_actions=False
        )
        sff_header = inspect_sff_header(archive.read(infos[sff_member.archive_index]))

    executable: list[str] = []
    runtime: list[str] = []
    declarative: list[str] = []
    other: list[str] = []
    for member in regular:
        suffix = PurePosixPath(member.normalized_name).suffix.casefold()
        if suffix in _EXECUTABLE_EXTENSIONS or member.normalized_name.casefold().endswith(
            ".cmd.exe"
        ):
            executable.append(member.normalized_name)
        elif suffix in _RUNTIME_LOGIC_EXTENSIONS:
            runtime.append(member.normalized_name)
        elif suffix in _DECLARATIVE_EXTENSIONS:
            declarative.append(member.normalized_name)
        else:
            other.append(member.normalized_name)
    return MugenCharacterArchiveAudit(
        archive_sha256=hashlib.sha256(archive_payload).hexdigest(),
        archive_bytes=len(archive_payload),
        inventory_sha256=manifest.inventory_sha256,
        member_count=len(manifest.members),
        definition_member=definition_member.normalized_name,
        definition_members=tuple(row[0].normalized_name for row in resolved_variants),
        definition_variants=tuple(row[1] for row in resolved_variants),
        air_member=air_member.normalized_name,
        sff_member=sff_member.normalized_name,
        definition=definition,
        actions=actions,
        sff_header=sff_header,
        executable_members=tuple(executable),
        runtime_logic_members=tuple(runtime),
        declarative_members=tuple(declarative),
        unclassified_members=tuple(other),
    )


def materialize_actions(
    actions: tuple[MugenAirAction, ...],
    sprites: tuple[MugenSffV1Sprite, ...],
) -> MugenActionPlan:
    """Render supported AIR actions to aligned, transparent RGBA canvases."""

    by_key: dict[tuple[int, int], list[MugenSffV1Sprite]] = {}
    for sprite in sprites:
        by_key.setdefault((sprite.group_number, sprite.image_number), []).append(sprite)

    admitted: list[MugenActionMaterialization] = []
    excluded: list[MugenActionExclusion] = []
    for action in actions:
        if not action.elements:
            excluded.append(
                MugenActionExclusion(action.action_number, "empty_action", "no elements")
            )
            continue
        resolved: list[tuple[MugenAirElement, MugenSffV1Sprite, int, int, bytes]] = []
        rejection: MugenActionExclusion | None = None
        for element in action.elements:
            key = (element.sprite_group, element.sprite_image)
            candidates = by_key.get(key, [])
            if not candidates:
                rejection = MugenActionExclusion(
                    action.action_number, "missing_sprite", f"missing SFF key {key}"
                )
                break
            unique_hashes = {candidate.rgba_sha256 for candidate in candidates}
            if len(unique_hashes) != 1:
                rejection = MugenActionExclusion(
                    action.action_number,
                    "ambiguous_sprite_key",
                    f"SFF key {key} has {len(unique_hashes)} pixel variants",
                )
                break
            if not element.x_offset.is_integer() or not element.y_offset.is_integer():
                rejection = MugenActionExclusion(
                    action.action_number,
                    "non_integral_offset",
                    f"line {element.source_line} has offset {element.x_offset},{element.y_offset}",
                )
                break
            if len(element.optional_tokens) > 1:
                rejection = MugenActionExclusion(
                    action.action_number,
                    "unsupported_air_transform",
                    f"line {element.source_line} options {element.optional_tokens!r}",
                )
                break
            sprite = candidates[0]
            transformed = _flip_rgba(
                sprite.rgba,
                width=sprite.width,
                height=sprite.height,
                horizontal=element.horizontal_flip,
                vertical=element.vertical_flip,
            )
            axis_x = sprite.width - 1 - sprite.axis_x if element.horizontal_flip else sprite.axis_x
            axis_y = sprite.height - 1 - sprite.axis_y if element.vertical_flip else sprite.axis_y
            left = int(element.x_offset) - axis_x
            top = int(element.y_offset) - axis_y
            resolved.append((element, sprite, left, top, transformed))
        if rejection is not None:
            excluded.append(rejection)
            continue

        world_left = min(row[2] for row in resolved)
        world_top = min(row[3] for row in resolved)
        world_right = max(row[2] + row[1].width for row in resolved)
        world_bottom = max(row[3] + row[1].height for row in resolved)
        canvas_width = world_right - world_left
        canvas_height = world_bottom - world_top
        frames: list[MugenActionFrame] = []
        for ordinal, (element, sprite, left, top, transformed) in enumerate(resolved):
            canvas = bytearray(canvas_width * canvas_height * 4)
            _paste_rgba(
                canvas,
                canvas_width=canvas_width,
                source=transformed,
                source_width=sprite.width,
                source_height=sprite.height,
                left=left - world_left,
                top=top - world_top,
            )
            rgba = bytes(canvas)
            frames.append(
                MugenActionFrame(
                    ordinal=ordinal,
                    sprite_group=element.sprite_group,
                    sprite_image=element.sprite_image,
                    duration_ticks=element.duration_ticks,
                    source_line=element.source_line,
                    source_rgba_sha256=sprite.rgba_sha256,
                    world_left=left,
                    world_top=top,
                    width=sprite.width,
                    height=sprite.height,
                    horizontal_flip=element.horizontal_flip,
                    vertical_flip=element.vertical_flip,
                    rgba=rgba,
                    rgba_sha256=hashlib.sha256(rgba).hexdigest(),
                )
            )
        admitted.append(
            MugenActionMaterialization(
                action_number=action.action_number,
                normalized_action=action.label.normalized_action,
                source_meaning=action.label.source_meaning,
                loop_mode=action.loop_mode,
                loop_start_index=action.loop_start_index,
                canvas_world_left=world_left,
                canvas_world_top=world_top,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                frames=tuple(frames),
            )
        )
    return MugenActionPlan(tuple(admitted), tuple(excluded))


def _section(parser: configparser.RawConfigParser, name: str) -> dict[str, str]:
    section = next((section for section in parser.sections() if section.casefold() == name), None)
    return {} if section is None else dict(parser.items(section, raw=True))


def _optional_value(values: dict[str, str], key: str) -> str | None:
    value = values.get(key.casefold())
    if value is None:
        return None
    normalized = _unquote(value.strip())
    return normalized or None


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _pair_of_ints(value: str | None) -> tuple[int, int] | None:
    values = _int_tuple(value)
    return (values[0], values[1]) if len(values) == 2 else None


def _int_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    result: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if not _INTEGER.match(stripped):
            return ()
        result.append(int(stripped))
    return tuple(result)


def _split_comment(line: str) -> tuple[str, str]:
    quoted: str | None = None
    for index, character in enumerate(line):
        if character in "\"'":
            quoted = None if quoted == character else character if quoted is None else quoted
        elif character == ";" and quoted is None:
            return line[:index], line[index + 1 :].strip()
    return line, ""


def _resolve_reference(
    manifest: ZipManifest,
    definition_path: str,
    raw_reference: str | None,
):
    if not raw_reference:
        raise ValueError("character DEF is missing a required file reference")
    parent = PurePosixPath(definition_path).parent
    reference = str(parent / raw_reference.replace("\\", "/"))
    folded = reference.casefold()
    matches = [
        member
        for member in manifest.members
        if member.is_regular_file and member.normalized_name.casefold() == folded
    ]
    if len(matches) != 1:
        raise ValueError(f"DEF reference {raw_reference!r} resolves to {len(matches)} members")
    return matches[0]


def _decode_sff_v1_pcx(
    payload: bytes,
    *,
    previous_palette: bytes | None,
    palette_reuse: bool,
) -> tuple[bytes, bytes, int, int]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            if image.mode not in {"L", "P"}:
                raise ValueError(f"SFF v1 PCX has unsupported mode {image.mode!r}")
            width, height = image.size
            indices = image.tobytes()
            raw_palette = image.getpalette()
    except (OSError, SyntaxError) as exc:
        raise ValueError(f"invalid SFF v1 PCX payload: {exc}") from exc
    if len(indices) != width * height:
        raise ValueError("SFF v1 PCX decoded pixel count does not match its dimensions")
    if raw_palette is not None:
        palette = bytes(raw_palette[:768])
        if len(palette) < 768:
            palette += b"\x00" * (768 - len(palette))
    elif palette_reuse and previous_palette is not None:
        palette = previous_palette
    else:
        raise ValueError("SFF v1 PCX omits its palette without a reusable predecessor")
    return indices, palette, width, height


def _indexed_rgba(indices: bytes, palette: bytes) -> bytes:
    if len(palette) != 768:
        raise ValueError("indexed RGB palette must contain exactly 256 colors")
    rgba = bytearray(len(indices) * 4)
    for output, index in enumerate(indices):
        palette_offset = index * 3
        rgba_offset = output * 4
        rgba[rgba_offset : rgba_offset + 3] = palette[palette_offset : palette_offset + 3]
        rgba[rgba_offset + 3] = 0 if index == 0 else 255
    return bytes(rgba)


def _flip_rgba(
    rgba: bytes,
    *,
    width: int,
    height: int,
    horizontal: bool,
    vertical: bool,
) -> bytes:
    if len(rgba) != width * height * 4:
        raise ValueError("RGBA byte count does not match dimensions")
    if not horizontal and not vertical:
        return rgba
    output = bytearray(len(rgba))
    for y in range(height):
        source_y = height - 1 - y if vertical else y
        for x in range(width):
            source_x = width - 1 - x if horizontal else x
            source = (source_y * width + source_x) * 4
            destination = (y * width + x) * 4
            output[destination : destination + 4] = rgba[source : source + 4]
    return bytes(output)


def _paste_rgba(
    canvas: bytearray,
    *,
    canvas_width: int,
    source: bytes,
    source_width: int,
    source_height: int,
    left: int,
    top: int,
) -> None:
    for y in range(source_height):
        source_start = y * source_width * 4
        destination_start = ((top + y) * canvas_width + left) * 4
        canvas[destination_start : destination_start + source_width * 4] = source[
            source_start : source_start + source_width * 4
        ]
