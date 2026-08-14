"""Pure, non-executing interpretation of M.U.G.E.N character packages.

M.U.G.E.N characters mix declarative media files (DEF, AIR, SFF, ACT) with
runtime program files (CMD/CNS and, in some community packages, native code).
This module intentionally interprets only the declarative identity and
animation layers.  It never imports or executes character logic.
"""

from __future__ import annotations

import hashlib
import io
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

import numpy as np
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
    41: ("jump", "jump_neutral_up"),
    42: ("jump", "jump_forward_up"),
    43: ("jump", "jump_backward_up"),
    44: ("jump", "jump_neutral_down"),
    45: ("jump", "jump_forward_down"),
    46: ("jump", "jump_backward_down"),
    47: ("jump", "jump_land"),
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
    150: ("block", "guard_hit_standing"),
    151: ("block", "guard_hit_crouching"),
    152: ("block", "guard_hit_air"),
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
    optional_fields: tuple[str, ...]
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
class MugenAirParseExclusion:
    """One malformed AIR element omitted by explicit recovery mode."""

    action_number: int
    line_number: int
    reason: str
    raw_line: str
    detail: str


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
class MugenSffV1DecodeExclusion:
    """One corrupt SFF v1 node skipped by explicit recovery mode."""

    archive_index: int
    group_number: int
    image_number: int
    reason: str
    detail: str


@dataclass(frozen=True)
class MugenSffV2Palette:
    archive_index: int
    group_number: int
    image_number: int
    color_count: int
    linked_palette_index: int | None
    rgba: bytes
    rgba_sha256: str


@dataclass(frozen=True)
class MugenSffV2Sprite:
    archive_index: int
    group_number: int
    image_number: int
    axis_x: int
    axis_y: int
    width: int
    height: int
    linked_sprite_index: int | None
    pixel_format: int
    color_depth: int
    palette_index: int | None
    indices: bytes | None
    rgba: bytes
    indices_sha256: str | None
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
    x_scale: float = 1.0
    y_scale: float = 1.0


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
    source_action_index: int = -1


@dataclass(frozen=True)
class MugenActionExclusion:
    action_number: int
    reason: Literal[
        "empty_action",
        "missing_sprite",
        "ambiguous_sprite_key",
        "non_integral_offset",
        "unsupported_air_transform",
        "unsupported_air_timing",
    ]
    detail: str
    source_action_index: int = -1


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
    """Parse the conservative Info/Files subset of a character DEF.

    Real collections commonly contain storyboard/AIR syntax or corrupt editor
    tails in files bearing a DEF extension. The MUGEN runtime only needs the
    key/value rows in ``[Info]`` and ``[Files]`` here, so unrelated malformed
    lines are retained as inert evidence but do not invalidate those sections.
    """

    text = decode_mugen_text(payload) if isinstance(payload, bytes) else payload
    comments = tuple(
        stripped[1:].strip()
        for line in text.splitlines()
        if (stripped := line.strip()).startswith(";") and stripped[1:].strip()
    )
    sections: dict[str, dict[str, str]] = {"info": {}, "files": {}}
    current: str | None = None
    for raw_line in text.splitlines():
        line = _strip_def_inline_comment(raw_line).strip()
        if not line:
            continue
        if line.startswith("["):
            if line.endswith("]"):
                name = line[1:-1].strip().casefold()
                current = name if name in sections else None
            else:
                current = None
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        normalized_key = key.strip().casefold()
        if normalized_key:
            sections[current][normalized_key] = value.strip()
    info = sections["info"]
    files = sections["files"]
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


def _strip_def_inline_comment(line: str) -> str:
    """Strip legacy semicolon comments even when authors omit preceding whitespace."""

    quoted = False
    output: list[str] = []
    for character in line:
        if character == '"':
            quoted = not quoted
        if character == ";" and not quoted:
            break
        output.append(character)
    return "".join(output)


def parse_air(
    payload: bytes | str,
    *,
    reject_duplicate_actions: bool = True,
    recover_invalid_elements: bool = False,
    exclusions: list[MugenAirParseExclusion] | None = None,
) -> tuple[MugenAirAction, ...]:
    """Parse ordered sprite references, timing, transforms, and loop points.

    Strict parsing remains the default. Recovery mode only omits rows that
    already look like sprite elements but contain invalid offset/duration
    fields; every omission is retained verbatim in ``exclusions``. It never
    guesses replacement coordinates or timing.
    """

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
            # Runtime parsers assign on each occurrence; the final marker wins.
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
            detail = f"invalid AIR element at line {line_number}: {raw_line!r}"
            if not recover_invalid_elements:
                raise ValueError(detail)
            if exclusions is not None:
                exclusions.append(
                    MugenAirParseExclusion(
                        action_number=number,
                        line_number=line_number,
                        reason="invalid_element_fields",
                        raw_line=raw_line,
                        detail=detail,
                    )
                )
            continue
        duration = int(fields[4])
        optional_fields = fields[5:]
        optional = tuple(value for value in optional_fields if value)
        flags = optional_fields[0].casefold() if optional_fields else ""
        elements.append(
            MugenAirElement(
                sprite_group=int(fields[0]),
                sprite_image=int(fields[1]),
                x_offset=float(fields[2]),
                y_offset=float(fields[3]),
                duration_ticks=duration,
                duration_seconds=None if duration < 0 else duration / 60.0,
                horizontal_flip="h" in flags,
                vertical_flip="v" in flags,
                optional_tokens=optional,
                optional_fields=optional_fields,
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


def decode_sff_v1(
    payload: bytes,
    *,
    initial_palette_rgb: bytes | None = None,
    recover_invalid_sprites: bool = False,
    exclusions: list[MugenSffV1DecodeExclusion] | None = None,
) -> tuple[MugenSffV1Sprite, ...]:
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

    sprite_slots: list[MugenSffV1Sprite | None] = []
    offset = first_offset
    if initial_palette_rgb is not None and len(initial_palette_rgb) != 768:
        raise ValueError("initial SFF v1 palette must contain exactly 768 RGB bytes")
    previous_palette: bytes | None = initial_palette_rgb
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
        actual_data_bytes = data_bytes
        # Several real legacy writers incorrectly place 512 in the declared
        # subheader-size field while still emitting the standard 32-byte record.
        # Deriving it from the linked-list boundary is exact and fail-closed.
        if actual_subheader_bytes < 32 or actual_subheader_bytes > subheader_bytes:
            recovery_end = offset + subheader_bytes
            if (
                not recover_invalid_sprites
                or record_end > len(payload)
                or recovery_end > record_end
            ):
                raise ValueError(
                    f"SFF v1 sprite {archive_index} has invalid derived subheader size "
                    f"{actual_subheader_bytes}"
                )
            actual_subheader_bytes = subheader_bytes
            actual_data_bytes = record_end - recovery_end
            if exclusions is not None:
                exclusions.append(
                    MugenSffV1DecodeExclusion(
                        archive_index=archive_index,
                        group_number=group,
                        image_number=image,
                        reason="invalid_declared_data_size_recovered",
                        detail=(
                            f"declared {data_bytes} bytes; exact record boundary contains "
                            f"{actual_data_bytes} bytes after the {subheader_bytes}-byte header"
                        ),
                    )
                )
        data_start = offset + actual_subheader_bytes
        data_end = data_start + actual_data_bytes
        if data_end > len(payload):
            raise ValueError(f"SFF v1 sprite {archive_index} data is out of bounds")

        linked_index: int | None = None
        failure: tuple[str, str] | None = None
        if data_bytes == 0:
            linked_index = link
            if linked_index >= len(sprite_slots) or sprite_slots[linked_index] is None:
                failure = (
                    "invalid_link",
                    f"links to unavailable index {linked_index}",
                )
            else:
                source = sprite_slots[linked_index]
                assert source is not None
                width, height = source.width, source.height
                indices, palette = source.indices, source.palette_rgb
        else:
            pcx_payload = payload[data_start:data_end]
            try:
                indices, palette, width, height = _decode_sff_v1_pcx(
                    pcx_payload,
                    previous_palette=previous_palette,
                    palette_reuse=bool(reuse) or (shared_palette and archive_index > 0),
                )
            except ValueError as error:
                failure = ("invalid_pcx", str(error))
                recovered_palette = _sff_v1_pcx_embedded_palette(pcx_payload)
                if recovered_palette is not None:
                    previous_palette = recovered_palette[1]
        if failure is not None:
            if not recover_invalid_sprites:
                raise ValueError(f"SFF v1 sprite {archive_index} {failure[1]}")
            if exclusions is not None:
                exclusions.append(
                    MugenSffV1DecodeExclusion(
                        archive_index=archive_index,
                        group_number=group,
                        image_number=image,
                        reason=failure[0],
                        detail=failure[1],
                    )
                )
            sprite_slots.append(None)
        else:
            previous_palette = palette
            rgba = _indexed_rgba(indices, palette)
            sprite_slots.append(
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
    return tuple(sprite for sprite in sprite_slots if sprite is not None)


def decode_sff_v2(
    payload: bytes,
) -> tuple[tuple[MugenSffV2Sprite, ...], tuple[MugenSffV2Palette, ...]]:
    """Decode SFF v2 sprites/palettes without invoking a M.U.G.E.N runtime.

    Supported pixel formats are raw indexed/RGB/RGBA, RLE8, RLE5, LZ5, and
    the PNG8/24/32 formats added by M.U.G.E.N 1.1. Every table and payload
    range is checked before decoding.
    """

    header = inspect_sff_header(payload)
    if header.format_family != "sff_v2":
        raise ValueError("decode_sff_v2 requires an SFF v2 payload")
    if len(payload) < 68:
        raise ValueError("SFF v2 payload is shorter than its header")
    (
        first_sprite_header,
        sprite_count,
        first_palette_header,
        palette_count,
        literal_data_offset,
        literal_data_bytes,
        translated_data_offset,
        translated_data_bytes,
    ) = struct.unpack_from("<8I", payload, 36)
    _bounded_region(payload, literal_data_offset, literal_data_bytes, "literal data")
    _bounded_region(payload, translated_data_offset, translated_data_bytes, "translated data")
    if first_sprite_header + sprite_count * 28 > len(payload):
        raise ValueError("SFF v2 sprite table is out of bounds")
    if first_palette_header + palette_count * 16 > len(payload):
        raise ValueError("SFF v2 palette table is out of bounds")

    palettes: list[MugenSffV2Palette] = []
    for archive_index in range(palette_count):
        offset = first_palette_header + archive_index * 16
        group, image, color_count, link, data_offset, data_bytes = struct.unpack_from(
            "<4H2I", payload, offset
        )
        linked: int | None = None
        if data_bytes == 0:
            linked = link
            if linked >= len(palettes):
                raise ValueError(
                    f"SFF v2 palette {archive_index} links to unavailable index {linked}"
                )
            rgba = palettes[linked].rgba
        else:
            start = literal_data_offset + data_offset
            _bounded_region(payload, start, data_bytes, f"palette {archive_index}")
            if data_bytes % 4:
                raise ValueError(f"SFF v2 palette {archive_index} byte count is not RGBA")
            raw_count = data_bytes // 4
            if raw_count > 256:
                raise ValueError(f"SFF v2 palette {archive_index} has over 256 colors")
            depth = 16
            while depth < raw_count:
                depth *= 2
            depth = min(depth, 256)
            rgba_array = bytearray(depth * 4)
            rgba_array[:data_bytes] = payload[start : start + data_bytes]
            # Version 2.0 stores RGBx and defines mask index zero. Version
            # 2.01+ stores the authored alpha channel.
            if header.version_bytes[1] == 0:
                for color in range(depth):
                    rgba_array[color * 4 + 3] = 0 if color == 0 else 255
            rgba = bytes(rgba_array)
        palettes.append(
            MugenSffV2Palette(
                archive_index=archive_index,
                group_number=group,
                image_number=image,
                color_count=color_count,
                linked_palette_index=linked,
                rgba=rgba,
                rgba_sha256=hashlib.sha256(rgba).hexdigest(),
            )
        )

    sprites: list[MugenSffV2Sprite] = []
    for archive_index in range(sprite_count):
        offset = first_sprite_header + archive_index * 28
        (
            group,
            image,
            width,
            height,
            axis_x,
            axis_y,
            link,
            pixel_format,
            color_depth,
            data_offset,
            data_bytes,
            palette_index,
            flags,
        ) = struct.unpack_from("<4H2hH2B2I2H", payload, offset)
        linked: int | None = None
        indices: bytes | None
        effective_palette: int | None = palette_index
        if data_bytes == 0:
            linked = link
            if linked >= len(sprites):
                raise ValueError(
                    f"SFF v2 sprite {archive_index} links to unavailable index {linked}"
                )
            source = sprites[linked]
            width, height = source.width, source.height
            indices, rgba = source.indices, source.rgba
            effective_palette = source.palette_index
        else:
            base = literal_data_offset if flags & 1 == 0 else translated_data_offset
            start = base + data_offset
            _bounded_region(payload, start, data_bytes, f"sprite {archive_index}")
            encoded = payload[start : start + data_bytes]
            indices, rgba = _decode_sff_v2_pixels(
                encoded,
                width=width,
                height=height,
                pixel_format=pixel_format,
                color_depth=color_depth,
                palette=(palettes[palette_index].rgba if palette_index < len(palettes) else None),
            )
            if indices is not None and palette_index >= len(palettes):
                raise ValueError(
                    f"SFF v2 sprite {archive_index} references absent palette {palette_index}"
                )
        if len(rgba) != width * height * 4:
            raise ValueError(f"SFF v2 sprite {archive_index} decoded RGBA size mismatch")
        sprites.append(
            MugenSffV2Sprite(
                archive_index=archive_index,
                group_number=group,
                image_number=image,
                axis_x=axis_x,
                axis_y=axis_y,
                width=width,
                height=height,
                linked_sprite_index=linked,
                pixel_format=pixel_format,
                color_depth=color_depth,
                palette_index=effective_palette,
                indices=indices,
                rgba=rgba,
                indices_sha256=(
                    hashlib.sha256(indices).hexdigest() if indices is not None else None
                ),
                rgba_sha256=hashlib.sha256(rgba).hexdigest(),
            )
        )
    return tuple(sprites), tuple(palettes)


def audit_character_zip_variants(
    archive_payload: bytes,
    *,
    limits: ArchiveLimits | None = None,
) -> tuple[MugenCharacterArchiveAudit, ...]:
    """Audit every distinct DEF-selected AIR/SFF pair without executing character logic."""

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
        grouped: dict[
            tuple[str, str],
            list[tuple[ZipMember, MugenCharacterDefinition, ZipMember, ZipMember]],
        ] = {}
        for row in resolved_variants:
            key = (row[2].normalized_name.casefold(), row[3].normalized_name.casefold())
            grouped.setdefault(key, []).append(row)
        decoded_groups = []
        for key in sorted(grouped):
            rows = grouped[key]
            definition_member, definition, air_member, sff_member = rows[0]
            actions = parse_air(
                archive.read(infos[air_member.archive_index]), reject_duplicate_actions=False
            )
            sff_header = inspect_sff_header(archive.read(infos[sff_member.archive_index]))
            decoded_groups.append(
                (rows, definition_member, definition, air_member, sff_member, actions, sff_header)
            )

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
    archive_sha256 = hashlib.sha256(archive_payload).hexdigest()
    return tuple(
        MugenCharacterArchiveAudit(
            archive_sha256=archive_sha256,
            archive_bytes=len(archive_payload),
            inventory_sha256=manifest.inventory_sha256,
            member_count=len(manifest.members),
            definition_member=definition_member.normalized_name,
            definition_members=tuple(row[0].normalized_name for row in rows),
            definition_variants=tuple(row[1] for row in rows),
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
        for (
            rows,
            definition_member,
            definition,
            air_member,
            sff_member,
            actions,
            sff_header,
        ) in decoded_groups
    )


def audit_character_zip(
    archive_payload: bytes,
    *,
    limits: ArchiveLimits | None = None,
) -> MugenCharacterArchiveAudit:
    """Audit a ZIP containing exactly one distinct character media pair."""

    variants = audit_character_zip_variants(archive_payload, limits=limits)
    if len(variants) != 1:
        raise ValueError(f"character archive contains {len(variants)} AIR/SFF media pairs")
    return variants[0]


def materialize_actions(
    actions: tuple[MugenAirAction, ...],
    sprites: tuple[MugenSffV1Sprite | MugenSffV2Sprite, ...],
) -> MugenActionPlan:
    """Render supported AIR actions to aligned, transparent RGBA canvases."""

    by_key: dict[tuple[int, int], list[MugenSffV1Sprite | MugenSffV2Sprite]] = {}
    for sprite in sprites:
        by_key.setdefault((sprite.group_number, sprite.image_number), []).append(sprite)

    admitted: list[MugenActionMaterialization] = []
    excluded: list[MugenActionExclusion] = []
    for source_action_index, action in enumerate(actions):
        if not action.elements:
            excluded.append(
                MugenActionExclusion(
                    action.action_number,
                    "empty_action",
                    "no elements",
                    source_action_index,
                )
            )
            continue
        unsupported_duration = next(
            (element for element in action.elements if element.duration_ticks < -1), None
        )
        if unsupported_duration is not None:
            excluded.append(
                MugenActionExclusion(
                    action.action_number,
                    "unsupported_air_timing",
                    (
                        f"line {unsupported_duration.source_line}: unsupported negative "
                        f"duration {unsupported_duration.duration_ticks}"
                    ),
                    source_action_index,
                )
            )
            continue
        resolved: list[
            tuple[
                MugenAirElement,
                MugenSffV1Sprite | MugenSffV2Sprite,
                int,
                int,
                bytes,
                int,
                int,
                float,
                float,
            ]
        ] = []
        rejection: MugenActionExclusion | None = None
        for element in action.elements:
            key = (element.sprite_group, element.sprite_image)
            candidates = by_key.get(key, [])
            if not candidates:
                rejection = MugenActionExclusion(
                    action.action_number,
                    "missing_sprite",
                    f"missing SFF key {key}",
                    source_action_index,
                )
                break
            unique_hashes = {candidate.rgba_sha256 for candidate in candidates}
            if len(unique_hashes) != 1:
                rejection = MugenActionExclusion(
                    action.action_number,
                    "ambiguous_sprite_key",
                    f"SFF key {key} has {len(unique_hashes)} pixel variants",
                    source_action_index,
                )
                break
            transform_error, x_scale, y_scale = _element_spatial_transform(element)
            if transform_error is not None:
                rejection = MugenActionExclusion(
                    action.action_number,
                    "unsupported_air_transform",
                    f"line {element.source_line}: {transform_error}",
                    source_action_index,
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
            transformed_width = sprite.width
            transformed_height = sprite.height
            axis_x = sprite.width - 1 - sprite.axis_x if element.horizontal_flip else sprite.axis_x
            axis_y = sprite.height - 1 - sprite.axis_y if element.vertical_flip else sprite.axis_y
            if x_scale != 1.0 or y_scale != 1.0:
                transformed, transformed_width, transformed_height = _scale_rgba_nearest(
                    transformed,
                    width=sprite.width,
                    height=sprite.height,
                    x_scale=x_scale,
                    y_scale=y_scale,
                )
                axis_x = _scale_axis(axis_x, x_scale)
                axis_y = _scale_axis(axis_y, y_scale)
            left = _round_half_away_from_zero(element.x_offset * x_scale) - axis_x
            top = _round_half_away_from_zero(element.y_offset * y_scale) - axis_y
            resolved.append(
                (
                    element,
                    sprite,
                    left,
                    top,
                    transformed,
                    transformed_width,
                    transformed_height,
                    x_scale,
                    y_scale,
                )
            )
        if rejection is not None:
            excluded.append(rejection)
            continue

        world_left = min(row[2] for row in resolved)
        world_top = min(row[3] for row in resolved)
        world_right = max(row[2] + row[5] for row in resolved)
        world_bottom = max(row[3] + row[6] for row in resolved)
        canvas_width = world_right - world_left
        canvas_height = world_bottom - world_top
        frames: list[MugenActionFrame] = []
        for ordinal, (
            element,
            sprite,
            left,
            top,
            transformed,
            transformed_width,
            transformed_height,
            x_scale,
            y_scale,
        ) in enumerate(resolved):
            canvas = bytearray(canvas_width * canvas_height * 4)
            _paste_rgba(
                canvas,
                canvas_width=canvas_width,
                source=transformed,
                source_width=transformed_width,
                source_height=transformed_height,
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
                    width=transformed_width,
                    height=transformed_height,
                    horizontal_flip=element.horizontal_flip,
                    vertical_flip=element.vertical_flip,
                    rgba=rgba,
                    rgba_sha256=hashlib.sha256(rgba).hexdigest(),
                    x_scale=x_scale,
                    y_scale=y_scale,
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
                source_action_index=source_action_index,
            )
        )
    return MugenActionPlan(tuple(admitted), tuple(excluded))


def _optional_value(values: dict[str, str], key: str) -> str | None:
    value = values.get(key.casefold())
    if value is None:
        return None
    content, _ = _split_comment(value)
    normalized = _unquote(content.strip())
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


def _bounded_region(payload: bytes, offset: int, size: int, label: str) -> None:
    if offset < 0 or size < 0 or offset > len(payload) or size > len(payload) - offset:
        raise ValueError(f"SFF v2 {label} region is out of bounds")


def _decode_sff_v2_pixels(
    payload: bytes,
    *,
    width: int,
    height: int,
    pixel_format: int,
    color_depth: int,
    palette: bytes | None,
) -> tuple[bytes | None, bytes]:
    pixel_count = width * height
    if width <= 0 or height <= 0 or pixel_count > 268_435_456:
        raise ValueError(f"invalid SFF v2 sprite dimensions: {width}x{height}")
    indexed: bytes | None = None
    if pixel_format == 0:
        if color_depth == 8:
            if len(payload) != pixel_count:
                raise ValueError("raw indexed SFF v2 sprite byte count mismatch")
            indexed = payload
        elif color_depth == 24:
            if len(payload) != pixel_count * 3:
                raise ValueError("raw RGB SFF v2 sprite byte count mismatch")
            rgb = np.frombuffer(payload, dtype=np.uint8).reshape(pixel_count, 3)
            rgba = np.empty((pixel_count, 4), dtype=np.uint8)
            rgba[:, :3] = rgb
            rgba[:, 3] = 255
            return None, rgba.tobytes()
        elif color_depth == 32:
            if len(payload) != pixel_count * 4:
                raise ValueError("raw RGBA SFF v2 sprite byte count mismatch")
            return None, payload
        else:
            raise ValueError(f"unsupported raw SFF v2 color depth: {color_depth}")
    elif pixel_format in {2, 3, 4}:
        if len(payload) < 4:
            raise ValueError("compressed SFF v2 sprite lacks its size prefix")
        declared = struct.unpack_from("<I", payload)[0]
        compressed = payload[4:]
        if pixel_format == 2:
            indexed = _sff_v2_rle8(compressed, pixel_count)
        elif pixel_format == 3:
            indexed = _sff_v2_rle5(compressed, pixel_count)
        else:
            indexed = _sff_v2_lz5(compressed, pixel_count)
        # Elecbyte-compatible files declare decompressed bytes (= pixels for
        # indexed sprites). Fighter Factory has also emitted a widespread
        # legacy bit-count prefix (= pixels * 8) while producing otherwise
        # valid RLE8/RLE5/LZ5 streams. Accept only those exact conventions and
        # still require the decoder to yield precisely width*height indices.
        if declared not in {0, pixel_count, pixel_count * 8}:
            raise ValueError(
                f"SFF v2 decompressed-size prefix {declared} differs from {pixel_count}"
            )
    elif pixel_format in {10, 11, 12}:
        if len(payload) < 4:
            raise ValueError("PNG SFF v2 sprite lacks its size prefix")
        try:
            with Image.open(io.BytesIO(payload[4:])) as image:
                image.load()
                if image.size != (width, height):
                    raise ValueError(
                        f"SFF v2 PNG size {image.size!r} differs from {(width, height)!r}"
                    )
                if pixel_format == 10:
                    if image.mode != "P":
                        raise ValueError("SFF v2 PNG8 sprite is not indexed")
                    indexed = image.tobytes()
                else:
                    return None, image.convert("RGBA").tobytes()
        except (OSError, SyntaxError) as error:
            raise ValueError(f"invalid embedded SFF v2 PNG: {error}") from error
    else:
        raise ValueError(f"unsupported SFF v2 pixel format: {pixel_format}")
    if indexed is None or len(indexed) != pixel_count:
        raise ValueError("indexed SFF v2 sprite pixel count mismatch")
    if palette is None:
        raise ValueError("indexed SFF v2 sprite has no resolvable palette")
    return indexed, _sff_v2_indexed_rgba(indexed, palette)


def _sff_v2_indexed_rgba(indices: bytes, palette: bytes) -> bytes:
    if len(palette) % 4:
        raise ValueError("SFF v2 palette byte count is not RGBA")
    colors = len(palette) // 4
    index_array = np.frombuffer(indices, dtype=np.uint8)
    maximum = int(index_array.max(initial=0))
    if maximum >= colors:
        raise ValueError(f"SFF v2 palette index {maximum} exceeds {colors} colors")
    palette_array = np.frombuffer(palette, dtype=np.uint8).reshape(colors, 4)
    return np.ascontiguousarray(palette_array[index_array]).tobytes()


def _sff_v2_rle8(payload: bytes, pixel_count: int) -> bytes:
    output = bytearray()
    cursor = 0
    while len(output) < pixel_count:
        if cursor >= len(payload):
            raise ValueError("truncated SFF v2 RLE8 stream")
        value = payload[cursor]
        cursor += 1
        count = 1
        if value & 0xC0 == 0x40:
            count = value & 0x3F
            if cursor >= len(payload):
                raise ValueError("truncated SFF v2 RLE8 run")
            value = payload[cursor]
            cursor += 1
        if len(output) + count > pixel_count:
            raise ValueError("SFF v2 RLE8 run exceeds pixel count")
        output.extend(bytes((value,)) * count)
    return bytes(output)


def _sff_v2_rle5(payload: bytes, pixel_count: int) -> bytes:
    output = bytearray()
    cursor = 0
    while len(output) < pixel_count:
        if cursor + 2 > len(payload):
            raise ValueError("truncated SFF v2 RLE5 packet")
        run_length = payload[cursor]
        cursor += 1
        control = payload[cursor]
        cursor += 1
        data_length = control & 0x7F
        color = 0
        if control & 0x80:
            if cursor >= len(payload):
                raise ValueError("truncated SFF v2 RLE5 color")
            color = payload[cursor]
            cursor += 1
        runs = [(run_length + 1, color)]
        for _ in range(data_length):
            if cursor >= len(payload):
                raise ValueError("truncated SFF v2 RLE5 data run")
            value = payload[cursor]
            cursor += 1
            runs.append(((value >> 5) + 1, value & 0x1F))
        for count, value in runs:
            if len(output) + count > pixel_count:
                raise ValueError("SFF v2 RLE5 run exceeds pixel count")
            output.extend(bytes((value,)) * count)
    return bytes(output)


def _sff_v2_lz5(payload: bytes, pixel_count: int) -> bytes:
    if not payload:
        raise ValueError("empty SFF v2 LZ5 stream")
    output = bytearray()
    cursor = 1
    control = payload[0]
    control_shift = 0
    recycled = 0
    recycled_bits = 0
    while len(output) < pixel_count:
        if cursor >= len(payload):
            raise ValueError("truncated SFF v2 LZ5 packet")
        value = payload[cursor]
        cursor += 1
        if control & (1 << control_shift):
            if value & 0x3F == 0:
                if cursor + 2 > len(payload):
                    raise ValueError("truncated SFF v2 LZ5 long copy")
                distance = ((value << 2) | payload[cursor]) + 1
                cursor += 1
                count = payload[cursor] + 3
                cursor += 1
            else:
                recycled |= (value & 0xC0) >> recycled_bits
                recycled_bits += 2
                count = (value & 0x3F) + 1
                if recycled_bits < 8:
                    if cursor >= len(payload):
                        raise ValueError("truncated SFF v2 LZ5 short copy")
                    distance = payload[cursor] + 1
                    cursor += 1
                else:
                    distance = recycled + 1
                    recycled = 0
                    recycled_bits = 0
            if distance > len(output) or len(output) + count > pixel_count:
                raise ValueError("invalid SFF v2 LZ5 back-reference")
            for _ in range(count):
                output.append(output[-distance])
        else:
            if value & 0xE0 == 0:
                if cursor >= len(payload):
                    raise ValueError("truncated SFF v2 LZ5 long run")
                count = payload[cursor] + 8
                cursor += 1
                color = value
            else:
                count = value >> 5
                color = value & 0x1F
            if len(output) + count > pixel_count:
                raise ValueError("SFF v2 LZ5 run exceeds pixel count")
            output.extend(bytes((color,)) * count)
        control_shift += 1
        if control_shift == 8 and len(output) < pixel_count:
            if cursor >= len(payload):
                raise ValueError("truncated SFF v2 LZ5 control byte")
            control = payload[cursor]
            cursor += 1
            control_shift = 0
    return bytes(output)


def _decode_sff_v1_pcx(
    payload: bytes,
    *,
    previous_palette: bytes | None,
    palette_reuse: bool,
) -> tuple[bytes, bytes, int, int]:
    if len(payload) < 128 or payload[0] != 0x0A:
        raise ValueError("invalid SFF v1 PCX header")
    encoding = payload[2]
    bits_per_pixel = payload[3]
    xmin, ymin, xmax, ymax = struct.unpack_from("<4H", payload, 4)
    if xmax < xmin or ymax < ymin:
        raise ValueError("invalid SFF v1 PCX bounds")
    width, height = xmax - xmin + 1, ymax - ymin + 1
    planes = payload[65]
    bytes_per_line = struct.unpack_from("<H", payload, 66)[0]
    if bits_per_pixel != 8 or planes != 1 or bytes_per_line < width:
        raise ValueError("SFF v1 PCX must be single-plane 8-bit indexed data with a valid stride")

    embedded_palette = None if palette_reuse else _sff_v1_pcx_embedded_palette(payload)
    palette_marker = embedded_palette[0] if embedded_palette is not None else -1
    raster_end = palette_marker if palette_marker >= 0 else len(payload)
    encoded = payload[128:raster_end]
    expected = bytes_per_line * height
    decoded = bytearray()
    cursor = 0
    while len(decoded) < expected and cursor < len(encoded):
        value = encoded[cursor]
        cursor += 1
        if encoding == 1 and value >= 0xC0:
            count = value & 0x3F
            if cursor >= len(encoded):
                raise ValueError("truncated SFF v1 PCX run")
            value = encoded[cursor]
            cursor += 1
        else:
            count = 1
        decoded.extend(bytes((value,)) * min(count, expected - len(decoded)))
    if len(decoded) != expected:
        raise ValueError(
            f"truncated SFF v1 PCX raster: expected {expected} bytes, got {len(decoded)}"
        )
    indices = b"".join(
        bytes(decoded[row * bytes_per_line : row * bytes_per_line + width]) for row in range(height)
    )

    if palette_marker >= 0:
        assert embedded_palette is not None
        palette = embedded_palette[1]
    elif palette_reuse and previous_palette is not None:
        palette = previous_palette
    else:
        raise ValueError("SFF v1 PCX omits its palette without a reusable predecessor")
    return indices, palette, width, height


def _sff_v1_pcx_embedded_palette(payload: bytes) -> tuple[int, bytes] | None:
    if len(payload) < 897:
        return None
    for marker in range(len(payload) - 769, 127, -1):
        if payload[marker] == 0x0C and marker + 769 <= len(payload):
            return marker, payload[marker + 1 : marker + 769]
    return None


def _indexed_rgba(indices: bytes, palette: bytes) -> bytes:
    if len(palette) != 768:
        raise ValueError("indexed RGB palette must contain exactly 256 colors")
    palette_array = np.frombuffer(palette, dtype=np.uint8).reshape(256, 3)
    index_array = np.frombuffer(indices, dtype=np.uint8)
    rgba = np.empty((len(indices), 4), dtype=np.uint8)
    rgba[:, :3] = palette_array[index_array]
    rgba[:, 3] = 255
    rgba[index_array == 0, 3] = 0
    return rgba.tobytes()


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


def _element_spatial_transform(element: MugenAirElement) -> tuple[str | None, float, float]:
    """Validate the subset of MUGEN 1.1 AIR transforms baked into RGBA frames."""

    fields = element.optional_fields
    flip = fields[0].casefold() if fields else ""
    if flip not in {"", "h", "v", "hv", "vh"}:
        return f"unsupported flip token {fields[0]!r}", 1.0, 1.0
    blend = fields[1] if len(fields) > 1 else ""
    if blend:
        return f"background-dependent blend token {blend!r}", 1.0, 1.0
    try:
        x_scale = float(fields[2]) if len(fields) > 2 and fields[2] else 1.0
        y_scale = float(fields[3]) if len(fields) > 3 and fields[3] else 1.0
        angle = float(fields[4]) if len(fields) > 4 and fields[4] else 0.0
    except ValueError:
        return f"non-numeric scale/angle fields {fields[2:5]!r}", 1.0, 1.0
    if not np.isfinite(x_scale) or not np.isfinite(y_scale) or x_scale <= 0 or y_scale <= 0:
        return f"invalid scale {x_scale},{y_scale}", 1.0, 1.0
    if not np.isfinite(angle) or angle != 0.0:
        return f"unsupported rotation angle {angle}", 1.0, 1.0
    if any(fields[5:]):
        return f"unsupported extra AIR fields {fields[5:]!r}", 1.0, 1.0
    return None, x_scale, y_scale


def _scale_rgba_nearest(
    rgba: bytes,
    *,
    width: int,
    height: int,
    x_scale: float,
    y_scale: float,
) -> tuple[bytes, int, int]:
    """Bake positive MUGEN 1.1 element scale with deterministic nearest sampling."""

    if len(rgba) != width * height * 4:
        raise ValueError("RGBA byte count does not match dimensions")
    output_width = max(1, _round_half_away_from_zero(width * x_scale))
    output_height = max(1, _round_half_away_from_zero(height * y_scale))
    source = np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4)
    source_x = np.minimum((np.arange(output_width) / x_scale).astype(int), width - 1)
    source_y = np.minimum((np.arange(output_height) / y_scale).astype(int), height - 1)
    output = np.ascontiguousarray(source[source_y[:, None], source_x[None, :]])
    return output.tobytes(), output_width, output_height


def _scale_axis(axis: int, scale: float) -> int:
    return _round_half_away_from_zero(axis * scale)


def _round_half_away_from_zero(value: float) -> int:
    if value >= 0:
        return int(np.floor(value + 0.5))
    return int(np.ceil(value - 0.5))


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
