from __future__ import annotations

import hashlib
import io
import struct
import zipfile

import pytest
from PIL import Image

from spritelab.adapters.mugen import (
    MugenSffV1Sprite,
    audit_character_zip,
    decode_sff_v1,
    inspect_sff_header,
    label_action_number,
    materialize_actions,
    parse_air,
    parse_character_def,
)


def test_def_preserves_identity_authorship_and_references() -> None:
    parsed = parse_character_def(
        b"""; mirrored by Example\r\n
[Info]\r\n
name = \"Hero\"\r\n
displayname = \"Anime Hero\"\r\n
author = \"Artist\"\r\n
versiondate = 01,02,2020\r\n
mugenversion = 1.1\r\n
localcoord = 320, 240\r\n
pal.defaults = 2, 1\r\n
[Files]\r\n
sprite = Hero.sff\r\n
anim = Hero.air\r\n
cmd = Hero.cmd ; runtime logic\r\n
"""
    )

    assert parsed.name == "Hero"
    assert parsed.display_name == "Anime Hero"
    assert parsed.author == "Artist"
    assert parsed.local_coord == (320, 240)
    assert parsed.palette_defaults == (2, 1)
    assert parsed.file("sprite") == "Hero.sff"
    assert parsed.file("cmd") == "Hero.cmd"
    assert parsed.source_comments == ("mirrored by Example",)


def test_air_parses_ticks_loopstart_flips_and_standard_actions() -> None:
    actions = parse_air(
        """; idle
[Begin Action 0]
0, 0, 1.5, -2, 6
Loopstart
0, 1, 0, 0, 12, HV

; light punch
[Begin Action 200]
Clsn1: 1
Clsn1[0] = 0,0,1,1
200,0,0,0,3
200,1,0,0,-1
"""
    )

    idle, attack = actions
    assert idle.label.normalized_action == "idle"
    assert idle.loop_mode == "intro_then_loop"
    assert idle.loop_start_index == 1
    assert idle.finite_duration_ticks == 18
    assert idle.elements[0].duration_seconds == pytest.approx(0.1)
    assert idle.elements[1].horizontal_flip is True
    assert idle.elements[1].vertical_flip is True
    assert attack.label.normalized_action == "attack"
    assert attack.loop_mode == "terminal_hold"
    assert attack.collision_1_declarations == 1


def test_air_rejects_duplicate_actions_and_invalid_durations() -> None:
    with pytest.raises(ValueError, match="duplicate action"):
        parse_air("[Begin Action 0]\n0,0,0,0,1\n[Begin Action 0]\n0,1,0,0,1")
    with pytest.raises(ValueError, match="invalid AIR duration"):
        parse_air("[Begin Action 0]\n0,0,0,0,-2")


def test_action_labels_distinguish_exact_and_recommended_ranges() -> None:
    assert label_action_number(20).normalized_action == "walk"
    assert label_action_number(181).normalized_action == "emote"
    assert label_action_number(230).normalized_action == "attack"
    assert label_action_number(3500).source_meaning == "hyper_attack"
    assert label_action_number(900).normalized_action is None


def test_sff_header_is_inspection_only_and_hash_bound() -> None:
    payload = b"ElecbyteSpr\x00" + bytes((0, 0, 0, 2)) + b"payload"
    header = inspect_sff_header(payload)

    assert header.signature == "ElecbyteSpr"
    assert header.version_bytes == (0, 0, 0, 2)
    assert header.format_family == "sff_v2"
    assert len(header.sha256) == 64


def test_sff_v1_decodes_palette_mask_and_linked_sprite() -> None:
    palette = [0] * 768
    palette[3:6] = [10, 20, 30]
    image = Image.new("P", (2, 1))
    image.putpalette(palette)
    image.putdata((0, 1))
    pcx = io.BytesIO()
    image.save(pcx, format="PCX")

    header = bytearray(512)
    header[:12] = b"ElecbyteSpr\x00"
    header[12:16] = bytes((0, 1, 0, 1))
    struct.pack_into("<4I", header, 16, 1, 2, 512, 32)
    header[32] = 1
    first_next = 512 + 32 + len(pcx.getvalue())
    first = struct.pack("<IIhhhhHB13x", first_next, len(pcx.getvalue()), 1, 2, 0, 0, 0, 0)
    linked = struct.pack("<IIhhhhHB13x", 0, 0, 3, 4, 0, 1, 0, 1)

    sprites = decode_sff_v1(bytes(header) + first + pcx.getvalue() + linked)

    assert len(sprites) == 2
    assert sprites[0].rgba == bytes((0, 0, 0, 0, 10, 20, 30, 255))
    assert sprites[1].linked_sprite_index == 0
    assert sprites[1].rgba_sha256 == sprites[0].rgba_sha256
    assert (sprites[1].axis_x, sprites[1].axis_y) == (3, 4)


def test_character_zip_audit_never_interprets_runtime_logic() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "hero/Hero.def",
            "[Info]\nname=Hero\nauthor=Artist\n[Files]\nsprite=Hero.sff\nanim=Hero.air\n",
        )
        archive.writestr("hero/Hero.air", "[Begin Action 20]\n20,0,0,0,6")
        sff = b"ElecbyteSpr\x00" + bytes((0, 0, 0, 1)) + struct.pack("<I", 0)
        archive.writestr("hero/Hero.sff", sff)
        archive.writestr("hero/Hero.cmd", "[Command]\ncommand = x")
        archive.writestr("hero/unsafe.exe", b"MZ")
        archive.writestr("hero/readme.txt", "credit me")

    audit = audit_character_zip(buffer.getvalue())

    assert audit.definition.name == "Hero"
    assert audit.actions[0].label.normalized_action == "walk"
    assert audit.sff_header.format_family == "sff_v1"
    assert audit.runtime_logic_members == ("hero/Hero.cmd",)
    assert audit.executable_members == ("hero/unsafe.exe",)
    assert audit.unclassified_members == ("hero/readme.txt",)


def test_action_materialization_aligns_axis_offsets_and_exact_flip() -> None:
    rgba = bytes((1, 2, 3, 255, 4, 5, 6, 255))
    palette = bytes(768)
    sprite = MugenSffV1Sprite(
        archive_index=0,
        group_number=20,
        image_number=0,
        axis_x=0,
        axis_y=0,
        width=2,
        height=1,
        linked_sprite_index=None,
        palette_reuse=False,
        indices=b"\x01\x02",
        palette_rgb=palette,
        rgba=rgba,
        indices_sha256=hashlib.sha256(b"\x01\x02").hexdigest(),
        palette_sha256=hashlib.sha256(palette).hexdigest(),
        rgba_sha256=hashlib.sha256(rgba).hexdigest(),
    )
    actions = parse_air("[Begin Action 20]\n20,0,3,4,6,H")

    plan = materialize_actions(actions, (sprite,))

    assert plan.excluded == ()
    action = plan.admitted[0]
    assert (action.canvas_world_left, action.canvas_world_top) == (2, 4)
    assert (action.canvas_width, action.canvas_height) == (2, 1)
    assert action.frames[0].rgba == bytes((4, 5, 6, 255, 1, 2, 3, 255))
