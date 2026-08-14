from __future__ import annotations

import hashlib
import io
import struct
import zipfile

import pytest
from PIL import Image

from spritelab.adapters.mugen import (
    MugenAirParseExclusion,
    MugenSffV1Sprite,
    audit_character_zip,
    audit_character_zip_variants,
    decode_sff_v1,
    decode_sff_v2,
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
anim = Hero.air;Animation data\r\n
cmd = Hero.cmd ; runtime logic\r\n
"""
    )

    assert parsed.name == "Hero"
    assert parsed.display_name == "Anime Hero"
    assert parsed.author == "Artist"
    assert parsed.local_coord == (320, 240)
    assert parsed.palette_defaults == (2, 1)
    assert parsed.file("sprite") == "Hero.sff"
    assert parsed.file("anim") == "Hero.air"
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


def test_standard_jump_and_guard_phase_labels_follow_elecbyte_numbers() -> None:
    assert label_action_number(44).source_meaning == "jump_neutral_down"
    assert label_action_number(47).source_meaning == "jump_land"
    assert label_action_number(150).source_meaning == "guard_hit_standing"


def test_air_rejects_duplicate_actions_and_preserves_unsupported_durations() -> None:
    with pytest.raises(ValueError, match="duplicate action"):
        parse_air("[Begin Action 0]\n0,0,0,0,1\n[Begin Action 0]\n0,1,0,0,1")
    action = parse_air("[Begin Action 5001]\n0,0,0,0,-2")[0]
    assert action.elements[0].duration_ticks == -2
    assert action.elements[0].duration_seconds is None


def test_air_recovery_omits_only_invalid_rows_with_verbatim_evidence() -> None:
    exclusions: list[MugenAirParseExclusion] = []

    action = parse_air(
        "[Begin Action 200]\n200,0,0,0,3\n200,1,0,broken,4\n200,2,1,2,5",
        recover_invalid_elements=True,
        exclusions=exclusions,
    )[0]

    assert [(row.sprite_group, row.sprite_image) for row in action.elements] == [
        (200, 0),
        (200, 2),
    ]
    assert action.finite_duration_ticks == 8
    assert exclusions == [
        MugenAirParseExclusion(
            action_number=200,
            line_number=3,
            reason="invalid_element_fields",
            raw_line="200,1,0,broken,4",
            detail="invalid AIR element at line 3: '200,1,0,broken,4'",
        )
    ]


def test_def_ignores_unrelated_malformed_editor_tail() -> None:
    parsed = parse_character_def(
        b"[Info]\nname=Hero\n[Files]\nanim=hero.air\nsprite=hero.sff\n[A\ngarbage\n"
    )

    assert parsed.name == "Hero"
    assert parsed.file("anim") == "hero.air"
    assert parsed.file("sprite") == "hero.sff"


def test_air_uses_final_loopstart_like_runtime_parser() -> None:
    action = parse_air("[Begin Action 0]\n0,0,0,0,1\nLoopstart\n0,1,0,0,1\nLoopstart\n0,2,0,0,1")[0]

    assert action.loop_start_index == 2


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


def test_sff_v1_recovery_skips_invalid_self_link_with_exact_evidence() -> None:
    image = Image.new("P", (1, 1))
    image.putpalette(bytes(768))
    image.putdata((0,))
    pcx = io.BytesIO()
    image.save(pcx, format="PCX")
    header = bytearray(512)
    header[:12] = b"ElecbyteSpr\x00"
    header[12:16] = bytes((0, 1, 0, 1))
    struct.pack_into("<4I", header, 16, 1, 2, 512, 32)
    first_next = 512 + 32 + len(pcx.getvalue())
    first = struct.pack("<IIhhhhHB13x", first_next, len(pcx.getvalue()), 0, 0, 0, 0, 0, 0)
    self_link = struct.pack("<IIhhhhHB13x", 0, 0, 0, 0, 9, 9, 1, 1)
    payload = bytes(header) + first + pcx.getvalue() + self_link
    exclusions = []

    sprites = decode_sff_v1(payload, recover_invalid_sprites=True, exclusions=exclusions)

    assert len(sprites) == 1
    assert exclusions[0].archive_index == 1
    assert exclusions[0].reason == "invalid_link"
    assert "unavailable index 1" in exclusions[0].detail


def test_sff_v2_decodes_palette_raw_and_rle8() -> None:
    palette = bytearray(1024)
    palette[4:8] = bytes((10, 20, 30, 255))
    raw = _sff_v2_fixture(bytes((0, 1, 1, 0)), palette, pixel_format=0)
    compressed = _sff_v2_fixture(struct.pack("<I", 4) + bytes((0x44, 1)), palette, pixel_format=2)
    legacy_bits = _sff_v2_fixture(struct.pack("<I", 32) + bytes((0x44, 1)), palette, pixel_format=2)

    raw_sprites, raw_palettes = decode_sff_v2(raw)
    compressed_sprites, _ = decode_sff_v2(compressed)
    legacy_sprites, _ = decode_sff_v2(legacy_bits)

    assert len(raw_palettes) == 1
    assert raw_palettes[0].rgba[3] == 0
    assert raw_sprites[0].rgba == bytes((0, 0, 0, 0, 10, 20, 30, 255, 10, 20, 30, 255, 0, 0, 0, 0))
    assert compressed_sprites[0].rgba == bytes((10, 20, 30, 255)) * 4
    assert legacy_sprites[0].rgba == compressed_sprites[0].rgba


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


def test_character_zip_variants_preserve_distinct_media_pairs() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in ("Alpha", "Beta"):
            archive.writestr(
                f"pack/{name}.def",
                f"[Info]\nname={name}\n[Files]\nsprite={name}.sff\nanim={name}.air\n",
            )
            archive.writestr(f"pack/{name}.air", "[Begin Action 0]\n0,0,0,0,6")
            archive.writestr(
                f"pack/{name}.sff",
                b"ElecbyteSpr\x00" + bytes((0, 0, 0, 1)) + struct.pack("<I", 0),
            )

    variants = audit_character_zip_variants(buffer.getvalue())

    assert [value.definition.name for value in variants] == ["Alpha", "Beta"]
    with pytest.raises(ValueError, match="2 AIR/SFF media pairs"):
        audit_character_zip(buffer.getvalue())


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


def test_action_materialization_bakes_mugen_11_scale_with_nearest_pixels() -> None:
    rgba = bytes((1, 2, 3, 255, 4, 5, 6, 255))
    palette = bytes(768)
    sprite = MugenSffV1Sprite(
        archive_index=0,
        group_number=0,
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
    actions = parse_air("[Begin Action 0]\n0,0,0,0,6, , ,2,2")

    plan = materialize_actions(actions, (sprite,))

    assert plan.excluded == ()
    action = plan.admitted[0]
    assert (action.canvas_width, action.canvas_height) == (4, 2)
    assert action.frames[0].x_scale == 2.0
    assert action.frames[0].y_scale == 2.0
    row = bytes((1, 2, 3, 255)) * 2 + bytes((4, 5, 6, 255)) * 2
    assert action.frames[0].rgba == row * 2


def test_action_materialization_defers_background_blend_and_rotation() -> None:
    rgba = bytes((1, 2, 3, 255))
    palette = bytes(768)
    sprite = MugenSffV1Sprite(
        archive_index=0,
        group_number=0,
        image_number=0,
        axis_x=0,
        axis_y=0,
        width=1,
        height=1,
        linked_sprite_index=None,
        palette_reuse=False,
        indices=b"\x01",
        palette_rgb=palette,
        rgba=rgba,
        indices_sha256=hashlib.sha256(b"\x01").hexdigest(),
        palette_sha256=hashlib.sha256(palette).hexdigest(),
        rgba_sha256=hashlib.sha256(rgba).hexdigest(),
    )
    actions = parse_air("[Begin Action 0]\n0,0,0,0,6, ,A\n[Begin Action 20]\n0,0,0,0,6, , ,1,1,45")

    plan = materialize_actions(actions, (sprite,))

    assert plan.admitted == ()
    assert [row.reason for row in plan.excluded] == [
        "unsupported_air_transform",
        "unsupported_air_transform",
    ]
    assert "background-dependent blend" in plan.excluded[0].detail
    assert "rotation angle" in plan.excluded[1].detail


def test_action_materialization_quarantines_unsupported_negative_timing() -> None:
    rgba = bytes((1, 2, 3, 255))
    palette = bytes(768)
    sprite = MugenSffV1Sprite(
        archive_index=0,
        group_number=0,
        image_number=0,
        axis_x=0,
        axis_y=0,
        width=1,
        height=1,
        linked_sprite_index=None,
        palette_reuse=False,
        indices=b"\x01",
        palette_rgb=palette,
        rgba=rgba,
        indices_sha256=hashlib.sha256(b"\x01").hexdigest(),
        palette_sha256=hashlib.sha256(palette).hexdigest(),
        rgba_sha256=hashlib.sha256(rgba).hexdigest(),
    )

    plan = materialize_actions(parse_air("[Begin Action 5001]\n0,0,0,0,-2"), (sprite,))

    assert plan.admitted == ()
    assert plan.excluded[0].reason == "unsupported_air_timing"
    assert "duration -2" in plan.excluded[0].detail


def _sff_v2_fixture(sprite_data: bytes, palette: bytearray, *, pixel_format: int) -> bytes:
    first_sprite = 68
    first_palette = first_sprite + 28
    literal_offset = first_palette + 16
    palette_bytes = bytes(palette)
    translated_offset = literal_offset + len(palette_bytes) + len(sprite_data)
    header = bytearray(68)
    header[:12] = b"ElecbyteSpr\x00"
    header[12:16] = bytes((0, 0, 0, 2))
    struct.pack_into(
        "<8I",
        header,
        36,
        first_sprite,
        1,
        first_palette,
        1,
        literal_offset,
        len(palette_bytes) + len(sprite_data),
        translated_offset,
        0,
    )
    sprite = struct.pack(
        "<4H2hH2B2I2H",
        0,
        0,
        2,
        2,
        1,
        2,
        0,
        pixel_format,
        8,
        len(palette_bytes),
        len(sprite_data),
        0,
        0,
    )
    palette_header = struct.pack("<4H2I", 1, 1, 256, 0, 0, len(palette_bytes))
    return bytes(header) + sprite + palette_header + palette_bytes + sprite_data
