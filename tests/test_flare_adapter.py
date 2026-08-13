from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from spritelab.adapters.flare import (
    EXPECTED_FLARE_ARCHIVE_SHA256,
    FLARE_DIRECTION_NAMES,
    FLARE_ENGINE_COMMIT,
    FLARE_GAME_COMMIT,
    FlareArchiveError,
    FlareParseError,
    Rectangle,
    audit_flare_archive,
    audit_known_flare_archive,
    direction_index,
    engine_tick_schedule,
    parse_animation_definition,
    resolve_animation_definition,
)

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "9c/8e/9c8e58bb704f55174928990a1a375c213ffb2847d7b928d6117a84db3e1215cc"
)


def _png_bytes(size: tuple[int, int]) -> bytes:
    image = Image.new("RGBA", size, (255, 0, 0, 127))
    output = BytesIO()
    metadata = PngInfo()
    metadata.add_text("Software", "fixture-generator")
    image.save(output, format="PNG", pnginfo=metadata)
    return output.getvalue()


def _compressed_animation(image: str = "images/wolf.png") -> str:
    return f"""image={image}

[stance]
frames=2
duration=200ms
type=looped
active_frame=all
frame=0,SW,0,0,8,8,3,7
frame=1,0,8,0,8,8,4,7
"""


def _symlink_info(name: str) -> ZipInfo:
    info = ZipInfo(name)
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    return info


def _synthetic_archive(tmp_path: Path) -> Path:
    path = tmp_path / "flare.zip"
    root = f"flare-game-{FLARE_GAME_COMMIT}"
    layer_lines = "\n".join(
        f"layer={token},main,body" for token in ("SW", "W", "NW", "N", "NE", "E", "SE", "S")
    )
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/", b"")
        archive.writestr(f"{root}/README", "Art/data CC-BY-SA 3.0 or later.\n")
        archive.writestr(_symlink_info(f"{root}/README.md"), "README")
        archive.writestr(f"{root}/LICENSE.txt", "Attribution-ShareAlike 3.0 Unported\n")
        archive.writestr(f"{root}/CREDITS.txt", "Art\nFixture Artist\n")
        archive.writestr(
            f"{root}/mods/fantasycore/settings.txt",
            "description=fixture base\n",
        )
        archive.writestr(
            f"{root}/mods/empyrean_campaign/settings.txt",
            "requires=fantasycore\n",
        )
        archive.writestr(
            f"{root}/mods/fantasycore/animations/enemies/wolf.txt",
            _compressed_animation(),
        )
        archive.writestr(
            f"{root}/mods/fantasycore/animations/avatar/male/sword.txt",
            _compressed_animation("images/sword.png"),
        )
        archive.writestr(
            f"{root}/mods/fantasycore/images/wolf.png",
            _png_bytes((16, 8)),
        )
        archive.writestr(
            f"{root}/mods/fantasycore/images/sword.png",
            _png_bytes((16, 8)),
        )
        archive.writestr(
            f"{root}/mods/fantasycore/engine/hero_layers.txt",
            layer_lines,
        )
        archive.writestr(
            f"{root}/mods/empyrean_campaign/enemies/wolf.txt",
            "name=Fixture Wolf\ncategories=wolf,animal\nhumanoid=false\n"
            "animations=animations/enemies/wolf.txt\n",
        )
        archive.writestr(
            f"{root}/mods/empyrean_campaign/items/items.txt",
            "[item]\nid=7\nname=Fixture Sword\nitem_type=main\ngfx=sword\n"
            "loot_animation=animations/enemies/wolf.txt\n",
        )
        archive.writestr(
            f"{root}/mods/empyrean_campaign/powers/powers.txt",
            "[power]\nid=9\nname=Howl\nanimation=animations/enemies/wolf.txt\n",
        )
    return path


def test_parser_preserves_rectangles_offsets_order_timing_loop_and_direction_fallback() -> None:
    definition = parse_animation_definition(
        _compressed_animation(),
        source_path="animations/enemies/wolf.txt",
        image_sizes={"images/wolf.png": (16, 8)},
    )
    (action,) = definition.actions

    assert definition.entity_family == "enemy"
    assert definition.identity == "enemies/wolf"
    assert action.source_action == "stance"
    assert action.normalized_action == "idle"
    assert action.source_frame_order == (0, 1)
    assert action.duration_literal == "200ms"
    assert action.duration_milliseconds == 200
    assert action.nominal_fps == 10
    assert action.animation_type == "looped"
    assert action.loop_mode == "loop"
    assert action.active_frames == "all"
    assert action.default_tick_schedule.tick_count == 12
    assert action.default_tick_schedule.per_frame_tick_counts == (6, 6)

    southwest = action.direction_tracks[0]
    assert southwest.direction_name == "southwest"
    assert [slot.explicit for slot in southwest.frames] == [True, True]
    assert [slot.frame.rectangle for slot in southwest.frames if slot.frame] == [
        Rectangle(0, 0, 8, 8),
        Rectangle(8, 0, 8, 8),
    ]
    assert southwest.frames[0].frame is not None
    assert southwest.frames[0].frame.offset.x == 3
    assert southwest.frames[0].frame.within_image_bounds is True

    north = action.direction_tracks[3]
    assert [slot.explicit for slot in north.frames] == [False, False]
    assert [slot.fallback_from_direction for slot in north.frames] == [0, 0]
    assert [slot.frame.index for slot in north.frames if slot.frame] == [0, 1]
    assert all(track.complete for track in action.direction_tracks)


def test_include_resolution_uses_final_image_binding_and_retains_origins() -> None:
    files = {
        "animations/base.txt": _compressed_animation("images/base.png"),
        "animations/dark.txt": (
            "color_mod=40,50,60\nINCLUDE animations/base.txt\nimage=images/dark.png\n"
        ),
    }
    definition = resolve_animation_definition(
        "animations/dark.txt",
        files,
        image_sizes={"images/base.png": (16, 8), "images/dark.png": (16, 8)},
    )

    assert definition.color_mod == (40, 50, 60)
    assert [item.logical_path for item in definition.image_bindings] == [
        "images/base.png",
        "images/dark.png",
    ]
    assert definition.source_documents == ("animations/dark.txt", "animations/base.txt")
    assert definition.includes[0].included_path == "animations/base.txt"
    frame = definition.actions[0].direction_tracks[0].frames[0].frame
    assert frame is not None
    assert frame.image_path == "images/dark.png"
    assert frame.location.logical_path == "animations/base.txt"


def test_engine_tick_schedule_and_direction_mapping_match_paired_engine_source() -> None:
    schedule = engine_tick_schedule(8, "533ms")
    assert schedule.tick_rate == 60
    assert schedule.tick_count == 32
    assert schedule.per_frame_tick_counts == (4,) * 8
    assert schedule.effective_duration_milliseconds == pytest.approx(533.3333333333)

    uneven = engine_tick_schedule(4, "100ms")
    assert uneven.frame_indices == (0, 1, 1, 2, 2, 3)
    assert engine_tick_schedule(2, "1s").per_frame_tick_counts == (30, 30)
    assert engine_tick_schedule(2, "0ms").frame_indices == ()
    direction_tokens = ("SW", "W", "NW", "N", "NE", "E", "SE", "S")
    assert [direction_index(token) for token in direction_tokens] == list(range(8))
    assert tuple(FLARE_DIRECTION_NAMES) == (
        "southwest",
        "west",
        "northwest",
        "north",
        "northeast",
        "east",
        "southeast",
        "south",
    )


def test_parser_rejects_unsafe_includes_bad_directions_and_out_of_range_indices() -> None:
    with pytest.raises(FlareParseError, match="safe relative path"):
        parse_animation_definition("INCLUDE ../outside.txt\n", source_path="animations/test.txt")
    with pytest.raises(FlareParseError, match="invalid direction"):
        parse_animation_definition(
            _compressed_animation().replace("0,SW,", "0,SIDEWAYS,"),
            source_path="animations/test.txt",
        )
    with pytest.raises(FlareParseError, match="outside declared count"):
        parse_animation_definition(
            _compressed_animation().replace("frame=1,0,", "frame=2,0,"),
            source_path="animations/test.txt",
        )
    with pytest.raises(FlareParseError, match="recursive INCLUDE chain"):
        resolve_animation_definition(
            "animations/a.txt",
            {
                "animations/a.txt": "INCLUDE animations/b.txt\n",
                "animations/b.txt": "INCLUDE animations/a.txt\n",
            },
        )


def test_synthetic_archive_audit_is_read_only_and_preserves_bindings(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    audit = audit_flare_archive(archive_path)

    assert audit.repository_commit == FLARE_GAME_COMMIT
    assert audit.engine_semantics_commit == FLARE_ENGINE_COMMIT
    assert audit.active_mods == ("fantasycore", "empyrean_campaign")
    assert audit.counts.animation_definition_file_count == 2
    assert audit.counts.action_count == 2
    assert audit.counts.explicit_frame_record_count == 4
    assert audit.counts.direction_track_count == 16
    assert audit.counts.explicit_direction_track_count == 2
    assert audit.counts.fallback_only_direction_track_count == 14
    assert audit.counts.unresolved_direction_track_count == 0
    assert audit.counts.direction_zero_fallback_slot_count == 28
    assert audit.counts.complete_eight_direction_action_count == 2
    assert audit.counts.referenced_source_image_count == 2
    assert audit.counts.out_of_bounds_frame_record_count == 0
    assert audit.counts.entity_binding_count == 1
    assert audit.counts.attachment_binding_count == 1
    assert audit.counts.hero_layer_direction_count == 8
    assert audit.png_metadata.readable_png_count == 2
    assert audit.png_metadata.metadata_field_counts == (("Software", 2),)
    assert audit.png_metadata.png_with_attribution_field_count == 0
    assert audit.symlinks[0].target == "README"

    (entity,) = audit.entities
    assert entity.display_name == "Fixture Wolf"
    assert entity.categories == ("wolf", "animal")
    assert entity.humanoid is False
    assert entity.animation_paths == ("animations/enemies/wolf.txt",)
    (attachment,) = audit.attachments
    assert attachment.layer_slot == "main"
    assert attachment.candidate_animation_paths == ("animations/avatar/male/sword.txt",)
    assert [layer.direction for layer in audit.hero_layers] == list(range(8))

    with pytest.raises(FlareArchiveError, match="SHA-256"):
        audit_known_flare_archive(archive_path)


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact CAS archive is not present")
def test_exact_cas_archive_audit() -> None:
    audit = audit_known_flare_archive(EXACT_ARCHIVE)
    counts = audit.counts

    assert audit.archive_sha256 == EXPECTED_FLARE_ARCHIVE_SHA256
    assert audit.archive_size_bytes == 626_244_475
    assert audit.repository_commit == FLARE_GAME_COMMIT
    assert audit.engine_semantics_commit == FLARE_ENGINE_COMMIT
    assert audit.root_prefix == f"flare-game-{FLARE_GAME_COMMIT}"
    assert (
        counts.zip_member_count,
        counts.regular_file_member_count,
        counts.directory_member_count,
        counts.symlink_member_count,
    ) == (2_727, 2_499, 227, 1)
    assert counts.expanded_member_bytes == 661_089_998
    assert counts.expanded_regular_file_bytes == 661_089_992
    assert counts.archive_png_file_count == 721
    assert counts.active_mod_png_file_count == 414
    assert (
        counts.animation_definition_file_count,
        counts.fantasycore_animation_definition_count,
        counts.empyrean_animation_definition_count,
    ) == (330, 310, 20)
    assert counts.included_animation_definition_count == 61
    assert counts.physical_action_declaration_count == 1_541
    assert counts.physical_explicit_frame_record_count == 54_869
    assert counts.action_count == 1_996
    assert counts.exact_geometry_action_count == 1_980
    assert counts.geometry_missing_action_count == 16
    assert counts.direction_track_count == 15_968
    assert counts.explicit_direction_track_count == 15_799
    assert counts.fallback_only_direction_track_count == 41
    assert counts.unresolved_direction_track_count == 128
    assert counts.explicit_frame_record_count == 70_897
    assert counts.effective_frame_slot_count == 71_608
    assert counts.explicit_frame_slot_count == 70_897
    assert counts.direction_zero_fallback_slot_count == 167
    assert counts.unresolved_frame_slot_count == 544
    assert counts.complete_eight_direction_action_count == 1_980
    assert (
        counts.play_once_action_count,
        counts.looped_action_count,
        counts.back_forth_action_count,
    ) == (1_474, 275, 247)
    assert counts.referenced_source_image_count == 296
    assert counts.missing_source_image_count == 0
    assert counts.out_of_bounds_frame_record_count == 0
    assert (
        counts.entity_binding_count,
        counts.concrete_entity_binding_count,
        counts.template_entity_binding_count,
    ) == (177, 144, 33)
    assert counts.enemy_binding_count == 156
    assert counts.npc_binding_count == 21
    assert counts.explicit_humanoid_binding_count == 77
    assert counts.animation_usage_count == 976
    assert counts.attachment_binding_count == 349
    assert counts.avatar_attachment_definition_count == 196
    assert counts.attachment_parent_mismatch_count == 0
    assert counts.hero_layer_direction_count == 8
    assert counts.evidence_document_count == 9
    assert audit.definition_family_counts == (
        ("avatar", 1),
        ("avatar_attachment", 196),
        ("enemy", 36),
        ("hero_parent", 1),
        ("loot", 39),
        ("npc", 15),
        ("power", 42),
    )
    assert audit.body_variant_counts == (("female", 65), ("female_dark", 65), ("male", 66))
    assert audit.usage_counts == (
        ("effect", 13),
        ("enemy", 156),
        ("item_loot", 558),
        ("npc", 21),
        ("power", 228),
    )

    assert audit.action_counts == (
        ("block", 229),
        ("cast", 232),
        ("cast_alt", 1),
        ("critdie", 34),
        ("dash_attack", 5),
        ("die", 233),
        ("hit", 232),
        ("power", 81),
        ("run", 230),
        ("run_alt", 4),
        ("shield_bash", 1),
        ("shoot", 227),
        ("spawn", 9),
        ("stance", 249),
        ("swing", 229),
    )
    assert audit.direction_explicit_frame_counts == (
        ("southwest", 8_883),
        ("west", 8_872),
        ("northwest", 8_872),
        ("north", 8_854),
        ("northeast", 8_854),
        ("east", 8_854),
        ("southeast", 8_854),
        ("south", 8_854),
    )

    assert audit.png_metadata.png_file_count == 721
    assert audit.png_metadata.readable_png_count == 721
    assert audit.png_metadata.png_with_text_count == 79
    assert audit.png_metadata.png_with_comment_count == 39
    assert audit.png_metadata.gimp_comment_count == 39
    assert audit.png_metadata.png_with_source_file_field_count == 16
    assert audit.png_metadata.png_with_attribution_field_count == 0
    assert audit.png_metadata.png_with_software_field_count == 18
    assert audit.png_metadata.metadata_field_counts == (
        ("Camera", 16),
        ("Comment", 39),
        ("Date", 16),
        ("File", 16),
        ("Frame", 16),
        ("RenderTime", 16),
        ("Scene", 16),
        ("Software", 18),
        ("Time", 16),
        ("chromaticity", 18),
        ("dpi", 135),
        ("exif", 16),
        ("gamma", 35),
        ("icc_profile", 1),
        ("srgb", 25),
        ("transparency", 6),
    )
    assert audit.symlinks[0].relative_path == "README.md"
    assert audit.symlinks[0].target == "README"

    hobgoblin = next(
        item
        for item in audit.definitions
        if item.logical_path == "animations/enemies/hobgoblin.txt"
    )
    stance = next(action for action in hobgoblin.actions if action.source_action == "stance")
    run = next(action for action in hobgoblin.actions if action.source_action == "run")
    assert (stance.declared_frame_count, stance.duration_literal, stance.animation_type) == (
        4,
        "800ms",
        "back_forth",
    )
    assert stance.direction_tracks[0].frames[0].frame is not None
    assert stance.direction_tracks[0].frames[0].frame.rectangle == Rectangle(956, 128, 83, 91)
    assert run.default_tick_schedule.per_frame_tick_counts == (4,) * 8
    assert all(
        frame.within_image_bounds is True
        for definition in audit.definitions
        for action in definition.actions
        for frame in action.raw_frames
    )

    ghost = next(
        item
        for item in audit.definitions
        if item.logical_path == "animations/enemies/zombie_ghost.txt"
    )
    assert ghost.color_mod == (191, 255, 191)
    assert ghost.alpha_mod == 127
    assert ghost.source_documents == (
        f"flare-game-{FLARE_GAME_COMMIT}/mods/empyrean_campaign/animations/enemies/"
        "zombie_ghost.txt",
        f"flare-game-{FLARE_GAME_COMMIT}/mods/fantasycore/animations/enemies/zombie.txt",
    )

    root_license = next(
        item for item in audit.evidence_documents if item.relative_path == "LICENSE.txt"
    )
    assert root_license.sha256 == "3f941b3b89cf7b8370ceb83cc76d2120d471b58735d8ca60238a751a48d7f72f"
    assert root_license.detected_license_identifiers == ("CC-BY-SA-3.0",)
    assert {issue.code for issue in audit.issues} == {
        "usage_outside_snapshot_animation_set",
        "timeline_without_sheet_geometry",
        "no_per_asset_credit_manifest",
    }
