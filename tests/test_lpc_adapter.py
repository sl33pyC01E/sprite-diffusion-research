from __future__ import annotations

import json

import pytest

from spritelab.adapters.lpc import (
    LpcParseError,
    classify_lpc_path,
    credit_filename_candidates,
    group_credits_by_filename,
    parse_animation_layout,
    parse_credits_csv,
    parse_palette_definition,
    parse_sheet_definition,
    sheet_animation_cues,
)

ROOT = "Universal-LPC-Spritesheet-Character-Generator-deadbeef"


def test_classifies_action_recolor_and_builds_palette_independent_identity() -> None:
    info = classify_lpc_path(f"{ROOT}/spritesheets/body/tail/cat/adult/fg/halfslash/blonde.png")

    assert info.kind == "sheet"
    assert info.is_sheet_candidate
    assert info.repository_relative_path.startswith("spritesheets/")
    assert info.content_path == "body/tail/cat/adult/fg/halfslash/blonde.png"
    assert info.layer_identity == "ulpc:body/tail/cat/adult/fg"
    assert info.category == "body"
    assert info.source_action == "halfslash"
    assert info.normalized_action == "attack"
    assert info.body_type == "adult"
    assert info.plane == "fg"
    assert info.palette == "blonde"
    assert info.entity_family == "anthropomorphic_animal"


def test_classifies_plain_action_and_preserves_non_palette_suffix() -> None:
    plain = classify_lpc_path("spritesheets/arms/armour/plate/male/idle.png")
    suffix = classify_lpc_path("spritesheets/dress/kimono/sleeves/universal/shoot/female_front.png")

    assert plain.layer_identity == "ulpc:arms/armour/plate/male"
    assert plain.palette is None
    assert plain.normalized_action == "idle"
    assert suffix.layer_identity == "ulpc:dress/kimono/sleeves/universal/female_front"
    assert suffix.palette is None


def test_only_spritesheets_png_is_a_sheet_candidate() -> None:
    readme_image = classify_lpc_path(f"{ROOT}/readme-images/example.png")
    ui_image = classify_lpc_path(f"{ROOT}/sources/github-mark.png")
    tool_image = classify_lpc_path(f"{ROOT}/tools/layout/universal-expanded.png")
    credits = classify_lpc_path(f"{ROOT}/CREDITS.csv")
    definition = classify_lpc_path(f"{ROOT}/sheet_definitions/body/body.json")

    assert readme_image.kind == "documentation_asset"
    assert ui_image.kind == "ui_or_source"
    assert tool_image.kind == "tool_asset"
    assert credits.kind == "credits"
    assert definition.kind == "sheet_definition"
    assert not any(
        item.is_sheet_candidate
        for item in (readme_image, ui_image, tool_image, credits, definition)
    )


@pytest.mark.parametrize("path", ["../spritesheets/a.png", "/spritesheets/a.png", "C:/a.png"])
def test_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(LpcParseError):
        classify_lpc_path(path)


def test_parse_sheet_definition_retains_layers_actions_recolors_and_credit() -> None:
    fixture = {
        "name": "Cat Tail",
        "type_name": "tail",
        "priority": 12,
        "tags": ["animal"],
        "required_tags": ["adult"],
        "match_body_color": True,
        "layer_1": {
            "zPos": 5,
            "male": "body/tail/cat/adult/bg/",
            "female": "body/tail/cat/adult/bg/",
        },
        "layer_2": {
            "zPos": 105,
            "male": "body/tail/cat/adult/fg/",
            "female": "body/tail/cat/adult/fg/",
            "custom_animation": "slash_128",
            "is_mask": True,
        },
        "animations": ["walk", "run", "1h_halfslash"],
        "variants": ["fur_brown"],
        "aliases": {"Brown": "fur_brown"},
        "credits": [
            {
                "file": "body/tail/cat",
                "notes": "original and expanded",
                "authors": ["Artist One", "Artist Two"],
                "licenses": ["OGA-BY 3.0", "CC-BY 3.0"],
                "urls": ["https://example.test/source"],
            }
        ],
        "recolors": {
            "color_1": {
                "type_name": "fur",
                "label": "Fur",
                "material": "hair",
                "base": "fur_brown",
                "palettes": ["ulpc", "lpcr"],
            }
        },
    }

    definition = parse_sheet_definition(
        json.dumps(fixture), source_path=f"{ROOT}/sheet_definitions/body/tail_cat.json"
    )

    assert definition.name == "Cat Tail"
    assert definition.type_name == "tail"
    assert definition.body_types == ("female", "male")
    assert definition.animations == ("walk", "run", "1h_halfslash")
    assert definition.tags == ("animal",)
    assert definition.priority == 12
    assert definition.match_body_color
    assert definition.layers[1].custom_animation == "slash_128"
    assert definition.layers[1].is_mask
    assert definition.layers[1].path_for("male") == "body/tail/cat/adult/fg"
    assert definition.credits[0].authors == ("Artist One", "Artist Two")
    assert definition.credits[0].licenses == ("OGA-BY 3.0", "CC-BY 3.0")
    assert definition.recolor_rules[0].material == "hair"
    assert definition.recolor_rules[0].palettes == ("ulpc", "lpcr")


def test_parse_sheet_definition_rejects_missing_layers() -> None:
    with pytest.raises(LpcParseError, match="no layer_N"):
        parse_sheet_definition({"name": "Broken", "type_name": "body"})


def test_parse_credits_csv_trims_lists_and_preserves_duplicate_claims() -> None:
    payload = (
        "\ufefffilename,notes,authors,licenses,urls\r\n"
        '"body/bodies/male/walk.png","first, note","A One,B Two",'
        '"OGA-BY 3.0,CC-BY-SA 3.0","https://one.test,https://two.test"\r\n'
        '"body/bodies/male/walk.png","second","C Three","CC0","https://three.test"\r\n'
    )

    rows = parse_credits_csv(payload)
    grouped = group_credits_by_filename(rows)

    assert len(rows) == 2
    assert rows[0].notes == "first, note"
    assert rows[0].authors == ("A One", "B Two")
    assert rows[0].licenses == ("OGA-BY 3.0", "CC-BY-SA 3.0")
    assert len(grouped["body/bodies/male/walk.png"]) == 2


def test_credit_candidates_handle_recolors_aliases_and_plane_reordering() -> None:
    recolor = credit_filename_candidates("spritesheets/body/tail/cat/adult/fg/halfslash/blonde.png")
    after_action = credit_filename_candidates(
        "spritesheets/backpack/basket_contents/ore/hurt/bg.png"
    )
    custom = credit_filename_candidates(
        "spritesheets/weapon/sword/arming/attack_slash/bg/brass.png"
    )

    assert "body/tail/cat/adult/fg/1h_halfslash.png" in recolor
    assert "backpack/basket_contents/ore/bg/hurt.png" in after_action
    assert "weapon/sword/arming/attack_slash/bg/slash_128.png" in custom


def test_credit_candidates_support_definition_custom_animation_without_path_action() -> None:
    candidates = credit_filename_candidates(
        "spritesheets/body/wheelchair/adult/background/black.png",
        custom_animation="wheelchair",
    )
    assert "body/wheelchair/adult/background/wheelchair.png" in candidates


def test_parse_animation_layout_groups_rows_and_normalizes_directions() -> None:
    layout = {
        "frame_size": [64, 64],
        "size": [3, 2],
        "rows": [
            [
                {"name": "idle", "direction": "n", "frame": 0},
                {"name": "idle", "direction": "n", "frame": 1},
                None,
            ],
            [
                {"name": "hurt", "direction": "s", "frame": 0},
                {"name": "hurt", "direction": "s", "frame": 1},
                {"name": "hurt", "direction": "s", "frame": 2},
            ],
        ],
    }

    parsed = parse_animation_layout(layout)

    assert (parsed.frame_width, parsed.frame_height) == (64, 64)
    assert (parsed.columns, parsed.rows) == (3, 2)
    assert parsed.actions == ("idle", "hurt")
    assert parsed.direction_layouts[0].direction == "north"
    assert parsed.direction_layouts[0].frame_indices == (0, 1)
    assert parsed.direction_layouts[1].direction == "south"


def test_sheet_animation_cues_cover_directions_and_oversize_geometry() -> None:
    standard = sheet_animation_cues("spritesheets/body/bodies/male/run.png", width=512, height=256)
    oversize = sheet_animation_cues(
        "spritesheets/weapon/polearm/dragonspear/background/walk/brass.png",
        width=1664,
        height=512,
    )
    hurt = sheet_animation_cues("spritesheets/body/bodies/male/hurt.png", width=384, height=64)

    assert [cue.direction for cue in standard] == ["north", "west", "south", "east"]
    assert all(cue.frame_count == 8 and cue.frame_size == 64 for cue in standard)
    assert all(cue.loopable and cue.canonical_geometry for cue in standard)
    assert all(cue.frame_count == 13 and cue.frame_size == 128 for cue in oversize)
    assert all(not cue.canonical_geometry for cue in oversize)
    assert len(hurt) == 1 and hurt[0].direction == "south"
    assert len({cue.stable_id for cue in standard}) == 4


def test_sheet_animation_cues_reject_malformed_geometry_or_can_quarantine() -> None:
    path = "spritesheets/head/heads/skeleton/adult/halfslash.png"
    with pytest.raises(LpcParseError, match="not divisible"):
        sheet_animation_cues(path, width=384, height=254)
    assert sheet_animation_cues(path, width=384, height=254, strict=False) == ()


def test_parse_palette_definition_retains_ordered_ramps() -> None:
    parsed = parse_palette_definition(
        {
            "steel": ["#000000", "#FFFFFF"],
            "gold": ["#120000", "#FFD700"],
        },
        source_path=f"{ROOT}/palette_definitions/metal/metal_ulpc.json",
    )

    assert parsed.material == "metal"
    assert parsed.scheme == "ulpc"
    assert [palette.name for palette in parsed.palettes] == ["steel", "gold"]
    assert parsed.palettes[0].colors == ("#000000", "#FFFFFF")


def test_parse_palette_definition_rejects_meta_or_invalid_color() -> None:
    with pytest.raises(LpcParseError, match="non-meta"):
        parse_palette_definition(
            {"type": "material"},
            source_path="palette_definitions/metal/meta_metal.json",
        )
    with pytest.raises(LpcParseError, match="#RRGGBB"):
        parse_palette_definition(
            {"steel": ["transparent"]},
            source_path="palette_definitions/metal/metal_ulpc.json",
        )
