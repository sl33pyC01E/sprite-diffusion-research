from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.opensurge import (
    EXPECTED_OPEN_SURGE_ARCHIVE_SHA256,
    OPEN_SURGE_COMMIT,
    OpenSurgeArchiveError,
    OpenSurgeSpriteParseError,
    audit_known_open_surge_archive,
    audit_open_surge_archive,
    classify_sprite_entity,
    interpret_action_comment,
    parse_copyright_data,
    parse_sprite_script,
)

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "14/48/1448d51c3e8bc8122ae893aad96e73f9c9ef1ffb74457109fa8cfa0ef10d0206"
)


def _png_bytes(size: tuple[int, int]) -> bytes:
    image = Image.new("P", size)
    image.putpalette([0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254)
    output = BytesIO()
    image.save(output, format="PNG", transparency=0)
    return output.getvalue()


def _synthetic_script() -> str:
    return """
// File: test.spr
// Description: parser fixture
// Author: Ada Artist
// License: MIT

// art by Pixel Person
sprite "Test Creature"
{
    source_file "images/test.png"
    source_rect 8 4 24 8
    frame_size 8 8
    hot_spot 4 8
    action_spot 6 4

    // charging
    animation 0
    {
        repeat TRUE
        fps 12.5
        data 0 1 1 2
        repeat_from 1
        action_spot 7 3
    }

    animation 7 // running
    {
        repeat TRUE
        fps 20
        data 2 1 0
    }

    // running to charging
    transition 7 to 0
    {
        repeat FALSE
        fps 10
        data 1 1
    }
}
"""


def _synthetic_archive(
    tmp_path: Path,
    *,
    shader_source: str = (
        '"const vec3 MASK_COLOR = vec3(1.0, 0.0, 1.0);\\n"\n"p *= float(p.rgb != MASK_COLOR);\\n"\n'
    ),
) -> Path:
    archive_path = tmp_path / "opensurge.zip"
    root = f"opensurge-{OPEN_SURGE_COMMIT}"
    copyright_csv = (
        "Type;File;License;Author;Website;Notes\n"
        "image;images/test.png;CC-BY-4.0;Ada Artist;example.test;fixture art\n"
    )
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/LICENSE", "GNU GENERAL PUBLIC LICENSE\nVersion 3\n")
        archive.writestr(f"{root}/README.md", "Open Surge test. License: GPLv3.\n")
        archive.writestr(f"{root}/licenses/MIT-license.txt", "MIT License\n")
        archive.writestr(f"{root}/src/misc/copyright_data.csv", copyright_csv)
        archive.writestr(
            f"{root}/src/core/sprite.c",
            "GPL version 3 or later; row-major sprite fixture.\n",
        )
        archive.writestr(
            f"{root}/src/core/animation.c",
            "GPL version 3 or later; repeat_from fixture.\n",
        )
        archive.writestr(
            f"{root}/src/core/color.c",
            "bool color_is_transparent(unsigned char r, unsigned char g, "
            "unsigned char b, unsigned char a) {\n"
            "    return (a == 0) || (r == 255 && g == 0 && b == 255);\n"
            "}\n",
        )
        archive.writestr(
            f"{root}/src/core/shader.c",
            shader_source,
        )
        archive.writestr(f"{root}/sprites/enemies/test.spr", _synthetic_script())
        archive.writestr(f"{root}/images/test.png", _png_bytes((40, 16)))
    return archive_path


def _sprite_map(audit: object) -> dict[str, object]:
    return {sprite.identity: sprite for sprite in audit.sprites}  # type: ignore[attr-defined]


def _animation(sprite: object, animation_id: int) -> object:
    matches = [
        animation
        for animation in sprite.animations  # type: ignore[attr-defined]
        if animation.animation_id == animation_id
    ]
    assert len(matches) == 1
    return matches[0]


def test_parser_preserves_timing_repeated_occurrences_loop_tail_and_anchors() -> None:
    (sprite,) = parse_sprite_script(
        _synthetic_script(),
        relative_path="sprites/enemies/test.spr",
        source_image_sizes={"images/test.png": (40, 16)},
    )
    charging = _animation(sprite, 0)

    assert sprite.identity == "Test Creature"
    assert (sprite.source_rect.x, sprite.source_rect.y) == (8, 4)
    assert (sprite.frame_size.x, sprite.frame_size.y) == (8, 8)
    assert (sprite.hot_spot.x, sprite.hot_spot.y) == (4, 8)
    assert (sprite.action_spot.x, sprite.action_spot.y) == (6, 4)
    assert sprite.source_rect_within_image is True
    assert sprite.asset_credit is None
    assert sprite.source_header_authors == ("Ada Artist",)
    assert sprite.source_header_licenses == ("MIT",)
    assert "art by Pixel Person" in sprite.source_comments

    assert charging.animation_id == 0
    assert charging.source_label == "charging"
    assert charging.normalized_action is None
    assert charging.normalized_action_basis == "unresolved_comment"
    assert charging.fps == 12.5
    assert charging.fps_source_token == "12.5"
    assert charging.data == (0, 1, 1, 2)
    assert charging.intro_data == (0,)
    assert charging.loop_data == (1, 1, 2)
    assert charging.repeat is True
    assert charging.repeat_from == charging.effective_repeat_from == 1
    assert charging.loop_mode == "intro_then_loop"
    assert charging.action_spot_overridden
    assert (charging.action_spot.x, charging.action_spot.y) == (7, 3)
    assert [frame.source_frame_index for frame in charging.frame_occurrences] == [0, 1, 1, 2]
    assert [frame.in_loop_tail for frame in charging.frame_occurrences] == [
        False,
        True,
        True,
        True,
    ]
    assert [
        (frame.left, frame.top, frame.right, frame.bottom) for frame in charging.frame_occurrences
    ] == [(8, 4, 16, 12), (16, 4, 24, 12), (16, 4, 24, 12), (24, 4, 32, 12)]

    running = _animation(sprite, 7)
    assert running.source_label == "running"
    assert running.source_label_basis == "inline_comment"
    assert running.normalized_action == "run"
    assert running.action_spot == sprite.action_spot
    assert not running.action_spot_overridden

    (transition,) = sprite.transitions
    assert transition.animation_id is None
    assert transition.transition_from == 7
    assert transition.transition_to == 0
    assert transition.transition_ordinal == 0
    assert transition.source_label == "running to charging"
    assert transition.normalized_action is None
    assert transition.data == (1, 1)
    assert transition.loop_mode == "one_shot"


def test_action_interpretation_is_explicit_and_unknown_labels_stay_unknown() -> None:
    assert interpret_action_comment("animal 17: running") == (
        "run",
        "structured_comment_mapping",
        None,
        "animal 17",
    )
    assert interpret_action_comment("left wing-flap")[:3] == (
        "fly",
        "exact_comment_mapping",
        "left",
    )
    assert interpret_action_comment("warming up") == (
        None,
        "unresolved_comment",
        None,
        None,
    )
    assert interpret_action_comment("action!")[0] is None


def test_comment_only_arrows_are_preserved_but_not_normalized() -> None:
    source = """
sprite X {
 source_file images/x.png
 source_rect 0 0 1 1
 frame_size 1 1
 // --->
 animation 0 {
  repeat TRUE
  fps 8
  data 0
 }
}
"""
    (sprite,) = parse_sprite_script(source, relative_path="sprites/legacy/items/x.spr")
    assert sprite.animations[0].source_label == "--->"
    assert sprite.animations[0].normalized_action is None


def test_parser_rejects_duplicate_animation_ids_and_missing_data() -> None:
    duplicate = _synthetic_script().replace("animation 7 // running", "animation 0 // running")
    with pytest.raises(OpenSurgeSpriteParseError, match="duplicate animation ID"):
        parse_sprite_script(duplicate, relative_path="sprites/enemies/test.spr")

    missing = _synthetic_script().replace("        data 0 1 1 2\n", "")
    with pytest.raises(OpenSurgeSpriteParseError, match="missing required 'data'"):
        parse_sprite_script(missing, relative_path="sprites/enemies/test.spr")


def test_copyright_manifest_preserves_asset_level_provenance() -> None:
    rows = parse_copyright_data(
        "Type;File;License;Author;Website;Notes\n"
        "image;images/wolf.png;CC-BY-3.0;A. Artist;example.test;Edited by B\n"
    )
    assert len(rows) == 1
    assert rows[0].file_path == "images/wolf.png"
    assert rows[0].license_expression == "CC-BY-3.0"
    assert rows[0].author == "A. Artist"
    assert rows[0].notes == "Edited by B"
    assert rows[0].line_number == 2


def test_entity_classification_marks_complete_characters_and_components_separately() -> None:
    wolf = classify_sprite_entity("sprites/bosses/giant_wolf.spr", "Giant Wolf")
    hand = classify_sprite_entity("sprites/bosses/giant_wolf.spr", "Giant Wolf's Hand")
    surge = classify_sprite_entity("sprites/players/surge.spr", "Surge")
    unknown = classify_sprite_entity("sprites/enemies/new_enemy.spr", "Future Enemy")

    assert wolf.primary_entity_class == "animal"
    assert wolf.subject_role == "boss_character"
    assert "quadruped" in wolf.morphology_tags
    assert hand.subject_role == "character_component"
    assert hand.parent_subject == "Giant Wolf"
    assert surge.entity_class_candidates == ("animal", "humanoid")
    assert "rabbit" in surge.morphology_tags
    assert unknown.primary_entity_class == "unknown"
    assert unknown.entity_class_candidates == ()


def test_synthetic_archive_joins_image_credit_and_reports_source_geometry(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    audit = audit_open_surge_archive(archive_path)
    sprite = audit.sprites[0]

    assert audit.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert audit.repository_commit == OPEN_SURGE_COMMIT
    assert audit.counts.sprite_script_file_count == 1
    assert audit.counts.sprite_definition_count == 1
    assert audit.counts.regular_animation_count == 2
    assert audit.counts.transition_count == 1
    assert audit.counts.invalid_transition_endpoint_count == 0
    assert audit.counts.frame_occurrence_count == 9
    assert audit.counts.repeated_frame_occurrence_count == 2
    assert audit.counts.unique_source_sheet_count == 1
    assert audit.counts.credited_unique_source_sheet_count == 1
    assert audit.counts.source_rect_out_of_image_count == 0
    assert sprite.asset_credit.license_expression == "CC-BY-4.0"
    assert sprite.asset_credit.author == "Ada Artist"
    assert audit.scripts[0].artwork_comments == ("art by Pixel Person",)
    assert audit.source_sheets[0].sha256 == hashlib.sha256(_png_bytes((40, 16))).hexdigest()
    assert audit.to_dict()["sprites"][0]["animations"][0]["data"] == (0, 1, 1, 2)
    evidence = {document.relative_path: document for document in audit.evidence_documents}
    assert evidence["src/core/color.c"].relevant_line_numbers == (2,)
    assert evidence["src/core/shader.c"].relevant_line_numbers == (1, 2)


def test_known_archive_rejects_an_unpinned_payload(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    with pytest.raises(OpenSurgeArchiveError, match="digest mismatch"):
        audit_known_open_surge_archive(archive_path)


def test_archive_audit_rejects_shader_without_exact_color_key_evidence(
    tmp_path: Path,
) -> None:
    archive_path = _synthetic_archive(tmp_path, shader_source="/* no color key */\n")
    with pytest.raises(OpenSurgeArchiveError, match="does not establish"):
        audit_open_surge_archive(archive_path)


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact CAS archive is not present")
def test_exact_cas_archive_audit() -> None:
    audit = audit_known_open_surge_archive(EXACT_ARCHIVE)
    sprites = _sprite_map(audit)

    assert audit.archive_sha256 == EXPECTED_OPEN_SURGE_ARCHIVE_SHA256
    assert audit.repository_commit == OPEN_SURGE_COMMIT
    assert audit.root_prefix == f"opensurge-{OPEN_SURGE_COMMIT}"
    assert audit.counts.zip_member_count == 1197
    assert audit.counts.file_member_count == 1066
    assert audit.counts.archive_png_file_count == 164
    assert audit.counts.sprite_script_file_count == 116
    assert audit.counts.sprite_script_with_header_license_count == 82
    assert audit.counts.sprite_definition_count == 357
    assert audit.counts.regular_animation_count == 893
    assert audit.counts.transition_count == 24
    assert audit.counts.invalid_transition_endpoint_count == 0
    assert audit.counts.total_timeline_count == 917
    assert audit.counts.frame_occurrence_count == 3540
    assert audit.counts.repeated_frame_occurrence_count == 1493
    assert audit.counts.repeat_true_count == 679
    assert audit.counts.repeat_false_count == 238
    assert audit.counts.repeat_from_declaration_count == 9
    assert audit.counts.comment_labeled_timeline_count == 708
    assert audit.counts.normalized_action_timeline_count == 236
    assert audit.counts.unresolved_action_timeline_count == 681

    assert audit.counts.source_sheet_reference_count == 357
    assert audit.counts.unique_source_sheet_count == 96
    assert audit.counts.missing_source_sheet_count == 0
    assert audit.counts.copyright_data_row_count == 285
    assert audit.counts.copyright_image_row_count == 135
    assert audit.counts.credited_unique_source_sheet_count == 96
    assert audit.counts.uncredited_unique_source_sheet_count == 0
    assert audit.counts.source_rect_out_of_image_count == 8
    assert audit.counts.source_rect_grid_incompatible_count == 1
    assert audit.counts.invalid_declared_frame_index_count == 0
    assert audit.counts.referenced_frame_out_of_image_occurrence_count == 3

    assert audit.counts.standalone_character_subject_count == 22
    assert audit.counts.enemy_character_subject_count == 11
    assert audit.counts.boss_character_subject_count == 4
    assert audit.counts.animal_character_subject_count == 17
    assert audit.counts.creature_character_subject_count == 5
    assert audit.counts.quadruped_character_subject_count == 4

    surge = sprites["Surge"]
    surge_run = _animation(surge, 2)
    assert surge.entity.primary_entity_class == "animal"
    assert surge.source_file == "images/players/surge.png"
    assert surge_run.source_label == "running"
    assert surge_run.normalized_action == "run"
    assert surge_run.fps == 20
    assert surge_run.data == (9, 10, 11, 12, 13, 14, 15, 16)
    assert surge_run.frame_occurrences[0].left == 64
    assert surge_run.frame_occurrences[0].top == 96

    surge_brake = _animation(surge, 7)
    assert surge_brake.source_label == "braking"
    assert surge_brake.normalized_action is None
    assert surge_brake.repeat_from == 2
    assert surge_brake.intro_data == (29, 30)
    assert surge_brake.loop_data == (31, 32)

    transition = next(
        item
        for item in surge.transitions
        if item.transition_from == 12 and item.transition_to == 32
    )
    assert transition.source_label == "from breathing to falling"
    assert transition.data == (58, 58, 58, 58, 58, 59)
    assert transition.effective_repeat is False

    wolf_head = sprites["Giant Wolf's Head"]
    wolf_hurt = _animation(wolf_head, 2)
    assert wolf_hurt.source_label == "got hit"
    assert wolf_hurt.normalized_action == "hurt"
    assert wolf_hurt.data == (2, 2, 1, 1, 1, 1, 1, 1, 1)
    assert wolf_head.asset_credit.license_expression == "CC-BY-3.0"
    assert wolf_head.asset_credit.author == "João Victor (Race the Hedgehog)"

    animals = sprites["Animal"]
    animal_17_run = _animation(animals, 35)
    assert animal_17_run.source_label == "animal 17: running"
    assert animal_17_run.normalized_action == "run"
    assert animal_17_run.source_variant_hint == "animal 17"
    assert animal_17_run.data == (52, 53)
    assert animals.source_rect_grid_compatible is False
    assert animals.asset_credit.license_expression == "CC-BY-3.0"
    assert "SecularSteve" in animals.asset_credit.notes

    assert {issue.code for issue in audit.issues} == {
        "declared_source_rect_exceeds_image",
        "source_rect_not_multiple_of_frame_size",
        "referenced_frame_cell_exceeds_image",
        "license_scope_is_asset_specific",
        "unresolved_action_comments_preserved",
    }
    image_licenses = {
        credit.license_expression for credit in audit.asset_credits if credit.asset_type == "image"
    }
    assert image_licenses == {
        "CC-BY-3.0",
        "CC-BY-4.0",
        "CC-BY-SA-3.0",
        "CC-BY-SA-4.0",
        "CC0-1.0",
        "Giftware",
        "MIT",
    }
    evidence = {document.relative_path: document for document in audit.evidence_documents}
    assert evidence["src/misc/copyright_data.csv"].sha256 == (
        "e606587ee6f597532bd34cab5f3d8df455f4a99f0f4bf849febb6a9b016556c8"
    )
    assert evidence["LICENSE"].sha256 == (
        "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
    )
    assert evidence["src/core/color.c"].sha256 == (
        "036b39ec0ba2fa42ef3b6dd5e16bb789e77e9684c008d3ea4147ebd51dbd19a4"
    )
    assert evidence["src/core/color.c"].relevant_line_numbers == (190,)
    assert evidence["src/core/shader.c"].sha256 == (
        "98c2cda978c67bc85f97b33b8692ab143d0fbb4aa113a0d3558cf5cba9d37dfa"
    )
    assert evidence["src/core/shader.c"].relevant_line_numbers == (111, 116)
