from __future__ import annotations

import hashlib
import json
import stat
import struct
from dataclasses import FrozenInstanceError
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from spritelab.adapters.wesnoth import (
    EXPECTED_WESNOTH_ARCHIVE_SHA256,
    WESNOTH_COMMIT,
    WesnothArchiveError,
    audit_known_wesnoth_archive,
    audit_wesnoth_archive,
    expand_image_expression,
    known_wesnoth_cas_path,
    primary_declarations_for,
)

PINNED_COUNTS = {
    "archive_members": 30_482,
    "archive_files": 29_038,
    "cfg_files": 2_618,
    "unit_cfg_files": 620,
    "png_files": 20_264,
    "image_files": 21_257,
    "unit_type_declarations": 600,
    "entity_records": 600,
    "unique_unit_ids": 573,
    "duplicate_unit_id_groups": 21,
    "duplicate_unit_id_excess": 27,
    "unresolved_entity_ids": 0,
    "base_unit_inheritances": 64,
    "animation_records": 1_677,
    "variant_animation_records": 163,
    "looping_animation_records": 301,
    "one_shot_animation_records": 1_376,
    "animations_with_primary_frames": 1_604,
    "safe_primary_animations": 604,
    "quarantined_primary_animations": 1_000,
    "primary_frame_declarations": 3_171,
    "expanded_primary_frames": 8_640,
    "resolved_primary_frames": 8_597,
    "unresolved_primary_frames": 43,
    "transformed_primary_frames": 41,
    "safe_primary_frame_occurrences": 2_526,
    "unique_resolved_primary_image_members": 4_968,
    "unique_safe_primary_image_members": 1_275,
    "auxiliary_frame_declarations": 883,
    "conditional_animations": 140,
    "macro_affected_animations": 1_004,
}
PINNED_AUDIT_SHA256 = "21c4d48184cb99495609de0c238a107d23e9c19d4000009df4f6893447b1e9e8"


def _png_header(width: int = 2, height: int = 3) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
    )


def _fixture_wml() -> str:
    return """\
[unit_type]
    id=Fixture Rider
    name= _ "Fixture Rider"
    race=human
    image="units/humans/fixture.png"
    profile="portraits/fixture.webp"
    [base_unit]
        id=Parent Unit
    [/base_unit]
    [standing_anim]
        start_time=0
        [if]
            direction=s,se,sw
            [frame]
                image="units/humans/fixture-stand-s-[1~2].png:100"
                offset="0~0.1"
            [/frame]
        [/if]
        [else]
            direction=n,ne,nw
            [frame]
                image="units/humans/fixture-stand-n-[1~2].png:[80,120]"
            [/frame]
        [/else]
    [/standing_anim]
    [movement_anim]
        direction=ne,nw
        [frame]
            image="units/humans/fixture-move-[1~2].png:[80,120]"
            layer=42
            directional_x=3
        [/frame]
    [/movement_anim]
    [attack_anim]
        [filter_attack]
            name=sword
            range=melee
        [/filter_attack]
        start_time=-100
        [frame]
            image="units/humans/fixture-attack-[1~2].png:50~RC(magenta>red)"
        [/frame]
        [missile_frame]
            image="projectiles/fixture.png:70"
            image_diagonal="projectiles/fixture-ne.png:70"
        [/missile_frame]
        {SOUND:HIT sword.ogg 0}
    [/attack_anim]
    [variation]
        variation_id=mounted
        [standing_anim]
            [frame]
                image="units/humans/fixture-mounted.png"
            [/frame]
        [/standing_anim]
    [/variation]
[/unit_type]
"""


def _make_fixture_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / "wesnoth-fixture.zip"
    root = "wesnoth-fixture"
    pngs = (
        "data/core/images/units/humans/fixture.png",
        "data/core/images/units/humans/fixture-stand-s-1.png",
        "data/core/images/units/humans/fixture-stand-s-2.png",
        "data/core/images/units/humans/fixture-stand-n-1.png",
        "data/core/images/units/humans/fixture-stand-n-2.png",
        "data/core/images/units/humans/fixture-move-1.png",
        "data/core/images/units/humans/fixture-move-2.png",
        "data/core/images/units/humans/fixture-attack-1.png",
        "data/core/images/units/humans/fixture-attack-2.png",
        "data/core/images/units/humans/fixture-mounted.png",
        "data/core/images/projectiles/fixture.png",
        "data/core/images/projectiles/fixture-ne.png",
        "data/core/images/portraits/fixture.png",
        "data/core/images/halo/fixture.png",
        "data/core/images/terrain/fixture.png",
        "data/core/images/icons/fixture.png",
        "data/core/images/maps/fixture.png",
    )
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{root}/data/core/units/humans/Fixture.cfg", _fixture_wml())
        archive.writestr(
            f"{root}/README.md",
            "Most art is GPL v2+; newer contributions are CC BY-SA 4.0.",
        )
        archive.writestr(f"{root}/COPYING", "GNU GENERAL PUBLIC LICENSE Version 2")
        archive.writestr(f"{root}/data/COPYING.txt", "GNU GENERAL PUBLIC LICENSE Version 2")
        archive.writestr(
            f"{root}/copyrights.csv",
            "Date,File,License,Author - Real Name(other name);Real Name(other name);etc,"
            "Notes,Needs Update,MD5\n"
            "2026/01/01,data/core/music/example.ogg,CC0,Artist,,,,\n",
        )
        archive.writestr(f"{root}/src/units/animation.cpp", 'anim["cycles"] = true;')
        archive.writestr(f"{root}/src/units/frame.cpp", "result.auto_hflip = true;")
        for ordinal, png in enumerate(pngs):
            archive.writestr(f"{root}/{png}", _png_header(ordinal + 1, ordinal + 2))
    return archive_path


def test_progressive_image_expansion_matches_runtime_subset() -> None:
    expanded = expand_image_expression("units/wolf-[1~3].png:[80,100,120]")
    assert tuple(frame.logical_path for frame in expanded) == (
        "units/wolf-1.png",
        "units/wolf-2.png",
        "units/wolf-3.png",
    )
    assert tuple(frame.duration_milliseconds for frame in expanded) == (80, 100, 120)
    assert all(frame.exact_timing for frame in expanded)

    padded = expand_image_expression("units/wolf-[01~03].png:50")
    assert tuple(frame.logical_path for frame in padded) == (
        "units/wolf-01.png",
        "units/wolf-02.png",
        "units/wolf-03.png",
    )
    assert tuple(frame.duration_milliseconds for frame in padded) == (50, 50, 50)

    parallel = expand_image_expression("a[1,2]b[3~4].png:25")
    assert tuple(frame.logical_path for frame in parallel) == ("a1b3.png", "a2b4.png")

    repeated = expand_image_expression("a[1*3].png:12")
    assert tuple(frame.logical_path for frame in repeated) == ("a1.png",) * 3


def test_outer_duration_uses_engine_residual_time_chunk() -> None:
    evenly_divided = expand_image_expression(
        "units/wolf-[1~3].png", explicit_duration_literal="300"
    )
    assert tuple(frame.duration_milliseconds for frame in evenly_divided) == (100, 100, 100)

    mixed = expand_image_expression(
        "units/wolf-[1~2].png:40,units/wolf-3.png",
        explicit_duration_literal="200",
    )
    # Engine divides (200 - 80 specified ms) by all three items. Inline
    # durations remain 40 ms and the unspecified item receives that chunk.
    assert tuple(frame.duration_milliseconds for frame in mixed) == (40, 40, 40)


def test_image_functions_and_invalid_expressions_are_not_flattened() -> None:
    transformed = expand_image_expression("units/wolf.png~RC(magenta>red):100")
    assert transformed[0].logical_path == "units/wolf.png"
    assert transformed[0].inline_modifiers == "~RC(magenta>red)"

    macro = expand_image_expression("{BASE_IMAGE}:100")
    assert macro[0].logical_path is None
    assert "unexpanded_image_variable_or_macro" in macro[0].issues

    bad = expand_image_expression("units/wolf-[1~3].png:[20,30]")
    assert all(not frame.exact_timing for frame in bad)
    assert all(frame.duration_milliseconds is None for frame in bad)


def test_fixture_audit_preserves_actions_layers_facing_and_quarantine(tmp_path: Path) -> None:
    archive_path = _make_fixture_archive(tmp_path)
    audit = audit_wesnoth_archive(archive_path)

    assert audit.archive_root == "wesnoth-fixture"
    assert audit.counts.entity_records == 1
    assert audit.counts.animation_records == 4
    assert audit.counts.animations_with_primary_frames == 4
    assert audit.counts.safe_primary_animations == 2
    assert audit.counts.conditional_animations == 1
    assert audit.counts.macro_affected_animations == 1
    assert audit.counts.auxiliary_frame_declarations == 2
    assert audit.counts.entity_class_counts == (("humanoid", 1),)
    assert audit.counts.action_counts == (("attack", 1), ("idle", 2), ("move", 1))

    entity = audit.entities[0]
    assert entity.unit_id == "Fixture Rider"
    assert entity.base_unit_ids == ("Parent Unit",)
    assert entity.unresolved_inheritance
    assert entity.entity_class == "humanoid"

    standing = next(
        animation
        for animation in entity.animations
        if animation.source_tag == "standing_anim" and not animation.variant_path
    )
    assert standing.normalized_action == "idle"
    assert standing.effective_cycles
    assert standing.loop_mode == "loop"
    assert standing.loop_basis == "engine_forces_standing_cycles"
    assert not standing.primary_timeline_exact
    assert "conditional_runtime_track" in standing.quarantine_reasons
    assert {declaration.directions for declaration in primary_declarations_for(standing)} == {
        ("s", "se", "sw"),
        ("n", "ne", "nw"),
    }

    movement = next(
        animation for animation in entity.animations if animation.source_tag == "movement_anim"
    )
    assert movement.normalized_action == "move"
    assert movement.loop_mode == "one_shot"
    assert movement.safe_primary_source_sequence
    movement_frame = primary_declarations_for(movement)[0]
    assert movement_frame.directions == ("ne", "nw")
    assert movement_frame.layer_literal == "42"
    assert movement_frame.directional_x_literal == "3"
    assert {
        (attribute.name, attribute.value) for attribute in movement_frame.context_attributes
    } >= {
        ("direction", "ne,nw"),
        ("layer", "42"),
        ("directional_x", "3"),
    }
    assert movement_frame.effective_auto_hflip
    assert not movement_frame.effective_auto_vflip
    assert tuple(frame.duration_milliseconds for frame in movement_frame.frames) == (80, 120)
    assert all(frame.resolution.width is not None for frame in movement_frame.frames)

    attack = next(
        animation for animation in entity.animations if animation.source_tag == "attack_anim"
    )
    assert attack.attack_name_filters == ("sword",)
    assert attack.attack_range_filters == ("melee",)
    assert not attack.safe_primary_source_sequence
    assert "unexpanded_wml_macro" in attack.quarantine_reasons
    primary = primary_declarations_for(attack)[0]
    assert "inline_image_path_function" in primary.quarantine_reasons
    roles = {declaration.render_role for declaration in attack.frame_declarations}
    assert roles == {"primary_unit", "projectile"}

    mounted = next(animation for animation in entity.animations if animation.variant_path)
    assert mounted.variant_path[0].startswith("variation:mounted@")
    assert mounted.safe_primary_source_sequence


def test_fixture_rights_are_repository_scoped_not_promoted_per_asset(tmp_path: Path) -> None:
    audit = audit_wesnoth_archive(_make_fixture_archive(tmp_path))
    assert audit.rights.repository_license_expression == "GPL-2.0-or-later"
    assert audit.rights.per_asset_license is None
    assert audit.rights.per_asset_attribution is None
    assert audit.rights.copyrights_csv_rows == 1
    assert audit.rights.copyrights_csv_image_rows == 0
    assert {evidence.evidence_scope for evidence in audit.rights.evidence} == {
        "repository_and_art_collection",
        "repository",
        "data_tree",
        "listed_exception_files_only",
    }


def test_records_are_immutable_and_canonical_serialization_is_stable(tmp_path: Path) -> None:
    archive_path = _make_fixture_archive(tmp_path)
    first = audit_wesnoth_archive(archive_path)
    second = audit_wesnoth_archive(archive_path)
    assert first == second
    assert first.canonical_json() == second.canonical_json()
    assert json.loads(first.canonical_json())["audit_record_sha256"] == first.audit_record_sha256
    with pytest.raises(FrozenInstanceError):
        first.commit = "changed"  # type: ignore[misc]


def test_archive_gate_rejects_unsafe_member(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    info = ZipInfo("root/link")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(path, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(WesnothArchiveError, match="non-regular archive member"):
        audit_wesnoth_archive(path)


def test_known_archive_gate_rejects_wrong_digest(tmp_path: Path) -> None:
    path = _make_fixture_archive(tmp_path)
    with pytest.raises(WesnothArchiveError, match="SHA-256 mismatch"):
        audit_known_wesnoth_archive(path)


def test_known_cas_path_uses_four_character_sharding() -> None:
    path = known_wesnoth_cas_path(Path("data/raw"))
    assert path == (
        Path("data/raw")
        / "objects"
        / "sha256"
        / EXPECTED_WESNOTH_ARCHIVE_SHA256[:2]
        / EXPECTED_WESNOTH_ARCHIVE_SHA256[2:4]
        / EXPECTED_WESNOTH_ARCHIVE_SHA256
    )


def test_exact_pinned_cas_archive_when_available() -> None:
    path = known_wesnoth_cas_path(Path("data/raw"))
    if not path.is_file():
        pytest.skip("pinned Wesnoth CAS archive is not present")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EXPECTED_WESNOTH_ARCHIVE_SHA256
    audit = audit_known_wesnoth_archive(path)
    assert audit.commit == WESNOTH_COMMIT
    for field_name, expected in PINNED_COUNTS.items():
        assert getattr(audit.counts, field_name) == expected
    assert audit.counts.action_counts == (
        ("attack", 931),
        ("death", 70),
        ("defend", 105),
        ("emote", 46),
        ("heal", 26),
        ("idle", 366),
        ("move", 98),
        ("move_transition", 4),
        ("spawn", 13),
        ("teleport", 6),
        ("unknown", 12),
    )
    assert audit.counts.entity_class_counts == (
        ("animal", 18),
        ("construct", 18),
        ("creature", 78),
        ("humanoid", 217),
        ("monster", 76),
        ("undead", 50),
        ("unknown", 132),
        ("vehicle", 11),
    )
    assert audit.rights.copyrights_csv_rows == 396
    assert audit.rights.copyrights_csv_image_rows == 0
    assert audit.audit_record_sha256 == PINNED_AUDIT_SHA256
