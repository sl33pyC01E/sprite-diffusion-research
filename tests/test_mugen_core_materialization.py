from __future__ import annotations

import hashlib

from spritelab.adapters.mugen import MugenSffV1Sprite, parse_air
from spritelab.mugen_core_materialization import select_mugen_core_materializations


def _sprite(group: int, image: int, rgba: bytes) -> MugenSffV1Sprite:
    return MugenSffV1Sprite(
        archive_index=image,
        group_number=group,
        image_number=image,
        axis_x=0,
        axis_y=0,
        width=1,
        height=1,
        linked_sprite_index=None,
        palette_reuse=False,
        indices=b"\x01",
        palette_rgb=bytes(768),
        rgba=rgba,
        indices_sha256=hashlib.sha256(b"\x01").hexdigest(),
        palette_sha256=hashlib.sha256(bytes(768)).hexdigest(),
        rgba_sha256=hashlib.sha256(rgba).hexdigest(),
    )


def test_core_selection_stops_after_two_pixel_distinct_attacks() -> None:
    actions = parse_air(
        """
[Begin Action 0]
0,0,0,0,1
[Begin Action 20]
0,1,0,0,1
[Begin Action 42]
0,2,0,0,1
[Begin Action 120]
0,3,0,0,1
[Begin Action 200]
0,4,0,0,1
[Begin Action 201]
0,5,0,0,1
[Begin Action 202]
0,6,0,0,1
"""
    )
    red = bytes((255, 0, 0, 255))
    sprites = tuple(
        _sprite(0, image, red if image in {4, 5} else bytes((image, image, image, 255)))
        for image in range(7)
    )

    plan = select_mugen_core_materializations(actions, sprites)

    assert [row.slot for row in plan.selected] == [
        "idle",
        "walk",
        "jump",
        "block",
        "attack_a",
        "attack_b",
    ]
    assert [row.materialized.action_number for row in plan.selected[-2:]] == [200, 202]


def test_core_selection_retains_attempt_failures_without_dropping_character() -> None:
    actions = parse_air(
        "[Begin Action 0]\n9,9,0,0,1\n"
        "[Begin Action 20]\n0,0,0,0,1\n"
        "[Begin Action 200]\n0,0,0,0,1\n"
        "[Begin Action 210]\n0,1,0,0,1\n"
    )
    sprites = (
        _sprite(0, 0, bytes((1, 2, 3, 255))),
        _sprite(0, 1, bytes((3, 2, 1, 255))),
    )

    plan = select_mugen_core_materializations(actions, sprites)

    assert [row.slot for row in plan.selected] == ["walk", "attack_a", "attack_b"]
    assert plan.exclusions[0].exclusion.reason == "missing_sprite"
    assert plan.exclusions[0].source_action_index == 0


def test_core_selection_can_fail_closed_with_only_exact_exclusions() -> None:
    actions = parse_air("[Begin Action 0]\n9,9,0,0,1\n[Begin Action 200]\n8,8,0,0,1\n")

    plan = select_mugen_core_materializations(actions, ())

    assert plan.selected == ()
    assert [row.exclusion.reason for row in plan.exclusions] == [
        "missing_sprite",
        "missing_sprite",
    ]
    assert [row.source_action_index for row in plan.exclusions] == [0, 1]
