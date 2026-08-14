from __future__ import annotations

from dataclasses import dataclass

from spritelab.mugen_schema import (
    canonical_available_slot_action_numbers,
    canonical_core_motion_action_numbers,
    canonical_six_slot_action_numbers,
    measure_core_schema_coverage,
    schema_phase,
    schema_verb,
)


@dataclass(frozen=True)
class _Action:
    action_number: int


def test_schema_preserves_standard_verb_ranges() -> None:
    assert schema_verb(0) == "idle"
    assert schema_verb(20) == "walk"
    assert schema_verb(47) == "jump"
    assert schema_verb(130) == "block"
    assert schema_verb(200) == "normal_attack"
    assert schema_verb(799) == "normal_attack"
    assert schema_verb(1000) == "special_attack"
    assert schema_verb(2999) == "special_attack"
    assert schema_verb(3000) == "super_attack"
    assert schema_verb(4999) == "super_attack"
    assert schema_verb(800) is None
    assert schema_verb(5000) is None


def test_schema_retains_jump_and_block_phase() -> None:
    assert schema_phase(42) == "forward_up"
    assert schema_phase(45) == "forward_down"
    assert schema_phase(47) == "land"
    assert schema_phase(120) == "start_standing"
    assert schema_phase(130) == "hold_standing"
    assert schema_phase(140) == "end_standing"
    assert schema_phase(200) is None


def test_six_slot_view_prefers_standard_standing_actions() -> None:
    coverage = measure_core_schema_coverage(
        tuple(_Action(value) for value in (0, 20, 21, 40, 41, 42, 120, 130, 140, 201, 210, 600))
    )
    assert coverage.complete_six_slot_core is True
    assert canonical_six_slot_action_numbers(coverage) == {
        "attack_a": 210,
        "attack_b": 201,
        "block": 120,
        "idle": 0,
        "jump": 42,
        "walk": 20,
    }


def test_incomplete_character_remains_measurable_without_fake_slots() -> None:
    coverage = measure_core_schema_coverage(tuple(_Action(value) for value in (0, 20, 200)))
    assert coverage.complete_six_slot_core is False
    assert coverage.idle_action_numbers == (0,)
    assert coverage.walk_action_numbers == (20,)
    assert coverage.attack_action_numbers == (200,)
    assert canonical_six_slot_action_numbers(coverage) is None
    assert canonical_available_slot_action_numbers(coverage) == {
        "attack_a": 200,
        "idle": 0,
        "walk": 20,
    }


def test_core_motion_view_composes_standard_jump_and_guard_phases() -> None:
    coverage = measure_core_schema_coverage(
        tuple(_Action(value) for value in (0, 20, 40, 42, 45, 47, 120, 130, 140, 200, 210))
    )

    assert canonical_core_motion_action_numbers(coverage) == {
        "attack_a": (200,),
        "attack_b": (210,),
        "block": (120, 130, 140),
        "idle": (0,),
        "jump": (40, 42, 45, 47),
        "walk": (20,),
    }
