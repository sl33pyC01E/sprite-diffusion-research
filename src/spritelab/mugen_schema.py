"""MUGEN AIR schema semantics independent of any model geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class NumberedAction(Protocol):
    action_number: int


@dataclass(frozen=True, slots=True)
class MugenCoreSchemaCoverage:
    """Availability of standard fighter verbs in one AIR definition."""

    attack_action_numbers: tuple[int, ...]
    block_action_numbers: tuple[int, ...]
    idle_action_numbers: tuple[int, ...]
    jump_action_numbers: tuple[int, ...]
    walk_action_numbers: tuple[int, ...]

    @property
    def complete_six_slot_core(self) -> bool:
        return bool(
            self.idle_action_numbers
            and self.walk_action_numbers
            and self.jump_action_numbers
            and self.block_action_numbers
            and len(self.attack_action_numbers) >= 2
        )


def schema_verb(action_number: int) -> str | None:
    """Map standard AIR number ranges to corpus verbs without collapsing variants."""

    if action_number == 0:
        return "idle"
    if action_number in {20, 21}:
        return "walk"
    if 40 <= action_number <= 47:
        return "jump"
    if 120 <= action_number <= 155:
        return "block"
    if 200 <= action_number <= 799:
        return "normal_attack"
    if 1000 <= action_number <= 2999:
        return "special_attack"
    if 3000 <= action_number <= 4999:
        return "super_attack"
    return None


def schema_phase(action_number: int) -> str | None:
    """Retain standard sub-action phase instead of treating every verb as a loop."""

    phases = {
        40: "start",
        41: "neutral_up",
        42: "forward_up",
        43: "backward_up",
        44: "neutral_down",
        45: "forward_down",
        46: "backward_down",
        47: "land",
        120: "start_standing",
        121: "start_crouching",
        122: "start_air",
        130: "hold_standing",
        131: "hold_crouching",
        132: "hold_air",
        140: "end_standing",
        141: "end_crouching",
        142: "end_air",
        150: "hit_standing",
        151: "hit_crouching",
        152: "hit_air",
    }
    return phases.get(action_number)


def measure_core_schema_coverage(actions: tuple[NumberedAction, ...]) -> MugenCoreSchemaCoverage:
    """Measure standard slots while retaining every distinct authored action number."""

    numbers = tuple(sorted({action.action_number for action in actions}))
    return MugenCoreSchemaCoverage(
        attack_action_numbers=tuple(number for number in numbers if 200 <= number <= 799),
        block_action_numbers=tuple(number for number in numbers if 120 <= number <= 155),
        idle_action_numbers=tuple(number for number in numbers if number == 0),
        jump_action_numbers=tuple(number for number in numbers if 40 <= number <= 47),
        walk_action_numbers=tuple(number for number in numbers if number in {20, 21}),
    )


def canonical_six_slot_action_numbers(
    coverage: MugenCoreSchemaCoverage,
) -> dict[str, int] | None:
    """Select a deterministic six-slot view without changing the authoritative corpus."""

    if not coverage.complete_six_slot_core:
        return None
    return canonical_available_slot_action_numbers(coverage)


def canonical_available_slot_action_numbers(
    coverage: MugenCoreSchemaCoverage,
) -> dict[str, int]:
    """Select every available canonical slot while leaving absent slots absent."""

    output: dict[str, int] = {}
    if coverage.idle_action_numbers:
        output["idle"] = 0
    if coverage.walk_action_numbers:
        output["walk"] = _first_present(coverage.walk_action_numbers, (20, 21))
    if coverage.jump_action_numbers:
        output["jump"] = _first_present(
            coverage.jump_action_numbers, (42, 41, 43, 46, 40, 45, 47, 44)
        )
    if coverage.block_action_numbers:
        output["block"] = _first_present(
            coverage.block_action_numbers,
            (120, 130, 140, 121, 131, 141, 122, 132, 142, 150, 151, 152),
        )
    attacks = ordered_attack_action_numbers(coverage.attack_action_numbers)
    if attacks:
        output["attack_a"] = attacks[0]
    if len(attacks) >= 2:
        output["attack_b"] = attacks[1]
    return output


def canonical_core_motion_action_numbers(
    coverage: MugenCoreSchemaCoverage,
) -> dict[str, tuple[int, ...]]:
    """Build action-number plans, composing MUGEN's authored jump/guard phases.

    This is a derived training view only. Every source action remains separately
    addressable in the authoritative AIR catalog.
    """

    scalar = canonical_available_slot_action_numbers(coverage)
    output = {
        slot: (action_number,)
        for slot, action_number in scalar.items()
        if slot not in {"jump", "block"}
    }
    jumps = set(coverage.jump_action_numbers)
    if jumps:
        if 42 in jumps:
            preference = (40, 42, 45, 47)
        elif 41 in jumps:
            preference = (40, 41, 44, 47)
        elif 43 in jumps:
            preference = (40, 43, 46, 47)
        else:
            preference = tuple(sorted(jumps))
        output["jump"] = tuple(number for number in preference if number in jumps)
    blocks = set(coverage.block_action_numbers)
    if blocks:
        standing = tuple(number for number in (120, 130, 140) if number in blocks)
        crouching = tuple(number for number in (121, 131, 141) if number in blocks)
        aerial = tuple(number for number in (122, 132, 142) if number in blocks)
        output["block"] = standing or crouching or aerial or (min(blocks),)
    return output


def ordered_attack_action_numbers(action_numbers: tuple[int, ...]) -> tuple[int, ...]:
    """Order standard attacks for canonical views while retaining all source variants."""

    return tuple(sorted(set(action_numbers), key=_attack_preference))


def _first_present(values: tuple[int, ...], preference: tuple[int, ...]) -> int:
    available = set(values)
    for value in preference:
        if value in available:
            return value
    return values[0]


def _attack_preference(action_number: int) -> tuple[int, int, int]:
    # Standard standing attacks are preferred for the two canonical view slots.
    stance_rank = 0 if 200 <= action_number <= 399 else 1 if action_number <= 599 else 2
    conventional_rank = 0 if action_number % 10 == 0 else 1
    return stance_rank, conventional_rank, action_number
