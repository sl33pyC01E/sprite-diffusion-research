"""Bounded canonical-core selection over fully decoded M.U.G.E.N sprites."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from spritelab.adapters.mugen import (
    MugenActionExclusion,
    MugenActionMaterialization,
    MugenAirAction,
    MugenSffV1Sprite,
    MugenSffV2Sprite,
    materialize_actions,
)
from spritelab.mugen_schema import (
    MugenCoreSchemaCoverage,
    canonical_available_slot_action_numbers,
    measure_core_schema_coverage,
    ordered_attack_action_numbers,
    schema_verb,
)


@dataclass(frozen=True, slots=True)
class MugenSelectedCoreAction:
    slot: str
    materialized: MugenActionMaterialization
    source_action: MugenAirAction
    source_action_index: int


@dataclass(frozen=True, slots=True)
class MugenCoreAttemptExclusion:
    exclusion: MugenActionExclusion
    source_action_index: int


@dataclass(frozen=True, slots=True)
class MugenCoreMaterializationPlan:
    source_coverage: MugenCoreSchemaCoverage
    resolved_coverage: MugenCoreSchemaCoverage
    selected: tuple[MugenSelectedCoreAction, ...]
    exclusions: tuple[MugenCoreAttemptExclusion, ...]


def select_mugen_core_materializations(
    actions: tuple[MugenAirAction, ...],
    sprites: tuple[MugenSffV1Sprite | MugenSffV2Sprite, ...],
) -> MugenCoreMaterializationPlan:
    """Render only canonical core candidates, stopping after two distinct attacks."""

    candidates = tuple(
        (source_index, action)
        for source_index, action in enumerate(actions)
        if schema_verb(action.action_number) in {"idle", "walk", "jump", "block", "normal_attack"}
    )
    source_coverage = measure_core_schema_coverage(tuple(action for _, action in candidates))
    requested = canonical_available_slot_action_numbers(source_coverage)
    candidates_by_number: dict[int, list[tuple[int, MugenAirAction]]] = {}
    for source_index, action in candidates:
        candidates_by_number.setdefault(action.action_number, []).append((source_index, action))
    exclusions: list[MugenCoreAttemptExclusion] = []
    admitted_by_number: dict[int, tuple[MugenActionMaterialization, MugenAirAction, int]] = {}

    def resolve(number: int) -> tuple[MugenActionMaterialization, MugenAirAction, int] | None:
        if number in admitted_by_number:
            return admitted_by_number[number]
        for source_index, action in candidates_by_number.get(number, []):
            plan = materialize_actions((action,), sprites)
            exclusions.extend(MugenCoreAttemptExclusion(row, source_index) for row in plan.excluded)
            if plan.admitted:
                result = (plan.admitted[0], action, source_index)
                admitted_by_number[number] = result
                return result
        return None

    selected: dict[str, tuple[MugenActionMaterialization, MugenAirAction, int]] = {}
    for slot in ("idle", "walk", "jump", "block"):
        number = requested.get(slot)
        if number is not None and (result := resolve(number)) is not None:
            selected[slot] = result
    seen_attack_hashes: set[str] = set()
    for number in ordered_attack_action_numbers(source_coverage.attack_action_numbers):
        result = resolve(number)
        if result is None:
            continue
        digest = materialized_pixel_sha256(result[0])
        if digest in seen_attack_hashes:
            continue
        seen_attack_hashes.add(digest)
        slot = "attack_a" if "attack_a" not in selected else "attack_b"
        selected[slot] = result
        if slot == "attack_b":
            break
    ordered = tuple(
        MugenSelectedCoreAction(slot, *selected[slot])
        for slot in ("idle", "walk", "jump", "block", "attack_a", "attack_b")
        if slot in selected
    )
    resolved_coverage = measure_core_schema_coverage(tuple(row.materialized for row in ordered))
    return MugenCoreMaterializationPlan(
        source_coverage=source_coverage,
        resolved_coverage=resolved_coverage,
        selected=ordered,
        exclusions=tuple(exclusions),
    )


def materialized_pixel_sha256(materialized: MugenActionMaterialization) -> str:
    digest = hashlib.sha256()
    digest.update(f"{materialized.canvas_height}x{materialized.canvas_width}\0".encode())
    for frame in materialized.frames:
        digest.update(frame.rgba)
    return digest.hexdigest()
