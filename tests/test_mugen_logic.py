from spritelab.adapters.mugen_logic import parse_mugen_state_logic, structure_mugen_action


def test_static_cns_parser_binds_literal_hitdef_semantics() -> None:
    states = parse_mugen_state_logic(
        b"""
[Statedef 200]
type = S
movetype = A
anim = 201

[State 200, Hit]
type = HitDef
trigger1 = Time = 0
attr = S, NA
animtype = Light
damage = 30, 5

[Statedef 1000]
type = A
movetype = A
anim = 1010
[State 1000, Beam]
type = HitDef
attr = A, SP
animtype = Hard
damage = 80
"""
    )

    assert len(states) == 2
    assert states[0].animation_number == 201
    assert states[0].hitdefs[0].attack_attribute == "NA"
    assert states[0].hitdefs[0].animation_type == "light"
    assert states[0].hitdefs[0].damage == 30
    assert states[1].hitdefs[0].attack_attribute == "SP"

    normal = structure_mugen_action(
        201,
        "recommended_attack_range",
        comments=("Standing Light Punch",),
        hitdefs=states[0].hitdefs,
    )
    special = structure_mugen_action(
        1010,
        "special_attack",
        comments=("Energy beam",),
        hitdefs=states[1].hitdefs,
    )
    assert (normal.verb, normal.attack_tier, normal.attack_strength, normal.attack_form) == (
        "normal_attack",
        "normal",
        "light",
        "punch",
    )
    assert normal.stance == "standing"
    assert (special.verb, special.attack_tier, special.attack_strength) == (
        "special_attack",
        "special",
        "heavy",
    )
    assert special.attack_form == "projectile"


def test_reserved_action_structure_preserves_phase_stance_and_direction() -> None:
    assert structure_mugen_action(120, "guard_start_standing").verb == "block"
    block = structure_mugen_action(131, "guard_crouching")
    walk = structure_mugen_action(21, "walking_backwards")
    get_up = structure_mugen_action(5120, "get_up")

    assert (block.stance, block.phase) == ("crouching", "sustain")
    assert (walk.verb, walk.direction, walk.phase) == ("walk", "backward", "sustain")
    assert (get_up.verb, get_up.stance) == ("get_up", "lying_to_standing")


def test_nonliteral_cns_expressions_remain_unresolved() -> None:
    state = parse_mugen_state_logic(
        b"""
[Statedef 300]
type = var(0)
anim = ifelse(var(1), 301, 302)
[State 300, hit]
type = HitDef
attr = S, SA
damage = 10 + var(2)
"""
    )[0]

    assert state.animation_number is None
    assert state.stance_code is None
    assert state.hitdefs[0].damage is None
    assert state.hitdefs[0].attack_attribute == "SA"


def test_conflicting_evidence_stays_ambiguous_even_after_later_matching_claim() -> None:
    state = parse_mugen_state_logic(
        b"""
[Statedef 200]
type = S
anim = 200
[State 200, first]
type = HitDef
attr = S, NA
animtype = Light
[State 200, second]
type = HitDef
attr = S, SA
animtype = Hard
[State 200, third]
type = HitDef
attr = S, NA
animtype = Light
"""
    )[0]

    action = structure_mugen_action(
        200,
        "recommended_attack_range",
        comments=("light heavy punch kick",),
        hitdefs=state.hitdefs,
    )

    assert action.attack_tier is None
    assert action.attack_strength is None
    assert action.attack_form is None
    assert action.stance == "standing"
