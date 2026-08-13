"""Static, non-executing M.U.G.E.N action semantics from CNS and AIR evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from spritelab.adapters.mugen import decode_mugen_text

_SECTION = re.compile(r"^\[\s*(statedef|state)\s+([+-]?\d+)(?:\s*,[^]]*)?\]$", re.IGNORECASE)
_ASSIGNMENT = re.compile(r"^([A-Za-z][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*$")
_INTEGER = re.compile(r"^[+-]?\d+$")
_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
_COMMENT_TERM = {
    "light": re.compile(r"\b(?:light|weak|jab)\b", re.IGNORECASE),
    "medium": re.compile(r"\b(?:medium|middle)\b", re.IGNORECASE),
    "heavy": re.compile(r"\b(?:heavy|hard|strong|fierce)\b", re.IGNORECASE),
}
_FORM_TERM = {
    "punch": re.compile(r"\b(?:punch|jab|uppercut)\b", re.IGNORECASE),
    "kick": re.compile(r"\b(?:kick|knee|stomp)\b", re.IGNORECASE),
    "weapon": re.compile(r"\b(?:slash|sword|blade|weapon)\b", re.IGNORECASE),
    "projectile": re.compile(r"\b(?:projectile|shot|shoot|beam|blast|fireball)\b", re.IGNORECASE),
    "throw": re.compile(r"\b(?:throw|grab)\b", re.IGNORECASE),
}


@dataclass(frozen=True, slots=True)
class MugenHitDefEvidence:
    state_number: int
    animation_number: int | None
    stance_code: str | None
    move_type: str | None
    attack_attribute: str | None
    animation_type: str | None
    damage: float | None
    source_line: int


@dataclass(frozen=True, slots=True)
class MugenStateLogic:
    state_number: int
    animation_number: int | None
    stance_code: str | None
    move_type: str | None
    hitdefs: tuple[MugenHitDefEvidence, ...]
    source_line: int


@dataclass(frozen=True, slots=True)
class MugenActionSemanticClaim:
    field: str
    value: str
    method: str
    evidence: str


@dataclass(frozen=True, slots=True)
class MugenStructuredAction:
    action_number: int
    verb: str
    attack_tier: str | None
    attack_strength: str | None
    attack_form: str | None
    stance: str | None
    direction: str | None
    phase: str | None
    claims: tuple[MugenActionSemanticClaim, ...]


@dataclass(slots=True)
class _StateBuilder:
    number: int
    source_line: int
    animation: int | None = None
    stance: str | None = None
    move_type: str | None = None
    hitdefs: list[MugenHitDefEvidence] | None = None

    def __post_init__(self) -> None:
        self.hitdefs = []


@dataclass(slots=True)
class _ControllerBuilder:
    state: _StateBuilder
    source_line: int
    controller_type: str | None = None
    attack_attribute: str | None = None
    animation_type: str | None = None
    damage: float | None = None


def parse_mugen_state_logic(payload: bytes) -> tuple[MugenStateLogic, ...]:
    """Parse literal Statedef/HitDef facts without evaluating triggers or code."""

    text = decode_mugen_text(payload)
    states: list[_StateBuilder] = []
    current_state: _StateBuilder | None = None
    controller: _ControllerBuilder | None = None

    def finish_controller() -> None:
        nonlocal controller
        if controller is None:
            return
        if controller.controller_type == "hitdef":
            assert controller.state.hitdefs is not None
            controller.state.hitdefs.append(
                MugenHitDefEvidence(
                    state_number=controller.state.number,
                    animation_number=controller.state.animation,
                    stance_code=controller.state.stance,
                    move_type=controller.state.move_type,
                    attack_attribute=controller.attack_attribute,
                    animation_type=controller.animation_type,
                    damage=controller.damage,
                    source_line=controller.source_line,
                )
            )
        controller = None

    for line_number, raw in enumerate(text.splitlines(), start=1):
        content = raw.split(";", 1)[0].strip()
        if not content:
            continue
        section = _SECTION.fullmatch(content)
        if section is not None:
            finish_controller()
            kind, number_text = section.groups()
            number = int(number_text)
            if kind.casefold() == "statedef":
                current_state = _StateBuilder(number, line_number)
                states.append(current_state)
            elif current_state is not None and number == current_state.number:
                controller = _ControllerBuilder(current_state, line_number)
            else:
                controller = None
            continue
        assignment = _ASSIGNMENT.fullmatch(content)
        if assignment is None or current_state is None:
            continue
        key, value = assignment.groups()
        key = key.casefold()
        literal = value.strip()
        if controller is None:
            if key == "anim" and _INTEGER.fullmatch(literal):
                current_state.animation = int(literal)
            elif key == "type" and literal.casefold() in {"s", "c", "a", "l", "u"}:
                current_state.stance = literal.upper()
            elif key == "movetype" and literal.casefold() in {"a", "i", "h", "u"}:
                current_state.move_type = literal.upper()
        elif key == "type":
            controller.controller_type = literal.casefold()
        elif key == "attr":
            parts = tuple(part.strip().upper() for part in literal.split(","))
            controller.attack_attribute = parts[1] if len(parts) >= 2 else None
        elif key == "animtype":
            controller.animation_type = literal.casefold()
        elif key == "damage":
            first = literal.split(",", 1)[0].strip()
            controller.damage = float(first) if _NUMBER.fullmatch(first) else None
    finish_controller()
    return tuple(
        MugenStateLogic(
            state.number,
            state.animation,
            state.stance,
            state.move_type,
            tuple(state.hitdefs or ()),
            state.source_line,
        )
        for state in states
    )


def structure_mugen_action(
    action_number: int,
    source_meaning: str | None,
    *,
    comments: tuple[str, ...] = (),
    hitdefs: tuple[MugenHitDefEvidence, ...] = (),
) -> MugenStructuredAction:
    """Merge official numeric semantics with literal comments and HitDef facts."""

    verb, meaning_tier, meaning_stance, direction, phase = _meaning_semantics(source_meaning)
    claims: list[MugenActionSemanticClaim] = []
    if source_meaning is not None:
        for field, value in (
            ("verb", verb),
            ("attack_tier", meaning_tier),
            ("stance", meaning_stance),
            ("direction", direction),
            ("phase", phase),
        ):
            if value is not None:
                claims.append(
                    MugenActionSemanticClaim(
                        field,
                        value,
                        "elecbyte_action_number_standard_or_range",
                        f"action={action_number};source_meaning={source_meaning}",
                    )
                )
    tier_values = {meaning_tier} if meaning_tier is not None else set()
    strength_values: set[str] = set()
    form_values: set[str] = set()
    stance_values = {meaning_stance} if meaning_stance is not None else set()
    for hitdef in hitdefs:
        hit_tier, hit_form = _hitdef_attribute(hitdef.attack_attribute)
        if hit_tier is not None:
            tier_values.add(hit_tier)
            claims.append(
                MugenActionSemanticClaim(
                    "attack_tier",
                    hit_tier,
                    "literal_hitdef_attr",
                    f"state={hitdef.state_number};line={hitdef.source_line};attr={hitdef.attack_attribute}",
                )
            )
        if hit_form is not None:
            form_values.add(hit_form)
            claims.append(
                MugenActionSemanticClaim(
                    "attack_form",
                    hit_form,
                    "literal_hitdef_attr",
                    f"state={hitdef.state_number};line={hitdef.source_line};attr={hitdef.attack_attribute}",
                )
            )
        hit_strength = _animation_type_strength(hitdef.animation_type)
        if hit_strength is not None:
            strength_values.add(hit_strength)
            claims.append(
                MugenActionSemanticClaim(
                    "attack_strength",
                    hit_strength,
                    "literal_hitdef_animtype",
                    f"state={hitdef.state_number};line={hitdef.source_line};animtype={hitdef.animation_type}",
                )
            )
        hit_stance = _stance(hitdef.stance_code)
        if hit_stance is not None:
            stance_values.add(hit_stance)
            claims.append(
                MugenActionSemanticClaim(
                    "stance",
                    hit_stance,
                    "literal_statedef_type",
                    f"state={hitdef.state_number};line={hitdef.source_line};type={hitdef.stance_code}",
                )
            )
    comment_text = " | ".join(value.strip() for value in comments if value.strip())
    if comment_text:
        for value, pattern in _COMMENT_TERM.items():
            if pattern.search(comment_text):
                strength_values.add(value)
                claims.append(
                    MugenActionSemanticClaim(
                        "attack_strength", value, "literal_air_comment_vocabulary", comment_text
                    )
                )
        for value, pattern in _FORM_TERM.items():
            if pattern.search(comment_text):
                form_values.add(value)
                claims.append(
                    MugenActionSemanticClaim(
                        "attack_form", value, "literal_air_comment_vocabulary", comment_text
                    )
                )
    tier = _unambiguous(tier_values)
    strength = _unambiguous(strength_values)
    form = _unambiguous(form_values)
    stance = _unambiguous(stance_values)
    if tier is not None and verb in {"attack", "unknown"}:
        verb = f"{tier}_attack" if tier != "super" else "super_attack"
    return MugenStructuredAction(
        action_number,
        verb,
        tier,
        strength,
        form,
        stance,
        direction,
        phase,
        tuple(claims),
    )


def _meaning_semantics(
    meaning: str | None,
) -> tuple[str, str | None, str | None, str | None, str | None]:
    exact: dict[str, tuple[str, str | None, str | None, str | None]] = {
        "standing": ("idle", "standing", "neutral", "sustain"),
        "stand_turning": ("turn", "standing", None, "one_shot"),
        "crouch_turning": ("turn", "crouching", None, "one_shot"),
        "stand_to_crouch": ("crouch", "standing_to_crouching", None, "start"),
        "crouching": ("crouch", "crouching", "neutral", "sustain"),
        "crouch_to_stand": ("crouch", "crouching_to_standing", None, "end"),
        "walking_forwards": ("walk", "standing", "forward", "sustain"),
        "walking_backwards": ("walk", "standing", "backward", "sustain"),
        "jump_start": ("jump", "standing", "vertical", "start"),
        "jump_neutral": ("jump", "airborne", "vertical", "sustain"),
        "jump_forwards": ("jump", "airborne", "forward", "sustain"),
        "jump_backwards": ("jump", "airborne", "backward", "sustain"),
        "jump_land": ("land", "standing", "vertical", "end"),
        "run_forwards": ("run", "standing", "forward", "sustain"),
        "hop_backwards": ("backstep", "airborne", "backward", "one_shot"),
        "air_recover": ("recover", "airborne", None, "one_shot"),
        "air_fall": ("hurt", "airborne", None, "sustain"),
        "tripped": ("hurt", "airborne", "backward", "one_shot"),
        "lie_down_hit": ("hurt", "lying", None, "one_shot"),
        "hit_ground_from_fall": ("hurt", "lying", None, "one_shot"),
        "get_up": ("get_up", "lying_to_standing", None, "one_shot"),
        "lie_down": ("death", "lying", None, "sustain"),
        "lie_dead_first_rounds": ("death", "lying", None, "sustain"),
        "lie_dead_final_round": ("death", "lying", None, "sustain"),
        "dizzy": ("dizzy", "standing", None, "sustain"),
        "lose": ("defeat", "standing", None, "one_shot"),
        "time_over": ("defeat", "standing", None, "one_shot"),
        "win": ("victory", "standing", None, "one_shot"),
        "alternate_win": ("victory", "standing", None, "one_shot"),
        "intro": ("intro", "standing", None, "one_shot"),
        "alternate_intro": ("intro", "standing", None, "one_shot"),
    }
    if meaning in exact:
        verb, stance, direction, phase = exact[meaning]
        return verb, None, stance, direction, phase
    if meaning == "recommended_attack_range":
        return "attack", "normal", None, None, "one_shot"
    if meaning == "special_attack":
        return "attack", "special", None, None, "one_shot"
    if meaning == "hyper_attack":
        return "attack", "super", None, None, "one_shot"
    if meaning is not None and meaning.startswith("guard_start_"):
        return "block", None, _guard_stance(meaning), None, "start"
    if meaning is not None and meaning.startswith("guard_end_"):
        return "block", None, _guard_stance(meaning), None, "end"
    if meaning is not None and meaning.startswith("guard_"):
        return "block", None, _guard_stance(meaning), None, "sustain"
    return "unknown", None, None, None, None


def _guard_stance(meaning: str) -> str | None:
    if meaning.endswith("standing"):
        return "standing"
    if meaning.endswith("crouching"):
        return "crouching"
    if meaning.endswith("air"):
        return "airborne"
    return None


def _hitdef_attribute(value: str | None) -> tuple[str | None, str | None]:
    if value is None or len(value) != 2:
        return None, None
    tier = {"N": "normal", "S": "special", "H": "super"}.get(value[0])
    # ``A`` means a generic attack and is not specific enough to compete with
    # literal punch/kick/weapon comment evidence. Projectile and throw are
    # semantically distinct mechanisms and remain useful canonical forms.
    form = {"P": "projectile", "T": "throw"}.get(value[1])
    return tier, form


def _animation_type_strength(value: str | None) -> str | None:
    if value is None:
        return None
    return {"light": "light", "medium": "medium", "hard": "heavy"}.get(value.casefold())


def _stance(value: str | None) -> str | None:
    return {"S": "standing", "C": "crouching", "A": "airborne", "L": "lying"}.get(value)


def _unambiguous(values: set[str]) -> str | None:
    return next(iter(values)) if len(values) == 1 else None
