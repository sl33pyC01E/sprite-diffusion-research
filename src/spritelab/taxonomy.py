from __future__ import annotations

import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NormalizedLabel:
    value: str
    confidence: float
    method: str


@dataclass(frozen=True)
class MotionCondition:
    source_action: str | None
    normalized_action: str
    action_family: str
    direction: str
    view: str
    loopable_default: bool | None
    confidence: float
    method: str


@dataclass(frozen=True)
class Taxonomy:
    version: str
    entity_classes: frozenset[str]
    entity_aliases: dict[str, str]
    action_families: dict[str, tuple[str, ...]]
    action_aliases: dict[str, str]
    loop_defaults: dict[str, bool]
    views: frozenset[str]
    directions: frozenset[str]

    @property
    def action_to_family(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for family, actions in self.action_families.items():
            for action in actions:
                if action in result:
                    raise ValueError(f"Action {action!r} appears in multiple families")
                result[action] = family
        return result

    def normalize_entity_class(self, raw: str | None) -> NormalizedLabel:
        return _normalize_label(
            raw,
            canonical=self.entity_classes,
            aliases=self.entity_aliases,
            fallback="unknown",
        )

    def normalize_action(self, raw: str | None) -> NormalizedLabel:
        return _normalize_label(
            raw,
            canonical=frozenset(self.action_to_family),
            aliases=self.action_aliases,
            fallback="unknown",
        )

    def normalize_direction(self, raw: str | None) -> NormalizedLabel:
        return _normalize_label(
            raw,
            canonical=self.directions,
            aliases={
                "n": "up",
                "north": "up",
                "s": "down",
                "south": "down",
                "e": "right",
                "east": "right",
                "w": "left",
                "west": "left",
                "ne": "up_right",
                "nw": "up_left",
                "se": "down_right",
                "sw": "down_left",
            },
            fallback="unknown",
        )

    def normalize_view(self, raw: str | None) -> NormalizedLabel:
        return _normalize_label(
            raw,
            canonical=self.views,
            aliases={
                "profile": "side",
                "3_4": "three_quarter",
                "threequarters": "three_quarter",
                "topdown": "top_down",
                "iso": "isometric",
            },
            fallback="unknown",
        )

    def motion_condition(
        self,
        *,
        action: str | None,
        direction: str | None = None,
        view: str | None = None,
    ) -> MotionCondition:
        action_label = self.normalize_action(action)
        direction_label = self.normalize_direction(direction)
        view_label = self.normalize_view(view)
        return MotionCondition(
            source_action=action,
            normalized_action=action_label.value,
            action_family=self.action_to_family.get(action_label.value, "other"),
            direction=direction_label.value,
            view=view_label.value,
            loopable_default=self.loop_defaults.get(action_label.value),
            confidence=min(
                action_label.confidence,
                direction_label.confidence if direction else 1.0,
                view_label.confidence if view else 1.0,
            ),
            method="/".join((action_label.method, direction_label.method, view_label.method)),
        )


def load_taxonomy(path: Path) -> Taxonomy:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    taxonomy = Taxonomy(
        version=str(raw["version"]),
        entity_classes=frozenset(map(str, raw["entity_classes"]["values"])),
        entity_aliases={str(key): str(value) for key, value in raw["entity_aliases"].items()},
        action_families={
            str(family): tuple(map(str, actions))
            for family, actions in raw["action_families"].items()
        },
        action_aliases={str(key): str(value) for key, value in raw["action_aliases"].items()},
        loop_defaults={str(key): bool(value) for key, value in raw["loop_defaults"].items()},
        views=frozenset(map(str, raw["views"]["values"])),
        directions=frozenset(map(str, raw["directions"]["values"])),
    )
    _validate_taxonomy(taxonomy)
    return taxonomy


def _normalize_label(
    raw: str | None,
    *,
    canonical: frozenset[str],
    aliases: dict[str, str],
    fallback: str,
) -> NormalizedLabel:
    if not raw:
        return NormalizedLabel(fallback, 0.0, "missing")
    slug = _slug(raw)
    if slug in canonical:
        return NormalizedLabel(slug, 1.0, "exact")
    if slug in aliases:
        return NormalizedLabel(aliases[slug], 0.98, "alias")

    candidates = sorted(canonical | aliases.keys(), key=lambda value: (-len(value), value))
    padded = f"_{slug}_"
    for candidate in candidates:
        if f"_{candidate}_" in padded:
            value = aliases.get(candidate, candidate)
            return NormalizedLabel(value, 0.85, "embedded_token")
    return NormalizedLabel(fallback, 0.0, "unmapped")


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")


def _validate_taxonomy(taxonomy: Taxonomy) -> None:
    action_to_family = taxonomy.action_to_family
    for alias, target in taxonomy.action_aliases.items():
        if target not in action_to_family:
            raise ValueError(f"Action alias {alias!r} targets unknown action {target!r}")
    for alias, target in taxonomy.entity_aliases.items():
        if target not in taxonomy.entity_classes:
            raise ValueError(f"Entity alias {alias!r} targets unknown class {target!r}")
    for action in taxonomy.loop_defaults:
        if action not in action_to_family:
            raise ValueError(f"Loop default targets unknown action {action!r}")
    for required in ("unknown",):
        if required not in action_to_family:
            raise ValueError(f"Required action is absent: {required}")
        if required not in taxonomy.entity_classes:
            raise ValueError(f"Required entity class is absent: {required}")
