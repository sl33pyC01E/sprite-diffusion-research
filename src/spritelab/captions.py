"""Provenance-aware text and structured conditioning for sprite sequences."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from spritelab.dataset import SequenceSample


@dataclass(frozen=True, slots=True)
class SpriteCaption:
    sequence_id: str
    text: str
    identity_label: str
    description: str
    description_basis: str
    source_prompt: str | None
    source_prompt_scope: str | None
    entity_class: str
    action: str
    view: str
    direction: str
    loop_mode: str


@dataclass(frozen=True, slots=True)
class SpriteGenerationRequest:
    description: str
    entity_class: str
    action: str
    view: str = "unknown"
    direction: str = "unknown"
    loop_mode: str = "unknown"

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("description cannot be empty")
        for name in ("entity_class", "action", "view", "direction", "loop_mode"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")

    @property
    def text(self) -> str:
        return _render_text(
            description=self.description.strip(),
            entity_class=self.entity_class,
            action=self.action,
            view=self.view,
            direction=self.direction,
            loop_mode=self.loop_mode,
        )


def build_sprite_caption(sample: SequenceSample) -> SpriteCaption:
    """Compile one caption without promoting collection prompts to identity facts."""

    metadata = sample.metadata if isinstance(sample.metadata, Mapping) else {}
    sequence_metadata = metadata.get("sequence_metadata")
    if not isinstance(sequence_metadata, Mapping):
        sequence_metadata = {}
    source_prompt = _optional_text(sequence_metadata.get("prompt"))
    source_prompt_scope = _optional_text(sequence_metadata.get("prompt_scope"))
    identity_key = _identity_key(metadata, fallback=sample.identity_id)
    identity_label = _humanize_identity(identity_key)
    if source_prompt is not None and source_prompt_scope == "identity":
        description = source_prompt
        description_basis = "source_identity_prompt"
    else:
        description = identity_label
        description_basis = "external_identity_key"
    return SpriteCaption(
        sequence_id=sample.sequence_id,
        text=_render_text(
            description=description,
            entity_class=sample.entity_class,
            action=sample.action,
            view=sample.view,
            direction=sample.direction,
            loop_mode=sample.loop_mode,
        ),
        identity_label=identity_label,
        description=description,
        description_basis=description_basis,
        source_prompt=source_prompt,
        source_prompt_scope=source_prompt_scope,
        entity_class=sample.entity_class,
        action=sample.action,
        view=sample.view,
        direction=sample.direction,
        loop_mode=sample.loop_mode,
    )


def _identity_key(metadata: Mapping[str, Any], *, fallback: str) -> str:
    sequence_metadata = metadata.get("sequence_metadata")
    subject_fallback: str | None = None
    subjects = metadata.get("subjects")
    if isinstance(subjects, list | tuple):
        ordered = sorted(
            (subject for subject in subjects if isinstance(subject, Mapping)),
            key=lambda subject: (
                subject.get("role") != "primary",
                str(subject.get("entity_id", "")).encode("utf-8"),
            ),
        )
        for subject in ordered:
            key = _optional_text(subject.get("external_identity_key"))
            if key:
                subject_fallback = key
                break
    if isinstance(sequence_metadata, Mapping):
        for key_name in (
            "source_class",
            "identity_label",
            "source_identity",
            "character_name",
            "entity_label",
        ):
            key = _optional_text(sequence_metadata.get(key_name))
            if key:
                return key
    return subject_fallback or fallback


def _humanize_identity(value: str) -> str:
    leaf = re.split(r"[:/\\]", value)[-1]
    leaf = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", leaf)
    leaf = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", leaf)
    leaf = re.sub(r"\.(?:png|gif|webp|apng|jpg|jpeg)$", "", leaf, flags=re.IGNORECASE)
    words = re.sub(r"[^0-9A-Za-z]+", " ", leaf).strip().casefold()
    return words or "unknown subject"


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _render_text(
    *,
    description: str,
    entity_class: str,
    action: str,
    view: str,
    direction: str,
    loop_mode: str,
) -> str:
    clauses = [description.strip(), f"{entity_class} entity"]
    if action not in {"unknown", "other", "custom"}:
        clauses.append(f"{action.replace('_', ' ')} action")
    if view not in {"unknown", "other"}:
        clauses.append(f"{view.replace('_', ' ')} view")
    if direction not in {"unknown", "none"}:
        clauses.append(f"facing {direction.replace('_', ' ')}")
    if loop_mode == "loop":
        clauses.append("seamless loop")
    elif loop_mode == "one_shot":
        clauses.append("one-shot animation")
    elif loop_mode == "ping_pong":
        clauses.append("ping-pong loop")
    clauses.extend(("transparent background", "pixel art animated sprite"))
    return ", ".join(clauses)
