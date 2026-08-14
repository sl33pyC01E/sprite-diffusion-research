"""Strict VLM role adjudication for reference-conditioned MUGEN motion clips."""

from __future__ import annotations

import base64
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from spritelab.spark_caption import canonical_json_bytes

PRESENCE_VALUES = ("all_frames", "some_frames", "no_frames", "ambiguous")
RELATION_VALUES = (
    "same_primary_subject",
    "transformation_or_costume_change",
    "different_or_assist_subject",
    "ambiguous",
)
SECONDARY_CONTENT_VALUES = (
    "projectile",
    "detached_effect",
    "weapon_only",
    "assist_subject",
    "full_screen_effect",
    "camera_zoom",
    "none",
)
ACTION_MATCH_VALUES = ("clear", "plausible", "mismatch", "ambiguous")
TRAINING_ROLE_VALUES = (
    "primary_subject_motion",
    "primary_subject_with_effects",
    "transformation",
    "assist_or_replacement",
    "effect_only",
    "ambiguous",
)

MOTION_ROLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "primary_subject_presence": {"type": "string", "enum": list(PRESENCE_VALUES)},
        "subject_identity_relation": {"type": "string", "enum": list(RELATION_VALUES)},
        "secondary_content": {
            "type": "array",
            "items": {"type": "string", "enum": list(SECONDARY_CONTENT_VALUES)},
            "uniqueItems": True,
        },
        "action_match": {"type": "string", "enum": list(ACTION_MATCH_VALUES)},
        "training_role": {"type": "string", "enum": list(TRAINING_ROLE_VALUES)},
    },
    "required": [
        "primary_subject_presence",
        "subject_identity_relation",
        "secondary_content",
        "action_match",
        "training_role",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class MotionRoleSampleConfig:
    per_verb: int = 4
    include_pixel_statuses: tuple[str, ...] = ("all_pass",)

    def __post_init__(self) -> None:
        if isinstance(self.per_verb, bool) or not isinstance(self.per_verb, int):
            raise ValueError("per_verb must be an integer")
        if self.per_verb <= 0:
            raise ValueError("per_verb must be positive")
        allowed = {"all_pass", "mixed", "all_fail"}
        if (
            not self.include_pixel_statuses
            or len(set(self.include_pixel_statuses)) != len(self.include_pixel_statuses)
            or any(value not in allowed for value in self.include_pixel_statuses)
        ):
            raise ValueError("include_pixel_statuses is invalid")


def motion_role_prompt_sha256() -> str:
    """Hash the exact role prompt and strict schema."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": MOTION_ROLE_SCHEMA,
                "system": _SYSTEM_PROMPT,
                "user_template": _USER_TEMPLATE,
            }
        )
    ).hexdigest()


def motion_role_vlm_request(*, model: str, sheet_png: bytes, expected_verb: str) -> dict[str, Any]:
    """Build an OpenAI-compatible deterministic role-adjudication request."""

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be non-empty text")
    if not isinstance(sheet_png, bytes) or not sheet_png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("sheet_png must be PNG bytes")
    if not isinstance(expected_verb, str) or not expected_verb.strip():
        raise ValueError("expected_verb must be non-empty text")
    data_url = "data:image/png;base64," + base64.b64encode(sheet_png).decode("ascii")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _USER_TEMPLATE.format(expected_verb=expected_verb.strip()),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 384,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "mugen_motion_role",
                "strict": True,
                "schema": MOTION_ROLE_SCHEMA,
            },
        },
    }


def parse_motion_role_vlm_response(content: str) -> dict[str, Any]:
    """Strictly parse and normalize one motion-role JSON response."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("motion-role response must be non-empty text")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("motion-role response is invalid JSON") from error
    if not isinstance(value, dict) or set(value) != set(MOTION_ROLE_SCHEMA["required"]):
        raise ValueError("motion-role response fields differ")
    scalar_enums = {
        "primary_subject_presence": PRESENCE_VALUES,
        "subject_identity_relation": RELATION_VALUES,
        "action_match": ACTION_MATCH_VALUES,
        "training_role": TRAINING_ROLE_VALUES,
    }
    for key, allowed in scalar_enums.items():
        if value[key] not in allowed:
            raise ValueError(f"motion-role field {key} is invalid")
    secondary = value["secondary_content"]
    if (
        not isinstance(secondary, list)
        or not secondary
        or len(secondary) != len(set(secondary))
        or any(item not in SECONDARY_CONTENT_VALUES for item in secondary)
        or ("none" in secondary and len(secondary) != 1)
    ):
        raise ValueError("motion-role secondary_content is invalid")
    value["secondary_content"] = sorted(secondary, key=lambda item: item.encode())
    value["conservative_same_subject_motion"] = conservative_same_subject_motion(value)
    return value


def conservative_same_subject_motion(decision: dict[str, Any]) -> bool:
    """Admit only clips whose primary reference subject remains visibly consistent."""

    return bool(
        decision.get("primary_subject_presence") == "all_frames"
        and decision.get("subject_identity_relation") == "same_primary_subject"
        and decision.get("action_match") in {"clear", "plausible"}
        and decision.get("training_role")
        in {"primary_subject_motion", "primary_subject_with_effects"}
        and "assist_subject" not in decision.get("secondary_content", [])
    )


def stratified_motion_role_sample(
    motion_plan: dict[str, Any],
    pixel_audit: dict[str, Any],
    *,
    config: MotionRoleSampleConfig | None = None,
) -> dict[str, Any]:
    """Select stable per-verb examples across splits for precision auditing."""

    selection = config or MotionRoleSampleConfig()
    records = _counted_records(motion_plan, "records", "sequences", "motion plan")
    pixel_records = _counted_records(pixel_audit, "records", "sequences", "pixel audit")
    pixel_by_sequence = _unique(pixel_records, "sequence_id", "pixel audit")
    candidates: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        sequence_id = _text(record, "sequence_id")
        pixel = pixel_by_sequence.get(sequence_id)
        if pixel is None:
            raise ValueError(f"pixel audit lacks sequence {sequence_id}")
        if pixel.get("pixel_gate_status") not in selection.include_pixel_statuses:
            continue
        conditioning = record.get("conditioning")
        if not isinstance(conditioning, dict):
            raise ValueError(f"conditioning is missing for {sequence_id}")
        verb = _text(conditioning, "verb")
        candidates[verb].append(record)
    output = []
    for verb in sorted(candidates, key=lambda item: item.encode()):
        ordered = sorted(
            candidates[verb],
            key=lambda record: (
                hashlib.sha256(
                    f"mugen_motion_role_sample_v1\0{verb}\0{record['sequence_id']}".encode()
                ).digest(),
                str(record["sequence_id"]).encode(),
            ),
        )
        split_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in ordered:
            split_groups[_text(record, "split")].append(record)
        chosen = []
        split_order = ("train", "validation", "test")
        while len(chosen) < selection.per_verb:
            changed = False
            for split in split_order:
                if split_groups[split] and len(chosen) < selection.per_verb:
                    chosen.append(split_groups[split].pop(0))
                    changed = True
            if not changed:
                break
        for record in chosen:
            output.append(
                {
                    "expected_verb": verb,
                    "identity_id": record["identity_id"],
                    "reference": record["reference"],
                    "sequence_id": record["sequence_id"],
                    "split": record["split"],
                    "target": record["target"],
                }
            )
    return {
        "artifact_kind": "mugen_motion_role_vlm_sample",
        "config": {
            "include_pixel_statuses": list(selection.include_pixel_statuses),
            "per_verb": selection.per_verb,
        },
        "counts": {
            "records": len(output),
            "verbs": len(candidates),
        },
        "prompt_sha256": motion_role_prompt_sha256(),
        "records": output,
        "schema_version": 1,
    }


def _counted_records(
    artifact: dict[str, Any], key: str, count_key: str, label: str
) -> list[dict[str, Any]]:
    records = artifact.get(key)
    counts = artifact.get("counts")
    if (
        not isinstance(records, list)
        or not isinstance(counts, dict)
        or counts.get(count_key) != len(records)
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ValueError(f"{label} record count differs")
    return records


def _unique(records: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        value = _text(record, key)
        if value in output:
            raise ValueError(f"{label} has duplicate {key}: {value}")
        output[value] = record
    return output


def _text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


_SYSTEM_PROMPT = (
    "You are a strict visual curator for small 2D fighting-game sprite sequences. "
    "Compare panels literally. Never identify a proper name, franchise, or artist. "
    "Return only the requested JSON object."
)

_USER_TEMPLATE = """REF is the canonical primary fighter. Panels 0 through 7 are one
animation labeled {expected_verb}. Decide whether that same fighter remains visibly present,
whether another fighter/assist replaces or joins it, and whether the visible motion plausibly
matches the label. Pose, facing, deformation, weapons, projectiles, and effects may change.
A transformation or costume/body identity change is not the same primary subject. Use
primary_subject_with_effects only when the canonical fighter remains present and recognizable.
Use effect_only when the fighter disappears and the panels show only effects/projectiles.
Use ambiguous rather than guessing."""
