"""Literal structured caption contracts for remote sprite VLMs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

SYSTEM_PROMPT = """You are a literal visual annotator for small 2D sprite artwork.
Describe only visible evidence in the supplied image. Never identify or guess a character,
person, franchise, anime, game, artist, or proper name, even if you recognize the design.
Do not mention the background, image format, resolution, pixelation, or artistic quality.
Do not infer hidden body parts, personality, story, powers, gender, ethnicity, or age.
Use short concrete visual phrases. Put ambiguous details only in uncertain_visible_features.
Return exactly one JSON object matching the requested schema and no prose."""

USER_PROMPT = """Create a literal appearance record for the one visible sprite subject.
Use lowercase phrases. Colors should be ordinary color words. Keep each list concise.
subject_type must be one biological/object category from the schema, never "sprite",
"character", an art style, or a proper name. Separate upper clothing, lower clothing,
footwear, armor, accessories, and held equipment. If a field is not visibly supported,
use an empty string or empty list."""

STRING_FIELDS = (
    "subject_type",
    "body_build",
    "pose",
    "facing",
    "skin_or_surface",
    "hair",
    "face",
    "upper_body_clothing",
    "lower_body_clothing",
    "footwear",
    "armor",
)
LIST_FIELDS = (
    "accessories",
    "equipment",
    "dominant_colors",
    "secondary_colors",
    "distinctive_visible_features",
    "uncertain_visible_features",
)
ALL_FIELDS = STRING_FIELDS + LIST_FIELDS
SUBJECT_TYPES = (
    "humanoid",
    "animal",
    "robot",
    "creature",
    "monster",
    "object",
    "projectile",
    "effect",
    "unknown",
)

CAPTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        **{name: {"type": "string"} for name in STRING_FIELDS if name != "subject_type"},
        "subject_type": {
            "type": "string",
            "enum": list(SUBJECT_TYPES),
        },
        **{
            name: {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            }
            for name in LIST_FIELDS
        },
    },
    "required": list(ALL_FIELDS),
    "additionalProperties": False,
}


def caption_prompt_sha256() -> str:
    """Hash the exact prompt and schema contract used for caption generation."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "json_schema": CAPTION_JSON_SCHEMA,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": USER_PROMPT,
            }
        )
    ).hexdigest()


def parse_structured_caption(content: str) -> dict[str, str | list[str]]:
    """Parse and strictly validate one VLM caption JSON object."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("caption response content must be non-empty text")
    value = _decode_json_object(content)
    if set(value) != set(ALL_FIELDS):
        raise ValueError(
            f"caption fields mismatch: expected {sorted(ALL_FIELDS)!r}, got {sorted(value)!r}"
        )
    normalized: dict[str, str | list[str]] = {}
    for name in STRING_FIELDS:
        item = value[name]
        if not isinstance(item, str):
            raise ValueError(f"caption field {name} must be a string")
        phrase = _normalize_phrase(item)
        normalized[name] = "" if name != "subject_type" and _is_empty_phrase(phrase) else phrase
    if normalized["subject_type"] not in SUBJECT_TYPES:
        raise ValueError("caption field subject_type is outside the supported taxonomy")
    for name in LIST_FIELDS:
        item = value[name]
        if not isinstance(item, list) or len(item) > 8:
            raise ValueError(f"caption field {name} must be a list with at most 8 items")
        phrases: list[str] = []
        for raw in item:
            if not isinstance(raw, str):
                raise ValueError(f"caption field {name} must contain only strings")
            phrase = _normalize_phrase(raw)
            if phrase and not _is_empty_phrase(phrase) and phrase not in phrases:
                phrases.append(phrase)
        normalized[name] = phrases
    return normalized


def structured_training_prompt(
    caption: Mapping[str, str | Sequence[str]],
    *,
    entity_class: str,
    include_pose_and_facing: bool = True,
) -> str:
    """Convert validated visible facts into the still generator's canonical text prompt."""

    validated = parse_structured_caption(json.dumps(dict(caption)))
    if not isinstance(entity_class, str) or not entity_class.strip():
        raise ValueError("entity_class must be non-empty text")
    if not isinstance(include_pose_and_facing, bool):
        raise TypeError("include_pose_and_facing must be a boolean")
    pieces = ["2D sprite on a transparent background"]
    subject = _join_nonempty(
        [
            validated["subject_type"],
            validated["body_build"],
            entity_class.strip().casefold(),
        ]
    )
    if subject:
        pieces.append(subject)
    fields = (
        "skin_or_surface",
        "hair",
        "face",
        "upper_body_clothing",
        "lower_body_clothing",
        "footwear",
        "armor",
        "accessories",
        "equipment",
    )
    if include_pose_and_facing:
        fields = ("pose", "facing", *fields)
    for field in fields:
        value = validated[field]
        text = ", ".join(value) if isinstance(value, list) else value
        if text:
            pieces.append(text)
    colors = list(validated["dominant_colors"]) + list(validated["secondary_colors"])
    if colors:
        pieces.append(f"visible colors: {', '.join(dict.fromkeys(colors))}")
    features = validated["distinctive_visible_features"]
    if features:
        pieces.append(", ".join(features))
    pieces.append("full subject, centered, crisp hard edges")
    return ". ".join(pieces) + "."


def openai_vision_request(
    *,
    model: str,
    png_data_url: str,
    max_tokens: int = 2048,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    """Build the exact OpenAI-compatible deterministic VLM request."""

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be non-empty text")
    if not isinstance(png_data_url, str) or not png_data_url.startswith("data:image/png;base64,"):
        raise ValueError("png_data_url must contain a base64 PNG data URL")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    if not isinstance(enable_thinking, bool):
        raise TypeError("enable_thinking must be a boolean")
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image_url", "image_url": {"url": png_data_url}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "literal_sprite_caption",
                "strict": True,
                "schema": CAPTION_JSON_SCHEMA,
            },
        },
    }


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _decode_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("caption response is not a JSON object") from error
    if not isinstance(value, dict):
        raise ValueError("caption response must be a JSON object")
    return value


def _normalize_phrase(value: str) -> str:
    return " ".join(value.strip().split()).strip(" .").casefold()


def _is_empty_phrase(value: str) -> bool:
    return value in {
        "n/a",
        "none",
        "none visible",
        "not applicable",
        "not discernible",
        "not visible",
        "unclear",
        "unknown",
    }


def _join_nonempty(values: Sequence[object]) -> str:
    retained = [str(value) for value in values if isinstance(value, str) and value]
    return ", ".join(dict.fromkeys(retained))
