from __future__ import annotations

import json

import pytest

from spritelab.spark_caption import (
    ALL_FIELDS,
    caption_prompt_sha256,
    openai_vision_request,
    parse_structured_caption,
    structured_training_prompt,
)


def _caption() -> dict[str, object]:
    return {
        "subject_type": " Humanoid ",
        "body_build": "Tall and lean",
        "pose": "standing with one fist raised",
        "facing": "three-quarter right",
        "skin_or_surface": "light skin",
        "hair": "not visible",
        "face": "lower face covered",
        "upper_body_clothing": "blue tunic",
        "lower_body_clothing": "blue trousers",
        "footwear": "black boots",
        "armor": "silver chest armor",
        "accessories": ["red scarf", "Red scarf"],
        "equipment": ["short sword at left hip"],
        "dominant_colors": ["blue", "silver"],
        "secondary_colors": ["black", "white"],
        "distinctive_visible_features": ["red shoulder guard"],
        "uncertain_visible_features": ["possible narrow cape"],
    }


def test_structured_caption_normalizes_and_builds_visual_only_prompt() -> None:
    parsed = parse_structured_caption(json.dumps(_caption()))
    assert set(parsed) == set(ALL_FIELDS)
    assert parsed["subject_type"] == "humanoid"
    assert parsed["hair"] == ""
    assert parsed["accessories"] == ["red scarf"]
    prompt = structured_training_prompt(parsed, entity_class="humanoid")
    assert prompt.startswith("2D sprite on a transparent background")
    assert "humanoid, tall and lean" in prompt
    assert "red shoulder guard" in prompt
    assert "possible narrow cape" not in prompt
    appearance = structured_training_prompt(
        parsed, entity_class="humanoid", include_pose_and_facing=False
    )
    assert "standing with one fist raised" not in appearance
    assert "three-quarter right" not in appearance


def test_caption_parser_rejects_schema_drift_and_non_json() -> None:
    value = _caption()
    value.pop("face")
    with pytest.raises(ValueError, match="fields mismatch"):
        parse_structured_caption(json.dumps(value))
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_structured_caption("the subject wears armor")


def test_openai_request_is_deterministic_and_structured() -> None:
    first = openai_vision_request(model="qwen", png_data_url="data:image/png;base64,AA==")
    second = openai_vision_request(model="qwen", png_data_url="data:image/png;base64,AA==")
    assert first == second
    assert first["temperature"] == 0
    assert first["max_tokens"] == 2048
    assert first["chat_template_kwargs"] == {"enable_thinking": True}
    assert first["response_format"]["type"] == "json_schema"
    assert len(caption_prompt_sha256()) == 64
