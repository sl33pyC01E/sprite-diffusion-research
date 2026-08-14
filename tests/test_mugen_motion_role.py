from __future__ import annotations

import json

import pytest

from spritelab.mugen_motion_role import (
    MotionRoleSampleConfig,
    conservative_same_subject_motion,
    motion_role_vlm_request,
    parse_motion_role_vlm_response,
    stratified_motion_role_sample,
)


def _decision(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "primary_subject_presence": "all_frames",
        "subject_identity_relation": "same_primary_subject",
        "secondary_content": ["projectile"],
        "action_match": "clear",
        "training_role": "primary_subject_with_effects",
    }
    value.update(overrides)
    return value


def test_parser_admits_same_subject_with_effects() -> None:
    parsed = parse_motion_role_vlm_response(json.dumps(_decision()))

    assert parsed["conservative_same_subject_motion"] is True


@pytest.mark.parametrize(
    "override",
    [
        {"primary_subject_presence": "some_frames"},
        {"subject_identity_relation": "transformation_or_costume_change"},
        {"secondary_content": ["assist_subject"]},
        {"action_match": "mismatch"},
        {"training_role": "effect_only"},
    ],
)
def test_conservative_gate_rejects_role_leakage(override: dict[str, object]) -> None:
    assert conservative_same_subject_motion(_decision(**override)) is False


def test_parser_rejects_none_mixed_with_real_secondary_content() -> None:
    with pytest.raises(ValueError, match="secondary_content"):
        parse_motion_role_vlm_response(
            json.dumps(_decision(secondary_content=["none", "projectile"]))
        )


def test_request_binds_expected_verb_and_png() -> None:
    request = motion_role_vlm_request(
        model="qwen-test", sheet_png=b"\x89PNG\r\n\x1a\nbytes", expected_verb="block"
    )

    assert request["model"] == "qwen-test"
    assert "block" in request["messages"][1]["content"][0]["text"]
    assert request["response_format"]["json_schema"]["strict"] is True


def test_stratified_sample_round_robins_splits_deterministically() -> None:
    records = []
    pixels = []
    for index, split in enumerate(("train", "train", "validation", "test", "test")):
        sequence_id = f"sequence_{index}"
        records.append(
            {
                "conditioning": {"verb": "walk"},
                "identity_id": f"identity_{index}",
                "reference": {},
                "sequence_id": sequence_id,
                "split": split,
                "target": {},
            }
        )
        pixels.append({"sequence_id": sequence_id, "pixel_gate_status": "all_pass"})
    plan = {"counts": {"sequences": len(records)}, "records": records}
    audit = {"counts": {"sequences": len(pixels)}, "records": pixels}

    first = stratified_motion_role_sample(plan, audit, config=MotionRoleSampleConfig(per_verb=3))
    second = stratified_motion_role_sample(plan, audit, config=MotionRoleSampleConfig(per_verb=3))

    assert first == second
    assert {record["split"] for record in first["records"]} == {
        "train",
        "validation",
        "test",
    }
