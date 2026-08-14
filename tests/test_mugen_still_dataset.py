from __future__ import annotations

import hashlib
import json

import pytest

from spritelab.mugen_still_dataset import (
    action_phrase,
    build_mugen_still_training_plan,
    compact_appearance_prompt,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_still_plan_joins_caption_taxonomy_and_preserves_hierarchy(tmp_path) -> None:
    materialization = {
        "sequence_count": 1,
        "sequences": [
            {
                "entity_class": "humanoid",
                "identity_id": "identity-a",
                "output": {
                    "array_content_sha256": _digest("array"),
                    "file_sha256": _digest("file"),
                    "relative_path": "clips/a.npy",
                    "shape": [8, 128, 128, 4],
                },
                "sequence_id": "sequence-a",
                "split": "train",
                "view": "side",
            }
        ],
    }
    taxonomy = {
        "sequence_count": 1,
        "records": [
            {
                "attack_form": "weapon",
                "attack_strength": "light",
                "attack_tier": "normal",
                "direction": None,
                "identity_id": "identity-a",
                "sequence_id": "sequence-a",
                "split": "train",
                "stance": None,
                "verb": "normal_attack",
            }
        ],
    }
    structured = {
        "accessories": [],
        "armor": "silver shoulder armor",
        "body_build": "athletic",
        "distinctive_visible_features": ["red scarf"],
        "dominant_colors": ["blue", "silver"],
        "equipment": ["short sword"],
        "face": "lower face covered",
        "facing": "right",
        "footwear": "black boots",
        "hair": "white hair",
        "lower_body_clothing": "blue trousers",
        "pose": "standing",
        "secondary_colors": ["red"],
        "skin_or_surface": "light skin",
        "subject_type": "humanoid",
        "uncertain_visible_features": [],
        "upper_body_clothing": "blue tunic",
    }
    captions = {
        "caption_count": 1,
        "records": [
            {
                "caption_input": {"file_sha256": _digest("png")},
                "entity_class": "humanoid",
                "identity_id": "identity-a",
                "reference_array_sha256": _digest("reference"),
                "request_body_sha256": _digest("request"),
                "split": "train",
                "structured_caption": structured,
            }
        ],
    }
    paths = []
    for name, value in (
        ("materialization.json", materialization),
        ("taxonomy.json", taxonomy),
        ("captions.json", captions),
    ):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    plan = build_mugen_still_training_plan(*paths)
    record = plan["records"][0]
    assert plan["counts"]["sequences"] == 1
    assert plan["sampler_contract"]["hierarchy"] == ["identity", "verb", "sequence", "frame"]
    assert record["conditioning"]["action_phrase"] == "performing a normal attack"
    assert "performing a normal attack" in record["prompt"]
    assert "standing" not in record["prompt"]
    assert "right" not in record["prompt"]
    assert record["target"]["frame_count"] == 8

    eligibility = {
        "artifact_kind": "mugen_subject_bearing_still_frame_eligibility",
        "counts": {"sequences": 1},
        "records": [
            {
                "eligible_frame_indices": [1, 3],
                "identity_id": "identity-a",
                "sequence_id": "sequence-a",
                "split": "train",
            }
        ],
        "source": {
            "caption_manifest_file_sha256": hashlib.sha256(paths[2].read_bytes()).hexdigest(),
            "materialization_file_sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        },
    }
    eligibility_path = tmp_path / "eligibility.json"
    eligibility_path.write_text(json.dumps(eligibility), encoding="utf-8")
    clean_plan = build_mugen_still_training_plan(*paths, eligibility_path)
    clean_target = clean_plan["records"][0]["target"]
    assert clean_plan["schema_version"] == 2
    assert clean_plan["counts"]["eligible_frames"] == 2
    assert clean_target["eligible_frame_indices"] == [1, 3]
    assert clean_target["frame_sampling"] == "uniform_subject_bearing_logical_frame_index"


def test_action_phrase_rejects_unknown() -> None:
    assert action_phrase("block") == "blocking in a defensive stance"
    with pytest.raises(ValueError, match="unsupported"):
        action_phrase("dance")


def test_compact_prompt_keeps_whole_priority_facts_without_duplicates() -> None:
    structured = {
        "subject_type": "humanoid",
        "body_build": "slender humanoid",
        "skin_or_surface": "armored black and red surface",
        "hair": "",
        "face": "red helmet",
        "upper_body_clothing": "black and red armor with white trim",
        "lower_body_clothing": "black pants with red lining",
        "footwear": "black boots",
        "armor": "black and red armor with white trim",
        "equipment": ["long silver sword"],
        "accessories": ["gold buckle"],
        "distinctive_visible_features": ["red helmet", "white eye design"],
        "dominant_colors": ["black", "red", "white"],
        "secondary_colors": ["gold"],
    }

    prompt = compact_appearance_prompt(structured, entity_class="humanoid", maximum_words=32)

    assert len(prompt.split()) <= 32
    assert prompt.count("black and red armor with white trim") == 1
    assert "long silver sword" in prompt
    assert not prompt.endswith(";")
