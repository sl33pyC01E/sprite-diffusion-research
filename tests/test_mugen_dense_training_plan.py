from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spritelab.mugen_dense_training_plan import (
    build_mugen_dense_still_training_plan,
    export_mugen_dense_still_training_plan,
)
from spritelab.storage import DiskGuard


def _fixture(tmp_path: Path) -> Path:
    sequences = []
    for index, action in enumerate(("idle", "walk", "jump", "block", "attack_a", "attack_b")):
        sequences.append(
            {
                "action": action,
                "caption": {
                    "description": "a stocky fighter in green clothing",
                    "reference_frame_array_content_sha256": "f" * 64,
                    "reference_frame_index": 2,
                },
                "entity_class": "humanoid",
                "identity_id": "identity-a",
                "output": {
                    "array_content_sha256": f"{index + 1:064x}",
                    "file_sha256": f"{index + 11:064x}",
                    "relative_path": f"{action}.npy",
                    "shape": [8, 128, 128, 4],
                },
                "sequence_id": f"sequence-{action}",
                "split": "train",
            }
        )
    value = {
        "artifact_kind": "mugen_dense_captioned_materialization_bridge",
        "model_eligibility": {"conditional_generation": True},
        "sequence_count": len(sequences),
        "sequences": sequences,
    }
    path = tmp_path / "captioned.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_dense_still_plan_uses_all_idle_frames_for_one_appearance_clip(tmp_path: Path) -> None:
    materialization = _fixture(tmp_path)

    plan = build_mugen_dense_still_training_plan(materialization)

    assert plan["counts"]["canonical_references"] == 1
    assert plan["counts"]["sequences"] == 1
    assert plan["counts"]["source_action_sequences"] == 6
    assert plan["schema_version"] == 6
    assert plan["counts"]["eligible_training_frames"] == 8
    reference = plan["records"][0]
    assert reference["conditioning"]["verb"] == "canonical_reference"
    assert reference["prompt"].endswith("neutral side-view reference")
    assert "attack" not in reference["prompt"]
    assert "walking" not in reference["prompt"]
    assert reference["target"]["eligible_frame_indices"] == list(range(8))
    assert reference["target"]["reference_frame_index"] == 2
    assert reference["target"]["reference_frame_array_content_sha256"] == "f" * 64
    assert reference["prompt_projection"] == "legacy_literal_description_v1"
    assert plan["sampler_contract"]["motion_or_action_text_in_prompt"] is False
    assert plan["sampler_contract"]["caption_reference"] == ("fixed_verified_idle_temporal_medoid")


def test_dense_still_plan_export_is_no_clobber(tmp_path: Path) -> None:
    materialization = _fixture(tmp_path)
    output = tmp_path / "plan.json"
    guard = DiskGuard(tmp_path, 0)

    digest = export_mugen_dense_still_training_plan(materialization, output, disk_guard=guard)

    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        export_mugen_dense_still_training_plan(materialization, output, disk_guard=guard)


def test_dense_still_plan_compacts_hash_bound_structured_caption(tmp_path: Path) -> None:
    materialization = _fixture(tmp_path)
    value = json.loads(materialization.read_text(encoding="utf-8"))
    caption_manifest = {
        "record_count": 1,
        "records": [
            {
                "identity_id": "identity-a",
                "structured_caption": {
                    "subject_type": "humanoid",
                    "body_build": "stocky fighter",
                    "skin_or_surface": "tan skin",
                    "hair": "spiky black hair",
                    "face": "brown eyes",
                    "upper_body_clothing": "green sleeveless shirt",
                    "lower_body_clothing": "gray pants",
                    "footwear": "brown boots",
                    "equipment": ["silver sword"],
                    "dominant_colors": ["green", "gray", "brown"],
                },
            }
        ],
    }
    caption_bytes = json.dumps(caption_manifest).encode()
    caption_path = tmp_path / "captions.json"
    caption_path.write_bytes(caption_bytes)
    value["source"] = {
        "caption_manifest_file_sha256": hashlib.sha256(caption_bytes).hexdigest(),
        "caption_manifest_path": str(caption_path),
    }
    materialization.write_text(json.dumps(value), encoding="utf-8")

    plan = build_mugen_dense_still_training_plan(materialization)

    reference = plan["records"][0]
    assert reference["prompt_projection"] == "structured_priority_whole_phrase_32_words_v1"
    assert "green sleeveless shirt" in reference["prompt"]
    assert "neutral side view" in reference["prompt"]
    assert (
        plan["source"]["caption_manifest_file_sha256"] == hashlib.sha256(caption_bytes).hexdigest()
    )
