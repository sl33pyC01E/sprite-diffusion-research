from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from spritelab.mugen_motion_dataset import (
    _array_sha256,
    build_mugen_reference_motion_plan,
    export_mugen_reference_motion_plan,
)


def _write_json(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path):
    clips = tmp_path / "clips"
    latents = tmp_path / "latents"
    clips.mkdir(parents=True)
    latents.mkdir()
    sequence_rows = []
    taxonomy_rows = []
    latent_rows = []
    for index, sequence_id in enumerate(("sequence-ref", "sequence-walk")):
        pixel_file_sha = str(index + 1) * 64
        pixel_array_sha = str(index + 3) * 64
        sequence_rows.append(
            {
                "action": "idle" if index == 0 else "walk",
                "entity_class": "humanoid",
                "identity_id": "identity-a",
                "loop_mode": "loop",
                "output": {
                    "array_content_sha256": pixel_array_sha,
                    "file_sha256": pixel_file_sha,
                    "relative_path": f"clips/{sequence_id}.npy",
                    "shape": [8, 128, 128, 4],
                },
                "sequence_id": sequence_id,
                "split": "train",
                "timing": {"duration_ms": [100] * 8, "phase": [i / 8 for i in range(8)]},
            }
        )
        taxonomy_rows.append(
            {
                "attack_form": None,
                "attack_strength": None,
                "attack_tier": None,
                "direction": "forward" if index else None,
                "identity_id": "identity-a",
                "legacy_action": "idle" if index == 0 else "walk",
                "phase": "one_shot",
                "sequence_id": sequence_id,
                "split": "train",
                "stance": "standing",
                "verb": "idle" if index == 0 else "walk",
            }
        )
        array = np.full((8, 8, 64, 64), index, dtype=np.float16)
        latent_path = latents / f"{sequence_id}.npy"
        np.save(latent_path, array, allow_pickle=False)
        latent_rows.append(
            {
                "array_content_sha256": _array_sha256(array),
                "dtype": "float16",
                "file_sha256": hashlib.sha256(latent_path.read_bytes()).hexdigest(),
                "identity_id": "identity-a",
                "relative_path": f"latents/{sequence_id}.npy",
                "sequence_id": sequence_id,
                "shape": [8, 8, 64, 64],
                "source": {
                    "array_content_sha256": pixel_array_sha,
                    "file_sha256": pixel_file_sha,
                },
                "split": "train",
            }
        )
    materialization = tmp_path / "materialization.json"
    taxonomy = tmp_path / "taxonomy.json"
    still = tmp_path / "still.json"
    latent = tmp_path / "latent.json"
    _write_json(
        materialization,
        {"schema_version": 1, "sequence_count": 2, "sequences": sequence_rows},
    )
    _write_json(
        taxonomy,
        {
            "artifact_kind": "mugen_materialized_structured_action_taxonomy",
            "sequence_count": 2,
            "records": taxonomy_rows,
        },
    )
    _write_json(
        still,
        {
            "artifact_kind": "mugen_canonical_appearance_still_training_plan",
            "counts": {"identities": 1},
            "records": [
                {
                    "caption_reference": {"identity_reference_array_sha256": "a" * 64},
                    "identity_id": "identity-a",
                    "prompt": "orange armored fighter",
                    "sample_id": "still-a",
                    "sequence_id": "sequence-ref",
                    "target": {
                        "array_content_sha256": "3" * 64,
                        "eligible_frame_indices": [2],
                        "file_sha256": "1" * 64,
                    },
                }
            ],
        },
    )
    _write_json(
        latent,
        {
            "artifact_kind": "mugen_frozen_rgba_autoencoder_latent_cache",
            "record_count": 2,
            "records": latent_rows,
        },
    )
    return materialization, taxonomy, still, latent


def test_plan_joins_reference_action_and_latent_with_explicit_reference_relation(tmp_path) -> None:
    paths = _fixture(tmp_path)
    plan = build_mugen_reference_motion_plan(*paths)

    assert plan["counts"] == {
        "actions": {"idle": 1, "walk": 1},
        "entity_classes": {"humanoid": 2},
        "identities": 1,
        "reference_target_relation": {"cross_sequence": 1, "same_sequence": 1},
        "sequences": 2,
        "splits": {"train": 2},
    }
    assert plan["records"][0]["reference_target_relation"] == "same_sequence"
    record = plan["records"][1]
    assert record["conditioning"]["verb"] == "walk"
    assert record["reference_target_relation"] == "cross_sequence"
    assert record["reference"]["frame_index"] == 2
    assert record["reference"]["latent"]["frame_shape"] == [8, 64, 64]
    assert record["target"]["latent"]["shape"] == [8, 8, 64, 64]
    assert record["target"]["phase"] == [i / 8 for i in range(8)]


def test_plan_rejects_latent_source_mismatch_and_export_is_no_clobber(tmp_path) -> None:
    paths = _fixture(tmp_path)
    latent_path = paths[-1]
    latent = json.loads(latent_path.read_text())
    latent["records"][1]["source"]["file_sha256"] = "f" * 64
    _write_json(latent_path, latent)
    with pytest.raises(ValueError, match="source pixels differ"):
        build_mugen_reference_motion_plan(*paths)

    paths = _fixture(tmp_path / "second")
    output = tmp_path / "motion-plan.json"
    published, digest = export_mugen_reference_motion_plan(*paths, output)
    assert published == output.resolve()
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="replace"):
        export_mugen_reference_motion_plan(*paths, output)
