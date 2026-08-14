from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.broad_train import BroadTrainingContractError, prepare_broad_corpus
from spritelab.mugen_dense_compat import (
    export_mugen_dense_autoencoder_materialization,
    export_mugen_dense_captioned_materialization,
)
from spritelab.storage import DiskGuard
from spritelab.training_data import load_materialized_training_clips


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    value = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    value[:, 20:28, 30:38] = (20, 40, 60, 255)
    value[4:, 28:36, 30:38] = (60, 40, 20, 255)
    path = source / "idle.npy"
    np.save(path, value, allow_pickle=False)
    array = {
        "array_content_sha256": _array_sha256(value),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "relative_path": path.name,
    }
    dense = {
        "artifact_kind": "mugen_dense_reference_motion_training_manifest",
        "quality_audit": {
            "file_sha256": "f" * 64,
            "selected_tier": "dense",
        },
        "records": [
            {
                "actions": [
                    {
                        "array": array,
                        "loop_mode": "loop",
                        "record_id": "sequence-a",
                        "slot": "idle",
                        "temporal_selection": {"target_phases": [index / 8 for index in range(8)]},
                    }
                ],
                "identity": {"label": "fighter alpha"},
                "identity_id": "identity-a",
                "reference": {
                    "frame_array_content_sha256": _array_sha256(np.ascontiguousarray(value[0]))
                },
                "sff_sha256": "a" * 64,
                "source_index": 0,
                "split": "train",
                "variant_id": "variant-a",
            }
        ],
        "schema_version": 1,
        "source_materializations": [{"root": str(source)}],
    }
    dense_path = tmp_path / "dense.json"
    dense_path.write_text(json.dumps(dense), encoding="utf-8")
    return dense_path


def test_dense_bridge_loads_through_verified_training_loader(tmp_path: Path) -> None:
    dense = _fixture(tmp_path)
    output = tmp_path / "bridge.json"

    digest = export_mugen_dense_autoencoder_materialization(
        dense, output, disk_guard=DiskGuard(tmp_path, 0)
    )
    clips = load_materialized_training_clips(output, split="train")

    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert len(clips) == 1
    assert clips[0].rgba.shape == (8, 128, 128, 4)
    assert clips[0].request.description == "fighter alpha"
    assert clips[0].request.action == "idle"
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["model_eligibility"]["conditional_generation"] is False
    with pytest.raises(BroadTrainingContractError, match="not eligible for conditional_generation"):
        prepare_broad_corpus(output, target_size=128, target_frames=8)


def test_dense_bridge_is_no_clobber(tmp_path: Path) -> None:
    dense = _fixture(tmp_path)
    output = tmp_path / "bridge.json"
    guard = DiskGuard(tmp_path, 0)
    export_mugen_dense_autoencoder_materialization(dense, output, disk_guard=guard)

    with pytest.raises(FileExistsError):
        export_mugen_dense_autoencoder_materialization(dense, output, disk_guard=guard)


def test_dense_bridge_preserves_one_shot_phases(tmp_path: Path) -> None:
    dense = _fixture(tmp_path)
    value = json.loads(dense.read_text(encoding="utf-8"))
    action = value["records"][0]["actions"][0]
    action["loop_mode"] = "one_shot"
    action["temporal_selection"]["target_phases"] = [index / 7 for index in range(8)]
    dense.write_text(json.dumps(value), encoding="utf-8")
    output = tmp_path / "bridge.json"

    export_mugen_dense_autoencoder_materialization(dense, output, disk_guard=DiskGuard(tmp_path, 0))
    clips = load_materialized_training_clips(output, split="train")

    assert clips[0].source_loop_mode == "one_shot"
    assert clips[0].frame_phases[-1] == 1.0


def test_captioned_dense_bridge_enables_conditional_loading(tmp_path: Path) -> None:
    dense = _fixture(tmp_path)
    dense_value = json.loads(dense.read_text(encoding="utf-8"))
    reference_sha256 = dense_value["records"][0]["reference"]["frame_array_content_sha256"]
    caption = {
        "artifact_kind": "mugen_dense_literal_visual_caption_dataset",
        "record_count": 1,
        "records": [
            {
                "identity_id": "identity-a",
                "frame_index": 0,
                "reference_frame_array_content_sha256": reference_sha256,
                "request_body_sha256": "c" * 64,
                "split": "train",
                "structured_caption": {"subject_type": "humanoid"},
                "training_appearance_prompt": "a stocky fighter in green clothing",
                "variant_id": "variant-a",
            }
        ],
    }
    caption_path = tmp_path / "captions.json"
    caption_path.write_text(json.dumps(caption), encoding="utf-8")
    output = tmp_path / "captioned.json"

    export_mugen_dense_captioned_materialization(
        dense, caption_path, output, disk_guard=DiskGuard(tmp_path, 0)
    )
    clips = load_materialized_training_clips(output, split="train")

    assert clips[0].request.description == "a stocky fighter in green clothing"
    assert clips[0].request.entity_class == "humanoid"
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["model_eligibility"]["conditional_generation"] is True
