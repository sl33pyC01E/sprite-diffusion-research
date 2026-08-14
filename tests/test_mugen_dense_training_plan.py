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
                "caption": {"description": "a stocky fighter in green clothing"},
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


def test_dense_still_plan_covers_all_six_actions(tmp_path: Path) -> None:
    materialization = _fixture(tmp_path)

    plan = build_mugen_dense_still_training_plan(materialization)

    assert plan["counts"]["sequences"] == 6
    assert set(plan["counts"]["actions"]) == {
        "idle",
        "walk",
        "jump",
        "block",
        "attack_a",
        "attack_b",
    }
    attack = next(row for row in plan["records"] if row["conditioning"]["verb"] == "attack_a")
    assert attack["prompt"].endswith("performing standard attack A; side view")
    assert attack["target"]["eligible_frame_indices"] == list(range(8))


def test_dense_still_plan_export_is_no_clobber(tmp_path: Path) -> None:
    materialization = _fixture(tmp_path)
    output = tmp_path / "plan.json"
    guard = DiskGuard(tmp_path, 0)

    digest = export_mugen_dense_still_training_plan(materialization, output, disk_guard=guard)

    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        export_mugen_dense_still_training_plan(materialization, output, disk_guard=guard)
