from __future__ import annotations

import json
from pathlib import Path

import pytest

from spritelab.mugen_motion_training import (
    MugenMotionTrainingSelectionConfig,
    build_mugen_motion_training_manifest,
    export_mugen_motion_training_manifest,
)


def _write_fixture(root: Path) -> tuple[Path, Path]:
    records = []
    pixels = []
    for index, (verb, split, status) in enumerate(
        (
            ("idle", "train", "all_pass"),
            ("walk", "validation", "all_pass"),
            ("idle", "test", "mixed"),
            ("special_attack", "train", "all_pass"),
        )
    ):
        sequence_id = f"sequence_{index}"
        identity_id = f"identity_{index}"
        records.append(
            {
                "conditioning": {"verb": verb},
                "entity_class": "humanoid",
                "identity_id": identity_id,
                "reference": {},
                "reference_target_relation": "cross_sequence",
                "sample_id": f"sample_{index}",
                "sequence_id": sequence_id,
                "split": split,
                "target": {},
            }
        )
        pixels.append(
            {
                "frames": [
                    {
                        "anchored_overlap": 0.8,
                        "bbox_iou": 0.7,
                        "candidate_palette_coverage": 0.6 + index * 0.01,
                        "occupancy_ratio": 1.0,
                        "palette_histogram_intersection": 0.5,
                    }
                    for _ in range(8)
                ],
                "pixel_gate_pass_indices": list(range(8)) if status == "all_pass" else [0],
                "pixel_gate_status": status,
                "sequence_id": sequence_id,
            }
        )
    plan = {
        "artifact_kind": "mugen_reference_conditioned_latent_motion_plan",
        "counts": {"sequences": len(records)},
        "records": records,
    }
    audit = {
        "artifact_kind": "mugen_subject_bearing_frame_pixel_gate",
        "counts": {"sequences": len(pixels)},
        "records": pixels,
    }
    plan_path = root / "plan.json"
    audit_path = root / "audit.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return plan_path, audit_path


def test_manifest_requires_approved_verb_and_all_frames(tmp_path: Path) -> None:
    plan, audit = _write_fixture(tmp_path)

    artifact = build_mugen_motion_training_manifest(
        plan,
        audit,
        config=MugenMotionTrainingSelectionConfig(verbs=("idle", "walk")),
    )

    assert [record["sequence_id"] for record in artifact["records"]] == [
        "sequence_0",
        "sequence_1",
    ]
    assert artifact["counts"]["splits"] == {"train": 1, "validation": 1}
    assert artifact["counts"]["exclusions"] == {
        "pixel_gate:mixed": 1,
        "verb:special_attack": 1,
    }


def test_export_is_canonical_and_no_clobber(tmp_path: Path) -> None:
    plan, audit = _write_fixture(tmp_path)
    output = tmp_path / "manifest.json"
    config = MugenMotionTrainingSelectionConfig(verbs=("idle", "walk"))

    path, digest = export_mugen_motion_training_manifest(plan, audit, output, config=config)

    assert path == output.resolve()
    assert len(digest) == 64
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        export_mugen_motion_training_manifest(plan, audit, output, config=config)


def test_canonical_mode_selects_one_highest_quality_identity_verb(tmp_path: Path) -> None:
    plan, audit = _write_fixture(tmp_path)
    plan_value = json.loads(plan.read_text(encoding="utf-8"))
    audit_value = json.loads(audit.read_text(encoding="utf-8"))
    duplicate = dict(plan_value["records"][0])
    duplicate["sequence_id"] = "sequence_better"
    duplicate["sample_id"] = "sample_better"
    plan_value["records"].append(duplicate)
    plan_value["counts"]["sequences"] += 1
    pixel = json.loads(json.dumps(audit_value["records"][0]))
    pixel["sequence_id"] = "sequence_better"
    for frame in pixel["frames"]:
        frame["candidate_palette_coverage"] = 0.99
    audit_value["records"].append(pixel)
    audit_value["counts"]["sequences"] += 1
    plan.write_text(json.dumps(plan_value), encoding="utf-8")
    audit.write_text(json.dumps(audit_value), encoding="utf-8")

    artifact = build_mugen_motion_training_manifest(
        plan,
        audit,
        config=MugenMotionTrainingSelectionConfig(
            verbs=("idle", "walk"), one_sequence_per_identity_verb=True
        ),
    )

    assert "sequence_better" in {record["sequence_id"] for record in artifact["records"]}
    assert artifact["counts"]["exclusions"]["noncanonical_identity_verb_variant"] == 1
