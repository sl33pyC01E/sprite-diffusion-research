from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from spritelab.mugen_dense_manifest import build_mugen_dense_manifest
from spritelab.mugen_materialization_view import normalize_mugen_materialization
from spritelab.mugen_stream_quality import export_mugen_stream_quality_audit
from spritelab.storage import DiskGuard


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _legacy_root(root: Path, *, clipping: int = 0) -> Path:
    root.mkdir()
    slots = ("idle", "walk", "jump", "block", "attack_a", "attack_b")
    clips = []
    slot_record_ids = {}
    world = {"scale": 1.0, "target_size": 128}
    for index, slot in enumerate(slots):
        value = np.zeros((8, 128, 128, 4), dtype=np.uint8)
        value[:, 20:28, 50 + index : 58 + index] = (index + 1, 2, 3, 255)
        value[4:, 28:32, 50 + index : 58 + index] = (4, 5, 6, 255)
        path = root / f"{slot}.npy"
        np.save(path, value, allow_pickle=False)
        record_id = f"record-{slot}"
        slot_record_ids[slot] = record_id
        clips.append(
            {
                "action_number": index,
                "array": {
                    "array_content_sha256": _array_sha256(value),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "relative_path": path.name,
                },
                "clipped_visible_pixels": clipping if slot == "attack_a" else 0,
                "identity_id": "legacy-identity",
                "loop_mode": "loop",
                "record_id": record_id,
                "schema_phase": None,
                "schema_verb": slot,
                "slot": slot,
                "temporal_selection": {
                    "source_frame_count": 2,
                    "target_phases": [frame / 8 for frame in range(8)],
                },
                "world_view_transform": world,
            }
        )
    manifest = {
        "artifact_kind": "mugen_fixed_schema_core_training_view",
        "characters": [
            {
                "complete_six_slot_core": True,
                "definitions": None,
                "identity_id": "legacy-identity",
                "resource": {"title": "Literal Legacy Fighter"},
                "slot_record_ids": slot_record_ids,
                "source": {
                    "air_member": "fighter.air",
                    "archive_sha256": "b" * 64,
                    "sff_member": "fighter.sff",
                    "sff_sha256": "a" * 64,
                },
                "world_view_transform": world,
            }
        ],
        "clips": clips,
        "schema_version": 2,
    }
    (root / "materialization.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_legacy_v2_zero_copy_view_enters_dense_pipeline(tmp_path: Path) -> None:
    root = _legacy_root(tmp_path / "legacy")
    manifest_path = root / "materialization.json"
    payload = manifest_path.read_bytes()
    view = normalize_mugen_materialization(
        json.loads(payload), manifest_sha256=hashlib.sha256(payload).hexdigest()
    )

    assert view.projection_contract == "legacy_fixed_schema_v2_zero_copy"
    assert len(view.characters) == 1
    character = view.characters[0]
    assert character["identity_label_provenance_only"] == "Literal Legacy Fighter"
    assert character["source"]["sff"]["sha256"] == "a" * 64
    assert {clip["source_action_index"] for clip in character["clips"]} == {-1}

    quality = tmp_path / "quality.json"
    export_mugen_stream_quality_audit((root,), quality, disk_guard=DiskGuard(tmp_path, 0))
    dense = build_mugen_dense_manifest((root,), quality)

    assert dense["counts"]["characters"] == 1
    assert dense["source_materializations"][0]["projection_contract"] == (
        "legacy_fixed_schema_v2_zero_copy"
    )
    assert dense["records"][0]["identity"]["label"] == "Literal Legacy Fighter"
    assert {row["slot"] for row in dense["records"][0]["actions"]} == {
        "idle",
        "walk",
        "jump",
        "block",
        "attack_a",
        "attack_b",
    }


def test_legacy_v2_view_retains_but_quality_excludes_reported_clipping(
    tmp_path: Path,
) -> None:
    root = _legacy_root(tmp_path / "legacy", clipping=1)
    payload = (root / "materialization.json").read_bytes()

    view = normalize_mugen_materialization(
        json.loads(payload), manifest_sha256=hashlib.sha256(payload).hexdigest()
    )
    attack = next(clip for clip in view.characters[0]["clips"] if clip["slot"] == "attack_a")
    assert attack["clipped_visible_pixels"] == 1
    quality = tmp_path / "quality.json"
    export_mugen_stream_quality_audit((root,), quality, disk_guard=DiskGuard(tmp_path, 0))
    audit = json.loads(quality.read_bytes())
    assert audit["quality_rows"][0]["broad_eligible"] is False
    assert "visible_pixel_clipping" in audit["quality_rows"][0]["dense_exclusion_reasons"]
