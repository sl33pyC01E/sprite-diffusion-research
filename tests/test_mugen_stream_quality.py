from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.mugen_stream_quality import (
    MugenStreamQualityPolicy,
    build_mugen_stream_quality_audit,
    export_mugen_stream_quality_audit,
)
from spritelab.storage import DiskGuard


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _fixture(root: Path, *, scale: float = 1.0, dynamic_slots: int = 6) -> Path:
    root.mkdir()
    clips = []
    slots = ("idle", "walk", "jump", "block", "attack_a", "attack_b")
    for index, slot in enumerate(slots):
        value = np.zeros((8, 128, 128, 4), dtype=np.uint8)
        value[:, 30:34, 60 + index : 64 + index, :] = (index + 1, 2, 3, 255)
        if index < dynamic_slots:
            value[4:, 34:38, 60 + index : 64 + index, :] = (4, 5, 6, 255)
        path = root / f"{slot}.npy"
        np.save(path, value, allow_pickle=False)
        clips.append(
            {
                "action_number": index,
                "array": {
                    "array_content_sha256": _array_sha256(value),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "relative_path": path.name,
                },
                "clipped_visible_pixels": 0,
                "loop_mode": "loop",
                "slot": slot,
                "temporal_selection": {"source_frame_count": 2 if index < dynamic_slots else 1},
            }
        )
    character = {
        "clips": clips,
        "complete_six_slot_core": True,
        "identity_id": "identity-1",
        "source": {"sff": {"sha256": "a" * 64}},
        "variant_id": "variant-1",
        "world_view_transform": {"scale": scale},
    }
    (root / "materialization.json").write_text(
        json.dumps({"characters": [character], "projection_version": 2}), encoding="utf-8"
    )
    return root


def test_quality_audit_verifies_arrays_and_builds_dense_tier(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "materialized")

    audit = build_mugen_stream_quality_audit((root,))

    assert audit["counts"]["characters"] == 1
    assert audit["counts"]["dense_eligible_characters"] == 1
    assert audit["quality_rows"][0]["dynamic_slots"] == 6
    assert audit["quality_rows"][0]["distinct_slot_arrays"] == 6
    idle = next(row for row in audit["quality_rows"][0]["clip_metrics"] if row["slot"] == "idle")
    assert idle["medoid_frame_index"] == 0
    assert idle["medoid_frame_array_content_sha256"] == idle["frame_array_content_sha256"][0]


def test_quality_audit_retains_broad_character_with_explicit_dense_reason(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path / "materialized", scale=0.25, dynamic_slots=1)

    row = build_mugen_stream_quality_audit((root,))["quality_rows"][0]

    assert row["broad_eligible"] is True
    assert row["dense_eligible"] is False
    assert row["dense_exclusion_reasons"] == [
        "view_scale_below_minimum",
        "insufficient_dynamic_slots",
    ]


def test_zero_scale_floor_admits_any_positive_fitted_scale(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "materialized", scale=0.0000001)

    row = build_mugen_stream_quality_audit(
        (root,), policy=MugenStreamQualityPolicy(minimum_view_scale=0)
    )["quality_rows"][0]

    assert row["dense_eligible"] is True
    assert "view_scale_below_minimum" not in row["dense_exclusion_reasons"]


def test_quality_audit_rejects_nonpositive_source_scale(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "materialized", scale=0)

    with pytest.raises(ValueError, match="world view scale must be finite and positive"):
        build_mugen_stream_quality_audit(
            (root,), policy=MugenStreamQualityPolicy(minimum_view_scale=0)
        )


def test_quality_audit_rejects_tampered_array(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "materialized")
    value = np.load(root / "idle.npy", allow_pickle=False)
    value[0, 0, 0] = (1, 2, 3, 4)
    np.save(root / "idle.npy", value, allow_pickle=False)

    with pytest.raises(ValueError, match="array file hash differs"):
        build_mugen_stream_quality_audit((root,))


def test_quality_export_is_no_clobber(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "materialized")
    output = tmp_path / "quality.json"
    guard = DiskGuard(tmp_path, 0)

    digest = export_mugen_stream_quality_audit(
        (root,), output, policy=MugenStreamQualityPolicy(), disk_guard=guard
    )

    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        export_mugen_stream_quality_audit((root,), output, disk_guard=guard)
