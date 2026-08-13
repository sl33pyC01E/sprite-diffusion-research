from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.training_audit import (
    build_materialization_training_audit,
    export_materialization_training_audit,
)


def _array_sha256(array: np.ndarray) -> str:
    header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    root = tmp_path / "materialized"
    rows = []
    specifications = (
        ("wolf-idle", "wolf", "idle", "train", "pack-a", 10),
        ("wolf-run", "wolf", "run", "train", "pack-a", 20),
        ("fox-idle", "fox", "idle", "validation", "pack-a", 30),
        ("bot-run", "bot", "run", "test", "pack-b", 40),
    )
    for sequence_id, identity, action, split, pack, value in specifications:
        array = np.zeros((2, 4, 4, 4), dtype=np.uint8)
        array[:, 1:3, 1:3, 0] = value
        array[:, 1:3, 1:3, 3] = 255
        relative = Path("clips") / split / f"{sequence_id}.npy"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        rows.append(
            {
                "action": action,
                "caption": {"description": f"{identity} sprite"},
                "direction": "unknown",
                "entity_class": "animal" if identity != "bot" else "robot",
                "frame_count": 2,
                "identity_id": identity,
                "loop_mode": "loop",
                "output": {
                    "array_content_sha256": _array_sha256(array),
                    "dtype": "uint8",
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "format": "numpy_npy_v1",
                    "relative_path": relative.as_posix(),
                    "shape": list(array.shape),
                    "size_bytes": path.stat().st_size,
                },
                "provenance": {
                    "source_blob_sha256": [hashlib.sha256(identity.encode()).hexdigest()],
                    "source_id": "fixture",
                    "source_pack_id": pack,
                },
                "quality_tier": "F0",
                "sequence_id": sequence_id,
                "split": split,
                "target_bucket": [4, 4],
                "timing": {"duration_ms": [100, 100], "phase": [0.0, 0.5]},
                "view": "side",
            }
        )
    manifest = {
        "schema_version": 1,
        "sequence_count": len(rows),
        "sequences": rows,
        "source_snapshot": {
            "canonical_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "schema_version": 1,
        },
    }
    path = root / "materialization.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_audit_reports_coverage_contrasts_and_pack_overlap(tmp_path: Path) -> None:
    artifact = build_materialization_training_audit(_manifest(tmp_path), target_frames=2)

    assert artifact["schema_version"] == 2
    assert artifact["coverage"]["sequence_count"] == 4
    assert artifact["coverage"]["identity_count"] == 3
    assert artifact["coverage"]["multi_action_identity_count"] == 1
    assert artifact["coverage"]["source_loop_mode_counts"] == {"loop": 4}
    assert artifact["splits"]["train"]["action_counts"] == {"idle": 1, "run": 1}
    assert artifact["endpoint_action_contrasts"]["partitions"]["train:4x4"] == {
        "conflicting_same_action_rows": 0,
        "contrast_group_count": 1,
        "cross_action_alias_rows_omitted": 0,
        "duplicate_target_rows_omitted": 0,
        "endpoint_excluded_row_count": 0,
        "no_target_distinct_rows_omitted": 0,
        "selected_representative_count": 2,
        "sequence_count": 2,
    }
    assert artifact["split_leakage_audit"]["identity_id"] == []
    assert artifact["split_leakage_audit"]["source_blob_sha256"] == []
    assert artifact["split_leakage_audit"]["source_pack_id"] == [
        {"splits": ["train", "validation"], "value": "pack-a"}
    ]


def test_export_is_canonical_and_no_clobber(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    output = tmp_path / "audit.json"
    result = export_materialization_training_audit(
        manifest,
        output,
        target_frames=2,
    )

    assert result.sequence_count == 4
    assert result.fixed_frame_count == 4
    assert result.artifact_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError, match="Refusing"):
        export_materialization_training_audit(manifest, output, target_frames=2)


def test_audit_rejects_manifest_count_mismatch(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["sequence_count"] = 99
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="sequence_count mismatch"):
        build_materialization_training_audit(manifest, target_frames=2)


def test_audit_excludes_cross_action_pixel_aliases_from_causal_contrasts(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload["sequences"]
    idle = next(row for row in rows if row["sequence_id"] == "wolf-idle")
    run = next(row for row in rows if row["sequence_id"] == "wolf-run")
    run_path = manifest.parent / run["output"]["relative_path"]
    idle_path = manifest.parent / idle["output"]["relative_path"]
    run_path.write_bytes(idle_path.read_bytes())
    run["output"].update(
        {
            "array_content_sha256": idle["output"]["array_content_sha256"],
            "file_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
            "size_bytes": run_path.stat().st_size,
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    artifact = build_materialization_training_audit(manifest, target_frames=2)

    assert artifact["coverage"]["sequence_count"] == 4
    assert artifact["endpoint_action_contrasts"]["partitions"]["train:4x4"] == {
        "conflicting_same_action_rows": 0,
        "contrast_group_count": 0,
        "cross_action_alias_rows_omitted": 1,
        "duplicate_target_rows_omitted": 0,
        "endpoint_excluded_row_count": 2,
        "no_target_distinct_rows_omitted": 1,
        "selected_representative_count": 0,
        "sequence_count": 2,
    }
