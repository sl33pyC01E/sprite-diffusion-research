from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.pixel_quality import (
    PixelQualityDetectionConfig,
    build_materialized_pixel_quality_audit,
    export_materialized_pixel_quality_audit,
)
from spritelab.training_data import TrainingDataError


def _array_sha256(array: np.ndarray) -> str:
    header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _fixture_manifest(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "materialized"
    opaque_magenta = np.zeros((2, 3, 3, 4), dtype=np.uint8)
    opaque_magenta[0, 0, 0] = (255, 0, 255, 255)
    opaque_magenta[1, 1, 1] = (255, 0, 255, 255)

    transparent_magenta = np.zeros((2, 3, 3, 4), dtype=np.uint8)
    transparent_magenta[:, 0, 0, :3] = (255, 0, 255)

    near_magenta = np.zeros((2, 3, 3, 4), dtype=np.uint8)
    near_magenta[0, 0, 1] = (254, 0, 255, 255)
    near_magenta[1, 2, 2] = (254, 0, 255, 255)

    specifications = (
        ("opaque-magenta", "source-a", "train", opaque_magenta),
        ("transparent-magenta", "source-a", "validation", transparent_magenta),
        ("near-magenta", "source-b", "train", near_magenta),
    )
    records = []
    paths: dict[str, Path] = {}
    for sequence_id, source_id, split, array in specifications:
        relative = Path("clips") / split / f"{sequence_id}.npy"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        paths[sequence_id] = path
        records.append(
            {
                "action": "idle",
                "caption": {"description": sequence_id},
                "direction": "unknown",
                "entity_class": "unknown",
                "frame_count": 2,
                "identity_id": sequence_id,
                "loop_mode": "one_shot",
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
                    "source_blob_sha256": [hashlib.sha256(sequence_id.encode()).hexdigest()],
                    "source_id": source_id,
                },
                "quality_tier": "F0",
                "sequence_id": sequence_id,
                "split": split,
                "target_bucket": [3, 3],
                "timing": {"duration_ms": [100, 100], "phase": [0.0, 1.0]},
                "view": "unknown",
            }
        )
    payload = {
        "schema_version": 1,
        "sequence_count": len(records),
        "sequences": records,
        "source_snapshot": {
            "canonical_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "schema_version": 1,
        },
    }
    manifest = root / "materialization.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, paths


def test_exact_opaque_sentinel_and_alpha_audit(tmp_path: Path) -> None:
    manifest, _ = _fixture_manifest(tmp_path)

    artifact = build_materialized_pixel_quality_audit(manifest)

    assert artifact["verification"]["verified_clip_count"] == 3
    assert artifact["manifest"]["file_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert artifact["detection_config"]["purpose"] == "detection_only"
    assert artifact["interpretation"]["pixel_mutation"] is False
    summary = artifact["summary"]
    assert summary["pixel_count"] == 54
    assert summary["visible_canvas"] == {"count": 4, "denominator": 54, "fraction": 4 / 54}
    assert summary["alpha_distribution"]["fully_opaque"]["count"] == 4
    assert summary["alpha_distribution"]["fully_transparent"]["count"] == 50
    assert summary["alpha_distribution"]["partially_transparent"]["count"] == 0
    sentinel = summary["opaque_rgb_sentinels"]["#ff00ff"]
    assert sentinel["opaque_exact_match_pixels"]["count"] == 2
    assert sentinel["affected_frames"]["count"] == 2
    assert sentinel["affected_clips"]["count"] == 1
    assert summary["border_occupancy"]["visible_pixels"]["count"] == 3
    assert summary["corner_occupancy"]["visible_pixels"]["count"] == 2
    assert summary["clip_flags"]["fully_transparent"]["count"] == 1

    clips = {row["sequence_id"]: row for row in artifact["clips"]}
    assert (
        clips["transparent-magenta"]["statistics"]["opaque_rgb_sentinels"]["#ff00ff"][
            "affected_clip"
        ]
        is False
    )
    assert (
        clips["near-magenta"]["statistics"]["opaque_rgb_sentinels"]["#ff00ff"]["affected_clip"]
        is False
    )


def test_source_grouping_filter_and_custom_exact_color(tmp_path: Path) -> None:
    manifest, _ = _fixture_manifest(tmp_path)

    filtered = build_materialized_pixel_quality_audit(manifest, source_ids=("source-a",))
    assert filtered["verification"]["verified_clip_count"] == 3
    assert filtered["selection"]["selected_clip_count"] == 2
    assert set(filtered["sources"]) == {"source-a"}
    assert set(filtered["splits"]) == {"train", "validation"}
    assert {(row["source_id"], row["split"]) for row in filtered["source_splits"]} == {
        ("source-a", "train"),
        ("source-a", "validation"),
    }

    near = build_materialized_pixel_quality_audit(
        manifest,
        config=PixelQualityDetectionConfig(opaque_rgb_sentinels=((254, 0, 255),)),
        source_ids=("source-b",),
    )
    sentinel = near["summary"]["opaque_rgb_sentinels"]["#fe00ff"]
    assert sentinel["opaque_exact_match_pixels"]["count"] == 2
    assert sentinel["affected_clips"]["count"] == 1

    with pytest.raises(ValueError, match="absent from manifest"):
        build_materialized_pixel_quality_audit(manifest, source_ids=("missing",))


def test_audit_rejects_file_hash_failure(tmp_path: Path) -> None:
    manifest, paths = _fixture_manifest(tmp_path)
    damaged = np.zeros((2, 3, 3, 4), dtype=np.uint8)
    damaged[..., 3] = 255
    with paths["opaque-magenta"].open("wb") as handle:
        np.save(handle, damaged, allow_pickle=False)

    with pytest.raises(TrainingDataError, match="clip (size|file SHA-256) mismatch"):
        build_materialized_pixel_quality_audit(manifest)


def test_export_is_canonical_hash_verified_and_no_clobber(tmp_path: Path) -> None:
    manifest, _ = _fixture_manifest(tmp_path)
    output = tmp_path / "pixel-quality.json"

    result = export_materialized_pixel_quality_audit(
        manifest,
        output,
        source_ids=("source-b",),
    )

    assert result.verified_clip_count == 3
    assert result.selected_clip_count == 1
    assert result.artifact_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError, match="Refusing"):
        export_materialized_pixel_quality_audit(manifest, output)


@pytest.mark.parametrize(
    "colors",
    (
        ((255, 0),),
        ((256, 0, 0),),
        ((True, 0, 0),),
        ((1, 2, 3), (1, 2, 3)),
    ),
)
def test_detection_config_rejects_ambiguous_colors(colors: object) -> None:
    with pytest.raises(ValueError):
        PixelQualityDetectionConfig(opaque_rgb_sentinels=colors)  # type: ignore[arg-type]
