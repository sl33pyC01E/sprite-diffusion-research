from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.training_data import (
    TrainingDataError,
    collate_materialized_clips,
    load_materialized_training_clips,
    model_to_rgba_uint8,
    rgba_uint8_to_model,
)


def _array_sha256(array: np.ndarray) -> str:
    header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _manifest(tmp_path: Path, array: np.ndarray, *, loop_mode: str = "one_shot") -> Path:
    clip = tmp_path / "clips" / "train" / "clip.npy"
    clip.parent.mkdir(parents=True)
    with clip.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    payload = {
        "schema_version": 1,
        "sequence_count": 1,
        "sequences": [
            {
                "action": "run",
                "caption": {"description": "red fox"},
                "direction": "right",
                "entity_class": "animal",
                "frame_count": array.shape[0],
                "identity_id": "fox-red",
                "loop_mode": loop_mode,
                "output": {
                    "array_content_sha256": _array_sha256(array),
                    "dtype": "uint8",
                    "file_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                    "format": "numpy_npy_v1",
                    "relative_path": "clips/train/clip.npy",
                    "shape": list(array.shape),
                    "size_bytes": clip.stat().st_size,
                },
                "provenance": {
                    "source_blob_sha256": ["a" * 64],
                    "source_id": "fixture",
                },
                "quality_tier": "F0",
                "sequence_id": "fox-run",
                "split": "train",
                "target_bucket": [array.shape[2], array.shape[1]],
                "timing": {
                    "duration_ms": [100] * array.shape[0],
                    "phase": (
                        [index / array.shape[0] for index in range(array.shape[0])]
                        if loop_mode == "loop"
                        else [index / (array.shape[0] - 1) for index in range(array.shape[0])]
                    ),
                },
                "view": "side",
            }
        ],
        "source_snapshot": {
            "canonical_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "schema_version": 1,
        },
    }
    path = tmp_path / "materialization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_verified_loader_and_temporal_selection(tmp_path: Path) -> None:
    array = np.zeros((3, 4, 4, 4), dtype=np.uint8)
    array[0, ..., 0] = 255
    array[1, ..., 1] = 255
    array[2, ..., 2] = 255
    array[..., 3] = 255
    manifest = _manifest(tmp_path, array)

    (clip,) = load_materialized_training_clips(
        manifest,
        target_bucket=4,
        target_frames=5,
    )

    assert clip.rgba.shape == (5, 4, 4, 4)
    assert clip.temporal_selection is not None
    assert clip.temporal_selection.source_ordinals == (0, 0, 1, 1, 2)
    assert clip.frame_phases == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert clip.source_duration_ms == (100.0, 100.0, 100.0)
    assert sum(clip.duration_ms) == pytest.approx(300.0)
    assert clip.temporal_duration_method == ("selected_authored_duration_weights_preserve_total_v1")
    batch = collate_materialized_clips((clip,))
    assert batch.clean.shape == (1, 5, 4, 4, 4)
    assert batch.frame_phases.shape == (1, 5)
    assert batch.requests[0].description == "red fox"


def test_rgba_model_conversion_is_premultiplied_and_reversible() -> None:
    rgba = np.array(
        [[[[255, 128, 0, 255], [200, 100, 50, 128], [99, 88, 77, 0]]]],
        dtype=np.uint8,
    )

    model = rgba_uint8_to_model(rgba)
    restored = model_to_rgba_uint8(model)

    assert model.shape == (1, 4, 1, 3)
    assert model[0, 0, 0, 2] == -1.0
    assert np.array_equal(restored[..., 0, :], rgba[..., 0, :])
    assert np.max(np.abs(restored[..., 1, :].astype(int) - rgba[..., 1, :].astype(int))) <= 1
    assert tuple(restored[0, 0, 2]) == (0, 0, 0, 0)


def test_loader_rejects_file_and_array_manifest_tampering(tmp_path: Path) -> None:
    array = np.zeros((2, 2, 2, 4), dtype=np.uint8)
    manifest = _manifest(tmp_path, array)
    payload = json.loads(manifest.read_text())
    payload["sequences"][0]["output"]["file_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))

    with pytest.raises(TrainingDataError, match="file SHA-256 mismatch"):
        load_materialized_training_clips(manifest)

    payload["sequences"][0]["output"]["file_sha256"] = hashlib.sha256(
        (tmp_path / "clips/train/clip.npy").read_bytes()
    ).hexdigest()
    payload["sequences"][0]["output"]["array_content_sha256"] = "1" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TrainingDataError, match="array SHA-256 mismatch"):
        load_materialized_training_clips(manifest)


def test_loader_rejects_path_escape_and_filtered_requested_id(tmp_path: Path) -> None:
    array = np.zeros((2, 2, 2, 4), dtype=np.uint8)
    manifest = _manifest(tmp_path, array)
    payload = json.loads(manifest.read_text())
    payload["sequences"][0]["output"]["relative_path"] = "../outside.npy"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TrainingDataError, match="unsafe"):
        load_materialized_training_clips(manifest)

    manifest = _manifest(tmp_path / "filtered", array)
    with pytest.raises(TrainingDataError, match="excluded"):
        load_materialized_training_clips(
            manifest,
            sequence_ids=("fox-run",),
            split="test",
        )


def test_batch_rejects_mixed_model_shapes(tmp_path: Path) -> None:
    first = np.zeros((2, 2, 2, 4), dtype=np.uint8)
    second = np.zeros((2, 3, 3, 4), dtype=np.uint8)
    clip_one = load_materialized_training_clips(_manifest(tmp_path / "one", first))[0]
    clip_two = load_materialized_training_clips(_manifest(tmp_path / "two", second))[0]

    with pytest.raises(ValueError, match="one model shape"):
        collate_materialized_clips((clip_one, clip_two))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda record: record["output"].update({"format": "raw"}), "output format"),
        (lambda record: record["output"].update({"dtype": "float32"}), "dtype"),
        (lambda record: record.update({"frame_count": 99}), "frame_count mismatch"),
        (lambda record: record.update({"target_bucket": [9, 9]}), "target_bucket mismatch"),
    ),
)
def test_loader_rejects_false_array_semantics(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    array = np.zeros((2, 2, 2, 4), dtype=np.uint8)
    manifest = _manifest(tmp_path, array)
    payload = json.loads(manifest.read_text())
    mutation(payload["sequences"][0])  # type: ignore[operator]
    manifest.write_text(json.dumps(payload))

    with pytest.raises(TrainingDataError, match=message):
        load_materialized_training_clips(manifest)


def test_loader_rejects_invalid_native_phase_without_resampling(tmp_path: Path) -> None:
    array = np.zeros((3, 2, 2, 4), dtype=np.uint8)
    manifest = _manifest(tmp_path, array)
    payload = json.loads(manifest.read_text())
    payload["sequences"][0]["timing"]["phase"] = [0.0, 0.8, 0.4]
    manifest.write_text(json.dumps(payload))

    with pytest.raises(TrainingDataError, match="nondecreasing"):
        load_materialized_training_clips(manifest)


def test_loader_rejects_manifest_count_and_provenance_shape(tmp_path: Path) -> None:
    array = np.zeros((2, 2, 2, 4), dtype=np.uint8)
    manifest = _manifest(tmp_path, array)
    payload = json.loads(manifest.read_text())
    payload["sequence_count"] = 2
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TrainingDataError, match="sequence_count mismatch"):
        load_materialized_training_clips(manifest)

    payload["sequence_count"] = 1
    payload["sequences"][0]["provenance"] = "not-an-object"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TrainingDataError, match="provenance must be an object"):
        load_materialized_training_clips(manifest)


def test_loader_rejects_npz_container_disguised_as_npy(tmp_path: Path) -> None:
    array = np.zeros((2, 2, 2, 4), dtype=np.uint8)
    manifest = _manifest(tmp_path, array)
    payload = json.loads(manifest.read_text())
    clip = tmp_path / "clips" / "train" / "clip.npy"
    with clip.open("wb") as handle:
        np.savez(handle, rgba=array)
    payload["sequences"][0]["output"]["file_sha256"] = hashlib.sha256(clip.read_bytes()).hexdigest()
    payload["sequences"][0]["output"]["size_bytes"] = clip.stat().st_size
    manifest.write_text(json.dumps(payload))

    with pytest.raises(TrainingDataError, match="must contain one NumPy array"):
        load_materialized_training_clips(manifest)


def test_intro_then_loop_projects_only_verified_contiguous_tail(tmp_path: Path) -> None:
    array = np.zeros((4, 2, 2, 4), dtype=np.uint8)
    for index in range(4):
        array[index, ..., 0] = 10 + index
        array[index, ..., 3] = 255
    manifest = _manifest(tmp_path, array)
    payload = json.loads(manifest.read_text())
    record = payload["sequences"][0]
    record["loop_mode"] = "intro_then_loop"
    record["timing"] = {
        "duration_ms": [10.0, 20.0, 30.0, 40.0],
        "phase": [None, None, 0.0, 0.5],
    }
    manifest.write_text(json.dumps(payload))

    native = load_materialized_training_clips(manifest)[0]
    assert native.source_loop_mode == "intro_then_loop"
    assert native.request.loop_mode == "loop"
    assert native.rgba[:, 0, 0, 0].tolist() == [12, 13]
    assert native.frame_phases == (0.0, 0.5)
    assert native.source_duration_ms == (10.0, 20.0, 30.0, 40.0)
    assert native.duration_ms == (30.0, 40.0)
    assert native.intro_loop_projection is not None
    assert native.intro_loop_projection.prefix_frame_count == 2
    assert native.intro_loop_projection.loop_source_ordinals == (2, 3)
    assert native.intro_loop_projection.discarded_prefix_duration_ms == 30.0
    assert native.intro_loop_projection.loop_total_duration_ms == 70.0

    resampled = load_materialized_training_clips(manifest, target_frames=4)[0]
    assert resampled.rgba.shape == (4, 2, 2, 4)
    assert sum(resampled.duration_ms) == pytest.approx(70.0)
    assert resampled.temporal_duration_method == (
        "intro_then_loop_verified_tail_then_selected_duration_weights_preserve_loop_total_v1"
    )


def test_intro_then_loop_rejects_noncontiguous_or_unphased_tail(tmp_path: Path) -> None:
    array = np.zeros((4, 2, 2, 4), dtype=np.uint8)
    manifest = _manifest(tmp_path, array)
    payload = json.loads(manifest.read_text())
    record = payload["sequences"][0]
    record["loop_mode"] = "intro_then_loop"
    record["timing"]["phase"] = [None, 0.0, None, 0.5]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TrainingDataError, match="not contiguous"):
        load_materialized_training_clips(manifest)

    record["timing"]["phase"] = [None, None, None, None]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(TrainingDataError, match="no phased loop tail"):
        load_materialized_training_clips(manifest)
