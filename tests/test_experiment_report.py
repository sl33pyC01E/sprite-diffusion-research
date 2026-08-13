from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.experiment_report import _premultiplied_distance, evaluate_overfit_experiment
from spritelab.training_data import load_materialized_training_clips


def _array_sha256(array: np.ndarray) -> str:
    header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _fixture(tmp_path: Path, *, target_frames: int = 2) -> tuple[Path, Path]:
    array = np.zeros((2, 4, 4, 4), dtype=np.uint8)
    array[:, 1:3, 1:3, (0, 3)] = 255
    clip_path = tmp_path / "materialized/clips/train/clip.npy"
    clip_path.parent.mkdir(parents=True)
    with clip_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    manifest = {
        "schema_version": 1,
        "sequence_count": 1,
        "sequences": [
            {
                "action": "idle",
                "caption": {"description": "red square"},
                "direction": "unknown",
                "entity_class": "object",
                "frame_count": 2,
                "identity_id": "red-square",
                "loop_mode": "loop",
                "output": {
                    "array_content_sha256": _array_sha256(array),
                    "dtype": "uint8",
                    "file_sha256": hashlib.sha256(clip_path.read_bytes()).hexdigest(),
                    "format": "numpy_npy_v1",
                    "relative_path": "clips/train/clip.npy",
                    "shape": list(array.shape),
                    "size_bytes": clip_path.stat().st_size,
                },
                "provenance": {"source_blob_sha256": ["a" * 64], "source_id": "fixture"},
                "quality_tier": "F0",
                "sequence_id": "red-idle",
                "split": "train",
                "target_bucket": [4, 4],
                "timing": {"duration_ms": [100, 100], "phase": [0.0, 0.5]},
                "view": "unknown",
            }
        ],
        "source_snapshot": {
            "canonical_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "schema_version": 1,
        },
    }
    manifest_path = tmp_path / "materialized/materialization.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    experiment = tmp_path / "experiment"
    target = load_materialized_training_clips(
        manifest_path,
        sequence_ids=("red-idle",),
        split="train",
        target_bucket=4,
        target_frames=target_frames,
    )[0].rgba
    sample = experiment / "samples/sample.npy"
    sample.parent.mkdir(parents=True)
    with sample.open("wb") as handle:
        np.save(handle, target, allow_pickle=False)
    report = {
        "config": {"target_bucket": 4, "target_frames": target_frames},
        "sample_files": [
            {
                "file_sha256": hashlib.sha256(sample.read_bytes()).hexdigest(),
                "path": "samples/sample.npy",
                "sequence_id": "red-idle",
            }
        ],
        "sequence_ids": ["red-idle"],
    }
    (experiment / "overfit-report.json").write_text(json.dumps(report), encoding="utf-8")
    return experiment, manifest_path


def test_exact_target_evaluation_is_hash_verified_and_perfect_for_copy(tmp_path: Path) -> None:
    experiment, manifest = _fixture(tmp_path)
    output = evaluate_overfit_experiment(experiment, manifest)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["sample_count"] == 1
    assert payload["schema_version"] == 2
    assert payload["alpha_visibility_threshold"] == 0
    assert payload["aggregate_mean"]["premultiplied_rgba_mae"] == 0
    assert payload["aggregate_mean"]["alpha_iou"] == 1
    assert payload["aggregate_mean"]["alpha_precision"] == 1
    assert payload["aggregate_mean"]["alpha_recall"] == 1
    assert payload["aggregate_mean"]["target_visible_premultiplied_rgba_mae"] == 0
    assert payload["aggregate_mean"]["target_background_premultiplied_rgba_mae"] == 0
    assert payload["aggregate_mean"]["predicted_visible_canvas_fraction"] == pytest.approx(1 / 4)
    assert payload["aggregate_mean"]["target_visible_canvas_fraction"] == pytest.approx(1 / 4)
    assert payload["aggregate_mean"]["predicted_to_target_visible_canvas_ratio"] == 1
    assert payload["samples"][0]["matched_target"]["exact_frame_match_fraction"] == 1
    assert payload["samples"][0]["matched_target"]["alpha_visibility_threshold"] == 0
    assert payload["causal_action_pair_separation"] == []
    row = payload["samples"][0]
    assert row["source_materialized_array_sha256"] == _array_sha256(
        np.load(manifest.parent / "clips/train/clip.npy", allow_pickle=False)
    )
    assert row["training_target"]["array_sha256"] == row["source_materialized_array_sha256"]

    with pytest.raises(FileExistsError, match="Refusing"):
        evaluate_overfit_experiment(experiment, manifest)


def test_exact_target_hash_describes_retimed_tensor_not_source_array(tmp_path: Path) -> None:
    experiment, manifest = _fixture(tmp_path, target_frames=3)
    output = evaluate_overfit_experiment(experiment, manifest)
    row = json.loads(output.read_text(encoding="utf-8"))["samples"][0]

    assert row["training_target"]["shape"] == [3, 4, 4, 4]
    assert row["training_target"]["temporal_selection"] is not None
    assert row["training_target"]["array_sha256"] != row["source_materialized_array_sha256"]


def test_exact_target_evaluation_rejects_sample_hash_mismatch(tmp_path: Path) -> None:
    experiment, manifest = _fixture(tmp_path)
    sample = experiment / "samples/sample.npy"
    sample.write_bytes(sample.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evaluate_overfit_experiment(experiment, manifest)


def test_exact_target_evaluation_reports_sparse_background_noise(tmp_path: Path) -> None:
    experiment, manifest = _fixture(tmp_path)
    sample_path = experiment / "samples/sample.npy"
    sample = np.load(sample_path, allow_pickle=False)
    sample[:, 0, 0, (1, 3)] = 255
    with sample_path.open("wb") as handle:
        np.save(handle, sample, allow_pickle=False)
    report_path = experiment / "overfit-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["sample_files"][0]["file_sha256"] = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")

    output = evaluate_overfit_experiment(
        experiment,
        manifest,
        alpha_threshold=127,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    aggregate = payload["aggregate_mean"]

    assert payload["alpha_visibility_threshold"] == 127
    assert aggregate["premultiplied_rgba_mae"] == pytest.approx(1 / 32)
    assert aggregate["alpha_precision"] == pytest.approx(4 / 5)
    assert aggregate["alpha_recall"] == 1
    assert aggregate["target_visible_premultiplied_rgba_mae"] == 0
    assert aggregate["target_background_premultiplied_rgba_mae"] == pytest.approx(1 / 24)
    assert aggregate["predicted_visible_canvas_fraction"] == pytest.approx(5 / 16)
    assert aggregate["target_visible_canvas_fraction"] == pytest.approx(1 / 4)
    assert aggregate["predicted_to_target_visible_canvas_ratio"] == pytest.approx(5 / 4)
    assert payload["samples"][0]["matched_target"]["alpha_visibility_threshold"] == 127


def test_premultiplied_pair_distance_ignores_hidden_rgb() -> None:
    left = np.zeros((1, 2, 2, 4), dtype=np.uint8)
    right = left.copy()
    right[..., 0] = 255

    assert _premultiplied_distance(left, right) == 0

    right[0, 0, 0, 3] = 255
    assert _premultiplied_distance(left, right) > 0
