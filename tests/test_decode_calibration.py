from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.decode import GlobalPaletteDecodeConfig, global_palette_decode_rgba
from spritelab.decode_calibration import (
    CalibrationArrayRef,
    export_global_palette_size_calibration,
    export_hard_alpha_threshold_calibration,
)


def _write_array(path: Path, rgba: np.ndarray) -> Path:
    with path.open("wb") as handle:
        np.save(handle, rgba, allow_pickle=False)
    return path


def _array_hash(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def test_calibration_finds_perfect_hard_mask_and_labels_training_estimate(
    tmp_path: Path,
) -> None:
    source = np.zeros((2, 2, 2, 4), dtype=np.uint8)
    source[:, 0, 1] = (20, 40, 60, 160)
    target = np.zeros_like(source)
    target[:, 0, 1] = (20, 40, 60, 255)
    source_path = _write_array(tmp_path / "source.npy", source)
    target_path = _write_array(tmp_path / "target.npy", target)

    result = export_hard_alpha_threshold_calibration(
        [CalibrationArrayRef("sprite-idle", source_path)],
        [CalibrationArrayRef("sprite-idle", target_path)],
        [192, 128],
        tmp_path / "calibration.json",
        estimate_kind="training_target_estimate",
    )

    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert result.selected_threshold == 128
    assert artifact["estimate"]["held_out"] is False
    assert "not held-out validation" in artifact["estimate"]["interpretation"]
    selected = artifact["selection"]["selected_aggregate_metrics"]
    assert selected == {
        "alpha_iou": 1.0,
        "alpha_mae": 0.0,
        "premultiplied_rgba_mae": 0.0,
    }


def test_calibration_ties_choose_lowest_numeric_threshold(tmp_path: Path) -> None:
    empty = np.zeros((1, 2, 2, 4), dtype=np.uint8)
    source_path = _write_array(tmp_path / "source.npy", empty)
    target_path = _write_array(tmp_path / "target.npy", empty)

    result = export_hard_alpha_threshold_calibration(
        [CalibrationArrayRef("empty", source_path)],
        [CalibrationArrayRef("empty", target_path)],
        [224, 32, 128],
        tmp_path / "tie.json",
        estimate_kind="held_out_validation",
    )

    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert result.selected_threshold == 32
    assert artifact["thresholds"] == [32, 128, 224]
    assert artifact["estimate"]["held_out"] is True
    assert all(
        row["aggregate_metrics"]["alpha_iou"] == 1.0 for row in artifact["threshold_results"]
    )


def test_calibration_rejects_mismatched_pair_order(tmp_path: Path) -> None:
    rgba = np.zeros((1, 1, 1, 4), dtype=np.uint8)
    a = _write_array(tmp_path / "a.npy", rgba)
    b = _write_array(tmp_path / "b.npy", rgba)

    with pytest.raises(ValueError, match="order must match exactly"):
        export_hard_alpha_threshold_calibration(
            [CalibrationArrayRef("a", a), CalibrationArrayRef("b", b)],
            [CalibrationArrayRef("b", b), CalibrationArrayRef("a", a)],
            [128],
            tmp_path / "order.json",
            estimate_kind="held_out_validation",
        )


def test_calibration_rejects_mismatched_shapes_and_dtype(tmp_path: Path) -> None:
    source = _write_array(tmp_path / "source.npy", np.zeros((1, 2, 2, 4), dtype=np.uint8))
    target = _write_array(tmp_path / "target.npy", np.zeros((1, 3, 2, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="identical .* shapes"):
        export_hard_alpha_threshold_calibration(
            [CalibrationArrayRef("sample", source)],
            [CalibrationArrayRef("sample", target)],
            [128],
            tmp_path / "shape.json",
            estimate_kind="held_out_validation",
        )

    wrong_dtype = _write_array(tmp_path / "float.npy", np.zeros((1, 2, 2, 4), dtype=np.float32))
    with pytest.raises(TypeError, match="dtype uint8"):
        export_hard_alpha_threshold_calibration(
            [CalibrationArrayRef("sample", wrong_dtype)],
            [CalibrationArrayRef("sample", target)],
            [128],
            tmp_path / "dtype.json",
            estimate_kind="held_out_validation",
        )


def test_calibration_artifact_is_canonical_hashed_and_no_clobber(tmp_path: Path) -> None:
    source_array = np.zeros((1, 2, 2, 4), dtype=np.uint8)
    source_array[0, 0, 0] = (10, 20, 30, 100)
    target_array = np.zeros_like(source_array)
    source_path = _write_array(tmp_path / "source.npy", source_array)
    target_path = _write_array(tmp_path / "target.npy", target_array)
    output = tmp_path / "artifact.json"

    result = export_hard_alpha_threshold_calibration(
        [CalibrationArrayRef("sample", source_path)],
        [CalibrationArrayRef("sample", target_path)],
        [64, 128],
        output,
        estimate_kind="held_out_validation",
    )

    payload = output.read_bytes()
    artifact = json.loads(payload)
    pair = artifact["inputs"]["pairs"][0]
    assert payload == (
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    assert result.artifact_sha256 == hashlib.sha256(payload).hexdigest()
    assert pair["source"]["file_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert pair["target"]["file_sha256"] == hashlib.sha256(target_path.read_bytes()).hexdigest()
    assert pair["source"]["array_sha256"] == _array_hash(source_array)
    assert pair["target"]["array_sha256"] == _array_hash(target_array)

    with pytest.raises(FileExistsError, match="Refusing"):
        export_hard_alpha_threshold_calibration(
            [CalibrationArrayRef("sample", source_path)],
            [CalibrationArrayRef("sample", target_path)],
            [64, 128],
            output,
            estimate_kind="held_out_validation",
        )


def test_palette_calibration_selects_target_matching_generated_only_palette(
    tmp_path: Path,
) -> None:
    source = np.zeros((2, 2, 2, 4), dtype=np.uint8)
    source[:, 0, 0] = (10, 20, 30, 255)
    source[:, 0, 1] = (20, 30, 40, 255)
    source[:, 1, 0] = (220, 230, 240, 255)
    target = global_palette_decode_rgba(
        source,
        config=GlobalPaletteDecodeConfig(alpha_threshold=128, maximum_colors=2),
    )
    source_path = _write_array(tmp_path / "source.npy", source)
    target_path = _write_array(tmp_path / "target.npy", target)

    result = export_global_palette_size_calibration(
        [CalibrationArrayRef("sprite-walk", source_path)],
        [CalibrationArrayRef("sprite-walk", target_path)],
        [4, 2],
        tmp_path / "palette.json",
        alpha_threshold=128,
        estimate_kind="training_target_estimate",
    )

    artifact = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    assert result.selected_maximum_colors == 2
    assert artifact["palette_sizes"] == [2, 4]
    assert artifact["parameters"] == {"alpha_threshold": 128}
    assert artifact["decode_operation"]["reference_or_target_palette_used"] is False
    assert artifact["selection"]["selected_aggregate_metrics"]["premultiplied_rgba_mae"] == 0.0


def test_palette_calibration_ties_choose_smaller_palette_and_refuses_clobber(
    tmp_path: Path,
) -> None:
    empty = np.zeros((1, 2, 2, 4), dtype=np.uint8)
    source_path = _write_array(tmp_path / "source.npy", empty)
    target_path = _write_array(tmp_path / "target.npy", empty)
    output = tmp_path / "palette.json"
    arguments = (
        [CalibrationArrayRef("empty", source_path)],
        [CalibrationArrayRef("empty", target_path)],
        [32, 8],
        output,
    )

    result = export_global_palette_size_calibration(
        *arguments,
        alpha_threshold=192,
        estimate_kind="held_out_validation",
    )

    assert result.selected_maximum_colors == 8
    payload = output.read_bytes()
    assert result.artifact_sha256 == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError, match="Refusing"):
        export_global_palette_size_calibration(
            *arguments,
            alpha_threshold=192,
            estimate_kind="held_out_validation",
        )


@pytest.mark.parametrize("palette_sizes", [[], [1], [257], [8, 8], [True]])
def test_palette_calibration_rejects_invalid_palette_sizes(
    tmp_path: Path, palette_sizes: list[int]
) -> None:
    rgba = np.zeros((1, 1, 1, 4), dtype=np.uint8)
    source = _write_array(tmp_path / "source.npy", rgba)
    target = _write_array(tmp_path / "target.npy", rgba)

    with pytest.raises((TypeError, ValueError)):
        export_global_palette_size_calibration(
            [CalibrationArrayRef("sample", source)],
            [CalibrationArrayRef("sample", target)],
            palette_sizes,
            tmp_path / "invalid.json",
            alpha_threshold=128,
            estimate_kind="held_out_validation",
        )
