from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from scripts.evaluate_mugen_latent_still_v1 import (
    _aggregate_metrics,
    _array_sha256,
    _export_target_preview,
    _load_generated_frame,
    _matched_metrics,
)


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def test_export_target_preview_verifies_and_publishes_exact_frame(tmp_path) -> None:
    array = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    array[3, 20:24, 30:35] = [17, 34, 51, 255]
    clips = tmp_path / "clips"
    clips.mkdir()
    payload = _npy_bytes(array)
    source = clips / "sequence.npy"
    source.write_bytes(payload)
    output = tmp_path / "output"
    output.mkdir()
    record = {
        "prompt": "a compact blue fighter",
        "sequence_id": "sequence-1",
        "target": {
            "array_content_sha256": _array_sha256(array),
            "eligible_frame_indices": list(range(8)),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "reference_frame_array_content_sha256": _array_sha256(np.ascontiguousarray(array[3])),
            "reference_frame_index": 3,
            "relative_path": "clips/sequence.npy",
        },
    }

    result = _export_target_preview(record, output, index=2, plan_root=tmp_path)

    assert result["frame_index"] == 3
    assert result["sequence_id"] == "sequence-1"
    native = Image.open(output / result["native_png"]["path"])
    preview = Image.open(output / result["preview_png"]["path"])
    assert native.size == (128, 128)
    assert preview.size == (512, 512)
    assert native.getpixel((31, 21)) == (17, 34, 51, 255)
    assert preview.getpixel((124, 84)) == (17, 34, 51, 255)


def test_export_target_preview_rejects_tampered_source(tmp_path) -> None:
    array = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    source = tmp_path / "sequence.npy"
    source.write_bytes(_npy_bytes(array))
    output = tmp_path / "output"
    output.mkdir()
    record = {
        "prompt": "fighter",
        "sequence_id": "sequence-1",
        "target": {
            "array_content_sha256": _array_sha256(array),
            "eligible_frame_indices": [0],
            "file_sha256": "0" * 64,
            "relative_path": source.name,
        },
    }

    with pytest.raises(ValueError, match="file SHA-256"):
        _export_target_preview(record, output, index=0, plan_root=tmp_path)


def test_generated_frame_and_matched_metrics_are_hash_bound(tmp_path) -> None:
    target = np.zeros((128, 128, 4), dtype=np.uint8)
    target[20:28, 30:38] = [30, 60, 90, 255]
    prediction = target.copy()
    prediction[20:24, 30:34, :3] = [60, 90, 120]
    payload = _npy_bytes(prediction)
    path = tmp_path / "sample.npy"
    path.write_bytes(payload)
    sample = {
        "array": {
            "array_content_sha256": _array_sha256(prediction),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "path": path.name,
        }
    }

    loaded = _load_generated_frame(sample, tmp_path)
    metrics = _matched_metrics(loaded, target, alpha_threshold=127)
    aggregate = _aggregate_metrics([metrics, metrics])

    assert np.array_equal(loaded, prediction)
    assert metrics["alpha_iou"] == 1
    assert metrics["premultiplied_rgba_mae"] > 0
    assert aggregate["count"] == 2
    assert aggregate["mean"]["premultiplied_rgba_mae"] == metrics["premultiplied_rgba_mae"]

    sample["array"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file SHA-256"):
        _load_generated_frame(sample, tmp_path)
