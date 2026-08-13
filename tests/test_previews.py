from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.previews import export_npy_clip_preview, export_rgba_clip_preview


def _clip() -> np.ndarray:
    rgba = np.zeros((2, 2, 3, 4), dtype=np.uint8)
    rgba[:, 0, 0, 0] = 103
    rgba[0, :, 1:, (0, 3)] = 255
    rgba[1, :, 1:, (1, 3)] = 255
    return rgba


def test_preview_preserves_frames_timing_and_nearest_pixels(tmp_path: Path) -> None:
    sample = tmp_path / "sample.npy"
    with sample.open("wb") as handle:
        np.save(handle, _clip(), allow_pickle=False)

    result = export_npy_clip_preview(
        sample,
        tmp_path / "preview",
        artifact_stem="rat-run",
        duration_ms=(100.0, 250.0),
        loop_mode="loop",
        integer_scale=2,
        source_report_sha256="a" * 64,
    )

    with Image.open(result.animated_png_path) as animation:
        assert animation.n_frames == 2
        assert animation.size == (6, 4)
        assert animation.info["loop"] == 0
        animation.seek(0)
        assert animation.info["duration"] == pytest.approx(100)
        assert animation.convert("RGBA").getpixel((0, 0)) == (0, 0, 0, 0)
        assert animation.convert("RGBA").getpixel((2, 0)) == (255, 0, 0, 255)
        animation.seek(1)
        assert animation.info["duration"] == pytest.approx(250)
        assert animation.convert("RGBA").getpixel((0, 0)) == (0, 0, 0, 0)
        assert animation.convert("RGBA").getpixel((2, 0)) == (0, 255, 0, 255)
    with Image.open(result.contact_sheet_path) as sheet:
        assert sheet.size == (12, 4)
        assert sheet.convert("RGBA").getpixel((1, 1)) == (0, 0, 0, 0)
        assert sheet.convert("RGBA").getpixel((3, 1)) == (255, 0, 0, 255)
        assert sheet.convert("RGBA").getpixel((9, 1)) == (0, 255, 0, 255)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["duration_ms"] == [100.0, 250.0]
    assert metadata["resampling"] == "nearest_positive_integer"
    assert metadata["invisible_rgb_policy"] == "zero_where_alpha_is_zero"
    assert metadata["source_sample_sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()
    assert metadata["source_report_sha256"] == "a" * 64

    with pytest.raises(FileExistsError, match="Refusing"):
        export_npy_clip_preview(
            sample,
            tmp_path / "preview",
            artifact_stem="rat-run",
            duration_ms=(100.0, 250.0),
            loop_mode="loop",
        )


def test_preview_rejects_unsafe_or_lossy_inputs(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="uint8"):
        export_rgba_clip_preview(
            _clip().astype(np.float32),
            tmp_path,
            artifact_stem="clip",
            duration_ms=(100.0, 100.0),
            loop_mode="loop",
        )
    with pytest.raises(ValueError, match="artifact_stem"):
        export_rgba_clip_preview(
            _clip(),
            tmp_path,
            artifact_stem="../escape",
            duration_ms=(100.0, 100.0),
            loop_mode="loop",
        )
    with pytest.raises(ValueError, match="length"):
        export_rgba_clip_preview(
            _clip(),
            tmp_path,
            artifact_stem="clip",
            duration_ms=(100.0,),
            loop_mode="loop",
        )


def test_preview_can_preserve_duplicate_temporal_frame_slots(tmp_path: Path) -> None:
    rgba = np.repeat(_clip()[:1], 4, axis=0)

    result = export_rgba_clip_preview(
        rgba,
        tmp_path,
        artifact_stem="held-pose",
        duration_ms=(25.0, 50.0, 75.0, 100.0),
        loop_mode="loop",
        integer_scale=2,
        preserve_frame_slots=True,
    )

    with Image.open(result.animated_png_path) as animation:
        assert animation.n_frames == 4
        for index, duration in enumerate((25, 50, 75, 100)):
            animation.seek(index)
            assert animation.info["duration"] == pytest.approx(duration)
            expected_native = rgba[index].copy()
            expected_native[..., :3][expected_native[..., 3] == 0] = 0
            expected = np.repeat(np.repeat(expected_native, 2, axis=0), 2, axis=1)
            assert np.array_equal(np.asarray(animation.convert("RGBA")), expected)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["preserve_frame_slots"] is True
