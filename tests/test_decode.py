from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.decode import (
    GlobalPaletteDecodeConfig,
    HardAlphaDecodeConfig,
    export_global_palette_decode,
    export_hard_alpha_decode,
    global_palette_decode_rgba,
    hard_alpha_decode_rgba,
)


def test_hard_alpha_decode_is_explicit_and_does_not_change_visible_rgb() -> None:
    rgba = np.array(
        [[[[10, 20, 30, 0], [40, 50, 60, 127], [70, 80, 90, 128]]]],
        dtype=np.uint8,
    )

    decoded = hard_alpha_decode_rgba(rgba)

    assert decoded.tolist() == [[[[0, 0, 0, 0], [0, 0, 0, 0], [70, 80, 90, 255]]]]
    assert rgba[0, 0, 1].tolist() == [40, 50, 60, 127]


def test_decode_export_records_hashes_and_refuses_clobber(tmp_path: Path) -> None:
    rgba = np.zeros((2, 3, 4, 4), dtype=np.uint8)
    rgba[:, 1, 1, :] = (1, 2, 3, 200)
    source = tmp_path / "source.npy"
    with source.open("wb") as handle:
        np.save(handle, rgba, allow_pickle=False)

    result = export_hard_alpha_decode(
        source,
        tmp_path / "decoded.npy",
        config=HardAlphaDecodeConfig(threshold=192),
    )

    decoded = np.load(result.array_path, allow_pickle=False)
    assert decoded[:, 1, 1, :].tolist() == [[1, 2, 3, 255], [1, 2, 3, 255]]
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["threshold"] == 192
    assert metadata["source"]["file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (
        metadata["decoded"]["file_sha256"]
        == hashlib.sha256(result.array_path.read_bytes()).hexdigest()
    )

    with pytest.raises(FileExistsError, match="Refusing"):
        export_hard_alpha_decode(source, tmp_path / "decoded.npy")


def test_hard_alpha_decode_rejects_invalid_contracts() -> None:
    with pytest.raises(TypeError, match="integer"):
        HardAlphaDecodeConfig(threshold=True)
    with pytest.raises(ValueError, match="between"):
        HardAlphaDecodeConfig(threshold=0)
    with pytest.raises(TypeError, match="uint8"):
        hard_alpha_decode_rgba(np.zeros((1, 2, 2, 4), dtype=np.float32))


def test_global_palette_decode_shares_one_palette_without_dithering() -> None:
    rgba = np.array(
        [
            [[[10, 10, 10, 255], [40, 40, 40, 255], [99, 88, 77, 100]]],
            [[[80, 80, 80, 255], [120, 120, 120, 255], [1, 2, 3, 0]]],
        ],
        dtype=np.uint8,
    )
    settings = GlobalPaletteDecodeConfig(alpha_threshold=128, maximum_colors=2)

    first = global_palette_decode_rgba(rgba, config=settings)
    second = global_palette_decode_rgba(rgba, config=settings)

    visible = first[..., 3] > 0
    assert np.array_equal(first, second)
    assert len(np.unique(first[..., :3][visible], axis=0)) <= 2
    assert set(np.unique(first[..., 3]).tolist()) <= {0, 255}
    assert np.all(first[..., :3][~visible] == 0)
    assert rgba[0, 0, 2].tolist() == [99, 88, 77, 100]


def test_global_palette_export_records_method_counts_and_refuses_clobber(
    tmp_path: Path,
) -> None:
    rgba = np.zeros((2, 2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[0, ..., :3] = ((10, 20, 30), (40, 50, 60), (70, 80, 90))
    rgba[1, ..., :3] = ((100, 110, 120), (130, 140, 150), (160, 170, 180))
    source = tmp_path / "source.npy"
    with source.open("wb") as handle:
        np.save(handle, rgba, allow_pickle=False)

    result = export_global_palette_decode(
        source,
        tmp_path / "palette.npy",
        config=GlobalPaletteDecodeConfig(alpha_threshold=192, maximum_colors=2),
    )

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert result.visible_colors_before == 6
    assert 1 <= result.visible_colors_after <= 2
    assert metadata["operation"]["palette_fit_scope"].startswith("all visible")
    assert metadata["operation"]["reference_or_target_palette_used"] is False
    assert metadata["parameters"] == {"alpha_threshold": 192, "maximum_colors": 2}
    assert metadata["runtime"]["pillow_version"]

    with pytest.raises(FileExistsError, match="Refusing"):
        export_global_palette_decode(source, tmp_path / "palette.npy")


def test_global_palette_config_rejects_invalid_values() -> None:
    with pytest.raises(TypeError, match="maximum_colors"):
        GlobalPaletteDecodeConfig(maximum_colors=True)
    with pytest.raises(ValueError, match="between 2 and 256"):
        GlobalPaletteDecodeConfig(maximum_colors=1)
