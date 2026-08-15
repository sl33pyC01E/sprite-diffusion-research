from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scripts.render_mugen_anchored_motion_checkpoint_v1 import (  # noqa: E402
    _comparison_sheet,
    _config_from_checkpoint,
    _keypose_config_from_checkpoint,
    _rgba_uint8,
)
from spritelab.anchored_motion_train import AnchoredMotionTrainingConfig  # noqa: E402
from spritelab.latent_keypose_train import LatentKeyposeTrainingConfig  # noqa: E402


def test_config_from_checkpoint_restores_nested_model() -> None:
    expected = AnchoredMotionTrainingConfig(device="cpu", precision="float32")

    actual = _config_from_checkpoint(asdict(expected))

    assert actual == expected
    assert actual.anchor_frame_indices == (0, 4, 7)
    assert actual.model.num_frames == 8


def test_config_from_checkpoint_rejects_missing_model() -> None:
    with pytest.raises(ValueError, match="model config"):
        _config_from_checkpoint({"steps": 10})


def test_keypose_config_from_checkpoint_restores_direct_contract() -> None:
    expected = LatentKeyposeTrainingConfig(
        device="cpu", precision="float32", prediction_mode="direct_residual"
    )

    actual = _keypose_config_from_checkpoint(asdict(expected))

    assert actual == expected
    assert actual.keypose_frame_index == 4


def test_rgba_conversion_and_comparison_sheet_preserve_slots() -> None:
    value = torch.zeros((1, 8, 4, 4, 4), dtype=torch.float32)
    value[0, 4, :, 1:3, 1:3] = torch.tensor([0.25, 0.5, 0.75, 1.0]).view(4, 1, 1)

    rgba = _rgba_uint8(value)[0]
    sheet = _comparison_sheet(rgba, rgba)

    assert rgba.dtype == np.uint8
    assert rgba.shape == (8, 4, 4, 4)
    assert rgba[4, 1, 1].tolist() == [64, 128, 191, 255]
    assert sheet.size == (92 + 8 * 256, 28 + 2 * 256)
