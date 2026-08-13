from __future__ import annotations

from dataclasses import replace

import pytest

from spritelab.models import ConditioningSchema, FactorizedSpriteDiT, PixelDiTConfig

torch = pytest.importorskip("torch")


def _small_config() -> PixelDiTConfig:
    schema = replace(ConditioningSchema(), phase_bins=2)
    return PixelDiTConfig(
        height=8,
        width=8,
        num_frames=2,
        patch_size=2,
        model_dim=32,
        depth=2,
        num_heads=4,
        condition_dim=16,
        phase_harmonics=2,
        conditioning=schema,
    )


def test_factorized_pixeldit_preserves_video_shape() -> None:
    config = _small_config()
    model = FactorizedSpriteDiT(config)
    video = torch.randn(2, 2, 4, 8, 8)
    timesteps = torch.tensor([0.1, 0.9])
    conditioning = torch.randn(2, 6, 16)
    conditioning_mask = torch.ones(2, 6, dtype=torch.bool)
    frame_phase = torch.tensor([[0.0, 0.5], [0.25, 0.75]])

    output = model(
        video,
        timesteps,
        conditioning,
        conditioning_mask=conditioning_mask,
        frame_phase=frame_phase,
    )

    assert output.shape == video.shape
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0  # zero-initialized DiT prediction head


def test_factorized_pixeldit_supports_classifier_free_null_context() -> None:
    config = _small_config()
    model = FactorizedSpriteDiT(config)

    output = model(torch.randn(1, 2, 4, 8, 8), torch.tensor([0.5]))

    assert output.shape == (1, 2, 4, 8, 8)


def test_factorized_pixeldit_rejects_bad_context_mask() -> None:
    config = _small_config()
    model = FactorizedSpriteDiT(config)

    with pytest.raises(ValueError, match="must match"):
        model(
            torch.randn(1, 2, 4, 8, 8),
            torch.tensor([0.5]),
            torch.randn(1, 6, 16),
            conditioning_mask=torch.ones(1, 5, dtype=torch.bool),
        )
