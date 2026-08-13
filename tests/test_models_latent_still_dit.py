from __future__ import annotations

import pytest

from spritelab.models.latent_still_dit import (
    LatentStillDiT,
    LatentStillDiTConfig,
    validate_latent_still_shapes,
)

torch = pytest.importorskip("torch")


def _config() -> LatentStillDiTConfig:
    return LatentStillDiTConfig(
        latent_size=16,
        latent_channels=4,
        patch_size=2,
        model_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2,
        condition_dim=16,
        window_size=2,
        global_attention_every=2,
    )


def test_latent_still_dit_preserves_shape_and_zero_initial_output() -> None:
    config = _config()
    model = LatentStillDiT(config)
    latent = torch.randn(2, 4, 16, 16)
    timesteps = torch.tensor([0.25, 0.75])
    context = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])

    output = model(latent, timesteps, context, context_mask=mask)

    assert output.shape == latent.shape
    assert torch.count_nonzero(output).item() == 0
    output.sum().backward()
    assert model.final_projection.weight.grad is not None


def test_latent_still_dit_accepts_classifier_free_null_context() -> None:
    model = LatentStillDiT(_config())
    latent = torch.randn(1, 4, 16, 16)
    output = model(latent, torch.tensor([1.0]))
    assert output.shape == latent.shape


def test_latent_still_contract_rejects_bad_geometry_and_masks() -> None:
    config = _config()
    with pytest.raises(ValueError, match="latent must have shape"):
        validate_latent_still_shapes((2, 4, 8, 8), (2,), (2, 5, 16), config)
    with pytest.raises(ValueError, match="context"):
        validate_latent_still_shapes((2, 4, 16, 16), (2,), (2, 5, 8), config)
    model = LatentStillDiT(config)
    with pytest.raises(ValueError, match="context_mask"):
        model(
            torch.randn(2, 4, 16, 16),
            torch.ones(2),
            torch.randn(2, 5, 16),
            context_mask=torch.ones(2, 4, dtype=torch.bool),
        )


def test_latent_still_config_requires_window_compatible_grid() -> None:
    with pytest.raises(ValueError, match="window_size"):
        LatentStillDiTConfig(latent_size=24, patch_size=2, window_size=8)
