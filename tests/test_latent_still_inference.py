from __future__ import annotations

from contextlib import nullcontext

import pytest

from spritelab.latent_still_inference import (
    LatentStillInferenceConfig,
    heun_sample_latent_still,
)

torch = pytest.importorskip("torch")


class _ZeroVelocity(torch.nn.Module):
    def forward(
        self,
        latent,
        timesteps,
        context=None,
        *,
        context_mask=None,
    ):
        del timesteps, context, context_mask
        return torch.zeros_like(latent)


def test_heun_zero_velocity_preserves_noise() -> None:
    noise = torch.randn(2, 4, 8, 8)
    context = torch.randn(2, 5, 6)
    mask = torch.ones(2, 5, dtype=torch.bool)
    result = heun_sample_latent_still(
        _ZeroVelocity(),
        noise,
        context,
        mask,
        steps=4,
        guidance_scale=3,
        autocast_context=nullcontext,
    )
    assert torch.equal(result, noise)


def test_inference_config_rejects_invalid_sampler_values() -> None:
    assert LatentStillInferenceConfig().sample_steps == 32
    with pytest.raises(ValueError, match="sample_steps"):
        LatentStillInferenceConfig(sample_steps=0)
    with pytest.raises(ValueError, match="guidance_scale"):
        LatentStillInferenceConfig(guidance_scale=-1)
