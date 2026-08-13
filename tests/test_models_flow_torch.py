from __future__ import annotations

import pytest

from spritelab.models import (
    endpoint_sample_velocity_model,
    euler_sample_velocity_model,
    predict_clean_from_velocity,
    rectified_flow_mse,
    sample_rectified_flow_batch,
)

torch = pytest.importorskip("torch")


def _clip(value: float, *, batch: int = 2) -> torch.Tensor:
    return torch.full((batch, 2, 4, 4, 4), value, dtype=torch.float32)


def test_rectified_flow_path_and_clean_reconstruction_are_exact() -> None:
    clean = _clip(-0.25)
    noise = _clip(0.75)
    timesteps = torch.tensor([0.0, 0.75])

    batch = sample_rectified_flow_batch(clean, noise=noise, timesteps=timesteps)

    assert torch.equal(batch.noisy[0], clean[0])
    assert torch.allclose(batch.noisy[1], _clip(0.5)[1])
    assert torch.equal(batch.target_velocity, noise - clean)
    reconstructed = predict_clean_from_velocity(
        batch.noisy,
        batch.target_velocity,
        batch.timesteps,
    )
    assert torch.allclose(reconstructed, clean)
    assert rectified_flow_mse(batch.target_velocity, batch).item() == 0


def test_foreground_weight_uses_clean_alpha_without_changing_exact_zero() -> None:
    clean = _clip(-1.0, batch=1)
    clean[:, :, 3] = 1.0
    batch = sample_rectified_flow_batch(
        clean,
        noise=torch.zeros_like(clean),
        timesteps=torch.tensor([0.5]),
    )
    predicted = batch.target_velocity + 1

    ordinary = rectified_flow_mse(predicted, batch)
    weighted = rectified_flow_mse(predicted, batch, foreground_weight=3)

    assert ordinary.item() == pytest.approx(1.0)
    assert weighted.item() > ordinary.item()
    assert (
        rectified_flow_mse(
            batch.target_velocity,
            batch,
            foreground_weight=3,
        ).item()
        == 0
    )


def test_alpha_channel_weight_scales_only_alpha_residual() -> None:
    clean = _clip(-1.0, batch=1)
    batch = sample_rectified_flow_batch(
        clean,
        noise=torch.zeros_like(clean),
        timesteps=torch.tensor([0.5]),
    )
    predicted = batch.target_velocity + 1

    weighted = rectified_flow_mse(
        predicted,
        batch,
        alpha_channel_weight=4,
    )

    assert weighted.item() == pytest.approx((3 + 4) / 4)
    assert (
        rectified_flow_mse(
            batch.target_velocity,
            batch,
            alpha_channel_weight=4,
        ).item()
        == 0
    )


def test_backward_euler_sampler_recovers_clean_for_constant_exact_velocity() -> None:
    clean = _clip(-0.5, batch=1)
    noise = _clip(0.5, batch=1)
    velocity = noise - clean

    class ConstantVelocity:
        def __call__(
            self,
            state: torch.Tensor,
            timesteps: torch.Tensor,
            conditioning: torch.Tensor | None,
            *,
            conditioning_mask: torch.Tensor | None,
            frame_phase: torch.Tensor | None,
        ) -> torch.Tensor:
            del state, timesteps, conditioning, conditioning_mask, frame_phase
            return velocity

    sampled = euler_sample_velocity_model(ConstantVelocity(), noise, steps=8)

    assert torch.allclose(sampled, clean, atol=1e-6)


def test_endpoint_sampler_is_exactly_one_backward_euler_step() -> None:
    noise = _clip(0.5, batch=1)

    class TimeDependentVelocity:
        def __call__(
            self,
            state: torch.Tensor,
            timesteps: torch.Tensor,
            conditioning: torch.Tensor | None,
            *,
            conditioning_mask: torch.Tensor | None,
            frame_phase: torch.Tensor | None,
        ) -> torch.Tensor:
            del conditioning, conditioning_mask, frame_phase
            assert torch.equal(timesteps, torch.ones_like(timesteps))
            return state * 0.25

    model = TimeDependentVelocity()
    endpoint = endpoint_sample_velocity_model(model, noise)
    euler = euler_sample_velocity_model(model, noise, steps=1)

    assert torch.equal(endpoint, euler)


def test_flow_contract_rejects_bad_values_and_shapes() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        sample_rectified_flow_batch(_clip(2.0))
    with pytest.raises(ValueError, match=r"\[B, T, 4, H, W\]"):
        sample_rectified_flow_batch(torch.zeros(1, 4, 4, 4))
    with pytest.raises(ValueError, match="shape"):
        sample_rectified_flow_batch(
            _clip(0),
            timesteps=torch.tensor([0.5]),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sample_rectified_flow_batch(
            _clip(0),
            timesteps=torch.tensor([0.5, 1.5]),
        )
    with pytest.raises(ValueError, match="positive"):
        euler_sample_velocity_model(object(), _clip(0), steps=0)
    batch = sample_rectified_flow_batch(
        _clip(0),
        noise=_clip(0),
        timesteps=torch.tensor([0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="non-negative"):
        rectified_flow_mse(batch.target_velocity, batch, foreground_weight=-1)
    with pytest.raises(ValueError, match="non-negative"):
        rectified_flow_mse(batch.target_velocity, batch, alpha_channel_weight=-1)
