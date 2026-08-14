from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spritelab.models import (  # noqa: E402
    LatentMotionDiTConfig,
    ReferenceConditionedLatentMotionDiT,
    validate_latent_motion_shapes,
)


def _config() -> LatentMotionDiTConfig:
    return LatentMotionDiTConfig(
        latent_size=8,
        num_frames=2,
        latent_channels=4,
        patch_size=2,
        model_dim=16,
        depth=1,
        num_heads=4,
        condition_dim=12,
        phase_harmonics=2,
    )


def test_reference_conditioned_latent_motion_dit_shape_and_reference_path() -> None:
    config = _config()
    model = ReferenceConditionedLatentMotionDiT(config)
    video = torch.randn(2, 2, 4, 8, 8)
    reference_a = torch.randn(2, 4, 8, 8)
    reference_b = reference_a + 1
    timestep = torch.tensor([0.2, 0.8])
    condition = torch.randn(2, 3, 12)
    phase = torch.tensor([[0.0, 0.5], [0.0, 0.5]])

    initial = model(video, reference_a, timestep, condition, frame_phase=phase)
    assert initial.shape == video.shape
    assert torch.count_nonzero(initial) == 0

    torch.nn.init.normal_(model.final_layer.projection.weight)
    output_a = model(video, reference_a, timestep, condition, frame_phase=phase)
    output_b = model(video, reference_b, timestep, condition, frame_phase=phase)
    assert output_a.shape == video.shape
    assert not torch.equal(output_a, output_b)


def test_latent_motion_shape_contract_rejects_wrong_reference_and_phase() -> None:
    config = _config()
    assert validate_latent_motion_shapes((1, 2, 4, 8, 8), (1, 4, 8, 8), config) == 1
    with pytest.raises(ValueError, match="reference"):
        validate_latent_motion_shapes((1, 2, 4, 8, 8), (1, 4, 7, 8), config)

    model = ReferenceConditionedLatentMotionDiT(config)
    with pytest.raises(ValueError, match="frame_phase"):
        model(
            torch.randn(1, 2, 4, 8, 8),
            torch.randn(1, 4, 8, 8),
            torch.tensor([0.5]),
            frame_phase=torch.zeros(1, 3),
        )


def test_latent_motion_config_rejects_invalid_geometry() -> None:
    with pytest.raises(ValueError, match="divisible"):
        LatentMotionDiTConfig(latent_size=7, patch_size=2)
    with pytest.raises(ValueError, match="num_heads"):
        LatentMotionDiTConfig(model_dim=15, num_heads=4)
