from __future__ import annotations

import pytest

from spritelab.models.anchored_latent_motion_dit import (
    AnchoredActionConditionedLatentMotionDiT,
    apply_latent_anchors,
    masked_velocity_mse,
    validate_anchor_shapes,
)
from spritelab.models.latent_motion_dit import LatentMotionDiTConfig


def _config() -> LatentMotionDiTConfig:
    return LatentMotionDiTConfig(
        latent_size=8,
        num_frames=8,
        latent_channels=4,
        patch_size=2,
        model_dim=16,
        depth=1,
        num_heads=4,
        condition_dim=16,
    )


def test_validate_anchor_shapes_requires_matching_video_and_mask() -> None:
    assert validate_anchor_shapes((2, 8, 4, 8, 8), (2, 8, 4, 8, 8), (2, 8)) == (
        2,
        8,
    )

    with pytest.raises(ValueError, match="match video"):
        validate_anchor_shapes((2, 8, 4, 8, 8), (2, 7, 4, 8, 8), (2, 8))
    with pytest.raises(ValueError, match=r"\[B,T\]"):
        validate_anchor_shapes((2, 8, 4, 8, 8), (2, 8, 4, 8, 8), (2, 7))


def test_apply_latent_anchors_clamps_only_start_middle_end() -> None:
    torch = pytest.importorskip("torch")
    video = torch.randn((2, 8, 4, 3, 3))
    anchors = torch.randn_like(video)
    mask = torch.zeros((2, 8), dtype=torch.bool)
    mask[:, (0, 4, 7)] = True

    output = apply_latent_anchors(video, anchors, mask)

    assert torch.equal(output[:, 0], anchors[:, 0])
    assert torch.equal(output[:, 4], anchors[:, 4])
    assert torch.equal(output[:, 7], anchors[:, 7])
    assert torch.equal(output[:, 1:4], video[:, 1:4])
    assert torch.equal(output[:, 5:7], video[:, 5:7])


def test_masked_velocity_mse_ignores_anchor_errors() -> None:
    torch = pytest.importorskip("torch")
    predicted = torch.zeros((1, 8, 1, 1, 1), requires_grad=True)
    target = torch.ones_like(predicted)
    mask = torch.zeros((1, 8), dtype=torch.bool)
    mask[:, (0, 4, 7)] = True
    target = target.clone()
    target[:, mask[0]] = 100

    loss = masked_velocity_mse(predicted, target, mask)

    assert loss.item() == pytest.approx(1)
    loss.backward()
    assert predicted.grad is not None
    assert torch.all(predicted.grad[:, mask[0]] == 0)
    assert torch.all(predicted.grad[:, ~mask[0]] != 0)


def test_anchored_model_accepts_action_and_anchor_planes() -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    model = AnchoredActionConditionedLatentMotionDiT(config, 6)
    video = torch.randn((2, 8, 4, 8, 8))
    reference = torch.randn((2, 4, 8, 8))
    anchors = torch.zeros_like(video)
    anchors[:, 4] = torch.randn_like(anchors[:, 4])
    mask = torch.zeros((2, 8), dtype=torch.bool)
    mask[:, (0, 4, 7)] = True
    phases = torch.arange(8, dtype=torch.float32).div(8).expand(2, -1)

    output = model(
        video,
        reference,
        torch.ones((2,)),
        torch.tensor((0, 1)),
        frame_phase=phases,
        anchor_residuals=anchors,
        anchor_mask=mask,
    )

    assert output.shape == video.shape
    assert torch.isfinite(output).all()


def test_anchored_model_rejects_nonboolean_or_complete_anchor_mask() -> None:
    torch = pytest.importorskip("torch")
    config = _config()
    model = AnchoredActionConditionedLatentMotionDiT(config, 6)
    video = torch.zeros((1, 8, 4, 8, 8))
    reference = torch.zeros((1, 4, 8, 8))
    phases = torch.arange(8, dtype=torch.float32).div(8).unsqueeze(0)
    kwargs = {
        "video": video,
        "reference": reference,
        "timesteps": torch.ones((1,)),
        "action_indices": torch.zeros((1,), dtype=torch.long),
        "frame_phase": phases,
        "anchor_residuals": video,
    }

    with pytest.raises(ValueError, match="boolean"):
        model(anchor_mask=torch.zeros((1, 8)), **kwargs)
    with pytest.raises(ValueError, match="predicted frame"):
        model(anchor_mask=torch.ones((1, 8), dtype=torch.bool), **kwargs)
