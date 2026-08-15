from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spritelab.anchored_motion_train import (
    AnchoredMotionTrainingConfig,
    _anchored_batch,
    _trajectory_bundle_metrics,
    sample_anchored_motion_residual,
)
from spritelab.latent_motion_train import LatentMotionTrainingRow


def _row(index: int) -> LatentMotionTrainingRow:
    return LatentMotionTrainingRow(
        sequence_id=f"sequence-{index}",
        identity_id="identity",
        verb=f"verb-{index}",
        action_index=index,
        split="train",
        duration_ms=(125.0,) * 8,
        loop_mode="loop",
    )


def test_anchored_config_fixes_start_middle_start_contract() -> None:
    config = AnchoredMotionTrainingConfig()

    assert config.anchor_frame_indices == (0, 4, 7)
    assert config.canonical_middle_frame_index == 4

    with pytest.raises(ValueError, match=r"\(0,4,7\)"):
        AnchoredMotionTrainingConfig(anchor_frame_indices=(0, 3, 7))


def test_anchored_batch_inserts_reference_endpoints_and_middle_target() -> None:
    torch = pytest.importorskip("torch")
    target = np.zeros((6, 8, 4, 2, 2), dtype=np.float16)
    for frame in range(8):
        target[:, frame] = frame + 1
    reference = np.ones((6, 4, 2, 2), dtype=np.float16)
    rgba = np.zeros((6, 8, 2, 2, 4), dtype=np.uint8)
    phases = np.broadcast_to(np.arange(8, dtype=np.float32) / 8, (6, 8)).copy()
    corpus = SimpleNamespace(
        target_latents=target,
        reference_latents=reference,
        target_rgba=rgba,
        phases=phases,
        rows=tuple(_row(index) for index in range(6)),
    )
    mean = torch.zeros((1, 1, 4, 1, 1))
    std = torch.ones((1, 1, 4, 1, 1))

    clean, normalized_reference, _, _, actions, anchors, mask = _anchored_batch(
        torch,
        corpus,
        tuple(range(6)),
        config=AnchoredMotionTrainingConfig(),
        device=torch.device("cpu"),
        mean=mean,
        std=std,
    )

    assert torch.all(clean[:, 0] == 0)
    assert torch.all(clean[:, 7] == 0)
    assert torch.all(clean[:, 4] == 4)
    assert torch.equal(anchors[:, 4], clean[:, 4])
    assert torch.all(anchors[:, (0, 7)] == 0)
    assert mask[0].tolist() == [True, False, False, False, True, False, False, True]
    assert torch.all(normalized_reference == 1)
    assert actions.tolist() == list(range(6))


def test_trajectory_metrics_exclude_anchors_and_expose_collapse() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((6, 5, 4, 3, 3))
    for index in range(6):
        target[index, :, 0, index // 3, index % 3] = 1
        target[index, :, 3, index // 3, index % 3] = 1
    exact = _trajectory_bundle_metrics(torch, predicted_rgba=target, target_rgba=target)
    collapsed = _trajectory_bundle_metrics(
        torch, predicted_rgba=target[:1].expand_as(target), target_rgba=target
    )

    assert exact["premultiplied_rgba_mae"].item() == pytest.approx(0)
    assert exact["correct_target_preference_rate"].item() == pytest.approx(1)
    assert collapsed["generated_action_separation"].item() == pytest.approx(0)
    assert collapsed["correct_target_preference_rate"].item() < 1


def test_anchored_sampler_clamps_after_model_prediction() -> None:
    torch = pytest.importorskip("torch")
    noise = torch.randn((2, 8, 4, 2, 2))
    anchors = torch.randn_like(noise)
    mask = torch.zeros((2, 8), dtype=torch.bool)
    mask[:, (0, 4, 7)] = True

    class ConstantVelocity:
        def __call__(self, video, *_args, **_kwargs):
            return torch.ones_like(video)

    output = sample_anchored_motion_residual(
        torch,
        ConstantVelocity(),
        noise=noise,
        reference=torch.zeros((2, 4, 2, 2)),
        actions=torch.zeros((2,), dtype=torch.long),
        phases=torch.zeros((2, 8)),
        anchor_residuals=anchors,
        anchor_mask=mask,
    )

    assert torch.equal(output[:, 0], anchors[:, 0])
    assert torch.equal(output[:, 4], anchors[:, 4])
    assert torch.equal(output[:, 7], anchors[:, 7])
    assert torch.allclose(output[:, 1:4], noise[:, 1:4] - 1)
