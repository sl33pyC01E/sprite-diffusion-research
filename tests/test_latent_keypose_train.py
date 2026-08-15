from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spritelab.latent_keypose_train import (
    LatentKeyposeTrainingConfig,
    _build_keypose_model,
    _keypose_batch,
    _keypose_bundle_metrics,
    _keypose_prediction_contract,
    _pixel_action_bundle_loss,
    build_keypose_action_bundles,
)
from spritelab.latent_motion_train import LatentMotionTrainingRow
from spritelab.models.latent_keypose_unet import LatentKeyposeUNetConfig
from spritelab.models.latent_motion_dit import LatentMotionDiTConfig


def _row(identity: str, verb: str, action_index: int) -> LatentMotionTrainingRow:
    return LatentMotionTrainingRow(
        sequence_id=f"{identity}-{verb}",
        identity_id=identity,
        verb=verb,
        action_index=action_index,
        split="train",
        duration_ms=(125.0,) * 8,
        loop_mode="loop",
    )


def test_keypose_config_fixes_one_model_frame_at_source_frame_four() -> None:
    config = LatentKeyposeTrainingConfig()

    assert config.keypose_frame_index == 4
    assert config.model.num_frames == 1
    assert config.action_token_count == 4
    assert config.action_condition_scale == pytest.approx(2)
    assert config.prediction_mode == "endpoint_flow"

    with pytest.raises(ValueError, match="exactly one frame"):
        LatentKeyposeTrainingConfig(model=LatentMotionDiTConfig(num_frames=2, patch_size=4))


def test_direct_keypose_contract_removes_unnecessary_noise_prediction() -> None:
    torch = pytest.importorskip("torch")
    clean = torch.tensor((1.0, 2.0, 3.0))
    noise = torch.tensor((4.0, 5.0, 6.0))

    endpoint_input, endpoint_target = _keypose_prediction_contract(
        torch,
        clean_residual=clean,
        noise=noise,
        prediction_mode="endpoint_flow",
    )
    direct_input, direct_target = _keypose_prediction_contract(
        torch,
        clean_residual=clean,
        noise=noise,
        prediction_mode="direct_residual",
    )

    assert torch.equal(endpoint_input, noise)
    assert torch.equal(endpoint_target, noise - clean)
    assert torch.equal(direct_input, torch.zeros_like(clean))
    assert torch.equal(direct_target, -clean)
    assert torch.equal(direct_input - direct_target, clean)


def test_keypose_action_bundles_require_all_six_actions() -> None:
    verbs = ("attack_a", "attack_b", "block", "idle", "jump", "walk")
    rows = tuple(
        _row(identity, verb, action_index)
        for identity in ("a", "b")
        for action_index, verb in enumerate(verbs)
    ) + (_row("partial", "idle", 3),)
    corpus = SimpleNamespace(rows=rows, action_vocabulary=verbs)

    bundles = build_keypose_action_bundles(corpus, tuple(range(len(rows))))

    assert bundles == (tuple(range(6)), tuple(range(6, 12)))


def test_keypose_batch_selects_only_fixed_middle_frame() -> None:
    torch = pytest.importorskip("torch")
    targets = np.zeros((6, 8, 8, 2, 2), dtype=np.float16)
    rgba = np.zeros((6, 8, 4, 4, 4), dtype=np.uint8)
    for frame in range(8):
        targets[:, frame] = frame
        rgba[:, frame] = frame
    references = np.zeros((6, 8, 2, 2), dtype=np.float16)
    rows = tuple(_row("a", f"verb-{index}", index) for index in range(6))
    corpus = SimpleNamespace(
        target_latents=targets,
        reference_latents=references,
        target_rgba=rgba,
        rows=rows,
    )
    mean = torch.zeros((1, 8, 1, 1))
    std = torch.ones((1, 8, 1, 1))

    target, reference, pixels, phase, actions = _keypose_batch(
        torch,
        corpus,
        tuple(range(6)),
        frame_index=4,
        device=torch.device("cpu"),
        mean=mean,
        std=std,
    )

    assert target.shape == (6, 8, 2, 2)
    assert torch.all(target == 4)
    assert reference.shape == target.shape
    assert pixels.shape == (6, 4, 4, 4)
    assert torch.allclose(pixels, torch.full_like(pixels, 4 / 255))
    assert torch.allclose(phase, torch.full_like(phase, 0.5))
    assert actions.tolist() == list(range(6))


def test_keypose_metrics_expose_exact_and_collapsed_action_outputs() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((6, 1, 4, 3, 3))
    for index in range(6):
        target[index, 0, 0, index // 3, index % 3] = 1
        target[index, 0, 3, index // 3, index % 3] = 1
    exact = _keypose_bundle_metrics(torch, predicted_rgba=target, target_rgba=target)
    collapsed = _keypose_bundle_metrics(
        torch, predicted_rgba=target[:1].expand_as(target), target_rgba=target
    )

    assert exact["premultiplied_rgba_mae"].item() == pytest.approx(0)
    assert exact["correct_target_preference_rate"].item() == pytest.approx(1)
    assert exact["generated_action_separation"].item() == pytest.approx(
        exact["target_action_separation"].item()
    )
    assert collapsed["correct_target_preference_rate"].item() < 1
    assert collapsed["generated_action_separation"].item() == pytest.approx(0)


def test_keypose_action_loss_prefers_exact_bundle_and_backpropagates() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((6, 1, 4, 3, 3))
    for index in range(6):
        target[index, 0, 0, index // 3, index % 3] = 1
        target[index, 0, 3, index // 3, index % 3] = 1
    exact = _pixel_action_bundle_loss(torch, predicted_rgba=target, target_rgba=target)
    collapsed_prediction = target[:1].expand_as(target).clone().requires_grad_(True)
    collapsed = _pixel_action_bundle_loss(
        torch, predicted_rgba=collapsed_prediction, target_rgba=target
    )

    assert exact.item() == pytest.approx(0)
    assert collapsed.item() > exact.item()
    collapsed.backward()
    assert collapsed_prediction.grad is not None
    assert torch.isfinite(collapsed_prediction.grad).all()


def test_keypose_model_accepts_reference_verb_and_one_latent_frame() -> None:
    torch = pytest.importorskip("torch")
    config = LatentKeyposeTrainingConfig(
        device="cpu",
        precision="float32",
        model=LatentMotionDiTConfig(
            latent_size=8,
            num_frames=1,
            latent_channels=4,
            patch_size=2,
            model_dim=16,
            depth=1,
            num_heads=4,
            condition_dim=16,
        ),
    )
    model = _build_keypose_model(config, 6)
    video = torch.randn((6, 1, 4, 8, 8))
    reference = torch.randn((6, 4, 8, 8))
    times = torch.ones((6,))
    actions = torch.arange(6)
    phases = torch.full((6, 1), 0.5)

    output = model(video, reference, times, actions, frame_phase=phases)

    assert output.shape == video.shape
    assert torch.isfinite(output).all()


def test_identity_unet_preserves_multiscale_reference_path_and_action_gradient() -> None:
    torch = pytest.importorskip("torch")
    config = LatentKeyposeTrainingConfig(
        device="cpu",
        precision="float32",
        model_architecture="identity_unet",
        model=LatentMotionDiTConfig(
            latent_size=8,
            num_frames=1,
            latent_channels=4,
            patch_size=2,
            model_dim=16,
            depth=1,
            num_heads=4,
            condition_dim=16,
        ),
        unet=LatentKeyposeUNetConfig(
            latent_size=8,
            latent_channels=4,
            base_channels=8,
            channel_multipliers=(1, 2),
            residual_blocks=1,
            condition_dim=16,
            attention_heads=4,
        ),
    )
    model = _build_keypose_model(config, 6)
    video = torch.zeros((6, 1, 4, 8, 8))
    reference = torch.randn((6, 4, 8, 8))
    times = torch.ones((6,))
    actions = torch.arange(6)
    phases = torch.full((6, 1), 0.5)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    target = actions[:, None, None, None, None].expand_as(video).float().div(5)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = model(video, reference, times, actions, frame_phase=phases)
        torch.nn.functional.mse_loss(output, target).backward()
        if step < 2:
            optimizer.step()

    assert output.shape == video.shape
    assert torch.isfinite(output).all()
    assert model.output_convolution.weight.grad is not None
    assert torch.isfinite(model.output_convolution.weight.grad).all()
    assert model.action_embedding.weight.grad is not None
    assert torch.isfinite(model.action_embedding.weight.grad).all()
    assert model.action_embedding.weight.grad.abs().sum() > 0
