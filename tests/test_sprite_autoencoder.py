from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from spritelab.models.sprite_autoencoder import (  # noqa: E402
    SpriteAutoencoderConfig,
    SpriteReconstructionLossConfig,
    SpriteRGBAAutoencoder,
    sprite_reconstruction_loss,
)


def _tiny_config() -> SpriteAutoencoderConfig:
    return SpriteAutoencoderConfig(
        image_size=16,
        base_channels=8,
        latent_channels=4,
        channel_multipliers=(1, 2),
        residual_blocks=1,
    )


def test_rgba_autoencoder_shapes_and_loss_backpropagate() -> None:
    config = _tiny_config()
    model = SpriteRGBAAutoencoder(config)
    target = torch.zeros((2, 4, 16, 16), dtype=torch.float32)
    target[:, :3, 4:12, 5:11] = torch.tensor([0.9, 0.2, 0.1]).view(1, 3, 1, 1)
    target[:, 3, 4:12, 5:11] = 1

    latent = model.encode(target)
    logits = model.decode_logits(latent)
    decoded = model.decode(latent)
    loss = sprite_reconstruction_loss(logits, target)
    loss.total.backward()

    assert latent.shape == (2, 4, 8, 8)
    assert logits.shape == target.shape
    assert decoded.shape == target.shape
    assert bool(((decoded >= 0) & (decoded <= 1)).all())
    assert all(
        bool(torch.isfinite(value)) and float(value) > 0
        for value in (
            loss.total,
            loss.premultiplied_rgba_l1,
            loss.alpha_bce,
            loss.visible_rgb_l1,
            loss.edge_l1,
        )
    )
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_autoencoder_contract_rejects_bad_geometry_and_loss_inputs() -> None:
    with pytest.raises(ValueError, match="divisible"):
        replace(_tiny_config(), image_size=15)
    with pytest.raises(ValueError, match="at least one"):
        SpriteReconstructionLossConfig(
            premultiplied_rgba_weight=0,
            alpha_bce_weight=0,
            visible_rgb_weight=0,
            edge_weight=0,
        )
    model = SpriteRGBAAutoencoder(_tiny_config())
    with pytest.raises(ValueError, match="RGBA tensor"):
        model(torch.zeros((1, 4, 15, 16)))
    with pytest.raises(ValueError, match="same shape"):
        sprite_reconstruction_loss(
            torch.zeros((1, 4, 16, 16)),
            torch.zeros((2, 4, 16, 16)),
        )
