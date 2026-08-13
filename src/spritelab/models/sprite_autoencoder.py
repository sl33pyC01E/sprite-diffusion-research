"""Sprite-specific continuous RGBA autoencoder for latent animation models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - torch-free environment
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingSpriteAutoencoderTorchError(RuntimeError):
    """Raised when the autoencoder is constructed without PyTorch."""


@dataclass(frozen=True, slots=True)
class SpriteAutoencoderConfig:
    """Architecture contract for an exact-size RGBA sprite autoencoder."""

    image_size: int = 128
    base_channels: int = 64
    latent_channels: int = 16
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    residual_blocks: int = 2

    def __post_init__(self) -> None:
        for name in ("image_size", "base_channels", "latent_channels", "residual_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.channel_multipliers) < 2:
            raise ValueError("channel_multipliers must contain at least two stages")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.channel_multipliers
        ):
            raise ValueError("channel_multipliers must be positive integers")
        if self.image_size % self.downsample_factor:
            raise ValueError("image_size must be divisible by the downsample factor")

    @property
    def downsample_factor(self) -> int:
        return 2 ** (len(self.channel_multipliers) - 1)

    @property
    def latent_size(self) -> int:
        return self.image_size // self.downsample_factor


@dataclass(frozen=True, slots=True)
class SpriteReconstructionLossConfig:
    """Transparent-sprite loss weights; all terms remain separately reported."""

    premultiplied_rgba_weight: float = 1.0
    alpha_bce_weight: float = 1.0
    visible_rgb_weight: float = 0.5
    edge_weight: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "premultiplied_rgba_weight",
            "alpha_bce_weight",
            "visible_rgb_weight",
            "edge_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not any(getattr(self, name) > 0 for name in self.__dataclass_fields__):
            raise ValueError("at least one reconstruction loss weight must be positive")


@dataclass(frozen=True, slots=True)
class SpriteReconstructionLoss:
    total: Any
    premultiplied_rgba_l1: Any
    alpha_bce: Any
    visible_rgb_l1: Any
    edge_l1: Any


if torch is not None and nn is not None and F is not None:

    def _normalization(channels: int) -> nn.GroupNorm:
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    class _ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.network = nn.Sequential(
                _normalization(channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                _normalization(channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value + self.network(value)

    class _Upsample(nn.Module):
        def __init__(self, source_channels: int, target_channels: int) -> None:
            super().__init__()
            self.convolution = nn.Conv2d(source_channels, target_channels, 3, padding=1)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            value = F.interpolate(value, scale_factor=2, mode="nearest")
            return self.convolution(value)

    class SpriteRGBAAutoencoder(nn.Module):
        """Encode RGBA frames to a spatial latent and reconstruct RGB/alpha logits.

        Nearest-neighbor decoder upsampling avoids learned transposed-convolution
        checkerboards. RGB and alpha remain distinct until loss/decode time so a
        sparse transparent canvas cannot dominate character color learning.
        """

        def __init__(self, config: SpriteAutoencoderConfig) -> None:
            super().__init__()
            self.config = config
            channels = tuple(config.base_channels * value for value in config.channel_multipliers)
            encoder: list[nn.Module] = [nn.Conv2d(4, channels[0], 3, padding=1)]
            for stage, channel_count in enumerate(channels):
                encoder.extend(_ResidualBlock(channel_count) for _ in range(config.residual_blocks))
                if stage + 1 < len(channels):
                    encoder.append(
                        nn.Conv2d(channel_count, channels[stage + 1], 4, stride=2, padding=1)
                    )
            encoder.extend(
                (
                    _normalization(channels[-1]),
                    nn.SiLU(),
                    nn.Conv2d(channels[-1], config.latent_channels, 3, padding=1),
                )
            )
            self.encoder = nn.Sequential(*encoder)

            decoder: list[nn.Module] = [
                nn.Conv2d(config.latent_channels, channels[-1], 3, padding=1)
            ]
            for stage in range(len(channels) - 1, -1, -1):
                channel_count = channels[stage]
                decoder.extend(_ResidualBlock(channel_count) for _ in range(config.residual_blocks))
                if stage > 0:
                    decoder.append(_Upsample(channel_count, channels[stage - 1]))
            decoder.extend(
                (
                    _normalization(channels[0]),
                    nn.SiLU(),
                    nn.Conv2d(channels[0], 4, 3, padding=1),
                )
            )
            self.decoder = nn.Sequential(*decoder)

        def encode(self, rgba_unit: torch.Tensor) -> torch.Tensor:
            _validate_rgba(rgba_unit, self.config.image_size)
            return self.encoder(rgba_unit)

        def decode_logits(self, latent: torch.Tensor) -> torch.Tensor:
            expected = (
                latent.shape[0],
                self.config.latent_channels,
                self.config.latent_size,
                self.config.latent_size,
            )
            if tuple(latent.shape) != expected:
                raise ValueError(
                    f"latent must have shape {expected!r}; got {tuple(latent.shape)!r}"
                )
            return self.decoder(latent)

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.decode_logits(latent))

        def forward(self, rgba_unit: torch.Tensor) -> torch.Tensor:
            return self.decode_logits(self.encode(rgba_unit))

    def sprite_reconstruction_loss(
        logits: torch.Tensor,
        target_rgba_unit: torch.Tensor,
        *,
        config: SpriteReconstructionLossConfig | None = None,
    ) -> SpriteReconstructionLoss:
        """Measure sparse RGBA reconstruction without rewarding invisible RGB."""

        if config is None:
            config = SpriteReconstructionLossConfig()
        _validate_rgba(target_rgba_unit, target_rgba_unit.shape[-1])
        if logits.shape != target_rgba_unit.shape:
            raise ValueError("logits and target_rgba_unit must have the same shape")
        if not bool(torch.isfinite(logits).all()) or not bool(
            torch.isfinite(target_rgba_unit).all()
        ):
            raise ValueError("reconstruction tensors must be finite")
        if bool((target_rgba_unit < 0).any()) or bool((target_rgba_unit > 1).any()):
            raise ValueError("target_rgba_unit must remain in [0, 1]")
        predicted = torch.sigmoid(logits)
        target_rgb = target_rgba_unit[:, :3]
        target_alpha = target_rgba_unit[:, 3:4]
        predicted_rgb = predicted[:, :3]
        predicted_alpha = predicted[:, 3:4]
        target_pm = torch.cat((target_rgb * target_alpha, target_alpha), dim=1)
        predicted_pm = torch.cat((predicted_rgb * predicted_alpha, predicted_alpha), dim=1)
        premultiplied = F.l1_loss(predicted_pm, target_pm)
        alpha_bce = F.binary_cross_entropy_with_logits(logits[:, 3:4], target_alpha)
        visible_denominator = target_alpha.sum().clamp_min(1.0) * 3
        visible_rgb = (torch.abs(predicted_rgb - target_rgb) * target_alpha).sum()
        visible_rgb = visible_rgb / visible_denominator
        edge = _edge_l1(predicted_pm, target_pm)
        total = (
            config.premultiplied_rgba_weight * premultiplied
            + config.alpha_bce_weight * alpha_bce
            + config.visible_rgb_weight * visible_rgb
            + config.edge_weight * edge
        )
        return SpriteReconstructionLoss(total, premultiplied, alpha_bce, visible_rgb, edge)

    def _validate_rgba(value: torch.Tensor, image_size: int) -> None:
        expected = (value.shape[0], 4, image_size, image_size)
        if value.ndim != 4 or tuple(value.shape) != expected:
            raise ValueError(
                f"RGBA tensor must have shape {expected!r}; got {tuple(value.shape)!r}"
            )
        if not value.is_floating_point():
            raise TypeError("RGBA tensor must use floating point")

    def _edge_l1(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        predicted_x = predicted[..., :, 1:] - predicted[..., :, :-1]
        target_x = target[..., :, 1:] - target[..., :, :-1]
        predicted_y = predicted[..., 1:, :] - predicted[..., :-1, :]
        target_y = target[..., 1:, :] - target[..., :-1, :]
        return 0.5 * (F.l1_loss(predicted_x, target_x) + F.l1_loss(predicted_y, target_y))

else:

    class SpriteRGBAAutoencoder:  # pragma: no cover - dependency boundary
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise MissingSpriteAutoencoderTorchError(
                "sprite autoencoding requires a platform-appropriate PyTorch installation"
            ) from _TORCH_IMPORT_ERROR

    def sprite_reconstruction_loss(*args: Any, **kwargs: Any) -> SpriteReconstructionLoss:
        del args, kwargs
        raise MissingSpriteAutoencoderTorchError(
            "sprite autoencoding requires a platform-appropriate PyTorch installation"
        ) from _TORCH_IMPORT_ERROR
