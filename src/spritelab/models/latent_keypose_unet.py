"""Identity-preserving latent image-to-image U-Net for canonical action poses."""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency boundary
    torch = None
    F = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingLatentKeyposeUNetTorchError(RuntimeError):
    """Raised when the optional key-pose U-Net runtime is unavailable."""


@dataclass(frozen=True, slots=True)
class LatentKeyposeUNetConfig:
    """Geometry for reference latent plus verb to one canonical pose latent."""

    latent_size: int = 64
    latent_channels: int = 8
    base_channels: int = 96
    channel_multipliers: tuple[int, ...] = (1, 2, 3, 4)
    residual_blocks: int = 2
    condition_dim: int = 384
    attention_heads: int = 6
    dropout: float = 0.0
    phase_harmonics: int = 4

    def __post_init__(self) -> None:
        for name in (
            "latent_size",
            "latent_channels",
            "base_channels",
            "residual_blocks",
            "condition_dim",
            "attention_heads",
            "phase_harmonics",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.channel_multipliers, tuple)
            or not self.channel_multipliers
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.channel_multipliers
            )
        ):
            raise ValueError("channel_multipliers must be a non-empty tuple of positive integers")
        downsample_factor = 2 ** (len(self.channel_multipliers) - 1)
        if self.latent_size % downsample_factor:
            raise ValueError("latent_size must be divisible by the U-Net downsample factor")
        deepest_channels = self.base_channels * self.channel_multipliers[-1]
        if deepest_channels % self.attention_heads:
            raise ValueError("deepest U-Net channels must be divisible by attention_heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


if torch is not None and nn is not None and F is not None:
    from .pixeldit import _timestep_embedding

    def _normalization(channels: int) -> nn.GroupNorm:
        groups = min(32, channels)
        while channels % groups:
            groups -= 1
        return nn.GroupNorm(groups, channels, eps=1e-6)

    class _ConditionedResidualBlock(nn.Module):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            condition_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.input_norm = _normalization(input_channels)
            self.input_convolution = nn.Conv2d(
                input_channels, output_channels, kernel_size=3, padding=1
            )
            self.output_norm = _normalization(output_channels)
            self.condition_projection = nn.Sequential(
                nn.SiLU(), nn.Linear(condition_dim, output_channels * 2)
            )
            self.dropout = nn.Dropout(dropout)
            self.output_convolution = nn.Conv2d(
                output_channels, output_channels, kernel_size=3, padding=1
            )
            self.skip = (
                nn.Identity()
                if input_channels == output_channels
                else nn.Conv2d(input_channels, output_channels, kernel_size=1)
            )
            nn.init.zeros_(self.output_convolution.weight)
            nn.init.zeros_(self.output_convolution.bias)

        def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            hidden = self.input_convolution(F.silu(self.input_norm(value)))
            scale, shift = self.condition_projection(condition).chunk(2, dim=-1)
            hidden = self.output_norm(hidden)
            hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
            hidden = self.output_convolution(self.dropout(F.silu(hidden)))
            return self.skip(value) + hidden

    class _SpatialAttention(nn.Module):
        def __init__(self, channels: int, heads: int) -> None:
            super().__init__()
            self.norm = _normalization(channels)
            self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
            self.projection = nn.Linear(channels, channels)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            batch, channels, height, width = value.shape
            tokens = self.norm(value).flatten(2).transpose(1, 2)
            attended = self.attention(tokens, tokens, tokens, need_weights=False)[0]
            attended = (
                self.projection(attended).transpose(1, 2).reshape(batch, channels, height, width)
            )
            return value + attended

    class _EncoderLevel(nn.Module):
        def __init__(
            self,
            input_channels: int,
            output_channels: int,
            condition_dim: int,
            block_count: int,
            dropout: float,
        ) -> None:
            super().__init__()
            blocks = []
            for block_index in range(block_count):
                blocks.append(
                    _ConditionedResidualBlock(
                        input_channels if block_index == 0 else output_channels,
                        output_channels,
                        condition_dim,
                        dropout,
                    )
                )
            self.blocks = nn.ModuleList(blocks)

        def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                value = block(value, condition)
            return value

    class _DecoderLevel(_EncoderLevel):
        pass

    class ReferenceActionLatentKeyposeUNet(nn.Module):
        """Predict a pose residual while retaining reference features at every scale."""

        def __init__(
            self,
            action_count: int,
            config: LatentKeyposeUNetConfig | None = None,
            *,
            action_condition_scale: float = 2.0,
            action_token_count: int = 4,
        ) -> None:
            super().__init__()
            if (
                isinstance(action_count, bool)
                or not isinstance(action_count, int)
                or action_count <= 0
            ):
                raise ValueError("action_count must be a positive integer")
            self.config = config or LatentKeyposeUNetConfig()
            if not math.isfinite(action_condition_scale) or action_condition_scale <= 0:
                raise ValueError("action_condition_scale must be finite and positive")
            if (
                isinstance(action_token_count, bool)
                or not isinstance(action_token_count, int)
                or action_token_count <= 0
            ):
                raise ValueError("action_token_count must be a positive integer")
            self.action_condition_scale = action_condition_scale
            self.action_token_count = action_token_count
            self.action_count = action_count
            widths = tuple(
                self.config.base_channels * multiplier
                for multiplier in self.config.channel_multipliers
            )
            self.action_embedding = nn.Embedding(
                action_count * action_token_count, self.config.condition_dim
            )
            self.action_norm = nn.LayerNorm(self.config.condition_dim)
            self.timestep_mlp = nn.Sequential(
                nn.Linear(self.config.condition_dim, self.config.condition_dim * 4),
                nn.SiLU(),
                nn.Linear(self.config.condition_dim * 4, self.config.condition_dim),
            )
            phase_width = 1 + 2 * self.config.phase_harmonics
            self.phase_mlp = nn.Sequential(
                nn.Linear(phase_width, self.config.condition_dim * 2),
                nn.SiLU(),
                nn.Linear(self.config.condition_dim * 2, self.config.condition_dim),
            )
            self.input_convolution = nn.Conv2d(
                self.config.latent_channels * 2, widths[0], kernel_size=3, padding=1
            )
            encoders = []
            downsamples = []
            current = widths[0]
            for level, width in enumerate(widths):
                encoders.append(
                    _EncoderLevel(
                        current,
                        width,
                        self.config.condition_dim,
                        self.config.residual_blocks,
                        self.config.dropout,
                    )
                )
                current = width
                if level + 1 < len(widths):
                    downsamples.append(
                        nn.Conv2d(width, widths[level + 1], kernel_size=4, stride=2, padding=1)
                    )
                    current = widths[level + 1]
            self.encoders = nn.ModuleList(encoders)
            self.downsamples = nn.ModuleList(downsamples)
            self.middle = nn.ModuleList(
                (
                    _ConditionedResidualBlock(
                        widths[-1],
                        widths[-1],
                        self.config.condition_dim,
                        self.config.dropout,
                    ),
                    _SpatialAttention(widths[-1], self.config.attention_heads),
                    _ConditionedResidualBlock(
                        widths[-1],
                        widths[-1],
                        self.config.condition_dim,
                        self.config.dropout,
                    ),
                )
            )
            self.upsamples = nn.ModuleList()
            self.decoders = nn.ModuleList()
            current = widths[-1]
            for level in range(len(widths) - 1, -1, -1):
                width = widths[level]
                if level < len(widths) - 1:
                    self.upsamples.append(nn.Conv2d(current, width, kernel_size=3, padding=1))
                    current = width
                self.decoders.append(
                    _DecoderLevel(
                        current + width,
                        width,
                        self.config.condition_dim,
                        self.config.residual_blocks,
                        self.config.dropout,
                    )
                )
                current = width
            self.output_norm = _normalization(widths[0])
            self.output_convolution = nn.Conv2d(
                widths[0], self.config.latent_channels, kernel_size=3, padding=1
            )
            nn.init.zeros_(self.output_convolution.weight)
            nn.init.zeros_(self.output_convolution.bias)

        def _condition(
            self,
            timesteps: torch.Tensor,
            action_indices: torch.Tensor,
            frame_phase: torch.Tensor,
        ) -> torch.Tensor:
            offsets = torch.arange(
                self.action_token_count, device=action_indices.device, dtype=action_indices.dtype
            )
            token_indices = action_indices[:, None] * self.action_token_count + offsets[None]
            action = self.action_norm(self.action_embedding(token_indices)).mean(dim=1)
            timestep = self.timestep_mlp(
                _timestep_embedding(timesteps, self.config.condition_dim).to(action.dtype)
            )
            phase = frame_phase[:, 0].to(dtype=torch.float32)
            frequencies = torch.arange(
                1,
                self.config.phase_harmonics + 1,
                device=phase.device,
                dtype=torch.float32,
            )
            angles = 2 * math.pi * phase.unsqueeze(-1) * frequencies
            phase_features = torch.cat(
                (phase.unsqueeze(-1), torch.sin(angles), torch.cos(angles)), dim=-1
            )
            return (
                self.action_condition_scale * action
                + timestep
                + self.phase_mlp(phase_features).to(action.dtype)
            )

        def forward(
            self,
            video: torch.Tensor,
            reference: torch.Tensor,
            timesteps: torch.Tensor,
            action_indices: torch.Tensor,
            *,
            frame_phase: torch.Tensor,
        ) -> torch.Tensor:
            batch = reference.shape[0]
            expected_reference = (
                batch,
                self.config.latent_channels,
                self.config.latent_size,
                self.config.latent_size,
            )
            expected_video = (batch, 1, *expected_reference[1:])
            if tuple(reference.shape) != expected_reference:
                raise ValueError(f"reference must have shape {expected_reference!r}")
            if tuple(video.shape) != expected_video:
                raise ValueError(f"video must have shape {expected_video!r}")
            if tuple(timesteps.shape) != (batch,):
                raise ValueError("timesteps must have shape [B]")
            if tuple(action_indices.shape) != (batch,):
                raise ValueError("action_indices must have shape [B]")
            if tuple(frame_phase.shape) != (batch, 1):
                raise ValueError("frame_phase must have shape [B,1]")
            condition = self._condition(timesteps, action_indices, frame_phase)
            value = self.input_convolution(torch.cat((reference, video[:, 0]), dim=1))
            skips = []
            for level, encoder in enumerate(self.encoders):
                value = encoder(value, condition)
                skips.append(value)
                if level < len(self.downsamples):
                    value = self.downsamples[level](value)
            value = self.middle[0](value, condition)
            value = self.middle[1](value)
            value = self.middle[2](value, condition)
            upsample_index = 0
            for decoder_index, level in enumerate(range(len(skips) - 1, -1, -1)):
                if level < len(skips) - 1:
                    value = F.interpolate(value, scale_factor=2, mode="nearest")
                    value = self.upsamples[upsample_index](value)
                    upsample_index += 1
                value = self.decoders[decoder_index](
                    torch.cat((value, skips[level]), dim=1), condition
                )
            output = self.output_convolution(F.silu(self.output_norm(value)))
            return output.unsqueeze(1)


else:

    class ReferenceActionLatentKeyposeUNet:
        """Import-safe placeholder when PyTorch is unavailable."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise MissingLatentKeyposeUNetTorchError(
                "ReferenceActionLatentKeyposeUNet requires PyTorch"
            ) from _TORCH_IMPORT_ERROR
