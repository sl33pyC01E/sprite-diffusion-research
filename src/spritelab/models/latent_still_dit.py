"""Quality-first windowed DiT for 2x-compressed sprite still latents."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency boundary
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingLatentStillTorchError(RuntimeError):
    """Raised when the latent still model is requested without PyTorch."""


@dataclass(frozen=True, slots=True)
class LatentStillDiTConfig:
    """Fixed geometry and capacity contract for one sprite latent."""

    latent_size: int = 64
    latent_channels: int = 8
    patch_size: int = 2
    model_dim: int = 512
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    condition_dim: int = 768
    window_size: int = 8
    global_attention_every: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "latent_size",
            "latent_channels",
            "patch_size",
            "model_dim",
            "depth",
            "num_heads",
            "condition_dim",
            "window_size",
            "global_attention_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.latent_size % self.patch_size:
            raise ValueError("latent_size must be divisible by patch_size")
        if self.grid_size % self.window_size:
            raise ValueError("patch grid must be divisible by window_size")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must remain in [0,1)")

    @property
    def grid_size(self) -> int:
        return self.latent_size // self.patch_size

    @property
    def token_count(self) -> int:
        return self.grid_size**2

    @property
    def patch_vector_width(self) -> int:
        return self.latent_channels * self.patch_size**2


def validate_latent_still_shapes(
    latent_shape: tuple[int, ...],
    timestep_shape: tuple[int, ...],
    context_shape: tuple[int, ...] | None,
    config: LatentStillDiTConfig,
) -> None:
    """Validate public tensor shapes without constructing a model."""

    if len(latent_shape) != 4:
        raise ValueError("latent must have shape [B,C,H,W]")
    expected = (config.latent_channels, config.latent_size, config.latent_size)
    if latent_shape[0] <= 0 or latent_shape[1:] != expected:
        raise ValueError(f"latent must have shape [B,{expected[0]},{expected[1]},{expected[2]}]")
    if timestep_shape != (latent_shape[0],):
        raise ValueError("timesteps must have shape [B]")
    if context_shape is not None:
        if len(context_shape) != 3:
            raise ValueError("context must have shape [B,L,D]")
        if (
            context_shape[0] != latent_shape[0]
            or context_shape[1] <= 0
            or context_shape[2] != config.condition_dim
        ):
            raise ValueError(
                f"context must have shape [B,L,{config.condition_dim}] with positive L"
            )


if torch is not None and nn is not None:

    def _timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
        half = width // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        if embedding.shape[-1] < width:
            embedding = torch.nn.functional.pad(embedding, (0, width - embedding.shape[-1]))
        return embedding

    def _modulate(
        value: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        return value * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    class _LatentStillDiTBlock(nn.Module):
        def __init__(self, config: LatentStillDiTConfig, *, global_attention: bool) -> None:
            super().__init__()
            width = config.model_dim
            self.config = config
            self.global_attention = global_attention
            self.self_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.context_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.self_attention = nn.MultiheadAttention(
                width, config.num_heads, dropout=config.dropout, batch_first=True
            )
            self.context_attention = nn.MultiheadAttention(
                width, config.num_heads, dropout=config.dropout, batch_first=True
            )
            hidden = int(width * config.mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(width, hidden),
                nn.GELU(approximate="tanh"),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, width),
                nn.Dropout(config.dropout),
            )
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 9))

        def forward(
            self,
            tokens: torch.Tensor,
            global_condition: torch.Tensor,
            context: torch.Tensor,
            context_mask: torch.Tensor,
        ) -> torch.Tensor:
            parameters = self.modulation(global_condition).chunk(9, dim=-1)
            (
                self_shift,
                self_scale,
                self_gate,
                context_shift,
                context_scale,
                context_gate,
                mlp_shift,
                mlp_scale,
                mlp_gate,
            ) = parameters
            self_inputs = _modulate(self.self_norm(tokens), self_shift, self_scale)
            if self.global_attention:
                self_outputs = self.self_attention(
                    self_inputs, self_inputs, self_inputs, need_weights=False
                )[0]
            else:
                windows = self._partition_windows(self_inputs)
                windows = self.self_attention(windows, windows, windows, need_weights=False)[0]
                self_outputs = self._merge_windows(windows, batch_size=tokens.shape[0])
            tokens = tokens + self_gate.unsqueeze(1) * self_outputs
            context_inputs = _modulate(self.context_norm(tokens), context_shift, context_scale)
            context_outputs = self.context_attention(
                context_inputs,
                context,
                context,
                key_padding_mask=~context_mask,
                need_weights=False,
            )[0]
            tokens = tokens + context_gate.unsqueeze(1) * context_outputs
            mlp_inputs = _modulate(self.mlp_norm(tokens), mlp_shift, mlp_scale)
            return tokens + mlp_gate.unsqueeze(1) * self.mlp(mlp_inputs)

        def _partition_windows(self, tokens: torch.Tensor) -> torch.Tensor:
            batch, _, width = tokens.shape
            grid = self.config.grid_size
            window = self.config.window_size
            groups = grid // window
            value = tokens.reshape(batch, grid, grid, width)
            value = value.reshape(batch, groups, window, groups, window, width)
            return value.permute(0, 1, 3, 2, 4, 5).reshape(
                batch * groups * groups, window * window, width
            )

        def _merge_windows(self, windows: torch.Tensor, *, batch_size: int) -> torch.Tensor:
            grid = self.config.grid_size
            window = self.config.window_size
            groups = grid // window
            width = windows.shape[-1]
            value = windows.reshape(batch_size, groups, groups, window, window, width)
            return value.permute(0, 1, 3, 2, 4, 5).reshape(batch_size, grid * grid, width)

    class LatentStillDiT(nn.Module):
        """Text-conditioned velocity model over continuous sprite latents."""

        def __init__(self, config: LatentStillDiTConfig | None = None) -> None:
            super().__init__()
            self.config = config or LatentStillDiTConfig()
            self.patch_embedding = nn.Conv2d(
                self.config.latent_channels,
                self.config.model_dim,
                kernel_size=self.config.patch_size,
                stride=self.config.patch_size,
            )
            self.position = nn.Parameter(
                torch.zeros(1, self.config.token_count, self.config.model_dim)
            )
            self.context_projection = nn.Linear(self.config.condition_dim, self.config.model_dim)
            self.null_context = nn.Parameter(torch.zeros(1, 1, self.config.model_dim))
            self.timestep_mlp = nn.Sequential(
                nn.Linear(self.config.model_dim, self.config.model_dim * 4),
                nn.SiLU(),
                nn.Linear(self.config.model_dim * 4, self.config.model_dim),
            )
            self.blocks = nn.ModuleList(
                _LatentStillDiTBlock(
                    self.config,
                    global_attention=(index + 1) % self.config.global_attention_every == 0,
                )
                for index in range(self.config.depth)
            )
            self.final_norm = nn.LayerNorm(
                self.config.model_dim, elementwise_affine=False, eps=1e-6
            )
            self.final_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(self.config.model_dim, self.config.model_dim * 2)
            )
            self.final_projection = nn.Linear(self.config.model_dim, self.config.patch_vector_width)
            self._initialize()

        def _initialize(self) -> None:
            nn.init.normal_(self.position, std=0.02)
            nn.init.normal_(self.null_context, std=0.02)
            for block in self.blocks:
                nn.init.zeros_(block.modulation[-1].weight)
                nn.init.zeros_(block.modulation[-1].bias)
            nn.init.zeros_(self.final_modulation[-1].weight)
            nn.init.zeros_(self.final_modulation[-1].bias)
            nn.init.zeros_(self.final_projection.weight)
            nn.init.zeros_(self.final_projection.bias)

        def forward(
            self,
            latent: torch.Tensor,
            timesteps: torch.Tensor,
            context: torch.Tensor | None = None,
            *,
            context_mask: torch.Tensor | None = None,
            context_dropout_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            validate_latent_still_shapes(
                tuple(latent.shape),
                tuple(timesteps.shape),
                tuple(context.shape) if context is not None else None,
                self.config,
            )
            if not latent.is_floating_point() or not timesteps.is_floating_point():
                raise TypeError("latent and timesteps must use floating point")
            batch_size = latent.shape[0]
            projected_context, mask, pooled = self._prepare_context(
                context,
                context_mask,
                context_dropout_mask,
                batch_size=batch_size,
                device=latent.device,
            )
            global_condition = (
                self.timestep_mlp(
                    _timestep_embedding(timesteps, self.config.model_dim).to(latent.dtype)
                )
                + pooled
            )
            tokens = self.patch_embedding(latent).flatten(2).transpose(1, 2)
            tokens = tokens + self.position.to(dtype=tokens.dtype)
            for block in self.blocks:
                tokens = block(tokens, global_condition, projected_context, mask)
            shift, scale = self.final_modulation(global_condition).chunk(2, dim=-1)
            patches = self.final_projection(_modulate(self.final_norm(tokens), shift, scale))
            return self._unpatchify(patches)

        def _prepare_context(
            self,
            context: torch.Tensor | None,
            context_mask: torch.Tensor | None,
            context_dropout_mask: torch.Tensor | None,
            *,
            batch_size: int,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if context is None:
                if context_mask is not None:
                    raise ValueError("context_mask cannot be supplied without context")
                if context_dropout_mask is not None:
                    raise ValueError("context_dropout_mask cannot be supplied without context")
                projected = self.null_context.expand(batch_size, -1, -1)
                mask = torch.ones((batch_size, 1), device=device, dtype=torch.bool)
                return projected, mask, projected[:, 0]
            projected = self.context_projection(context)
            if context_mask is None:
                mask = torch.ones((batch_size, context.shape[1]), device=device, dtype=torch.bool)
            else:
                if tuple(context_mask.shape) != (batch_size, context.shape[1]):
                    raise ValueError("context_mask must match [B,L]")
                mask = context_mask.to(device=device, dtype=torch.bool)
                if not bool(mask.any(dim=1).all()):
                    raise ValueError("each context row must retain at least one token")
            if context_dropout_mask is not None:
                if tuple(context_dropout_mask.shape) != (batch_size,):
                    raise ValueError("context_dropout_mask must have shape [B]")
                dropout_rows = context_dropout_mask.to(device=device, dtype=torch.bool)
                if bool(dropout_rows.any()):
                    null_rows = self.null_context.expand(batch_size, projected.shape[1], -1)
                    projected = torch.where(dropout_rows[:, None, None], null_rows, projected)
                    null_mask = torch.zeros_like(mask)
                    null_mask[:, 0] = True
                    mask = torch.where(dropout_rows[:, None], null_mask, mask)
            weights = mask.to(dtype=projected.dtype).unsqueeze(-1)
            pooled = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return projected, mask, pooled

        def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
            batch = patches.shape[0]
            grid = self.config.grid_size
            patch = self.config.patch_size
            value = patches.reshape(
                batch,
                grid,
                grid,
                patch,
                patch,
                self.config.latent_channels,
            )
            return value.permute(0, 5, 1, 3, 2, 4).reshape(
                batch,
                self.config.latent_channels,
                self.config.latent_size,
                self.config.latent_size,
            )

else:

    class LatentStillDiT:  # pragma: no cover - optional dependency boundary
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise MissingLatentStillTorchError(
                "LatentStillDiT requires a platform-appropriate PyTorch installation"
            ) from _TORCH_IMPORT_ERROR
