"""Optional-PyTorch factorized spatiotemporal PixelDiT skeleton."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .config import (
    PixelDiTConfig,
    validate_conditioning_shape,
    validate_phase_shape,
    validate_video_shape,
)

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - branch is exercised without torch
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingTorchError(RuntimeError):
    """Raised when the model is constructed without a PyTorch installation."""


def torch_available() -> bool:
    """Return whether the optional PyTorch runtime imported successfully."""

    return torch is not None and nn is not None


def require_torch() -> None:
    """Raise an actionable error when the optional model runtime is unavailable."""

    if not torch_available():
        raise MissingTorchError(
            "FactorizedSpriteDiT requires a platform-appropriate PyTorch installation; "
            "the torch-free configuration and validators remain available"
        ) from _TORCH_IMPORT_ERROR


if torch is not None and nn is not None:

    def _timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
        """Create sinusoidal diffusion-time embeddings."""

        half = width // 2
        if half == 0:
            return timesteps.float().unsqueeze(-1)
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
        inputs: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        while shift.ndim < inputs.ndim:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        return inputs * (1 + scale) + shift

    class FactorizedDiTBlock(nn.Module):
        """Spatial attention, temporal attention, context attention, then MLP."""

        def __init__(self, config: PixelDiTConfig) -> None:
            super().__init__()
            width = config.model_dim
            self.spatial_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.temporal_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.context_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.spatial_attention = nn.MultiheadAttention(
                width,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.temporal_attention = nn.MultiheadAttention(
                width,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.context_attention = nn.MultiheadAttention(
                width,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            hidden_width = int(width * config.mlp_ratio)
            self.mlp = nn.Sequential(
                nn.Linear(width, hidden_width),
                nn.GELU(approximate="tanh"),
                nn.Dropout(config.dropout),
                nn.Linear(hidden_width, width),
                nn.Dropout(config.dropout),
            )
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 12))

        def forward(
            self,
            tokens: torch.Tensor,
            global_condition: torch.Tensor,
            context: torch.Tensor,
            context_mask: torch.Tensor,
        ) -> torch.Tensor:
            batch, frames, patches, width = tokens.shape
            parameters = self.modulation(global_condition).chunk(12, dim=-1)
            (
                spatial_shift,
                spatial_scale,
                spatial_gate,
                temporal_shift,
                temporal_scale,
                temporal_gate,
                context_shift,
                context_scale,
                context_gate,
                mlp_shift,
                mlp_scale,
                mlp_gate,
            ) = parameters

            spatial_inputs = _modulate(
                self.spatial_norm(tokens), spatial_shift, spatial_scale
            ).reshape(batch * frames, patches, width)
            spatial_outputs = self.spatial_attention(
                spatial_inputs,
                spatial_inputs,
                spatial_inputs,
                need_weights=False,
            )[0].reshape(batch, frames, patches, width)
            tokens = tokens + spatial_gate[:, None, None, :] * spatial_outputs

            temporal_inputs = _modulate(self.temporal_norm(tokens), temporal_shift, temporal_scale)
            temporal_inputs = temporal_inputs.permute(0, 2, 1, 3).reshape(
                batch * patches, frames, width
            )
            temporal_outputs = self.temporal_attention(
                temporal_inputs,
                temporal_inputs,
                temporal_inputs,
                need_weights=False,
            )[0]
            temporal_outputs = temporal_outputs.reshape(batch, patches, frames, width).permute(
                0, 2, 1, 3
            )
            tokens = tokens + temporal_gate[:, None, None, :] * temporal_outputs

            flat_tokens = tokens.reshape(batch, frames * patches, width)
            context_inputs = _modulate(self.context_norm(flat_tokens), context_shift, context_scale)
            context_outputs = self.context_attention(
                context_inputs,
                context,
                context,
                key_padding_mask=~context_mask,
                need_weights=False,
            )[0]
            flat_tokens = flat_tokens + context_gate[:, None, :] * context_outputs
            tokens = flat_tokens.reshape(batch, frames, patches, width)

            mlp_inputs = _modulate(self.mlp_norm(tokens), mlp_shift, mlp_scale)
            return tokens + mlp_gate[:, None, None, :] * self.mlp(mlp_inputs)

    class FinalLayer(nn.Module):
        """Conditioned normalization and projection back to RGBA patch vectors."""

        def __init__(self, config: PixelDiTConfig) -> None:
            super().__init__()
            width = config.model_dim
            self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 2))
            self.projection = nn.Linear(width, config.patch_grid.patch_vector_width)

        def forward(self, tokens: torch.Tensor, global_condition: torch.Tensor) -> torch.Tensor:
            shift, scale = self.modulation(global_condition).chunk(2, dim=-1)
            return self.projection(_modulate(self.norm(tokens), shift, scale))

    class FactorizedSpriteDiT(nn.Module):
        """Fixed-resolution native-RGBA diffusion transformer.

        Public video layout is ``[batch, frames, channels, height, width]``.
        Context may be pooled ``[B, D]`` or tokenized ``[B, L, D]``. Production
        callers should supply identity/entity/action/view/direction/loop tokens in
        the schema order, followed by any free-text tokens. Explicit phase values
        are normalized to ``[0, 1]`` and supplied separately per frame.
        """

        def __init__(self, config: PixelDiTConfig | None = None) -> None:
            super().__init__()
            self.config = config or PixelDiTConfig()
            grid = self.config.patch_grid
            self.patch_embedding = nn.Conv3d(
                self.config.channels,
                self.config.model_dim,
                kernel_size=(1, self.config.patch_size, self.config.patch_size),
                stride=(1, self.config.patch_size, self.config.patch_size),
            )
            self.spatial_position = nn.Parameter(
                torch.zeros(1, 1, grid.tokens_per_frame, self.config.model_dim)
            )
            phase_feature_width = 1 + 2 * self.config.phase_harmonics
            self.phase_projection = nn.Linear(phase_feature_width, self.config.model_dim)
            self.condition_projection = nn.Linear(self.config.condition_dim, self.config.model_dim)
            self.null_context = nn.Parameter(torch.zeros(1, 1, self.config.model_dim))
            self.timestep_mlp = nn.Sequential(
                nn.Linear(self.config.model_dim, self.config.model_dim * 4),
                nn.SiLU(),
                nn.Linear(self.config.model_dim * 4, self.config.model_dim),
            )
            self.blocks = nn.ModuleList(
                FactorizedDiTBlock(self.config) for _ in range(self.config.depth)
            )
            self.final_layer = FinalLayer(self.config)
            self._initialize_weights()

        def _initialize_weights(self) -> None:
            nn.init.normal_(self.spatial_position, std=0.02)
            nn.init.normal_(self.null_context, std=0.02)
            for block in self.blocks:
                nn.init.zeros_(block.modulation[-1].weight)
                nn.init.zeros_(block.modulation[-1].bias)
            nn.init.zeros_(self.final_layer.modulation[-1].weight)
            nn.init.zeros_(self.final_layer.modulation[-1].bias)
            nn.init.zeros_(self.final_layer.projection.weight)
            nn.init.zeros_(self.final_layer.projection.bias)

        def _prepare_context(
            self,
            conditioning: torch.Tensor | None,
            conditioning_mask: torch.Tensor | None,
            *,
            batch_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if conditioning is None:
                if conditioning_mask is not None:
                    raise ValueError("conditioning_mask cannot be supplied without conditioning")
                context = self.null_context.expand(batch_size, -1, -1)
                mask = torch.ones((batch_size, 1), device=context.device, dtype=torch.bool)
                return context, mask, context[:, 0]

            validate_conditioning_shape(
                tuple(conditioning.shape), self.config, batch_size=batch_size
            )
            if conditioning.ndim == 2:
                conditioning = conditioning.unsqueeze(1)
            context = self.condition_projection(conditioning)
            token_count = context.shape[1]
            if conditioning_mask is None:
                mask = torch.ones(
                    (batch_size, token_count), device=context.device, dtype=torch.bool
                )
            else:
                if tuple(conditioning_mask.shape) != (batch_size, token_count):
                    raise ValueError(
                        "conditioning_mask must match [B, L]; "
                        f"got {tuple(conditioning_mask.shape)!r}"
                    )
                mask = conditioning_mask.to(device=context.device, dtype=torch.bool)
                if not bool(mask.any(dim=1).all()):
                    raise ValueError("each conditioning row must keep at least one token")
            weights = mask.to(dtype=context.dtype).unsqueeze(-1)
            pooled = (context * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return context, mask, pooled

        def _phase_features(
            self,
            frame_phase: torch.Tensor | None,
            *,
            batch_size: int,
            device: torch.device,
        ) -> torch.Tensor:
            if frame_phase is None:
                frame_phase = (
                    torch.arange(self.config.num_frames, device=device, dtype=torch.float32)
                    / self.config.num_frames
                )
                frame_phase = frame_phase.unsqueeze(0).expand(batch_size, -1)
            else:
                validate_phase_shape(tuple(frame_phase.shape), self.config, batch_size=batch_size)
                frame_phase = frame_phase.to(device=device, dtype=torch.float32)
                if not bool(torch.isfinite(frame_phase).all()):
                    raise ValueError("frame_phase values must be finite")
                if bool((frame_phase < 0).any()) or bool((frame_phase > 1).any()):
                    raise ValueError("frame_phase values must be in [0, 1]")
            frequencies = torch.arange(
                1,
                self.config.phase_harmonics + 1,
                device=device,
                dtype=torch.float32,
            )
            angles = 2 * math.pi * frame_phase.unsqueeze(-1) * frequencies
            features = torch.cat(
                (frame_phase.unsqueeze(-1), torch.sin(angles), torch.cos(angles)), dim=-1
            )
            return self.phase_projection(features).unsqueeze(2)

        def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
            config = self.config
            grid = config.patch_grid
            batch = patches.shape[0]
            patches = patches.reshape(
                batch,
                config.num_frames,
                grid.rows,
                grid.columns,
                config.patch_size,
                config.patch_size,
                config.channels,
            )
            return (
                patches.permute(0, 1, 6, 2, 4, 3, 5)
                .contiguous()
                .reshape(
                    batch,
                    config.num_frames,
                    config.channels,
                    config.height,
                    config.width,
                )
            )

        def forward(
            self,
            video: torch.Tensor,
            timesteps: torch.Tensor,
            conditioning: torch.Tensor | None = None,
            *,
            conditioning_mask: torch.Tensor | None = None,
            frame_phase: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Predict an RGBA flow/noise target with the same shape as ``video``."""

            validate_video_shape(tuple(video.shape), self.config)
            batch_size = video.shape[0]
            if tuple(timesteps.shape) != (batch_size,):
                raise ValueError(
                    f"timesteps must have shape {(batch_size,)!r}; got {tuple(timesteps.shape)!r}"
                )
            context, context_mask, pooled_context = self._prepare_context(
                conditioning, conditioning_mask, batch_size=batch_size
            )
            timestep_condition = self.timestep_mlp(
                _timestep_embedding(timesteps, self.config.model_dim).to(video.dtype)
            )
            global_condition = timestep_condition + pooled_context

            embedded = self.patch_embedding(video.permute(0, 2, 1, 3, 4))
            embedded = embedded.flatten(3).permute(0, 2, 3, 1)
            tokens = embedded + self.spatial_position
            tokens = tokens + self._phase_features(
                frame_phase, batch_size=batch_size, device=video.device
            ).to(dtype=tokens.dtype)
            for block in self.blocks:
                tokens = block(tokens, global_condition, context, context_mask)
            return self._unpatchify(self.final_layer(tokens, global_condition))


else:

    class FactorizedSpriteDiT:
        """Import-safe placeholder used when the optional PyTorch runtime is absent."""

        def __init__(
            self,
            config: PixelDiTConfig | None = None,
            *args: object,
            **kwargs: object,
        ) -> None:
            del config, args, kwargs
            require_torch()


def validate_model_input_shapes(
    video_shape: Sequence[int],
    conditioning_shape: Sequence[int] | None,
    config: PixelDiTConfig,
) -> None:
    """Validate model-facing shapes without importing or constructing tensors."""

    validate_video_shape(video_shape, config)
    if conditioning_shape is not None:
        validate_conditioning_shape(conditioning_shape, config, batch_size=tuple(video_shape)[0])
