"""Reference-conditioned latent DiT for sprite animation residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import PatchGrid

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency branch
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingLatentMotionTorchError(RuntimeError):
    """Raised when the optional latent-motion model runtime is unavailable."""


@dataclass(frozen=True, slots=True)
class LatentMotionDiTConfig:
    """Fixed geometry for reference-still to eight-frame latent animation."""

    latent_size: int = 64
    num_frames: int = 8
    latent_channels: int = 8
    patch_size: int = 2
    model_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    condition_dim: int = 768
    phase_harmonics: int = 4

    def __post_init__(self) -> None:
        for name in (
            "latent_size",
            "num_frames",
            "latent_channels",
            "patch_size",
            "model_dim",
            "depth",
            "num_heads",
            "condition_dim",
            "phase_harmonics",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.latent_size % self.patch_size:
            raise ValueError("latent_size must be divisible by patch_size")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be finite and positive")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def height(self) -> int:
        return self.latent_size

    @property
    def width(self) -> int:
        return self.latent_size

    @property
    def channels(self) -> int:
        return self.latent_channels

    @property
    def patch_grid(self) -> PatchGrid:
        return PatchGrid(
            frames=self.num_frames,
            rows=self.latent_size // self.patch_size,
            columns=self.latent_size // self.patch_size,
            patch_size=self.patch_size,
            channels=self.latent_channels,
        )


def validate_latent_motion_shapes(
    video_shape: tuple[int, ...],
    reference_shape: tuple[int, ...],
    config: LatentMotionDiTConfig,
) -> int:
    """Validate noised target ``[B,T,C,H,W]`` and reference ``[B,C,H,W]``."""

    if len(video_shape) != 5:
        raise ValueError("video must have shape [B, T, C, H, W]")
    batch = video_shape[0]
    if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
        raise ValueError("video batch must be positive")
    expected_video = (
        batch,
        config.num_frames,
        config.latent_channels,
        config.latent_size,
        config.latent_size,
    )
    expected_reference = (
        batch,
        config.latent_channels,
        config.latent_size,
        config.latent_size,
    )
    if video_shape != expected_video:
        raise ValueError(f"video must have shape {expected_video!r}; got {video_shape!r}")
    if reference_shape != expected_reference:
        raise ValueError(
            f"reference must have shape {expected_reference!r}; got {reference_shape!r}"
        )
    return batch


if torch is not None and nn is not None:
    from .pixeldit import FactorizedDiTBlock, FinalLayer, _timestep_embedding

    class ReferenceConditionedLatentMotionDiT(nn.Module):
        """Predict latent animation residual/noise while preserving a reference still."""

        def __init__(self, config: LatentMotionDiTConfig | None = None) -> None:
            super().__init__()
            self.config = config or LatentMotionDiTConfig()
            grid = self.config.patch_grid
            self.video_patch_embedding = nn.Conv3d(
                self.config.latent_channels,
                self.config.model_dim,
                kernel_size=(1, self.config.patch_size, self.config.patch_size),
                stride=(1, self.config.patch_size, self.config.patch_size),
            )
            self.reference_patch_embedding = nn.Conv2d(
                self.config.latent_channels,
                self.config.model_dim,
                kernel_size=self.config.patch_size,
                stride=self.config.patch_size,
            )
            self.spatial_position = nn.Parameter(
                torch.zeros(1, 1, grid.tokens_per_frame, self.config.model_dim)
            )
            self.reference_type = nn.Parameter(torch.zeros(1, 1, 1, self.config.model_dim))
            phase_width = 1 + 2 * self.config.phase_harmonics
            self.phase_projection = nn.Linear(phase_width, self.config.model_dim)
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
            nn.init.normal_(self.reference_type, std=0.02)
            nn.init.normal_(self.null_context, std=0.02)
            for block in self.blocks:
                nn.init.zeros_(block.modulation[-1].weight)
                nn.init.zeros_(block.modulation[-1].bias)
            nn.init.zeros_(self.final_layer.modulation[-1].weight)
            nn.init.zeros_(self.final_layer.modulation[-1].bias)
            nn.init.zeros_(self.final_layer.projection.weight)
            nn.init.zeros_(self.final_layer.projection.bias)

        def forward(
            self,
            video: torch.Tensor,
            reference: torch.Tensor,
            timesteps: torch.Tensor,
            conditioning: torch.Tensor | None = None,
            *,
            conditioning_mask: torch.Tensor | None = None,
            frame_phase: torch.Tensor | None = None,
        ) -> torch.Tensor:
            batch = validate_latent_motion_shapes(
                tuple(video.shape), tuple(reference.shape), self.config
            )
            if tuple(timesteps.shape) != (batch,):
                raise ValueError(f"timesteps must have shape {(batch,)!r}")
            context, mask, pooled = self._prepare_context(
                conditioning, conditioning_mask, batch=batch, device=video.device
            )
            global_condition = (
                self.timestep_mlp(
                    _timestep_embedding(timesteps, self.config.model_dim).to(video.dtype)
                )
                + pooled
            )
            embedded = self.video_patch_embedding(video.permute(0, 2, 1, 3, 4))
            tokens = embedded.flatten(3).permute(0, 2, 3, 1)
            reference_tokens = self.reference_patch_embedding(reference).flatten(2).transpose(1, 2)
            tokens = tokens + reference_tokens.unsqueeze(1) + self.reference_type
            tokens = tokens + self.spatial_position
            tokens = tokens + self._phase_features(
                frame_phase, batch=batch, device=video.device
            ).to(tokens.dtype)
            for block in self.blocks:
                tokens = block(tokens, global_condition, context, mask)
            return self._unpatchify(self.final_layer(tokens, global_condition))

        def _prepare_context(
            self,
            conditioning: torch.Tensor | None,
            conditioning_mask: torch.Tensor | None,
            *,
            batch: int,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if conditioning is None:
                if conditioning_mask is not None:
                    raise ValueError("conditioning_mask requires conditioning")
                context = self.null_context.expand(batch, -1, -1)
                mask = torch.ones((batch, 1), dtype=torch.bool, device=device)
                return context, mask, context[:, 0]
            if conditioning.ndim == 2:
                conditioning = conditioning.unsqueeze(1)
            if (
                conditioning.ndim != 3
                or conditioning.shape[0] != batch
                or conditioning.shape[2] != self.config.condition_dim
            ):
                raise ValueError("conditioning must have shape [B, L, condition_dim]")
            context = self.condition_projection(conditioning)
            if conditioning_mask is None:
                mask = torch.ones(context.shape[:2], dtype=torch.bool, device=device)
            else:
                if tuple(conditioning_mask.shape) != tuple(context.shape[:2]):
                    raise ValueError("conditioning_mask must match [B, L]")
                mask = conditioning_mask.to(device=device, dtype=torch.bool)
                if not bool(mask.any(dim=1).all()):
                    raise ValueError("each conditioning row must retain a token")
            weights = mask.to(context.dtype).unsqueeze(-1)
            pooled = (context * weights).sum(1) / weights.sum(1).clamp_min(1)
            return context, mask, pooled

        def _phase_features(
            self,
            frame_phase: torch.Tensor | None,
            *,
            batch: int,
            device: torch.device,
        ) -> torch.Tensor:
            if frame_phase is None:
                phase = torch.arange(
                    self.config.num_frames, device=device, dtype=torch.float32
                ).div(self.config.num_frames)
                frame_phase = phase.unsqueeze(0).expand(batch, -1)
            elif tuple(frame_phase.shape) != (batch, self.config.num_frames):
                raise ValueError("frame_phase must match [B, T]")
            else:
                frame_phase = frame_phase.to(device=device, dtype=torch.float32)
            if not bool(torch.isfinite(frame_phase).all()) or bool(
                ((frame_phase < 0) | (frame_phase > 1)).any()
            ):
                raise ValueError("frame_phase values must be finite and in [0, 1]")
            frequencies = torch.arange(
                1, self.config.phase_harmonics + 1, device=device, dtype=torch.float32
            )
            angles = 2 * math.pi * frame_phase.unsqueeze(-1) * frequencies
            features = torch.cat(
                (frame_phase.unsqueeze(-1), torch.sin(angles), torch.cos(angles)), dim=-1
            )
            return self.phase_projection(features).unsqueeze(2)

        def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
            grid = self.config.patch_grid
            patches = patches.reshape(
                patches.shape[0],
                self.config.num_frames,
                grid.rows,
                grid.columns,
                self.config.patch_size,
                self.config.patch_size,
                self.config.latent_channels,
            )
            return (
                patches.permute(0, 1, 6, 2, 4, 3, 5)
                .contiguous()
                .reshape(
                    patches.shape[0],
                    self.config.num_frames,
                    self.config.latent_channels,
                    self.config.latent_size,
                    self.config.latent_size,
                )
            )

else:

    class ReferenceConditionedLatentMotionDiT:
        """Import-safe placeholder when PyTorch is unavailable."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise MissingLatentMotionTorchError(
                "ReferenceConditionedLatentMotionDiT requires PyTorch"
            ) from _TORCH_IMPORT_ERROR
