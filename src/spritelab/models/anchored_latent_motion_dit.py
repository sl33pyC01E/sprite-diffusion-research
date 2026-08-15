"""Action-conditioned latent DiT with hard start/middle/end frame anchors."""

from __future__ import annotations

import math

from .latent_motion_dit import LatentMotionDiTConfig, ReferenceConditionedLatentMotionDiT

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency boundary
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingAnchoredMotionTorchError(RuntimeError):
    """Raised when the optional anchored-motion runtime is unavailable."""


def validate_anchor_shapes(
    video_shape: tuple[int, ...],
    anchor_shape: tuple[int, ...],
    anchor_mask_shape: tuple[int, ...],
) -> tuple[int, int]:
    """Validate video/anchor ``[B,T,C,H,W]`` and mask ``[B,T]`` geometry."""

    if len(video_shape) != 5:
        raise ValueError("video must have shape [B,T,C,H,W]")
    if anchor_shape != video_shape:
        raise ValueError("anchor residuals must match video shape")
    batch, frames = video_shape[:2]
    if anchor_mask_shape != (batch, frames):
        raise ValueError("anchor mask must have shape [B,T]")
    return batch, frames


if torch is not None and nn is not None:

    class AnchoredActionConditionedLatentMotionDiT(nn.Module):
        """Predict missing latent trajectory frames around immutable anchors."""

        def __init__(
            self,
            config: LatentMotionDiTConfig,
            action_count: int,
            *,
            action_token_count: int = 4,
            action_condition_scale: float = 2.0,
        ) -> None:
            super().__init__()
            if isinstance(action_count, bool) or not isinstance(action_count, int):
                raise ValueError("action_count must be an integer")
            if action_count <= 1:
                raise ValueError("action_count must exceed one")
            if isinstance(action_token_count, bool) or not isinstance(action_token_count, int):
                raise ValueError("action_token_count must be an integer")
            if action_token_count <= 0:
                raise ValueError("action_token_count must be positive")
            if not math.isfinite(action_condition_scale) or action_condition_scale <= 0:
                raise ValueError("action_condition_scale must be finite and positive")
            self.config = config
            self.action_token_count = action_token_count
            self.action_condition_scale = action_condition_scale
            self.dit = ReferenceConditionedLatentMotionDiT(config)
            self.action_embedding = nn.Embedding(action_count, config.condition_dim)
            self.action_norm = nn.LayerNorm(config.condition_dim)
            self.action_token_projection = nn.Linear(
                config.condition_dim, action_token_count * config.condition_dim
            )
            self.action_token_norm = nn.LayerNorm(config.condition_dim)
            phase_width = 1 + 2 * config.phase_harmonics
            self.action_frame_mlp = nn.Sequential(
                nn.Linear(config.condition_dim + phase_width, config.model_dim * 2),
                nn.SiLU(),
                nn.Linear(config.model_dim * 2, config.model_dim),
            )
            self.anchor_adapter = nn.Conv3d(
                config.latent_channels + 1,
                config.latent_channels,
                kernel_size=1,
            )
            self._initialize_conditioning()

        def _initialize_conditioning(self) -> None:
            with torch.no_grad():
                self.action_token_projection.weight.zero_()
                self.action_token_projection.bias.zero_()
                identity = torch.eye(
                    self.config.condition_dim,
                    dtype=self.action_token_projection.weight.dtype,
                )
                for token_index in range(self.action_token_count):
                    start = token_index * self.config.condition_dim
                    self.action_token_projection.weight[
                        start : start + self.config.condition_dim
                    ].copy_(identity)
                nn.init.normal_(self.action_frame_mlp[-1].weight, std=0.02)
                self.action_frame_mlp[-1].bias.zero_()
                nn.init.normal_(self.anchor_adapter.weight, std=0.02)
                self.anchor_adapter.bias.zero_()

        def _action_conditioning(
            self, action_indices: torch.Tensor, frame_phase: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            base = self.action_norm(self.action_embedding(action_indices))
            context = self.action_token_projection(base).reshape(
                base.shape[0], self.action_token_count, self.config.condition_dim
            )
            context = self.action_token_norm(context) * self.action_condition_scale
            frequencies = torch.arange(
                1,
                self.config.phase_harmonics + 1,
                device=frame_phase.device,
                dtype=torch.float32,
            )
            phase = frame_phase.to(dtype=torch.float32)
            angles = 2 * math.pi * phase.unsqueeze(-1) * frequencies
            phase_features = torch.cat(
                (phase.unsqueeze(-1), torch.sin(angles), torch.cos(angles)), dim=-1
            )
            frame_input = torch.cat(
                (
                    base.unsqueeze(1).expand(-1, self.config.num_frames, -1),
                    phase_features,
                ),
                dim=-1,
            )
            frame_conditioning = self.action_frame_mlp(frame_input) * self.action_condition_scale
            return context, frame_conditioning

        def forward(
            self,
            video: torch.Tensor,
            reference: torch.Tensor,
            timesteps: torch.Tensor,
            action_indices: torch.Tensor,
            *,
            frame_phase: torch.Tensor,
            anchor_residuals: torch.Tensor,
            anchor_mask: torch.Tensor,
        ) -> torch.Tensor:
            batch, frames = validate_anchor_shapes(
                tuple(video.shape), tuple(anchor_residuals.shape), tuple(anchor_mask.shape)
            )
            if frames != self.config.num_frames:
                raise ValueError("anchor frame count differs from model configuration")
            if anchor_mask.dtype != torch.bool:
                raise ValueError("anchor mask must be boolean")
            if not bool(anchor_mask.any(dim=1).all()):
                raise ValueError("every sample must contain at least one anchor")
            if not bool((~anchor_mask).any(dim=1).all()):
                raise ValueError("every sample must retain at least one predicted frame")
            if tuple(action_indices.shape) != (batch,):
                raise ValueError("action indices must have shape [B]")
            mask_channel = anchor_mask.to(device=video.device, dtype=video.dtype).view(
                batch, frames, 1, 1, 1
            )
            mask_channel = mask_channel.expand(
                -1, -1, 1, self.config.latent_size, self.config.latent_size
            )
            anchor_input = torch.cat(
                (anchor_residuals.to(video.dtype), mask_channel), dim=2
            ).permute(0, 2, 1, 3, 4)
            anchor_features = self.anchor_adapter(anchor_input).permute(0, 2, 1, 3, 4)
            context, frame_conditioning = self._action_conditioning(action_indices, frame_phase)
            return self.dit(
                video + anchor_features,
                reference,
                timesteps,
                context,
                frame_phase=frame_phase,
                frame_conditioning=frame_conditioning,
            )

    def apply_latent_anchors(
        video: torch.Tensor,
        anchor_residuals: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replace all anchored time slots exactly, without interpolation."""

        batch, frames = validate_anchor_shapes(
            tuple(video.shape), tuple(anchor_residuals.shape), tuple(anchor_mask.shape)
        )
        if anchor_mask.dtype != torch.bool:
            raise ValueError("anchor mask must be boolean")
        mask = anchor_mask.view(batch, frames, 1, 1, 1)
        return torch.where(mask, anchor_residuals, video)

    def masked_velocity_mse(
        predicted_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        anchor_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Supervise only frames that are not supplied as immutable anchors."""

        batch, frames = validate_anchor_shapes(
            tuple(predicted_velocity.shape),
            tuple(target_velocity.shape),
            tuple(anchor_mask.shape),
        )
        if anchor_mask.dtype != torch.bool:
            raise ValueError("anchor mask must be boolean")
        missing = (~anchor_mask).view(batch, frames, 1, 1, 1)
        if not bool(missing.any()):
            raise ValueError("masked velocity loss requires a predicted frame")
        squared = (predicted_velocity - target_velocity).float().square()
        return (squared * missing).sum() / missing.expand_as(squared).sum()

else:

    class AnchoredActionConditionedLatentMotionDiT:
        """Import-safe placeholder when PyTorch is unavailable."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise MissingAnchoredMotionTorchError(
                "AnchoredActionConditionedLatentMotionDiT requires PyTorch"
            ) from _TORCH_IMPORT_ERROR

    def apply_latent_anchors(*_args: object, **_kwargs: object) -> None:
        raise MissingAnchoredMotionTorchError(
            "apply_latent_anchors requires PyTorch"
        ) from _TORCH_IMPORT_ERROR

    def masked_velocity_mse(*_args: object, **_kwargs: object) -> None:
        raise MissingAnchoredMotionTorchError(
            "masked_velocity_mse requires PyTorch"
        ) from _TORCH_IMPORT_ERROR
