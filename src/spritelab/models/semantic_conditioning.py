"""Optional frozen-semantic-vector adapter for sprite generation conditions."""

from __future__ import annotations

from typing import Any

from .conditioning import EncodedConditionBatch, SpriteConditionEncoder
from .config import ConditioningSchema

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - torch-free environment
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingSemanticConditioningTorchError(RuntimeError):
    """Raised when the semantic adapter is constructed without PyTorch."""


if torch is not None and nn is not None:

    class SemanticSpriteConditionEncoder(nn.Module):
        """Fuse a frozen semantic description vector into the text-summary token."""

        def __init__(
            self,
            schema: ConditioningSchema,
            *,
            condition_dim: int,
            semantic_dim: int,
            max_text_bytes: int = 96,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            if (
                isinstance(semantic_dim, bool)
                or not isinstance(semantic_dim, int)
                or semantic_dim <= 0
            ):
                raise ValueError("semantic_dim must be a positive integer")
            self.base = SpriteConditionEncoder(
                schema,
                condition_dim=condition_dim,
                max_text_bytes=max_text_bytes,
                dropout=dropout,
            )
            self.semantic_dim = semantic_dim
            self.semantic_projection = nn.Sequential(
                nn.LayerNorm(semantic_dim),
                nn.Linear(semantic_dim, condition_dim),
                nn.SiLU(),
                nn.Linear(condition_dim, condition_dim),
            )
            self.semantic_norm = nn.LayerNorm(condition_dim)

        @property
        def context_tokens(self) -> int:
            return self.base.context_tokens

        def forward(
            self,
            batch: EncodedConditionBatch,
            semantic_vectors: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            context, mask = self.base(batch)
            expected = (batch.batch_size, self.semantic_dim)
            if tuple(semantic_vectors.shape) != expected:
                raise ValueError(
                    f"semantic_vectors must have shape {expected!r}; "
                    f"got {tuple(semantic_vectors.shape)!r}"
                )
            if not semantic_vectors.is_floating_point():
                raise TypeError("semantic_vectors must use floating point")
            if not bool(torch.isfinite(semantic_vectors).all()):
                raise ValueError("semantic_vectors must be finite")
            projected = self.semantic_projection(
                semantic_vectors.to(device=context.device, dtype=context.dtype)
            )
            fused = context.clone()
            fused[:, 0] = self.semantic_norm(context[:, 0] + projected)
            return fused, mask

else:

    class SemanticSpriteConditionEncoder:  # pragma: no cover - dependency boundary
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            raise MissingSemanticConditioningTorchError(
                "semantic conditioning requires a platform-appropriate PyTorch installation"
            ) from _TORCH_IMPORT_ERROR
