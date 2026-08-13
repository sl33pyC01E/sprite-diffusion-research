"""Deterministic text tokenization and optional-PyTorch condition encoding."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .config import ConditioningSchema

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
BYTE_TOKEN_OFFSET = 3
BYTE_VOCAB_SIZE = BYTE_TOKEN_OFFSET + 256
STRUCTURED_TOKEN_COUNT = 6


class GenerationConditionLike(Protocol):
    """Structural input accepted from the public generation-request schema."""

    description: str
    entity_class: str
    action: str
    view: str
    direction: str
    loop_mode: str


@dataclass(frozen=True, slots=True)
class EncodedConditionBatch:
    """Torch-free, padded token and categorical IDs for a prompt batch."""

    descriptions: tuple[str, ...]
    text_token_ids: tuple[tuple[int, ...], ...]
    text_attention_mask: tuple[tuple[bool, ...], ...]
    entity_ids: tuple[int, ...]
    action_ids: tuple[int, ...]
    view_ids: tuple[int, ...]
    direction_ids: tuple[int, ...]
    loop_mode_ids: tuple[int, ...]
    max_text_bytes: int

    @property
    def batch_size(self) -> int:
        return len(self.descriptions)

    @property
    def text_tokens(self) -> int:
        return self.max_text_bytes + 2

    @property
    def context_tokens(self) -> int:
        return STRUCTURED_TOKEN_COUNT + self.text_tokens


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")


def _utf8_prefix(text: str, *, maximum_bytes: int) -> bytes:
    """Return the longest NFC prefix that ends at a Unicode code-point boundary."""

    normalized = unicodedata.normalize("NFC", text)
    output = bytearray()
    for character in normalized:
        encoded = character.encode("utf-8")
        if len(output) + len(encoded) > maximum_bytes:
            break
        output.extend(encoded)
    return bytes(output)


def encode_utf8_text(
    text: str, *, max_text_bytes: int = 96
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    """Encode one prompt as fixed-width UTF-8 byte IDs with BOS/EOS and padding."""

    _positive_integer("max_text_bytes", max_text_bytes)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    encoded = _utf8_prefix(text.strip(), maximum_bytes=max_text_bytes)
    token_ids = [BOS_TOKEN_ID]
    token_ids.extend(BYTE_TOKEN_OFFSET + value for value in encoded)
    token_ids.append(EOS_TOKEN_ID)
    token_count = max_text_bytes + 2
    attention_mask = [True] * len(token_ids)
    padding = token_count - len(token_ids)
    token_ids.extend([PAD_TOKEN_ID] * padding)
    attention_mask.extend([False] * padding)
    return tuple(token_ids), tuple(attention_mask)


def encode_generation_conditions(
    requests: Sequence[GenerationConditionLike],
    schema: ConditioningSchema,
    *,
    max_text_bytes: int = 96,
) -> EncodedConditionBatch:
    """Encode prompts while retaining independent categorical steering channels."""

    _positive_integer("max_text_bytes", max_text_bytes)
    if not requests:
        raise ValueError("at least one generation condition is required")
    vocabularies = {
        "entity_class": schema.entity_classes,
        "action": schema.action_classes,
        "view": schema.view_classes,
        "direction": schema.direction_classes,
        "loop_mode": schema.loop_modes,
    }
    lookup = {
        name: {label: index for index, label in enumerate(vocabulary)}
        for name, vocabulary in vocabularies.items()
    }
    descriptions: list[str] = []
    text_token_ids: list[tuple[int, ...]] = []
    text_attention_mask: list[tuple[bool, ...]] = []
    categorical_ids: dict[str, list[int]] = {name: [] for name in vocabularies}
    for request in requests:
        description = request.description.strip()
        token_ids, attention_mask = encode_utf8_text(
            description,
            max_text_bytes=max_text_bytes,
        )
        descriptions.append(description)
        text_token_ids.append(token_ids)
        text_attention_mask.append(attention_mask)
        for name, vocabulary in vocabularies.items():
            value = getattr(request, name)
            if value not in lookup[name]:
                raise ValueError(f"unknown {name}: {value!r}; expected one of {vocabulary!r}")
            categorical_ids[name].append(lookup[name][value])
    return EncodedConditionBatch(
        descriptions=tuple(descriptions),
        text_token_ids=tuple(text_token_ids),
        text_attention_mask=tuple(text_attention_mask),
        entity_ids=tuple(categorical_ids["entity_class"]),
        action_ids=tuple(categorical_ids["action"]),
        view_ids=tuple(categorical_ids["view"]),
        direction_ids=tuple(categorical_ids["direction"]),
        loop_mode_ids=tuple(categorical_ids["loop_mode"]),
        max_text_bytes=max_text_bytes,
    )


try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised in the base environment
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


class MissingConditioningTorchError(RuntimeError):
    """Raised when the trainable encoder is constructed without PyTorch."""


if torch is not None and nn is not None:

    class SpriteConditionEncoder(nn.Module):
        """Small trainable byte-text and structured-label conditioning baseline.

        Context order is ``description summary, entity, action, view, direction,
        loop mode, text bytes``. The description summary occupies the identity slot
        in the denoiser contract; no closed-set identity embedding is required at
        inference time.
        """

        def __init__(
            self,
            schema: ConditioningSchema,
            *,
            condition_dim: int,
            max_text_bytes: int = 96,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            _positive_integer("condition_dim", condition_dim)
            _positive_integer("max_text_bytes", max_text_bytes)
            if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
                raise ValueError(f"dropout must be in [0, 1); got {dropout!r}")
            self.schema = schema
            self.condition_dim = condition_dim
            self.max_text_bytes = max_text_bytes
            self.text_embedding = nn.Embedding(
                BYTE_VOCAB_SIZE,
                condition_dim,
                padding_idx=PAD_TOKEN_ID,
            )
            self.text_position = nn.Embedding(max_text_bytes + 2, condition_dim)
            self.token_type = nn.Embedding(7, condition_dim)
            self.entity_embedding = nn.Embedding(len(schema.entity_classes), condition_dim)
            self.action_embedding = nn.Embedding(len(schema.action_classes), condition_dim)
            self.view_embedding = nn.Embedding(len(schema.view_classes), condition_dim)
            self.direction_embedding = nn.Embedding(len(schema.direction_classes), condition_dim)
            self.loop_embedding = nn.Embedding(len(schema.loop_modes), condition_dim)
            self.summary_projection = nn.Sequential(
                nn.LayerNorm(condition_dim),
                nn.Linear(condition_dim, condition_dim),
                nn.SiLU(),
                nn.Linear(condition_dim, condition_dim),
            )
            self.output_norm = nn.LayerNorm(condition_dim)
            self.dropout = nn.Dropout(dropout)

        @property
        def context_tokens(self) -> int:
            return STRUCTURED_TOKEN_COUNT + self.max_text_bytes + 2

        def forward(
            self,
            batch: EncodedConditionBatch,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Convert a deterministic CPU batch into model context and mask."""

            if batch.max_text_bytes != self.max_text_bytes:
                raise ValueError(
                    "encoded max_text_bytes must match encoder; "
                    f"got {batch.max_text_bytes} and {self.max_text_bytes}"
                )
            device = self.text_embedding.weight.device
            text_token_ids = torch.as_tensor(
                batch.text_token_ids,
                dtype=torch.long,
                device=device,
            )
            text_attention_mask = torch.as_tensor(
                batch.text_attention_mask,
                dtype=torch.bool,
                device=device,
            )
            categorical = tuple(
                torch.as_tensor(values, dtype=torch.long, device=device)
                for values in (
                    batch.entity_ids,
                    batch.action_ids,
                    batch.view_ids,
                    batch.direction_ids,
                    batch.loop_mode_ids,
                )
            )
            return self.forward_tensors(
                text_token_ids,
                text_attention_mask,
                *categorical,
            )

        def forward_tensors(
            self,
            text_token_ids: torch.Tensor,
            text_attention_mask: torch.Tensor,
            entity_ids: torch.Tensor,
            action_ids: torch.Tensor,
            view_ids: torch.Tensor,
            direction_ids: torch.Tensor,
            loop_mode_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Encode already-collated tensors for a training data loader."""

            expected_text = (text_token_ids.shape[0], self.max_text_bytes + 2)
            if tuple(text_token_ids.shape) != expected_text:
                raise ValueError(
                    f"text_token_ids must have shape {expected_text!r}; "
                    f"got {tuple(text_token_ids.shape)!r}"
                )
            if tuple(text_attention_mask.shape) != expected_text:
                raise ValueError(
                    "text_attention_mask must match text_token_ids; "
                    f"got {tuple(text_attention_mask.shape)!r} and {expected_text!r}"
                )
            batch_size = text_token_ids.shape[0]
            categorical_ids = (
                entity_ids,
                action_ids,
                view_ids,
                direction_ids,
                loop_mode_ids,
            )
            if any(tuple(values.shape) != (batch_size,) for values in categorical_ids):
                raise ValueError("each categorical ID tensor must have shape [B]")
            if text_token_ids.dtype != torch.long:
                raise ValueError("text_token_ids must use torch.long")
            if text_attention_mask.dtype != torch.bool:
                raise ValueError("text_attention_mask must use torch.bool")
            positions = torch.arange(
                self.max_text_bytes + 2,
                device=text_token_ids.device,
            )
            text = self.text_embedding(text_token_ids)
            text = text + self.text_position(positions).unsqueeze(0)
            text = text + self.token_type.weight[6].view(1, 1, -1)
            mask_weights = text_attention_mask.unsqueeze(-1).to(text.dtype)
            pooled = (text * mask_weights).sum(dim=1) / mask_weights.sum(dim=1).clamp_min(1)
            summary = self.summary_projection(pooled) + self.token_type.weight[0]
            structured = torch.stack(
                (
                    summary,
                    self.entity_embedding(entity_ids) + self.token_type.weight[1],
                    self.action_embedding(action_ids) + self.token_type.weight[2],
                    self.view_embedding(view_ids) + self.token_type.weight[3],
                    self.direction_embedding(direction_ids) + self.token_type.weight[4],
                    self.loop_embedding(loop_mode_ids) + self.token_type.weight[5],
                ),
                dim=1,
            )
            context = torch.cat((structured, text), dim=1)
            structured_mask = torch.ones(
                (batch_size, STRUCTURED_TOKEN_COUNT),
                dtype=torch.bool,
                device=text_attention_mask.device,
            )
            context_mask = torch.cat((structured_mask, text_attention_mask), dim=1)
            return self.output_norm(self.dropout(context)), context_mask

else:

    class SpriteConditionEncoder:  # pragma: no cover - trivial dependency boundary
        """Dependency-error placeholder when PyTorch is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            raise MissingConditioningTorchError(
                "SpriteConditionEncoder requires a platform-appropriate PyTorch installation; "
                "deterministic prompt tokenization remains available"
            ) from _TORCH_IMPORT_ERROR
