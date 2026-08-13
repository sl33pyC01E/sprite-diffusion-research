from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from spritelab.models.conditioning import (
    SpriteConditionEncoder,
    encode_generation_conditions,
)
from spritelab.models.config import ConditioningSchema, PixelDiTConfig
from spritelab.models.pixeldit import FactorizedSpriteDiT

torch = pytest.importorskip("torch")


@dataclass(frozen=True)
class Request:
    description: str
    entity_class: str = "animal"
    action: str = "run"
    view: str = "side"
    direction: str = "right"
    loop_mode: str = "loop"


def test_trainable_encoder_emits_dit_context_and_mask() -> None:
    schema = ConditioningSchema()
    encoded = encode_generation_conditions(
        [Request("small red fox"), Request("clockwork bird", action="fly")],
        schema,
        max_text_bytes=12,
    )
    encoder = SpriteConditionEncoder(schema, condition_dim=16, max_text_bytes=12)

    context, mask = encoder(encoded)

    assert context.shape == (2, 20, 16)
    assert mask.shape == (2, 20)
    assert mask.dtype == torch.bool
    assert mask[:, :6].all()
    assert torch.isfinite(context).all()


def test_encoder_rejects_mismatched_token_width() -> None:
    schema = ConditioningSchema()
    encoded = encode_generation_conditions([Request("fox")], schema, max_text_bytes=8)
    encoder = SpriteConditionEncoder(schema, condition_dim=8, max_text_bytes=9)

    with pytest.raises(ValueError, match="max_text_bytes"):
        encoder(encoded)


def test_prompt_and_structured_controls_feed_pixeldit_end_to_end() -> None:
    schema = replace(ConditioningSchema(), phase_bins=2)
    config = PixelDiTConfig(
        height=8,
        width=8,
        num_frames=2,
        patch_size=2,
        model_dim=32,
        depth=1,
        num_heads=4,
        condition_dim=16,
        phase_harmonics=2,
        conditioning=schema,
    )
    encoded = encode_generation_conditions(
        [Request("small red fox"), Request("clockwork bird", action="fly")],
        schema,
        max_text_bytes=12,
    )
    encoder = SpriteConditionEncoder(schema, condition_dim=16, max_text_bytes=12)
    denoiser = FactorizedSpriteDiT(config)
    context, mask = encoder(encoded)

    output = denoiser(
        torch.randn(2, 2, 4, 8, 8),
        torch.tensor([0.2, 0.8]),
        context,
        conditioning_mask=mask,
        frame_phase=torch.tensor([[0.0, 0.5], [0.0, 0.5]]),
    )

    assert output.shape == (2, 2, 4, 8, 8)
    assert torch.count_nonzero(output) == 0
