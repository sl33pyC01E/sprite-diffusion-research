from __future__ import annotations

import pytest


def test_semantic_conditioning_changes_only_the_summary_token_before_model_attention() -> None:
    torch = pytest.importorskip("torch")
    from spritelab.captions import SpriteGenerationRequest
    from spritelab.models.conditioning import encode_generation_conditions
    from spritelab.models.config import ConditioningSchema
    from spritelab.models.semantic_conditioning import SemanticSpriteConditionEncoder

    schema = ConditioningSchema()
    requests = (
        SpriteGenerationRequest("red fox", "animal", "run", "side", "right", "loop"),
        SpriteGenerationRequest("blue wolf", "animal", "run", "side", "right", "loop"),
    )
    encoded = encode_generation_conditions(requests, schema, max_text_bytes=12)
    encoder = SemanticSpriteConditionEncoder(
        schema, condition_dim=16, semantic_dim=8, max_text_bytes=12
    )
    zero = torch.zeros((2, 8))
    one = torch.zeros((2, 8))
    one[:, 0] = 1

    zero_context, zero_mask = encoder(encoded, zero)
    one_context, one_mask = encoder(encoded, one)

    assert zero_context.shape == one_context.shape == (2, encoder.context_tokens, 16)
    assert torch.equal(zero_mask, one_mask)
    assert not torch.equal(zero_context[:, 0], one_context[:, 0])
    assert torch.equal(zero_context[:, 1:], one_context[:, 1:])


def test_semantic_conditioning_rejects_bad_vector_contracts() -> None:
    torch = pytest.importorskip("torch")
    from spritelab.captions import SpriteGenerationRequest
    from spritelab.models.conditioning import encode_generation_conditions
    from spritelab.models.config import ConditioningSchema
    from spritelab.models.semantic_conditioning import SemanticSpriteConditionEncoder

    schema = ConditioningSchema()
    encoded = encode_generation_conditions(
        (SpriteGenerationRequest("fox", "animal", "run", "side", "right", "loop"),),
        schema,
        max_text_bytes=8,
    )
    encoder = SemanticSpriteConditionEncoder(
        schema, condition_dim=8, semantic_dim=4, max_text_bytes=8
    )
    with pytest.raises(ValueError, match="shape"):
        encoder(encoded, torch.zeros((1, 3)))
    with pytest.raises(TypeError, match="floating"):
        encoder(encoded, torch.zeros((1, 4), dtype=torch.long))
    with pytest.raises(ValueError, match="finite"):
        encoder(encoded, torch.full((1, 4), float("nan")))
