from __future__ import annotations

from dataclasses import dataclass

import pytest

from spritelab.models.conditioning import (
    BOS_TOKEN_ID,
    BYTE_TOKEN_OFFSET,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    encode_generation_conditions,
    encode_utf8_text,
)
from spritelab.models.config import ConditioningSchema


@dataclass(frozen=True)
class Request:
    description: str
    entity_class: str = "animal"
    action: str = "run"
    view: str = "side"
    direction: str = "right"
    loop_mode: str = "loop"


def test_utf8_encoding_is_fixed_width_and_preserves_codepoint_boundaries() -> None:
    token_ids, mask = encode_utf8_text("AéZ", max_text_bytes=3)

    assert token_ids == (
        BOS_TOKEN_ID,
        BYTE_TOKEN_OFFSET + ord("A"),
        BYTE_TOKEN_OFFSET + 0xC3,
        BYTE_TOKEN_OFFSET + 0xA9,
        EOS_TOKEN_ID,
    )
    assert mask == (True, True, True, True, True)

    clipped, clipped_mask = encode_utf8_text("éZ", max_text_bytes=1)
    assert clipped == (BOS_TOKEN_ID, EOS_TOKEN_ID, PAD_TOKEN_ID)
    assert clipped_mask == (True, True, False)


def test_generation_conditions_retain_independent_steering_ids() -> None:
    schema = ConditioningSchema()
    batch = encode_generation_conditions(
        [
            Request("striped wolf", action="run"),
            Request("striped wolf", action="idle", direction="left"),
        ],
        schema,
        max_text_bytes=16,
    )

    assert batch.batch_size == 2
    assert batch.context_tokens == 24
    assert batch.entity_ids[0] == batch.entity_ids[1]
    assert batch.action_ids[0] != batch.action_ids[1]
    assert batch.direction_ids[0] != batch.direction_ids[1]
    assert batch.text_token_ids[0] == batch.text_token_ids[1]


def test_generation_condition_rejects_unknown_labels_and_empty_batches() -> None:
    schema = ConditioningSchema()
    with pytest.raises(ValueError, match="at least one"):
        encode_generation_conditions([], schema)
    with pytest.raises(ValueError, match="unknown action"):
        encode_generation_conditions([Request("wolf", action="teleport")], schema)
    with pytest.raises(ValueError, match="non-empty"):
        encode_generation_conditions([Request("  ")], schema)
