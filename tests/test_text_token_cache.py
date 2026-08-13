from __future__ import annotations

import numpy as np
import pytest

from spritelab.text_token_cache import TextTokenCacheError, validate_text_token_arrays


def test_text_token_arrays_require_exact_geometry_and_finite_values() -> None:
    embeddings = np.zeros((2, 77, 768), dtype=np.float16)
    input_ids = np.ones((2, 77), dtype=np.int32)
    attention_mask = np.ones((2, 77), dtype=np.bool_)
    validate_text_token_arrays(embeddings, input_ids, attention_mask, row_count=2)

    embeddings[0, 0, 0] = np.nan
    with pytest.raises(TextTokenCacheError, match="non-finite"):
        validate_text_token_arrays(embeddings, input_ids, attention_mask, row_count=2)


def test_text_token_arrays_reject_empty_prompt_mask() -> None:
    embeddings = np.zeros((1, 77, 768), dtype=np.float16)
    input_ids = np.ones((1, 77), dtype=np.int32)
    attention_mask = np.zeros((1, 77), dtype=np.bool_)
    with pytest.raises(TextTokenCacheError, match="every prompt"):
        validate_text_token_arrays(embeddings, input_ids, attention_mask, row_count=1)
