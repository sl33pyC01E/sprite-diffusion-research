from __future__ import annotations

import numpy as np
import pytest

from spritelab.sd_control_cache import composite_rgba_on_background


def test_rgba_composite_is_exact_and_preserves_shape() -> None:
    value = np.array(
        [[[[255, 0, 0, 255], [0, 255, 0, 0], [0, 0, 255, 128]]]],
        dtype=np.uint8,
    )
    result = composite_rgba_on_background(value, background_rgb=(127, 127, 127))
    assert result.dtype == np.float32
    assert result.shape == (1, 1, 3, 3)
    assert result[0, 0, 0].tolist() == [1.0, 0.0, 0.0]
    assert result[0, 0, 1].tolist() == pytest.approx([127 / 255] * 3)
    alpha = 128 / 255
    assert result[0, 0, 2].tolist() == pytest.approx(
        [(127 / 255) * (1 - alpha), (127 / 255) * (1 - alpha), alpha + (127 / 255) * (1 - alpha)]
    )


def test_rgba_composite_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="uint8"):
        composite_rgba_on_background(np.zeros((1, 2, 2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="background_rgb"):
        composite_rgba_on_background(
            np.zeros((1, 2, 2, 4), dtype=np.uint8), background_rgb=(0, 999, 0)
        )
