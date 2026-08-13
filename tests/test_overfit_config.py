from __future__ import annotations

import pytest

from spritelab.overfit import TinyOverfitConfig


def test_tiny_overfit_config_validates_geometry_and_optimization() -> None:
    config = TinyOverfitConfig(steps=2, target_bucket=32, target_frames=4)
    unweighted = TinyOverfitConfig(foreground_weight=0)
    no_endpoint = TinyOverfitConfig(matched_endpoint_weight=0)

    assert config.steps == 2
    assert config.target_bucket == 32
    assert unweighted.foreground_weight == 0
    assert no_endpoint.matched_endpoint_weight == 0

    with pytest.raises(ValueError, match="divisible"):
        TinyOverfitConfig(target_bucket=30, patch_size=4)
    with pytest.raises(ValueError, match="model_dim"):
        TinyOverfitConfig(model_dim=127, num_heads=4)
    with pytest.raises(ValueError, match="learning_rate"):
        TinyOverfitConfig(learning_rate=0)
    with pytest.raises(ValueError, match="foreground_weight"):
        TinyOverfitConfig(foreground_weight=-1)
    with pytest.raises(ValueError, match="matched_endpoint_weight"):
        TinyOverfitConfig(matched_endpoint_weight=-1)
    with pytest.raises(ValueError, match="precision"):
        TinyOverfitConfig(precision="float16")  # type: ignore[arg-type]
