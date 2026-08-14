from __future__ import annotations

import pytest

from spritelab.sd_lora_inference import SDLoraInferenceConfig


def test_sd_lora_inference_config_validates_sampling() -> None:
    config = SDLoraInferenceConfig()
    assert config.sample_steps == 50
    assert config.guidance_scale == 5
    assert config.weights_variant == "ema"
    with pytest.raises(ValueError, match="sample_steps"):
        SDLoraInferenceConfig(sample_steps=0)
    with pytest.raises(ValueError, match="guidance_scale"):
        SDLoraInferenceConfig(guidance_scale=-1)
    with pytest.raises(ValueError, match="weights_variant"):
        SDLoraInferenceConfig(weights_variant="other")  # type: ignore[arg-type]
