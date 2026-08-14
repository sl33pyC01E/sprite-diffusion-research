from __future__ import annotations

import pytest

from spritelab.sd_lora_train import SDLoraTrainingConfig, sd_lora_target_modules


def test_sd_lora_config_is_explicit_and_validated() -> None:
    config = SDLoraTrainingConfig(steps=4, warmup_steps=1, checkpoint_every=2)
    assert config.rank == config.alpha == 16
    assert config.conditioning_dropout_probability == 0.1
    with pytest.raises(ValueError, match="warmup_steps"):
        SDLoraTrainingConfig(steps=4, warmup_steps=4)
    with pytest.raises(ValueError, match="conditioning_dropout_probability"):
        SDLoraTrainingConfig(conditioning_dropout_probability=1.1)
    assert sd_lora_target_modules("attention_resnet")[-3:] == (
        "conv1",
        "conv2",
        "conv_shortcut",
    )
    with pytest.raises(ValueError, match="target_profile"):
        SDLoraTrainingConfig(target_profile="unknown")  # type: ignore[arg-type]
