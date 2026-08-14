from __future__ import annotations

import pytest

from spritelab.latent_motion_overfit import (
    LatentMotionOverfitConfig,
    LatentMotionOverfitError,
    _load_initial_checkpoint,
)


def test_objective_weights_require_at_least_one_enabled_objective() -> None:
    with pytest.raises(ValueError, match="at least one training objective"):
        LatentMotionOverfitConfig(base_weight=0, endpoint_weight=0, pixel_endpoint_weight=0)


def test_endpoint_only_configuration_is_explicitly_supported() -> None:
    config = LatentMotionOverfitConfig(base_weight=0, endpoint_weight=1)

    assert config.base_weight == 0
    assert config.endpoint_weight == 1


def test_pixel_only_endpoint_configuration_is_explicitly_supported() -> None:
    config = LatentMotionOverfitConfig(base_weight=0, endpoint_weight=0, pixel_endpoint_weight=1)

    assert config.pixel_endpoint_weight == 1


@pytest.mark.parametrize("name", ["base_weight", "endpoint_weight", "pixel_endpoint_weight"])
def test_objective_weights_reject_negative_values(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        LatentMotionOverfitConfig(**{name: -0.01})


@pytest.mark.parametrize(
    ("path", "expected_sha256"),
    [("checkpoint.pt", None), (None, "0" * 64)],
)
def test_initial_checkpoint_path_and_hash_are_atomic_contract(
    path: str | None, expected_sha256: str | None
) -> None:
    with pytest.raises(LatentMotionOverfitError, match="must be provided together"):
        _load_initial_checkpoint(
            None,
            path,
            expected_sha256=expected_sha256,
            model=None,
            conditioner=None,
            corpus=None,
            model_config=None,
        )
