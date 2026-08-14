from __future__ import annotations

from dataclasses import asdict

import pytest

from spritelab.latent_motion_train import (
    LatentMotionTrainingConfig,
    LatentMotionTrainingRow,
    _config_from_dict,
    _ema_update,
    build_matched_action_index,
    sample_matched_action_pair,
)


def _row(identity: str, verb: str, action_index: int) -> LatentMotionTrainingRow:
    return LatentMotionTrainingRow(
        sequence_id=f"{identity}-{verb}",
        identity_id=identity,
        verb=verb,
        action_index=action_index,
        split="train",
        duration_ms=(100.0,) * 8,
        loop_mode="loop",
    )


def test_matched_index_excludes_single_action_identities() -> None:
    rows = (
        _row("a", "idle", 0),
        _row("a", "walk", 1),
        _row("b", "idle", 0),
    )

    index = build_matched_action_index(rows, (0, 1, 2))

    assert index == {"a": {"idle": 0, "walk": 1}}


def test_matched_sampler_returns_same_identity_distinct_actions() -> None:
    torch = pytest.importorskip("torch")
    rows = (_row("a", "idle", 0), _row("a", "walk", 1))
    index = build_matched_action_index(rows, (0, 1))

    pair = sample_matched_action_pair(index, generator=torch.Generator(device="cpu").manual_seed(7))

    assert {rows[index].verb for index in pair} == {"idle", "walk"}
    assert rows[pair[0]].identity_id == rows[pair[1]].identity_id


def test_config_requires_an_endpoint_objective() -> None:
    with pytest.raises(ValueError, match="at least one endpoint objective"):
        LatentMotionTrainingConfig(latent_endpoint_weight=0, pixel_endpoint_weight=0)


def test_checkpoint_config_round_trips_nested_model() -> None:
    config = LatentMotionTrainingConfig()

    reconstructed = _config_from_dict(asdict(config))

    assert reconstructed == config


def test_zero_decay_ema_copies_raw_weights_during_warmup() -> None:
    torch = pytest.importorskip("torch")
    raw = torch.nn.Linear(2, 2)
    ema = torch.nn.Linear(2, 2)
    with torch.no_grad():
        raw.weight.fill_(2)
        raw.bias.fill_(3)
        ema.weight.zero_()
        ema.bias.zero_()

    _ema_update(torch, ema, raw, 0.0)

    assert torch.equal(ema.weight, raw.weight)
    assert torch.equal(ema.bias, raw.bias)
