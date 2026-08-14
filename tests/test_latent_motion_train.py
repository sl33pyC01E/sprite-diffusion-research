from __future__ import annotations

from dataclasses import asdict

import pytest

from spritelab.latent_motion_train import (
    LatentMotionTrainingConfig,
    LatentMotionTrainingError,
    LatentMotionTrainingRow,
    _balanced_pairs_from_index,
    _config_from_dict,
    _ema_update,
    _paired_action_metrics,
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


def test_paired_action_metrics_reward_correct_and_causal_outputs() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((2, 2, 4, 2, 2))
    target[1] = 1
    predicted = target.clone()
    permuted = torch.flip(target, (0,))

    metrics = _paired_action_metrics(
        torch,
        predicted_pm=predicted,
        permuted_pm=permuted,
        target_pm=target,
    )

    assert metrics["action_separation_ratio"].item() == pytest.approx(1)
    assert metrics["action_correct_target_preference_rate"].item() == pytest.approx(1)
    assert metrics["action_swap_moves_toward_replacement_rate"].item() == pytest.approx(1)
    assert metrics["action_correct_target_margin"].item() == pytest.approx(1)


def test_paired_action_metrics_reject_shape_mismatch() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((2, 2, 4, 2, 2))

    with pytest.raises(ValueError, match="share one shape"):
        _paired_action_metrics(
            torch,
            predicted_pm=target[:1],
            permuted_pm=target,
            target_pm=target,
        )


def test_balanced_pairs_cover_identities_then_diversify_verbs() -> None:
    index = {
        "a": {"crouch": 0, "idle": 1, "walk": 2},
        "b": {"crouch": 3, "idle": 4},
        "c": {"run": 5, "walk": 6},
    }

    pairs = _balanced_pairs_from_index(index, 4)

    assert set(pairs[:3]) == {(0, 2), (3, 4), (5, 6)}
    assert pairs[3] == (1, 2)


def test_balanced_pairs_validate_limit_and_empty_index() -> None:
    with pytest.raises(ValueError, match="positive"):
        _balanced_pairs_from_index({}, 0)
    with pytest.raises(LatentMotionTrainingError, match="no matched action pairs"):
        _balanced_pairs_from_index({}, 1)
