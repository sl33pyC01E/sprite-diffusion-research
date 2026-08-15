from __future__ import annotations

from dataclasses import asdict

import pytest

from scripts.run_mugen_latent_motion_refinement_v1 import refinement_config
from spritelab.latent_motion_train import (
    LatentMotionTrainingConfig,
    LatentMotionTrainingError,
    LatentMotionTrainingRow,
    _balanced_pairs_from_index,
    _checkpoint_action_vocabulary,
    _config_from_dict,
    _ema_checkpoint_artifact_kind,
    _ema_update,
    _matched_action_contrast_loss,
    _paired_action_metrics,
    _paired_appearance_metrics,
    _paired_temporal_motion_metrics,
    _sample_motion_residual,
    _sample_target_distinct_pair,
    _sample_training_times,
    _target_distinct_pairs_from_index,
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


def test_target_distinct_pairs_exclude_action_aliases() -> None:
    torch = pytest.importorskip("torch")
    index = {"a": {"idle": 0, "run": 1, "walk": 2}}
    pairs = _target_distinct_pairs_from_index(index, {0: "idle", 1: "move", 2: "move"})

    assert pairs == {"a": ((0, 1), (0, 2))}
    sampled = _sample_target_distinct_pair(
        pairs, generator=torch.Generator(device="cpu").manual_seed(3)
    )
    assert sampled in pairs["a"]


def test_config_requires_a_denoising_objective() -> None:
    with pytest.raises(ValueError, match="at least one denoising objective"):
        LatentMotionTrainingConfig(latent_endpoint_weight=0, pixel_endpoint_weight=0)


def test_config_rejects_negative_action_contrast_weight() -> None:
    with pytest.raises(ValueError, match="action_contrast_weight"):
        LatentMotionTrainingConfig(action_contrast_weight=-0.01)


def test_flow_config_requires_a_multistep_sampler() -> None:
    with pytest.raises(ValueError, match="at least two inference steps"):
        LatentMotionTrainingConfig(time_sampling="uniform", inference_steps=1)

    config = LatentMotionTrainingConfig(
        time_sampling="uniform",
        endpoint_sample_probability=0.25,
        inference_steps=8,
        sampler_algorithm="heun",
    )

    assert config.inference_steps == 8


def test_default_is_corpus_scale_mixed_flow_not_endpoint_memorization() -> None:
    config = LatentMotionTrainingConfig()

    assert config.time_sampling == "uniform"
    assert config.endpoint_sample_probability == pytest.approx(0.25)
    assert config.inference_steps == 16
    assert config.sampler_algorithm == "heun"
    assert config.model.model_dim == 384
    assert config.model.depth == 12


def test_endpoint_refinement_control_differs_only_in_action_weight() -> None:
    control = refinement_config("endpoint-control3000")
    action = refinement_config("endpoint-action3000")

    assert control.action_contrast_weight == pytest.approx(0)
    assert action.action_contrast_weight == pytest.approx(1)
    assert asdict(control) == {
        **asdict(action),
        "action_contrast_weight": 0.0,
    }
    assert action.time_sampling == "endpoint"
    assert action.pixel_endpoint_weight == pytest.approx(2)


def test_endpoint_refinement_rejects_unknown_profile() -> None:
    with pytest.raises(ValueError, match="unsupported refinement profile"):
        refinement_config("not-a-profile")


def test_training_times_are_shared_across_a_matched_pair() -> None:
    torch = pytest.importorskip("torch")
    config = LatentMotionTrainingConfig(
        time_sampling="uniform",
        endpoint_sample_probability=0,
        inference_steps=2,
    )

    times = _sample_training_times(
        torch,
        batch=2,
        config=config,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(11),
    )

    assert times.shape == (2,)
    assert times[0].item() == pytest.approx(times[1].item())
    assert 0 <= times[0].item() < 1


@pytest.mark.parametrize("algorithm", ["euler", "heun"])
def test_rectified_flow_sampler_recovers_a_constant_velocity_path(algorithm: str) -> None:
    torch = pytest.importorskip("torch")
    noise = torch.ones((2, 1, 1, 1, 1))

    class ConstantVelocity:
        def __call__(self, state, reference, times, actions, *, frame_phase):
            del reference, times, actions, frame_phase
            return torch.ones_like(state)

    result = _sample_motion_residual(
        torch,
        ConstantVelocity(),
        noise=noise,
        reference=torch.zeros((2, 1, 1, 1)),
        actions=torch.zeros((2,), dtype=torch.long),
        phases=torch.zeros((2, 1)),
        inference_steps=4,
        sampler_algorithm=algorithm,
    )

    assert torch.allclose(result, torch.zeros_like(noise))


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


def test_matched_action_contrast_loss_matches_authored_delta() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((2, 1, 1, 2, 2))
    target[1] = 1

    exact = _matched_action_contrast_loss(
        torch,
        estimated_clean=target,
        target_clean=target,
    )
    collapsed = _matched_action_contrast_loss(
        torch,
        estimated_clean=torch.zeros_like(target),
        target_clean=target,
    )

    assert exact.item() == pytest.approx(0)
    assert collapsed.item() == pytest.approx(1)


def test_matched_action_contrast_loss_rejects_shape_mismatch() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((2, 1, 1, 2, 2))

    with pytest.raises(ValueError, match="share one shape"):
        _matched_action_contrast_loss(
            torch,
            estimated_clean=target[:1],
            target_clean=target,
        )


def test_paired_appearance_metrics_separate_sprite_and_canvas_error() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((2, 1, 4, 2, 2))
    target[:, :, :, 0, 0] = 1
    predicted = target.clone()
    predicted[:, :, :, 0, 0] = 0.5
    predicted[:, :, :, 1, 1] = 0.25

    metrics = _paired_appearance_metrics(
        torch,
        predicted_pm=predicted,
        target_pm=target,
    )

    assert metrics["foreground_premultiplied_rgba_mae"].item() == pytest.approx(0.5)
    assert metrics["background_premultiplied_rgba_mae"].item() == pytest.approx(1 / 12)
    assert metrics["alpha_iou_127"].item() == pytest.approx(1)
    assert metrics["alpha_precision_127"].item() == pytest.approx(1)
    assert metrics["alpha_recall_127"].item() == pytest.approx(1)
    assert metrics["foreground_occupancy_ratio"].item() == pytest.approx(1)


def test_paired_appearance_metrics_reject_shape_mismatch() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((2, 1, 4, 2, 2))

    with pytest.raises(ValueError, match="share one shape"):
        _paired_appearance_metrics(
            torch,
            predicted_pm=target[:1],
            target_pm=target,
        )


def test_paired_temporal_motion_metrics_expose_static_collapse() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((2, 3, 4, 2, 2))
    target[:, 1] = 1
    predicted = target * 0.25

    metrics = _paired_temporal_motion_metrics(
        torch,
        predicted_pm=predicted,
        target_pm=target,
    )

    assert metrics["target_temporal_magnitude"].item() == pytest.approx(1)
    assert metrics["generated_temporal_magnitude"].item() == pytest.approx(0.25)
    assert "temporal_motion_ratio" not in metrics


def test_paired_temporal_motion_metrics_reject_shape_mismatch() -> None:
    torch = pytest.importorskip("torch")
    target = torch.zeros((2, 2, 4, 2, 2))

    with pytest.raises(ValueError, match="share one shape"):
        _paired_temporal_motion_metrics(
            torch,
            predicted_pm=target[:1],
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


@pytest.mark.parametrize(
    "artifact_kind",
    (
        "mugen_reference_latent_motion_ema_inference_checkpoint",
        "mugen_reference_latent_motion_resume_checkpoint",
    ),
)
def test_evaluator_accepts_inference_and_resume_ema_checkpoints(
    artifact_kind: str,
) -> None:
    assert _ema_checkpoint_artifact_kind({"artifact_kind": artifact_kind}) == artifact_kind


def test_evaluator_rejects_unknown_checkpoint_kind() -> None:
    with pytest.raises(LatentMotionTrainingError, match="wrong artifact kind"):
        _ema_checkpoint_artifact_kind({"artifact_kind": "untrusted"})


def test_resume_checkpoint_uses_corpus_bound_action_vocabulary() -> None:
    checkpoint = {"corpus": {"action_vocabulary": ["idle", "walk"]}}

    assert _checkpoint_action_vocabulary(
        checkpoint, "mugen_reference_latent_motion_resume_checkpoint"
    ) == ["idle", "walk"]


def test_inference_checkpoint_requires_top_level_action_vocabulary() -> None:
    checkpoint = {"corpus": {"action_vocabulary": ["idle", "walk"]}}

    assert (
        _checkpoint_action_vocabulary(
            checkpoint, "mugen_reference_latent_motion_ema_inference_checkpoint"
        )
        is None
    )
