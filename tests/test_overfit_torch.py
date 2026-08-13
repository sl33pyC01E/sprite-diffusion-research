from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import spritelab.overfit as overfit_module
from spritelab.models.conditioning import EncodedConditionBatch
from spritelab.overfit import (
    OverfitContinuationContractError,
    TinyOverfitConfig,
    _build_endpoint_contrast_plan,
    _identity_grouped_noise,
    _permute_actions_for_endpoint_plan,
    _slice_encoded_conditions,
    continue_tiny_overfit,
    run_tiny_overfit,
)
from spritelab.storage import HashMismatch

torch = pytest.importorskip("torch")


def _array_sha256(array: np.ndarray) -> str:
    header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    array = np.zeros((2, 8, 8, 4), dtype=np.uint8)
    array[:, 2:6, 2:6, 0] = 255
    array[:, 2:6, 2:6, 3] = 255
    clip = tmp_path / "clips/train/red.npy"
    clip.parent.mkdir(parents=True, exist_ok=True)
    with clip.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    payload = {
        "schema_version": 1,
        "sequence_count": 1,
        "sequences": [
            {
                "action": "idle",
                "caption": {"description": "red square"},
                "direction": "unknown",
                "entity_class": "object",
                "frame_count": 2,
                "identity_id": "red-square",
                "loop_mode": "loop",
                "output": {
                    "array_content_sha256": _array_sha256(array),
                    "dtype": "uint8",
                    "file_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
                    "format": "numpy_npy_v1",
                    "relative_path": "clips/train/red.npy",
                    "shape": list(array.shape),
                    "size_bytes": clip.stat().st_size,
                },
                "provenance": {
                    "source_blob_sha256": ["a" * 64],
                    "source_id": "fixture",
                },
                "quality_tier": "F0",
                "sequence_id": "red-idle",
                "split": "train",
                "target_bucket": [8, 8],
                "timing": {"duration_ms": [100, 100], "phase": [0.0, 0.5]},
                "view": "unknown",
            }
        ],
        "source_snapshot": {
            "canonical_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "schema_version": 1,
        },
    }
    path = tmp_path / "materialization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _multi_action_manifest(tmp_path: Path) -> Path:
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    second = deepcopy(payload["sequences"][0])
    array = np.zeros((2, 8, 8, 4), dtype=np.uint8)
    array[:, 1:7, 3:5, 1] = 255
    array[:, 1:7, 3:5, 3] = 255
    clip = tmp_path / "clips/train/red-run.npy"
    with clip.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    second["action"] = "run"
    second["caption"]["description"] = "red square"
    second["sequence_id"] = "red-run"
    second["output"] = {
        "array_content_sha256": _array_sha256(array),
        "dtype": "uint8",
        "file_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "format": "numpy_npy_v1",
        "relative_path": "clips/train/red-run.npy",
        "shape": list(array.shape),
        "size_bytes": clip.stat().st_size,
    }
    payload["sequences"].append(second)
    payload["sequence_count"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _tiny_config(*, steps: int, alpha_channel_weight: float = 1.0) -> TinyOverfitConfig:
    return TinyOverfitConfig(
        target_bucket=8,
        target_frames=2,
        patch_size=2,
        model_dim=16,
        depth=1,
        num_heads=4,
        condition_dim=8,
        max_text_bytes=8,
        alpha_channel_weight=alpha_channel_weight,
        steps=steps,
        log_every=1,
        sample_steps=2,
        device="cpu",
        seed=29,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_nested_equal(left: object, right: object) -> None:
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right)
        assert torch.equal(left, right)
        return
    if isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, list | tuple) or isinstance(right, list | tuple):
        assert isinstance(left, type(right))
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_equal(left_value, right_value)
        return
    assert left == right


def test_tiny_overfit_runs_end_to_end_on_cpu(tmp_path: Path) -> None:
    result = run_tiny_overfit(
        _manifest(tmp_path),
        tmp_path / "output",
        config=TinyOverfitConfig(
            target_bucket=8,
            target_frames=2,
            patch_size=2,
            model_dim=16,
            depth=1,
            num_heads=4,
            condition_dim=8,
            max_text_bytes=8,
            steps=2,
            log_every=1,
            sample_steps=2,
            device="cpu",
            seed=7,
        ),
    )

    assert result.report_path.is_file()
    assert result.checkpoint_path.is_file()
    assert len(result.sample_paths) == 1
    assert result.sample_paths[0].name == f"{hashlib.sha256(b'red-idle').hexdigest()}.npy"
    sample = np.load(result.sample_paths[0], allow_pickle=False)
    assert sample.shape == (2, 8, 8, 4)
    report = json.loads(result.report_path.read_text())
    assert report["artifact_kind"] == "tiny_corpus_memorization_diagnostic"
    assert report["sequence_ids"] == ["red-idle"]
    assert len(report["history"]) == 2
    assert report["initial_loss"] == result.initial_loss
    assert report["final_loss"] == result.final_loss
    assert report["matched_endpoint_final_loss"] is None
    assert report["matched_endpoint_sequence_ids"] == []
    assert report["config"]["alpha_channel_weight"] == 1.0
    assert (
        report["checkpoint"]["file_sha256"]
        == hashlib.sha256(result.checkpoint_path.read_bytes()).hexdigest()
    )
    assert (
        report["materialization_manifest_file_sha256"]
        == hashlib.sha256((tmp_path / "materialization.json").read_bytes()).hexdigest()
    )
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu")
    assert checkpoint["config"]["alpha_channel_weight"] == 1.0
    assert checkpoint["sequence_ids"] == ("red-idle",)
    assert checkpoint["rng_state"]["numpy"]["bit_generator"] == "MT19937"

    with pytest.raises(FileExistsError, match="Refusing"):
        run_tiny_overfit(
            _manifest(tmp_path),
            tmp_path / "output",
            config=TinyOverfitConfig(
                target_bucket=8,
                target_frames=2,
                patch_size=2,
                model_dim=16,
                depth=1,
                num_heads=4,
                condition_dim=8,
                max_text_bytes=8,
                steps=1,
                log_every=1,
                sample_steps=1,
                device="cpu",
            ),
        )


def test_tiny_overfit_preflights_existing_hashed_sample(tmp_path: Path) -> None:
    output = tmp_path / "output"
    existing = output / "samples" / f"{hashlib.sha256(b'red-idle').hexdigest()}.npy"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"preserve me")

    with pytest.raises(FileExistsError, match="existing sample"):
        run_tiny_overfit(
            _manifest(tmp_path),
            output,
            config=TinyOverfitConfig(
                target_bucket=8,
                target_frames=2,
                patch_size=2,
                model_dim=16,
                depth=1,
                num_heads=4,
                condition_dim=8,
                max_text_bytes=8,
                steps=1,
                log_every=1,
                sample_steps=1,
                device="cpu",
            ),
        )

    assert existing.read_bytes() == b"preserve me"


def test_continuation_matches_uninterrupted_cpu_training(tmp_path: Path) -> None:
    manifest = _multi_action_manifest(tmp_path)
    uninterrupted = run_tiny_overfit(
        manifest,
        tmp_path / "uninterrupted",
        config=_tiny_config(steps=4),
    )
    parent = run_tiny_overfit(
        manifest,
        tmp_path / "parent",
        config=_tiny_config(steps=2),
    )
    continued = continue_tiny_overfit(
        manifest,
        parent.checkpoint_path,
        parent.report_path,
        tmp_path / "continued",
        expected_parent_checkpoint_sha256=_file_sha256(parent.checkpoint_path),
        expected_parent_report_sha256=_file_sha256(parent.report_path),
        additional_steps=2,
        config=_tiny_config(steps=4),
    )

    full_checkpoint = torch.load(
        uninterrupted.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    continued_checkpoint = torch.load(
        continued.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    for field in (
        "denoiser",
        "condition_encoder",
        "optimizer",
        "rng_state",
        "diagnostic_tensor_sha256",
    ):
        _assert_nested_equal(full_checkpoint[field], continued_checkpoint[field])
    assert full_checkpoint["step"] == continued_checkpoint["step"] == 4
    assert full_checkpoint["config"] == continued_checkpoint["config"]
    for full_sample, continued_sample in zip(
        uninterrupted.sample_paths,
        continued.sample_paths,
        strict=True,
    ):
        assert full_sample.read_bytes() == continued_sample.read_bytes()

    full_report = json.loads(uninterrupted.report_path.read_text(encoding="utf-8"))
    continued_report = json.loads(continued.report_path.read_text(encoding="utf-8"))
    assert full_report["history"] == continued_report["history"]
    assert full_report["final_loss"] == continued_report["final_loss"]
    assert full_report["final_training_loss"] == continued_report["final_training_loss"]
    assert continued_report["continuation"] == continued_checkpoint["continuation"]
    assert continued_report["continuation"]["parent_checkpoint_sha256"] == _file_sha256(
        parent.checkpoint_path
    )
    assert continued_report["continuation"]["parent_report_sha256"] == _file_sha256(
        parent.report_path
    )
    assert continued_report["continuation"]["parent_step"] == 2
    assert continued_report["continuation"]["additional_steps"] == 2
    assert continued_report["continuation"]["cumulative_step"] == 4


def test_continuation_rejects_zero_hash_and_config_mismatch_without_writes(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path)
    parent = run_tiny_overfit(
        manifest,
        tmp_path / "parent",
        config=_tiny_config(steps=1),
    )
    arguments = {
        "manifest_path": manifest,
        "parent_checkpoint_path": parent.checkpoint_path,
        "parent_report_path": parent.report_path,
        "expected_parent_checkpoint_sha256": _file_sha256(parent.checkpoint_path),
        "expected_parent_report_sha256": _file_sha256(parent.report_path),
    }

    zero_output = tmp_path / "zero"
    with pytest.raises(ValueError, match="additional_steps"):
        continue_tiny_overfit(
            output_directory=zero_output,
            additional_steps=0,
            **arguments,
        )
    assert not zero_output.exists()

    hash_output = tmp_path / "bad-hash"
    with pytest.raises(HashMismatch, match="checkpoint SHA-256"):
        continue_tiny_overfit(
            output_directory=hash_output,
            additional_steps=1,
            expected_parent_checkpoint_sha256="0" * 64,
            **{
                key: value
                for key, value in arguments.items()
                if key != "expected_parent_checkpoint_sha256"
            },
        )
    assert not hash_output.exists()

    config_output = tmp_path / "bad-config"
    with pytest.raises(OverfitContinuationContractError, match="config differs"):
        continue_tiny_overfit(
            output_directory=config_output,
            additional_steps=1,
            config=_tiny_config(steps=2, alpha_channel_weight=4),
            **arguments,
        )
    assert not config_output.exists()


def test_continuation_rejects_sequence_order_and_rolls_back_private_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _multi_action_manifest(tmp_path)
    parent = run_tiny_overfit(
        manifest,
        tmp_path / "parent",
        config=_tiny_config(steps=1),
    )
    arguments = {
        "manifest_path": manifest,
        "parent_checkpoint_path": parent.checkpoint_path,
        "parent_report_path": parent.report_path,
        "expected_parent_checkpoint_sha256": _file_sha256(parent.checkpoint_path),
        "expected_parent_report_sha256": _file_sha256(parent.report_path),
        "additional_steps": 1,
    }

    ordered_output = tmp_path / "wrong-order"
    with pytest.raises(OverfitContinuationContractError, match="ordered parent"):
        continue_tiny_overfit(
            output_directory=ordered_output,
            sequence_ids=("red-run", "red-idle"),
            **arguments,
        )
    assert not ordered_output.exists()

    def fail_after_validation(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("synthetic post-validation failure")

    monkeypatch.setattr(overfit_module, "_run_tiny_overfit", fail_after_validation)
    rollback_output = tmp_path / "rollback"
    with pytest.raises(RuntimeError, match="synthetic"):
        continue_tiny_overfit(
            output_directory=rollback_output,
            **arguments,
        )
    assert not rollback_output.exists()
    assert not tuple(tmp_path.glob(".rollback.*.continuation.tmp"))


def test_tiny_overfit_reports_matched_endpoint_conditioning_loss(tmp_path: Path) -> None:
    result = run_tiny_overfit(
        _multi_action_manifest(tmp_path),
        tmp_path / "output",
        config=TinyOverfitConfig(
            target_bucket=8,
            target_frames=2,
            patch_size=2,
            model_dim=16,
            depth=1,
            num_heads=4,
            condition_dim=8,
            max_text_bytes=8,
            steps=1,
            log_every=1,
            sample_steps=1,
            device="cpu",
            seed=11,
        ),
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["matched_endpoint_sequence_ids"] == ["red-idle", "red-run"]
    assert report["matched_endpoint_initial_loss"] > 0
    assert report["matched_endpoint_final_loss"] > 0
    assert report["matched_endpoint_action_permuted_final_loss"] > 0
    assert "matched_endpoint_loss" in report["history"][0]
    assert len(report["action_swap_matrix"]) == 2
    assert {
        (row["original_action"], row["replacement_action"]) for row in report["action_swap_matrix"]
    } == {("idle", "run"), ("run", "idle")}
    assert report["input_clips"][0]["request"]["action"] in {"idle", "run"}


def test_alpha_channel_weight_is_recorded_in_training_artifacts(tmp_path: Path) -> None:
    result = run_tiny_overfit(
        _manifest(tmp_path),
        tmp_path / "output-alpha",
        config=TinyOverfitConfig(
            target_bucket=8,
            target_frames=2,
            patch_size=2,
            model_dim=16,
            depth=1,
            num_heads=4,
            condition_dim=8,
            max_text_bytes=8,
            alpha_channel_weight=4,
            steps=1,
            log_every=1,
            sample_steps=1,
            device="cpu",
        ),
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert report["config"]["alpha_channel_weight"] == 4
    assert checkpoint["config"]["alpha_channel_weight"] == 4
    assert "multiplier is 4" in report["training_conditioning_contract"]


def test_zero_endpoint_weight_keeps_diagnostic_but_skips_training_forward(tmp_path: Path) -> None:
    result = run_tiny_overfit(
        _multi_action_manifest(tmp_path),
        tmp_path / "output-zero",
        config=TinyOverfitConfig(
            target_bucket=8,
            target_frames=2,
            patch_size=2,
            model_dim=16,
            depth=1,
            num_heads=4,
            condition_dim=8,
            max_text_bytes=8,
            matched_endpoint_weight=0,
            steps=1,
            log_every=1,
            sample_steps=1,
            device="cpu",
            seed=13,
        ),
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["matched_endpoint_initial_loss"] > 0
    assert report["matched_endpoint_final_loss"] > 0
    assert "matched_endpoint_loss" not in report["history"][0]


def test_sample_noise_is_matched_within_identity() -> None:
    clean = torch.zeros((3, 2, 4, 2, 2))
    clips = (
        SimpleNamespace(identity_id="rat"),
        SimpleNamespace(identity_id="cat"),
        SimpleNamespace(identity_id="rat"),
    )

    noise = _identity_grouped_noise(
        torch,
        clean,
        clips,
        generator=torch.Generator().manual_seed(17),
    )

    assert torch.equal(noise[0], noise[2])
    assert not torch.equal(noise[0], noise[1])


def test_endpoint_rows_and_action_permutation_stay_within_identity() -> None:
    def clip(identity: str, action: str, sequence_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            identity_id=identity,
            sequence_id=sequence_id,
            frame_phases=(0.0, 0.5),
            request=SimpleNamespace(
                action=action,
                description=identity,
                entity_class="animal",
                view="side",
                direction="right",
                loop_mode="loop",
            ),
        )

    clips = (
        clip("rat", "idle", "rat-idle"),
        clip("cat", "idle", "cat-idle"),
        clip("rat", "run", "rat-run-a"),
        clip("rat", "run", "rat-run-b"),
    )
    clean = np.zeros((4, 2, 4, 2, 2), dtype=np.float32)
    clean[0] = -1
    clean[2:] = 1
    batch = EncodedConditionBatch(
        descriptions=("rat", "cat", "rat", "rat"),
        text_token_ids=((1,), (2,), (3,), (4,)),
        text_attention_mask=((True,),) * 4,
        entity_ids=(1, 2, 1, 1),
        action_ids=(10, 10, 20, 20),
        view_ids=(0, 0, 0, 0),
        direction_ids=(0, 0, 0, 0),
        loop_mode_ids=(1, 1, 1, 1),
        max_text_bytes=1,
    )

    plan = _build_endpoint_contrast_plan(clips, clean)
    permuted = _permute_actions_for_endpoint_plan(batch, plan)

    assert plan.selected_indices == (0, 2)
    assert len(plan.exclusions) == 1
    assert plan.exclusions[0].sequence_id == "rat-run-b"
    assert plan.exclusions[0].reason == "byte_identical_duplicate_target_uses_one_representative"
    assert permuted is not None
    assert permuted.action_ids == (20, 10, 10, 20)
    sliced = _slice_encoded_conditions(permuted, plan.selected_indices)
    assert sliced.descriptions == ("rat", "rat")
    assert sliced.action_ids == (20, 10)

    clean[3] = 0.5
    conflicted = _build_endpoint_contrast_plan(clips, clean)
    assert conflicted.selected_indices == ()
    assert {row.reason for row in conflicted.exclusions} == {
        "conflicting_targets_for_identical_action_and_non_action_conditions",
        "no_target_distinct_multi_action_contrast_after_conflict_and_alias_filter",
    }


def test_endpoint_plan_excludes_cross_action_pixel_aliases_without_dropping_base_rows() -> None:
    def clip(action: str, sequence_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            identity_id="worker",
            sequence_id=sequence_id,
            frame_phases=(0.0, 0.5),
            request=SimpleNamespace(
                action=action,
                description="worker",
                entity_class="humanoid",
                view="side",
                direction="right",
                loop_mode="loop",
            ),
        )

    clips = (clip("carry", "carry"), clip("walk", "walk"))
    clean = np.zeros((2, 2, 4, 2, 2), dtype=np.float32)

    plan = _build_endpoint_contrast_plan(clips, clean)

    assert len(clips) == 2  # Both rows remain available to ordinary base training.
    assert plan.groups == ()
    assert plan.selected_indices == ()
    assert [(row.sequence_id, row.reason) for row in plan.exclusions] == [
        ("carry", "no_target_distinct_multi_action_contrast_after_conflict_and_alias_filter"),
        ("walk", "byte_identical_cross_action_target_uses_one_representative"),
    ]


def test_endpoint_plan_keeps_one_action_per_target_when_three_actions_include_alias() -> None:
    def clip(action: str) -> SimpleNamespace:
        return SimpleNamespace(
            identity_id="worker",
            sequence_id=action,
            frame_phases=(0.0, 0.5),
            request=SimpleNamespace(
                action=action,
                description="worker",
                entity_class="humanoid",
                view="side",
                direction="right",
                loop_mode="loop",
            ),
        )

    clips = (clip("carry"), clip("haul"), clip("walk"))
    clean = np.zeros((3, 2, 4, 2, 2), dtype=np.float32)
    clean[2] = 1

    plan = _build_endpoint_contrast_plan(clips, clean)

    assert plan.selected_indices == (0, 2)
    assert plan.groups[0].actions == ("carry", "walk")
    assert [(row.sequence_id, row.reason) for row in plan.exclusions] == [
        ("haul", "byte_identical_cross_action_target_uses_one_representative")
    ]
