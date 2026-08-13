from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from spritelab.captions import SpriteGenerationRequest
from spritelab.inference import (
    CheckpointInferenceConfig,
    run_checkpoint_inference,
)
from spritelab.models.conditioning import SpriteConditionEncoder
from spritelab.models.config import ConditioningSchema, PixelDiTConfig
from spritelab.models.pixeldit import FactorizedSpriteDiT
from spritelab.storage import HashMismatch

torch = pytest.importorskip("torch")


class _RecordingDiskGuard:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def require_capacity(self, additional_bytes: int = 0, *, label: str = "write") -> None:
        self.calls.append((additional_bytes, label))


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _checkpoint_bundle(tmp_path: Path) -> tuple[Path, str, Path, str]:
    schema = replace(ConditioningSchema(), phase_bins=2)
    model_config = PixelDiTConfig(
        height=4,
        width=4,
        num_frames=2,
        patch_size=2,
        model_dim=8,
        depth=1,
        num_heads=2,
        condition_dim=8,
        phase_harmonics=2,
        conditioning=schema,
    )
    denoiser = FactorizedSpriteDiT(model_config)
    encoder = SpriteConditionEncoder(schema, condition_dim=8, max_text_bytes=8)
    generator = torch.Generator(device="cpu").manual_seed(71)
    with torch.no_grad():
        for parameter in tuple(denoiser.parameters()) + tuple(encoder.parameters()):
            parameter.copy_(
                torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
                * 0.15
            )
    training_config = {
        "target_bucket": 4,
        "target_frames": 2,
        "patch_size": 2,
        "model_dim": 8,
        "depth": 1,
        "num_heads": 2,
        "condition_dim": 8,
        "max_text_bytes": 8,
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "foreground_weight": 2.0,
        "matched_endpoint_weight": 1.0,
        "steps": 1,
        "log_every": 1,
        "sample_steps": 2,
        "seed": 5,
        "device": "cpu",
        "precision": "float32",
    }
    checkpoint_payload = {
        "condition_encoder": encoder.state_dict(),
        "config": training_config,
        "denoiser": denoiser.state_dict(),
        "model_config": asdict(model_config),
        "runtime": {
            "cuda_version": getattr(torch.version, "cuda", None),
            "cudnn_version": None,
            "deterministic_algorithms_enabled": False,
            "device": "cpu",
            "device_name": None,
            "torch_version": str(torch.__version__),
        },
        "step": 1,
    }
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(checkpoint_payload, checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    report_payload = {
        "artifact_kind": "tiny_corpus_memorization_diagnostic",
        "checkpoint": {
            "file_sha256": checkpoint_sha256,
            "path": checkpoint.name,
        },
        "config": training_config,
    }
    report = tmp_path / "overfit-report.json"
    report.write_bytes(_canonical_bytes(report_payload))
    return (
        checkpoint,
        checkpoint_sha256,
        report,
        hashlib.sha256(report.read_bytes()).hexdigest(),
    )


def _requests() -> tuple[SpriteGenerationRequest, SpriteGenerationRequest]:
    common = {
        "description": "  red square adventurer  ",
        "entity_class": "object",
        "view": "unknown",
        "direction": "unknown",
        "loop_mode": "loop",
    }
    return (
        SpriteGenerationRequest(action="idle", **common),
        SpriteGenerationRequest(action="run", **common),
    )


def test_same_seed_replays_exactly_and_shared_noise_exposes_action_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, checkpoint_sha, source_report, source_report_sha = _checkpoint_bundle(tmp_path)
    load_calls: list[dict[str, object]] = []
    original_load = torch.load

    def recording_load(*args: object, **kwargs: object) -> object:
        load_calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    inference = CheckpointInferenceConfig(
        seed=90210,
        sample_steps=3,
        noise_strategy="shared",
        device="cpu",
    )
    phases = ((0.0, 0.5), (0.0, 0.5))
    guard = _RecordingDiskGuard()
    first = run_checkpoint_inference(
        checkpoint,
        tmp_path / "first",
        _requests(),
        phases,
        expected_checkpoint_sha256=checkpoint_sha,
        source_report_path=source_report,
        expected_source_report_sha256=source_report_sha,
        config=inference,
        disk_guard=guard,
    )
    second = run_checkpoint_inference(
        checkpoint,
        tmp_path / "second",
        _requests(),
        phases,
        expected_checkpoint_sha256=checkpoint_sha,
        source_report_path=source_report,
        expected_source_report_sha256=source_report_sha,
        config=inference,
    )

    assert all(call["weights_only"] is True for call in load_calls)
    assert all(call["map_location"] == "cpu" for call in load_calls)
    assert first.noise_sha256 == second.noise_sha256
    assert first.report_sha256 == second.report_sha256
    assert {label for _, label in guard.calls} == {
        "checkpoint inference outputs",
        "checkpoint inference report",
        "checkpoint inference sample",
    }
    first_arrays = tuple(np.load(path, allow_pickle=False) for path in first.sample_paths)
    second_arrays = tuple(np.load(path, allow_pickle=False) for path in second.sample_paths)
    assert all(
        np.array_equal(left, right) for left, right in zip(first_arrays, second_arrays, strict=True)
    )
    assert not np.array_equal(first_arrays[0], first_arrays[1])

    report = json.loads(first.report_path.read_text(encoding="utf-8"))
    assert report["checkpoint"]["file_sha256"] == checkpoint_sha
    assert report["source_report"]["file_sha256"] == source_report_sha
    assert report["rng"]["noise_strategy"] == "shared"
    assert report["checkpoint"]["stored_training_config"]["alpha_channel_weight"] == 1.0
    assert report["rng"]["noise_row_sha256"][0] == report["rng"]["noise_row_sha256"][1]
    assert report["samples"][0]["request"]["description"] == "  red square adventurer  "
    assert report["samples"][0]["model_text_prompt"] == "red square adventurer"
    assert report["samples"][0]["structured_labels"]["action"] == "idle"
    assert report["samples"][1]["structured_labels"]["action"] == "run"
    assert report["samples"][0]["dtype"] == "uint8"
    assert report["samples"][0]["shape"] == [2, 4, 4, 4]
    assert (
        report["samples"][0]["file_sha256"]
        == hashlib.sha256(first.sample_paths[0].read_bytes()).hexdigest()
    )
    assert "not evidence" in report["claim_scope"]["limit"]


def test_endpoint_sampler_matches_one_step_euler_and_is_reported(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha, _, _ = _checkpoint_bundle(tmp_path)
    common = {
        "checkpoint_path": checkpoint,
        "requests": (_requests()[0],),
        "frame_phases": ((0.0, 0.5),),
        "expected_checkpoint_sha256": checkpoint_sha,
    }
    endpoint = run_checkpoint_inference(
        **common,
        output_directory=tmp_path / "endpoint",
        config=CheckpointInferenceConfig(
            seed=41,
            sample_steps=1,
            sampler_algorithm="endpoint",
        ),
    )
    euler = run_checkpoint_inference(
        **common,
        output_directory=tmp_path / "euler",
        config=CheckpointInferenceConfig(seed=41, sample_steps=1),
    )

    assert np.array_equal(
        np.load(endpoint.sample_paths[0], allow_pickle=False),
        np.load(euler.sample_paths[0], allow_pickle=False),
    )
    report = json.loads(endpoint.report_path.read_text(encoding="utf-8"))
    assert report["sampler"]["algorithm"] == "direct_t1_endpoint_velocity"


def test_endpoint_sampler_requires_one_step() -> None:
    with pytest.raises(ValueError, match="requires sample_steps=1"):
        CheckpointInferenceConfig(
            seed=1,
            sample_steps=2,
            sampler_algorithm="endpoint",
        )


def test_corrupt_checkpoint_hash_is_rejected_before_loading(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha, _, _ = _checkpoint_bundle(tmp_path)
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")

    with pytest.raises(HashMismatch, match="Checkpoint SHA-256 mismatch"):
        run_checkpoint_inference(
            checkpoint,
            tmp_path / "output",
            (_requests()[0],),
            ((0.0, 0.5),),
            expected_checkpoint_sha256=checkpoint_sha,
            config=CheckpointInferenceConfig(seed=1, sample_steps=1),
        )


def test_phase_shapes_and_structured_labels_are_validated(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha, _, _ = _checkpoint_bundle(tmp_path)
    config = CheckpointInferenceConfig(seed=1, sample_steps=1)

    with pytest.raises(ValueError, match="frame_phases length"):
        run_checkpoint_inference(
            checkpoint,
            tmp_path / "bad-shape",
            (_requests()[0],),
            ((0.0,),),
            expected_checkpoint_sha256=checkpoint_sha,
            config=config,
        )

    unknown_action = SpriteGenerationRequest(
        description="red square",
        entity_class="object",
        action="teleport_sideways",
        loop_mode="loop",
    )
    with pytest.raises(ValueError, match="unknown action"):
        run_checkpoint_inference(
            checkpoint,
            tmp_path / "bad-label",
            (unknown_action,),
            ((0.0, 0.5),),
            expected_checkpoint_sha256=checkpoint_sha,
            config=config,
        )

    with pytest.raises(ValueError, match="one row per request"):
        run_checkpoint_inference(
            checkpoint,
            tmp_path / "bad-batch",
            _requests(),
            ((0.0, 0.5),),
            expected_checkpoint_sha256=checkpoint_sha,
            config=config,
        )


def test_inference_refuses_to_clobber_any_existing_artifact(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha, _, _ = _checkpoint_bundle(tmp_path)
    destination = tmp_path / "output"
    arguments = {
        "checkpoint_path": checkpoint,
        "output_directory": destination,
        "requests": (_requests()[0],),
        "frame_phases": ((0.0, 0.5),),
        "expected_checkpoint_sha256": checkpoint_sha,
        "config": CheckpointInferenceConfig(seed=8, sample_steps=1),
    }
    run_checkpoint_inference(**arguments)

    with pytest.raises(FileExistsError, match="Refusing to replace"):
        run_checkpoint_inference(**arguments)


def test_source_report_hash_and_checkpoint_link_are_verified(tmp_path: Path) -> None:
    checkpoint, checkpoint_sha, source_report, source_report_sha = _checkpoint_bundle(tmp_path)
    common = {
        "checkpoint_path": checkpoint,
        "output_directory": tmp_path / "output",
        "requests": (_requests()[0],),
        "frame_phases": ((0.0, 0.5),),
        "expected_checkpoint_sha256": checkpoint_sha,
        "config": CheckpointInferenceConfig(seed=1, sample_steps=1),
        "source_report_path": source_report,
    }
    with pytest.raises(HashMismatch, match="Source report SHA-256 mismatch"):
        run_checkpoint_inference(
            **common,
            expected_source_report_sha256="0" * 64,
        )

    payload = json.loads(source_report.read_text(encoding="utf-8"))
    payload["checkpoint"]["file_sha256"] = "1" * 64
    source_report.write_bytes(_canonical_bytes(payload))
    new_source_sha = hashlib.sha256(source_report.read_bytes()).hexdigest()
    assert new_source_sha != source_report_sha
    with pytest.raises(HashMismatch, match="different checkpoint"):
        run_checkpoint_inference(
            **common,
            expected_source_report_sha256=new_source_sha,
        )
