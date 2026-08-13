from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import spritelab.autoencoder_train as training  # noqa: E402
from spritelab.autoencoder_train import (  # noqa: E402
    SpriteAutoencoderTrainingConfig,
    identity_action_frame_index,
    run_autoencoder_training,
    sample_balanced_frames,
    validation_frame_plan,
)
from spritelab.broad_train import PreparedBroadCorpus, PreparedBroadRow  # noqa: E402
from spritelab.models.sprite_autoencoder import SpriteAutoencoderConfig  # noqa: E402


def _row(identity: str, action: str, seed: int, *, split: str = "train") -> PreparedBroadRow:
    generator = np.random.default_rng(seed)
    rgba = np.zeros((2, 16, 16, 4), dtype=np.uint8)
    rgba[:, 4:12, 4:12, :3] = generator.integers(0, 256, size=(2, 8, 8, 3), dtype=np.uint8)
    rgba[:, 4:12, 4:12, 3] = 255
    return PreparedBroadRow(
        sequence_id=f"sequence-{identity}-{action}",
        identity_id=identity,
        action=action,
        split=split,
        request=SimpleNamespace(description=identity),
        rgba=rgba,
        frame_phases=(0.0, 0.5),
        source_size=(16, 16),
        source_file_sha256=f"{seed:064x}",
        normalized_array_sha256=f"{seed + 10:064x}",
    )


def _corpus() -> PreparedBroadCorpus:
    return PreparedBroadCorpus(
        train=(_row("a", "idle", 1), _row("a", "run", 2), _row("b", "idle", 3)),
        validation=(
            _row("c", "idle", 4, split="validation"),
            _row("d", "run", 5, split="validation"),
        ),
        materialization_manifest_sha256="1" * 64,
        source_snapshot_canonical_sha256="2" * 64,
        source_snapshot_manifest_sha256="3" * 64,
        corpus_sha256="4" * 64,
        spatial_transform="fixture",
    )


def test_frame_plans_are_stable_and_cover_valid_frames() -> None:
    corpus = _corpus()
    index = identity_action_frame_index(corpus.train)
    first = sample_balanced_frames(
        corpus.train, index, batch_size=20, generator=np.random.default_rng(8)
    )
    second = sample_balanced_frames(
        corpus.train, index, batch_size=20, generator=np.random.default_rng(8)
    )

    assert first == second
    assert all(0 <= row < len(corpus.train) for row, _ in first)
    assert all(0 <= frame < corpus.train[row].rgba.shape[0] for row, frame in first)
    assert validation_frame_plan(corpus.validation, maximum_frames=3) == validation_frame_plan(
        corpus.validation, maximum_frames=3
    )


def test_tiny_cpu_training_publishes_hash_bound_bundle(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus()
    monkeypatch.setattr(training, "prepare_broad_corpus", lambda *args, **kwargs: corpus)
    config = SpriteAutoencoderTrainingConfig(
        architecture=SpriteAutoencoderConfig(
            image_size=16,
            base_channels=8,
            latent_channels=4,
            channel_multipliers=(1, 2),
            residual_blocks=1,
        ),
        batch_size=2,
        checkpoint_every=2,
        device="cpu",
        horizontal_flip_probability=0,
        log_every=1,
        precision="float32",
        steps=2,
        validate_every=1,
        validation_frames=2,
        warmup_steps=0,
    )
    result = run_autoencoder_training("fixture.json", tmp_path / "run", config=config)

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["artifact_kind"] == "sprite_rgba_autoencoder_training"
    assert report["corpus"]["corpus_sha256"] == "4" * 64
    assert report["final_validation"]["frames"] == 2
    assert report["latent_contract"] == {
        "channels": 4,
        "continuous": True,
        "downsample_factor": 2,
        "height": 8,
        "width": 8,
    }
    assert result.checkpoint_path.is_file()
    assert len(result.checkpoint_sha256) == len(result.report_sha256) == 64
    with pytest.raises(FileExistsError, match="replace"):
        run_autoencoder_training("fixture.json", tmp_path / "run", config=config)
