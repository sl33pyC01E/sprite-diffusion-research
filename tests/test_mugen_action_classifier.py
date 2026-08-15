from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

torch = pytest.importorskip("torch")

from spritelab.latent_motion_train import (  # noqa: E402
    LatentMotionTrainingCorpus,
    LatentMotionTrainingRow,
)
from spritelab.mugen_action_classifier import (  # noqa: E402
    MugenActionClassifierConfig,
    MugenActionClassifierTrainingConfig,
    MugenLatentActionClassifier,
    dense_action_bundles,
    latent_action_batch,
)

ACTIONS = ("attack_a", "attack_b", "block", "idle", "jump", "walk")


def _corpus() -> LatentMotionTrainingCorpus:
    rows = tuple(
        LatentMotionTrainingRow(
            sequence_id=f"{identity}-{action}",
            identity_id=identity,
            verb=action,
            action_index=action_index,
            split="train",
            duration_ms=(100.0,) * 8,
            loop_mode="loop",
        )
        for identity in ("a", "b")
        for action_index, action in enumerate(ACTIONS)
    )
    target = np.zeros((12, 8, 8, 8, 8), dtype=np.float16)
    reference = np.zeros((12, 8, 8, 8), dtype=np.float16)
    for index, row in enumerate(rows):
        target[index, :, row.action_index] = row.action_index + 1
    return LatentMotionTrainingCorpus(
        rows=rows,
        train_indices=tuple(range(12)),
        validation_indices=tuple(range(12)),
        test_indices=tuple(range(12)),
        action_vocabulary=ACTIONS,
        target_latents=target,
        reference_latents=reference,
        target_rgba=np.zeros((12, 8, 16, 16, 4), dtype=np.uint8),
        phases=np.zeros((12, 8), dtype=np.float32),
        channel_mean=(0.0,) * 8,
        channel_standard_deviation=(2.0,) * 8,
        autoencoder_checkpoint_path=Path("codec.pt"),
        autoencoder_architecture={},
        contract={"test": True},
    )


def test_dense_action_bundles_preserve_vocabulary_order() -> None:
    corpus = _corpus()

    bundles = dense_action_bundles(corpus, corpus.train_indices)

    assert bundles == (tuple(range(6)), tuple(range(6, 12)))


def test_latent_action_batch_is_reference_relative_and_normalized() -> None:
    corpus = _corpus()
    bundles = dense_action_bundles(corpus, corpus.train_indices)[:1]

    residual, labels = latent_action_batch(torch, corpus, bundles, device=torch.device("cpu"))

    assert residual.shape == (6, 8, 8, 8, 8)
    assert labels.tolist() == list(range(6))
    assert residual[5, 5].unique().tolist() == [3.0]


def test_action_classifier_emits_logits_and_features() -> None:
    model = MugenLatentActionClassifier(
        MugenActionClassifierConfig(base_channels=8, feature_dim=16)
    )

    logits, features = model(torch.randn(3, 8, 8, 32, 32))

    assert logits.shape == (3, 6)
    assert features.shape == (3, 16)


def test_action_classifier_config_rejects_invalid_training_contract() -> None:
    with pytest.raises(ValueError, match="minimum_learning_rate"):
        MugenActionClassifierTrainingConfig(learning_rate=1e-4, minimum_learning_rate=2e-4)
    with pytest.raises(ValueError, match="device"):
        MugenActionClassifierTrainingConfig(device="spark")
