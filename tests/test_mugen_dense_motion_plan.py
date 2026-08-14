from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from spritelab.latent_motion_train import load_latent_motion_training_corpus
from spritelab.mugen_dense_motion_plan import (
    build_mugen_dense_motion_plan,
    build_mugen_dense_motion_training_manifest,
)


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    sequences = []
    latents = []
    for identity_index, split in enumerate(("train", "validation", "test")):
        for action_index, action in enumerate(
            ("idle", "walk", "jump", "block", "attack_a", "attack_b")
        ):
            sequence_id = f"sequence-{identity_index}-{action}"
            pixel_path = tmp_path / f"pixels-{identity_index}-{action}.npy"
            pixel_value = np.zeros((8, 128, 128, 4), dtype=np.uint8)
            pixel_value[:, 20:28, 30 + action_index : 38 + action_index] = (
                identity_index + 1,
                action_index + 1,
                10,
                255,
            )
            np.save(pixel_path, pixel_value, allow_pickle=False)
            latent_value = np.full(
                (8, 8, 64, 64), identity_index * 10 + action_index, dtype=np.float16
            )
            latent_path = tmp_path / f"latent-{identity_index}-{action}.npy"
            np.save(latent_path, latent_value, allow_pickle=False)
            pixel_file_sha = hashlib.sha256(pixel_path.read_bytes()).hexdigest()
            pixel_array_sha = _array_sha256(pixel_value)
            sequences.append(
                {
                    "action": action,
                    "caption": {
                        "reference_frame_array_content_sha256": f"{identity_index + 201:064x}",
                        "reference_frame_index": 0,
                    },
                    "direction": "unknown",
                    "entity_class": "humanoid",
                    "identity_id": f"identity-{identity_index}",
                    "loop_mode": "loop" if action in {"idle", "walk"} else "one_shot",
                    "output": {
                        "array_content_sha256": pixel_array_sha,
                        "file_sha256": pixel_file_sha,
                        "relative_path": pixel_path.name,
                        "shape": [8, 128, 128, 4],
                    },
                    "sequence_id": sequence_id,
                    "split": split,
                    "timing": {
                        "duration_ms": [125.0] * 8,
                        "phase": (
                            [index / 8 for index in range(8)]
                            if action in {"idle", "walk"}
                            else [index / 7 for index in range(8)]
                        ),
                    },
                    "view": "side",
                }
            )
            latents.append(
                {
                    "array_content_sha256": _array_sha256(latent_value),
                    "dtype": "float16",
                    "file_sha256": hashlib.sha256(latent_path.read_bytes()).hexdigest(),
                    "identity_id": f"identity-{identity_index}",
                    "relative_path": latent_path.name,
                    "sequence_id": sequence_id,
                    "shape": [8, 8, 64, 64],
                    "source": {
                        "array_content_sha256": pixel_array_sha,
                        "file_sha256": pixel_file_sha,
                        "relative_path": pixel_path.name,
                    },
                    "split": split,
                }
            )
    materialization = {
        "artifact_kind": "mugen_dense_captioned_materialization_bridge",
        "sequence_count": len(sequences),
        "sequences": sequences,
    }
    materialization_path = tmp_path / "materialization.json"
    materialization_path.write_text(json.dumps(materialization), encoding="utf-8")
    checkpoint_path = tmp_path / "autoencoder.pt"
    checkpoint_path.write_bytes(b"fixture checkpoint identity")
    latent = {
        "artifact_kind": "mugen_frozen_rgba_autoencoder_latent_cache",
        "codec": {
            "architecture": {
                "image_size": 128,
                "latent_channels": 8,
            },
            "checkpoint_file_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "checkpoint_path": str(checkpoint_path),
        },
        "normalization": {
            "channel_mean": [0.0] * 8,
            "channel_standard_deviation": [1.0] * 8,
        },
        "record_count": len(latents),
        "records": latents,
        "source": {
            "materialization_file_sha256": hashlib.sha256(
                materialization_path.read_bytes()
            ).hexdigest(),
            "materialization_path": str(materialization_path),
        },
    }
    latent_path = tmp_path / "latents.json"
    latent_path.write_text(json.dumps(latent), encoding="utf-8")
    return materialization_path, latent_path


def test_dense_motion_plan_preserves_six_actions_and_idle_reference(tmp_path: Path) -> None:
    materialization, latent = _fixture(tmp_path)

    plan = build_mugen_dense_motion_plan(materialization, latent)

    assert plan["counts"]["sequences"] == 18
    assert set(plan["counts"]["actions"]) == {
        "idle",
        "walk",
        "jump",
        "block",
        "attack_a",
        "attack_b",
    }
    attack = next(row for row in plan["records"] if row["sequence_id"] == "sequence-0-attack_a")
    assert attack["reference"]["sequence_id"] == "sequence-0-idle"
    assert attack["target"]["phase"][-1] == 1.0
    assert len(attack["reference"]["latent"]["frame_array_content_sha256"]) == 64
    assert plan["source"]["latent_manifest"]["scope"]["unused_latent_sequences"] == 0

    plan_path = tmp_path / "motion-plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    training = build_mugen_dense_motion_training_manifest(plan_path)
    assert training["counts"]["sequences"] == 18
    assert training["config"]["one_sequence_per_identity_verb"] is True

    training_path = tmp_path / "training.json"
    training_path.write_text(json.dumps(training), encoding="utf-8")
    corpus = load_latent_motion_training_corpus(training_path, verify_hashes=True)
    assert corpus.target_latents.shape == (18, 8, 8, 64, 64)
    assert corpus.reference_latents.shape == (18, 8, 64, 64)
    assert set(corpus.action_vocabulary) == {
        "idle",
        "walk",
        "jump",
        "block",
        "attack_a",
        "attack_b",
    }


def test_dense_motion_plan_accepts_verified_broad_latent_superset(tmp_path: Path) -> None:
    broad_materialization, latent = _fixture(tmp_path)
    broad = json.loads(broad_materialization.read_bytes())
    selected = [row for row in broad["sequences"] if row["identity_id"] != "identity-2"]
    subset = {
        **broad,
        "sequence_count": len(selected),
        "sequences": selected,
    }
    subset_path = tmp_path / "dense-subset.json"
    subset_path.write_text(json.dumps(subset), encoding="utf-8")

    plan = build_mugen_dense_motion_plan(subset_path, latent)

    assert plan["counts"]["sequences"] == 12
    scope = plan["source"]["latent_manifest"]["scope"]
    assert scope["joined_sequences"] == 12
    assert scope["unused_latent_sequences"] == 6
    assert scope["source_materialization_path"] == str(broad_materialization.resolve())
