from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.latent_still_train import (
    LatentStillRow,
    LatentStillTrainingConfig,
    build_hierarchical_sampler_index,
    load_latent_still_corpus,
    normalize_latents,
    run_latent_still_training,
    sample_hierarchical_batch,
)
from spritelab.models.latent_still_dit import LatentStillDiTConfig

torch = pytest.importorskip("torch")


def _row(
    identity: str,
    verb: str,
    sequence: str,
    eligible: tuple[int, ...] = tuple(range(8)),
) -> LatentStillRow:
    return LatentStillRow(
        sequence_id=sequence,
        identity_id=identity,
        verb=verb,
        split="train",
        prompt="sprite",
        prompt_row=0,
        latent_path=Path("unused.npy"),
        latent_file_sha256="0" * 64,
        latent_array_sha256="1" * 64,
        eligible_frame_indices=eligible,
    )


def test_hierarchical_sampler_balances_identity_then_verb() -> None:
    rows = (
        _row("a", "idle", "a-idle"),
        _row("a", "walk", "a-walk-1"),
        _row("a", "walk", "a-walk-2"),
        _row("b", "attack", "b-attack"),
    )
    index = build_hierarchical_sampler_index(rows, tuple(range(len(rows))))
    generator = torch.Generator(device="cpu").manual_seed(7)
    selection = sample_hierarchical_batch(index, batch_size=2000, generator=generator)
    identities = [rows[row].identity_id for row, _ in selection]
    a_verbs = [rows[row].verb for row, _ in selection if rows[row].identity_id == "a"]

    assert 850 < identities.count("a") < 1150
    assert 400 < a_verbs.count("idle") < 600
    assert all(0 <= frame < 8 for _, frame in selection)


def test_hierarchical_sampler_uses_only_subject_bearing_frames() -> None:
    rows = (_row("a", "attack", "a-attack", (2, 5)),)
    index = build_hierarchical_sampler_index(rows, (0,))
    selection = sample_hierarchical_batch(
        index,
        batch_size=100,
        generator=torch.Generator(device="cpu").manual_seed(8),
        frame_indices_by_row=tuple(row.eligible_frame_indices for row in rows),
    )
    assert {frame for _, frame in selection} == {2, 5}


def test_latent_normalization_is_channelwise_float32() -> None:
    value = np.empty((2, 8, 64, 64), dtype=np.float16)
    for channel in range(8):
        value[:, channel] = channel + 2
    normalized = normalize_latents(
        value,
        tuple(float(channel) for channel in range(8)),
        (2.0,) * 8,
    )
    assert normalized.dtype == np.float32
    assert normalized.shape == value.shape
    assert np.all(normalized == 1)


def test_training_config_validates_schedule_and_small_model() -> None:
    config = LatentStillTrainingConfig(
        steps=4,
        warmup_steps=1,
        checkpoint_every=2,
        validate_every=2,
        log_every=1,
        model=LatentStillDiTConfig(
            latent_size=16,
            latent_channels=4,
            patch_size=2,
            model_dim=32,
            depth=2,
            num_heads=4,
            mlp_ratio=2,
            condition_dim=16,
            window_size=2,
            global_attention_every=2,
        ),
    )
    assert config.steps == 4
    with pytest.raises(ValueError, match="warmup_steps"):
        LatentStillTrainingConfig(steps=4, warmup_steps=4)


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _write_npy(path: Path, value: np.ndarray) -> tuple[str, str]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), _array_sha256(value)


def test_corpus_loader_verifies_plan_latent_and_text_closure(tmp_path) -> None:
    plan_path = tmp_path / "plan.json"
    records = []
    latent_records = []
    prompts = ("red fighter; idle", "blue fighter; walk")
    for index, split in enumerate(("train", "validation")):
        sequence = f"sequence-{index}"
        identity = f"identity-{index}"
        records.append(
            {
                "conditioning": {"verb": ("idle", "walk")[index]},
                "identity_id": identity,
                "prompt": prompts[index],
                "sequence_id": sequence,
                "split": split,
            }
        )
        latent_value = np.full((8, 8, 64, 64), index, dtype=np.float16)
        relative = f"latents/{sequence}.npy"
        file_sha, array_sha = _write_npy(tmp_path / "latents" / relative, latent_value)
        latent_records.append(
            {
                "array_content_sha256": array_sha,
                "file_sha256": file_sha,
                "identity_id": identity,
                "relative_path": relative,
                "sequence_id": sequence,
                "split": split,
            }
        )
    extra_value = np.zeros((8, 8, 64, 64), dtype=np.float16)
    extra_file_sha, extra_array_sha = _write_npy(
        tmp_path / "latents/latents/sequence-extra.npy", extra_value
    )
    latent_records.append(
        {
            "array_content_sha256": extra_array_sha,
            "file_sha256": extra_file_sha,
            "identity_id": "identity-extra",
            "relative_path": "latents/sequence-extra.npy",
            "sequence_id": "sequence-extra",
            "split": "test",
        }
    )
    plan = {
        "artifact_kind": "mugen_latent_still_sequence_training_plan",
        "counts": {"sequences": 2},
        "records": records,
        "source": {"materialization_file_sha256": "a" * 64},
    }
    plan_payload = (json.dumps(plan, sort_keys=True, separators=(",", ":")) + "\n").encode()
    plan_path.write_bytes(plan_payload)
    latent_manifest = {
        "artifact_kind": "mugen_frozen_rgba_autoencoder_latent_cache",
        "normalization": {
            "channel_mean": [0.0] * 8,
            "channel_standard_deviation": [1.0] * 8,
        },
        "record_count": 3,
        "records": latent_records,
        "source": {"materialization_file_sha256": "a" * 64},
    }
    latent_manifest_path = tmp_path / "latents" / "manifest.json"
    latent_manifest_path.write_text(json.dumps(latent_manifest), encoding="utf-8")
    embeddings = np.zeros((2, 77, 768), dtype=np.float16)
    masks = np.ones((2, 77), dtype=np.bool_)
    arrays = {}
    for name, value in (("embeddings", embeddings), ("attention_mask", masks)):
        file_sha, array_sha = _write_npy(tmp_path / "text" / f"{name}.npy", value)
        arrays[name] = {
            "array_content_sha256": array_sha,
            "file_sha256": file_sha,
            "path": f"{name}.npy",
        }
    text_manifest = {
        "arrays": arrays,
        "artifact_kind": "frozen_clip_token_hidden_state_cache",
        "prompt_count": 2,
        "rows": [{"prompt": prompt, "row_index": index} for index, prompt in enumerate(prompts)],
        "source": {"training_plan_file_sha256": hashlib.sha256(plan_payload).hexdigest()},
    }
    text_manifest_path = tmp_path / "text" / "manifest.json"
    text_manifest_path.write_text(json.dumps(text_manifest), encoding="utf-8")

    corpus = load_latent_still_corpus(plan_path, latent_manifest_path, text_manifest_path)

    assert len(corpus.rows) == 2
    assert corpus.train_indices == (0,)
    assert corpus.validation_indices == (1,)
    assert corpus.contract["train_identities"] == 1

    result = run_latent_still_training(
        plan_path,
        latent_manifest_path,
        text_manifest_path,
        tmp_path / "run",
        config=LatentStillTrainingConfig(
            batch_size=1,
            gradient_accumulation=1,
            steps=2,
            warmup_steps=0,
            checkpoint_every=1,
            validate_every=1,
            log_every=1,
            validation_rows=1,
            device="cpu",
            precision="float32",
            model=LatentStillDiTConfig(
                latent_size=64,
                latent_channels=8,
                patch_size=8,
                model_dim=32,
                depth=2,
                num_heads=4,
                mlp_ratio=2,
                condition_dim=768,
                window_size=2,
                global_attention_every=2,
            ),
        ),
    )
    checkpoint = torch.load(result.training_checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 2
    assert result.report_path.is_file()
