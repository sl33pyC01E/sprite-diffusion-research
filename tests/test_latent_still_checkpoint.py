from __future__ import annotations

import hashlib
import json

import pytest

from spritelab.latent_still_checkpoint import (
    LatentStillCheckpointError,
    export_latent_still_intermediate_ema,
)
from spritelab.storage import DiskGuard

torch = pytest.importorskip("torch")


def test_intermediate_ema_export_is_safe_hash_bound_and_no_clobber(tmp_path) -> None:
    latent_manifest = {
        "artifact_kind": "mugen_frozen_rgba_autoencoder_latent_cache",
        "normalization": {
            "channel_mean": [0.0] * 8,
            "channel_standard_deviation": [1.0] * 8,
        },
    }
    latent_bytes = json.dumps(latent_manifest).encode()
    latent_path = tmp_path / "latents.json"
    latent_path.write_bytes(latent_bytes)
    latent_sha256 = hashlib.sha256(latent_bytes).hexdigest()
    training = {
        "artifact_kind": "mugen_latent_still_dit_resume_checkpoint",
        "config": {"model": {"model_dim": 32}},
        "corpus": {"latent_manifest_file_sha256": latent_sha256},
        "ema_model": {"weight": torch.ones(2)},
        "ema_policy": {"decay": 0.9},
        "step": 10,
    }
    training_path = tmp_path / "training.pt"
    torch.save(training, training_path)
    training_sha256 = hashlib.sha256(training_path.read_bytes()).hexdigest()
    output = tmp_path / "ema.pt"

    result, digest = export_latent_still_intermediate_ema(
        training_path,
        latent_path,
        output,
        expected_training_checkpoint_sha256=training_sha256,
        expected_latent_manifest_sha256=latent_sha256,
        disk_guard=DiskGuard(tmp_path, 0),
    )

    assert result == output
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    value = torch.load(output, map_location="cpu", weights_only=True)
    assert value["artifact_kind"] == "mugen_latent_still_dit_ema_inference_checkpoint"
    assert value["step"] == 10
    assert value["lineage"]["source_training_checkpoint_sha256"] == training_sha256
    with pytest.raises(FileExistsError):
        export_latent_still_intermediate_ema(
            training_path,
            latent_path,
            output,
            expected_training_checkpoint_sha256=training_sha256,
            expected_latent_manifest_sha256=latent_sha256,
            disk_guard=DiskGuard(tmp_path, 0),
        )


def test_intermediate_ema_export_rejects_latent_lineage_mismatch(tmp_path) -> None:
    latent_path = tmp_path / "latents.json"
    latent_path.write_text(
        json.dumps(
            {
                "artifact_kind": "mugen_frozen_rgba_autoencoder_latent_cache",
                "normalization": {
                    "channel_mean": [0.0] * 8,
                    "channel_standard_deviation": [1.0] * 8,
                },
            }
        ),
        encoding="utf-8",
    )
    training_path = tmp_path / "training.pt"
    torch.save(
        {
            "artifact_kind": "mugen_latent_still_dit_resume_checkpoint",
            "config": {"model": {}},
            "corpus": {"latent_manifest_file_sha256": "f" * 64},
            "ema_model": {"weight": torch.ones(1)},
            "step": 1,
        },
        training_path,
    )
    with pytest.raises(LatentStillCheckpointError, match="latent corpus differs"):
        export_latent_still_intermediate_ema(
            training_path,
            latent_path,
            tmp_path / "ema.pt",
            expected_training_checkpoint_sha256=hashlib.sha256(
                training_path.read_bytes()
            ).hexdigest(),
            expected_latent_manifest_sha256=hashlib.sha256(latent_path.read_bytes()).hexdigest(),
            disk_guard=DiskGuard(tmp_path, 0),
        )
