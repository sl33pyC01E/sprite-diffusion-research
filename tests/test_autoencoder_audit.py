from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import spritelab.autoencoder_audit as audit  # noqa: E402
from spritelab.autoencoder_audit import (  # noqa: E402
    AutoencoderAuditContractError,
    export_autoencoder_reconstruction_audit,
    select_identity_diverse_frames,
)
from spritelab.autoencoder_train import SpriteAutoencoderTrainingConfig  # noqa: E402
from spritelab.broad_train import PreparedBroadCorpus, PreparedBroadRow  # noqa: E402
from spritelab.models.sprite_autoencoder import (  # noqa: E402
    SpriteAutoencoderConfig,
    SpriteRGBAAutoencoder,
)


def _row(identity: str, action: str, seed: int, *, split: str) -> PreparedBroadRow:
    import numpy as np

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
        train=(
            _row("a", "idle", 1, split="train"),
            _row("a", "run", 2, split="train"),
            _row("b", "idle", 3, split="train"),
        ),
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


def _checkpoint(tmp_path, corpus: PreparedBroadCorpus) -> tuple[object, str]:
    architecture = SpriteAutoencoderConfig(
        image_size=16,
        base_channels=8,
        latent_channels=4,
        channel_multipliers=(1, 2),
        residual_blocks=1,
    )
    config = SpriteAutoencoderTrainingConfig(
        architecture=architecture,
        device="cpu",
        precision="float32",
        steps=2,
        warmup_steps=0,
    )
    model = SpriteRGBAAutoencoder(architecture)
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "artifact_kind": "sprite_rgba_autoencoder_resume_checkpoint",
            "config": asdict(config),
            "corpus": {
                "corpus_sha256": corpus.corpus_sha256,
                "materialization_manifest_sha256": corpus.materialization_manifest_sha256,
                "source_snapshot_canonical_sha256": corpus.source_snapshot_canonical_sha256,
                "source_snapshot_manifest_sha256": corpus.source_snapshot_manifest_sha256,
                "train_rows": len(corpus.train),
                "validation_rows": len(corpus.validation),
            },
            "ema": model.state_dict(),
            "model": model.state_dict(),
            "runtime": {"torch": str(torch.__version__)},
            "schema_version": 1,
            "step": 2,
        },
        path,
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_identity_diverse_selection_uses_each_identity_first() -> None:
    corpus = _corpus()
    selected = select_identity_diverse_frames(corpus.validation, maximum_frames=2)
    assert len(selected) == 2
    assert {corpus.validation[row].identity_id for row, _ in selected} == {"c", "d"}
    assert selected == select_identity_diverse_frames(corpus.validation, maximum_frames=2)


def test_reconstruction_audit_is_hash_bound_no_clobber_and_stock_safe_load(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus()
    prepare_arguments = {}

    def prepare(*args, **kwargs):
        prepare_arguments.update(kwargs)
        return corpus

    monkeypatch.setattr(audit, "prepare_broad_corpus", prepare)
    checkpoint, checkpoint_sha256 = _checkpoint(tmp_path, corpus)
    output = tmp_path / "audit"

    result = export_autoencoder_reconstruction_audit(
        "fixture.json",
        checkpoint,
        output,
        expected_checkpoint_sha256=checkpoint_sha256,
        maximum_frames=2,
        integer_scale=1,
    )

    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert prepare_arguments["usage"] == "autoencoder"
    assert report["artifact_kind"] == ("sprite_rgba_autoencoder_held_out_reconstruction_audit")
    assert report["display_gallery"]["sample_count"] == 2
    assert report["checkpoint"]["load_contract"] == (
        "torch.load(weights_only=True,map_location='cpu')"
    )
    assert len(report["samples"]) == 2
    assert (
        report["target_array"]["file_sha256"]
        == hashlib.sha256(result.target_array_path.read_bytes()).hexdigest()
    )
    assert result.gallery_path.is_file()
    with pytest.raises(FileExistsError, match="replace"):
        export_autoencoder_reconstruction_audit(
            "fixture.json",
            checkpoint,
            output,
            expected_checkpoint_sha256=checkpoint_sha256,
        )
    with pytest.raises(AutoencoderAuditContractError, match="mismatch"):
        export_autoencoder_reconstruction_audit(
            "fixture.json",
            checkpoint,
            tmp_path / "bad",
            expected_checkpoint_sha256="0" * 64,
        )
