from __future__ import annotations

import hashlib
import json

import pytest

from scripts.compact_completed_training_checkpoints_v1 import (
    compact_completed_training_checkpoints,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture_run(
    tmp_path,
    monkeypatch,
    *,
    report_kind="mugen_latent_still_dit_training",
):
    experiments = tmp_path / "data/experiments"
    run = experiments / "run"
    run.mkdir(parents=True)
    monkeypatch.setattr(
        "scripts.compact_completed_training_checkpoints_v1.EXPERIMENTS",
        experiments.resolve(),
    )
    payloads = {}
    for step in (10_000, 20_000, 30_000, 40_000, 50_000):
        name = f"training-step-{step:07d}.pt"
        payload = f"checkpoint-{step}".encode()
        (run / name).write_bytes(payload)
        payloads[name] = payload
    final_name = "training-step-0050000.pt"
    checkpoint_key = (
        "checkpoint" if report_kind == "sprite_rgba_autoencoder_training" else "training_checkpoint"
    )
    report = {
        "artifact_kind": report_kind,
        checkpoint_key: {
            "file_sha256": _sha(payloads[final_name]),
            "path": final_name,
        },
    }
    (run / "training-report.json").write_text(json.dumps(report), encoding="utf-8")
    return run, payloads


def test_compaction_keeps_first_and_final_and_resumes(tmp_path, monkeypatch) -> None:
    run, payloads = _fixture_run(tmp_path, monkeypatch)

    result = compact_completed_training_checkpoints(run)
    repeated = compact_completed_training_checkpoints(run)

    assert result == repeated
    assert result["kept"] == ["training-step-0010000.pt", "training-step-0050000.pt"]
    assert result["removed_bytes"] == sum(
        len(payloads[f"training-step-{step:07d}.pt"]) for step in (20_000, 30_000, 40_000)
    )
    assert sorted(path.name for path in run.glob("training-step-*.pt")) == result["kept"]
    assert (run / "checkpoint-compaction-plan.json").is_file()
    assert (run / "checkpoint-compaction-result.json").is_file()


def test_compaction_resumes_from_plan_after_partial_unlinks(tmp_path, monkeypatch) -> None:
    run, payloads = _fixture_run(tmp_path, monkeypatch)
    module = "scripts.compact_completed_training_checkpoints_v1"
    real_unlink = __import__("pathlib").Path.unlink
    removed = 0

    def interrupting_unlink(path, *args, **kwargs):
        nonlocal removed
        if path.name.startswith("training-step-"):
            removed += 1
            if removed == 2:
                raise KeyboardInterrupt
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(f"{module}.Path.unlink", interrupting_unlink)
    with pytest.raises(KeyboardInterrupt):
        compact_completed_training_checkpoints(run)
    monkeypatch.setattr(f"{module}.Path.unlink", real_unlink)

    result = compact_completed_training_checkpoints(run)

    assert result["removed_bytes"] == sum(
        len(payloads[f"training-step-{step:07d}.pt"]) for step in (20_000, 30_000, 40_000)
    )
    assert sorted(path.name for path in run.glob("training-step-*.pt")) == result["kept"]


def test_compaction_rejects_tampered_final_and_outside_root(tmp_path, monkeypatch) -> None:
    run, _payloads = _fixture_run(tmp_path, monkeypatch)
    (run / "training-step-0050000.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="final training checkpoint hash"):
        compact_completed_training_checkpoints(run)
    with pytest.raises(ValueError, match="data/experiments"):
        compact_completed_training_checkpoints(tmp_path / "outside")


def test_compaction_accepts_autoencoder_report_schema(tmp_path, monkeypatch) -> None:
    run, _payloads = _fixture_run(
        tmp_path,
        monkeypatch,
        report_kind="sprite_rgba_autoencoder_training",
    )

    result = compact_completed_training_checkpoints(run)

    assert result["kept"] == ["training-step-0010000.pt", "training-step-0050000.pt"]
