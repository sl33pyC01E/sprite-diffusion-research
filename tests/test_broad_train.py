from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.broad_train import (
    BroadTrainingConfig,
    PreparedBroadRow,
    identity_action_index,
    prepare_broad_corpus,
    resize_rgba_nearest,
    run_broad_training,
    sample_balanced_indices,
)


def _row(sequence_id: str, identity: str, action: str) -> PreparedBroadRow:
    return PreparedBroadRow(
        sequence_id=sequence_id,
        identity_id=identity,
        action=action,
        split="train",
        request=object(),
        rgba=np.zeros((2, 2, 2, 4), dtype=np.uint8),
        frame_phases=(0.0, 0.5),
        source_size=(2, 2),
        source_file_sha256="a" * 64,
        normalized_array_sha256="b" * 64,
    )


def test_resize_rgba_nearest_has_explicit_floor_index_semantics() -> None:
    rgba = np.zeros((1, 2, 2, 4), dtype=np.uint8)
    rgba[0, ..., 0] = np.asarray([[1, 2], [3, 4]], dtype=np.uint8)
    expanded = resize_rgba_nearest(rgba, 4)
    assert expanded[0, ..., 0].tolist() == [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]
    assert resize_rgba_nearest(expanded, 2)[0, ..., 0].tolist() == [[1, 2], [3, 4]]


def test_identity_action_index_is_stable_and_preserves_variants() -> None:
    rows = (
        _row("z", "wolf", "walk"),
        _row("a", "ant", "idle"),
        _row("y", "wolf", "walk"),
        _row("b", "ant", "attack"),
    )
    assert identity_action_index(rows) == {
        "ant": {"attack": (3,), "idle": (1,)},
        "wolf": {"walk": (0, 2)},
    }


def test_balanced_sampler_is_generator_reproducible() -> None:
    torch = pytest.importorskip("torch")
    index = identity_action_index((_row("a", "ant", "idle"), _row("b", "wolf", "walk")))
    left = torch.Generator(device="cpu").manual_seed(91)
    right = torch.Generator(device="cpu").manual_seed(91)
    assert sample_balanced_indices(index, batch_size=12, generator=left) == (
        sample_balanced_indices(index, batch_size=12, generator=right)
    )


def test_broad_config_rejects_invalid_geometry_and_schedule() -> None:
    with pytest.raises(ValueError, match="divisible"):
        BroadTrainingConfig(target_size=63, patch_size=8)
    with pytest.raises(ValueError, match="warmup_steps"):
        BroadTrainingConfig(steps=10, warmup_steps=10)
    with pytest.raises(ValueError, match="minimum_learning_rate"):
        BroadTrainingConfig(learning_rate=1e-4, minimum_learning_rate=2e-4)
    with pytest.raises(ValueError, match="horizontal_flip_probability"):
        BroadTrainingConfig(horizontal_flip_probability=1.1)


def _fixture_manifest(tmp_path: Path) -> Path:
    sequences = []
    for index, (split, identity, action) in enumerate(
        (("train", "red-fox", "run"), ("validation", "blue-wolf", "idle"))
    ):
        array = np.zeros((2, 4, 4, 4), dtype=np.uint8)
        array[..., index, 0] = 100 + index
        array[..., index, 3] = 255
        path = tmp_path / "clips" / split / f"{identity}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
        sequences.append(
            {
                "action": action,
                "caption": {"description": identity.replace("-", " ")},
                "direction": "right",
                "entity_class": "animal",
                "frame_count": 2,
                "identity_id": identity,
                "loop_mode": "loop",
                "output": {
                    "array_content_sha256": hashlib.sha256(
                        header + array.tobytes(order="C")
                    ).hexdigest(),
                    "dtype": "uint8",
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "format": "numpy_npy_v1",
                    "relative_path": path.relative_to(tmp_path).as_posix(),
                    "shape": list(array.shape),
                    "size_bytes": path.stat().st_size,
                },
                "provenance": {
                    "source_blob_sha256": [f"{index + 1:x}" * 64],
                    "source_id": "fixture",
                },
                "quality_tier": "F0",
                "sequence_id": f"sequence-{index}",
                "split": split,
                "target_bucket": [4, 4],
                "timing": {"duration_ms": [100, 100], "phase": [0.0, 0.5]},
                "view": "side",
            }
        )
    manifest = {
        "schema_version": 1,
        "sequence_count": len(sequences),
        "sequences": sequences,
        "source_snapshot": {
            "canonical_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "schema_version": 1,
        },
    }
    path = tmp_path / "materialization.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_prepare_broad_corpus_preserves_disjoint_splits_and_hashes(tmp_path: Path) -> None:
    corpus = prepare_broad_corpus(_fixture_manifest(tmp_path), target_size=8, target_frames=2)

    assert [row.identity_id for row in corpus.train] == ["red-fox"]
    assert [row.identity_id for row in corpus.validation] == ["blue-wolf"]
    assert corpus.train[0].rgba.shape == (2, 8, 8, 4)
    assert len(corpus.corpus_sha256) == 64


def test_one_step_cpu_training_exports_resume_and_inference_checkpoints(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    result = run_broad_training(
        _fixture_manifest(tmp_path),
        tmp_path / "experiment",
        config=BroadTrainingConfig(
            target_size=4,
            target_frames=2,
            patch_size=2,
            model_dim=16,
            depth=1,
            num_heads=4,
            condition_dim=16,
            max_text_bytes=8,
            batch_size=1,
            gradient_accumulation=1,
            horizontal_flip_probability=1.0,
            learning_rate=1e-4,
            minimum_learning_rate=1e-5,
            warmup_steps=0,
            steps=1,
            log_every=1,
            validate_every=1,
            checkpoint_every=1,
            seed=7,
            device="cpu",
            precision="float32",
        ),
    )

    assert result.training_checkpoint_path.is_file()
    assert result.inference_checkpoint_path.is_file()
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["corpus"]["identity_overlap"] == 0
    assert report["final_validation"]["sample_count"] == 1
