from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.mugen_dense_manifest import (
    build_mugen_dense_manifest,
    export_mugen_dense_manifest,
)
from spritelab.mugen_stream_quality import export_mugen_stream_quality_audit
from spritelab.storage import DiskGuard


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _root(
    root: Path,
    *,
    variant: str,
    sff_sha: str,
    shared_idle: np.ndarray,
    color_offset: int = 0,
    display_name: str | None = None,
    scale: float = 1.0,
    identity_id: str | None = None,
) -> Path:
    root.mkdir()
    clips = []
    for index, slot in enumerate(("idle", "walk", "jump", "block", "attack_a", "attack_b")):
        if slot == "idle":
            value = shared_idle.copy()
        else:
            value = np.zeros((8, 128, 128, 4), dtype=np.uint8)
            value[:, 20:24, 30 + index : 34 + index] = (
                index + color_offset,
                2,
                3,
                255,
            )
            value[4:, 24:28, 30 + index : 34 + index] = (
                4 + color_offset,
                5,
                6,
                255,
            )
        path = root / f"{variant}-{slot}.npy"
        np.save(path, value, allow_pickle=False)
        clips.append(
            {
                "action_number": index,
                "array": {
                    "array_content_sha256": _array_sha256(value),
                    "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "relative_path": path.name,
                },
                "clipped_visible_pixels": 0,
                "loop_mode": "loop",
                "record_id": f"record-{variant}-{slot}",
                "schema_phase": None,
                "schema_verb": slot,
                "slot": slot,
                "source_action_index": index,
                "temporal_selection": {
                    "source_frame_count": 2,
                    "target_phases": [frame / 8 for frame in range(8)],
                },
            }
        )
    character = {
        "clips": clips,
        "complete_six_slot_core": True,
        "definitions": [{"display_name": display_name or variant, "name": variant}],
        "identity_id": identity_id or f"identity-{variant}",
        "source": {
            "air": {"sha256": "b" * 64},
            "archive_sha256": "c" * 64,
            "sff": {"sha256": sff_sha},
        },
        "variant_id": variant,
        "world_view_transform": {"scale": scale},
    }
    (root / "materialization.json").write_text(
        json.dumps({"characters": [character], "projection_version": 2}), encoding="utf-8"
    )
    return root


def test_dense_manifest_keeps_exact_array_duplicates_in_one_split(tmp_path: Path) -> None:
    idle = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    idle[:, 10:14, 10:14] = (1, 2, 3, 255)
    idle[4:, 14:18, 10:14] = (3, 2, 1, 255)
    first = _root(tmp_path / "first", variant="variant-a", sff_sha="a" * 64, shared_idle=idle)
    second = _root(tmp_path / "second", variant="variant-b", sff_sha="d" * 64, shared_idle=idle)
    quality = tmp_path / "quality.json"
    export_mugen_stream_quality_audit((first, second), quality, disk_guard=DiskGuard(tmp_path, 0))

    manifest = build_mugen_dense_manifest((first, second), quality)

    assert manifest["counts"]["characters"] == 2
    assert manifest["counts"]["components"] == 1
    assert len({row["split"] for row in manifest["records"]}) == 1
    assert manifest["records"][0]["reference"]["selection_method"] == (
        "premultiplied_rgba_temporal_medoid_v1"
    )
    assert len(manifest["records"][0]["reference"]["frame_array_content_sha256"]) == 64
    train_probe = manifest["evaluation_probes"][manifest["records"][0]["split"]]
    assert {row["variant_id"] for row in train_probe} == {"variant-a", "variant-b"}
    assert set(train_probe[0]["actions"]) == {
        "idle",
        "walk",
        "jump",
        "block",
        "attack_a",
        "attack_b",
    }


def test_dense_manifest_keeps_normalized_identity_labels_in_one_split(tmp_path: Path) -> None:
    first_idle = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    first_idle[:, 10:14, 10:14] = (1, 2, 3, 255)
    first_idle[4:, 14:18, 10:14] = (3, 2, 1, 255)
    second_idle = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    second_idle[:, 20:24, 20:24] = (7, 8, 9, 255)
    second_idle[4:, 24:28, 20:24] = (9, 8, 7, 255)
    first = _root(
        tmp_path / "first",
        variant="variant-a",
        sff_sha="a" * 64,
        shared_idle=first_idle,
        display_name="M. Bison",
    )
    second = _root(
        tmp_path / "second",
        variant="variant-b",
        sff_sha="d" * 64,
        shared_idle=second_idle,
        color_offset=10,
        display_name="m bison",
    )
    quality = tmp_path / "quality.json"
    export_mugen_stream_quality_audit((first, second), quality, disk_guard=DiskGuard(tmp_path, 0))

    manifest = build_mugen_dense_manifest((first, second), quality)

    assert manifest["counts"]["components"] == 1
    assert len({row["split"] for row in manifest["records"]}) == 1
    assert any(
        "identity_label:m bison" in component["tokens"] for component in manifest["components"]
    )


def test_same_sff_variants_get_distinct_training_ids_in_one_split(tmp_path: Path) -> None:
    first_idle = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    first_idle[:, 10:14, 10:14] = (1, 2, 3, 255)
    first_idle[4:, 14:18, 10:14] = (3, 2, 1, 255)
    second_idle = first_idle.copy()
    first = _root(
        tmp_path / "first",
        variant="variant-a",
        sff_sha="a" * 64,
        shared_idle=first_idle,
        identity_id="source-sff-identity",
    )
    second = _root(
        tmp_path / "second",
        variant="variant-b",
        sff_sha="a" * 64,
        shared_idle=second_idle,
        color_offset=10,
        identity_id="source-sff-identity",
    )
    quality = tmp_path / "quality.json"
    export_mugen_stream_quality_audit((first, second), quality, disk_guard=DiskGuard(tmp_path, 0))

    manifest = build_mugen_dense_manifest((first, second), quality)

    assert len({row["identity_id"] for row in manifest["records"]}) == 2
    assert {row["source_identity_id"] for row in manifest["records"]} == {"source-sff-identity"}
    assert len({row["split"] for row in manifest["records"]}) == 1


def test_dense_subset_inherits_broad_universe_split(tmp_path: Path) -> None:
    first_idle = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    first_idle[:, 10:14, 10:14] = (1, 2, 3, 255)
    first_idle[4:, 14:18, 10:14] = (3, 2, 1, 255)
    second_idle = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    second_idle[:, 20:24, 20:24] = (7, 8, 9, 255)
    second_idle[4:, 24:28, 20:24] = (9, 8, 7, 255)
    dense_root = _root(
        tmp_path / "dense-root",
        variant="variant-dense",
        sff_sha="a" * 64,
        shared_idle=first_idle,
        display_name="Shared Family",
    )
    broad_only_root = _root(
        tmp_path / "broad-root",
        variant="variant-broad-only",
        sff_sha="d" * 64,
        shared_idle=second_idle,
        color_offset=10,
        display_name="shared-family",
        scale=0.25,
    )
    quality = tmp_path / "quality.json"
    roots = (dense_root, broad_only_root)
    export_mugen_stream_quality_audit(roots, quality, disk_guard=DiskGuard(tmp_path, 0))

    dense = build_mugen_dense_manifest(roots, quality, tier="dense")
    broad = build_mugen_dense_manifest(roots, quality, tier="broad")

    assert dense["counts"]["characters"] == 1
    assert dense["counts"]["split_universe_characters"] == 2
    assert dense["counts"]["selected_components"] == 1
    dense_record = dense["records"][0]
    broad_record = next(
        row for row in broad["records"] if row["variant_id"] == dense_record["variant_id"]
    )
    assert dense_record["split"] == broad_record["split"]
    assert any(
        set(component["variant_ids"]) == {"variant-dense", "variant-broad-only"}
        for component in dense["components"]
    )


def test_dense_manifest_rejects_materialization_changed_after_audit(tmp_path: Path) -> None:
    idle = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    idle[:, 10:14, 10:14] = (1, 2, 3, 255)
    root = _root(tmp_path / "root", variant="variant-a", sff_sha="a" * 64, shared_idle=idle)
    quality = tmp_path / "quality.json"
    export_mugen_stream_quality_audit((root,), quality, disk_guard=DiskGuard(tmp_path, 0))
    with (root / "materialization.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")

    with pytest.raises(ValueError, match="does not bind materialization"):
        build_mugen_dense_manifest((root,), quality)


def test_dense_manifest_export_is_no_clobber(tmp_path: Path) -> None:
    idle = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    idle[:, 10:14, 10:14] = (1, 2, 3, 255)
    root = _root(tmp_path / "root", variant="variant-a", sff_sha="a" * 64, shared_idle=idle)
    quality = tmp_path / "quality.json"
    guard = DiskGuard(tmp_path, 0)
    export_mugen_stream_quality_audit((root,), quality, disk_guard=guard)
    output = tmp_path / "dense.json"

    digest = export_mugen_dense_manifest((root,), quality, output, disk_guard=guard)

    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        export_mugen_dense_manifest((root,), quality, output, disk_guard=guard)
