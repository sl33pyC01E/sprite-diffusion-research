from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.mugen_dense_caption import (
    export_mugen_dense_caption_inputs,
    load_mugen_dense_caption_references,
)
from spritelab.storage import DiskGuard


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    value = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    value[3, 20:28, 30:38] = (100, 50, 25, 255)
    array_path = source / "idle.npy"
    np.save(array_path, value, allow_pickle=False)
    frame_sha = _array_sha256(np.ascontiguousarray(value[3]))
    dense = {
        "artifact_kind": "mugen_dense_reference_motion_training_manifest",
        "records": [
            {
                "identity": {"label": "secret franchise name"},
                "identity_id": "identity-a",
                "reference": {
                    "array": {
                        "array_content_sha256": _array_sha256(value),
                        "file_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
                        "relative_path": array_path.name,
                    },
                    "frame_array_content_sha256": frame_sha,
                    "frame_index": 3,
                    "selection_method": "premultiplied_rgba_temporal_medoid_v1",
                },
                "source_index": 0,
                "split": "train",
                "variant_id": "variant-a",
            }
        ],
        "source_materializations": [{"root": str(source)}],
    }
    path = tmp_path / "dense.json"
    path.write_text(json.dumps(dense), encoding="utf-8")
    return path


def test_dense_caption_reference_and_export_are_exact(tmp_path: Path) -> None:
    dense = _fixture(tmp_path)
    references = load_mugen_dense_caption_references(dense)
    output = tmp_path / "caption-inputs"

    digest = export_mugen_dense_caption_inputs(dense, output, disk_guard=DiskGuard(tmp_path, 0))

    assert len(references) == 1
    assert references[0].frame_index == 3
    manifest_path = output / "manifest.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == digest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["records"][0]
    image_path = output.joinpath(*record["caption_input"]["relative_path"].split("/"))
    assert (
        hashlib.sha256(image_path.read_bytes()).hexdigest()
        == record["caption_input"]["file_sha256"]
    )
    assert Image.open(image_path).size == (512, 512)
    assert "secret franchise name" not in json.dumps(manifest["caption_contract"])


def test_dense_caption_export_is_no_clobber(tmp_path: Path) -> None:
    dense = _fixture(tmp_path)
    output = tmp_path / "caption-inputs"
    guard = DiskGuard(tmp_path, 0)
    export_mugen_dense_caption_inputs(dense, output, disk_guard=guard)

    with pytest.raises(FileExistsError):
        export_mugen_dense_caption_inputs(dense, output, disk_guard=guard)
