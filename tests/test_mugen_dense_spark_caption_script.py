from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mugen_dense_spark_captions_v1.py"
    spec = importlib.util.spec_from_file_location("run_mugen_dense_spark_captions_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_caption_input_loader_verifies_files_and_hides_private_runtime_path(
    tmp_path: Path,
) -> None:
    image = b"png fixture"
    image_path = tmp_path / "image.png"
    image_path.write_bytes(image)
    manifest = {
        "artifact_kind": "mugen_dense_literal_visual_caption_input_dataset",
        "record_count": 1,
        "records": [
            {
                "caption_input": {
                    "file_sha256": hashlib.sha256(image).hexdigest(),
                    "relative_path": image_path.name,
                    "size_bytes": len(image),
                },
                "variant_id": "variant-a",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    records, payload = _module().load_caption_inputs(path)

    assert json.loads(payload)["record_count"] == 1
    assert records[0]["_image_path"] == str(image_path)
    assert "_image_path" not in json.loads(payload)["records"][0]


def test_caption_input_loader_rejects_tampered_file(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"changed")
    manifest = {
        "artifact_kind": "mugen_dense_literal_visual_caption_input_dataset",
        "record_count": 1,
        "records": [
            {
                "caption_input": {
                    "file_sha256": "a" * 64,
                    "relative_path": image_path.name,
                    "size_bytes": 7,
                },
                "variant_id": "variant-a",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="hash differs"):
        _module().load_caption_inputs(path)
