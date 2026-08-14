"""Publish the generic local source contract for the sprite-specific SD checkpoint."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.storage import DiskGuard  # noqa: E402

REVISION = "8229c9b6e928103f0e657cfe6b14d902cb2101d6"
MODEL_ID = "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator"
MODEL = ROOT / f"data/models/sd-pixelart-spritesheet-{REVISION}"
EVIDENCE = ROOT / "data/index/reports/sd-pixelart-spritesheet-source-index-v1.json"
EVIDENCE_SHA256 = "fd3d6898d01901256215ee04e19142d9d36ec32ae7be1fc0ca09101239233167"
OUTPUT = MODEL / "source-index.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace training source index: {OUTPUT}")
    evidence_bytes = EVIDENCE.read_bytes()
    if hashlib.sha256(evidence_bytes).hexdigest() != EVIDENCE_SHA256:
        raise RuntimeError("sprite checkpoint evidence index differs")
    evidence = json.loads(evidence_bytes)
    files = evidence.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("sprite checkpoint evidence files are absent")
    for record in files:
        path = MODEL / record["relative_path"]
        if _file_sha256(path) != record["file_sha256"]:
            raise RuntimeError(f"sprite checkpoint file differs: {record['relative_path']}")
    artifact = {
        "artifact_kind": "local_pretrained_diffusion_source_index",
        "derived_from_source_index_file_sha256": EVIDENCE_SHA256,
        "files": files,
        "model_id": MODEL_ID,
        "resolved_revision": REVISION,
        "schema_version": 1,
        "serialization": {
            "format": "pytorch_bin",
            "loader_contract": "diffusers_torch_load_weights_only_true",
        },
    }
    payload = _canonical_json(artifact)
    DiskGuard(ROOT, min_free_bytes=100 * 1024**3).require_capacity(
        len(payload) + 1024**2, label="sprite checkpoint training source index"
    )
    with OUTPUT.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {"path": str(OUTPUT), "sha256": hashlib.sha256(payload).hexdigest()},
            sort_keys=True,
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


if __name__ == "__main__":
    main()
