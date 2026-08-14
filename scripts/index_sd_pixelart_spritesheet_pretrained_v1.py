"""Publish exact provenance for the pinned SD pixel-art sprite-sheet baseline."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.storage import DiskGuard  # noqa: E402

REPOSITORY = "Onodofthenorth/SD_PixelArt_SpriteSheet_Generator"
REVISION = "8229c9b6e928103f0e657cfe6b14d902cb2101d6"
MODEL = ROOT / f"data/models/sd-pixelart-spritesheet-{REVISION}"
OUTPUT = ROOT / "data/index/reports/sd-pixelart-spritesheet-source-index-v1.json"
EXPECTED_FILES = {
    ".gitattributes": "5609513b096029ff8f97ea53d470f7c77324257540d2bc89a07e4afb39253df8",
    "README.md": "a6888ea0fa64d0bb98ec8ab9227af38d39f7cd26aa2f9b07c2d60d1df7e1529d",
    "feature_extractor/preprocessor_config.json": (
        "2a1da83b5e1032aaeef397552ddb408dca0d8cd1dc58f61bf6abf38d6f33a0a2"
    ),
    "model_index.json": "e2f6f22e274374010aec30c79eb2d6e53fff4a36ac0225523a92cb2d41a83347",
    "scheduler/scheduler_config.json": (
        "07a12844f77ecf6c43f8907304c9cd58ba5d9e02eadb151cf284e8e6c6268a7b"
    ),
    "text_encoder/config.json": (
        "eb1d0ff4bbdfb3ae1ccdb94491df5728dacf0e9db59bdc5e139e3490774dae33"
    ),
    "text_encoder/pytorch_model.bin": (
        "b0546e93f1be86db0a8872a5e1a6c8eb3a161577486abb720eee437a3f64d30c"
    ),
    "tokenizer/merges.txt": ("9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
    "tokenizer/special_tokens_map.json": (
        "c4864a9376a8401918425bed71fc14fc0e81f9b59ec45c1cf96cccb2df508eac"
    ),
    "tokenizer/tokenizer_config.json": (
        "00439066fcba73de57644cf41e4e3b9f2dbb09d7f3fc2005898ba52399045882"
    ),
    "tokenizer/vocab.json": ("e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"),
    "unet/config.json": "78f474de6bab3d893868f37be97b636ae65c0df3073ed3256ca458ff599b5f96",
    "unet/diffusion_pytorch_model.bin": (
        "1516cb50f37947da4a33777cb5d785a88ff06388ff819b4aee30da4be09a0431"
    ),
    "vae/config.json": "65d2c77722ca3f6510e7a65ab292dd29d66703d25bfd51544144981c888dae34",
    "vae/diffusion_pytorch_model.bin": (
        "1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc"
    ),
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace source index: {OUTPUT}")
    files = []
    for relative, expected_sha256 in sorted(
        EXPECTED_FILES.items(), key=lambda item: item[0].encode()
    ):
        path = MODEL / relative
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"sprite-sheet model file differs: {relative}")
        files.append(
            {
                "file_sha256": actual_sha256,
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "url": f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{relative}",
            }
        )
    metadata = MODEL / ".cache/huggingface/download/model_index.json.metadata"
    metadata_revision = metadata.read_text(encoding="utf-8").splitlines()[0]
    if metadata_revision != REVISION:
        raise RuntimeError("download metadata revision differs")
    artifact = {
        "artifact_kind": "external_pretrained_latent_sprite_baseline_source_index",
        "files": files,
        "model": {
            "architecture_claim": "Stable Diffusion latent sprite-sheet checkpoint",
            "repository": REPOSITORY,
            "revision": REVISION,
            "trigger_tokens": {
                "back": "PixelartBSS",
                "front": "PixelartFSS",
                "left": "PixelartLSS",
                "right": "PixelartRSS",
            },
        },
        "rights": {
            "license_identifier_claim": "Apache-2.0",
            "model_card_file_sha256": EXPECTED_FILES["README.md"],
            "scope": "model repository claim; upstream training assets not re-audited",
        },
        "schema_version": 1,
    }
    payload = _canonical_json(artifact)
    DiskGuard(ROOT, min_free_bytes=100 * 1024**3).require_capacity(
        len(payload) + 1024**2, label="sprite-sheet model source index"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
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
