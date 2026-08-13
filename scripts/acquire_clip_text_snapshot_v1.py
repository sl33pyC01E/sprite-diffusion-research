"""Acquire and index the exact frozen CLIP text snapshot used for semantic conditioning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "openai/clip-vit-base-patch32"
REVISION = "c7244be81152024ce0e99ac8d2e373a8953d9f9a"
OUTPUT = ROOT / f"data/models/openai-clip-vit-base-patch32-{REVISION[:12]}"
ALLOW_PATTERNS = (
    "config.json",
    "merges.txt",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace CLIP snapshot: {OUTPUT}")
    info = model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"resolved revision differs: {info.sha}")
    remote = {
        sibling.rfilename: {
            "lfs_sha256": sibling.lfs.sha256 if sibling.lfs is not None else None,
            "size_bytes": sibling.size,
        }
        for sibling in info.siblings
        if sibling.rfilename in ALLOW_PATTERNS
    }
    if set(remote) != set(ALLOW_PATTERNS):
        raise RuntimeError(f"pinned revision is missing files: {set(ALLOW_PATTERNS) - set(remote)}")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=OUTPUT,
        allow_patterns=list(ALLOW_PATTERNS),
    )
    files = []
    for relative in ALLOW_PATTERNS:
        path = OUTPUT / relative
        if not path.is_file():
            raise RuntimeError(f"downloaded snapshot is missing {relative}")
        local_sha = _sha256_file(path)
        expected_lfs = remote[relative]["lfs_sha256"]
        if expected_lfs is not None and local_sha != expected_lfs:
            raise RuntimeError(f"LFS SHA-256 mismatch for {relative}")
        files.append(
            {
                "file_sha256": local_sha,
                "relative_path": relative,
                "remote_lfs_sha256": expected_lfs,
                "size_bytes": path.stat().st_size,
            }
        )
    index = {
        "artifact_kind": "frozen_semantic_text_encoder_source_index",
        "canonical_model_url": f"https://huggingface.co/{MODEL_ID}",
        "immutable_revision_url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "license_evidence_url": "https://github.com/openai/CLIP/blob/main/LICENSE",
        "model_id": MODEL_ID,
        "resolved_revision": REVISION,
        "files": files,
        "scope": "frozen CLIP text projection and tokenizer for research conditioning",
        "schema_version": 1,
    }
    payload = json.dumps(index, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    (OUTPUT / "source-index.json").write_text(payload, encoding="utf-8", newline="\n")
    print(
        {
            "output": str(OUTPUT),
            "revision": REVISION,
            "source_index_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    )


if __name__ == "__main__":
    main()
