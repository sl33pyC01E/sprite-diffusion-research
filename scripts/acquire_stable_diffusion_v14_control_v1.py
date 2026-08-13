"""Acquire the exact minimal Stable Diffusion v1.4 LoRA-control components."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.storage import DiskGuard  # noqa: E402

MODEL_ID = "CompVis/stable-diffusion-v1-4"
REVISION = "eb7ecef2ce03788573d2863ef3a7e501ee25cd6c"
OUTPUT = ROOT / f"data/models/stable-diffusion-v1-4-{REVISION[:12]}-training-components"
ALLOW_PATTERNS = (
    "nREADME.md",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)


def main() -> None:
    index_path = OUTPUT / "source-index.json"
    if index_path.exists():
        raise FileExistsError(f"Refusing to replace Stable Diffusion source index: {index_path}")
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
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
    missing = set(ALLOW_PATTERNS) - set(remote)
    if missing:
        raise RuntimeError(f"pinned revision is missing files: {sorted(missing)!r}")
    expected_bytes = sum(int(record["size_bytes"] or 0) for record in remote.values())
    guard.require_capacity(expected_bytes + 512 * 1024**2, label="Stable Diffusion v1.4 control")
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
        "artifact_kind": "stable_diffusion_lora_control_source_index",
        "canonical_model_url": f"https://huggingface.co/{MODEL_ID}",
        "files": files,
        "immutable_revision_url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "license": "CreativeML-OpenRAIL-M",
        "license_evidence_url": f"https://huggingface.co/{MODEL_ID}/blob/{REVISION}/nREADME.md",
        "model_id": MODEL_ID,
        "resolved_revision": REVISION,
        "schema_version": 1,
        "scope": (
            "non-canonical RGB pretrained still-image quality control; UNet LoRA only; "
            "no safety checker or inference pipeline components acquired"
        ),
    }
    payload = json.dumps(index, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    with index_path.open("xb") as handle:
        handle.write(payload.encode())
        handle.flush()
    print(
        {
            "expected_bytes": expected_bytes,
            "output": str(OUTPUT),
            "revision": REVISION,
            "source_index_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
