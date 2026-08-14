"""Publish exact provenance for the external Alucard sprite baseline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safetensors import safe_open  # noqa: E402

from spritelab.storage import DiskGuard  # noqa: E402

MODEL_REVISION = "b8e7602fc8e676d0b0bc0abb11d2cda665c560d8"
CODE_COMMIT = "02d1c60a16142015f7838a6a033da5e6ac9ce4f7"
MODEL = ROOT / f"data/models/alucard-{MODEL_REVISION}"
CODE = ROOT / f"data/models/alucard-source-{CODE_COMMIT}"
OUTPUT = ROOT / "data/index/reports/alucard-pretrained-source-index-v1.json"
EXPECTED_FILES = {
    ".gitattributes": "11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361",
    "LICENSE": "71386f5c714dd4723f26e8b8c485175edf79e0b463a374734f7496ce03419924",
    "README.md": "22fe3096d158416b397fc48282e7984635fcae8583e5a661c596991fd0beb9ae",
    "alucard_model.safetensors": (
        "2f502cc676c9fc34009d6c57caa4e782512a2643f436bc16408f477c352ccc2c"
    ),
    "config.json": "c155993ff2a9d9eec9bd24cfcf02b20344042025f36a347a87579a02bd1cb0c5",
}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace source index: {OUTPUT}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != CODE_COMMIT:
        raise RuntimeError("Alucard code checkout commit differs")
    files = []
    for name, expected_sha256 in sorted(EXPECTED_FILES.items(), key=lambda item: item[0].encode()):
        path = MODEL / name
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"Alucard model file hash differs: {name}")
        files.append(
            {
                "file_sha256": actual_sha256,
                "name": name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "url": (
                    f"https://huggingface.co/evilsocket/alucard/resolve/{MODEL_REVISION}/{name}"
                ),
            }
        )
    weights = MODEL / "alucard_model.safetensors"
    tensor_count = 0
    parameter_count = 0
    dtypes = set()
    with safe_open(weights, framework="pt", device="cpu") as handle:
        for key in handle.keys():  # noqa: SIM118 - safe_open is not a Mapping
            tensor = handle.get_tensor(key)
            tensor_count += 1
            parameter_count += tensor.numel()
            dtypes.add(str(tensor.dtype))
    if parameter_count != 31_956_228:
        raise RuntimeError("Alucard parameter count differs")
    artifact = {
        "artifact_kind": "external_pretrained_sprite_baseline_source_index",
        "code": {
            "commit": CODE_COMMIT,
            "path": str(CODE.resolve()),
            "repository_url": "https://github.com/evilsocket/alucard",
        },
        "model": {
            "architecture_claim": "32M pixel-space RGBA flow-matching U-Net",
            "files": files,
            "parameter_count": parameter_count,
            "repository_url": "https://huggingface.co/evilsocket/alucard",
            "revision": MODEL_REVISION,
            "tensor_count": tensor_count,
            "tensor_dtypes": sorted(dtypes),
        },
        "rights": {
            "license_file_sha256": EXPECTED_FILES["LICENSE"],
            "license_identifier_claim": "FAIR-1.0.0",
            "scope": "model and repository stated license; upstream training assets not re-audited",
            "use_context": "noncommercial research baseline",
        },
        "schema_version": 1,
    }
    payload = _canonical_json(artifact)
    DiskGuard(ROOT, min_free_bytes=100 * 1024**3).require_capacity(
        len(payload) + 1024**2, label="Alucard source index"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "path": str(OUTPUT),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
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
