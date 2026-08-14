"""Download and hash-index the pinned CogVideoX image-to-video model on Spark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import model_info, snapshot_download

MODEL_ID = "THUDM/CogVideoX-5b-I2V"
REVISION = "a6f0f4858a8395e7429d82493864ce92bf73af11"
ROOT = Path("/home/sleepy/sprite-lab-cogvideox")
OUTPUT = ROOT / "CogVideoX-5b-I2V-a6f0f4858a83"


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    info = model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.private or info.gated:
        raise RuntimeError("CogVideoX model identity or access policy differs")
    expected_files = {
        sibling.rfilename: sibling.size for sibling in info.siblings if sibling.rfilename
    }
    complete = bool(expected_files) and all(
        (OUTPUT / relative).is_file()
        and (expected_bytes is None or (OUTPUT / relative).stat().st_size == expected_bytes)
        for relative, expected_bytes in expected_files.items()
    )
    if not complete:
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            local_dir=OUTPUT,
            max_workers=4,
        )
    missing_or_wrong = [
        relative
        for relative, expected_bytes in expected_files.items()
        if not (OUTPUT / relative).is_file()
        or (expected_bytes is not None and (OUTPUT / relative).stat().st_size != expected_bytes)
    ]
    if missing_or_wrong:
        raise RuntimeError(f"CogVideoX snapshot closure differs: {missing_or_wrong}")
    records = []
    for path in sorted(
        (path for path in OUTPUT.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(OUTPUT).as_posix().encode(),
    ):
        relative = path.relative_to(OUTPUT).as_posix()
        if relative == "source-index.json" or relative.startswith(".cache/"):
            continue
        records.append(
            {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": file_sha256(path),
            }
        )
    source_index = {
        "artifact_kind": "huggingface_model_snapshot_index",
        "files": records,
        "license_claim": {
            "scope": "model repository; retain upstream license without reinterpretation",
            "spdx": None,
            "upstream_value": info.card_data.license if info.card_data else None,
        },
        "model_id": MODEL_ID,
        "requested_revision": REVISION,
        "resolved_revision": info.sha,
        "schema_version": 1,
        "source_url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
    }
    payload = (
        json.dumps(source_index, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    path = OUTPUT / "source-index.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError("existing CogVideoX source index differs")
    else:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "bytes": sum(record["bytes"] for record in records),
                "files": len(records),
                "output": str(OUTPUT),
                "source_index_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
