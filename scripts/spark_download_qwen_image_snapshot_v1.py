"""Resumably acquire the exact upstream Qwen-Image Diffusers snapshot on Spark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Qwen/Qwen-Image"
REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
EXPECTED_TOTAL_BYTES = 57_704_594_653
DEFAULT_TARGET = Path(f"/home/sleepy/sprite-lab-qwen/Qwen-Image-{REVISION[:12]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--min-free-gib", type=int, default=100)
    args = parser.parse_args()
    target = args.target.resolve()
    floor = args.min_free_gib * 1024**3
    manifest_path = target / "spritelab-snapshot-manifest.json"
    if manifest_path.exists():
        manifest = _object(manifest_path.read_bytes())
        if (
            manifest.get("repo_id") != REPO_ID
            or manifest.get("revision") != REVISION
            or manifest.get("total_bytes") != EXPECTED_TOTAL_BYTES
        ):
            raise RuntimeError("existing Qwen snapshot manifest has a different contract")
        _verify_manifest(target, manifest)
        print(json.dumps({"status": "already_complete", "manifest": str(manifest_path)}))
        return 0

    info = HfApi().model_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise RuntimeError(f"Hugging Face resolved {info.sha}, expected {REVISION}")
    siblings = sorted(info.siblings, key=lambda sibling: sibling.rfilename.encode("utf-8"))
    total = sum(sibling.size or 0 for sibling in siblings)
    if total != EXPECTED_TOTAL_BYTES:
        raise RuntimeError(f"Qwen snapshot size is {total}, expected {EXPECTED_TOTAL_BYTES}")
    target.mkdir(parents=True, exist_ok=True)

    missing_bytes = 0
    for sibling in siblings:
        expected_size = sibling.size
        if expected_size is None:
            raise RuntimeError(f"missing size metadata for {sibling.rfilename}")
        destination = target / sibling.rfilename
        if destination.exists():
            if destination.stat().st_size != expected_size:
                raise RuntimeError(
                    f"existing final file has wrong size: {destination}; partial cache is retained"
                )
            _verify_lfs(
                destination,
                sibling.lfs.sha256 if sibling.lfs is not None else None,
            )
        else:
            missing_bytes += expected_size
    _require_capacity(target, missing_bytes, floor, "complete Qwen Image snapshot")

    records = []
    for position, sibling in enumerate(siblings, 1):
        expected_size = sibling.size
        assert expected_size is not None
        destination = target / sibling.rfilename
        if not destination.exists():
            _require_capacity(target, expected_size, floor, sibling.rfilename)
            downloaded = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=sibling.rfilename,
                    revision=REVISION,
                    local_dir=target,
                )
            ).resolve()
            if downloaded != destination.resolve():
                raise RuntimeError(
                    f"download destination differs: got {downloaded}, expected {destination}"
                )
        if destination.stat().st_size != expected_size:
            raise RuntimeError(f"downloaded file size differs: {destination}")
        sha256 = _file_sha256(destination)
        if sibling.lfs is not None and sha256 != sibling.lfs.sha256:
            raise RuntimeError(f"downloaded LFS SHA-256 differs: {destination}")
        records.append(
            {
                "blob_id": sibling.blob_id,
                "lfs_sha256": sibling.lfs.sha256 if sibling.lfs is not None else None,
                "path": sibling.rfilename,
                "sha256": sha256,
                "size_bytes": expected_size,
            }
        )
        print(
            json.dumps(
                {
                    "file": sibling.rfilename,
                    "position": position,
                    "sha256": sha256,
                    "total": len(siblings),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    artifact: dict[str, Any] = {
        "artifact_kind": "spritelab_huggingface_model_snapshot",
        "files": records,
        "license": "apache-2.0",
        "repo_id": REPO_ID,
        # Spark's pinned Comfy environment is Python 3.10 and lacks datetime.UTC.
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "revision": REVISION,
        "schema_version": 1,
        "total_bytes": EXPECTED_TOTAL_BYTES,
    }
    payload = _canonical(artifact)
    _require_capacity(target, len(payload), floor, "snapshot manifest")
    with manifest_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_lfs(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is not None and _file_sha256(path) != expected_sha256:
        raise RuntimeError(f"existing LFS SHA-256 differs: {path}")


def _verify_manifest(root: Path, manifest: dict[str, Any]) -> None:
    records = manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError("snapshot manifest files are absent")
    for record in records:
        if not isinstance(record, dict):
            raise RuntimeError("snapshot manifest file record is invalid")
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("snapshot manifest path is invalid")
        path = (root / relative).resolve()
        if root != path and root not in path.parents:
            raise RuntimeError(f"snapshot manifest path escapes root: {relative}")
        if path.stat().st_size != record.get("size_bytes") or _file_sha256(path) != record.get(
            "sha256"
        ):
            raise RuntimeError(f"snapshot manifest file differs: {relative}")


def _require_capacity(root: Path, needed: int, floor: int, label: str) -> None:
    free = shutil.disk_usage(root).free
    if free - needed < floor:
        raise RuntimeError(
            f"disk floor would be crossed for {label}: free={free}, needed={needed}, floor={floor}"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("snapshot manifest is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("snapshot manifest must be an object")
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
