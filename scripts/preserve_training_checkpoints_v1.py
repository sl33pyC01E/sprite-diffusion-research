"""Preserve rotating training checkpoints as immutable hard-linked snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

REQUIRED_FILES = (
    "optimizer.bin",
    "pytorch_lora_weights.safetensors",
    "random_states_0.pkl",
    "scheduler.bin",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--once", action="store_true")
    arguments = parser.parse_args()
    if arguments.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    source = arguments.source.resolve()
    archive = arguments.archive.resolve()
    if source == archive or source in archive.parents or archive in source.parents:
        raise ValueError("source and archive trees must not contain one another")
    if not source.is_dir():
        raise FileNotFoundError(f"checkpoint source is absent: {source}")
    archive.mkdir(parents=True, exist_ok=True)

    prior_signatures: dict[str, tuple[tuple[str, int, int], ...]] = {}
    verified_existing: set[str] = set()
    trainer_absent_polls = 0
    while True:
        current = scan_complete_checkpoints(source)
        for name, signature in current.items():
            target = archive / name
            if target.exists():
                if name not in verified_existing:
                    verify_existing_snapshot(source / name, target)
                    publish_preservation_report(source / name, target, archive)
                    verified_existing.add(name)
                continue
            if not arguments.once and prior_signatures.get(name) != signature:
                continue
            preserve_checkpoint(source / name, target, archive)
            verified_existing.add(name)
            print(f"preserved {source / name} -> {target}", flush=True)
        prior_signatures = current
        if arguments.once:
            return 0
        if _pid_exists(arguments.trainer_pid):
            trainer_absent_polls = 0
        else:
            trainer_absent_polls += 1
            if trainer_absent_polls >= 2:
                # The first absent poll establishes a final stable signature;
                # the second gives the final checkpoint one preservation pass.
                return 0
        time.sleep(arguments.poll_seconds)


def scan_complete_checkpoints(
    source: Path,
) -> dict[str, tuple[tuple[str, int, int], ...]]:
    checkpoints = {}
    for path in sorted(source.glob("checkpoint-*"), key=lambda value: value.name.encode("utf-8")):
        if not path.is_dir() or any(not (path / name).is_file() for name in REQUIRED_FILES):
            continue
        checkpoints[path.name] = _tree_signature(path)
    return checkpoints


def preserve_checkpoint(source: Path, target: Path, archive: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to replace preserved checkpoint: {target}")
    partial = archive / f".{target.name}.partial-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, partial, copy_function=os.link)
        verify_existing_snapshot(source, partial)
        os.replace(partial, target)
    except BaseException:
        print(f"checkpoint preservation failed; retained partial tree: {partial}", flush=True)
        raise
    publish_preservation_report(source, target, archive)


def verify_existing_snapshot(source: Path, target: Path) -> None:
    source_rows = _tree_rows(source)
    target_rows = _tree_rows(target)
    if [row["relative_path"] for row in source_rows] != [
        row["relative_path"] for row in target_rows
    ]:
        raise RuntimeError(f"preserved checkpoint file set differs: {target}")
    for source_row, target_row in zip(source_rows, target_rows, strict=True):
        for key in ("size_bytes", "file_sha256"):
            if source_row[key] != target_row[key]:
                raise RuntimeError(f"preserved checkpoint differs at {source_row['relative_path']}")


def publish_preservation_report(source: Path, target: Path, archive: Path) -> None:
    report_path = archive / f"{target.name}-preservation.json"
    rows = _tree_rows(target)
    report = {
        "artifact_kind": "hardlinked_training_checkpoint_preservation",
        "schema_version": 1,
        "source": str(source),
        "preserved": str(target),
        "method": "same-filesystem hard link; source rotation cannot remove preserved inode",
        "files": rows,
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
    }
    payload = _canonical_json(report)
    if report_path.exists():
        existing = report_path.read_bytes()
        if existing != payload:
            raise RuntimeError(f"preservation report differs: {report_path}")
        return
    with report_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _tree_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    rows = []
    for path in sorted(
        (value for value in root.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(root).as_posix().encode("utf-8"),
    ):
        stat = path.stat()
        rows.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    return tuple(rows)


def _tree_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(
        (value for value in root.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(root).as_posix().encode("utf-8"),
    ):
        stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": stat.st_size,
                "file_sha256": _file_sha256(path),
            }
        )
    return rows


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
