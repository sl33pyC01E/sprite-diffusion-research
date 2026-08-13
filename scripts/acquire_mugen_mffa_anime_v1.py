"""Acquire the indexed MFFA anime character archives into the guarded CAS.

The append-only journal makes the long collection pass power-loss resumable. It
contains one canonical JSON record per completed URL; the immutable summary is
published only after every selected candidate has either completed or failed.
No downloaded character code is executed or imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.fetch import HttpFetcher  # noqa: E402
from spritelab.storage import ContentAddressedStore, DiskFloorReached, DiskGuard  # noqa: E402

DISCOVERY = ROOT / "data/index/reports/mugen-mffa-anime-discovery-v2.json"
OUTPUT = ROOT / "data/index/reports/mugen-mffa-anime-zip-acquisition-v1.json"
JOURNAL = ROOT / "data/index/reports/mugen-mffa-anime-zip-acquisition-v1.jsonl"
USER_AGENT = "SpriteLab-Research/0.1 provenance-indexed noncommercial POC"


def _console(message: str) -> None:
    print(message.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_journal(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"journal line {line_number} is not newline terminated")
            record = json.loads(line)
            url = record["requested_url"]
            if url in records and records[url] != record:
                raise ValueError(f"conflicting journal rows for {url}")
            records[url] = record
    return records


def _append_journal(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to replace acquisition index: {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extensions", nargs="+", default=("zip",))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--journal", type=Path, default=JOURNAL)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    journal_path = args.journal.resolve()
    extensions = {value.casefold().lstrip(".") for value in args.extensions}
    if not extensions or any(not value for value in extensions):
        raise ValueError("extensions must be non-empty")
    if output.exists():
        raise FileExistsError(f"refusing to replace acquisition index: {output}")

    discovery_bytes = DISCOVERY.read_bytes()
    discovery = json.loads(discovery_bytes)
    selected: list[tuple[dict[str, object], dict[str, object]]] = []
    for resource in discovery["resources"]:
        for candidate in resource["download_candidates"]:
            filename = str(candidate.get("filename") or "")
            extension = filename.rsplit(".", 1)[-1].casefold() if "." in filename else ""
            if extension in extensions:
                selected.append((resource, candidate))
    selected.sort(key=lambda row: str(row[1]["final_url"]).encode())
    journal = _load_journal(journal_path)
    selected_urls = {str(candidate["final_url"]) for _, candidate in selected}
    unexpected = set(journal) - selected_urls
    if unexpected:
        raise ValueError(f"journal contains {len(unexpected)} URLs outside this selection")

    guard = DiskGuard(ROOT, 100 * 1024**3)
    if args.preflight_only:
        print(
            {
                "discovery_sha256": hashlib.sha256(discovery_bytes).hexdigest(),
                "extensions": sorted(extensions),
                "free_bytes": guard.status().free_bytes,
                "resumable_completed": len(journal),
                "selected": len(selected),
            }
        )
        return 0

    fetcher = HttpFetcher(
        ContentAddressedStore(ROOT / "data", guard),
        user_agent=USER_AGENT,
        timeout_seconds=180,
        max_retries=5,
    )
    failures: list[dict[str, object]] = []
    floor_reached = False
    for position, (resource, candidate) in enumerate(selected, 1):
        url = str(candidate["final_url"])
        if url in journal:
            _console(f"[{position}/{len(selected)}] resume {candidate['filename']}")
            continue
        try:
            result = fetcher.fetch(url)
        except DiskFloorReached as error:
            floor_reached = True
            failures.append({"requested_url": url, "error": f"{type(error).__name__}: {error}"})
            _console(f"disk floor reached before {candidate['filename']}")
            break
        except Exception as error:  # retain failure evidence while continuing independent URLs
            failures.append({"requested_url": url, "error": f"{type(error).__name__}: {error}"})
            _console(f"[{position}/{len(selected)}] FAILED {candidate['filename']}: {error}")
            continue
        record = {
            "archive": {
                "bytes": result.blob.size_bytes,
                "cas_path": str(result.blob.path),
                "sha256": result.blob.sha256,
            },
            "candidate": candidate,
            "requested_url": url,
            "resource": {
                "canonical_url": resource["canonical_url"],
                "listing_text": resource["listing_text"],
                "page_sha256": resource["page_sha256"],
                "resource_id": resource["resource_id"],
                "title": resource["title"],
            },
            "retrieval": {
                "content_length": result.content_length,
                "etag": result.etag,
                "final_url": result.final_url,
                "last_modified": result.last_modified,
                "mime_type": result.mime_type,
                "resumed_from": result.resumed_from,
                "status_code": result.status_code,
            },
        }
        _append_journal(journal_path, record)
        journal[url] = record
        _console(
            f"[{position}/{len(selected)}] {candidate['filename']}: "
            f"{result.blob.size_bytes} bytes; free={guard.status().free_bytes}"
        )

    artifact = {
        "schema_version": 1,
        "artifact_kind": "mugen_mffa_anime_archive_acquisition_index",
        "claim_limit": "Public fan uploads; no permissive rights inference.",
        "discovery_index": {
            "path": str(DISCOVERY),
            "sha256": hashlib.sha256(discovery_bytes).hexdigest(),
        },
        "disk_floor_bytes": guard.min_free_bytes,
        "extensions": sorted(extensions),
        "selected_count": len(selected),
        "completed_count": len(journal),
        "completed_bytes": sum(int(row["archive"]["bytes"]) for row in journal.values()),
        "failures": failures,
        "floor_reached": floor_reached,
        "complete": len(journal) == len(selected),
        "items": [journal[url] for url in sorted(journal, key=str.encode)],
        "trust_boundary": "Archives were stored as inert bytes; no MUGEN character code executed.",
    }
    payload = (
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    guard.require_capacity(len(payload), label="MFFA acquisition index")
    _atomic_no_clobber(output, payload)
    print(
        {
            "complete": artifact["complete"],
            "completed_bytes": artifact["completed_bytes"],
            "completed_count": artifact["completed_count"],
            "index_sha256": hashlib.sha256(payload).hexdigest(),
            "journal_sha256": _sha(journal_path),
        }
    )
    return 0 if artifact["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
