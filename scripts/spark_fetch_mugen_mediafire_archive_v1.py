"""Resolve and resumably acquire one indexed MediaFire MUGEN archive on Spark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

USER_AGENT = "SpriteLab-Research/0.1 provenance-indexed noncommercial POC"
MAX_HTML_BYTES = 2 * 1024**2
CHUNK_BYTES = 8 * 1024**2
SYNC_BYTES = 64 * 1024**2
SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)\b", re.IGNORECASE)
CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--section-ordinal", type=int, required=True)
    parser.add_argument("--target-root", type=Path, default=Path("/home/sleepy/sprite-lab-mugen"))
    parser.add_argument("--min-free-gib", type=int, default=100)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    report_file = args.metadata_report.resolve()
    report_sha256 = _expect_hash(
        report_file, args.expected_report_sha256, "priority metadata report"
    )
    report = _object(report_file.read_bytes(), "priority metadata report")
    record = _select_record(report, args.section_ordinal)
    target_root = args.target_root.resolve()
    floor = args.min_free_gib * 1024**3
    expected_size = _positive_int(record.get("declared_size_bytes"), "declared size")
    target_root.mkdir(parents=True, exist_ok=True)

    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30, read=120, write=120, pool=30),
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    ) as client:
        resolved = _resolve_mediafire(client, _text(record, "url"))
        if resolved["declared_size_bytes"] != expected_size:
            raise RuntimeError(
                "fresh MediaFire size differs from the indexed declaration: "
                f"{resolved['declared_size_bytes']} != {expected_size}"
            )
        if resolved["filename"] != _text(record, "filename"):
            raise RuntimeError("fresh MediaFire filename differs from the indexed declaration")
        partial = (
            target_root
            / "partials"
            / (hashlib.sha256(_text(record, "url").encode()).hexdigest() + ".partial")
        )
        existing = partial.stat().st_size if partial.exists() else 0
        if existing > expected_size:
            raise RuntimeError(f"partial exceeds expected archive size: {partial}")
        needed = expected_size - existing
        _require_capacity(target_root, needed, floor, "indexed MediaFire archive")
        preflight = {
            "declared_size_bytes": expected_size,
            "direct_download_url": resolved["direct_download_url"],
            "existing_partial_bytes": existing,
            "filename": resolved["filename"],
            "free_bytes": shutil.disk_usage(target_root).free,
            "landing_url": record["url"],
            "metadata_report_sha256": report_sha256,
            "needed_bytes": needed,
            "section_source_ordinal": args.section_ordinal,
        }
        if args.preflight_only:
            print(json.dumps({"status": "ready", **preflight}, sort_keys=True))
            return 0
        partial.parent.mkdir(parents=True, exist_ok=True)
        if existing < expected_size:
            _download(
                client,
                resolved["direct_download_url"],
                partial,
                existing=existing,
                expected_size=expected_size,
                floor=floor,
            )
        if partial.stat().st_size != expected_size:
            raise RuntimeError("completed partial size differs")
        archive_sha256 = _file_sha256(partial)
        cas_path = target_root / "objects" / "sha256" / archive_sha256[:2] / archive_sha256[2:4]
        cas_path /= archive_sha256
        cas_path.parent.mkdir(parents=True, exist_ok=True)
        if cas_path.exists():
            if cas_path.stat().st_size != expected_size or _file_sha256(cas_path) != archive_sha256:
                raise RuntimeError(f"existing CAS object differs: {cas_path}")
            partial_retained = True
        else:
            os.replace(partial, cas_path)
            partial_retained = False
        acquisitions = target_root / "acquisitions"
        acquisitions.mkdir(parents=True, exist_ok=True)
        acquisition_path = acquisitions / f"mikazuki-section-{args.section_ordinal}-v1.json"
        if acquisition_path.exists():
            raise FileExistsError(f"Refusing to replace acquisition report: {acquisition_path}")
        artifact = {
            "artifact_kind": "mugen_mediafire_archive_acquisition",
            "archive": {
                "cas_path": str(cas_path),
                "filename": resolved["filename"],
                "sha256": archive_sha256,
                "size_bytes": expected_size,
            },
            "provider": {
                "direct_download_url_at_retrieval": resolved["direct_download_url"],
                "landing_html_sha256": resolved["landing_html_sha256"],
                "landing_url": record["url"],
                "name": "MediaFire",
            },
            "retrieval": {
                "partial_retained_because_cas_preexisted": partial_retained,
                # Spark's pinned environment is Python 3.10 and lacks datetime.UTC.
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                "resumed_from_bytes": existing,
            },
            "schema_version": 1,
            "source": {
                "metadata_report_path": str(report_file),
                "metadata_report_sha256": report_sha256,
                "section_source_ordinal": args.section_ordinal,
                "section_title": record.get("section_title"),
            },
        }
        payload = _canonical(artifact)
        _require_capacity(target_root, len(payload), floor, "acquisition report")
        with acquisition_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "acquisition_report": str(acquisition_path),
                "acquisition_report_sha256": hashlib.sha256(payload).hexdigest(),
                "archive_sha256": archive_sha256,
                "archive_size_bytes": expected_size,
                "cas_path": str(cas_path),
                "status": "complete",
            },
            sort_keys=True,
        )
    )
    return 0


def _select_record(report: dict[str, object], section_ordinal: int) -> dict[str, object]:
    records = report.get("records")
    if not isinstance(records, list):
        raise RuntimeError("priority metadata report records are absent")
    selected = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("section_source_ordinal") == section_ordinal
        and record.get("status") == "resolved_single_file"
    ]
    if len(selected) != 1:
        raise RuntimeError(
            f"section {section_ordinal} resolves to {len(selected)} single-file records"
        )
    record = selected[0]
    if urlparse(_text(record, "url")).netloc.casefold() not in {
        "mediafire.com",
        "www.mediafire.com",
    }:
        raise RuntimeError("selected record is not a MediaFire landing page")
    return record


def _resolve_mediafire(client: httpx.Client, url: str) -> dict[str, object]:
    chunks = []
    total = 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").casefold()
        if "text/html" not in content_type:
            raise RuntimeError(f"MediaFire landing response is not HTML: {content_type!r}")
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > MAX_HTML_BYTES:
                raise RuntimeError("MediaFire landing HTML exceeds bounded metadata policy")
            chunks.append(chunk)
    payload = b"".join(chunks)
    soup = BeautifulSoup(payload, "html.parser")
    filename_node = soup.select_one(".filename")
    details_node = soup.select_one(".details")
    download_node = soup.select_one("a#downloadButton[href]")
    if filename_node is None or details_node is None or download_node is None:
        raise RuntimeError("MediaFire landing metadata is incomplete")
    match = SIZE_PATTERN.search(details_node.get_text(" ", strip=True))
    if match is None:
        raise RuntimeError("MediaFire landing size is absent")
    filename = " ".join(filename_node.get_text(" ", strip=True).split())
    return {
        "declared_size_bytes": _parse_size(match.group(1), match.group(2)),
        "direct_download_url": download_node["href"],
        "filename": filename,
        "landing_html_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _download(
    client: httpx.Client,
    url: str,
    partial: Path,
    *,
    existing: int,
    expected_size: int,
    floor: int,
) -> None:
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with client.stream("GET", url, headers=headers) as response:
        if response.status_code not in ({206} if existing else {200, 206}):
            raise RuntimeError(
                f"MediaFire returned HTTP {response.status_code}; existing partial retained"
            )
        append = existing > 0
        if response.status_code == 206:
            match = CONTENT_RANGE.match(response.headers.get("content-range") or "")
            if match is None:
                raise RuntimeError("MediaFire partial response lacks valid Content-Range")
            start, end, total = map(int, match.groups())
            if start != existing or end < start or total != expected_size:
                raise RuntimeError("MediaFire Content-Range differs from indexed archive")
        elif existing:
            raise RuntimeError("MediaFire ignored Range; existing partial retained")
        declared = response.headers.get("content-length")
        if declared and declared.isdigit():
            expected_remaining = expected_size - existing
            if int(declared) != expected_remaining:
                raise RuntimeError("MediaFire response Content-Length differs")
        mode = "ab" if append else "xb"
        written_since_sync = 0
        with partial.open(mode) as handle:
            for chunk in response.iter_bytes(CHUNK_BYTES):
                if not chunk:
                    continue
                _require_capacity(partial.parent, len(chunk), floor, "MediaFire download chunk")
                handle.write(chunk)
                written_since_sync += len(chunk)
                if written_since_sync >= SYNC_BYTES:
                    handle.flush()
                    os.fsync(handle.fileno())
                    written_since_sync = 0
            handle.flush()
            os.fsync(handle.fileno())


def _parse_size(number: str, unit: str) -> int:
    powers = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}
    return round(float(number) * 1024 ** powers[unit.upper()])


def _require_capacity(root: Path, needed: int, floor: int, label: str) -> None:
    free = shutil.disk_usage(root).free
    if free - needed < floor:
        raise RuntimeError(
            f"disk floor would be crossed for {label}: free={free}, needed={needed}, floor={floor}"
        )


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeError(f"{label} must be a positive integer")
    return value


def _text(record: dict[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{key} must be non-empty text")
    return value


def _expect_hash(path: Path, expected: str, label: str) -> str:
    actual = _file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 differs: expected {expected}, got {actual}")
    return actual


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
