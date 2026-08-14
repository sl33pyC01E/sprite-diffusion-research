"""Resumably acquire exact indexed Google Drive MUGEN archives into local CAS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlencode, urlparse

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/index/reports/mugen-mikazuki-priority-download-metadata-v1.json"
REPORT_SHA256 = "4afcea23a2098ca5b03031c72b8a77fb2d86f0efaf58e7652e090eb499577ed1"
USER_AGENT = "SpriteLab-Research/0.1 provenance-indexed noncommercial POC"
CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
CHUNK_BYTES = 8 * 1024**2
SYNC_BYTES = 64 * 1024**2
# Drive intermittently ignores otherwise valid 256 MiB ranges for these very large
# public objects. 64 MiB remains resumable while keeping request overhead modest.
RANGE_REQUEST_BYTES = 64 * 1024**2
MAX_HTML_BYTES = 1024**2
MAX_CONFIRMATION_REFRESHES = 512


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section-ordinal", type=int, action="append", required=True)
    parser.add_argument("--min-free-gib", type=int, default=100)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--full-stream", action="store_true")
    args = parser.parse_args()
    if len(set(args.section_ordinal)) != len(args.section_ordinal):
        raise ValueError("section ordinals must be unique")
    report_payload = REPORT.read_bytes()
    if hashlib.sha256(report_payload).hexdigest() != REPORT_SHA256:
        raise RuntimeError("Mikazuki metadata report SHA-256 differs")
    report = _object(report_payload, "Mikazuki metadata report")
    floor = args.min_free_gib * 1024**3
    selected = [_select_record(report, ordinal) for ordinal in args.section_ordinal]
    with httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30, read=180, write=180, pool=30),
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    ) as client:
        for record in selected:
            result = _acquire_record(
                client,
                record,
                floor=floor,
                full_stream=args.full_stream,
                preflight_only=args.preflight_only,
            )
            print(json.dumps(result, sort_keys=True), flush=True)
    return 0


def _acquire_record(
    client: httpx.Client,
    record: dict[str, object],
    *,
    floor: int,
    full_stream: bool,
    preflight_only: bool,
) -> dict[str, object]:
    ordinal = _positive_int(record.get("section_source_ordinal"), "section ordinal")
    exact_indexed_size = _positive_int(record.get("declared_size_bytes"), "declared size")
    confirmed = _resolve_drive_confirmation(client, _text(record, "direct_download_url"))
    exact_remote_size = _probe_exact_size(client, confirmed["download_url"])
    if exact_remote_size != exact_indexed_size:
        raise RuntimeError(
            f"section {ordinal} Drive size differs: {exact_remote_size} != {exact_indexed_size}"
        )
    partial_suffix = ".full.part" if full_stream else ".part"
    partial = ROOT / "data/.partials" / f"mugen-mikazuki-section-{ordinal}{partial_suffix}"
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > exact_remote_size:
        raise RuntimeError(f"partial exceeds exact Drive object size: {partial}")
    needed = exact_remote_size - existing
    _require_capacity(ROOT, needed, floor, f"Mikazuki section {ordinal}")
    preflight = {
        "declared_size_bytes": exact_indexed_size,
        "existing_partial_bytes": existing,
        "filename": _text(record, "filename"),
        "free_bytes": shutil.disk_usage(ROOT).free,
        "needed_bytes": needed,
        "section_source_ordinal": ordinal,
        "status": "ready",
    }
    if preflight_only:
        return preflight
    report_path = ROOT / f"data/index/reports/mugen-mikazuki-section-{ordinal}-acquisition-v1.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to replace acquisition report: {report_path}")
    partial.parent.mkdir(parents=True, exist_ok=True)
    if existing < exact_remote_size:
        if full_stream:
            if existing:
                raise RuntimeError(
                    f"full-stream staging file is incomplete and non-resumable: {partial}"
                )
            _download_full_stream(
                client,
                confirmed["download_url"],
                partial,
                expected_size=exact_remote_size,
                floor=floor,
            )
            download_transport = {
                "confirmation_refresh_html_sha256": [],
                "used_full_stream_verified_prefix": False,
                "used_unconditional_full_stream": True,
            }
        else:
            download_transport = _download(
                client,
                confirmed["download_url"],
                partial,
                indexed_direct_url=_text(record, "direct_download_url"),
                existing=existing,
                expected_size=exact_remote_size,
                floor=floor,
            )
    else:
        download_transport = {
            "confirmation_refresh_html_sha256": [],
            "used_unconditional_full_stream": full_stream,
            "used_full_stream_verified_prefix": False,
        }
    if partial.stat().st_size != exact_remote_size:
        raise RuntimeError("completed partial size differs from exact Drive object")
    archive_sha256 = _file_sha256(partial)
    destination = (
        ROOT / "data/raw/objects/sha256" / archive_sha256[:2] / archive_sha256[2:4] / archive_sha256
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if (
            destination.stat().st_size != exact_remote_size
            or _file_sha256(destination) != archive_sha256
        ):
            raise RuntimeError(f"existing CAS object differs: {destination}")
        cas_preexisted = True
        retained_partial = str(partial)
    else:
        os.replace(partial, destination)
        cas_preexisted = False
        retained_partial = None
    artifact = {
        "archive": {
            "cas_path": str(destination),
            "filename": record["filename"],
            "sha256": archive_sha256,
            "size_bytes": exact_remote_size,
        },
        "artifact_kind": "mugen_mikazuki_google_drive_archive_acquisition",
        "provider": {
            "confirmation_html_sha256": confirmed["confirmation_html_sha256"],
            **download_transport,
            "download_url_at_retrieval": confirmed["download_url"],
            "landing_url": record["url"],
            "name": "Google Drive",
        },
        "retrieval": {
            "cas_preexisted": cas_preexisted,
            "partial_retained_because_cas_preexisted": retained_partial,
            "resumed_from_bytes": existing,
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
        },
        "schema_version": 1,
        "source": {
            "metadata_report_path": str(REPORT),
            "metadata_report_sha256": REPORT_SHA256,
            "section_source_ordinal": ordinal,
            "section_title": record.get("section_title"),
        },
    }
    payload = _canonical(artifact)
    _require_capacity(ROOT, len(payload), floor, "Drive acquisition report")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "acquisition_report": str(report_path),
        "acquisition_report_sha256": hashlib.sha256(payload).hexdigest(),
        "archive_sha256": archive_sha256,
        "archive_size_bytes": exact_remote_size,
        "cas_path": str(destination),
        "section_source_ordinal": ordinal,
        "status": "complete",
    }


def _download_full_stream(
    client: httpx.Client,
    url: str,
    partial: Path,
    *,
    expected_size: int,
    floor: int,
) -> None:
    """Download one exact Drive object as a single stream, without range semantics."""

    _require_capacity(partial.parent, expected_size, floor, "Drive full stream")
    received = 0
    written_since_sync = 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").casefold()
        if "text/html" in content_type:
            raise RuntimeError("Google Drive full stream returned HTML instead of the archive")
        with partial.open("xb") as handle:
            for chunk in response.iter_bytes(CHUNK_BYTES):
                if not chunk:
                    continue
                received += len(chunk)
                if received > expected_size:
                    raise RuntimeError("Google Drive full stream exceeds exact indexed size")
                _require_capacity(partial.parent, len(chunk), floor, "Drive full-stream chunk")
                handle.write(chunk)
                written_since_sync += len(chunk)
                if written_since_sync >= SYNC_BYTES:
                    handle.flush()
                    os.fsync(handle.fileno())
                    written_since_sync = 0
                if received % (1024**3) < len(chunk):
                    _console(
                        f"full stream {received}/{expected_size} bytes "
                        f"({received / expected_size:.1%})"
                    )
            handle.flush()
            os.fsync(handle.fileno())
    if received != expected_size:
        raise RuntimeError(f"Google Drive full stream differs: {received} != {expected_size}")


def _resolve_drive_confirmation(client: httpx.Client, direct_url: str) -> dict[str, str]:
    parsed = urlparse(direct_url)
    if parsed.netloc.casefold() != "drive.usercontent.google.com":
        raise RuntimeError("indexed direct URL is not Google Drive usercontent")
    response = client.get(direct_url)
    response.raise_for_status()
    if len(response.content) > MAX_HTML_BYTES:
        raise RuntimeError("Google Drive confirmation HTML exceeds bound")
    content_type = (response.headers.get("content-type") or "").casefold()
    if "text/html" not in content_type:
        raise RuntimeError("Google Drive did not return its confirmation page")
    soup = BeautifulSoup(response.content, "html.parser")
    form = soup.find("form", attrs={"action": True})
    if form is None:
        raise RuntimeError("Google Drive confirmation form is absent")
    action = str(form.get("action"))
    if urlparse(action).netloc.casefold() != "drive.usercontent.google.com":
        raise RuntimeError("Google Drive confirmation action has an unexpected host")
    parameters = {
        str(node.get("name")): str(node.get("value") or "")
        for node in form.find_all("input")
        if node.get("name")
    }
    if parameters.get("confirm") != "t" or not parameters.get("id") or not parameters.get("uuid"):
        raise RuntimeError("Google Drive confirmation parameters are incomplete")
    return {
        "confirmation_html_sha256": hashlib.sha256(response.content).hexdigest(),
        "download_url": action + "?" + urlencode(parameters),
    }


def _probe_exact_size(client: httpx.Client, url: str) -> int:
    with client.stream("GET", url, headers={"Range": "bytes=0-0"}) as response:
        if response.status_code != 206:
            raise RuntimeError(f"Google Drive range probe returned HTTP {response.status_code}")
        match = CONTENT_RANGE.match(response.headers.get("content-range") or "")
        if match is None or tuple(map(int, match.groups()[:2])) != (0, 0):
            raise RuntimeError("Google Drive range probe has invalid Content-Range")
        total = int(match.group(3))
        if response.headers.get("content-length") != "1" or total <= 0:
            raise RuntimeError("Google Drive range probe has invalid exact size")
        return total


def _download(
    client: httpx.Client,
    url: str,
    partial: Path,
    *,
    indexed_direct_url: str,
    existing: int,
    expected_size: int,
    floor: int,
) -> dict[str, object]:
    position = existing
    mode = "ab" if existing else "xb"
    refresh_hashes: list[str] = []
    used_full_stream = False
    with partial.open(mode) as handle:
        while position < expected_size:
            end = min(position + RANGE_REQUEST_BYTES, expected_size) - 1
            _require_capacity(
                partial.parent,
                end - position + 1,
                floor,
                "Drive ranged download",
            )
            while True:
                range_complete = False
                with client.stream(
                    "GET", url, headers={"Range": f"bytes={position}-{end}"}
                ) as response:
                    if response.status_code not in {200, 206}:
                        raise RuntimeError(
                            "Google Drive range returned HTTP "
                            f"{response.status_code}; partial retained"
                        )
                    if response.status_code == 200:
                        if response.headers.get("content-length") != str(expected_size):
                            _console(f"Drive throttled range at byte {position}; cooling down")
                        else:
                            _append_full_stream_after_verified_prefix(
                                response,
                                partial=partial,
                                append_handle=handle,
                                verified_prefix_bytes=position,
                                expected_size=expected_size,
                                floor=floor,
                            )
                            position = expected_size
                            used_full_stream = True
                            range_complete = True
                    else:
                        match = CONTENT_RANGE.match(response.headers.get("content-range") or "")
                        if match is None:
                            raise RuntimeError("Google Drive download lacks valid Content-Range")
                        start, actual_end, total = map(int, match.groups())
                        if start != position or actual_end != end or total != expected_size:
                            raise RuntimeError(
                                "Google Drive download range differs from exact object"
                            )
                        expected_chunk = end - position + 1
                        if response.headers.get("content-length") != str(expected_chunk):
                            raise RuntimeError("Google Drive ranged Content-Length differs")
                        received = 0
                        written_since_sync = 0
                        for chunk in response.iter_bytes(CHUNK_BYTES):
                            if not chunk:
                                continue
                            _require_capacity(
                                partial.parent, len(chunk), floor, "Drive download chunk"
                            )
                            handle.write(chunk)
                            received += len(chunk)
                            written_since_sync += len(chunk)
                            if written_since_sync >= SYNC_BYTES:
                                handle.flush()
                                os.fsync(handle.fileno())
                                written_since_sync = 0
                        if received != expected_chunk:
                            raise RuntimeError(
                                f"Google Drive ranged body differs: {received} != {expected_chunk}"
                            )
                        handle.flush()
                        os.fsync(handle.fileno())
                        range_complete = True
                if range_complete:
                    break
                if len(refresh_hashes) >= MAX_CONFIRMATION_REFRESHES:
                    raise RuntimeError("Google Drive confirmation refresh bound exceeded")
                refreshed = _resolve_drive_confirmation(client, indexed_direct_url)
                url = refreshed["download_url"]
                refresh_hashes.append(refreshed["confirmation_html_sha256"])
                time.sleep(15)
                _console(
                    f"Drive ignored range at byte {position}; refreshed confirmation "
                    f"({len(refresh_hashes)})"
                )
            if not used_full_stream:
                position = end + 1
            _console(
                f"section download {position}/{expected_size} bytes "
                f"({position / expected_size:.1%})"
            )
    return {
        "confirmation_refresh_html_sha256": refresh_hashes,
        "used_full_stream_verified_prefix": used_full_stream,
    }


def _append_full_stream_after_verified_prefix(
    response: httpx.Response,
    *,
    partial: Path,
    append_handle: BinaryIO,
    verified_prefix_bytes: int,
    expected_size: int,
    floor: int,
) -> None:
    """Accept Drive's full stream only after exact comparison with our partial prefix."""

    streamed = 0
    written_since_sync = 0
    with partial.open("rb") as prefix_handle:
        for chunk in response.iter_bytes(CHUNK_BYTES):
            if not chunk:
                continue
            cursor = 0
            if streamed < verified_prefix_bytes:
                prefix_count = min(len(chunk), verified_prefix_bytes - streamed)
                expected_prefix = prefix_handle.read(prefix_count)
                if expected_prefix != chunk[:prefix_count]:
                    raise RuntimeError(f"Drive full stream differs from partial at byte {streamed}")
                cursor = prefix_count
            tail = chunk[cursor:]
            if tail:
                _require_capacity(partial.parent, len(tail), floor, "Drive full-stream tail")
                append_handle.write(tail)
                written_since_sync += len(tail)
                if written_since_sync >= SYNC_BYTES:
                    append_handle.flush()
                    os.fsync(append_handle.fileno())
                    written_since_sync = 0
            streamed += len(chunk)
    if streamed != expected_size:
        raise RuntimeError(f"Drive full stream size differs: {streamed} != {expected_size}")
    append_handle.flush()
    os.fsync(append_handle.fileno())
    if partial.stat().st_size != expected_size:
        raise RuntimeError("Drive full-stream append produced an unexpected partial size")


def _select_record(report: dict[str, object], ordinal: int) -> dict[str, object]:
    records = report.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Mikazuki metadata records are absent")
    selected = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("section_source_ordinal") == ordinal
        and record.get("status") == "resolved_single_file"
        and record.get("provider_domain") == "drive.google.com"
    ]
    if len(selected) != 1:
        raise RuntimeError(f"section {ordinal} resolves to {len(selected)} Drive files")
    return selected[0]


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


def _console(message: str) -> None:
    print(message.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


if __name__ == "__main__":
    sys.exit(main())
