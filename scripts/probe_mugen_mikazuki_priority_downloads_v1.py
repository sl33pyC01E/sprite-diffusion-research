"""Resolve size/name metadata for priority Mikazuki links without archive reads."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.storage import ContentAddressedStore, DiskGuard  # noqa: E402

DISCOVERY = ROOT / "data/index/reports/mugen-mikazuki-roster-discovery-v2.json"
EXPECTED_DISCOVERY_SHA256 = "31ac109cbe8a7766bce4b83faad65f515fae7b8ef2616daa8a1156feacb608e7"
OUTPUT = ROOT / "data/index/reports/mugen-mikazuki-priority-download-metadata-v1.json"
USER_AGENT = "SpriteLab-Research/0.1 provenance-indexed noncommercial POC"
DRIVE_FILE = re.compile(r"drive\.google\.com/file/d/([^/?#]+)")
DRIVE_SIZE = re.compile(r'\[null,null,"(\d{5,})"\]')
MEDIAFIRE_SIZE = re.compile(r"(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)\b", re.IGNORECASE)
MAX_METADATA_BYTES = 2 * 1024 * 1024


class MetadataProbeError(RuntimeError):
    """Raised when a link violates the bounded metadata-only fetch policy."""


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace download metadata index: {OUTPUT}")
    discovery_bytes = DISCOVERY.read_bytes()
    discovery_sha256 = hashlib.sha256(discovery_bytes).hexdigest()
    if discovery_sha256 != EXPECTED_DISCOVERY_SHA256:
        raise RuntimeError("Mikazuki discovery index hash differs")
    discovery = json.loads(discovery_bytes)
    selected = []
    for section in discovery["sections"]:
        if not section["anime_or_jus_priority"]:
            continue
        for link in section["links"]:
            if link["role"] == "archive_download_entry":
                selected.append((section, link))
    selected.sort(key=lambda row: (row[0]["source_ordinal"], row[1]["url"].encode("utf-8")))

    guard = DiskGuard(ROOT, 100 * 1024**3)
    store = ContentAddressedStore(ROOT / "data", guard)
    records = []
    with httpx.Client(
        follow_redirects=True,
        timeout=5,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    ) as client:
        for position, (section, link) in enumerate(selected, 1):
            try:
                metadata = probe_link(client, store, link["url"])
            except httpx.HTTPError as error:
                metadata = _unresolved("metadata_http_error", f"{type(error).__name__}: {error}")
            except MetadataProbeError as error:
                metadata = _unresolved("metadata_policy_rejected", str(error))
            records.append(
                {
                    "section_title": section["title"],
                    "section_source_ordinal": section["source_ordinal"],
                    "anchor_text": link["anchor_text"],
                    "url": link["url"],
                    "provider_domain": link["domain"],
                    **metadata,
                }
            )
            _console(f"[{position}/{len(selected)}] {metadata['status']}: {section['title'][:72]}")

    known_sizes = [
        record["declared_size_bytes"] for record in records if record["declared_size_bytes"]
    ]
    artifact = {
        "artifact_kind": "mugen_mikazuki_priority_download_metadata_probe",
        "schema_version": 1,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "path": str(DISCOVERY.resolve()),
            "file_sha256": discovery_sha256,
        },
        "scope": {
            "selection": "discovery-v2 sections with anime_or_jus_priority=true",
            "network_policy": (
                "Landing/view HTML metadata only. Direct archive response bodies are never opened."
            ),
            "rights": "Resolved names and sizes are source evidence, not reuse permission.",
            "unresolved": (
                "Folder listings, encrypted MEGA metadata, link shorteners, and unsupported "
                "providers "
                "remain explicit rather than being guessed."
            ),
        },
        "counts": {
            "selected_link_occurrences": len(records),
            "resolved_single_files": sum(
                record["status"] == "resolved_single_file" for record in records
            ),
            "unresolved_entries": sum(
                record["status"] != "resolved_single_file" for record in records
            ),
            "known_size_entries": len(known_sizes),
            "known_size_bytes": sum(known_sizes),
            "maximum_single_file_bytes": max(known_sizes, default=0),
        },
        "records": records,
    }
    payload = _canonical_json(artifact)
    guard.require_capacity(len(payload), label="Mikazuki priority metadata report")
    _atomic_no_clobber(OUTPUT, payload)
    print(
        json.dumps(
            {
                "output": str(OUTPUT.resolve()),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                **artifact["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


def probe_link(client: httpx.Client, store: ContentAddressedStore, url: str) -> dict[str, object]:
    domain = urlparse(url).netloc.casefold()
    drive_match = DRIVE_FILE.search(url)
    if drive_match:
        return _probe_drive_file(client, store, url, drive_match.group(1))
    if domain.endswith("drive.google.com"):
        return _unresolved(
            "provider_collection_or_folder", "Google Drive link is not a single file"
        )
    if domain.endswith("mediafire.com") and "/file/" in url:
        return _probe_mediafire_file(client, store, url)
    if domain.endswith("mediafire.com"):
        return _unresolved(
            "provider_collection_or_folder", "MediaFire folder inventory not expanded"
        )
    if domain == "mega.nz" or domain.endswith(".mega.nz"):
        return _unresolved("encrypted_provider_metadata", "MEGA public metadata not decrypted")
    if domain in {"tii.la", "www.tii.la"}:
        return _unresolved("link_shortener", "Shortener is retained without navigation")
    return _unresolved("unsupported_provider_metadata", f"No metadata probe for {domain}")


def _probe_drive_file(
    client: httpx.Client, store: ContentAddressedStore, url: str, file_id: str
) -> dict[str, object]:
    body = _bounded_html(client, url)
    snapshot = store.put_bytes(body)
    soup = BeautifulSoup(body, "html.parser")
    title_meta = soup.select_one('meta[property="og:title"]')
    title = title_meta.get("content") if title_meta else None
    text = body.decode("utf-8", errors="replace")
    sizes = [int(match.group(1)) for match in DRIVE_SIZE.finditer(text)]
    unique_sizes = sorted(set(sizes))
    if len(unique_sizes) > 1:
        raise RuntimeError(f"Ambiguous Google Drive size metadata for {file_id}: {unique_sizes}")
    size = unique_sizes[0] if unique_sizes else None
    mime_match = re.search(r'itemJson:.*?"(application/[^"\\]+)"', text, re.DOTALL)
    return {
        "status": "resolved_single_file" if title and size else "single_file_metadata_incomplete",
        "file_id": file_id,
        "filename": title,
        "mime_type": mime_match.group(1) if mime_match else None,
        "declared_size_bytes": size,
        "declared_size_text": f"{size} bytes" if size else None,
        "direct_download_url": (
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
        ),
        "metadata_snapshot": _snapshot_record(snapshot),
    }


def _probe_mediafire_file(
    client: httpx.Client, store: ContentAddressedStore, url: str
) -> dict[str, object]:
    body = _bounded_html(client, url)
    snapshot = store.put_bytes(body)
    soup = BeautifulSoup(body, "html.parser")
    filename_node = soup.select_one(".filename")
    download = soup.select_one("a#downloadButton[href]")
    details = soup.select_one(".details")
    size_match = MEDIAFIRE_SIZE.search(details.get_text(" ", strip=True) if details else "")
    size_text = size_match.group(0) if size_match else None
    size = _parse_size(size_match.group(1), size_match.group(2)) if size_match else None
    filename = _plain(filename_node.get_text(" ", strip=True)) if filename_node else None
    return {
        "status": (
            "resolved_single_file"
            if filename and size and download is not None
            else "single_file_metadata_incomplete"
        ),
        "file_id": None,
        "filename": filename,
        "mime_type": None,
        "declared_size_bytes": size,
        "declared_size_text": size_text,
        "declared_size_parse_basis": "binary powers of 1024",
        "direct_download_url": download.get("href") if download else None,
        "metadata_snapshot": _snapshot_record(snapshot),
    }


def _unresolved(reason: str, note: str) -> dict[str, object]:
    return {
        "status": reason,
        "note": note,
        "file_id": None,
        "filename": None,
        "mime_type": None,
        "declared_size_bytes": None,
        "declared_size_text": None,
        "direct_download_url": None,
        "metadata_snapshot": None,
    }


def _bounded_html(client: httpx.Client, url: str) -> bytes:
    chunks = []
    total = 0
    with client.stream("GET", url) as response:
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").casefold()
        if "text/html" not in content_type:
            raise MetadataProbeError(
                f"Refusing non-HTML response {content_type or '<absent>'} from {response.url}"
            )
        declared = response.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_METADATA_BYTES:
            raise MetadataProbeError(
                f"Refusing declared metadata body of {declared} bytes from {response.url}"
            )
        for chunk in response.iter_bytes():
            total += len(chunk)
            if total > MAX_METADATA_BYTES:
                raise MetadataProbeError(
                    f"Refusing metadata body exceeding {MAX_METADATA_BYTES} bytes from "
                    f"{response.url}"
                )
            chunks.append(chunk)
    return b"".join(chunks)


def _snapshot_record(snapshot: object) -> dict[str, object]:
    return {
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "cas_path": str(snapshot.path),
        "already_present": snapshot.existed,
    }


def _parse_size(number: str, unit: str) -> int:
    scale = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}
    return round(float(number) * scale[unit.casefold()])


def _plain(value: str) -> str:
    return " ".join(value.split())


def _console(value: str) -> None:
    print(value.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _canonical_json(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with open(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
        if path.exists():
            raise FileExistsError(f"Refusing to replace metadata report: {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
