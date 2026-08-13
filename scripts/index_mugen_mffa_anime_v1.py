from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
CATEGORY_URL = "https://mugen.doomjoshuaboy.com/resources/categories/anime-manga-characters.16/"
OUTPUT = ROOT / "data/index/reports/mugen-mffa-anime-discovery-v2.json"
USER_AGENT = "SpriteLab-Research/0.1 provenance-indexed noncommercial POC"
_RESOURCE_ID = re.compile(r"\.(\d+)/$")
_SIZE = re.compile(r"^(\d+(?:\.\d+)?)\s*(B|KB|MB|GB)$", re.IGNORECASE)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace discovery index: {OUTPUT}")
    with httpx.Client(
        follow_redirects=True,
        timeout=60,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    ) as client:
        category_pages, resources = _crawl_category(client)
        records: list[dict[str, object]] = []
        for position, resource in enumerate(resources, 1):
            record = _inspect_resource(client, resource)
            records.append(record)
            print(
                f"[{position}/{len(resources)}] {record['title']}: "
                f"{len(record['download_candidates'])} candidates"
            )
            time.sleep(0.05)

    downloadable_bytes = sum(
        candidate["declared_bytes"] or 0
        for record in records
        for candidate in record["download_candidates"]
    )
    artifact = {
        "schema_version": 1,
        "artifact_kind": "mugen_mffa_anime_metadata_discovery_index",
        "category_url": CATEGORY_URL,
        "retrieval_policy": "Public pages only; metadata probing never reads archive bodies.",
        "rights_scope": "Uploader/author/category claims are evidence, not license grants.",
        "category_pages": category_pages,
        "resource_count": len(records),
        "download_candidate_count": sum(len(record["download_candidates"]) for record in records),
        "known_download_bytes": downloadable_bytes,
        "resources": records,
    }
    payload = (
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    free = _free_bytes()
    floor = 100 * 1024**3
    if free - len(payload) < floor:
        raise RuntimeError("discovery index would cross the 100-GiB disk floor")
    _atomic_no_clobber(OUTPUT, payload)
    print(
        {
            "resource_count": len(records),
            "download_candidate_count": artifact["download_candidate_count"],
            "known_download_bytes": downloadable_bytes,
            "index_sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
    return 0


def _crawl_category(
    client: httpx.Client,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    first = client.get(CATEGORY_URL)
    first.raise_for_status()
    soup = BeautifulSoup(first.text, "html.parser")
    pages = max(
        [
            int(a.get_text(strip=True))
            for a in soup.select(".pageNav a")
            if a.get_text(strip=True).isdigit()
        ],
        default=1,
    )
    category_pages: list[dict[str, object]] = []
    resources: dict[int, dict[str, str]] = {}
    for page in range(1, pages + 1):
        url = CATEGORY_URL if page == 1 else f"{CATEGORY_URL}?page={page}"
        response = first if page == 1 else client.get(url)
        response.raise_for_status()
        body = response.content
        category_pages.append(
            {"page": page, "url": str(response.url), "sha256": hashlib.sha256(body).hexdigest()}
        )
        page_soup = BeautifulSoup(body, "html.parser")
        for item in page_soup.select(".structItem--resource"):
            title_link = next(
                (
                    a
                    for a in item.select("a[href]")
                    if _RESOURCE_ID.search(urljoin(CATEGORY_URL, a.get("href", "")))
                    and a.get_text(" ", strip=True)
                ),
                None,
            )
            if title_link is None:
                continue
            resource_url = urljoin(CATEGORY_URL, title_link["href"])
            match = _RESOURCE_ID.search(resource_url)
            if match is None:
                continue
            resource_id = int(match.group(1))
            plain = " ".join(item.get_text(" ", strip=True).split())
            resources[resource_id] = {
                "resource_id": str(resource_id),
                "title": title_link.get_text(" ", strip=True),
                "url": resource_url,
                "listing_text": plain,
            }
        time.sleep(0.05)
    return category_pages, [resources[key] for key in sorted(resources)]


def _inspect_resource(client: httpx.Client, resource: dict[str, str]) -> dict[str, object]:
    response = client.get(resource["url"])
    response.raise_for_status()
    body = response.content
    soup = BeautifulSoup(body, "html.parser")
    canonical = soup.select_one('link[rel="canonical"]')
    download_link = next(
        (
            urljoin(str(response.url), a["href"])
            for a in soup.select("a[href]")
            if "/resources/" in a["href"] and a["href"].rstrip("/").casefold().endswith("/download")
        ),
        None,
    )
    candidates = _probe_download(client, download_link) if download_link else []
    description = soup.select_one(".resourceBody") or soup.select_one(".bbWrapper")
    return {
        **resource,
        "canonical_url": canonical["href"] if canonical else str(response.url),
        "page_sha256": hashlib.sha256(body).hexdigest(),
        "description_text": (
            " ".join(description.get_text(" ", strip=True).split()) if description else None
        ),
        "download_entry_url": download_link,
        "download_candidates": candidates,
    }


def _probe_download(client: httpx.Client, url: str) -> list[dict[str, object]]:
    with client.stream("GET", url) as response:
        content_type = (response.headers.get("content-type") or "").casefold()
        if "text/html" not in content_type:
            return [_candidate_from_response(response, display_name=None)]
        body = response.read()
    chooser = BeautifulSoup(body, "html.parser")
    results: list[dict[str, object]] = []
    for link in chooser.select('a[href*="/download"]'):
        if link.get_text(" ", strip=True).casefold() != "download":
            continue
        direct_url = urljoin(url, link["href"])
        container = link.parent
        name = (
            next(
                (
                    value.strip()
                    for value in container.stripped_strings
                    if value.strip().casefold() != "download" and _SIZE.match(value.strip()) is None
                ),
                None,
            )
            if container
            else None
        )
        size_text = (
            next(
                (
                    value.strip()
                    for value in container.stripped_strings
                    if _SIZE.match(value.strip())
                ),
                None,
            )
            if container
            else None
        )
        declared = _parse_size(size_text)
        with client.stream("GET", direct_url) as probe:
            candidate = _candidate_from_response(probe, display_name=name)
        if declared is not None:
            candidate["declared_bytes"] = declared
            candidate["declared_size_text"] = size_text
        results.append(candidate)
    return results


def _candidate_from_response(
    response: httpx.Response, display_name: str | None
) -> dict[str, object]:
    length = response.headers.get("content-length")
    filename = _content_disposition_filename(response.headers.get("content-disposition"))
    return {
        "display_name": display_name,
        "filename": filename,
        "requested_url": str(response.request.url),
        "final_url": str(response.url),
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type"),
        "declared_bytes": int(length) if length and length.isdigit() else None,
        "etag": response.headers.get("etag"),
        "last_modified": response.headers.get("last-modified"),
    }


def _content_disposition_filename(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', value, re.IGNORECASE)
    return match.group(1).strip() if match else None


def _parse_size(value: str | None) -> int | None:
    match = _SIZE.match(value or "")
    if not match:
        return None
    scale = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3}[match.group(2).casefold()]
    return round(float(match.group(1)) * scale)


def _free_bytes() -> int:
    return (
        os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
        if hasattr(os, "statvfs")
        else __import__("shutil").disk_usage(ROOT).free
    )


def _atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to replace discovery index: {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
