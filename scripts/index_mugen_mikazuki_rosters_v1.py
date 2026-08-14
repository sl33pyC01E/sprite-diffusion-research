"""Index Mikazuki's public MUGEN roster page without downloading archives."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.storage import ContentAddressedStore, DiskGuard  # noqa: E402

LANDING_URL = "https://sites.google.com/view/mikazukithemugenitecreations/mugen-rosters"
OUTPUT = ROOT / "data/index/reports/mugen-mikazuki-roster-discovery-v2.json"
USER_AGENT = "SpriteLab-Research/0.1 provenance-indexed noncommercial POC"
CHARACTER_COUNT = re.compile(
    r"(?:over\s*)?(\d[\d,]*)\s*\+?"
    r"(?:\s+[A-Za-z][A-Za-z0-9_-]*){0,3}\s+(?:chars?|characters?)\b",
    re.IGNORECASE,
)
ANIME_KEYWORDS = (
    "anime",
    "bleach",
    "dragon ball",
    "fate",
    "hatsune miku",
    "jojo",
    "jus",
    "manga",
    "melty blood",
    "naruto",
    "one piece",
    "touhou",
)
DOWNLOAD_DOMAINS = (
    "1024tera.com",
    "1024terabox.com",
    "drive.google.com",
    "dropbox.com",
    "mega.nz",
    "mediafire.com",
    "tii.la",
)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace discovery index: {OUTPUT}")
    with httpx.Client(
        follow_redirects=True,
        timeout=90,
        headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
    ) as client:
        response = client.get(LANDING_URL)
        response.raise_for_status()
    body = response.content
    guard = DiskGuard(ROOT, 100 * 1024**3)
    snapshot = ContentAddressedStore(ROOT / "data", guard).put_bytes(body)
    sections = parse_sections(body, str(response.url))
    links = [link for section in sections for link in section["links"]]
    download_links = [link for link in links if link["role"] == "archive_download_entry"]
    unique_download_urls = sorted({link["url"] for link in download_links})
    artifact = {
        "artifact_kind": "mugen_mikazuki_public_roster_metadata_discovery_index",
        "schema_version": 1,
        "retrieval": {
            "requested_url": LANDING_URL,
            "final_url": str(response.url),
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "status_code": response.status_code,
            "content_type": response.headers.get("content-type"),
            "etag": response.headers.get("etag"),
            "last_modified": response.headers.get("last-modified"),
        },
        "landing_snapshot": {
            "sha256": snapshot.sha256,
            "size_bytes": snapshot.size_bytes,
            "cas_path": str(snapshot.path),
            "already_present": snapshot.existed,
        },
        "scope": {
            "policy": "Public landing-page metadata only; archive bodies are not read.",
            "rights": (
                "Uploader, title, roster membership, and link presence are evidence only; "
                "they do not establish creator identity or a reuse license."
            ),
            "youtube": "Preview URLs are metadata only and are not downloaded.",
            "relevance_heuristic": {
                "method": "case-insensitive title keyword match",
                "keywords": list(ANIME_KEYWORDS),
                "meaning": "discovery priority only; not a semantic or quality ground truth",
            },
        },
        "counts": {
            "sections": len(sections),
            "sections_with_downloads": sum(bool(section["download_count"]) for section in sections),
            "anime_or_jus_priority_sections": sum(
                section["anime_or_jus_priority"] for section in sections
            ),
            "download_link_occurrences": len(download_links),
            "unique_download_urls": len(unique_download_urls),
            "preview_link_occurrences": sum(link["role"] == "youtube_preview" for link in links),
            "maximum_claimed_character_count": max(
                (section["claimed_character_count"] or 0 for section in sections), default=0
            ),
        },
        "unique_download_urls": unique_download_urls,
        "sections": sections,
    }
    payload = _canonical_json(artifact)
    guard.require_capacity(len(payload), label="Mikazuki roster discovery index")
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


def parse_sections(body: bytes, base_url: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(body, "html.parser")
    current: dict[str, object] | None = None
    sections: list[dict[str, object]] = []
    for node in soup.select("h1,h2,h3,a[href]"):
        if node.name in {"h1", "h2", "h3"}:
            title = _plain(node.get_text(" ", strip=True))
            if not title:
                continue
            if current is not None and current["links"]:
                sections.append(_finish_section(current))
            current = {"title": title, "heading_level": node.name, "links": []}
            continue
        if current is None:
            continue
        href = urljoin(base_url, node.get("href", ""))
        parsed = urlparse(href)
        if parsed.scheme not in {"http", "https"}:
            continue
        role = _link_role(parsed.netloc.casefold(), _plain(node.get_text(" ", strip=True)))
        if role is None:
            continue
        link = {
            "anchor_text": _plain(node.get_text(" ", strip=True)) or None,
            "domain": parsed.netloc.casefold(),
            "role": role,
            "url": href,
        }
        if link not in current["links"]:
            current["links"].append(link)
    if current is not None and current["links"]:
        sections.append(_finish_section(current))
    for ordinal, section in enumerate(sections):
        section["source_ordinal"] = ordinal
    return sections


def _finish_section(section: dict[str, object]) -> dict[str, object]:
    title = str(section["title"])
    links = list(section["links"])
    download_count = sum(link["role"] == "archive_download_entry" for link in links)
    count_match = CHARACTER_COUNT.search(title)
    return {
        "title": title,
        "heading_level": section["heading_level"],
        "title_sha256": hashlib.sha256(title.encode("utf-8")).hexdigest(),
        "anime_or_jus_priority": any(keyword in title.casefold() for keyword in ANIME_KEYWORDS),
        "claimed_character_count": (
            int(count_match.group(1).replace(",", "")) if count_match else None
        ),
        "download_count": download_count,
        "links": links,
    }


def _link_role(domain: str, anchor_text: str) -> str | None:
    if domain.endswith("youtube.com") or domain == "youtu.be":
        return "youtube_preview"
    if any(
        domain == candidate or domain.endswith(f".{candidate}") for candidate in DOWNLOAD_DOMAINS
    ):
        return "archive_download_entry"
    return None


def _plain(value: str) -> str:
    return " ".join(value.split())


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
            raise FileExistsError(f"Refusing to replace discovery index: {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
