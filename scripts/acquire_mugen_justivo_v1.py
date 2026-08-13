from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.adapters.mugen import audit_character_zip  # noqa: E402
from spritelab.fetch import HttpFetcher  # noqa: E402
from spritelab.storage import ContentAddressedStore, DiskGuard  # noqa: E402

LANDING_URL = "https://mugen.justivo.com/"
OUTPUT = ROOT / "data/index/reports/mugen-justivo-acquisition-v2.json"


@dataclass(frozen=True)
class LandingCharacter:
    name: str
    game: str | None
    author: str | None
    download_url: str


class _LandingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[LandingCharacter] = []
        self._in_row = False
        self._in_big = False
        self._text: list[str] = []
        self._name: list[str] = []
        self._download: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag.casefold() == "tr":
            self._in_row = True
            self._text = []
            self._name = []
            self._download = None
        elif self._in_row and tag.casefold() == "big":
            self._in_big = True
        elif self._in_row and tag.casefold() == "a":
            href = attributes.get("href") or ""
            absolute = urljoin(LANDING_URL, href)
            if absolute.casefold().endswith(".zip") and self._download is None:
                self._download = absolute

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "big":
            self._in_big = False
        elif tag.casefold() == "tr" and self._in_row:
            text = " ".join(" ".join(self._text).split())
            name = " ".join(" ".join(self._name).split())
            if name and self._download:
                self.rows.append(
                    LandingCharacter(
                        name=name,
                        game=_field(text, "Game:", "Author:"),
                        author=_field(text, "Author:", "Download Here"),
                        download_url=self._download,
                    )
                )
            self._in_row = False

    def handle_data(self, data: str) -> None:
        if self._in_row:
            self._text.append(data)
            if self._in_big:
                self._name.append(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace acquisition index: {OUTPUT}")
    guard = DiskGuard(ROOT, 100 * 1024**3)
    store = ContentAddressedStore(ROOT / "data", guard)
    fetcher = HttpFetcher(
        store,
        user_agent="SpriteLab-Research/0.1 provenance-indexed noncommercial POC",
        timeout_seconds=120,
    )
    landing = fetcher.fetch(LANDING_URL)
    landing_bytes = landing.blob.path.read_bytes()
    landing_parser = _LandingParser()
    landing_parser.feed(landing_bytes.decode("iso-8859-1"))
    rows = tuple(landing_parser.rows)
    if not rows:
        raise RuntimeError("landing page yielded no ZIP character rows")
    if args.preflight_only:
        print(
            {
                "free_bytes": guard.status().free_bytes,
                "landing_sha256": landing.blob.sha256,
                "zip_rows": len(rows),
                "output_absent": True,
            }
        )
        return 0

    items: list[dict[str, object]] = []
    for position, row in enumerate(rows, 1):
        result = fetcher.fetch(row.download_url)
        payload = result.blob.path.read_bytes()
        try:
            audit = audit_character_zip(payload)
            audit_summary: dict[str, object] = {
                "status": "audited",
                "inventory_sha256": audit.inventory_sha256,
                "internal_name": audit.definition.name,
                "internal_display_name": audit.definition.display_name,
                "internal_author": audit.definition.author,
                "action_count": len(audit.actions),
                "frame_occurrence_count": sum(len(action.elements) for action in audit.actions),
                "sff_sha256": audit.sff_header.sha256,
                "sff_format_family": audit.sff_header.format_family,
                "executable_members": audit.executable_members,
                "runtime_logic_members": audit.runtime_logic_members,
            }
        except Exception as exc:  # preserve a failed pack as evidence, continue the collection
            audit_summary = {"status": "audit_failed", "error": f"{type(exc).__name__}: {exc}"}
        items.append(
            {
                "landing_claim": asdict(row),
                "retrieval": {
                    "requested_url": result.requested_url,
                    "final_url": result.final_url,
                    "status_code": result.status_code,
                    "mime_type": result.mime_type,
                    "content_length": result.content_length,
                    "etag": result.etag,
                    "last_modified": result.last_modified,
                },
                "archive": {
                    "sha256": result.blob.sha256,
                    "bytes": result.blob.size_bytes,
                    "cas_path": str(result.blob.path),
                },
                "audit": audit_summary,
            }
        )
        print(f"[{position}/{len(rows)}] {row.name}: {result.blob.size_bytes} bytes")

    artifact = {
        "schema_version": 1,
        "artifact_kind": "mugen_public_mirror_acquisition_index",
        "collection_scope": "Simple MUGEN 2004 individually listed ZIP mirrors only",
        "rights_scope": (
            "Unknown/unverified per pack; landing and internal author claims are evidence, "
            "not a permissive-license inference."
        ),
        "trust_boundary": "No character or runtime logic was executed.",
        "landing": {
            "url": LANDING_URL,
            "sha256": landing.blob.sha256,
            "bytes": landing.blob.size_bytes,
            "etag": landing.etag,
            "last_modified": landing.last_modified,
            "cas_path": str(landing.blob.path),
        },
        "item_count": len(items),
        "items": items,
    }
    payload = (
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    guard.require_capacity(len(payload), label="MUGEN acquisition index")
    _atomic_no_clobber(OUTPUT, payload)
    print({"items": len(items), "index_sha256": hashlib.sha256(payload).hexdigest()})
    return 0


def _field(text: str, start: str, end: str) -> str | None:
    if start not in text:
        return None
    value = text.split(start, 1)[1]
    if end in value:
        value = value.split(end, 1)[0]
    value = value.strip()
    return value or None


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
            raise FileExistsError(f"refusing to replace acquisition index: {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
