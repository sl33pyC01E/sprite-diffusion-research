from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from spritelab.db import IndexDB
from spritelab.storage import ContentAddressedStore, DiskFloorReached, StoredBlob

_CONTENT_RANGE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$", re.IGNORECASE)
_UNSATISFIED_RANGE = re.compile(r"^bytes\s+\*/(\d+)$", re.IGNORECASE)
_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


class FetchError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class FetchResult:
    blob: StoredBlob
    requested_url: str
    final_url: str
    status_code: int
    mime_type: str | None
    content_length: int | None
    etag: str | None
    last_modified: str | None
    resumed_from: int


class HttpFetcher:
    """Resumable HTTP fetches into the immutable object store.

    A deterministic partial file is retained after transient failures. Servers that
    honor byte ranges resume it; servers that do not are safely restarted.
    """

    def __init__(
        self,
        store: ContentAddressedStore,
        *,
        user_agent: str,
        timeout_seconds: float = 60,
        max_retries: int = 4,
        chunk_bytes: int = 1024 * 1024,
        client: httpx.Client | None = None,
        extra_headers: dict[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        self.store = store
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.chunk_bytes = chunk_bytes
        self._client = client
        self.extra_headers = dict(extra_headers or {})
        self._sleep = sleep

    def fetch(self, url: str, *, expected_sha256: str | None = None) -> FetchResult:
        self.store.initialize()
        partial = self.store.partial_path(url)
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._attempt(url, partial, expected_sha256=expected_sha256)
            except DiskFloorReached:
                raise
            except (httpx.HTTPError, FetchError) as error:
                last_error = error
                status = error.status_code if isinstance(error, FetchError) else None
                if attempt >= self.max_retries or (
                    status is not None and status not in _RETRYABLE_STATUS
                ):
                    raise
                self._sleep(min(2**attempt, 30))
        raise FetchError(f"Failed to fetch {url}: {last_error}")

    def _attempt(
        self,
        url: str,
        partial: Path,
        *,
        expected_sha256: str | None,
    ) -> FetchResult:
        existing = partial.stat().st_size if partial.exists() else 0
        headers = {
            **self.extra_headers,
            "User-Agent": self.user_agent,
            "Accept-Encoding": "identity",
        }
        if existing:
            headers["Range"] = f"bytes={existing}-"

        if self._client is None:
            with httpx.Client(
                follow_redirects=True,
                timeout=self.timeout_seconds,
            ) as client:
                return self._stream_response(
                    client,
                    url,
                    partial,
                    existing,
                    headers,
                    expected_sha256,
                )
        return self._stream_response(
            self._client,
            url,
            partial,
            existing,
            headers,
            expected_sha256,
        )

    def _stream_response(
        self,
        client: httpx.Client,
        url: str,
        partial: Path,
        existing: int,
        headers: dict[str, str],
        expected_sha256: str | None,
    ) -> FetchResult:
        with client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416 and existing:
                total = _parse_unsatisfied_total(response.headers.get("Content-Range"))
                if total == existing:
                    blob = self.store.commit_partial(partial, expected_sha256=expected_sha256)
                    return _result(response, url, blob, existing)
            if response.status_code in _RETRYABLE_STATUS:
                raise FetchError(
                    f"Retryable HTTP {response.status_code} for {url}",
                    status_code=response.status_code,
                )
            if response.status_code < 200 or response.status_code >= 300:
                raise FetchError(
                    f"HTTP {response.status_code} for {url}",
                    status_code=response.status_code,
                )

            append = response.status_code == 206 and existing > 0
            if response.status_code == 206:
                range_start, range_total = _parse_content_range(
                    response.headers.get("Content-Range")
                )
                if range_start != existing:
                    raise FetchError(f"Server resumed {url} at {range_start}, expected {existing}")
            else:
                range_total = None
                append = False

            remaining = _parse_nonnegative_int(response.headers.get("Content-Length"))
            if remaining is not None:
                self.store.guard.require_capacity(remaining, label=f"download {url}")

            mode = "ab" if append else "wb"
            resumed_from = existing if append else 0
            with partial.open(mode) as handle:
                for chunk in response.iter_bytes(self.chunk_bytes):
                    if not chunk:
                        continue
                    self.store.guard.require_capacity(len(chunk), label=f"download chunk {url}")
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())

            completed_size = partial.stat().st_size
            declared_total = range_total if range_total is not None else remaining
            if declared_total is not None and completed_size != declared_total:
                raise FetchError(
                    f"Incomplete response for {url}: expected {declared_total} bytes, "
                    f"have {completed_size}"
                )
            blob = self.store.commit_partial(partial, expected_sha256=expected_sha256)
            return _result(response, url, blob, resumed_from)


def fetch_indexed(
    *,
    fetcher: HttpFetcher,
    database: IndexDB,
    url: str,
    run_id: str | None = None,
    item_id: str | None = None,
    role: str = "original",
    original_filename: str | None = None,
    expected_sha256: str | None = None,
) -> tuple[str, FetchResult]:
    """Fetch a URL and atomically record its retrieval/blob provenance facts."""
    database.initialize()
    retrieval_id = database.start_retrieval(url=url, run_id=run_id, item_id=item_id)
    try:
        result = fetcher.fetch(url, expected_sha256=expected_sha256)
        database.register_blob(
            sha256=result.blob.sha256,
            size_bytes=result.blob.size_bytes,
            storage_path=result.blob.path,
            mime_type=result.mime_type,
        )
        database.finish_retrieval(
            retrieval_id,
            status_code=result.status_code,
            etag=result.etag,
            last_modified=result.last_modified,
            mime_type=result.mime_type,
            content_length=result.content_length,
            blob_sha256=result.blob.sha256,
        )
        if item_id is not None:
            database.link_item_blob(
                item_id=item_id,
                blob_sha256=result.blob.sha256,
                role=role,
                original_url=url,
                original_filename=original_filename,
            )
        return retrieval_id, result
    except BaseException as error:
        database.finish_retrieval(
            retrieval_id,
            status_code=getattr(error, "status_code", None),
            error=f"{type(error).__name__}: {error}",
        )
        raise


def _parse_nonnegative_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _parse_content_range(value: str | None) -> tuple[int, int | None]:
    match = _CONTENT_RANGE.match(value or "")
    if not match:
        raise FetchError(f"Invalid Content-Range header: {value!r}")
    start = int(match.group(1))
    end = int(match.group(2))
    total = None if match.group(3) == "*" else int(match.group(3))
    if end < start or (total is not None and end >= total):
        raise FetchError(f"Invalid Content-Range bounds: {value!r}")
    return start, total


def _parse_unsatisfied_total(value: str | None) -> int | None:
    match = _UNSATISFIED_RANGE.match(value or "")
    return int(match.group(1)) if match else None


def _result(
    response: httpx.Response,
    requested_url: str,
    blob: StoredBlob,
    resumed_from: int,
) -> FetchResult:
    return FetchResult(
        blob=blob,
        requested_url=requested_url,
        final_url=str(response.url),
        status_code=response.status_code,
        mime_type=response.headers.get("Content-Type", "").split(";", 1)[0] or None,
        content_length=blob.size_bytes,
        etag=response.headers.get("ETag"),
        last_modified=response.headers.get("Last-Modified"),
        resumed_from=resumed_from,
    )
