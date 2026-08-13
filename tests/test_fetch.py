from pathlib import Path

import httpx
import pytest

from spritelab.fetch import HttpFetcher
from spritelab.storage import ContentAddressedStore, DiskFloorReached, DiskGuard


def make_fetcher(
    tmp_path: Path,
    handler: httpx.MockTransport,
    *,
    floor: int = 0,
) -> tuple[HttpFetcher, ContentAddressedStore]:
    store = ContentAddressedStore(tmp_path, DiskGuard(tmp_path, floor))
    client = httpx.Client(transport=handler, follow_redirects=True)
    return (
        HttpFetcher(
            store,
            user_agent="spritelab-test",
            max_retries=0,
            chunk_bytes=2,
            client=client,
            sleep=lambda _seconds: None,
        ),
        store,
    )


def test_fetch_to_content_addressed_store(tmp_path: Path) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Length": "6", "Content-Type": "image/png"},
            content=b"abcdef",
            request=request,
        )
    )
    fetcher, _store = make_fetcher(tmp_path, transport)

    result = fetcher.fetch("https://example.invalid/sprite.png")

    assert result.blob.path.read_bytes() == b"abcdef"
    assert result.mime_type == "image/png"
    assert result.resumed_from == 0


def test_fetch_resumes_matching_byte_range(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Range"] == "bytes=3-"
        return httpx.Response(
            206,
            headers={"Content-Length": "3", "Content-Range": "bytes 3-5/6"},
            content=b"def",
            request=request,
        )

    transport = httpx.MockTransport(handler)
    fetcher, store = make_fetcher(tmp_path, transport)
    store.initialize()
    store.partial_path("https://example.invalid/sprite.png").write_bytes(b"abc")

    result = fetcher.fetch("https://example.invalid/sprite.png")

    assert result.blob.path.read_bytes() == b"abcdef"
    assert result.resumed_from == 3


def test_fetch_checks_disk_floor_before_body(tmp_path: Path) -> None:
    class FiveByteGuard(DiskGuard):
        def require_capacity(self, additional_bytes: int = 0, *, label: str = "write") -> None:
            if additional_bytes > 5:
                raise DiskFloorReached(f"Refusing {label} deterministically")

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"Content-Length": "6"},
            content=b"abcdef",
            request=request,
        )
    )
    fetcher, store = make_fetcher(tmp_path, transport)
    store.guard = FiveByteGuard(tmp_path, 0)

    with pytest.raises(DiskFloorReached):
        fetcher.fetch("https://example.invalid/too-large.png")
