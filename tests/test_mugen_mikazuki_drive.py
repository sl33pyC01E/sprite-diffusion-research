from __future__ import annotations

from dataclasses import dataclass

import pytest

import scripts.acquire_mugen_mikazuki_drive_v1 as subject


@dataclass
class _Response:
    status_code: int
    headers: dict[str, str]
    body: bytes = b""

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def iter_bytes(self, _chunk_size: int):
        yield self.body

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.requests: list[tuple[str, dict[str, str]]] = []

    def stream(self, method: str, _url: str, *, headers: dict[str, str] | None = None) -> _Response:
        self.requests.append((method, headers or {}))
        return self.responses.pop(0)


def test_download_accepts_exact_full_stream_after_verifying_partial_prefix(tmp_path) -> None:
    partial = tmp_path / "archive.part"
    partial.write_bytes(b"abc")
    client = _Client([_Response(200, {"content-length": "6"}, b"abcdef")])

    transport = subject._download(
        client,
        "https://drive.usercontent.google.com/initial",
        partial,
        indexed_direct_url="https://drive.usercontent.google.com/download?id=x",
        existing=3,
        expected_size=6,
        floor=0,
    )

    assert partial.read_bytes() == b"abcdef"
    assert transport == {
        "confirmation_refresh_html_sha256": [],
        "used_full_stream_verified_prefix": True,
    }
    assert client.requests == [("GET", {"Range": "bytes=3-5"})]


def test_full_stream_prefix_mismatch_does_not_append(tmp_path) -> None:
    partial = tmp_path / "archive.part"
    partial.write_bytes(b"abc")
    client = _Client([_Response(200, {"content-length": "6"}, b"abXdef")])

    with pytest.raises(RuntimeError, match="differs from partial"):
        subject._download(
            client,
            "https://drive.usercontent.google.com/initial",
            partial,
            indexed_direct_url="https://drive.usercontent.google.com/download?id=x",
            existing=3,
            expected_size=6,
            floor=0,
        )

    assert partial.read_bytes() == b"abc"


def test_throttled_range_cools_down_and_refreshes_confirmation(tmp_path, monkeypatch) -> None:
    partial = tmp_path / "archive.part"
    partial.write_bytes(b"abc")
    client = _Client(
        [
            _Response(200, {"content-length": "123"}),
            _Response(
                206,
                {"content-length": "3", "content-range": "bytes 3-5/6"},
                b"def",
            ),
        ]
    )
    sleeps: list[int] = []
    monkeypatch.setattr(subject.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        subject,
        "_resolve_drive_confirmation",
        lambda _client, _url: {
            "confirmation_html_sha256": "a" * 64,
            "download_url": "https://drive.usercontent.google.com/refreshed",
        },
    )

    transport = subject._download(
        client,
        "https://drive.usercontent.google.com/initial",
        partial,
        indexed_direct_url="https://drive.usercontent.google.com/download?id=x",
        existing=3,
        expected_size=6,
        floor=0,
    )

    assert partial.read_bytes() == b"abcdef"
    assert transport["confirmation_refresh_html_sha256"] == ["a" * 64]
    assert sleeps == [15]


def test_unconditional_full_stream_writes_exact_object(tmp_path) -> None:
    partial = tmp_path / "archive.full.part"
    client = _Client([_Response(200, {"content-type": "application/octet-stream"}, b"abcdef")])

    subject._download_full_stream(
        client,
        "https://drive.usercontent.google.com/full",
        partial,
        expected_size=6,
        floor=0,
    )

    assert partial.read_bytes() == b"abcdef"
    assert client.requests == [("GET", {})]
