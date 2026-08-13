from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path
from typing import BinaryIO

type MediaSource = str | os.PathLike[str] | bytes | bytearray | memoryview | BinaryIO


def read_source_bytes(source: MediaSource) -> bytes:
    """Read a media source without consuming a seekable caller-owned stream."""

    if isinstance(source, str | os.PathLike):
        return Path(source).read_bytes()
    if isinstance(source, bytes):
        return source
    if isinstance(source, bytearray | memoryview):
        return bytes(source)
    if not hasattr(source, "read"):
        raise TypeError("source must be a path, bytes, or a binary file object")

    position: int | None = None
    with suppress(AttributeError, OSError):
        position = source.tell()
        source.seek(0)

    payload = source.read()
    if position is not None:
        with suppress(AttributeError, OSError):
            source.seek(position)
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise TypeError("binary media source returned non-bytes data")
    return bytes(payload)
