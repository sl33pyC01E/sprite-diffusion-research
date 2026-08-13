from __future__ import annotations

import hashlib
import io

import numpy as np

from spritelab.latent_cache import latent_channel_statistics


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def test_latent_channel_statistics_verify_files_and_compute_per_channel(tmp_path) -> None:
    records = []
    for index, level in enumerate((1.0, 3.0)):
        value = np.empty((8, 8, 64, 64), dtype=np.float16)
        for channel in range(8):
            value[:, channel] = level + channel
        payload = _npy_bytes(value)
        relative = f"latents/{index}.npy"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append(
            {
                "array_content_sha256": _array_sha256(value),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "relative_path": relative,
            }
        )

    statistics = latent_channel_statistics(tmp_path, records)

    assert statistics["channel_mean"] == [2.0 + channel for channel in range(8)]
    assert statistics["channel_standard_deviation"] == [1.0] * 8
    assert statistics["scalar_count_per_channel"] == 16 * 64 * 64
