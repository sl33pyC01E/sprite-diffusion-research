"""Minimal Decord-compatible reader for the ARM64 Spark training host."""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

__version__ = "spritelab-pyav-shim-v1"


class _Bridge:
    def set_bridge(self, name: str) -> None:
        if name != "torch":
            raise ValueError("the SpriteLab Decord shim only supports the Torch bridge")


bridge = _Bridge()


class VideoReader:
    """Eager RGB video reader implementing the trainer's Decord subset."""

    def __init__(self, uri: str, *, width: int, height: int) -> None:
        path = Path(uri)
        if not path.is_file() or width <= 0 or height <= 0:
            raise ValueError("video path and output geometry must be valid")
        frames = []
        with av.open(str(path), mode="r") as container:
            for frame in container.decode(video=0):
                value = frame.to_ndarray(format="rgb24")
                if value.shape[:2] != (height, width):
                    value = np.asarray(
                        Image.fromarray(value).resize((width, height), Image.Resampling.BILINEAR),
                        dtype=np.uint8,
                    )
                frames.append(torch.from_numpy(np.ascontiguousarray(value)))
        if not frames:
            raise ValueError(f"video contains no decoded frames: {path}")
        self._frames = torch.stack(frames)

    def __len__(self) -> int:
        return int(self._frames.shape[0])

    def get_batch(self, indices: list[int]) -> torch.Tensor:
        if not indices or any(index < 0 or index >= len(self) for index in indices):
            raise IndexError("video frame indices are invalid")
        return self._frames[indices]
