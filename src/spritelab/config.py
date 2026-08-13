from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIB = 1024**3
MIB = 1024**2


def project_root() -> Path:
    configured = os.environ.get("SPRITELAB_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class StorageConfig:
    data_root: Path
    min_free_bytes: int
    download_chunk_bytes: int


@dataclass(frozen=True)
class NetworkConfig:
    user_agent: str
    timeout_seconds: float
    max_retries: int
    default_requests_per_second: float


@dataclass(frozen=True)
class IndexConfig:
    database: Path


@dataclass(frozen=True)
class NormalizationConfig:
    default_width: int
    default_height: int
    default_frames: int
    resampling: str


@dataclass(frozen=True)
class Config:
    project_root: Path
    storage: StorageConfig
    network: NetworkConfig
    index: IndexConfig
    normalization: NormalizationConfig


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_config(config_path: Path | None = None) -> Config:
    root = project_root()
    default_path = config_path or root / "configs" / "default.toml"
    raw = _read_toml(default_path)
    raw = _merge(raw, _read_toml(root / "config.local.toml"))

    storage_raw = raw.get("storage", {})
    data_root_raw = os.environ.get("SPRITELAB_DATA_ROOT", storage_raw.get("data_root", "data"))
    data_root = Path(data_root_raw).expanduser()
    if not data_root.is_absolute():
        data_root = root / data_root
    data_root = data_root.resolve()

    min_free_gib = float(
        os.environ.get("SPRITELAB_MIN_FREE_GIB", storage_raw.get("min_free_gib", 100.0))
    )
    database_raw = raw.get("index", {}).get("database", "index/spritelab.sqlite3")
    database = Path(database_raw)
    if not database.is_absolute():
        database = data_root / database

    network_raw = raw.get("network", {})
    normalization_raw = raw.get("normalization", {})
    return Config(
        project_root=root,
        storage=StorageConfig(
            data_root=data_root,
            min_free_bytes=int(min_free_gib * GIB),
            download_chunk_bytes=int(storage_raw.get("download_chunk_mib", 1) * MIB),
        ),
        network=NetworkConfig(
            user_agent=str(network_raw.get("user_agent", "SpriteDiffusionResearch/0.1")),
            timeout_seconds=float(network_raw.get("timeout_seconds", 60)),
            max_retries=int(network_raw.get("max_retries", 4)),
            default_requests_per_second=float(network_raw.get("default_requests_per_second", 0.5)),
        ),
        index=IndexConfig(database=database.resolve()),
        normalization=NormalizationConfig(
            default_width=int(normalization_raw.get("default_width", 64)),
            default_height=int(normalization_raw.get("default_height", 64)),
            default_frames=int(normalization_raw.get("default_frames", 8)),
            resampling=str(normalization_raw.get("resampling", "nearest")),
        ),
    )
