from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from spritelab.db import IndexDB


@dataclass(frozen=True)
class SourceDefinition:
    id: str
    name: str
    root_url: str
    adapter: str
    acquisition_mode: str
    default_requests_per_second: float
    terms_url: str
    license_url: str
    quality_tier: str
    notes: str

    def as_config(self) -> dict[str, Any]:
        return {
            "acquisition_mode": self.acquisition_mode,
            "default_requests_per_second": self.default_requests_per_second,
            "terms_url": self.terms_url,
            "license_url": self.license_url,
            "quality_tier": self.quality_tier,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SourceRegistry:
    schema_version: int
    snapshot_date: str
    sources: tuple[SourceDefinition, ...]

    def by_id(self, source_id: str) -> SourceDefinition:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(f"Unknown source: {source_id}")


def load_source_registry(path: Path) -> SourceRegistry:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    registry = raw.get("registry", {})
    definitions: list[SourceDefinition] = []
    seen: set[str] = set()
    for position, source_raw in enumerate(raw.get("sources", []), start=1):
        definition = _parse_source(source_raw, position)
        if definition.id in seen:
            raise ValueError(f"Duplicate source id: {definition.id}")
        seen.add(definition.id)
        definitions.append(definition)
    if not definitions:
        raise ValueError("Source registry contains no sources")
    return SourceRegistry(
        schema_version=int(registry.get("schema_version", 0)),
        snapshot_date=str(registry.get("snapshot_date", "")),
        sources=tuple(definitions),
    )


def sync_source_registry(database: IndexDB, registry: SourceRegistry) -> int:
    for source in registry.sources:
        database.register_source(
            source_id=source.id,
            kind=source.adapter,
            name=source.name,
            root_url=source.root_url,
            adapter_version=f"registry-v{registry.schema_version}",
            config=source.as_config(),
        )
    return len(registry.sources)


def _parse_source(raw: dict[str, Any], position: int) -> SourceDefinition:
    required = (
        "id",
        "name",
        "root_url",
        "adapter",
        "acquisition_mode",
        "default_requests_per_second",
        "terms_url",
        "license_url",
        "quality_tier",
        "notes",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Source #{position} lacks fields: {', '.join(missing)}")
    source_id = str(raw["id"])
    if not source_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in source_id
    ):
        raise ValueError(f"Invalid source id: {source_id!r}")
    rate = float(raw["default_requests_per_second"])
    if rate < 0:
        raise ValueError(f"Negative request rate for source {source_id}")
    for field in ("root_url", "terms_url", "license_url"):
        value = str(raw[field])
        if urlparse(value).scheme not in {"http", "https"}:
            raise ValueError(f"Invalid {field} for source {source_id}: {value!r}")
    return SourceDefinition(
        id=source_id,
        name=str(raw["name"]),
        root_url=str(raw["root_url"]),
        adapter=str(raw["adapter"]),
        acquisition_mode=str(raw["acquisition_mode"]),
        default_requests_per_second=rate,
        terms_url=str(raw["terms_url"]),
        license_url=str(raw["license_url"]),
        quality_tier=str(raw["quality_tier"]),
        notes=str(raw["notes"]),
    )
