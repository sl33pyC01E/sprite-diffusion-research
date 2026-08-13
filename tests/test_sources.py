from pathlib import Path

from spritelab.db import IndexDB
from spritelab.sources import load_source_registry, sync_source_registry


def test_project_source_registry_is_valid_and_syncable(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry = load_source_registry(project_root / "configs" / "sources.toml")

    assert len(registry.sources) >= 30
    assert len({source.id for source in registry.sources}) == len(registry.sources)
    assert registry.by_id("universal_lpc").quality_tier == "A_lossless_open"

    database = IndexDB(tmp_path / "index.sqlite3")
    assert sync_source_registry(database, registry) == len(registry.sources)
    assert database.counts()["sources"] == len(registry.sources)
