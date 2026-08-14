from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_script_module():
    path = (
        Path(__file__).resolve().parents[1] / "scripts" / "materialize_mugen_manual_rar_core_v1.py"
    )
    spec = importlib.util.spec_from_file_location("mugen_manual_rar_core_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _character(variant: str, air_sha: str, sff_sha: str) -> dict[str, object]:
    return {
        "identity_id": f"identity-{variant}",
        "source": {
            "air": {"sha256": air_sha},
            "sff": {
                "crc32": "0123abcd",
                "sha256": sff_sha,
                "size_bytes": 123,
            },
        },
        "variant_id": variant,
    }


def test_catalog_fingerprint_uses_air_hash_and_declared_sff_identity() -> None:
    module = _load_script_module()

    assert module._catalog_fingerprint(
        {
            "air": {"sha256": "a" * 64},
            "sff": {"crc32": "0123abcd", "size_bytes": 123},
        }
    ) == ("a" * 64, 123, "0123abcd")


def test_known_exact_pairs_merge_current_journal_and_manifest(tmp_path: Path) -> None:
    module = _load_script_module()
    journal_root = tmp_path / "journal"
    journal_root.mkdir()
    journal_row = _character("variant-b", "a" * 64, "b" * 64)
    (journal_root / "character-records.jsonl").write_text(
        json.dumps(journal_row) + "\n", encoding="utf-8"
    )
    manifest_root = tmp_path / "manifest"
    manifest_root.mkdir()
    manifest_row = _character("variant-a", "a" * 64, "c" * 64)
    (manifest_root / "materialization.json").write_text(
        json.dumps({"characters": [manifest_row]}), encoding="utf-8"
    )

    result = module._known_exact_pairs(
        {"variant-c": _character("variant-c", "d" * 64, "e" * 64)},
        (journal_root, manifest_root),
    )

    shared = result[("a" * 64, 123, "0123abcd")]
    assert [row["variant_id"] for row in shared] == ["variant-a", "variant-b"]
    assert result[("d" * 64, 123, "0123abcd")][0]["variant_id"] == "variant-c"
