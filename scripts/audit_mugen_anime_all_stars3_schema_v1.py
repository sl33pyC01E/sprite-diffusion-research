"""Publish the exact AIR schema catalog for extracted Anime All Stars 3 fighters."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_directory import audit_mugen_directory  # noqa: E402
from spritelab.mugen_schema import (  # noqa: E402
    canonical_six_slot_action_numbers,
    measure_core_schema_coverage,
    schema_phase,
    schema_verb,
)
from spritelab.storage import DiskGuard  # noqa: E402

ARCHIVE_REPORT = ROOT / "data/raw/acquisitions/mikazuki-section-13-v1.json"
INVENTORY = ROOT / "data/index/reports/mugen-anime-all-stars-3-inventory-verbose-v1.txt"
COLLECTION_ROOT = (
    ROOT
    / "data/staging/mugen-anime-all-stars-3-v1"
    / "Anime All Stars 3 (Requested By Pands)/chars"
)
SUPERSEDED = ROOT / "data/index/reports/mugen-anime-all-stars-3-air-schema-catalog-v1.json"
SUPERSEDED_SHA256 = "dde2e67ad22be6e9e0522b36f8955076e18b7fae219c044bfb99bb2cd737e430"
OUTPUT = ROOT / "data/index/reports/mugen-anime-all-stars-3-air-schema-catalog-v2.json"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace schema catalog: {OUTPUT}")
    guard = DiskGuard(ROOT, 100 * 1024**3)
    if _file_sha256(SUPERSEDED) != SUPERSEDED_SHA256:
        raise ValueError("superseded Anime All Stars 3 catalog SHA-256 differs")
    archive_report_bytes = ARCHIVE_REPORT.read_bytes()
    archive_report = json.loads(archive_report_bytes)
    archive = Path(archive_report["archive"]["cas_path"])
    _verify_file(
        archive,
        size=archive_report["archive"]["size_bytes"],
        sha256=archive_report["archive"]["sha256"],
    )
    audit = audit_mugen_directory(COLLECTION_ROOT)
    records = []
    slot_counts: Counter[str] = Counter()
    complete = 0
    total_actions = 0
    for variant in audit.variants:
        coverage = measure_core_schema_coverage(variant.actions)
        canonical = canonical_six_slot_action_numbers(coverage)
        complete += int(coverage.complete_six_slot_core)
        total_actions += len(variant.actions)
        for slot in ("idle", "walk", "jump", "block", "attack"):
            if getattr(coverage, f"{slot}_action_numbers"):
                slot_counts[slot] += 1
        identity_id = "mugen_" + variant.sff_sha256[:32]
        stable_variant = {
            "air_sha256": variant.air_sha256,
            "archive_sha256": archive_report["archive"]["sha256"],
            "sff_sha256": variant.sff_sha256,
        }
        variant_id = (
            "mugen_variant_"
            + hashlib.sha256(_canonical(stable_variant).rstrip(b"\n")).hexdigest()[:32]
        )
        definitions = []
        for path, definition in zip(variant.definition_paths, variant.definitions, strict=True):
            definitions.append(
                {
                    "author": definition.author,
                    "display_name": definition.display_name,
                    "file_sha256": _file_sha256(COLLECTION_ROOT / path),
                    "member_path": _archive_member(path),
                    "name": definition.name,
                    "path": path,
                    "source_comments": list(definition.source_comments),
                }
            )
        records.append(
            {
                "actions": [
                    _action_record(action, index) for index, action in enumerate(variant.actions)
                ],
                "air": {
                    "member_path": _archive_member(variant.air_path),
                    "path": variant.air_path,
                    "sha256": variant.air_sha256,
                },
                "canonical_six_slot_action_numbers": canonical,
                "core_coverage": {
                    **asdict(coverage),
                    "complete_six_slot_core": coverage.complete_six_slot_core,
                },
                "definitions": definitions,
                "identity_id": identity_id,
                "sff": {
                    "bytes": variant.sff_bytes,
                    "format_family": variant.sff_header.format_family,
                    "member_path": _archive_member(variant.sff_path),
                    "path": variant.sff_path,
                    "sha256": variant.sff_sha256,
                    "version_bytes": list(variant.sff_header.version_bytes),
                },
                "variant_id": variant_id,
            }
        )
    records.sort(key=lambda row: row["variant_id"].encode("utf-8"))
    artifact = {
        "artifact_kind": "mugen_anime_all_stars3_air_schema_catalog",
        "characters": records,
        "counts": {
            "actions": total_actions,
            "complete_six_slot_characters": complete,
            "definitions": audit.definition_count,
            "definition_failures": len(audit.failures),
            "incomplete_six_slot_characters": len(records) - complete,
            "slot_character_coverage": dict(sorted(slot_counts.items())),
            "unique_characters": len(records),
        },
        "definition_failures": [asdict(row) for row in audit.failures],
        "policy": {
            "admission": "every distinct DEF-resolved AIR/SFF pair with parseable media",
            "core_view": "idle, walk, jump, block, attack_a, attack_b",
            "exclusion": "no character is excluded for missing a core slot",
            "rights_scope": "unknown/unverified fan collection; no permissive inference",
            "runtime": "CMD/CNS/ST and executable content are never interpreted or executed",
            "timing": "raw authored AIR ticks retained, including unsupported negative values",
        },
        "schema_version": 2,
        "source": {
            "acquisition_report_path": str(ARCHIVE_REPORT),
            "acquisition_report_sha256": hashlib.sha256(archive_report_bytes).hexdigest(),
            "archive_sha256": archive_report["archive"]["sha256"],
            "archive_size_bytes": archive_report["archive"]["size_bytes"],
            "inventory_path": str(INVENTORY),
            "inventory_sha256": _file_sha256(INVENTORY),
            "landing_url": archive_report["provider"]["landing_url"],
            "section_source_ordinal": archive_report["source"]["section_source_ordinal"],
            "superseded_catalog_path": str(SUPERSEDED),
            "superseded_catalog_sha256": SUPERSEDED_SHA256,
        },
    }
    payload = _canonical(artifact)
    guard.require_capacity(len(payload), label="Anime All Stars 3 schema catalog")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "counts": artifact["counts"],
                "path": str(OUTPUT),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


def _action_record(action: object, source_action_index: int) -> dict[str, object]:
    return {
        "action_number": action.action_number,
        "collision_1_declarations": action.collision_1_declarations,
        "collision_2_declarations": action.collision_2_declarations,
        "elements": [asdict(element) for element in action.elements],
        "finite_duration_ticks": action.finite_duration_ticks,
        "label": asdict(action.label),
        "loop_mode": action.loop_mode,
        "loop_start_index": action.loop_start_index,
        "schema_phase": schema_phase(action.action_number),
        "schema_verb": schema_verb(action.action_number),
        "source_action_index": source_action_index,
        "source_comments": list(action.source_comments),
    }


def _archive_member(relative: str) -> str:
    return f"Anime All Stars 3 (Requested By Pands)/chars/{relative}"


def _verify_file(path: Path, *, size: int, sha256: str) -> None:
    if path.stat().st_size != size or _file_sha256(path) != sha256:
        raise ValueError(f"source archive identity differs: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
