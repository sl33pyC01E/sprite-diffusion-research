"""Build a resumable, model-independent AIR schema catalog for all decoded MFFA fighters."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.materialize_mugen_mffa_training_v1 as zip_common  # noqa: E402
from scripts.audit_mugen_mffa_rar7z_v1 import _inventory, _synthetic_character  # noqa: E402
from spritelab.adapters.mugen import audit_character_zip, audit_character_zip_variants  # noqa: E402
from spritelab.mugen_schema import (  # noqa: E402
    canonical_six_slot_action_numbers,
    measure_core_schema_coverage,
    schema_phase,
    schema_verb,
)
from spritelab.storage import DiskGuard  # noqa: E402

ZIP_SOURCE = ROOT / "data/index/reports/mugen-mffa-anime-zip-acquisition-v1.json"
RAR_SOURCE = ROOT / "data/index/reports/mugen-mffa-anime-rar7z-acquisition-v1.json"
ZIP_AUDIT = ROOT / "data/index/reports/mugen-mffa-anime-zip-corpus-audit-v3.json"
RAR_AUDIT = ROOT / "data/index/reports/mugen-mffa-anime-rar7z-corpus-audit-v1.json"
OUTPUT = ROOT / "data/index/reports/mugen-mffa-air-schema-catalog-v1.json"
CHARACTER_JOURNAL = ROOT / "data/index/reports/mugen-mffa-air-schema-catalog-v1.jsonl"
PACK_JOURNAL = ROOT / "data/index/reports/mugen-mffa-air-schema-pack-status-v1.jsonl"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace AIR schema catalog: {OUTPUT}")
    guard = DiskGuard(ROOT, 100 * 1024**3)
    guard.require_capacity(256 * 1024**2, label="MUGEN AIR schema catalog")
    sources = _sources()
    characters = _load_journal(CHARACTER_JOURNAL, "identity_id")
    packs = _load_journal(PACK_JOURNAL, "pack_key")
    for position, source in enumerate(sources, 1):
        pack_key = source["pack_key"]
        if pack_key in packs:
            _console(f"[{position}/{len(sources)}] resume {source['title']}")
            continue
        if not source["decoded"]:
            status = {
                "character_count": 0,
                "error": source.get("audit_error"),
                "pack_key": pack_key,
                "status": "excluded_by_existing_pixel_audit",
            }
        else:
            try:
                audits, payload = _audits(source)
                created = 0
                for audit in audits:
                    identity_id = _identity_id(source["archive_sha256"], audit.sff_header.sha256)
                    record = _character_record(source, payload, audit, identity_id)
                    existing = characters.get(identity_id)
                    if existing is not None:
                        if existing != record:
                            raise ValueError(f"resumed character differs: {identity_id}")
                    else:
                        _append(CHARACTER_JOURNAL, record)
                        characters[identity_id] = record
                    created += 1
                status = {
                    "character_count": created,
                    "pack_key": pack_key,
                    "status": "cataloged",
                }
            except Exception as error:
                status = {
                    "character_count": 0,
                    "error": f"{type(error).__name__}: {error}",
                    "pack_key": pack_key,
                    "status": "catalog_failed",
                }
        _append(PACK_JOURNAL, status)
        packs[pack_key] = status
        _console(
            f"[{position}/{len(sources)}] {source['title']}: "
            f"{status['status']} {status['character_count']}"
        )
    records = [characters[key] for key in sorted(characters)]
    pack_rows = [packs[key] for key in sorted(packs)]
    completeness = Counter(
        "complete" if record["core_coverage"]["complete_six_slot_core"] else "incomplete"
        for record in records
    )
    slot_counts: Counter[str] = Counter()
    for record in records:
        for slot, values in record["core_coverage"].items():
            if slot.endswith("_action_numbers") and values:
                slot_counts[slot.removesuffix("_action_numbers")] += 1
    source_payloads = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ZIP_SOURCE, RAR_SOURCE, ZIP_AUDIT, RAR_AUDIT)
    }
    artifact = {
        "artifact_kind": "mugen_air_schema_character_catalog",
        "characters": records,
        "counts": {
            "characters": len(records),
            "complete_six_slot_core": completeness["complete"],
            "incomplete_six_slot_core": completeness["incomplete"],
            "pack_status": dict(sorted(Counter(row["status"] for row in pack_rows).items())),
            "packs": len(pack_rows),
            "slot_character_coverage": dict(sorted(slot_counts.items())),
        },
        "policy": {
            "admission": "every character whose existing exact audit decoded its SFF",
            "authoritative_timing": "raw AIR integer ticks including zero and terminal -1 holds",
            "character_exclusion": "none for missing schema slots; availability is explicit",
            "core_view": "idle, walk, jump, block, attack_a, attack_b",
            "model_geometry": "none; catalog is variable-length source AIR metadata",
            "rights_scope": "Unknown/unverified fan uploads; no permissive inference",
            "trust_boundary": (
                "DEF/AIR/SFF headers read as inert data; character code never executed"
            ),
        },
        "schema_version": 1,
        "source": source_payloads,
    }
    payload = _canonical(artifact)
    guard.require_capacity(len(payload), label="MUGEN AIR schema catalog publication")
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


def _sources() -> list[dict[str, object]]:
    zip_source = json.loads(ZIP_SOURCE.read_bytes())
    rar_source = json.loads(RAR_SOURCE.read_bytes())
    zip_audit = {row["archive_sha256"]: row for row in json.loads(ZIP_AUDIT.read_bytes())["packs"]}
    rar_audit = {row["archive_sha256"]: row for row in json.loads(RAR_AUDIT.read_bytes())["packs"]}
    output = []
    for source_kind, document, audits in (
        ("zip", zip_source, zip_audit),
        ("rar7z", rar_source, rar_audit),
    ):
        for item in document["items"]:
            archive_sha256 = item["archive"]["sha256"]
            audit = audits[archive_sha256]
            output.append(
                {
                    "archive_path": item["archive"]["cas_path"],
                    "archive_sha256": archive_sha256,
                    "archive_size_bytes": item["archive"]["bytes"],
                    "audit_error": audit.get("error"),
                    "decoded": str(audit.get("decode_status", "")).startswith("decoded_"),
                    "pack_key": f"{source_kind}:{archive_sha256}",
                    "resource": item["resource"],
                    "source_kind": source_kind,
                    "title": item["resource"]["title"],
                }
            )
    return sorted(output, key=lambda row: str(row["pack_key"]).encode())


def _audits(source: dict[str, object]) -> tuple[tuple[object, ...], bytes]:
    archive_path = Path(str(source["archive_path"]))
    if archive_path.stat().st_size != source["archive_size_bytes"]:
        raise ValueError("archive size differs")
    if _file_sha256(archive_path) != source["archive_sha256"]:
        raise ValueError("archive SHA-256 differs")
    if source["source_kind"] == "zip":
        payload = archive_path.read_bytes()
        return audit_character_zip_variants(payload), payload
    names = _inventory(archive_path)
    payload, _ = _synthetic_character(archive_path, names)
    return (audit_character_zip(payload),), payload


def _character_record(
    source: dict[str, object], payload: bytes, audit: object, identity_id: str
) -> dict[str, object]:
    coverage = measure_core_schema_coverage(audit.actions)
    canonical = canonical_six_slot_action_numbers(coverage)
    air_payload = zip_common._member_bytes(payload, audit.air_member)
    actions = []
    for source_index, action in enumerate(audit.actions):
        actions.append(
            {
                "action_number": action.action_number,
                "collision_1_declarations": action.collision_1_declarations,
                "collision_2_declarations": action.collision_2_declarations,
                "elements": [
                    {
                        "duration_ticks": element.duration_ticks,
                        "horizontal_flip": element.horizontal_flip,
                        "optional_tokens": list(element.optional_tokens),
                        "source_line": element.source_line,
                        "sprite_group": element.sprite_group,
                        "sprite_image": element.sprite_image,
                        "vertical_flip": element.vertical_flip,
                        "x_offset": element.x_offset,
                        "y_offset": element.y_offset,
                    }
                    for element in action.elements
                ],
                "finite_duration_ticks": action.finite_duration_ticks,
                "loop_mode": action.loop_mode,
                "loop_start_index": action.loop_start_index,
                "normalized_action": action.label.normalized_action,
                "schema_phase": schema_phase(action.action_number),
                "schema_verb": schema_verb(action.action_number),
                "source_action_index": source_index,
                "source_comments": list(action.source_comments),
                "source_meaning": action.label.source_meaning,
            }
        )
    resource = source["resource"]
    return {
        "actions": actions,
        "air": {"member": audit.air_member, "sha256": hashlib.sha256(air_payload).hexdigest()},
        "canonical_six_slot_action_numbers": canonical,
        "core_coverage": {
            **asdict(coverage),
            "complete_six_slot_core": coverage.complete_six_slot_core,
        },
        "definition": {
            "authors": [value.author for value in audit.definition_variants],
            "display_names": [value.display_name for value in audit.definition_variants],
            "members": list(audit.definition_members),
            "names": [value.name for value in audit.definition_variants],
        },
        "identity_id": identity_id,
        "resource": resource,
        "source": {
            "archive_sha256": source["archive_sha256"],
            "archive_size_bytes": source["archive_size_bytes"],
            "landing_url": resource["canonical_url"],
            "source_kind": source["source_kind"],
        },
        "sff": {
            "format_family": audit.sff_header.format_family,
            "member": audit.sff_member,
            "sha256": audit.sff_header.sha256,
        },
    }


def _identity_id(archive_sha256: str, sff_sha256: str) -> str:
    return f"mugen_{archive_sha256[:16]}_{sff_sha256[:16]}"


def _load_journal(path: Path, key: str) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"{path.name} line {line_number} is unterminated")
            record = json.loads(line)
            value = record[key]
            if value in output and output[value] != record:
                raise ValueError(f"conflicting journal row for {value}")
            output[value] = record
    return output


def _append(path: Path, record: dict[str, object]) -> None:
    payload = _canonical(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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


def _console(message: str) -> None:
    print(message.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
