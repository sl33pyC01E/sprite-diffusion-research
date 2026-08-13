"""Materialize conservative MFFA RAR/7z fighter loops without path extraction."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.materialize_mugen_mffa_training_v1 as common  # noqa: E402
from scripts.audit_mugen_mffa_rar7z_v1 import (  # noqa: E402
    _inventory,
    _synthetic_character,
)
from spritelab.adapters.mugen import audit_character_zip  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

SOURCE = ROOT / "data/index/reports/mugen-mffa-anime-rar7z-acquisition-v1.json"
AUDIT = ROOT / "data/index/reports/mugen-mffa-anime-rar7z-corpus-audit-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-anime-rar7z-action-v1"
STAGE = ROOT / "data/processed/.mugen-mffa-anime-rar7z-action-v1.partial"
SEQUENCE_JOURNAL = STAGE / "sequence-records.jsonl"
PACK_JOURNAL = STAGE / "pack-records.jsonl"


def _console(message: str) -> None:
    print(message.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace materialization: {OUTPUT}")
    STAGE.mkdir(parents=True, exist_ok=True)
    common.STAGE = STAGE
    common.SEQUENCE_JOURNAL = SEQUENCE_JOURNAL
    guard = DiskGuard(ROOT, 100 * 1024**3)
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    audit_bytes = AUDIT.read_bytes()
    audited = {row["archive_sha256"]: row for row in json.loads(audit_bytes)["packs"]}
    sequence_rows = common._load_rows(SEQUENCE_JOURNAL, "sequence_id")
    pack_rows = common._load_rows(PACK_JOURNAL, "archive_sha256")
    items = sorted(source["items"], key=lambda value: value["archive"]["sha256"])
    for position, item in enumerate(items, 1):
        digest = item["archive"]["sha256"]
        if digest in pack_rows:
            _console(f"[{position}/{len(items)}] resume {item['resource']['title']}")
            continue
        audit_row = audited[digest]
        if not str(audit_row["decode_status"]).startswith("decoded_"):
            row = {
                "archive_sha256": digest,
                "error": audit_row.get("error"),
                "generated_sequences": 0,
                "resource_id": item["resource"]["resource_id"],
                "status": "excluded_pack_by_exact_audit",
            }
        else:
            before = len(sequence_rows)
            try:
                archive_path = Path(item["archive"]["cas_path"])
                names = _inventory(archive_path)
                synthetic, _ = _synthetic_character(archive_path, names)
                media = audit_character_zip(synthetic)
                generated, counts = common._materialize_variant(
                    item,
                    synthetic,
                    media,
                    guard=guard,
                    sequence_rows=sequence_rows,
                    source_archive_sha256=digest,
                )
                row = {
                    "archive_sha256": digest,
                    "counts": dict(sorted(counts.items())),
                    "generated_sequences": generated,
                    "resource_id": item["resource"]["resource_id"],
                    "status": "materialized",
                }
            except Exception as error:
                partial = len(sequence_rows) - before
                row = {
                    "archive_sha256": digest,
                    "error": f"{type(error).__name__}: {error}",
                    "generated_sequences": partial,
                    "resource_id": item["resource"]["resource_id"],
                    "status": "partially_materialized" if partial else "excluded_pack",
                }
        common._append(PACK_JOURNAL, row)
        pack_rows[digest] = row
        _console(
            f"[{position}/{len(items)}] {item['resource']['title']}: "
            f"{row['status']} {row['generated_sequences']}"
        )

    records = [sequence_rows[key] for key in sorted(sequence_rows)]
    split_counts = Counter(value["split"] for value in records)
    action_counts = Counter(value["action"] for value in records)
    artifact = {
        "schema_version": 1,
        "sequence_count": len(records),
        "source_snapshot": {
            "canonical_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "schema_version": 1,
        },
        "config": {
            "action_caps_per_identity": common.ACTION_CAPS,
            "action_counts": dict(sorted(action_counts.items())),
            "archive_occurrences": len(items),
            "exact_audit_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "identity_split": (
                "first_32_bits_archive_sha256_mod_10: 0 validation, 1 test, else train"
            ),
            "pack_results": [pack_rows[key] for key in sorted(pack_rows)],
            "rights_scope": "Unknown/unverified fan uploads; no permissive inference",
            "source_acquisition_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "spatial_normalization": (
                "identity_scale_world_origin_floor_nearest_rgba_v1; bottom-center"
            ),
            "split_counts": dict(sorted(split_counts.items())),
            "target_frames": common.TARGET_FRAMES,
            "target_size": common.TARGET_SIZE,
            "trust_boundary": "No paths unpacked and no MUGEN character code executed",
        },
        "sequences": records,
    }
    payload = (
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    manifest = STAGE / "materialization.json"
    if manifest.exists():
        raise FileExistsError(f"partial manifest already exists: {manifest}")
    guard.require_capacity(len(payload), label="MUGEN RAR/7z materialization manifest")
    with manifest.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(STAGE, OUTPUT)
    print(
        {
            "actions": dict(sorted(action_counts.items())),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "sequences": len(records),
            "splits": dict(sorted(split_counts.items())),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
