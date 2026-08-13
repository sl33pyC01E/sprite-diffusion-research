"""Merge ZIP and RAR/7z MFFA clips using no-copy hard links."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.materialize_mugen_mffa_training_v1 import _entity_class  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

INPUTS = (
    ROOT / "data/processed/mugen-mffa-anime-action-v1",
    ROOT / "data/processed/mugen-mffa-anime-rar7z-action-v1",
)
OUTPUT = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1"
STAGE = ROOT / "data/processed/.mugen-mffa-anime-combined-action-v1.partial"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if OUTPUT.exists() or STAGE.exists():
        raise FileExistsError("refusing to replace or resume combined MUGEN materialization")
    guard = DiskGuard(ROOT, 100 * 1024**3)
    manifests: list[tuple[Path, bytes, dict[str, object]]] = []
    for directory in INPUTS:
        path = directory / "materialization.json"
        payload = path.read_bytes()
        manifests.append((directory, payload, json.loads(payload)))
    combined_source = {
        "input_manifest_sha256": [hashlib.sha256(row[1]).hexdigest() for row in manifests],
        "method": "ordered_union_hardlink_v1",
    }
    canonical = hashlib.sha256(
        json.dumps(combined_source, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()

    records: dict[str, dict[str, object]] = {}
    origins: dict[str, Path] = {}
    for directory, _, artifact in manifests:
        for record in artifact["sequences"]:
            record = deepcopy(record)
            description = record["caption"]["description"]
            entity_class = _entity_class(description)
            record["entity_class"] = entity_class
            record["caption"]["entity_class"] = entity_class
            record["caption"]["text"] = (
                f"{description}, {entity_class} entity, {record['action']} action, "
                "looping animation, transparent background, pixel art animated sprite"
            )
            sequence_id = record["sequence_id"]
            if sequence_id in records and records[sequence_id] != record:
                raise ValueError(f"conflicting sequence record {sequence_id}")
            records[sequence_id] = record
            origins[sequence_id] = directory

    STAGE.mkdir(parents=True)
    try:
        for sequence_id in sorted(records):
            record = records[sequence_id]
            relative = Path(record["output"]["relative_path"])
            source = origins[sequence_id] / relative
            if _sha_file(source) != record["output"]["file_sha256"]:
                raise ValueError(f"input clip hash mismatch for {sequence_id}")
            target = STAGE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
        ordered = [records[key] for key in sorted(records)]
        actions = Counter(row["action"] for row in ordered)
        splits = Counter(row["split"] for row in ordered)
        artifact = {
            "schema_version": 1,
            "sequence_count": len(ordered),
            "source_snapshot": {
                "canonical_sha256": canonical,
                "manifest_sha256": canonical,
                "schema_version": 1,
            },
            "config": {
                "action_counts": dict(sorted(actions.items())),
                "input_materializations": combined_source["input_manifest_sha256"],
                "merge_method": combined_source["method"],
                "rights_scope": "Unknown/unverified fan uploads; no permissive inference",
                "split_counts": dict(sorted(splits.items())),
                "target_frames": 8,
                "target_size": 128,
            },
            "sequences": ordered,
        }
        payload = (
            json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
        ).encode()
        guard.require_capacity(len(payload), label="combined MUGEN manifest")
        with (STAGE / "materialization.json").open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(STAGE, OUTPUT)
    except Exception:
        # Leave the partial tree intact for forensic inspection; never overwrite it.
        raise
    print(
        {
            "actions": dict(sorted(actions.items())),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "sequences": len(ordered),
            "splits": dict(sorted(splits.items())),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
