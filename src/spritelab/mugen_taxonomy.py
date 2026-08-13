"""Deterministic structured-action projection for materialized M.U.G.E.N clips."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spritelab.adapters.mugen_logic import structure_mugen_action
from spritelab.storage import DiskGuard


def build_mugen_action_taxonomy(
    materialization_manifest: Path | str,
    output_path: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> str:
    """Project source action-number evidence without inventing unavailable detail."""

    source = Path(materialization_manifest).resolve()
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace MUGEN action taxonomy: {output}")
    source_bytes = source.read_bytes()
    try:
        manifest = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"materialization manifest is invalid JSON: {source}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("materialization manifest schema is unsupported")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or manifest.get("sequence_count") != len(sequences):
        raise ValueError("materialization sequence count does not match records")
    records: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    identity_actions: dict[str, set[str]] = defaultdict(set)
    evidence_gaps = Counter[str]()
    for sequence in sequences:
        if not isinstance(sequence, dict):
            raise ValueError("materialization sequence record must be an object")
        provenance = sequence.get("provenance")
        if not isinstance(provenance, dict):
            raise ValueError("materialization sequence provenance must be an object")
        action_number = provenance.get("source_action_number")
        source_meaning = provenance.get("source_meaning")
        if not isinstance(action_number, int):
            raise ValueError("MUGEN sequence lacks an integer source action number")
        if source_meaning is not None and not isinstance(source_meaning, str):
            raise ValueError("MUGEN source meaning must be text or null")
        structured = structure_mugen_action(action_number, source_meaning)
        record = {
            "action_number": action_number,
            "attack_form": structured.attack_form,
            "attack_strength": structured.attack_strength,
            "attack_tier": structured.attack_tier,
            "direction": structured.direction,
            "identity_id": sequence.get("identity_id"),
            "legacy_action": sequence.get("action"),
            "phase": structured.phase,
            "sequence_id": sequence.get("sequence_id"),
            "source_id": provenance.get("source_id"),
            "source_meaning": source_meaning,
            "split": sequence.get("split"),
            "stance": structured.stance,
            "verb": structured.verb,
        }
        for name in ("identity_id", "sequence_id", "source_id", "split"):
            if not isinstance(record[name], str) or not record[name]:
                raise ValueError(f"MUGEN sequence has invalid {name}")
        records.append(record)
        for field in (
            "verb",
            "attack_tier",
            "attack_strength",
            "attack_form",
            "stance",
            "direction",
            "phase",
        ):
            value = record[field]
            counts[field][value if isinstance(value, str) else "unresolved"] += 1
        identity_actions[record["identity_id"]].add(structured.verb)
        if structured.verb.endswith("_attack") and structured.attack_strength is None:
            evidence_gaps["attack_strength_requires_cns_hitdef_or_air_comment_join"] += 1
        if structured.verb.endswith("_attack") and structured.attack_form is None:
            evidence_gaps["attack_form_requires_cns_hitdef_or_air_comment_join"] += 1
    records.sort(key=lambda row: str(row["sequence_id"]).encode())
    report = {
        "artifact_kind": "mugen_materialized_structured_action_taxonomy",
        "counts": {
            field: dict(sorted(values.items(), key=lambda item: item[0].encode()))
            for field, values in sorted(counts.items(), key=lambda item: item[0].encode())
        },
        "evidence_contract": {
            "current": "source_action_number_and_materialized_source_meaning_only",
            "does_not_execute_character_code": True,
            "elecbyte_air_standard": "https://elecbyte.com/mugendocs-11b1/air.html",
            "elecbyte_cns_reference": "https://elecbyte.com/mugendocs/cns.html",
            "future_join": "literal_static_CNS_HitDef_and_AIR_comment_evidence",
            "numeric_range_does_not_infer_attack_strength_or_form": True,
        },
        "evidence_gaps": dict(sorted(evidence_gaps.items(), key=lambda item: item[0].encode())),
        "identity_count": len(identity_actions),
        "multi_verb_identity_count": sum(len(values) >= 2 for values in identity_actions.values()),
        "records": records,
        "schema_version": 1,
        "sequence_count": len(records),
        "source_materialization": {
            "file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "path": str(source),
        },
    }
    payload = _canonical_json(report)
    if disk_guard is not None:
        disk_guard.require_capacity(len(payload) + 65_536, label="MUGEN action taxonomy")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to replace MUGEN action taxonomy: {output}") from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
