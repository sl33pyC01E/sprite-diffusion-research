"""Materialize native variable-length pixels for canonical MUGEN schema slots."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.materialize_mugen_mffa_training_v1 as common  # noqa: E402
from scripts.audit_mugen_mffa_rar7z_v1 import _inventory, _synthetic_character  # noqa: E402
from spritelab.adapters.mugen import (  # noqa: E402
    audit_character_zip,
    audit_character_zip_variants,
    decode_sff_v1,
    decode_sff_v2,
    materialize_actions,
)
from spritelab.mugen_schema import (  # noqa: E402
    canonical_available_slot_action_numbers,
    measure_core_schema_coverage,
    ordered_attack_action_numbers,
    schema_phase,
    schema_verb,
)
from spritelab.storage import DiskGuard  # noqa: E402

ZIP_SOURCE = ROOT / "data/index/reports/mugen-mffa-anime-zip-acquisition-v1.json"
RAR_SOURCE = ROOT / "data/index/reports/mugen-mffa-anime-rar7z-acquisition-v1.json"
ZIP_AUDIT = ROOT / "data/index/reports/mugen-mffa-anime-zip-corpus-audit-v3.json"
RAR_AUDIT = ROOT / "data/index/reports/mugen-mffa-anime-rar7z-corpus-audit-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-schema-core-native-v1"
STAGE = ROOT / "data/processed/.mugen-mffa-schema-core-native-v1.partial"
CHARACTER_JOURNAL = STAGE / "character-records.jsonl"
PACK_JOURNAL = STAGE / "pack-records.jsonl"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace schema-core materialization: {OUTPUT}")
    STAGE.mkdir(parents=True, exist_ok=True)
    guard = DiskGuard(ROOT, 100 * 1024**3)
    characters = _load_journal(CHARACTER_JOURNAL, "identity_id")
    packs = _load_journal(PACK_JOURNAL, "pack_key")
    sources = _sources()
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
                    if identity_id in characters:
                        created += 1
                        continue
                    record = _materialize_character(source, payload, audit, identity_id, guard)
                    _append(CHARACTER_JOURNAL, record)
                    characters[identity_id] = record
                    created += 1
                status = {
                    "character_count": created,
                    "pack_key": pack_key,
                    "status": "materialized",
                }
            except Exception as error:
                status = {
                    "character_count": 0,
                    "error": f"{type(error).__name__}: {error}",
                    "pack_key": pack_key,
                    "status": "materialization_failed",
                }
        _append(PACK_JOURNAL, status)
        packs[pack_key] = status
        _console(
            f"[{position}/{len(sources)}] {source['title']}: "
            f"{status['status']} {status['character_count']}"
        )
    records = [characters[key] for key in sorted(characters)]
    pack_rows = [packs[key] for key in sorted(packs)]
    complete = sum(record["complete_six_slot_core"] for record in records)
    slot_counts = Counter(slot["slot"] for record in records for slot in record["slots"])
    artifact = {
        "artifact_kind": "mugen_native_variable_length_schema_core_materialization",
        "characters": records,
        "counts": {
            "characters": len(records),
            "complete_six_slot_core": complete,
            "incomplete_six_slot_core": len(records) - complete,
            "pack_status": dict(sorted(Counter(row["status"] for row in pack_rows).items())),
            "packs": len(pack_rows),
            "slot_records": dict(sorted(slot_counts.items())),
        },
        "policy": {
            "action_selection": (
                "standard AIR numbers; canonical view only; every other action remains in source "
                "catalog and immutable archive"
            ),
            "attack_distinctness": "attack_a and attack_b must have distinct rendered pixel hashes",
            "frame_admission": (
                "all resolvable authored frames including zero-tick and terminal holds"
            ),
            "model_geometry": "none; each NPY retains native variable T/H/W aligned action canvas",
            "rights_scope": "Unknown/unverified fan uploads; no permissive inference",
            "temporal_admission": (
                "loop, intro-loop, terminal-hold, static, and one-shot all retained"
            ),
        },
        "schema_version": 1,
        "source": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ZIP_SOURCE, RAR_SOURCE, ZIP_AUDIT, RAR_AUDIT)
        },
    }
    payload = _canonical(artifact)
    manifest = STAGE / "materialization.json"
    if manifest.exists():
        raise FileExistsError(f"partial materialization manifest already exists: {manifest}")
    guard.require_capacity(len(payload), label="schema-core materialization manifest")
    with manifest.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(STAGE, OUTPUT)
    print(
        json.dumps(
            {
                "counts": artifact["counts"],
                "path": str(OUTPUT / "materialization.json"),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


def _materialize_character(
    source: dict[str, object],
    payload: bytes,
    audit: object,
    identity_id: str,
    guard: DiskGuard,
) -> dict[str, object]:
    sff_payload = common._member_bytes(payload, audit.sff_member)
    if audit.sff_header.format_family == "sff_v1":
        sprites = decode_sff_v1(
            sff_payload,
            initial_palette_rgb=common._initial_palette(
                payload, audit.definition_member, audit.definition.file("pal1")
            ),
        )
    else:
        sprites, _ = decode_sff_v2(sff_payload)
    candidates = [
        (source_index, action)
        for source_index, action in enumerate(audit.actions)
        if schema_verb(action.action_number) in {"idle", "walk", "jump", "block", "normal_attack"}
    ]
    plan = materialize_actions(tuple(action for _, action in candidates), sprites)
    admitted_by_number: dict[int, list[tuple[object, object, int]]] = {}
    for materialized in plan.admitted:
        original_index, source_action = candidates[materialized.source_action_index]
        admitted_by_number.setdefault(materialized.action_number, []).append(
            (materialized, source_action, original_index)
        )
    admitted_actions = tuple(
        values[0][0] for _, values in sorted(admitted_by_number.items()) if values
    )
    coverage = measure_core_schema_coverage(admitted_actions)
    requested = canonical_available_slot_action_numbers(coverage)
    selected: dict[str, tuple[object, object, int]] = {}
    for slot in ("idle", "walk", "jump", "block"):
        action_number = requested.get(slot)
        if action_number is not None:
            selected[slot] = admitted_by_number[action_number][0]
    attack_numbers = ordered_attack_action_numbers(coverage.attack_action_numbers)
    seen_attack_hashes: set[str] = set()
    for action_number in attack_numbers:
        candidate = admitted_by_number[action_number][0]
        digest = _materialized_pixel_sha256(candidate[0])
        if digest in seen_attack_hashes:
            continue
        seen_attack_hashes.add(digest)
        slot = "attack_a" if "attack_a" not in selected else "attack_b"
        selected[slot] = candidate
        if slot == "attack_b":
            break
    slots = []
    for slot in ("idle", "walk", "jump", "block", "attack_a", "attack_b"):
        if slot not in selected:
            continue
        materialized, source_action, original_index = selected[slot]
        array = np.stack(
            [
                np.frombuffer(frame.rgba, dtype=np.uint8).reshape(
                    materialized.canvas_height, materialized.canvas_width, 4
                )
                for frame in materialized.frames
            ]
        )
        array = np.ascontiguousarray(array)
        stable = {
            "action_number": materialized.action_number,
            "air_member": audit.air_member,
            "archive_sha256": source["archive_sha256"],
            "identity_id": identity_id,
            "slot": slot,
            "source_action_index": original_index,
        }
        record_id = (
            "mugen_schema_action_"
            + hashlib.sha256(
                json.dumps(stable, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()[:32]
        )
        relative = Path("actions") / identity_id / f"{record_id}.npy"
        path = STAGE / relative
        guard.require_capacity(array.nbytes + 4096, label=f"MUGEN schema action {record_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        slots.append(
            {
                "action_number": materialized.action_number,
                "array": {
                    "array_content_sha256": _array_sha256(array),
                    "dtype": "uint8",
                    "file_sha256": _file_sha256(path),
                    "relative_path": relative.as_posix(),
                    "shape": list(array.shape),
                    "size_bytes": path.stat().st_size,
                },
                "canvas": {
                    "height": materialized.canvas_height,
                    "width": materialized.canvas_width,
                    "world_left": materialized.canvas_world_left,
                    "world_top": materialized.canvas_world_top,
                },
                "frames": [
                    {
                        "duration_ticks": frame.duration_ticks,
                        "horizontal_flip": frame.horizontal_flip,
                        "ordinal": frame.ordinal,
                        "source_line": frame.source_line,
                        "source_rgba_sha256": frame.source_rgba_sha256,
                        "sprite_group": frame.sprite_group,
                        "sprite_image": frame.sprite_image,
                        "vertical_flip": frame.vertical_flip,
                        "world_left": frame.world_left,
                        "world_top": frame.world_top,
                        "x_scale": frame.x_scale,
                        "y_scale": frame.y_scale,
                    }
                    for frame in materialized.frames
                ],
                "loop_mode": materialized.loop_mode,
                "loop_start_index": materialized.loop_start_index,
                "record_id": record_id,
                "schema_phase": schema_phase(materialized.action_number),
                "schema_verb": schema_verb(materialized.action_number),
                "slot": slot,
                "source_action_index": original_index,
                "source_comments": list(source_action.source_comments),
            }
        )
    available_slots = {slot["slot"] for slot in slots}
    required_slots = {"idle", "walk", "jump", "block", "attack_a", "attack_b"}
    exclusions = [
        {
            "action_number": exclusion.action_number,
            "detail": exclusion.detail,
            "reason": exclusion.reason,
            "selected_source_action_index": candidates[exclusion.source_action_index][0],
        }
        for exclusion in plan.excluded
    ]
    resource = source["resource"]
    return {
        "available_resolved_core_action_numbers": {
            "attack": list(coverage.attack_action_numbers),
            "block": list(coverage.block_action_numbers),
            "idle": list(coverage.idle_action_numbers),
            "jump": list(coverage.jump_action_numbers),
            "walk": list(coverage.walk_action_numbers),
        },
        "complete_six_slot_core": available_slots == required_slots,
        "definition": {
            "authors": [value.author for value in audit.definition_variants],
            "display_names": [value.display_name for value in audit.definition_variants],
            "members": list(audit.definition_members),
            "names": [value.name for value in audit.definition_variants],
        },
        "identity_id": identity_id,
        "missing_slots": sorted(required_slots - available_slots),
        "pixel_resolution_exclusions": exclusions,
        "resource": resource,
        "slots": slots,
        "source": {
            "air_member": audit.air_member,
            "archive_sha256": source["archive_sha256"],
            "landing_url": resource["canonical_url"],
            "sff_member": audit.sff_member,
            "sff_sha256": audit.sff_header.sha256,
            "source_kind": source["source_kind"],
        },
    }


def _materialized_pixel_sha256(materialized: object) -> str:
    digest = hashlib.sha256()
    digest.update(f"{materialized.canvas_height}x{materialized.canvas_width}\0".encode())
    for frame in materialized.frames:
        digest.update(frame.rgba)
    return digest.hexdigest()


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


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


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
