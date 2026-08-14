"""Materialize native canonical MUGEN actions from Anime All Stars 3."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import scripts.materialize_mugen_mffa_schema_core_v1 as common  # noqa: E402
from spritelab.adapters.mugen import (  # noqa: E402
    decode_sff_v1,
    decode_sff_v2,
    materialize_actions,
)
from spritelab.mugen_directory import audit_mugen_directory  # noqa: E402
from spritelab.mugen_schema import (  # noqa: E402
    canonical_available_slot_action_numbers,
    measure_core_schema_coverage,
    ordered_attack_action_numbers,
    schema_phase,
    schema_verb,
)
from spritelab.storage import DiskGuard  # noqa: E402

CATALOG = ROOT / "data/index/reports/mugen-anime-all-stars-3-air-schema-catalog-v2.json"
CATALOG_SHA256 = "6e19f47a77603daaff16b30b87afb65e41e674b13e5ae446b9dbfc9d3c933c81"
COLLECTION_ROOT = (
    ROOT
    / "data/staging/mugen-anime-all-stars-3-v1"
    / "Anime All Stars 3 (Requested By Pands)/chars"
)
OUTPUT = ROOT / "data/processed/mugen-anime-all-stars3-schema-core-native-v2"
STAGE = ROOT / "data/processed/.mugen-anime-all-stars3-schema-core-native-v2.partial"
CHARACTER_JOURNAL = STAGE / "character-records.jsonl"
STATUS_JOURNAL = STAGE / "status-records.jsonl"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace materialization: {OUTPUT}")
    catalog_bytes = CATALOG.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != CATALOG_SHA256:
        raise ValueError("Anime All Stars 3 schema catalog SHA-256 differs")
    catalog = json.loads(catalog_bytes)
    catalog_by_sff = {row["sff"]["sha256"]: row for row in catalog["characters"]}
    STAGE.mkdir(parents=True, exist_ok=True)
    guard = DiskGuard(ROOT, 100 * 1024**3)
    characters = common._load_journal(CHARACTER_JOURNAL, "identity_id")
    statuses = common._load_journal(STATUS_JOURNAL, "identity_id")
    audit = audit_mugen_directory(COLLECTION_ROOT)
    files = _casefold_file_index(COLLECTION_ROOT)
    for position, variant in enumerate(audit.variants, 1):
        identity_id = "mugen_" + variant.sff_sha256[:32]
        if identity_id in statuses:
            common._console(f"[{position}/{len(audit.variants)}] resume {identity_id}")
            continue
        try:
            expected = catalog_by_sff[variant.sff_sha256]
            record = _materialize_variant(variant, expected, identity_id, files, guard)
            common._append(CHARACTER_JOURNAL, record)
            characters[identity_id] = record
            status = {
                "identity_id": identity_id,
                "slot_count": len(record["slots"]),
                "status": "materialized",
            }
        except Exception as error:
            status = {
                "error": f"{type(error).__name__}: {error}",
                "identity_id": identity_id,
                "slot_count": 0,
                "status": "materialization_failed",
            }
        common._append(STATUS_JOURNAL, status)
        statuses[identity_id] = status
        common._console(
            f"[{position}/{len(audit.variants)}] {identity_id}: "
            f"{status['status']} {status['slot_count']}"
        )
    records = [characters[key] for key in sorted(characters)]
    status_rows = [statuses[key] for key in sorted(statuses)]
    complete = sum(record["complete_six_slot_core"] for record in records)
    slot_counts = Counter(slot["slot"] for record in records for slot in record["slots"])
    artifact = {
        "artifact_kind": "mugen_anime_all_stars3_native_schema_core_materialization",
        "characters": records,
        "counts": {
            "characters": len(records),
            "complete_six_slot_core": complete,
            "incomplete_six_slot_core": len(records) - complete,
            "slot_records": dict(sorted(slot_counts.items())),
            "status": dict(sorted(Counter(row["status"] for row in status_rows).items())),
        },
        "policy": {
            "action_selection": "canonical standard AIR view; source catalog retains every action",
            "attack_distinctness": "attack_a and attack_b require distinct rendered pixel hashes",
            "character_admission": "all decoded fighters retained; missing slots remain explicit",
            "geometry": "native variable T/H/W world-aligned action canvases",
            "rights_scope": "unknown/unverified fan collection; no permissive inference",
            "runtime": "no CMD/CNS/ST content interpreted or executed",
            "sff_v1_recovery": (
                "invalid sprite nodes are omitted with exact evidence; links to omitted nodes "
                "remain invalid and are also omitted; no pixels are guessed"
            ),
        },
        "schema_version": 2,
        "source": {
            "catalog_path": str(CATALOG),
            "catalog_sha256": CATALOG_SHA256,
        },
        "status_rows": status_rows,
    }
    payload = common._canonical(artifact)
    manifest = STAGE / "materialization.json"
    if manifest.exists():
        raise FileExistsError(f"partial manifest already exists: {manifest}")
    guard.require_capacity(len(payload), label="Anime All Stars 3 materialization manifest")
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


def _materialize_variant(
    variant: object,
    expected: dict[str, object],
    identity_id: str,
    files: dict[str, Path],
    guard: DiskGuard,
) -> dict[str, object]:
    sff_path = COLLECTION_ROOT / variant.sff_path
    sff_payload = sff_path.read_bytes()
    if hashlib.sha256(sff_payload).hexdigest() != variant.sff_sha256:
        raise ValueError("SFF SHA-256 differs from catalog")
    if variant.sff_header.format_family == "sff_v1":
        sff_exclusions = []
        sprites = decode_sff_v1(
            sff_payload,
            initial_palette_rgb=_initial_palette(variant, files),
            recover_invalid_sprites=True,
            exclusions=sff_exclusions,
        )
    else:
        sff_exclusions = []
        sprites, _ = decode_sff_v2(sff_payload)
    candidates = [
        (source_index, action)
        for source_index, action in enumerate(variant.actions)
        if schema_verb(action.action_number) in {"idle", "walk", "jump", "block", "normal_attack"}
    ]
    source_coverage = measure_core_schema_coverage(tuple(action for _, action in candidates))
    requested = canonical_available_slot_action_numbers(source_coverage)
    candidates_by_number: dict[int, list[tuple[int, object]]] = {}
    for source_index, action in candidates:
        candidates_by_number.setdefault(action.action_number, []).append((source_index, action))
    attempted_exclusions: list[dict[str, object]] = []
    admitted_by_number: dict[int, tuple[object, object, int]] = {}

    def resolve_number(number: int) -> tuple[object, object, int] | None:
        existing = admitted_by_number.get(number)
        if existing is not None:
            return existing
        for source_index, action in candidates_by_number.get(number, []):
            plan = materialize_actions((action,), sprites)
            attempted_exclusions.extend(
                {
                    "action_number": row.action_number,
                    "detail": row.detail,
                    "reason": row.reason,
                    "source_action_index": source_index,
                }
                for row in plan.excluded
            )
            if plan.admitted:
                resolved = (plan.admitted[0], action, source_index)
                admitted_by_number[number] = resolved
                return resolved
        return None

    selected: dict[str, tuple[object, object, int]] = {}
    for slot in ("idle", "walk", "jump", "block"):
        number = requested.get(slot)
        if number is not None:
            resolved = resolve_number(number)
            if resolved is not None:
                selected[slot] = resolved
    seen_attack_hashes: set[str] = set()
    for number in ordered_attack_action_numbers(source_coverage.attack_action_numbers):
        candidate = resolve_number(number)
        if candidate is None:
            continue
        digest = common._materialized_pixel_sha256(candidate[0])
        if digest in seen_attack_hashes:
            continue
        seen_attack_hashes.add(digest)
        slot = "attack_a" if "attack_a" not in selected else "attack_b"
        selected[slot] = candidate
        if slot == "attack_b":
            break
    resolved_coverage = measure_core_schema_coverage(
        tuple(candidate[0] for candidate in admitted_by_number.values())
    )
    slots = [
        _write_slot(identity_id, slot, *selected[slot], guard)
        for slot in ("idle", "walk", "jump", "block", "attack_a", "attack_b")
        if slot in selected
    ]
    available = {row["slot"] for row in slots}
    required = {"idle", "walk", "jump", "block", "attack_a", "attack_b"}
    return {
        "available_resolved_core_action_numbers": {
            "attack": list(resolved_coverage.attack_action_numbers),
            "block": list(resolved_coverage.block_action_numbers),
            "idle": list(resolved_coverage.idle_action_numbers),
            "jump": list(resolved_coverage.jump_action_numbers),
            "walk": list(resolved_coverage.walk_action_numbers),
        },
        "available_source_core_action_numbers": {
            "attack": list(source_coverage.attack_action_numbers),
            "block": list(source_coverage.block_action_numbers),
            "idle": list(source_coverage.idle_action_numbers),
            "jump": list(source_coverage.jump_action_numbers),
            "walk": list(source_coverage.walk_action_numbers),
        },
        "complete_six_slot_core": available == required,
        "definitions": expected["definitions"],
        "identity_id": identity_id,
        "missing_slots": sorted(required - available),
        "pixel_resolution_exclusions": attempted_exclusions,
        "source_sprite_decode_exclusions": [asdict(row) for row in sff_exclusions],
        "slots": slots,
        "source": {
            "air": expected["air"],
            "catalog_variant_id": expected["variant_id"],
            "sff": expected["sff"],
        },
    }


def _write_slot(
    identity_id: str,
    slot: str,
    materialized: object,
    source_action: object,
    original_index: int,
    guard: DiskGuard,
) -> dict[str, object]:
    array = np.ascontiguousarray(
        np.stack(
            [
                np.frombuffer(frame.rgba, dtype=np.uint8).reshape(
                    materialized.canvas_height, materialized.canvas_width, 4
                )
                for frame in materialized.frames
            ]
        )
    )
    stable = {
        "action_number": materialized.action_number,
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
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if existing.dtype != array.dtype or existing.shape != array.shape:
            raise ValueError(f"existing staged action geometry differs: {path}")
        if not np.array_equal(existing, array):
            raise ValueError(f"existing staged action pixels differ: {path}")
    else:
        guard.require_capacity(array.nbytes + 4096, label=f"Anime All Stars action {record_id}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "action_number": materialized.action_number,
        "array": {
            "array_content_sha256": common._array_sha256(array),
            "dtype": "uint8",
            "file_sha256": common._file_sha256(path),
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


def _casefold_file_index(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        key = path.relative_to(root).as_posix().casefold()
        if key in output:
            raise ValueError(f"case-colliding collection path: {key}")
        output[key] = path
    return output


def _initial_palette(variant: object, files: dict[str, Path]) -> bytes | None:
    reference = variant.definitions[0].file("pal1")
    if not reference:
        return None
    parent = PurePosixPath(variant.definition_paths[0]).parent
    candidate = str(parent / reference.replace("\\", "/")).lstrip("/").casefold()
    path = files.get(candidate)
    if path is None:
        return None
    payload = path.read_bytes()
    return payload[:768] if len(payload) >= 768 else None


if __name__ == "__main__":
    raise SystemExit(main())
