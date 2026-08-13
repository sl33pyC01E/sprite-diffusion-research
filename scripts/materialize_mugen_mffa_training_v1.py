"""Materialize conservative MFFA fighter loops as fixed model-ready RGBA clips."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.adapters.mugen import (  # noqa: E402
    MugenCharacterArchiveAudit,
    audit_character_zip_variants,
    decode_sff_v1,
    decode_sff_v2,
    materialize_actions,
)
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.temporal import select_temporal_frames  # noqa: E402

SOURCE = ROOT / "data/index/reports/mugen-mffa-anime-zip-acquisition-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-anime-action-v1"
STAGE = ROOT / "data/processed/.mugen-mffa-anime-action-v1.partial"
SEQUENCE_JOURNAL = STAGE / "sequence-records.jsonl"
PACK_JOURNAL = STAGE / "pack-records.jsonl"
TARGET_SIZE = 128
TARGET_FRAMES = 8
ACTION_MAP = {"block": "defend"}
ACTION_CAPS = {
    "attack": 16,
    "block": 4,
    "death": 4,
    "emote": 6,
    "hurt": 6,
    "idle": 4,
    "jump": 6,
    "run": 2,
    "spawn": 6,
    "walk": 4,
}


def _console(message: str) -> None:
    print(message.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(array: np.ndarray) -> str:
    header = f"{array.dtype.str}\0{'x'.join(str(value) for value in array.shape)}\0".encode()
    return hashlib.sha256(header + array.tobytes(order="C")).hexdigest()


def _load_rows(path: Path, key: str) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"{path.name} line {number} is unterminated")
            row = json.loads(line)
            value = row[key]
            if value in rows and rows[value] != row:
                raise ValueError(f"conflicting {path.name} row for {value}")
            rows[value] = row
    return rows


def _append(path: Path, row: dict[str, object]) -> None:
    payload = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _member_bytes(archive_payload: bytes, normalized_name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        matches = [
            info
            for info in archive.infolist()
            if PurePosixPath(info.filename.replace("\\", "/")).as_posix().lstrip("/").casefold()
            == normalized_name.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one member {normalized_name!r}, found {len(matches)}")
        return archive.read(matches[0])


def _initial_palette(payload: bytes, definition_member: str, reference: str | None) -> bytes | None:
    if not reference:
        return None
    normalized = str(PurePosixPath(definition_member).parent / reference.replace("\\", "/")).lstrip(
        "/"
    )
    try:
        value = _member_bytes(payload, normalized)
    except ValueError:
        return None
    return value[:768] if len(value) >= 768 else None


def _split(archive_sha256: str) -> str:
    bucket = int(archive_sha256[:8], 16) % 10
    return "validation" if bucket == 0 else "test" if bucket == 1 else "train"


def _place_world_canvas(
    rgba: np.ndarray,
    *,
    world_left: int,
    world_top: int,
    scale: float,
) -> np.ndarray:
    """Place MUGEN world origin at a stable bottom-center target anchor."""

    output = np.zeros((rgba.shape[0], TARGET_SIZE, TARGET_SIZE, 4), dtype=np.uint8)
    anchor_x = TARGET_SIZE // 2
    anchor_y = round(TARGET_SIZE * 0.9)
    source_x = np.floor(
        (np.arange(TARGET_SIZE, dtype=np.float64) - anchor_x) / scale - world_left
    ).astype(np.int64)
    source_y = np.floor(
        (np.arange(TARGET_SIZE, dtype=np.float64) - anchor_y) / scale - world_top
    ).astype(np.int64)
    valid_x = np.flatnonzero((source_x >= 0) & (source_x < rgba.shape[2]))
    valid_y = np.flatnonzero((source_y >= 0) & (source_y < rgba.shape[1]))
    if len(valid_x) and len(valid_y):
        output[:, valid_y[:, None], valid_x[None, :], :] = rgba[
            :, source_y[valid_y, None], source_x[None, valid_x], :
        ]
    return np.ascontiguousarray(output)


def _frame_visible_extent(rgba: bytes, height: int, width: int) -> int | None:
    alpha = np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4)[..., 3]
    points = np.argwhere(alpha > 0)
    if not len(points):
        return None
    visible_height, visible_width = points.max(axis=0) - points.min(axis=0) + 1
    return int(max(visible_height, visible_width))


def _identity_reference_extent(actions: tuple[object, ...]) -> float:
    preferred: list[int] = []
    fallback: list[int] = []
    for action in actions:
        extents = [
            extent
            for frame in action.frames
            if (
                extent := _frame_visible_extent(
                    frame.rgba, action.canvas_height, action.canvas_width
                )
            )
            is not None
        ]
        if not extents:
            continue
        representative = round(float(np.median(extents)))
        fallback.append(representative)
        if action.normalized_action in {"idle", "walk", "run"}:
            preferred.append(representative)
    candidates = preferred or fallback
    if not candidates:
        raise ValueError("no visible action frames")
    return float(np.median(candidates))


def _visible_extent_ok(rgba: np.ndarray) -> bool:
    extents: list[int] = []
    for frame in rgba:
        points = np.argwhere(frame[..., 3] > 0)
        if not len(points):
            continue
        height, width = points.max(axis=0) - points.min(axis=0) + 1
        extents.append(int(max(height, width)))
    return bool(extents) and float(np.median(extents)) >= 20


def _description(audit: object, resource: dict[str, object]) -> str:
    values = [
        getattr(audit.definition, "display_name", None),
        getattr(audit.definition, "name", None),
        resource.get("title"),
    ]
    for value in values:
        if isinstance(value, str):
            cleaned = " ".join(value.split())
            if cleaned:
                return cleaned[:160]
    return "anime fighting character"


def _entity_class(description: str) -> str:
    tokens = {token.strip("-_!?.'\"") for token in description.casefold().replace("/", " ").split()}
    if tokens & {"android", "gundam", "mecha", "mechazawa", "robot", "transformer"}:
        return "robot"
    if tokens & {
        "bear",
        "bird",
        "cat",
        "dog",
        "dragon",
        "gamera",
        "godzilla",
        "kaiju",
        "monster",
        "penguin",
        "wolf",
    }:
        return "creature"
    return "humanoid"


def _materialize_variant(
    item: dict[str, object],
    payload: bytes,
    audit: MugenCharacterArchiveAudit,
    *,
    guard: DiskGuard,
    sequence_rows: dict[str, dict[str, object]],
    source_archive_sha256: str | None = None,
) -> tuple[int, Counter[str]]:
    archive_sha256 = source_archive_sha256 or audit.archive_sha256
    sff_payload = _member_bytes(payload, audit.sff_member)
    if audit.sff_header.format_family == "sff_v1":
        sprites = decode_sff_v1(
            sff_payload,
            initial_palette_rgb=_initial_palette(
                payload, audit.definition_member, audit.definition.file("pal1")
            ),
        )
    else:
        sprites, _ = decode_sff_v2(sff_payload)
    air_sha256 = hashlib.sha256(_member_bytes(payload, audit.air_member)).hexdigest()

    selected = [
        (index, action)
        for index, action in enumerate(audit.actions)
        if action.label.normalized_action is not None
        and action.loop_mode == "loop"
        and sum(element.duration_ticks > 0 for element in action.elements) >= 2
    ]
    plan = materialize_actions(tuple(action for _, action in selected), sprites)
    description = _description(audit, item["resource"])
    entity_class = _entity_class(description)
    counts: Counter[str] = Counter()
    retained_by_action: Counter[str] = Counter()
    retained_hashes: dict[str, set[str]] = {}
    reference_extent = _identity_reference_extent(plan.admitted)
    identity_scale = min(4.0, TARGET_SIZE * 0.72 / reference_extent)
    generated = 0
    for action in plan.admitted:
        source_index, source_action = selected[action.source_action_index]
        positive = [
            (frame, element.duration_ticks)
            for frame, element in zip(action.frames, source_action.elements, strict=True)
            if element.duration_ticks > 0
        ]
        if len(positive) < 2 or len({frame.rgba_sha256 for frame, _ in positive}) < 2:
            counts["static_or_single_visible_frame"] += 1
            continue
        if action.canvas_width > 1024 or action.canvas_height > 1024:
            counts["oversized_action_canvas"] += 1
            continue
        durations = [ticks for _, ticks in positive]
        total = sum(durations)
        starts: list[float] = []
        elapsed = 0
        for ticks in durations:
            starts.append(elapsed / total)
            elapsed += ticks
        selection = select_temporal_frames(
            len(positive),
            TARGET_FRAMES,
            loop_mode="loop",
            source_phases=starts,
        )
        native = np.stack(
            [
                np.frombuffer(positive[index][0].rgba, dtype=np.uint8).reshape(
                    action.canvas_height, action.canvas_width, 4
                )
                for index in selection.source_ordinals
            ]
        )
        rgba = _place_world_canvas(
            native,
            world_left=action.canvas_world_left,
            world_top=action.canvas_world_top,
            scale=identity_scale,
        )
        if not _visible_extent_ok(rgba):
            counts["tiny_visible_extent"] += 1
            continue
        label = action.normalized_action
        if label is None:
            continue
        array_sha256 = _array_sha(rgba)
        if array_sha256 in retained_hashes.setdefault(label, set()):
            counts[f"duplicate_{label}"] += 1
            continue
        if retained_by_action[label] >= ACTION_CAPS[label]:
            counts[f"capped_{label}"] += 1
            continue
        retained_by_action[label] += 1
        retained_hashes[label].add(array_sha256)
        normalized_action = ACTION_MAP.get(action.normalized_action, action.normalized_action)
        identity_id = f"mugen_{archive_sha256[:16]}_{audit.sff_header.sha256[:16]}"
        stable = {
            "action_number": action.action_number,
            "air_sha256": air_sha256,
            "archive_sha256": archive_sha256,
            "source_action_index": source_index,
            "sff_sha256": audit.sff_header.sha256,
        }
        sequence_id = (
            "sequence_"
            + hashlib.sha256(
                json.dumps(stable, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()[:32]
        )
        relative = Path("clips") / f"{sequence_id}.npy"
        target = STAGE / relative
        if sequence_id in sequence_rows:
            existing = np.load(target, allow_pickle=False)
            if not np.array_equal(existing, rgba):
                raise ValueError(f"resumed array differs for {sequence_id}")
            generated += 1
            counts[normalized_action] += 1
            continue
        guard.require_capacity(rgba.nbytes + 4096, label=f"MUGEN clip {sequence_id}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = np.load(target, allow_pickle=False)
            if not np.array_equal(existing, rgba):
                raise ValueError(f"orphan array differs for {sequence_id}")
        else:
            with target.open("xb") as handle:
                np.save(handle, rgba, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
        record = {
            "action": normalized_action,
            "caption": {
                "action": normalized_action,
                "description": description,
                "description_basis": "internal_def_display_or_name",
                "direction": "unknown",
                "entity_class": entity_class,
                "identity_label": description,
                "loop_mode": "loop",
                "sequence_id": sequence_id,
                "source_prompt": None,
                "source_prompt_scope": None,
                "text": (
                    f"{description}, {entity_class} entity, {normalized_action} action, "
                    "looping animation, transparent background, pixel art animated sprite"
                ),
                "view": "side",
            },
            "direction": "unknown",
            "entity_class": entity_class,
            "frame_count": TARGET_FRAMES,
            "identity_id": identity_id,
            "loop_mode": "loop",
            "normalization": {
                "anchor": "bottom_center",
                "identity_reference_extent": reference_extent,
                "identity_scale": identity_scale,
                "source_canvas": [action.canvas_width, action.canvas_height],
                "source_world_origin": [
                    -action.canvas_world_left,
                    -action.canvas_world_top,
                ],
                "spatial_method": "identity_scale_world_origin_floor_nearest_rgba_v1",
                "temporal_method": selection.selection_method,
                "temporal_selection_sha256": selection.sha256,
                "temporal_source_ordinals": list(selection.source_ordinals),
            },
            "output": {
                "array_content_sha256": array_sha256,
                "dtype": "uint8",
                "file_sha256": _sha_file(target),
                "format": "numpy_npy_v1",
                "relative_path": relative.as_posix(),
                "shape": list(rgba.shape),
                "size_bytes": target.stat().st_size,
            },
            "provenance": {
                "air_member": audit.air_member,
                "archive_sha256": archive_sha256,
                "source_action_index": source_index,
                "source_action_number": action.action_number,
                "source_blob_sha256": [archive_sha256],
                "source_id": "mugen_mffa_anime",
                "source_meaning": action.source_meaning,
                "sff_member": audit.sff_member,
                "sff_sha256": audit.sff_header.sha256,
            },
            "quality_tier": "exact_source_pixels_resampled",
            "sample_weight": 1.0,
            "sequence_id": sequence_id,
            "split": _split(archive_sha256),
            "target_bucket": [TARGET_SIZE, TARGET_SIZE],
            "timing": {
                "duration_ms": [1000 / 60 * total / TARGET_FRAMES] * TARGET_FRAMES,
                "phase": list(selection.target_phases),
                "source_duration_ticks": durations,
            },
            "view": "side",
        }
        _append(SEQUENCE_JOURNAL, record)
        sequence_rows[sequence_id] = record
        generated += 1
        counts[normalized_action] += 1
    counts.update(f"excluded_{value.reason}" for value in plan.excluded)
    return generated, counts


def _materialize_pack(
    item: dict[str, object],
    *,
    guard: DiskGuard,
    sequence_rows: dict[str, dict[str, object]],
) -> tuple[int, Counter[str]]:
    archive = item["archive"]
    archive_path = Path(archive["cas_path"])
    payload = archive_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != archive["sha256"]:
        raise ValueError("archive CAS hash mismatch")
    audits = audit_character_zip_variants(payload)
    generated = 0
    counts: Counter[str] = Counter()
    counts["media_pairs"] = len(audits)
    for audit in audits:
        variant_generated, variant_counts = _materialize_variant(
            item,
            payload,
            audit,
            guard=guard,
            sequence_rows=sequence_rows,
        )
        generated += variant_generated
        counts.update(variant_counts)
    return generated, counts


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace materialization: {OUTPUT}")
    STAGE.mkdir(parents=True, exist_ok=True)
    guard = DiskGuard(ROOT, 100 * 1024**3)
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    sequence_rows = _load_rows(SEQUENCE_JOURNAL, "sequence_id")
    pack_rows = _load_rows(PACK_JOURNAL, "archive_sha256")
    items = sorted(source["items"], key=lambda value: value["archive"]["sha256"].encode())
    for position, item in enumerate(items, 1):
        digest = item["archive"]["sha256"]
        if digest in pack_rows:
            _console(f"[{position}/{len(items)}] resume {item['resource']['title']}")
            continue
        try:
            generated, counts = _materialize_pack(item, guard=guard, sequence_rows=sequence_rows)
            row = {
                "archive_sha256": digest,
                "counts": dict(sorted(counts.items())),
                "generated_sequences": generated,
                "resource_id": item["resource"]["resource_id"],
                "status": "materialized",
            }
        except Exception as error:
            row = {
                "archive_sha256": digest,
                "error": f"{type(error).__name__}: {error}",
                "generated_sequences": 0,
                "resource_id": item["resource"]["resource_id"],
                "status": "excluded_pack",
            }
        _append(PACK_JOURNAL, row)
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
            "action_policy": "Elecbyte exact/recommended labels; finite loops; known actions only",
            "action_caps_per_identity": ACTION_CAPS,
            "action_counts": dict(sorted(action_counts.items())),
            "archive_occurrences": len(items),
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
            "target_frames": TARGET_FRAMES,
            "target_size": TARGET_SIZE,
            "trust_boundary": "No MUGEN character code executed",
        },
        "sequences": records,
    }
    payload = (
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    manifest = STAGE / "materialization.json"
    if manifest.exists():
        raise FileExistsError(f"partial manifest already exists: {manifest}")
    guard.require_capacity(len(payload), label="MUGEN materialization manifest")
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
