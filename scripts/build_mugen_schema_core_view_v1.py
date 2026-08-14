"""Build a fixed eight-frame/128px derivative of native MUGEN schema actions."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_schema_view import (  # noqa: E402
    LeakageIdentity,
    assign_leakage_safe_splits,
    place_world_clip,
    plan_world_view,
    select_action_frames,
)
from spritelab.storage import DiskGuard  # noqa: E402

INPUT = ROOT / "data/processed/mugen-mffa-schema-core-native-v1/materialization.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-schema-core-b128-f8-v1"
STAGE = ROOT / "data/processed/.mugen-mffa-schema-core-b128-f8-v1.partial"
TARGET_SIZE = 128
TARGET_FRAMES = 8
PADDING = 8
MAXIMUM_SCALE = 4.0
PROJECTION_VERSION = 2


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError("Refusing to replace fixed MUGEN schema view")
    source_bytes = INPUT.read_bytes()
    source = json.loads(source_bytes)
    source_root = INPUT.parent
    guard = DiskGuard(ROOT, 100 * 1024**3)
    resumed_stage = STAGE.exists()
    STAGE.mkdir(parents=True, exist_ok=True)
    if resumed_stage:
        print(f"Resuming verified no-clobber staging directory: {STAGE}", flush=True)
    characters = []
    clips = []
    try:
        for index, character in enumerate(source["characters"], 1):
            record, derived = _build_character(character, source_root, guard)
            characters.append(record)
            clips.extend(derived)
            print(
                f"[{index}/{len(source['characters'])}] {character['identity_id']}: "
                f"{len(derived)} slots",
                flush=True,
            )
        clips_by_identity: dict[str, list[dict[str, object]]] = {}
        for clip in clips:
            clips_by_identity.setdefault(clip["identity_id"], []).append(clip)
        splits = assign_leakage_safe_splits(
            tuple(
                LeakageIdentity(
                    row["identity_id"],
                    _source_sff_sha(row["source"]),
                    tuple(
                        clip["array"]["array_content_sha256"]
                        for clip in clips_by_identity[row["identity_id"]]
                    ),
                )
                for row in characters
            )
        )
        for row in characters:
            row["split"] = splits[row["identity_id"]]
        for clip in clips:
            clip["split"] = splits[clip["identity_id"]]
        counts = Counter(clip["slot"] for clip in clips)
        artifact = {
            "artifact_kind": "mugen_fixed_schema_core_training_view",
            "characters": characters,
            "clips": clips,
            "counts": {
                "characters": len(characters),
                "clips": len(clips),
                "complete_six_slot_characters": sum(
                    row["complete_six_slot_core"] for row in characters
                ),
                "slots": dict(sorted(counts.items())),
                "splits": dict(sorted(Counter(row["split"] for row in characters).items())),
            },
            "policy": {
                "admission": "every native materialized slot; fixed shape is derivative only",
                "reference": "first fixed-view idle frame; absent only when source idle is absent",
                "spatial": (
                    "identity-consistent player-axis placement; non-attack body extents set scale; "
                    "positive nearest-neighbor scale capped at 4; attack clipping measured"
                ),
                "split": (
                    "90/5/5 deterministic connected components over exact SFF and derived "
                    "array hashes; no exact source/pixel duplicate crosses a split"
                ),
                "temporal": (
                    "eight whole authored frames; weighted ticks for loops; terminal/intro-loop "
                    "represented as one-shot; zero-tick and terminal frames receive "
                    "derivative-only "
                    "unit selection weight"
                ),
            },
            "schema_version": PROJECTION_VERSION,
            "source": {
                "materialization_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
                "materialization_path": str(INPUT),
            },
            "view": {
                "frame_count": TARGET_FRAMES,
                "maximum_scale": MAXIMUM_SCALE,
                "padding": PADDING,
                "target_size": TARGET_SIZE,
            },
        }
        payload = _canonical(artifact)
        manifest = STAGE / "materialization.json"
        guard.require_capacity(len(payload), label="fixed MUGEN schema view manifest")
        with manifest.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(STAGE, OUTPUT)
    except Exception:
        # Keep the no-clobber staging directory for forensic inspection/resume tooling.
        raise
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


def _build_character(
    character: dict[str, object], source_root: Path, guard: DiskGuard
) -> tuple[dict[str, object], list[dict[str, object]]]:
    native_slots = character["slots"]
    scale_slots = [row for row in native_slots if row["slot"] not in {"attack_a", "attack_b"}]
    if not scale_slots:
        scale_slots = native_slots
    extents = tuple(
        (
            row["canvas"]["world_left"],
            row["canvas"]["world_top"],
            row["canvas"]["world_left"] + row["canvas"]["width"],
            row["canvas"]["world_top"] + row["canvas"]["height"],
        )
        for row in scale_slots
    )
    transform = plan_world_view(
        extents,
        target_size=TARGET_SIZE,
        padding=PADDING,
        maximum_scale=MAXIMUM_SCALE,
    )
    split = _split(_source_sff_sha(character["source"]))
    clips = []
    reference_record_id = None
    for slot in native_slots:
        source_path = source_root / slot["array"]["relative_path"]
        _verify_file(source_path, slot["array"])
        native = np.load(source_path, allow_pickle=False)
        if _array_sha256(native) != slot["array"]["array_content_sha256"]:
            raise ValueError(f"native array content differs: {source_path}")
        placed = place_world_clip(
            native,
            world_left=slot["canvas"]["world_left"],
            world_top=slot["canvas"]["world_top"],
            transform=transform,
        )
        selection = select_action_frames(
            tuple(frame["duration_ticks"] for frame in slot["frames"]),
            loop_mode=slot["loop_mode"],
            target_frame_count=TARGET_FRAMES,
        )
        fixed = np.ascontiguousarray(placed.rgba[list(selection.source_ordinals)])
        stable = {
            "native_record_id": slot["record_id"],
            "target_frames": TARGET_FRAMES,
            "target_size": TARGET_SIZE,
            "view_version": PROJECTION_VERSION,
        }
        record_id = (
            "mugen_schema_view_"
            + hashlib.sha256(
                json.dumps(stable, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest()[:32]
        )
        relative = Path("clips") / character["identity_id"] / f"{record_id}.npy"
        path = STAGE / relative
        _write_array(path, fixed, guard, f"fixed MUGEN schema clip {record_id}")
        clip = {
            "action_number": slot["action_number"],
            "array": _array_record(path, relative, fixed),
            "clipped_visible_pixels": placed.clipped_visible_pixels,
            "identity_id": character["identity_id"],
            "loop_mode": selection.loop_mode,
            "native_record_id": slot["record_id"],
            "record_id": record_id,
            "schema_phase": slot["schema_phase"],
            "schema_verb": slot["schema_verb"],
            "slot": slot["slot"],
            "source": character["source"],
            "split": split,
            "temporal_selection": {
                **asdict(selection),
                "sha256": selection.sha256,
            },
            "world_view_transform": asdict(transform),
        }
        clips.append(clip)
        if slot["slot"] == "idle":
            reference_record_id = record_id
    return (
        {
            "complete_six_slot_core": character["complete_six_slot_core"],
            "definitions": character.get("definitions"),
            "identity_id": character["identity_id"],
            "reference_record_id": reference_record_id,
            "resource": character.get("resource"),
            "slot_record_ids": {row["slot"]: row["record_id"] for row in clips},
            "source": character["source"],
            "split": split,
            "world_view_transform": asdict(transform),
        },
        clips,
    )


def _split(sff_sha256: str) -> str:
    value = int(sff_sha256[:8], 16) % 100
    return "train" if value < 90 else "validation" if value < 95 else "test"


def _source_sff_sha(source: dict[str, object]) -> str:
    if "sff_sha256" in source:
        return str(source["sff_sha256"])
    sff = source.get("sff")
    if isinstance(sff, dict) and "sha256" in sff:
        return str(sff["sha256"])
    raise ValueError("native materialization source lacks an SFF SHA-256")


def _write_array(path: Path, array: np.ndarray, guard: DiskGuard, label: str) -> None:
    if path.exists():
        existing = np.load(path, allow_pickle=False)
        if existing.dtype != array.dtype or existing.shape != array.shape:
            raise ValueError(f"existing staged array geometry differs: {path}")
        if not np.array_equal(existing, array):
            raise ValueError(f"existing staged array content differs: {path}")
        return
    guard.require_capacity(array.nbytes + 4096, label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _array_record(path: Path, relative: Path, array: np.ndarray) -> dict[str, object]:
    return {
        "array_content_sha256": _array_sha256(array),
        "dtype": "uint8",
        "file_sha256": _file_sha256(path),
        "relative_path": relative.as_posix(),
        "shape": list(array.shape),
        "size_bytes": path.stat().st_size,
    }


def _verify_file(path: Path, record: dict[str, object]) -> None:
    if path.stat().st_size != record["size_bytes"]:
        raise ValueError(f"native array size differs: {path}")
    if _file_sha256(path) != record["file_sha256"]:
        raise ValueError(f"native array file hash differs: {path}")


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


if __name__ == "__main__":
    raise SystemExit(main())
