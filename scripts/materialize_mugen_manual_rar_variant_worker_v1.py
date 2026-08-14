"""Isolated one-variant worker for the large manual MUGEN RAR corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.adapters.mugen import (  # noqa: E402
    decode_sff_v1,
    decode_sff_v2,
    inspect_sff_header,
    parse_air,
)
from spritelab.mugen_core_materialization import (  # noqa: E402
    select_mugen_core_materializations,
)
from spritelab.mugen_schema import schema_phase, schema_verb  # noqa: E402
from spritelab.mugen_schema_view import (  # noqa: E402
    place_world_clip,
    plan_world_view,
    select_action_frames,
)
from spritelab.storage import DiskGuard  # noqa: E402

SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
PROFILES = {
    "iidx-jus-chibi-2000": {
        "archive": Path(
            "C:/Users/forre/Downloads/"
            "IIDX Distortion K-Shoot Mania - JUS & Chibi Edition w 2000+ Chars.rar"
        ),
        "archive_sha256": "eb9983574ebc441f44d668693c402befde62aac6eaa604e615652b660e4a596a",
        "allowed_extract_exit_codes": (0,),
        "metadata_root": ROOT / "data/staging/mugen-iidx-jus-chibi-2000-v1",
    },
    "anime-ascension-4000": {
        "archive": ROOT / "data/raw/manual/mugen-anime-ascension-4000-v1/"
        "anime-ascension-4000-physical-rar-v1.rar",
        "archive_sha256": "0a16a93be8971843ea1822cffd95942364e2b9f6ce05a1dd921ce490f1a71294",
        "allowed_extract_exit_codes": (2,),
        "metadata_root": ROOT / "data/staging/mugen-anime-ascension-4000-v1",
    },
}
TARGET_SIZE = 128
TARGET_FRAMES = 8
PADDING = 8
MAXIMUM_SCALE = 4.0
PROJECTION_VERSION = 2
EXCLUDED_NO_RENDERABLE_CORE_EXIT_CODE = 3
EXACT_DUPLICATE_EXIT_CODE = 4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(sys.stdin.buffer.read())
    profile = PROFILES[args.profile]
    air_path = profile["metadata_root"] / record["air"]["path"]
    air_payload = air_path.read_bytes()
    if hashlib.sha256(air_payload).hexdigest() != record["air"]["sha256"]:
        raise ValueError("AIR SHA-256 differs from catalog")
    air_exclusions = []
    actions = parse_air(
        air_payload,
        reject_duplicate_actions=False,
        recover_invalid_elements=True,
        exclusions=air_exclusions,
    )
    if [row.action_number for row in actions] != record["action_number_occurrences"]:
        raise ValueError("AIR action-number occurrences differ from catalog")

    extraction = subprocess.run(
        [
            str(SEVEN_ZIP),
            "x",
            "-so",
            "-sccUTF-8",
            "--",
            str(profile["archive"]),
            record["sff"]["path"],
        ],
        check=False,
        capture_output=True,
    )
    if extraction.returncode not in profile["allowed_extract_exit_codes"]:
        raise ValueError(f"unexpected 7-Zip extraction exit code: {extraction.returncode}")
    sff_payload = extraction.stdout
    if len(sff_payload) != record["sff"]["size_bytes"]:
        raise ValueError("extracted SFF size differs from catalog")
    crc32 = f"{zlib.crc32(sff_payload) & 0xFFFFFFFF:08x}"
    if record["sff"]["crc32"] is not None and crc32 != record["sff"]["crc32"]:
        raise ValueError("extracted SFF CRC32 differs from catalog")
    sff_sha256 = hashlib.sha256(sff_payload).hexdigest()
    duplicate_candidates = record.get("known_exact_pair_candidates", [])
    for candidate in duplicate_candidates:
        if candidate["sff_sha256"] != sff_sha256:
            continue
        duplicate = {
            "canonical_identity_id": candidate["identity_id"],
            "canonical_variant_id": candidate["variant_id"],
            "definitions": record["definitions"],
            "reason": "exact_air_and_sff_duplicate",
            "source": {
                "air": record["air"],
                "archive_sha256": profile["archive_sha256"],
                "sff": {
                    **record["sff"],
                    "crc32_verified": crc32,
                    "sha256": sff_sha256,
                    "seven_zip_exit_code": extraction.returncode,
                    "seven_zip_stderr_sha256": hashlib.sha256(extraction.stderr).hexdigest(),
                },
            },
            "status": "duplicate",
            "variant_id": record["variant_id"],
        }
        sys.stdout.buffer.write(_canonical(duplicate))
        return EXACT_DUPLICATE_EXIT_CODE
    header = inspect_sff_header(sff_payload)
    decode_exclusions = []
    if header.format_family == "sff_v1":
        sprites = decode_sff_v1(
            sff_payload,
            recover_invalid_sprites=True,
            exclusions=decode_exclusions,
        )
        palette_count = None
    else:
        sprites, palettes = decode_sff_v2(sff_payload)
        palette_count = len(palettes)
    plan = select_mugen_core_materializations(actions, sprites)
    if not plan.selected:
        exclusion = {
            "air_parse_exclusions": [asdict(row) for row in air_exclusions],
            "definitions": record["definitions"],
            "pixel_resolution_exclusions": [
                {
                    **asdict(row.exclusion),
                    "source_action_index": row.source_action_index,
                }
                for row in plan.exclusions
            ],
            "reason": "no_canonical_core_action_rendered",
            "source": {
                "air": record["air"],
                "archive_sha256": profile["archive_sha256"],
                "sff": {
                    **record["sff"],
                    "crc32_verified": crc32,
                    "decode_exclusions": [asdict(row) for row in decode_exclusions],
                    "decoded_palette_count": palette_count,
                    "decoded_sprite_count": len(sprites),
                    "format_family": header.format_family,
                    "sha256": sff_sha256,
                    "seven_zip_exit_code": extraction.returncode,
                    "seven_zip_stderr_sha256": hashlib.sha256(extraction.stderr).hexdigest(),
                },
            },
            "status": "excluded",
            "variant_id": record["variant_id"],
        }
        sys.stdout.buffer.write(_canonical(exclusion))
        return EXCLUDED_NO_RENDERABLE_CORE_EXIT_CODE

    scale_rows = plan.selected
    transform = plan_world_view(
        tuple(
            (
                row.materialized.canvas_world_left,
                row.materialized.canvas_world_top,
                row.materialized.canvas_world_left + row.materialized.canvas_width,
                row.materialized.canvas_world_top + row.materialized.canvas_height,
            )
            for row in scale_rows
        ),
        target_size=TARGET_SIZE,
        padding=PADDING,
        maximum_scale=MAXIMUM_SCALE,
    )
    identity_id = "mugen_" + sff_sha256[:32]
    guard = DiskGuard(ROOT, 100 * 1024**3)
    clips = []
    for selected in plan.selected:
        materialized = selected.materialized
        native = np.ascontiguousarray(
            np.stack(
                [
                    np.frombuffer(frame.rgba, dtype=np.uint8).reshape(
                        materialized.canvas_height,
                        materialized.canvas_width,
                        4,
                    )
                    for frame in materialized.frames
                ]
            )
        )
        placed = place_world_clip(
            native,
            world_left=materialized.canvas_world_left,
            world_top=materialized.canvas_world_top,
            transform=transform,
        )
        if placed.clipped_visible_pixels:
            raise ValueError(
                f"shared six-action view clipped {placed.clipped_visible_pixels} visible "
                f"pixels for slot {selected.slot}"
            )
        temporal = select_action_frames(
            tuple(frame.duration_ticks for frame in materialized.frames),
            loop_mode=materialized.loop_mode,
            target_frame_count=TARGET_FRAMES,
        )
        fixed = np.ascontiguousarray(placed.rgba[list(temporal.source_ordinals)])
        stable = {
            "action_number": materialized.action_number,
            "projection_version": PROJECTION_VERSION,
            "slot": selected.slot,
            "variant_id": record["variant_id"],
        }
        record_id = (
            "mugen_stream_core_" + hashlib.sha256(_canonical(stable).rstrip(b"\n")).hexdigest()[:32]
        )
        relative = Path("clips") / record["variant_id"] / f"{record_id}.npy"
        path = args.stage_root / relative
        _write_array(path, fixed, guard, f"streamed MUGEN clip {record_id}")
        clips.append(
            {
                "action_number": materialized.action_number,
                "array": _array_record(path, relative, fixed),
                "clipped_visible_pixels": placed.clipped_visible_pixels,
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
                    }
                    for frame in materialized.frames
                ],
                "identity_id": identity_id,
                "loop_mode": temporal.loop_mode,
                "record_id": record_id,
                "schema_phase": schema_phase(materialized.action_number),
                "schema_verb": schema_verb(materialized.action_number),
                "slot": selected.slot,
                "source_action_index": selected.source_action_index,
                "temporal_selection": {**asdict(temporal), "sha256": temporal.sha256},
                "variant_id": record["variant_id"],
                "world_view_transform": asdict(transform),
            }
        )
    available = {row["slot"] for row in clips}
    required = {"idle", "walk", "jump", "block", "attack_a", "attack_b"}
    output = {
        "air_parse_exclusions": [asdict(row) for row in air_exclusions],
        "clips": clips,
        "complete_six_slot_core": available == required,
        "definitions": record["definitions"],
        "identity_id": identity_id,
        "missing_slots": sorted(required - available),
        "pixel_resolution_exclusions": [
            {
                **asdict(row.exclusion),
                "source_action_index": row.source_action_index,
            }
            for row in plan.exclusions
        ],
        "source": {
            "air": record["air"],
            "archive_sha256": profile["archive_sha256"],
            "sff": {
                **record["sff"],
                "crc32_verified": crc32,
                "decode_exclusions": [asdict(row) for row in decode_exclusions],
                "decoded_palette_count": palette_count,
                "decoded_sprite_count": len(sprites),
                "format_family": header.format_family,
                "sha256": sff_sha256,
                "seven_zip_exit_code": extraction.returncode,
                "seven_zip_stderr_sha256": hashlib.sha256(extraction.stderr).hexdigest(),
            },
        },
        "variant_id": record["variant_id"],
        "world_view_transform": asdict(transform),
    }
    sys.stdout.buffer.write(_canonical(output))
    return 0


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
