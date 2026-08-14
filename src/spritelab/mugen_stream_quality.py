"""Exact quality audit for streamed fixed-core M.U.G.E.N materializations."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from spritelab.storage import DiskGuard

_REQUIRED_SLOTS = frozenset({"idle", "walk", "jump", "block", "attack_a", "attack_b"})


@dataclass(frozen=True, slots=True)
class MugenStreamQualityPolicy:
    """Explicit non-destructive tier thresholds for one six-action character."""

    minimum_view_scale: float = 0.5
    minimum_dynamic_slots: int = 3
    minimum_distinct_slot_arrays: int = 4

    def __post_init__(self) -> None:
        if not 0 < self.minimum_view_scale <= 4:
            raise ValueError("minimum_view_scale must be in (0, 4]")
        if not 0 <= self.minimum_dynamic_slots <= 6:
            raise ValueError("minimum_dynamic_slots must be between zero and six")
        if not 1 <= self.minimum_distinct_slot_arrays <= 6:
            raise ValueError("minimum_distinct_slot_arrays must be between one and six")


def build_mugen_stream_quality_audit(
    materialization_roots: tuple[Path | str, ...],
    *,
    policy: MugenStreamQualityPolicy | None = None,
) -> dict[str, Any]:
    """Verify every array and describe broad/dense tiers without dropping records."""

    active_policy = policy or MugenStreamQualityPolicy()
    if not materialization_roots:
        raise ValueError("at least one materialization root is required")
    source_rows = []
    characters = []
    seen_variants: set[str] = set()
    for value in materialization_roots:
        root = Path(value).resolve()
        manifest_path = root / "materialization.json"
        payload = manifest_path.read_bytes()
        manifest = _object(json.loads(payload), "materialization")
        if manifest.get("projection_version") != 2:
            raise ValueError(f"quality audit requires MUGEN projection version 2: {manifest_path}")
        rows = manifest.get("characters")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"materialization characters are invalid: {manifest_path}")
        for row in rows:
            variant_id = _text(row, "variant_id")
            if variant_id in seen_variants:
                raise ValueError(f"variant occurs in multiple inputs: {variant_id}")
            seen_variants.add(variant_id)
            characters.append((root, row))
        source_rows.append(
            {
                "character_count": len(rows),
                "manifest_file_sha256": hashlib.sha256(payload).hexdigest(),
                "manifest_path": str(manifest_path),
            }
        )

    quality_rows = []
    identity_variants: defaultdict[str, list[str]] = defaultdict(list)
    slot_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    scales = []
    for root, character in characters:
        row = _audit_character(root, character, active_policy)
        quality_rows.append(row)
        identity_variants[row["sff_sha256"]].append(row["variant_id"])
        slot_counts.update(row["slots"])
        reason_counts.update(row["dense_exclusion_reasons"])
        scales.append(row["view_scale"])
    quality_rows.sort(key=lambda row: row["variant_id"].encode("utf-8"))
    duplicate_groups = [
        {
            "sff_sha256": identity_id,
            "variant_ids": sorted(variant_ids, key=str.encode),
        }
        for identity_id, variant_ids in sorted(identity_variants.items())
        if len(variant_ids) > 1
    ]
    return {
        "artifact_kind": "mugen_streamed_core_quality_audit",
        "counts": {
            "broad_eligible_characters": sum(row["broad_eligible"] for row in quality_rows),
            "characters": len(quality_rows),
            "complete_six_slot_characters": sum(
                row["complete_six_slot_core"] for row in quality_rows
            ),
            "dense_eligible_characters": sum(row["dense_eligible"] for row in quality_rows),
            "dense_exclusion_reasons": dict(sorted(reason_counts.items())),
            "exact_sff_identity_duplicate_groups": len(duplicate_groups),
            "exact_sff_identity_duplicate_rows": sum(
                len(row["variant_ids"]) for row in duplicate_groups
            ),
            "slots": dict(sorted(slot_counts.items())),
            "unique_sff_identities": len(identity_variants),
        },
        "identity_duplicate_groups": duplicate_groups,
        "policy": asdict(active_policy),
        "quality_rows": quality_rows,
        "schema_version": 1,
        "source_materializations": source_rows,
        "view_scale_distribution": _distribution(scales),
    }


def export_mugen_stream_quality_audit(
    materialization_roots: tuple[Path | str, ...],
    output_path: Path | str,
    *,
    policy: MugenStreamQualityPolicy | None = None,
    disk_guard: DiskGuard | None = None,
) -> str:
    """Publish one canonical no-clobber audit and return its SHA-256."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace quality audit: {output}")
    artifact = build_mugen_stream_quality_audit(materialization_roots, policy=policy)
    payload = _canonical(artifact)
    guard = disk_guard or DiskGuard(output.anchor, 100 * 1024**3)
    guard.require_capacity(len(payload), label="MUGEN streamed quality audit")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace temporary audit: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return hashlib.sha256(payload).hexdigest()


def _audit_character(
    root: Path, character: dict[str, Any], policy: MugenStreamQualityPolicy
) -> dict[str, Any]:
    clips = character.get("clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"character has no clips: {character.get('variant_id')}")
    by_slot = {}
    clip_rows = []
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("clip must be an object")
        slot = _text(clip, "slot")
        if slot in by_slot:
            raise ValueError(f"character duplicates slot {slot!r}")
        by_slot[slot] = clip
        array = _object(clip.get("array"), "clip array")
        relative = _safe_relative(_text(array, "relative_path"))
        path = root.joinpath(*relative.parts)
        file_payload = path.read_bytes()
        if hashlib.sha256(file_payload).hexdigest() != _digest(array, "file_sha256"):
            raise ValueError(f"array file hash differs: {path}")
        value = np.load(path, allow_pickle=False)
        if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
            raise ValueError(f"array geometry differs: {path}: {value.dtype} {value.shape}")
        if _array_sha256(value) != _digest(array, "array_content_sha256"):
            raise ValueError(f"array content hash differs: {path}")
        alpha = value[..., 3]
        visible = int(np.count_nonzero(alpha))
        frame_visible_pixels = [int(np.count_nonzero(frame)) for frame in alpha]
        unique_frames = len({frame.tobytes(order="C") for frame in value})
        frame_hashes = [_array_sha256(frame) for frame in value]
        clip_rows.append(
            {
                "action_number": int(clip["action_number"]),
                "array_content_sha256": array["array_content_sha256"],
                "clipped_visible_pixels": int(clip["clipped_visible_pixels"]),
                "dynamic": unique_frames > 1,
                "frame_array_content_sha256": frame_hashes,
                "frame_visible_pixels": frame_visible_pixels,
                "loop_mode": _text(clip, "loop_mode"),
                "mean_visible_fraction": visible / (8 * 128 * 128),
                "slot": slot,
                "source_frame_count": int(
                    _object(clip.get("temporal_selection"), "temporal selection")[
                        "source_frame_count"
                    ]
                ),
                "unique_output_frames": unique_frames,
                "visible_pixels": visible,
            }
        )
    slots = set(by_slot)
    complete = slots == _REQUIRED_SLOTS
    if bool(character.get("complete_six_slot_core")) != complete:
        raise ValueError(f"complete-six flag differs: {character.get('variant_id')}")
    view = _object(character.get("world_view_transform"), "world view transform")
    scale = float(view["scale"])
    distinct_arrays = len({row["array_content_sha256"] for row in clip_rows})
    dynamic_slots = sum(row["dynamic"] for row in clip_rows)
    clipping = sum(row["clipped_visible_pixels"] for row in clip_rows)
    empty_slots = sorted(row["slot"] for row in clip_rows if row["visible_pixels"] == 0)
    empty_frames = [
        {"frame_index": index, "slot": row["slot"]}
        for row in clip_rows
        for index, visible in enumerate(row["frame_visible_pixels"])
        if visible == 0
    ]
    reasons = []
    if not complete:
        reasons.append("incomplete_six_slot_core")
    if clipping:
        reasons.append("visible_pixel_clipping")
    if empty_slots:
        reasons.append("empty_visible_slot")
    if empty_frames:
        reasons.append("empty_output_frame")
    if scale < policy.minimum_view_scale:
        reasons.append("view_scale_below_minimum")
    if dynamic_slots < policy.minimum_dynamic_slots:
        reasons.append("insufficient_dynamic_slots")
    if distinct_arrays < policy.minimum_distinct_slot_arrays:
        reasons.append("insufficient_distinct_slot_arrays")
    broad = not clipping and not empty_slots
    identity_id = _text(character, "identity_id")
    source = _object(character.get("source"), "character source")
    sff = _object(source.get("sff"), "character source SFF")
    sff_sha256 = _digest(sff, "sha256")
    return {
        "broad_eligible": broad,
        "clip_metrics": sorted(clip_rows, key=lambda row: row["slot"].encode("utf-8")),
        "complete_six_slot_core": complete,
        "dense_eligible": not reasons,
        "dense_exclusion_reasons": reasons,
        "distinct_slot_arrays": distinct_arrays,
        "dynamic_slots": dynamic_slots,
        "empty_output_frames": empty_frames,
        "identity_id": identity_id,
        "sff_sha256": sff_sha256,
        "slots": sorted(slots, key=str.encode),
        "variant_id": _text(character, "variant_id"),
        "view_scale": scale,
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "maximum": None, "median": None, "minimum": None, "p10": None}
    ordered = sorted(values)

    def percentile(numerator: int, denominator: int) -> float:
        return ordered[min(len(ordered) - 1, (len(ordered) - 1) * numerator // denominator)]

    return {
        "count": len(ordered),
        "maximum": ordered[-1],
        "median": percentile(1, 2),
        "minimum": ordered[0],
        "p10": percentile(1, 10),
    }


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe array path: {value!r}")
    return path


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _digest(value: dict[str, Any], key: str) -> str:
    result = _text(value, key)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{key} must be a lowercase SHA-256")
    return result


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be non-empty text")
    return result


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
