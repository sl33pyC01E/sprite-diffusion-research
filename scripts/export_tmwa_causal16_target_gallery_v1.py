"""Export exact fixed-eight TMWA source targets as a hash-bound preview gallery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spritelab.previews import export_rgba_clip_preview  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import load_materialized_training_clips  # noqa: E402

MANIFEST = (
    REPOSITORY / "data/processed/tmwa-model-ready-action-v1/"
    "tmwa-causal-down-idle-walk-16-materialization-v1.json"
)
EXPECTED_MANIFEST_SHA256 = "095205d49811a0102368d607a5bcb8dd9a0c2b057ebe33ddaeeda893bede45ca"
OUTPUT = REPOSITORY / "data/previews/tmwa-causal-down-idle-walk-16-targets-v2"
INDEX = OUTPUT / "index.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(value) for value in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _stem(description: str, action: str, sequence_id: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", description.casefold()).strip("-") or "entity"
    return f"target-{slug}-down-{action}-{sequence_id.removeprefix('sequence_')[:8]}"


def _write_no_clobber(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace target gallery index: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to replace target gallery index: {path}") from error
        temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    observed_manifest_sha256 = _sha256_file(MANIFEST)
    if observed_manifest_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            "subset manifest SHA-256 mismatch: expected "
            f"{EXPECTED_MANIFEST_SHA256}, got {observed_manifest_sha256}"
        )
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace target gallery directory: {OUTPUT}")
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    guard.require_capacity(256 * 1024**2, label="TMWA target gallery reserve")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw_by_sequence = {row["sequence_id"]: row for row in manifest["sequences"]}
    clips = load_materialized_training_clips(
        MANIFEST,
        split="train",
        target_bucket=64,
        target_frames=8,
    )
    rows: list[dict[str, object]] = []
    for clip in clips:
        raw = raw_by_sequence[clip.sequence_id]
        stem = _stem(clip.request.description, clip.request.action, clip.sequence_id)
        result = export_rgba_clip_preview(
            clip.rgba,
            OUTPUT,
            artifact_stem=stem,
            duration_ms=clip.duration_ms,
            loop_mode=clip.request.loop_mode,
            integer_scale=4,
            source_report_sha256=EXPECTED_MANIFEST_SHA256,
            preserve_frame_slots=True,
            overwrite=False,
            disk_guard=guard,
        )
        rows.append(
            {
                "action": clip.request.action,
                "animated_png": {
                    "file_sha256": result.animated_png_sha256,
                    "path": result.animated_png_path.name,
                },
                "contact_sheet": {
                    "file_sha256": result.contact_sheet_sha256,
                    "path": result.contact_sheet_path.name,
                },
                "description": clip.request.description,
                "direction": clip.request.direction,
                "duration_ms": list(clip.duration_ms),
                "entity_class": clip.request.entity_class,
                "fixed_rgba_target_sha256": _array_sha256(clip.rgba),
                "fixed_shape": list(clip.rgba.shape),
                "identity_id": clip.identity_id,
                "loop_mode": clip.request.loop_mode,
                "preview_metadata": {
                    "file_sha256": result.metadata_sha256,
                    "path": result.metadata_path.name,
                },
                "sequence_id": clip.sequence_id,
                "source_blob_sha256": list(clip.source_blob_sha256),
                "source_native_array_sha256": raw["output"]["array_content_sha256"],
                "source_native_file_sha256": raw["output"]["file_sha256"],
                "source_native_relative_path": raw["output"]["relative_path"],
            }
        )
    index = {
        "artifact_kind": "exact_materialized_source_target_preview_gallery",
        "display_contract": {
            "animated_format": "APNG",
            "integer_scale": 4,
            "invisible_rgb_policy": "zero_where_alpha_is_zero_display_only",
            "preserve_frame_slots": True,
            "resampling": "nearest_positive_integer",
        },
        "interpretation": (
            "Every image is a display-only derivative of an exact fixed-eight training "
            "target. These are source targets, not generated samples or model output."
        ),
        "row_count": len(rows),
        "rows": rows,
        "schema_version": 1,
        "supersedes": {
            "reason": (
                "Pillow coalesced byte-identical adjacent fixed-frame slots in APNGs; "
                "v2 preserves all eight temporal slots with exact per-slot durations"
            ),
            "retained_path": str(
                REPOSITORY / "data/previews/tmwa-causal-down-idle-walk-16-targets-v1"
            ),
        },
        "subset_manifest": {
            "file_sha256": EXPECTED_MANIFEST_SHA256,
            "path": str(MANIFEST),
            "selection": manifest["selection"],
        },
    }
    payload = (
        json.dumps(
            index,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    _write_no_clobber(INDEX, payload)
    print(
        json.dumps(
            {
                "index_path": str(INDEX),
                "index_sha256": hashlib.sha256(payload).hexdigest(),
                "row_count": len(rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
