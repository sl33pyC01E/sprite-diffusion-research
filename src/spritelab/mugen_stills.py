"""Canonical appearance references for two-stage M.U.G.E.N sprite generation."""

from __future__ import annotations

import hashlib
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class MugenStillReference:
    identity_id: str
    identity_label: str
    split: str
    entity_class: str
    sequence_id: str
    structured_verb: str
    legacy_action: str
    frame_index: int
    rgba: np.ndarray
    source_file_sha256: str
    source_array_sha256: str
    reference_array_sha256: str
    alpha_bbox_xywh: tuple[int, int, int, int] | None
    visible_pixel_count: int
    palette_facts: tuple[tuple[str, float], ...]


_VERB_PRIORITY = {
    "idle": 0,
    "intro": 1,
    "walk": 2,
    "run": 3,
    "crouch": 4,
    "turn": 5,
    "block": 6,
    "normal_attack": 7,
    "special_attack": 8,
    "super_attack": 9,
}

_COLORS = {
    "black": (20, 20, 20),
    "blue": (45, 90, 190),
    "brown": (120, 72, 38),
    "cyan": (45, 190, 205),
    "gray": (125, 125, 125),
    "green": (55, 145, 65),
    "orange": (225, 120, 30),
    "pink": (225, 110, 165),
    "purple": (135, 75, 165),
    "red": (195, 45, 45),
    "white": (225, 225, 225),
    "yellow": (225, 200, 45),
}


def load_mugen_still_references(
    materialization_manifest: Path | str,
    taxonomy_path: Path | str,
) -> tuple[MugenStillReference, ...]:
    """Select one medoid frame per identity with neutral-action preference."""

    manifest_file = Path(materialization_manifest).resolve()
    taxonomy_file = Path(taxonomy_path).resolve()
    manifest_bytes = manifest_file.read_bytes()
    taxonomy_bytes = taxonomy_file.read_bytes()
    manifest = _json_object(manifest_bytes, "materialization manifest")
    taxonomy = _json_object(taxonomy_bytes, "MUGEN action taxonomy")
    sequences = manifest.get("sequences")
    taxonomy_records = taxonomy.get("records")
    if not isinstance(sequences, list) or manifest.get("sequence_count") != len(sequences):
        raise ValueError("materialization sequence count does not match records")
    if not isinstance(taxonomy_records, list) or taxonomy.get("sequence_count") != len(
        taxonomy_records
    ):
        raise ValueError("taxonomy sequence count does not match records")
    taxonomy_by_sequence = {
        record.get("sequence_id"): record
        for record in taxonomy_records
        if isinstance(record, dict) and isinstance(record.get("sequence_id"), str)
    }
    if len(taxonomy_by_sequence) != len(taxonomy_records):
        raise ValueError("taxonomy sequence IDs must be unique strings")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sequence in sequences:
        if not isinstance(sequence, dict):
            raise ValueError("materialization sequence must be an object")
        sequence_id = sequence.get("sequence_id")
        identity_id = sequence.get("identity_id")
        if not isinstance(sequence_id, str) or not isinstance(identity_id, str):
            raise ValueError("materialization sequence/identity IDs must be strings")
        taxonomy_record = taxonomy_by_sequence.get(sequence_id)
        if taxonomy_record is None:
            raise ValueError(f"taxonomy is missing sequence: {sequence_id}")
        grouped[identity_id].append({"sequence": sequence, "taxonomy": taxonomy_record})
    output: list[MugenStillReference] = []
    for identity_id, candidates in sorted(grouped.items(), key=lambda item: item[0].encode()):
        candidates.sort(key=_candidate_key)
        selected = candidates[0]
        sequence = selected["sequence"]
        taxonomy_record = selected["taxonomy"]
        array, file_sha256 = _load_sequence_array(manifest_file.parent, sequence)
        frame_index = _medoid_frame(array)
        rgba = np.ascontiguousarray(array[frame_index])
        caption = sequence.get("caption")
        if not isinstance(caption, dict):
            raise ValueError("materialization caption must be an object")
        identity_label = caption.get("identity_label")
        split = sequence.get("split")
        entity_class = sequence.get("entity_class")
        legacy_action = sequence.get("action")
        structured_verb = taxonomy_record.get("verb")
        for name, value in (
            ("identity_label", identity_label),
            ("split", split),
            ("entity_class", entity_class),
            ("legacy_action", legacy_action),
            ("structured_verb", structured_verb),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"selected still has invalid {name}")
        bbox, visible_count = _alpha_geometry(rgba)
        output.append(
            MugenStillReference(
                identity_id=identity_id,
                identity_label=identity_label,
                split=split,
                entity_class=entity_class,
                sequence_id=sequence["sequence_id"],
                structured_verb=structured_verb,
                legacy_action=legacy_action,
                frame_index=frame_index,
                rgba=rgba,
                source_file_sha256=file_sha256,
                source_array_sha256=sequence["output"]["array_content_sha256"],
                reference_array_sha256=_array_sha256(rgba),
                alpha_bbox_xywh=bbox,
                visible_pixel_count=visible_count,
                palette_facts=_palette_facts(rgba),
            )
        )
    return tuple(output)


def compose_caption_input(rgba: np.ndarray, *, background_rgb: int = 127) -> np.ndarray:
    """Composite exact RGBA onto neutral gray for RGB-only caption models."""

    if not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8:
        raise TypeError("rgba must be a uint8 NumPy array")
    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError("rgba must have shape [H,W,4]")
    if isinstance(background_rgb, bool) or not isinstance(background_rgb, int):
        raise TypeError("background_rgb must be an integer")
    if not 0 <= background_rgb <= 255:
        raise ValueError("background_rgb must remain in [0,255]")
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    composite = rgba[..., :3].astype(np.float32) * alpha + background_rgb * (1 - alpha)
    return np.ascontiguousarray(np.rint(composite).clip(0, 255), dtype=np.uint8)


def filtered_appearance_caption(raw_caption: str) -> str:
    """Remove model claims caused by the declared captioning canvas, retaining raw elsewhere."""

    if not isinstance(raw_caption, str) or not raw_caption.strip():
        raise ValueError("raw_caption must be non-empty text")
    sentences = [value.strip() for value in raw_caption.replace("\n", " ").split(".")]
    retained = [
        value
        for value in sentences
        if value
        and "background" not in value.casefold()
        and "pixelated appearance" not in value.casefold()
    ]
    return ". ".join(retained) + ("." if retained else "")


def detailed_training_prompt(reference: MugenStillReference, raw_caption: str) -> str:
    """Merge provenance label, model observation, and deterministic palette facts."""

    appearance = filtered_appearance_caption(raw_caption)
    palette = ", ".join(name for name, _ in reference.palette_facts[:4])
    pieces = [
        "2D pixel art sprite on a transparent background",
        reference.identity_label,
        reference.entity_class,
    ]
    if appearance:
        pieces.append(appearance.rstrip("."))
    if palette:
        pieces.append(f"dominant visible colors: {palette}")
    pieces.append("full character, centered, crisp hard pixel edges")
    return ". ".join(pieces) + "."


def _candidate_key(candidate: dict[str, Any]) -> tuple[int, bytes]:
    taxonomy = candidate["taxonomy"]
    sequence = candidate["sequence"]
    verb = taxonomy.get("verb")
    priority = _VERB_PRIORITY.get(verb, 100)
    return priority, str(sequence.get("sequence_id")).encode()


def _load_sequence_array(root: Path, sequence: dict[str, Any]) -> tuple[np.ndarray, str]:
    output = sequence.get("output")
    if not isinstance(output, dict):
        raise ValueError("materialization output must be an object")
    relative = output.get("relative_path")
    expected_file_sha256 = output.get("file_sha256")
    expected_array_sha256 = output.get("array_content_sha256")
    if not all(
        isinstance(value, str) for value in (relative, expected_file_sha256, expected_array_sha256)
    ):
        raise ValueError("materialization output path/hashes must be strings")
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("materialization output escapes its root")
    payload = path.read_bytes()
    actual_file_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_file_sha256 != expected_file_sha256:
        raise ValueError(f"materialized file hash mismatch: {path}")
    try:
        value = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"materialized still source is unreadable: {path}") from error
    if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
        raise ValueError("materialized still source must be a uint8 array")
    if value.ndim != 4 or value.shape[-1] != 4 or value.shape[0] < 1:
        raise ValueError("materialized still source must have shape [T,H,W,4]")
    if _array_sha256(value) != expected_array_sha256:
        raise ValueError(f"materialized array hash mismatch: {path}")
    return np.ascontiguousarray(value), actual_file_sha256


def _medoid_frame(rgba: np.ndarray) -> int:
    unit = rgba.astype(np.float32) / 255
    alpha = unit[..., 3:4]
    premultiplied = np.concatenate((unit[..., :3] * alpha, alpha), axis=-1)
    scores = []
    for index in range(rgba.shape[0]):
        score = np.abs(premultiplied[index : index + 1] - premultiplied).mean(axis=(1, 2, 3)).sum()
        scores.append(float(score))
    return min(range(len(scores)), key=lambda index: (scores[index], index))


def _alpha_geometry(rgba: np.ndarray) -> tuple[tuple[int, int, int, int] | None, int]:
    visible = rgba[..., 3] > 0
    count = int(visible.sum())
    if not count:
        return None, 0
    ys, xs = np.nonzero(visible)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    return (left, top, right - left, bottom - top), count


def _palette_facts(rgba: np.ndarray) -> tuple[tuple[str, float], ...]:
    visible = rgba[..., 3] > 0
    if not bool(visible.any()):
        return ()
    rgb = rgba[..., :3][visible].astype(np.int32)
    anchors = np.asarray(list(_COLORS.values()), dtype=np.int32)
    distances = ((rgb[:, None, :] - anchors[None, :, :]) ** 2).sum(axis=2)
    assignments = distances.argmin(axis=1)
    counts = np.bincount(assignments, minlength=len(_COLORS))
    names = tuple(_COLORS)
    order = sorted(range(len(names)), key=lambda index: (-int(counts[index]), names[index]))
    total = int(counts.sum())
    return tuple(
        (names[index], float(counts[index] / total)) for index in order if counts[index] > 0
    )


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()
