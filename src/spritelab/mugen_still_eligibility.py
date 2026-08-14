"""Auditable subject-bearing frame selection for MUGEN still training."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard


@dataclass(frozen=True, slots=True)
class SubjectFramePixelGateConfig:
    """Conservative appearance-consistency thresholds for a captioned identity."""

    dilation_size: int = 15
    minimum_anchored_overlap: float = 0.20
    minimum_palette_histogram_intersection: float = 0.10
    minimum_candidate_palette_coverage: float = 0.15
    minimum_bbox_iou: float = 0.08
    minimum_occupancy_ratio: float = 0.20
    maximum_occupancy_ratio: float = 3.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.dilation_size, bool)
            or not isinstance(self.dilation_size, int)
            or self.dilation_size < 1
            or self.dilation_size % 2 != 1
        ):
            raise ValueError("dilation_size must be a positive odd integer")
        for name in (
            "minimum_anchored_overlap",
            "minimum_palette_histogram_intersection",
            "minimum_candidate_palette_coverage",
            "minimum_bbox_iou",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be in [0,1]")
        if (
            not isinstance(self.minimum_occupancy_ratio, (int, float))
            or isinstance(self.minimum_occupancy_ratio, bool)
            or self.minimum_occupancy_ratio <= 0
        ):
            raise ValueError("minimum_occupancy_ratio must be positive")
        if (
            not isinstance(self.maximum_occupancy_ratio, (int, float))
            or isinstance(self.maximum_occupancy_ratio, bool)
            or self.maximum_occupancy_ratio < self.minimum_occupancy_ratio
        ):
            raise ValueError("maximum_occupancy_ratio must be at least the minimum")


def frame_pixel_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    config: SubjectFramePixelGateConfig | None = None,
) -> dict[str, Any]:
    """Measure exact alpha geometry and coarse visible-palette retention."""

    gate = config or SubjectFramePixelGateConfig()
    _validate_rgba(candidate, "candidate")
    _validate_rgba(reference, "reference")
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference geometry differs")
    reference_mask = reference[..., 3] > 0
    if not bool(reference_mask.any()):
        raise ValueError("reference contains no visible subject pixels")
    reference_dilated = (
        np.asarray(
            Image.fromarray(reference_mask.astype(np.uint8) * 255).filter(
                ImageFilter.MaxFilter(gate.dilation_size)
            )
        )
        > 0
    )
    return _frame_pixel_metrics_with_features(
        candidate,
        reference_mask=reference_mask,
        reference_dilated=reference_dilated,
        reference_histogram=_palette_histogram(reference, reference_mask),
        config=gate,
    )


def _frame_pixel_metrics_with_features(
    candidate: np.ndarray,
    *,
    reference_mask: np.ndarray,
    reference_dilated: np.ndarray,
    reference_histogram: np.ndarray,
    config: SubjectFramePixelGateConfig,
) -> dict[str, Any]:
    candidate_mask = candidate[..., 3] > 0
    reference_count = int(reference_mask.sum())
    candidate_count = int(candidate_mask.sum())
    if reference_count == 0:
        raise ValueError("reference contains no visible subject pixels")
    if candidate_count == 0:
        return {
            "anchored_overlap": 0.0,
            "bbox_iou": 0.0,
            "candidate_palette_coverage": 0.0,
            "occupancy_ratio": 0.0,
            "palette_histogram_intersection": 0.0,
            "passes_pixel_gate": False,
            "visible_pixel_count": 0,
        }
    anchored_overlap = float(candidate_mask[reference_dilated].sum() / candidate_count)
    candidate_histogram = _palette_histogram(candidate, candidate_mask)
    palette_intersection = float(np.minimum(reference_histogram, candidate_histogram).sum())
    candidate_coverage = float(candidate_histogram[reference_histogram > 0].sum())
    bbox_iou = _bbox_iou(candidate_mask, reference_mask)
    occupancy_ratio = float(candidate_count / reference_count)
    passes = bool(
        anchored_overlap >= config.minimum_anchored_overlap
        and palette_intersection >= config.minimum_palette_histogram_intersection
        and candidate_coverage >= config.minimum_candidate_palette_coverage
        and bbox_iou >= config.minimum_bbox_iou
        and config.minimum_occupancy_ratio <= occupancy_ratio <= config.maximum_occupancy_ratio
    )
    return {
        "anchored_overlap": anchored_overlap,
        "bbox_iou": bbox_iou,
        "candidate_palette_coverage": candidate_coverage,
        "occupancy_ratio": occupancy_ratio,
        "palette_histogram_intersection": palette_intersection,
        "passes_pixel_gate": passes,
        "visible_pixel_count": candidate_count,
    }


def build_subject_frame_pixel_audit(
    materialization_path: Path | str,
    caption_manifest_path: Path | str,
    *,
    config: SubjectFramePixelGateConfig | None = None,
) -> dict[str, Any]:
    """Audit every logical frame against its identity's exact caption reference."""

    gate = config or SubjectFramePixelGateConfig()
    materialization_file = Path(materialization_path).resolve()
    caption_file = Path(caption_manifest_path).resolve()
    materialization_bytes = materialization_file.read_bytes()
    caption_bytes = caption_file.read_bytes()
    materialization = _json_object(materialization_bytes, "materialization")
    captions = _json_object(caption_bytes, "caption manifest")
    sequences = _counted_records(
        materialization.get("sequences"),
        materialization.get("sequence_count"),
        "materialization",
    )
    caption_records = _counted_records(
        captions.get("records"), captions.get("caption_count"), "caption manifest"
    )
    materialization_sha256 = hashlib.sha256(materialization_bytes).hexdigest()
    if captions.get("source", {}).get("materialization_file_sha256") != materialization_sha256:
        raise ValueError("caption manifest was not produced from this materialization")
    sequence_by_id = _unique(sequences, "sequence_id", "materialization")
    reference_by_identity: dict[
        str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
    ] = {}
    for caption in caption_records:
        identity_id = _text(caption, "identity_id")
        source_sequence = sequence_by_id.get(_text(caption, "sequence_id"))
        if source_sequence is None or source_sequence.get("identity_id") != identity_id:
            raise ValueError(f"caption reference sequence differs for {identity_id}")
        clip = _load_sequence(materialization_file.parent, source_sequence)
        frame_index = caption.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < 8
        ):
            raise ValueError(f"caption reference frame is invalid for {identity_id}")
        reference = np.ascontiguousarray(clip[frame_index])
        if _array_sha256(reference) != caption.get("reference_array_sha256"):
            raise ValueError(f"caption reference array differs for {identity_id}")
        if identity_id in reference_by_identity:
            raise ValueError(f"duplicate caption identity: {identity_id}")
        reference_mask = reference[..., 3] > 0
        reference_dilated = (
            np.asarray(
                Image.fromarray(reference_mask.astype(np.uint8) * 255).filter(
                    ImageFilter.MaxFilter(gate.dilation_size)
                )
            )
            > 0
        )
        reference_by_identity[identity_id] = (
            reference,
            reference_mask,
            reference_dilated,
            _palette_histogram(reference, reference_mask),
            {
                "array_content_sha256": caption["reference_array_sha256"],
                "caption_input_file_sha256": caption.get("caption_input", {}).get("file_sha256"),
                "frame_index": frame_index,
                "sequence_id": source_sequence["sequence_id"],
            },
        )
    identities = {_text(sequence, "identity_id") for sequence in sequences}
    if set(reference_by_identity) != identities:
        raise ValueError("caption references do not cover materialized identities exactly")
    output_records = []
    status_counts: Counter[str] = Counter()
    passed_frames = 0
    split_passed: Counter[str] = Counter()
    for sequence in sorted(sequences, key=lambda row: _text(row, "sequence_id").encode()):
        sequence_id = _text(sequence, "sequence_id")
        identity_id = _text(sequence, "identity_id")
        split = _text(sequence, "split")
        clip = _load_sequence(materialization_file.parent, sequence)
        reference, reference_mask, reference_dilated, reference_histogram, reference_record = (
            reference_by_identity[identity_id]
        )
        metrics = [
            _frame_pixel_metrics_with_features(
                frame,
                reference_mask=reference_mask,
                reference_dilated=reference_dilated,
                reference_histogram=reference_histogram,
                config=gate,
            )
            for frame in clip
        ]
        indices = [index for index, record in enumerate(metrics) if record["passes_pixel_gate"]]
        status = "all_pass" if len(indices) == 8 else "all_fail" if not indices else "mixed"
        status_counts[status] += 1
        passed_frames += len(indices)
        split_passed[split] += len(indices)
        output_records.append(
            {
                "identity_id": identity_id,
                "legacy_action": _text(sequence, "action"),
                "pixel_gate_pass_indices": indices,
                "pixel_gate_status": status,
                "reference": reference_record,
                "sequence_id": sequence_id,
                "split": split,
                "frames": [
                    {"frame_index": index, **record} for index, record in enumerate(metrics)
                ],
            }
        )
    return {
        "artifact_kind": "mugen_subject_bearing_frame_pixel_gate",
        "config": asdict(gate),
        "counts": {
            "frames": len(output_records) * 8,
            "pixel_gate_pass_frames": passed_frames,
            "pixel_gate_reject_frames": len(output_records) * 8 - passed_frames,
            "sequences": len(output_records),
            "split_pass_frames": dict(
                sorted(split_passed.items(), key=lambda item: item[0].encode())
            ),
            "status_sequences": dict(
                sorted(status_counts.items(), key=lambda item: item[0].encode())
            ),
        },
        "policy": {
            "all_fail": "exclude_sequence_from_still_training",
            "all_pass": "accept_all_frames_without_vlm",
            "mixed": "require_vlm_confirmation_and_intersect_with_pixel_gate",
            "scope": "still_generator_only; rejected effects remain valid animation evidence",
        },
        "records": output_records,
        "schema_version": 1,
        "source": {
            "caption_manifest_file_sha256": hashlib.sha256(caption_bytes).hexdigest(),
            "caption_manifest_path": str(caption_file),
            "materialization_file_sha256": materialization_sha256,
            "materialization_path": str(materialization_file),
        },
    }


def export_subject_frame_pixel_audit(
    materialization_path: Path | str,
    caption_manifest_path: Path | str,
    output_path: Path | str,
    *,
    config: SubjectFramePixelGateConfig | None = None,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Publish one canonical no-clobber pixel-gate audit."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace subject-frame audit: {output}")
    payload = canonical_json_bytes(
        build_subject_frame_pixel_audit(materialization_path, caption_manifest_path, config=config)
    )
    (disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)).require_capacity(
        len(payload) + 16 * 1024**2, label="MUGEN subject-frame audit"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return output, hashlib.sha256(payload).hexdigest()


def subject_contact_sheet(reference: np.ndarray, clip: np.ndarray) -> bytes:
    """Render REF plus frames 0..7 into a fixed 3x3 VLM decision sheet."""

    _validate_rgba(reference, "reference")
    if clip.dtype != np.uint8 or clip.shape != (8, *reference.shape):
        raise ValueError("clip must be uint8 [8,H,W,4] matching reference")
    cell_size = 256
    canvas = Image.new("RGB", (cell_size * 3, cell_size * 3), (38, 40, 44))
    labels = ("REF", "0", "1", "2", "3", "4", "5", "6", "7")
    for index, frame in enumerate((reference, *clip)):
        image = _checkerboard_composite(frame).resize(
            (cell_size, cell_size), Image.Resampling.NEAREST
        )
        x = (index % 3) * cell_size
        y = (index // 3) * cell_size
        canvas.paste(image, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((x, y, x + 54, y + 24), fill=(0, 0, 0))
        draw.text((x + 5, y + 5), labels[index], fill=(255, 255, 255))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def subject_frame_vlm_request(*, model: str, sheet_png: bytes) -> dict[str, Any]:
    """Build the deterministic strict-JSON VLM adjudication request."""

    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be non-empty text")
    if not isinstance(sheet_png, bytes) or not sheet_png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("sheet_png must be PNG bytes")
    schema = {
        "type": "object",
        "properties": {
            "same_primary_subject_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 7},
                "uniqueItems": True,
            },
            "ambiguous_indices": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0, "maximum": 7},
                "uniqueItems": True,
            },
        },
        "required": ["same_primary_subject_indices", "ambiguous_indices"],
        "additionalProperties": False,
    }
    data_url = "data:image/png;base64," + base64.b64encode(sheet_png).decode("ascii")
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict visual data curator. Compare sprite panels literally. "
                    "Return only the requested JSON and never identify a proper name or franchise."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "REF shows the canonical primary subject. Panels 0 through 7 are "
                            "animation frames. List an index only when the same primary subject is "
                            "visibly present as a recognizable body/entity. Exclude frames that "
                            "show only projectiles, energy, smoke, weapons, detached body parts, "
                            "shadows, summoned helpers, or a different entity. A changed pose is "
                            "valid. If the primary subject might be present but cannot be "
                            "determined "
                            "visually, put "
                            "the index only in ambiguous_indices."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0,
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "subject_frame_presence",
                "strict": True,
                "schema": schema,
            },
        },
    }


def parse_subject_frame_vlm_response(content: str) -> dict[str, list[int]]:
    """Strictly normalize one subject-frame decision response."""

    if not isinstance(content, str) or not content.strip():
        raise ValueError("VLM response must be non-empty text")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines)
    value = _json_object(text.encode(), "VLM response")
    expected = {"same_primary_subject_indices", "ambiguous_indices"}
    if set(value) != expected:
        raise ValueError("VLM response fields differ")
    output: dict[str, list[int]] = {}
    for key in sorted(expected):
        indices = value[key]
        if (
            not isinstance(indices, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 8
                for item in indices
            )
            or len(indices) != len(set(indices))
        ):
            raise ValueError(f"VLM field {key} is invalid")
        output[key] = sorted(indices)
    if set(output["same_primary_subject_indices"]) & set(output["ambiguous_indices"]):
        raise ValueError("VLM certain and ambiguous indices overlap")
    return output


def merge_subject_frame_eligibility(
    pixel_audit: dict[str, Any], vlm_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Intersect mixed-sequence pixel candidates with exact VLM confirmations."""

    if pixel_audit.get("artifact_kind") != "mugen_subject_bearing_frame_pixel_gate":
        raise ValueError("pixel audit has the wrong artifact kind")
    records = pixel_audit.get("records")
    if not isinstance(records, list) or pixel_audit.get("counts", {}).get("sequences") != len(
        records
    ):
        raise ValueError("pixel audit record count differs")
    vlm_by_id = _unique(vlm_records, "sequence_id", "VLM records")
    mixed_ids = {
        _text(record, "sequence_id")
        for record in records
        if record.get("pixel_gate_status") == "mixed"
    }
    if set(vlm_by_id) != mixed_ids:
        raise ValueError("VLM record closure differs from mixed pixel-gate sequences")
    output = []
    eligible_frames = 0
    excluded_sequences = 0
    ambiguous_frames = 0
    for record in records:
        sequence_id = _text(record, "sequence_id")
        status = record.get("pixel_gate_status")
        pixel_indices = record.get("pixel_gate_pass_indices")
        if not isinstance(pixel_indices, list):
            raise ValueError(f"pixel indices are missing for {sequence_id}")
        if status == "all_pass":
            eligible = list(range(8))
            method = "pixel_gate_all_pass"
            ambiguous: list[int] = []
        elif status == "all_fail":
            eligible = []
            method = "pixel_gate_all_fail"
            ambiguous = []
        elif status == "mixed":
            decision = vlm_by_id[sequence_id]
            confirmed = decision.get("same_primary_subject_indices")
            ambiguous = decision.get("ambiguous_indices")
            if not isinstance(confirmed, list) or not isinstance(ambiguous, list):
                raise ValueError(f"VLM indices are missing for {sequence_id}")
            eligible = sorted(set(pixel_indices) & set(confirmed))
            method = "pixel_gate_intersect_qwen35_122b_confirmation"
        else:
            raise ValueError(f"unknown pixel-gate status for {sequence_id}")
        eligible_frames += len(eligible)
        ambiguous_frames += len(ambiguous)
        excluded_sequences += not eligible
        output.append(
            {
                "ambiguous_frame_indices": sorted(ambiguous),
                "eligibility_method": method,
                "eligible_frame_indices": eligible,
                "identity_id": record.get("identity_id"),
                "sequence_id": sequence_id,
                "split": record.get("split"),
            }
        )
    pixel_payload = canonical_json_bytes(pixel_audit)
    vlm_payload = canonical_json_bytes(vlm_records)
    return {
        "artifact_kind": "mugen_subject_bearing_still_frame_eligibility",
        "counts": {
            "ambiguous_frames_excluded": ambiguous_frames,
            "eligible_frames": eligible_frames,
            "excluded_frames": len(output) * 8 - eligible_frames,
            "excluded_sequences": excluded_sequences,
            "retained_sequences": len(output) - excluded_sequences,
            "sequences": len(output),
        },
        "policy": {
            "all_pass_sequences": "all_pixel-gate frames accepted",
            "all_fail_sequences": "excluded from still generator only",
            "ambiguous_vlm_frames": "excluded",
            "mixed_sequences": "pixel-gate pass intersect exact VLM same-subject decision",
        },
        "records": output,
        "schema_version": 1,
        "source": {
            "caption_manifest_file_sha256": pixel_audit.get("source", {}).get(
                "caption_manifest_file_sha256"
            ),
            "materialization_file_sha256": pixel_audit.get("source", {}).get(
                "materialization_file_sha256"
            ),
            "pixel_audit_canonical_sha256": hashlib.sha256(pixel_payload).hexdigest(),
            "vlm_records_canonical_sha256": hashlib.sha256(vlm_payload).hexdigest(),
        },
    }


def _load_sequence(root: Path, sequence: dict[str, Any]) -> np.ndarray:
    output = sequence.get("output")
    if not isinstance(output, dict):
        raise ValueError("sequence output is missing")
    path = (root / _text(output, "relative_path")).resolve()
    if root not in path.parents:
        raise ValueError("sequence target escapes materialization root")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != output.get("file_sha256"):
        raise ValueError(f"sequence file hash differs: {path}")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise ValueError(f"sequence geometry differs: {path}")
    if _array_sha256(value) != output.get("array_content_sha256"):
        raise ValueError(f"sequence array hash differs: {path}")
    return np.ascontiguousarray(value)


def _checkerboard_composite(rgba: np.ndarray) -> Image.Image:
    height, width = rgba.shape[:2]
    yy, xx = np.indices((height, width))
    base = np.where(((xx // 8 + yy // 8) % 2)[..., None] == 0, 96, 128)
    background = np.repeat(base, 3, axis=2).astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) * alpha + background * (1 - alpha)
    return Image.fromarray(np.rint(rgb).astype(np.uint8), "RGB")


def _palette_histogram(rgba: np.ndarray, mask: np.ndarray) -> np.ndarray:
    code = (
        (rgba[..., 0].astype(np.int32) >> 4) << 8
        | (rgba[..., 1].astype(np.int32) >> 4) << 4
        | (rgba[..., 2].astype(np.int32) >> 4)
    )
    histogram = np.bincount(code[mask], minlength=4096).astype(np.float64)
    return histogram / histogram.sum()


def _bbox_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_y, left_x = np.nonzero(left)
    right_y, right_x = np.nonzero(right)
    left_box = (left_x.min(), left_y.min(), left_x.max() + 1, left_y.max() + 1)
    right_box = (right_x.min(), right_y.min(), right_x.max() + 1, right_y.max() + 1)
    width = max(0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
    height = max(0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1]))
    intersection = int(width * height)
    left_area = int((left_box[2] - left_box[0]) * (left_box[3] - left_box[1]))
    right_area = int((right_box[2] - right_box[0]) * (right_box[3] - right_box[1]))
    return float(intersection / (left_area + right_area - intersection))


def _validate_rgba(value: np.ndarray, label: str) -> None:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.uint8
        or value.ndim != 3
        or value.shape[-1] != 4
    ):
        raise ValueError(f"{label} must be a uint8 [H,W,4] array")


def _counted_records(value: Any, count: Any, label: str) -> list[dict[str, Any]]:
    if (
        not isinstance(value, list)
        or count != len(value)
        or not all(isinstance(record, dict) for record in value)
    ):
        raise ValueError(f"{label} record count differs")
    return value


def _unique(records: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        value = _text(record, key)
        if value in output:
            raise ValueError(f"{label} has duplicate {key}: {value}")
        output[value] = record
    return output


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"field {key} must be non-empty text")
    return result


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()
