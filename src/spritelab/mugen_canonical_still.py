"""Appearance-only canonical MUGEN still plan for the first pipeline stage."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from spritelab.mugen_still_dataset import compact_appearance_prompt
from spritelab.mugen_stills import compose_caption_input
from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard


def build_mugen_canonical_still_training_plan(
    sequence_plan_path: Path | str,
    caption_manifest_path: Path | str,
) -> dict[str, Any]:
    """Bind one VLM-inspected appearance-only reference frame per identity."""

    sequence_file = Path(sequence_plan_path).resolve()
    caption_file = Path(caption_manifest_path).resolve()
    sequence_bytes = sequence_file.read_bytes()
    caption_bytes = caption_file.read_bytes()
    sequence_plan = _object(sequence_bytes, "sequence plan")
    captions = _object(caption_bytes, "caption manifest")
    sequence_records = _counted_records(
        sequence_plan,
        count_path=("counts", "sequences"),
        records_key="records",
        label="sequence plan",
    )
    caption_records = _counted_records(
        captions,
        count_path=("caption_count",),
        records_key="records",
        label="caption manifest",
    )
    if captions.get("artifact_kind") != "mugen_canonical_still_structured_caption_dataset":
        raise ValueError("caption manifest has the wrong artifact kind")
    sequence_source = sequence_plan.get("source")
    if not isinstance(sequence_source, dict):
        raise ValueError("sequence plan source is absent")
    materialization_path = Path(_text(sequence_source, "materialization_path")).resolve()
    if _file_sha256(materialization_path) != sequence_source.get("materialization_file_sha256"):
        raise ValueError("sequence plan materialization differs")
    sequence_by_id = _unique(sequence_records, "sequence_id", "sequence plan")
    identity_by_id: dict[str, dict[str, Any]] = {}
    output_records = []
    split_counts: Counter[str] = Counter()
    prompts: set[str] = set()
    for caption in sorted(caption_records, key=lambda row: _text(row, "identity_id").encode()):
        identity_id = _text(caption, "identity_id")
        if identity_id in identity_by_id:
            raise ValueError(f"caption identity is duplicated: {identity_id}")
        identity_by_id[identity_id] = caption
        sequence_id = _text(caption, "sequence_id")
        sequence = sequence_by_id.get(sequence_id)
        if sequence is None:
            raise ValueError(f"caption sequence is absent from plan: {sequence_id}")
        split = _text(caption, "split")
        entity_class = _text(caption, "entity_class")
        if sequence.get("identity_id") != identity_id or sequence.get("split") != split:
            raise ValueError(f"caption identity/split differs for {identity_id}")
        if sequence.get("entity_class") != entity_class:
            raise ValueError(f"caption entity class differs for {identity_id}")
        frame_index = caption.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < 8
        ):
            raise ValueError(f"caption frame index is invalid for {identity_id}")
        full_training_prompt = _text(caption, "training_prompt")
        structured_caption = caption.get("structured_caption")
        if not isinstance(structured_caption, dict):
            raise ValueError(f"structured caption is absent for {identity_id}")
        prompt = (
            compact_appearance_prompt(
                structured_caption,
                entity_class=entity_class,
                maximum_words=38,
            )
            + "; full subject; centered; crisp hard edges"
        )
        target = sequence.get("target")
        if not isinstance(target, dict) or target.get("shape") != [8, 128, 128, 4]:
            raise ValueError(f"sequence target is invalid for {sequence_id}")
        if caption.get("source_file_sha256") != target.get("file_sha256") or caption.get(
            "source_array_sha256"
        ) != target.get("array_content_sha256"):
            raise ValueError(f"caption source target differs for {identity_id}")
        target_path = (materialization_path.parent / _text(target, "relative_path")).resolve()
        if materialization_path.parent not in target_path.parents:
            raise ValueError(f"sequence target escapes the materialization root: {sequence_id}")
        target_bytes = target_path.read_bytes()
        if hashlib.sha256(target_bytes).hexdigest() != target.get("file_sha256"):
            raise ValueError(f"sequence target bytes differ for {sequence_id}")
        target_array = np.load(io.BytesIO(target_bytes), allow_pickle=False)
        if target_array.dtype != np.uint8 or target_array.shape != (8, 128, 128, 4):
            raise ValueError(f"sequence target array differs for {sequence_id}")
        reference = np.ascontiguousarray(target_array[frame_index])
        if _array_sha256(reference) != caption.get("reference_array_sha256"):
            raise ValueError(f"caption reference array differs for {identity_id}")
        caption_input = caption.get("caption_input")
        if not isinstance(caption_input, dict):
            raise ValueError(f"caption input is absent for {identity_id}")
        caption_input_path = (caption_file.parent / _text(caption_input, "relative_path")).resolve()
        if caption_file.parent not in caption_input_path.parents:
            raise ValueError(f"caption input escapes the manifest root: {identity_id}")
        if _file_sha256(caption_input_path) != caption_input.get("file_sha256"):
            raise ValueError(f"caption input bytes differ for {identity_id}")
        expected_input = np.asarray(
            Image.fromarray(compose_caption_input(reference), mode="RGB").resize(
                (512, 512), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        with Image.open(caption_input_path) as image:
            actual_input = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if not np.array_equal(actual_input, expected_input):
            raise ValueError(f"caption input pixels differ for {identity_id}")
        sample_id = (
            "canonical_still_"
            + hashlib.sha256(
                (
                    f"mugen_canonical_still_v4\0{identity_id}\0{sequence_id}\0{frame_index}\0"
                    f"{hashlib.sha256(caption_bytes).hexdigest()}"
                ).encode()
            ).hexdigest()[:32]
        )
        output_records.append(
            {
                "caption_reference": {
                    "caption_input_file_sha256": caption_input["file_sha256"],
                    "full_training_prompt_sha256": hashlib.sha256(
                        full_training_prompt.encode()
                    ).hexdigest(),
                    "identity_reference_array_sha256": caption["reference_array_sha256"],
                    "request_body_sha256": _text(caption, "request_body_sha256"),
                },
                "conditioning": {
                    "action_phrase": None,
                    "attack_form": None,
                    "attack_strength": None,
                    "attack_tier": None,
                    "direction": None,
                    "source_structured_verb": caption.get("structured_verb"),
                    "stance": None,
                    "verb": "canonical_still",
                    "view": structured_caption.get("facing"),
                },
                "entity_class": entity_class,
                "identity_id": identity_id,
                "prompt": prompt,
                "sample_id": sample_id,
                "sequence_id": sequence_id,
                "split": split,
                "target": {
                    "array_content_sha256": target["array_content_sha256"],
                    "eligible_frame_indices": [frame_index],
                    "file_sha256": target["file_sha256"],
                    "frame_count": 8,
                    "frame_sampling": "exact_vlm_inspected_canonical_reference_frame",
                    "relative_path": target["relative_path"],
                    "shape": target["shape"],
                },
            }
        )
        prompts.add(prompt)
        split_counts[split] += 1
    sequence_identities = {record.get("identity_id") for record in sequence_records}
    if set(identity_by_id) != sequence_identities:
        raise ValueError("caption identity closure differs from sequence plan")
    return {
        "artifact_kind": "mugen_canonical_appearance_still_training_plan",
        "counts": {
            "eligible_frames": len(output_records),
            "identities": len(output_records),
            "prompts": len(prompts),
            "sequences": len(output_records),
            "split_sequences": dict(
                sorted(split_counts.items(), key=lambda item: item[0].encode())
            ),
        },
        "records": output_records,
        "sampler_contract": {
            "frame": "one_exact_vlm_inspected_reference_frame_per_identity",
            "hierarchy": ["identity", "canonical_still", "sequence", "frame"],
            "identity_split_disjoint": True,
            "stage": "text_to_canonical_appearance_still",
        },
        "schema_version": 4,
        "source": {
            "caption_manifest_file_sha256": hashlib.sha256(caption_bytes).hexdigest(),
            "caption_manifest_path": str(caption_file),
            "materialization_file_sha256": sequence_source["materialization_file_sha256"],
            "materialization_path": str(materialization_path),
            "sequence_plan_file_sha256": hashlib.sha256(sequence_bytes).hexdigest(),
            "sequence_plan_path": str(sequence_file),
        },
    }


def export_mugen_canonical_still_training_plan(
    sequence_plan_path: Path | str,
    caption_manifest_path: Path | str,
    output_path: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Publish the canonical appearance plan with no-clobber semantics."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace canonical still plan: {output}")
    plan = build_mugen_canonical_still_training_plan(sequence_plan_path, caption_manifest_path)
    payload = canonical_json_bytes(plan)
    (disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)).require_capacity(
        len(payload) + 16 * 1024**2,
        label="MUGEN canonical still training plan",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return output, hashlib.sha256(payload).hexdigest()


def _counted_records(
    value: dict[str, Any],
    *,
    count_path: tuple[str, ...],
    records_key: str,
    label: str,
) -> list[dict[str, Any]]:
    count: Any = value
    for key in count_path:
        count = count.get(key) if isinstance(count, dict) else None
    records = value.get(records_key)
    if (
        not isinstance(records, list)
        or count != len(records)
        or not all(isinstance(record, dict) for record in records)
    ):
        raise ValueError(f"{label} record count differs")
    return records


def _unique(records: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        value = _text(record, key)
        if value in output:
            raise ValueError(f"{label} contains duplicate {key}: {value}")
        output[value] = record
    return output


def _object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"field {key} must be non-empty text")
    return value


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
