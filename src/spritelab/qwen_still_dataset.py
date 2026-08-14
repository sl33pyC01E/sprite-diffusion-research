"""Hash-bound ImageFolder export for Qwen-Image sprite-style LoRA training."""

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

from spritelab.mugen_stills import compose_caption_input
from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard


def export_qwen_image_lora_dataset(
    canonical_plan_path: Path | str,
    output_directory: Path | str,
    *,
    split: str = "train",
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Export exact 512px RGB composites and captions without split leakage.

    The Hugging Face ImageFolder contract is ``metadata.jsonl`` plus image files.
    Source PNG bytes are copied exactly after independently verifying that their
    decoded pixels equal the selected RGBA target composited over RGB 127 and
    enlarged with nearest-neighbor sampling.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    plan_file = Path(canonical_plan_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace Qwen Image dataset: {output}")
    plan_bytes = plan_file.read_bytes()
    plan = _object(plan_bytes, "canonical still plan")
    if plan.get("artifact_kind") != "mugen_canonical_appearance_still_training_plan":
        raise ValueError("canonical still plan has the wrong artifact kind")
    records = plan.get("records")
    counts = plan.get("counts")
    if (
        not isinstance(records, list)
        or not all(isinstance(record, dict) for record in records)
        or not isinstance(counts, dict)
        or counts.get("sequences") != len(records)
    ):
        raise ValueError("canonical still plan record count differs")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise ValueError("canonical still plan source is absent")
    caption_file = Path(_text(source, "caption_manifest_path")).resolve()
    materialization_file = Path(_text(source, "materialization_path")).resolve()
    if _file_sha256(caption_file) != source.get("caption_manifest_file_sha256"):
        raise ValueError("caption manifest hash differs")
    if _file_sha256(materialization_file) != source.get("materialization_file_sha256"):
        raise ValueError("materialization manifest hash differs")
    captions = _object(caption_file.read_bytes(), "caption manifest")
    caption_records = captions.get("records")
    if (
        captions.get("artifact_kind") != "mugen_canonical_still_structured_caption_dataset"
        or not isinstance(caption_records, list)
        or captions.get("caption_count") != len(caption_records)
    ):
        raise ValueError("caption manifest record count differs")
    caption_by_identity = _unique(caption_records, "identity_id", "caption manifest")

    selected = sorted(
        (record for record in records if record.get("split") == split),
        key=lambda record: _text(record, "sample_id").encode("utf-8"),
    )
    expected_split_counts = counts.get("split_sequences")
    if not isinstance(expected_split_counts, dict) or expected_split_counts.get(split) != len(
        selected
    ):
        raise ValueError("canonical still split count differs")
    if not selected:
        raise ValueError(f"canonical still split is empty: {split}")

    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    source_bytes_total = 0
    verified: list[tuple[dict[str, Any], Path, bytes, str]] = []
    materialization_root = materialization_file.parent
    caption_root = caption_file.parent
    identity_counts: Counter[str] = Counter()
    for record in selected:
        identity_id = _text(record, "identity_id")
        caption = caption_by_identity.get(identity_id)
        if caption is None:
            raise ValueError(f"caption identity is absent: {identity_id}")
        target = record.get("target")
        caption_input = caption.get("caption_input")
        if not isinstance(target, dict) or not isinstance(caption_input, dict):
            raise ValueError(f"target or caption input is absent: {identity_id}")
        frame_indices = target.get("eligible_frame_indices")
        if (
            not isinstance(frame_indices, list)
            or len(frame_indices) != 1
            or isinstance(frame_indices[0], bool)
            or not isinstance(frame_indices[0], int)
        ):
            raise ValueError(f"eligible frame index is invalid: {identity_id}")
        frame_index = frame_indices[0]
        if caption.get("frame_index") != frame_index:
            raise ValueError(f"caption frame index differs: {identity_id}")
        target_path = _inside(materialization_root, _text(target, "relative_path"))
        target_bytes = target_path.read_bytes()
        if hashlib.sha256(target_bytes).hexdigest() != target.get("file_sha256"):
            raise ValueError(f"target file hash differs: {identity_id}")
        array = np.load(io.BytesIO(target_bytes), allow_pickle=False)
        if array.dtype != np.uint8 or array.shape != (8, 128, 128, 4):
            raise ValueError(f"target array shape or dtype differs: {identity_id}")
        if _array_sha256(array) != target.get("array_content_sha256"):
            raise ValueError(f"target array hash differs: {identity_id}")
        reference = np.ascontiguousarray(array[frame_index])
        if _array_sha256(reference) != record["caption_reference"].get(
            "identity_reference_array_sha256"
        ):
            raise ValueError(f"reference array hash differs: {identity_id}")
        input_path = _inside(caption_root, _text(caption_input, "relative_path"))
        input_bytes = input_path.read_bytes()
        input_sha256 = hashlib.sha256(input_bytes).hexdigest()
        if input_sha256 != caption_input.get("file_sha256") or input_sha256 != record[
            "caption_reference"
        ].get("caption_input_file_sha256"):
            raise ValueError(f"caption input file hash differs: {identity_id}")
        expected = np.asarray(
            Image.fromarray(compose_caption_input(reference), mode="RGB").resize(
                (512, 512), Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        )
        with Image.open(io.BytesIO(input_bytes)) as image:
            actual = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if actual.shape != (512, 512, 3) or not np.array_equal(actual, expected):
            raise ValueError(f"caption input pixels differ: {identity_id}")
        prompt = _qwen_prompt(_text(record, "prompt"))
        verified.append((record, input_path, input_bytes, prompt))
        source_bytes_total += len(input_bytes)
        identity_counts[identity_id] += 1
    if any(count != 1 for count in identity_counts.values()):
        raise ValueError("Qwen Image export must contain one still per identity")

    guard.require_capacity(
        source_bytes_total + 16 * 1024**2,
        label=f"Qwen Image {split} ImageFolder export",
    )
    output.mkdir(parents=True, exist_ok=False)
    image_root = output / "images"
    image_root.mkdir()
    metadata_lines = []
    manifest_records = []
    for record, _source_path, input_bytes, prompt in verified:
        filename = f"{_text(record, 'sample_id')}.png"
        relative = f"images/{filename}"
        destination = image_root / filename
        with destination.open("xb") as handle:
            handle.write(input_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        image_sha256 = hashlib.sha256(input_bytes).hexdigest()
        metadata_lines.append(
            json.dumps(
                {"file_name": relative, "text": prompt},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        manifest_records.append(
            {
                "entity_class": record.get("entity_class"),
                "identity_id": record["identity_id"],
                "image_file_sha256": image_sha256,
                "image_relative_path": relative,
                "prompt": prompt,
                "sample_id": record["sample_id"],
                "sequence_id": record["sequence_id"],
                "source_prompt": record["prompt"],
                "source_reference_array_sha256": record["caption_reference"][
                    "identity_reference_array_sha256"
                ],
            }
        )
    metadata_payload = ("\n".join(metadata_lines) + "\n").encode("utf-8")
    metadata_path = output / "metadata.jsonl"
    with metadata_path.open("xb") as handle:
        handle.write(metadata_payload)
        handle.flush()
        os.fsync(handle.fileno())
    artifact = {
        "artifact_kind": "mugen_qwen_image_lora_imagefolder",
        "counts": {
            "identities": len(manifest_records),
            "images": len(manifest_records),
            "prompts": len({record["prompt"] for record in manifest_records}),
        },
        "image_contract": {
            "alpha_projection": "exact_straight_alpha_over_rgb_127_then_round_uint8",
            "decoded_shape": [512, 512, 3],
            "display_or_training_role": "authoritative_qwen_image_rgb_training_target",
            "native_rgba_shape": [128, 128, 4],
            "resize": "nearest_neighbor_4x",
        },
        "metadata_file_sha256": hashlib.sha256(metadata_payload).hexdigest(),
        "records": manifest_records,
        "schema_version": 1,
        "source": {
            "canonical_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "canonical_plan_path": str(plan_file),
            "caption_manifest_file_sha256": source["caption_manifest_file_sha256"],
            "materialization_file_sha256": source["materialization_file_sha256"],
        },
        "split": split,
        "training_contract": {
            "caption_column": "text",
            "center_crop": True,
            "image_column": "image",
            "random_flip": False,
            "split_identity_disjoint_from": sorted(
                key for key, value in expected_split_counts.items() if key != split and value
            ),
        },
    }
    manifest_payload = canonical_json_bytes(artifact)
    manifest_path = output / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(manifest_payload)
        handle.flush()
        os.fsync(handle.fileno())
    return manifest_path, hashlib.sha256(manifest_payload).hexdigest()


def _qwen_prompt(prompt: str) -> str:
    marker = "transparent background"
    if marker not in prompt:
        raise ValueError("canonical prompt lacks the transparent-background contract")
    return prompt.replace(
        marker,
        "transparent-background sprite shown on a solid neutral gray preview canvas",
        1,
    )


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"relative path escapes source root: {relative}")
    return path


def _unique(records: list[object], key: str, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} record is not an object")
        value = _text(raw, key)
        if value in output:
            raise ValueError(f"{label} contains duplicate {key}: {value}")
        output[value] = raw
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
        raise ValueError(f"{key} must be non-empty text")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = f"{contiguous.dtype.str}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    return hashlib.sha256(header.encode() + contiguous.tobytes(order="C")).hexdigest()
