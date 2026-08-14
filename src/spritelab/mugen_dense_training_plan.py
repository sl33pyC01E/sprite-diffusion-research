"""Canonical appearance-still plan for the captioned M.U.G.E.N corpus."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from spritelab.storage import DiskGuard


def build_mugen_dense_still_training_plan(
    captioned_materialization_path: Path | str,
) -> dict[str, Any]:
    """Select one exact idle medoid per identity for appearance-only generation."""

    path = Path(captioned_materialization_path).resolve()
    payload = path.read_bytes()
    materialization = _object(json.loads(payload), "captioned materialization")
    if materialization.get("artifact_kind") != "mugen_dense_captioned_materialization_bridge":
        raise ValueError("captioned materialization has the wrong artifact kind")
    eligibility = _object(materialization.get("model_eligibility"), "model eligibility")
    if eligibility.get("conditional_generation") is not True:
        raise ValueError("captioned materialization is not conditional-generation eligible")
    sequences = materialization.get("sequences")
    if (
        not isinstance(sequences, list)
        or materialization.get("sequence_count") != len(sequences)
        or any(not isinstance(row, dict) for row in sequences)
    ):
        raise ValueError("captioned materialization sequence count differs")
    by_identity: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_sequences = set()
    for sequence in sequences:
        sequence_id = _text(sequence, "sequence_id")
        if sequence_id in seen_sequences:
            raise ValueError(f"duplicate sequence ID: {sequence_id}")
        seen_sequences.add(sequence_id)
        by_identity[_text(sequence, "identity_id")].append(sequence)

    records = []
    prompts: Counter[str] = Counter()
    identities_by_split: defaultdict[str, set[str]] = defaultdict(set)
    for identity_id, identity_sequences in sorted(
        by_identity.items(), key=lambda item: item[0].encode()
    ):
        idle_rows = [row for row in identity_sequences if row.get("action") == "idle"]
        if len(idle_rows) != 1:
            raise ValueError(f"identity must have exactly one idle reference: {identity_id}")
        idle = idle_rows[0]
        split = _text(idle, "split")
        if any(_text(row, "split") != split for row in identity_sequences):
            raise ValueError(f"identity sequences cross splits: {identity_id}")
        caption = _object(idle.get("caption"), "caption")
        appearance = _text(caption, "description").rstrip(" .;")
        reference_index = caption.get("reference_frame_index")
        if (
            isinstance(reference_index, bool)
            or not isinstance(reference_index, int)
            or not 0 <= reference_index < 8
        ):
            raise ValueError(f"canonical reference frame index differs: {identity_id}")
        reference_sha256 = _digest(caption, "reference_frame_array_content_sha256")
        for row in identity_sequences:
            row_caption = _object(row.get("caption"), "caption")
            if (
                _text(row_caption, "description") != _text(caption, "description")
                or row_caption.get("reference_frame_index") != reference_index
                or _digest(row_caption, "reference_frame_array_content_sha256") != reference_sha256
            ):
                raise ValueError(
                    f"identity caption/reference differs across actions: {identity_id}"
                )
        prompt = (
            f"{appearance}; full-body pixel art sprite; transparent background; "
            "neutral side-view reference"
        )
        target = _object(idle.get("output"), "output")
        for key in ("array_content_sha256", "file_sha256"):
            _digest(target, key)
        if target.get("shape") != [8, 128, 128, 4]:
            raise ValueError(f"target shape differs: {_text(idle, 'sequence_id')}")
        sequence_id = _text(idle, "sequence_id")
        records.append(
            {
                "conditioning": {
                    "action_phrase": "canonical neutral appearance reference",
                    "direction": "unknown",
                    "verb": "canonical_reference",
                    "view": "side",
                },
                "entity_class": _text(idle, "entity_class"),
                "identity_id": identity_id,
                "prompt": prompt,
                "sample_id": "still_reference_"
                + hashlib.sha256(
                    f"mugen_canonical_still_v1\0{identity_id}\0{sequence_id}".encode()
                ).hexdigest()[:32],
                "sequence_id": sequence_id,
                "split": split,
                "target": {
                    "array_content_sha256": target["array_content_sha256"],
                    "eligible_frame_indices": [reference_index],
                    "file_sha256": target["file_sha256"],
                    "frame_count": 8,
                    "frame_sampling": "fixed_premultiplied_rgba_temporal_medoid_idle_reference",
                    "reference_frame_array_content_sha256": reference_sha256,
                    "relative_path": _text(target, "relative_path"),
                    "shape": target["shape"],
                },
            }
        )
        prompts[prompt] += 1
        identities_by_split[split].add(identity_id)
    records.sort(key=lambda row: row["identity_id"].encode())
    return {
        "artifact_kind": "mugen_latent_still_sequence_training_plan",
        "counts": {
            "canonical_references": len(records),
            "identities": len(records),
            "prompts": len(prompts),
            "source_action_sequences": len(sequences),
            "split_identities": {
                split: len(values) for split, values in sorted(identities_by_split.items())
            },
        },
        "records": records,
        "sampler_contract": {
            "frame": "fixed_verified_idle_temporal_medoid",
            "hierarchy": ["identity", "canonical_reference"],
            "identity_component_split_disjoint": True,
            "motion_or_action_text_in_prompt": False,
            "one_target_frame_per_identity": True,
            "raw_sequence_frequency_is_not_sampling_weight": True,
        },
        "schema_version": 4,
        "source": {
            "materialization_file_sha256": hashlib.sha256(payload).hexdigest(),
            "materialization_path": str(path),
        },
    }


def export_mugen_dense_still_training_plan(
    captioned_materialization_path: Path | str,
    output_path: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> str:
    """Publish the plan canonically and no-clobber."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace dense still plan: {output}")
    artifact = build_mugen_dense_still_training_plan(captioned_materialization_path)
    payload = _canonical(artifact)
    (disk_guard or DiskGuard(Path(output.anchor), 100 * 1024**3)).require_capacity(
        len(payload), label="MUGEN dense still training plan"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace temporary dense still plan: {temporary}")
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


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be non-empty text")
    return result


def _digest(value: dict[str, Any], key: str) -> str:
    result = _text(value, key)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{key} must be a lowercase SHA-256")
    return result


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
