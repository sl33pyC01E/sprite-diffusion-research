"""Hash-bound MUGEN sequence plan for latent still-image training."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard

_ACTION_PHRASES = {
    "backstep": "stepping backward",
    "block": "blocking in a defensive stance",
    "crouch": "crouching",
    "death": "falling in a defeated pose",
    "defeat": "in a defeated pose",
    "dizzy": "staggering while dizzy",
    "get_up": "getting up from the ground",
    "hurt": "recoiling from a hit",
    "idle": "in an idle neutral stance",
    "intro": "appearing in an entrance pose",
    "jump": "jumping",
    "land": "landing",
    "normal_attack": "performing a normal attack",
    "recover": "recovering balance",
    "run": "running",
    "special_attack": "performing a special attack",
    "super_attack": "performing a super attack",
    "turn": "turning around",
    "victory": "in a victory pose",
    "walk": "walking",
}


def action_phrase(verb: str) -> str:
    """Return the canonical literal prompt phrase for one supported action verb."""

    if not isinstance(verb, str) or not verb:
        raise ValueError("verb must be non-empty text")
    try:
        return _ACTION_PHRASES[verb]
    except KeyError as error:
        raise ValueError(f"unsupported MUGEN still action verb: {verb}") from error


def compact_appearance_prompt(
    structured: dict[str, Any],
    *,
    entity_class: str,
    maximum_words: int = 40,
) -> str:
    """Render dense appearance facts for CLIP without duplicative prose.

    The remote VLM intentionally emits redundant evidence fields.  Stable
    Diffusion 1.x CLIP has only 77 tokens, so this projection keeps whole,
    priority-ordered facts and stops before a conservative whitespace-word
    budget.  It never slices a phrase or silently tokenizer-truncates it.
    """

    if not isinstance(structured, dict):
        raise ValueError("structured caption must be an object")
    if not isinstance(entity_class, str) or not entity_class.strip():
        raise ValueError("entity_class must be non-empty text")
    if isinstance(maximum_words, bool) or not isinstance(maximum_words, int):
        raise ValueError("maximum_words must be an integer")
    if maximum_words < 16:
        raise ValueError("maximum_words must be at least 16")

    candidates: list[str] = ["pixel art sprite", "transparent background", entity_class]
    for key in (
        "subject_type",
        "body_build",
        "skin_or_surface",
        "hair",
        "face",
        "upper_body_clothing",
        "lower_body_clothing",
        "footwear",
        "armor",
    ):
        value = structured.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    for key, limit in (
        ("equipment", 2),
        ("accessories", 2),
        ("distinctive_visible_features", 2),
    ):
        value = structured.get(key)
        if isinstance(value, list):
            candidates.extend(
                item.strip() for item in value[:limit] if isinstance(item, str) and item.strip()
            )
    colors = []
    for key in ("dominant_colors", "secondary_colors"):
        value = structured.get(key)
        if isinstance(value, list):
            colors.extend(item.strip() for item in value if isinstance(item, str) and item.strip())
    if colors:
        candidates.append("colors " + " ".join(dict.fromkeys(colors[:6])))

    selected: list[str] = []
    normalized: list[str] = []
    word_count = 0
    for candidate in candidates:
        phrase = " ".join(candidate.split())
        lowered = phrase.casefold()
        if not phrase or any(lowered == prior or lowered in prior for prior in normalized):
            continue
        words = len(phrase.split())
        if word_count + words > maximum_words:
            continue
        selected.append(phrase)
        normalized.append(lowered)
        word_count += words
    if not selected:
        raise ValueError("structured caption contains no promptable appearance facts")
    return "; ".join(selected)


def build_mugen_still_training_plan(
    materialization_path: Path | str,
    taxonomy_path: Path | str,
    caption_manifest_path: Path | str,
    frame_eligibility_path: Path | str | None = None,
) -> dict[str, Any]:
    """Build a compact sequence plan with hierarchical sampling and exact targets."""

    materialization_file = Path(materialization_path).resolve()
    taxonomy_file = Path(taxonomy_path).resolve()
    caption_file = Path(caption_manifest_path).resolve()
    materialization_bytes = materialization_file.read_bytes()
    taxonomy_bytes = taxonomy_file.read_bytes()
    caption_bytes = caption_file.read_bytes()
    materialization = _json_object(materialization_bytes, "materialization")
    taxonomy = _json_object(taxonomy_bytes, "taxonomy")
    captions = _json_object(caption_bytes, "caption manifest")
    eligibility_file = (
        Path(frame_eligibility_path).resolve() if frame_eligibility_path is not None else None
    )
    eligibility_bytes = eligibility_file.read_bytes() if eligibility_file is not None else None
    eligibility_by_sequence: dict[str, dict[str, Any]] | None = None
    if eligibility_bytes is not None:
        eligibility = _json_object(eligibility_bytes, "frame eligibility")
        if eligibility.get("artifact_kind") != ("mugen_subject_bearing_still_frame_eligibility"):
            raise ValueError("frame eligibility has the wrong artifact kind")
        eligibility_records = eligibility.get("records")
        eligibility_count = eligibility.get("counts", {}).get("sequences")
        if (
            not isinstance(eligibility_records, list)
            or eligibility_count != len(eligibility_records)
            or not all(isinstance(record, dict) for record in eligibility_records)
        ):
            raise ValueError("frame eligibility count does not match records")
        eligibility_by_sequence = _unique_index(
            eligibility_records, "sequence_id", "frame eligibility"
        )
        eligibility_source = eligibility.get("source")
        if not isinstance(eligibility_source, dict):
            raise ValueError("frame eligibility source is missing")
        if (
            eligibility_source.get("materialization_file_sha256")
            != hashlib.sha256(materialization_bytes).hexdigest()
        ):
            raise ValueError("frame eligibility materialization differs")
        if (
            eligibility_source.get("caption_manifest_file_sha256")
            != hashlib.sha256(caption_bytes).hexdigest()
        ):
            raise ValueError("frame eligibility captions differ")
    sequences = _records(materialization, "sequences", "sequence_count", "materialization")
    taxonomy_records = _records(taxonomy, "records", "sequence_count", "taxonomy")
    caption_records = _records(captions, "records", "caption_count", "caption manifest")
    caption_manifest_sha256 = hashlib.sha256(caption_bytes).hexdigest()
    taxonomy_by_sequence = _unique_index(taxonomy_records, "sequence_id", "taxonomy")
    caption_by_identity = _unique_index(caption_records, "identity_id", "caption manifest")
    materialized_identities = {
        record.get("identity_id")
        for record in sequences
        if isinstance(record.get("identity_id"), str)
    }
    if set(caption_by_identity) != materialized_identities:
        missing = sorted(materialized_identities - set(caption_by_identity), key=str.encode)
        extra = sorted(set(caption_by_identity) - materialized_identities, key=str.encode)
        raise ValueError(f"caption identity closure mismatch: missing={missing!r}, extra={extra!r}")
    output_records = []
    action_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    identity_splits: dict[str, str] = {}
    prompts: set[str] = set()
    eligible_frame_count = 0
    excluded_sequence_count = 0
    for sequence in sorted(sequences, key=lambda row: str(row.get("sequence_id")).encode()):
        sequence_id = _required_text(sequence, "sequence_id")
        identity_id = _required_text(sequence, "identity_id")
        split = _required_text(sequence, "split")
        entity_class = _required_text(sequence, "entity_class")
        view = _required_text(sequence, "view")
        previous_split = identity_splits.setdefault(identity_id, split)
        if previous_split != split:
            raise ValueError(f"identity crosses splits: {identity_id}")
        taxonomy_record = taxonomy_by_sequence.get(sequence_id)
        if taxonomy_record is None:
            raise ValueError(f"taxonomy is missing sequence {sequence_id}")
        if (
            taxonomy_record.get("identity_id") != identity_id
            or taxonomy_record.get("split") != split
        ):
            raise ValueError(f"taxonomy identity/split mismatch for {sequence_id}")
        verb = _required_text(taxonomy_record, "verb")
        eligible_frame_indices = list(range(8))
        if eligibility_by_sequence is not None:
            eligibility_record = eligibility_by_sequence.get(sequence_id)
            if eligibility_record is None:
                raise ValueError(f"frame eligibility is missing sequence {sequence_id}")
            if (
                eligibility_record.get("identity_id") != identity_id
                or eligibility_record.get("split") != split
            ):
                raise ValueError(f"frame eligibility identity/split differs for {sequence_id}")
            raw_indices = eligibility_record.get("eligible_frame_indices")
            if (
                not isinstance(raw_indices, list)
                or any(
                    isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 8
                    for index in raw_indices
                )
                or raw_indices != sorted(set(raw_indices))
            ):
                raise ValueError(f"eligible frame indices are invalid for {sequence_id}")
            eligible_frame_indices = raw_indices
            if not eligible_frame_indices:
                excluded_sequence_count += 1
                continue
        caption = caption_by_identity[identity_id]
        if caption.get("split") != split or caption.get("entity_class") != entity_class:
            raise ValueError(f"caption split/entity mismatch for {identity_id}")
        structured = caption.get("structured_caption")
        if not isinstance(structured, dict):
            raise ValueError(f"caption lacks structured facts for {identity_id}")
        caption_input = caption.get("caption_input")
        if not isinstance(caption_input, dict):
            raise ValueError(f"caption lacks input evidence for {identity_id}")
        caption_input_sha256 = _required_text(caption_input, "file_sha256")
        reference_sha256 = _required_text(caption, "reference_array_sha256")
        request_sha256 = _required_text(caption, "request_body_sha256")
        for label, digest in (
            ("caption input", caption_input_sha256),
            ("caption reference", reference_sha256),
            ("caption request", request_sha256),
        ):
            _validate_sha256(digest, f"{identity_id} {label}")
        appearance = compact_appearance_prompt(structured, entity_class=entity_class)
        prompt = f"{appearance}; {action_phrase(verb)}; {view} view"
        prompts.add(prompt)
        target = sequence.get("output")
        if not isinstance(target, dict):
            raise ValueError(f"sequence target is invalid for {sequence_id}")
        relative_path = _required_text(target, "relative_path")
        target_path = (materialization_file.parent / relative_path).resolve()
        if materialization_file.parent not in target_path.parents:
            raise ValueError(f"sequence target escapes materialization root: {sequence_id}")
        for name in ("file_sha256", "array_content_sha256"):
            _validate_sha256(_required_text(target, name), f"{sequence_id} {name}")
        shape = target.get("shape")
        if shape != [8, 128, 128, 4]:
            raise ValueError(
                f"sequence target geometry is unsupported for {sequence_id}: {shape!r}"
            )
        if eligibility_bytes is None:
            sample_payload = f"mugen_still_sequence_v1\0{sequence_id}\0{caption_manifest_sha256}"
        else:
            sample_payload = (
                f"mugen_subject_bearing_still_sequence_v2\0{sequence_id}\0"
                f"{caption_manifest_sha256}\0{hashlib.sha256(eligibility_bytes).hexdigest()}"
            )
        sample_id = "still_sequence_" + hashlib.sha256(sample_payload.encode()).hexdigest()[:32]
        output_records.append(
            {
                "caption_reference": {
                    "caption_input_file_sha256": caption_input_sha256,
                    "identity_reference_array_sha256": reference_sha256,
                    "request_body_sha256": request_sha256,
                },
                "conditioning": {
                    "action_phrase": action_phrase(verb),
                    "attack_form": taxonomy_record.get("attack_form"),
                    "attack_strength": taxonomy_record.get("attack_strength"),
                    "attack_tier": taxonomy_record.get("attack_tier"),
                    "direction": taxonomy_record.get("direction"),
                    "stance": taxonomy_record.get("stance"),
                    "verb": verb,
                    "view": view,
                },
                "entity_class": entity_class,
                "identity_id": identity_id,
                "prompt": prompt,
                "sample_id": sample_id,
                "sequence_id": sequence_id,
                "split": split,
                "target": {
                    "array_content_sha256": target["array_content_sha256"],
                    "file_sha256": target["file_sha256"],
                    "frame_count": 8,
                    "frame_sampling": (
                        "uniform_subject_bearing_logical_frame_index"
                        if eligibility_bytes is not None
                        else "uniform_unique_logical_frame_index"
                    ),
                    "relative_path": relative_path,
                    "shape": shape,
                    **(
                        {"eligible_frame_indices": eligible_frame_indices}
                        if eligibility_bytes is not None
                        else {}
                    ),
                },
            }
        )
        action_counts[verb] += 1
        split_counts[split] += 1
        eligible_frame_count += len(eligible_frame_indices)
    if eligibility_by_sequence is None and len(output_records) != len(taxonomy_by_sequence):
        raise ValueError("taxonomy contains sequence rows absent from materialization")
    if eligibility_by_sequence is not None and set(eligibility_by_sequence) != set(
        taxonomy_by_sequence
    ):
        raise ValueError("frame eligibility sequence closure differs from taxonomy")
    retained_identities = {record["identity_id"] for record in output_records}
    return {
        "artifact_kind": "mugen_latent_still_sequence_training_plan",
        "counts": {
            "action_sequences": dict(
                sorted(action_counts.items(), key=lambda item: item[0].encode())
            ),
            "identities": (
                len(retained_identities) if eligibility_bytes is not None else len(identity_splits)
            ),
            "prompts": len(prompts),
            "sequences": len(output_records),
            "split_sequences": dict(
                sorted(split_counts.items(), key=lambda item: item[0].encode())
            ),
            **(
                {
                    "eligible_frames": eligible_frame_count,
                    "excluded_identities": len(identity_splits) - len(retained_identities),
                    "excluded_sequences": excluded_sequence_count,
                }
                if eligibility_bytes is not None
                else {}
            ),
        },
        "records": output_records,
        "sampler_contract": {
            "frame": (
                "uniform_subject_bearing_frame_within_selected_sequence"
                if eligibility_bytes is not None
                else "uniform_logical_frame_within_selected_sequence"
            ),
            "hierarchy": ["identity", "verb", "sequence", "frame"],
            "identity_split_disjoint": True,
            "raw_sequence_frequency_is_not_sampling_weight": True,
        },
        "schema_version": 2 if eligibility_bytes is not None else 1,
        "source": {
            "caption_manifest_file_sha256": caption_manifest_sha256,
            "caption_manifest_path": str(caption_file),
            "materialization_file_sha256": hashlib.sha256(materialization_bytes).hexdigest(),
            "materialization_path": str(materialization_file),
            "taxonomy_file_sha256": hashlib.sha256(taxonomy_bytes).hexdigest(),
            "taxonomy_path": str(taxonomy_file),
            **(
                {
                    "frame_eligibility_file_sha256": hashlib.sha256(eligibility_bytes).hexdigest(),
                    "frame_eligibility_path": str(eligibility_file),
                }
                if eligibility_bytes is not None
                else {}
            ),
        },
    }


def export_mugen_still_training_plan(
    materialization_path: Path | str,
    taxonomy_path: Path | str,
    caption_manifest_path: Path | str,
    output_path: Path | str,
    *,
    frame_eligibility_path: Path | str | None = None,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Write one canonical no-clobber still-training plan."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace MUGEN still training plan: {output}")
    plan = build_mugen_still_training_plan(
        materialization_path,
        taxonomy_path,
        caption_manifest_path,
        frame_eligibility_path,
    )
    payload = canonical_json_bytes(plan)
    (disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)).require_capacity(
        len(payload) + 16 * 1024**2,
        label="MUGEN latent still sequence training plan",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return output, hashlib.sha256(payload).hexdigest()


def _records(value: dict[str, Any], key: str, count_key: str, label: str) -> list[dict[str, Any]]:
    records = value.get(key)
    if not isinstance(records, list) or value.get(count_key) != len(records):
        raise ValueError(f"{label} count does not match records")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{label} records must be objects")
    return records


def _unique_index(records: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        value = _required_text(record, key)
        if value in output:
            raise ValueError(f"{label} has duplicate {key}: {value}")
        output[value] = record
    return output


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"record field {key} must be non-empty text")
    return value


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def _validate_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
