"""Deterministic quality audit and review gallery for Spark sprite captions."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from spritelab.mugen_still_dataset import action_phrase, compact_appearance_prompt
from spritelab.storage import DiskGuard


class CaptionAuditError(ValueError):
    """Raised when the caption manifest or its referenced evidence differs."""


_GENERIC_PROVENANCE_LABELS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "index",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}


def build_caption_audit(
    manifest_path: Path | str,
    tokenizer_directory: Path | str,
    output_directory: Path | str,
    *,
    review_rows: int = 32,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Audit all captions, CLIP prompts, identity leakage, and a review sample."""

    try:
        from transformers import CLIPTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("caption token audit requires Transformers") from error
    if isinstance(review_rows, bool) or not isinstance(review_rows, int) or review_rows <= 0:
        raise ValueError("review_rows must be a positive integer")
    manifest_file = Path(manifest_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace caption audit output: {output}")
    manifest_bytes = manifest_file.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptionAuditError("caption manifest is invalid JSON") from error
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if (
        not isinstance(records, list)
        or manifest.get("caption_count") != len(records)
        or not all(isinstance(record, dict) for record in records)
    ):
        raise CaptionAuditError("caption manifest record count differs")
    tokenizer = CLIPTokenizer.from_pretrained(
        Path(tokenizer_directory).resolve(), local_files_only=True
    )
    if tokenizer.model_max_length != 77:
        raise CaptionAuditError("tokenizer does not have the SD1 77-token contract")
    entity_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    empty_counts: Counter[str] = Counter()
    token_counts = []
    identity_leakage = []
    reasoning_identity_mentions = []
    uncertain_prompt_leakage = []
    compact_records = []
    seen_identities = set()
    for record in records:
        identity_id = _text(record, "identity_id")
        if identity_id in seen_identities:
            raise CaptionAuditError(f"duplicate caption identity: {identity_id}")
        seen_identities.add(identity_id)
        entity_class = _text(record, "entity_class")
        split = _text(record, "split")
        verb = _text(record, "structured_verb")
        structured = record.get("structured_caption")
        if not isinstance(structured, dict):
            raise CaptionAuditError(f"structured caption is absent for {identity_id}")
        appearance = compact_appearance_prompt(structured, entity_class=entity_class)
        prompt = f"{appearance}; {action_phrase(verb)}; side view"
        token_count = len(tokenizer.encode(prompt, add_special_tokens=True))
        token_counts.append(token_count)
        if token_count > 77:
            raise CaptionAuditError(f"compact prompt exceeds CLIP limit for {identity_id}")
        entity_counts[entity_class] += 1
        subject_counts[str(structured.get("subject_type", ""))] += 1
        split_counts[split] += 1
        for key in (
            "skin_or_surface",
            "hair",
            "face",
            "upper_body_clothing",
            "lower_body_clothing",
            "footwear",
            "armor",
        ):
            if not isinstance(structured.get(key), str) or not structured.get(key, "").strip():
                empty_counts[key] += 1
        label = record.get("identity_label_provenance_only")
        label_normalized = " ".join(str(label).casefold().split()) if label is not None else ""
        structured_text = json.dumps(structured, ensure_ascii=False).casefold()
        training_prompt = str(record.get("training_prompt", "")).casefold()
        response_text = json.dumps(record.get("model_response", {}), ensure_ascii=False).casefold()
        label_is_distinctive = (
            len(label_normalized) >= 4 and label_normalized not in _GENERIC_PROVENANCE_LABELS
        )
        if label_is_distinctive and (
            label_normalized in structured_text or label_normalized in training_prompt
        ):
            identity_leakage.append(identity_id)
        if label_is_distinctive and label_normalized in response_text:
            reasoning_identity_mentions.append(identity_id)
        uncertain = structured.get("uncertain_visible_features")
        if isinstance(uncertain, list):
            certain_structure = dict(structured)
            certain_structure.pop("uncertain_visible_features", None)
            certain_text = json.dumps(certain_structure, ensure_ascii=False).casefold()
            leaked = [
                value
                for value in uncertain
                if isinstance(value, str)
                and value.strip()
                and value.casefold() in training_prompt
                and value.casefold() not in certain_text
            ]
            if leaked:
                uncertain_prompt_leakage.append({"identity_id": identity_id, "values": leaked})
        compact_records.append(
            {
                "compact_prompt": prompt,
                "entity_class": entity_class,
                "identity_id": identity_id,
                "input_path": _caption_input_path(manifest_file.parent, record),
                "split": split,
                "subject_type": structured.get("subject_type"),
                "token_count": token_count,
            }
        )
    selected = _balanced_review_selection(compact_records, review_rows)
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(64 * 1024**2, label="caption review audit")
    output.mkdir(parents=True, exist_ok=False)
    gallery_path = output / "caption-review-gallery.png"
    _render_gallery(selected, gallery_path)
    report = {
        "artifact_kind": "mugen_spark_structured_caption_quality_audit",
        "caption_manifest": {
            "file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "path": str(manifest_file),
        },
        "counts": {
            "captions": len(records),
            "empty_literal_fields": dict(sorted(empty_counts.items())),
            "entity_classes": dict(sorted(entity_counts.items())),
            "splits": dict(sorted(split_counts.items())),
            "subject_types": dict(sorted(subject_counts.items())),
        },
        "gallery": {
            "file_sha256": _file_sha256(gallery_path),
            "path": gallery_path.name,
            "rows": len(selected),
        },
        "identity_blinding_audit": {
            "reasoning_or_raw_response_full_label_mentions": reasoning_identity_mentions,
            "structured_or_training_prompt_full_label_leaks": identity_leakage,
        },
        "model_output_status": "remote_model_generated_unverified_visual_description",
        "review_selection": [
            {key: value for key, value in record.items() if key != "input_path"}
            for record in selected
        ],
        "tokenization": {
            "maximum": max(token_counts),
            "mean": sum(token_counts) / len(token_counts),
            "over_77": 0,
        },
        "uncertain_feature_training_prompt_leakage": uncertain_prompt_leakage,
    }
    report_path = output / "audit-report.json"
    payload = _canonical_json(report)
    with report_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return report_path, hashlib.sha256(payload).hexdigest()


def _balanced_review_selection(records: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["subject_type"])].append(record)
    for values in grouped.values():
        values.sort(key=lambda row: hashlib.sha256(row["identity_id"].encode()).digest())
    output = []
    keys = sorted(grouped, key=str.encode)
    while keys and len(output) < maximum:
        remaining = []
        for key in keys:
            if grouped[key] and len(output) < maximum:
                output.append(grouped[key].pop(0))
            if grouped[key]:
                remaining.append(key)
        keys = remaining
    return output


def _render_gallery(records: list[dict[str, Any]], output: Path) -> None:
    columns = 4
    card_width = 384
    card_height = 420
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * card_width, rows * card_height), (28, 30, 36))
    draw = ImageDraw.Draw(canvas)
    for index, record in enumerate(records):
        x = (index % columns) * card_width
        y = (index // columns) * card_height
        image = Image.open(record["input_path"]).convert("RGBA")
        image.thumbnail((256, 256), resample=Image.Resampling.NEAREST)
        panel = Image.new("RGBA", (256, 256), (127, 127, 127, 255))
        panel.alpha_composite(image, ((256 - image.width) // 2, (256 - image.height) // 2))
        canvas.paste(panel.convert("RGB"), (x + 64, y + 8))
        header = (
            f"{record['entity_class']} / {record['subject_type']} / {record['token_count']} tok"
        )
        draw.text((x + 8, y + 272), header, fill=(231, 235, 242))
        lines = _wrap_text(record["compact_prompt"], 54)[:7]
        for line_index, line in enumerate(lines):
            draw.text((x + 8, y + 294 + 16 * line_index), line, fill=(180, 188, 201))
    canvas.save(output, format="PNG", optimize=False)


def _wrap_text(value: str, width: int) -> list[str]:
    output = []
    current = ""
    for word in value.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            output.append(current)
            current = word
        else:
            current = candidate
    if current:
        output.append(current)
    return output


def _caption_input_path(root: Path, record: dict[str, Any]) -> Path:
    caption_input = record.get("caption_input")
    if not isinstance(caption_input, dict):
        raise CaptionAuditError("caption input record is absent")
    relative = _text(caption_input, "relative_path")
    path = (root / relative).resolve()
    if root not in path.parents or _file_sha256(path) != caption_input.get("file_sha256"):
        raise CaptionAuditError(f"caption input file differs: {relative}")
    return path


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise CaptionAuditError(f"field {key} must be non-empty text")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
