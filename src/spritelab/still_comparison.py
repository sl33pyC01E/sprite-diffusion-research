"""Exact target/generated galleries for the still-image quality experiments."""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from spritelab.sd_control_cache import composite_rgba_on_background
from spritelab.sprite_postprocess import (
    SpriteDisplayDecodeConfig,
    composite_rgba_on_checkerboard,
    decode_generated_rgb_sprite,
)
from spritelab.storage import DiskGuard


class StillComparisonError(ValueError):
    """Raised when an inference sample cannot be matched to exact target evidence."""


def build_sd_lora_ablation_comparison(
    inference_reports: list[tuple[str, Path | str, str]],
    plan_path: Path | str,
    output_directory: Path | str,
    *,
    target_sequence_ids: list[str] | None = None,
    display_decode_config: SpriteDisplayDecodeConfig | None = None,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Compare exact targets with two or more same-noise inference reports."""

    if len(inference_reports) < 2:
        raise ValueError("at least two inference reports are required")
    labels = [label for label, _, _ in inference_reports]
    if any(not isinstance(label, str) or not label.strip() for label in labels) or len(
        set(labels)
    ) != len(labels):
        raise ValueError("inference labels must be unique non-empty text")
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace still ablation comparison: {output}")
    loaded = []
    for label, raw_path, expected_sha256 in inference_reports:
        path = Path(raw_path).resolve()
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise StillComparisonError(f"inference report SHA-256 differs: {label}")
        report = _json_object(payload, f"{label} inference report")
        if report.get("artifact_kind") != "mugen_sd14_attention_lora_rgb_inference":
            raise StillComparisonError(f"inference report has wrong kind: {label}")
        samples = report.get("samples")
        if not isinstance(samples, list) or not samples:
            raise StillComparisonError(f"inference samples are absent: {label}")
        loaded.append((label, path, expected_sha256, report, samples))
    prompt_order = [_text(sample, "prompt") for sample in loaded[0][4]]
    if target_sequence_ids is not None and (
        len(target_sequence_ids) != len(prompt_order)
        or len(set(target_sequence_ids)) != len(target_sequence_ids)
        or any(not isinstance(value, str) or not value for value in target_sequence_ids)
    ):
        raise ValueError("target_sequence_ids must uniquely align with inference prompts")
    noise_sha256 = loaded[0][3].get("noise_batch_sha256")
    for label, _, _, report, samples in loaded[1:]:
        if [_text(sample, "prompt") for sample in samples] != prompt_order:
            raise StillComparisonError(f"inference prompt order differs: {label}")
        if report.get("noise_batch_sha256") != noise_sha256:
            raise StillComparisonError(f"inference noise batch differs: {label}")
    plan_file = Path(plan_path).resolve()
    plan_bytes = plan_file.read_bytes()
    plan = _json_object(plan_bytes, "training plan")
    records = plan.get("records")
    if not isinstance(records, list) or plan.get("counts", {}).get("sequences") != len(records):
        raise StillComparisonError("training plan record count differs")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise StillComparisonError("training plan source is absent")
    materialization_path = Path(_text(source, "materialization_path")).resolve()
    if _file_sha256(materialization_path) != source.get("materialization_file_sha256"):
        raise StillComparisonError("materialization manifest differs")
    target_by_prompt: dict[str, dict[str, Any]] = {}
    target_by_sequence: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda row: str(row.get("sequence_id")).encode()):
        if not isinstance(record, dict):
            continue
        prompt = record.get("prompt")
        sequence_id = record.get("sequence_id")
        if isinstance(prompt, str):
            target_by_prompt.setdefault(prompt, record)
        if isinstance(sequence_id, str) and sequence_id:
            if sequence_id in target_by_sequence:
                raise StillComparisonError("training plan has duplicate sequence IDs")
            target_by_sequence[sequence_id] = record
    rows = []
    for sample_index, prompt in enumerate(prompt_order):
        target_sequence_id = (
            target_sequence_ids[sample_index] if target_sequence_ids is not None else None
        )
        target_record = (
            target_by_sequence.get(target_sequence_id)
            if target_sequence_id is not None
            else target_by_prompt.get(prompt)
        )
        if target_record is None:
            raise StillComparisonError(
                f"no target uses sequence/prompt: {target_sequence_id!r} / {prompt!r}"
            )
        if target_record.get("prompt") != prompt:
            raise StillComparisonError("target sequence prompt differs from inference prompt")
        target = target_record.get("target")
        if not isinstance(target, dict):
            raise StillComparisonError("training target is absent")
        rgba = _load_target_rgba(materialization_path.parent, target)
        eligible = _eligible_indices(target)
        target_frame_index = eligible[0]
        target_rgb = composite_rgba_on_background(
            rgba[target_frame_index : target_frame_index + 1]
        )[0]
        generated = {}
        generated_display = {}
        generated_decode = {}
        generated_hashes = {}
        for label, report_path, _, _, samples in loaded:
            generated_record = samples[sample_index].get("downsample_128")
            if not isinstance(generated_record, dict):
                raise StillComparisonError(f"generated 128px record is absent: {label}")
            image_path = (report_path.parent / _text(generated_record, "path")).resolve()
            if _file_sha256(image_path) != generated_record.get("file_sha256"):
                raise StillComparisonError(f"generated PNG differs: {label}")
            generated[label] = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
            generated_hashes[label] = generated_record["file_sha256"]
            if display_decode_config is not None:
                decoded, decode_metadata = decode_generated_rgb_sprite(
                    generated[label], config=display_decode_config
                )
                generated_display[label] = composite_rgba_on_checkerboard(decoded)
                generated_decode[label] = decode_metadata
        rows.append(
            {
                "generated": generated,
                "generated_decode": generated_decode,
                "generated_display": generated_display,
                "generated_file_sha256": generated_hashes,
                "identity_id": target_record.get("identity_id"),
                "prompt": prompt,
                "sequence_id": target_record.get("sequence_id"),
                "target": np.rint(target_rgb * 255).clip(0, 255).astype(np.uint8),
                "target_display": composite_rgba_on_checkerboard(rgba[target_frame_index]),
                "target_array_content_sha256": target.get("array_content_sha256"),
                "target_frame_index": target_frame_index,
            }
        )
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(128 * 1024**2, label="still LoRA ablation comparison")
    output.mkdir(parents=True, exist_ok=False)
    gallery_path = output / "target-vs-lora-ablation-gallery.png"
    _render_ablation_gallery(rows, labels, gallery_path)
    display_gallery = None
    if display_decode_config is not None:
        display_gallery = output / "target-vs-lora-ablation-display-gallery.png"
        _render_ablation_gallery(
            rows,
            labels,
            display_gallery,
            target_key="target_display",
            generated_key="generated_display",
        )
    report = {
        "artifact_kind": "mugen_sd14_lora_same_noise_ablation_comparison",
        "claim": (
            "identity-held-out prompts with exact subject-bearing targets; all generated columns "
            "share prompt order and initial noise; qualitative comparison only"
        ),
        "gallery": {
            "file_sha256": _file_sha256(gallery_path),
            "path": gallery_path.name,
        },
        "display_decode": (
            {
                "claim": (
                    "exact target alpha is canonical; generated alpha and palette are "
                    "display-only derivatives inferred without target pixels"
                ),
                "config": asdict(display_decode_config),
                "gallery": {
                    "file_sha256": _file_sha256(display_gallery),
                    "path": display_gallery.name,
                },
            }
            if display_gallery is not None and display_decode_config is not None
            else None
        ),
        "inference_reports": [
            {"file_sha256": sha256, "label": label, "path": str(path)}
            for label, path, sha256, _, _ in loaded
        ],
        "noise_batch_sha256": noise_sha256,
        "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "rows": [
            {
                key: value
                for key, value in row.items()
                if key not in {"generated", "generated_display", "target", "target_display"}
            }
            for row in rows
        ],
        "target_selection": (
            "explicit_sequence_id" if target_sequence_ids is not None else "prompt_first_match"
        ),
    }
    report_path = output / "comparison-report.json"
    payload = _canonical_json(report)
    with report_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return report_path, hashlib.sha256(payload).hexdigest()


def build_sd_lora_target_comparison(
    inference_report_path: Path | str,
    plan_path: Path | str,
    output_directory: Path | str,
    *,
    expected_inference_report_sha256: str,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Render exact gray-composited targets beside SD-LoRA RGB generations."""

    inference_file = Path(inference_report_path).resolve()
    plan_file = Path(plan_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace still comparison: {output}")
    inference_bytes = inference_file.read_bytes()
    if hashlib.sha256(inference_bytes).hexdigest() != expected_inference_report_sha256:
        raise StillComparisonError("inference report SHA-256 mismatch")
    inference = _json_object(inference_bytes, "inference report")
    if inference.get("artifact_kind") != "mugen_sd14_attention_lora_rgb_inference":
        raise StillComparisonError("inference report has the wrong artifact kind")
    plan_bytes = plan_file.read_bytes()
    plan = _json_object(plan_bytes, "training plan")
    records = plan.get("records")
    count = (
        plan.get("counts", {}).get("sequences") if isinstance(plan.get("counts"), dict) else None
    )
    if not isinstance(records, list) or count != len(records):
        raise StillComparisonError("training plan record count differs")
    plan_source = plan.get("source")
    if not isinstance(plan_source, dict):
        raise StillComparisonError("training plan source is absent")
    materialization_path = Path(_text(plan_source, "materialization_path")).resolve()
    if _file_sha256(materialization_path) != plan_source.get("materialization_file_sha256"):
        raise StillComparisonError("materialization manifest differs")
    target_by_prompt = {}
    for record in sorted(records, key=lambda row: str(row.get("sequence_id")).encode()):
        prompt = record.get("prompt") if isinstance(record, dict) else None
        if isinstance(prompt, str):
            target_by_prompt.setdefault(prompt, record)
    samples = inference.get("samples")
    if not isinstance(samples, list) or not samples:
        raise StillComparisonError("inference samples are absent")
    rows = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise StillComparisonError("inference sample is invalid")
        prompt = _text(sample, "prompt")
        target_record = target_by_prompt.get(prompt)
        if target_record is None:
            raise StillComparisonError(f"no exact target uses prompt: {prompt!r}")
        target = target_record.get("target")
        if not isinstance(target, dict):
            raise StillComparisonError("training target record is absent")
        rgba = _load_target_rgba(materialization_path.parent, target)
        generated_record = sample.get("downsample_128")
        if not isinstance(generated_record, dict):
            raise StillComparisonError("generated 128px record is absent")
        generated_path = (inference_file.parent / _text(generated_record, "path")).resolve()
        if _file_sha256(generated_path) != generated_record.get("file_sha256"):
            raise StillComparisonError("generated PNG hash differs")
        eligible = _eligible_indices(target)
        target_frame_index = eligible[0]
        target_rgb = composite_rgba_on_background(
            rgba[target_frame_index : target_frame_index + 1]
        )[0]
        target_uint8 = np.rint(target_rgb * 255).clip(0, 255).astype(np.uint8)
        generated = np.asarray(Image.open(generated_path).convert("RGB"), dtype=np.uint8)
        rows.append(
            {
                "generated": generated,
                "generated_file_sha256": generated_record["file_sha256"],
                "identity_id": target_record.get("identity_id"),
                "prompt": prompt,
                "sequence_id": target_record.get("sequence_id"),
                "target": target_uint8,
                "target_array_content_sha256": target.get("array_content_sha256"),
                "target_frame_index": target_frame_index,
            }
        )
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(64 * 1024**2, label="still target comparison")
    output.mkdir(parents=True, exist_ok=False)
    gallery_path = output / "target-vs-generated-gallery.png"
    _render_gallery(rows, gallery_path)
    report = {
        "artifact_kind": "mugen_sd14_lora_exact_target_comparison",
        "claim": (
            "targets and generated images share prompts but not noise/pixel alignment; "
            "gallery is a qualitative held-out-identity comparison"
        ),
        "gallery": {
            "file_sha256": _file_sha256(gallery_path),
            "path": gallery_path.name,
        },
        "inference_report_file_sha256": expected_inference_report_sha256,
        "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "rows": [
            {key: value for key, value in row.items() if key not in {"generated", "target"}}
            for row in rows
        ],
    }
    report_path = output / "comparison-report.json"
    payload = _canonical_json(report)
    with report_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return report_path, hashlib.sha256(payload).hexdigest()


def _render_gallery(rows: list[dict[str, Any]], output: Path) -> None:
    width = 1024
    row_height = 560
    canvas = Image.new("RGB", (width, row_height * len(rows)), (25, 27, 32))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        top = index * row_height
        target = Image.fromarray(row["target"], mode="RGB").resize(
            (384, 384), resample=Image.Resampling.NEAREST
        )
        generated = Image.fromarray(row["generated"], mode="RGB").resize(
            (384, 384), resample=Image.Resampling.NEAREST
        )
        canvas.paste(target, (80, top + 40))
        canvas.paste(generated, (560, top + 40))
        draw.text(
            (80, top + 16),
            f"EXACT SUBJECT-BEARING TARGET (frame {row['target_frame_index']})",
            fill=(220, 226, 236),
        )
        draw.text((560, top + 16), "GENERATED (same prompt)", fill=(220, 226, 236))
        prompt_lines = _wrap(row["prompt"], 105)[:5]
        for line_index, line in enumerate(prompt_lines):
            draw.text((32, top + 442 + line_index * 18), line, fill=(178, 188, 204))
    canvas.save(output, format="PNG", optimize=False)


def _render_ablation_gallery(
    rows: list[dict[str, Any]],
    labels: list[str],
    output: Path,
    *,
    target_key: str = "target",
    generated_key: str = "generated",
) -> None:
    image_size = 320
    gap = 32
    left = 48
    columns = 1 + len(labels)
    width = left * 2 + columns * image_size + (columns - 1) * gap
    row_height = 450
    canvas = Image.new("RGB", (width, row_height * len(rows)), (25, 27, 32))
    draw = ImageDraw.Draw(canvas)
    headings = ["EXACT SUBJECT-BEARING TARGET", *labels]
    for row_index, row in enumerate(rows):
        top = row_index * row_height
        images = [row[target_key], *(row[generated_key][label] for label in labels)]
        for column, (heading, value) in enumerate(zip(headings, images, strict=True)):
            x = left + column * (image_size + gap)
            image = Image.fromarray(value, mode="RGB").resize(
                (image_size, image_size), resample=Image.Resampling.NEAREST
            )
            canvas.paste(image, (x, top + 42))
            draw.text((x, top + 16), heading, fill=(220, 226, 236))
        summary = f"frame {row['target_frame_index']} | {row['sequence_id']} | {row['prompt']}"
        for line_index, line in enumerate(_wrap(summary, max(100, width // 9))[:4]):
            draw.text((32, top + 374 + line_index * 18), line, fill=(178, 188, 204))
    canvas.save(output, format="PNG", optimize=False)


def _load_target_rgba(root: Path, target: dict[str, Any]) -> np.ndarray:
    path = (root / _text(target, "relative_path")).resolve()
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != target.get("file_sha256"):
        raise StillComparisonError("target file hash differs")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise StillComparisonError("target array geometry differs")
    return value


def _eligible_indices(target: dict[str, Any]) -> list[int]:
    eligible = target.get("eligible_frame_indices", [0])
    if (
        not isinstance(eligible, list)
        or not eligible
        or any(
            isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 8
            for index in eligible
        )
        or eligible != sorted(set(eligible))
    ):
        raise StillComparisonError("eligible target frame indices are invalid")
    return eligible


def _wrap(value: str, maximum: int) -> list[str]:
    output = []
    line = ""
    for word in value.split():
        candidate = f"{line} {word}".strip()
        if line and len(candidate) > maximum:
            output.append(line)
            line = word
        else:
            line = candidate
    if line:
        output.append(line)
    return output


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StillComparisonError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise StillComparisonError(f"{label} must be an object")
    return value


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise StillComparisonError(f"field {key} must be non-empty text")
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
