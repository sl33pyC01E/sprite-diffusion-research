"""Exact target/generated galleries for the still-image quality experiments."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from spritelab.sd_control_cache import composite_rgba_on_background
from spritelab.storage import DiskGuard


class StillComparisonError(ValueError):
    """Raised when an inference sample cannot be matched to exact target evidence."""


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
        target_path = (materialization_path.parent / _text(target, "relative_path")).resolve()
        target_payload = target_path.read_bytes()
        if hashlib.sha256(target_payload).hexdigest() != target.get("file_sha256"):
            raise StillComparisonError("target file hash differs")
        rgba = np.load(io.BytesIO(target_payload), allow_pickle=False)
        if rgba.dtype != np.uint8 or rgba.shape != (8, 128, 128, 4):
            raise StillComparisonError("target array geometry differs")
        generated_record = sample.get("downsample_128")
        if not isinstance(generated_record, dict):
            raise StillComparisonError("generated 128px record is absent")
        generated_path = (inference_file.parent / _text(generated_record, "path")).resolve()
        if _file_sha256(generated_path) != generated_record.get("file_sha256"):
            raise StillComparisonError("generated PNG hash differs")
        eligible = target.get("eligible_frame_indices", [0])
        if (
            not isinstance(eligible, list)
            or not eligible
            or any(
                isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 8
                for index in eligible
            )
        ):
            raise StillComparisonError("eligible target frame indices are invalid")
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
