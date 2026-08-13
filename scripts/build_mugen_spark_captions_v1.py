"""Caption canonical MUGEN stills with the Spark's pinned Qwen vision service."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_stills import (  # noqa: E402
    MugenStillReference,
    compose_caption_input,
    load_mugen_still_references,
)
from spritelab.spark_caption import (  # noqa: E402
    canonical_json_bytes,
    caption_prompt_sha256,
    openai_vision_request,
    parse_structured_caption,
    structured_training_prompt,
)
from spritelab.storage import DiskGuard  # noqa: E402

MATERIALIZATION = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
TAXONOMY = ROOT / "data/index/reports/mugen-mffa-action-taxonomy-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-canonical-still-captions-v3-spark-qwen35-122b"
MODEL_ID = "RedHatAI/Qwen3.5-122B-A10B-NVFP4"
MODEL_REVISION = "49d19c108259a21450c40b8af38828b0a97390d8"
SERVED_MODEL = "qwen3.5-122b"


class CaptionValidationFailure(ValueError):
    def __init__(self, message: str, failure_record: dict[str, Any]) -> None:
        super().__init__(message)
        self.failure_record = failure_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.request_timeout <= 0:
        raise ValueError("--request-timeout must be positive")
    if args.workers not in {1, 2}:
        raise ValueError("--workers must be 1 or 2 for the pinned Spark service")
    references = load_mugen_still_references(MATERIALIZATION, TAXONOMY)
    if args.limit is not None:
        references = references[: args.limit]
    output = args.output.resolve()
    preflight = {
        "base_url": args.base_url,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "output": str(output),
        "prompt_sha256": caption_prompt_sha256(),
        "reference_count": len(references),
        "served_model": SERVED_MODEL,
        "workers": args.workers,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    DiskGuard(ROOT, min_free_bytes=100 * 1024**3).require_capacity(
        128 * 1024**2, label="Spark MUGEN structured captions"
    )
    output.mkdir(parents=True, exist_ok=True)
    input_dir = output / "caption-inputs"
    input_dir.mkdir(exist_ok=True)
    journal_path = output / "caption-records.jsonl"
    failure_journal_path = output / "caption-failures.jsonl"
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"caption manifest already exists: {manifest_path}")
    completed = _load_journal(journal_path)
    pending = [reference for reference in references if reference.identity_id not in completed]
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    with (
        journal_path.open("a", encoding="utf-8", newline="\n") as journal,
        ThreadPoolExecutor(max_workers=args.workers) as executor,
    ):
        futures: dict[Future[dict[str, Any]], MugenStillReference] = {
            executor.submit(
                _caption_reference,
                reference,
                output=output,
                input_dir=input_dir,
                endpoint=endpoint,
                timeout=args.request_timeout,
            ): reference
            for reference in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            reference = futures[future]
            try:
                record = future.result()
            except CaptionValidationFailure as error:
                _append_failure(failure_journal_path, error.failure_record)
                raise
            journal.write(canonical_json_bytes(record).decode("utf-8"))
            journal.flush()
            os.fsync(journal.fileno())
            print(
                json.dumps(
                    {
                        "captioned": len(completed) + index,
                        "identity_id": reference.identity_id,
                        "remaining": len(pending) - index,
                        "training_prompt": record["training_prompt"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
    completed = _load_journal(journal_path)
    expected_ids = [reference.identity_id for reference in references]
    if set(completed) != set(expected_ids):
        raise RuntimeError("caption journal is incomplete after generation")
    records = [completed[identity_id] for identity_id in expected_ids]
    manifest = {
        "artifact_kind": "mugen_canonical_still_structured_caption_dataset",
        "caption_contract": {
            "identity_and_franchise_hidden_from_model": True,
            "model_output_is_unverified": True,
            "prompt_sha256": caption_prompt_sha256(),
            "training_prompt_excludes_identity_label": True,
            "uncertain_features_excluded_from_training_prompt": True,
        },
        "caption_count": len(records),
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "served_model": SERVED_MODEL,
            "service": "vllm_openai_compatible_on_user_spark",
        },
        "records": records,
        "schema_version": 1,
        "source": {
            "materialization_file_sha256": _file_sha256(MATERIALIZATION),
            "materialization_path": str(MATERIALIZATION),
            "taxonomy_file_sha256": _file_sha256(TAXONOMY),
            "taxonomy_path": str(TAXONOMY),
        },
        "supersedes_for_training": {
            "file_sha256": ("7b969ce8c4150274f0da03a1e2ff04c2ef3bf35379100717f2129cb8a61056f8"),
            "path": str(
                ROOT / "data/processed/mugen-mffa-canonical-still-captions-v2/manifest.json"
            ),
            "reason": "smaller_captioner_hallucinated_character_identity_and_equipment",
        },
    }
    manifest_payload = canonical_json_bytes(manifest)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def _caption_reference(
    reference: MugenStillReference,
    *,
    output: Path,
    input_dir: Path,
    endpoint: str,
    timeout: float,
) -> dict[str, Any]:
    input_payload = _caption_input_png(reference.rgba)
    input_sha256 = hashlib.sha256(input_payload).hexdigest()
    input_path = input_dir / f"{reference.identity_id}-{input_sha256[:12]}.png"
    if input_path.exists():
        if _file_sha256(input_path) != input_sha256:
            raise RuntimeError(f"existing caption input hash mismatch: {input_path}")
    else:
        with input_path.open("xb") as handle:
            handle.write(input_payload)
            handle.flush()
            os.fsync(handle.fileno())
    data_url = "data:image/png;base64," + base64.b64encode(input_payload).decode("ascii")
    request = openai_vision_request(model=SERVED_MODEL, png_data_url=data_url)
    request_payload = canonical_json_bytes(request)
    response_payload = _post_json(
        endpoint,
        request_payload,
        timeout=timeout,
        attempts=3,
    )
    response = _json_object(response_payload, "caption service response")
    content = _response_content(response)
    try:
        structured = parse_structured_caption(content)
    except ValueError as error:
        failure_record = {
            "error": str(error),
            "identity_id": reference.identity_id,
            "model_response": response,
            "model_response_file_sha256": hashlib.sha256(response_payload).hexdigest(),
            "request_body_sha256": hashlib.sha256(request_payload).hexdigest(),
        }
        raise CaptionValidationFailure(str(error), failure_record) from error
    return {
        "alpha_bbox_xywh": list(reference.alpha_bbox_xywh)
        if reference.alpha_bbox_xywh is not None
        else None,
        "caption_input": {
            "file_sha256": input_sha256,
            "relative_path": input_path.relative_to(output).as_posix(),
            "resize": "512x512_nearest_neighbor",
            "rgba_composite_background_rgb": [127, 127, 127],
            "size_bytes": len(input_payload),
        },
        "caption_source": "remote_model_generated_unverified_literal_visual",
        "entity_class": reference.entity_class,
        "frame_index": reference.frame_index,
        "identity_id": reference.identity_id,
        "identity_label_provenance_only": reference.identity_label,
        "legacy_action": reference.legacy_action,
        "model_response": response,
        "model_response_file_sha256": hashlib.sha256(response_payload).hexdigest(),
        "palette_facts": [
            {"fraction": fraction, "name": name} for name, fraction in reference.palette_facts
        ],
        "reference_array_sha256": reference.reference_array_sha256,
        "request_body_sha256": hashlib.sha256(request_payload).hexdigest(),
        "sequence_id": reference.sequence_id,
        "source_array_sha256": reference.source_array_sha256,
        "source_file_sha256": reference.source_file_sha256,
        "split": reference.split,
        "structured_caption": structured,
        "structured_verb": reference.structured_verb,
        "training_prompt": structured_training_prompt(
            structured, entity_class=reference.entity_class
        ),
        "visible_pixel_count": reference.visible_pixel_count,
    }


def _caption_input_png(rgba) -> bytes:
    composite = compose_caption_input(rgba)
    image = Image.fromarray(composite).resize((512, 512), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


def _post_json(url: str, payload: bytes, *, timeout: float, attempts: int) -> bytes:
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as error:
            if attempt == attempts:
                if isinstance(error, urllib.error.HTTPError):
                    detail = error.read().decode("utf-8", errors="replace")[:2000]
                    raise RuntimeError(
                        f"caption service failed after {attempts} attempts: {error}; {detail}"
                    ) from error
                raise RuntimeError(
                    f"caption service failed after {attempts} attempts: {error}"
                ) from error
            time.sleep(attempt * 2)
    raise AssertionError("unreachable retry loop")


def _response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("caption response must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("caption response choice has no text content")
    return content


def _load_journal(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    output: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"caption journal has invalid line {line_number}") from error
            identity_id = record.get("identity_id") if isinstance(record, dict) else None
            if not isinstance(identity_id, str) or identity_id in output:
                raise RuntimeError(f"caption journal has invalid identity on line {line_number}")
            output[identity_id] = record
    return output


def _append_failure(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json_bytes(record).decode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
