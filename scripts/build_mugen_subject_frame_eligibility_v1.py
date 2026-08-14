"""Build the pixel gate and Spark-adjudicated MUGEN still-frame eligibility."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_still_eligibility import (  # noqa: E402
    export_subject_frame_pixel_audit,
    merge_subject_frame_eligibility,
    parse_subject_frame_vlm_response,
    subject_contact_sheet,
    subject_frame_vlm_request,
)
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

MATERIALIZATION = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
CAPTIONS = (
    ROOT / "data/processed/mugen-mffa-canonical-still-captions-v3-spark-qwen35-122b/manifest.json"
)
PIXEL_AUDIT = ROOT / "data/index/reports/mugen-mffa-subject-frame-pixel-gate-v1.json"
VLM_ROOT = ROOT / "data/processed/mugen-mffa-subject-frame-vlm-v1"
ELIGIBILITY = ROOT / "data/processed/mugen-mffa-subject-bearing-still-eligibility-v1.json"
ENDPOINT = "http://spark:8080/v1/chat/completions"
MODEL = "qwen3.5-122b"


def main() -> None:
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    if not PIXEL_AUDIT.exists():
        path, sha256 = export_subject_frame_pixel_audit(
            MATERIALIZATION, CAPTIONS, PIXEL_AUDIT, disk_guard=guard
        )
        print(json.dumps({"pixel_audit": str(path), "sha256": sha256}, sort_keys=True))
    pixel_bytes = PIXEL_AUDIT.read_bytes()
    pixel_audit = _object(pixel_bytes, "pixel audit")
    records = pixel_audit.get("records")
    if not isinstance(records, list):
        raise RuntimeError("pixel audit records are missing")
    mixed = [record for record in records if record.get("pixel_gate_status") == "mixed"]
    materialization = _object(MATERIALIZATION.read_bytes(), "materialization")
    captions = _object(CAPTIONS.read_bytes(), "captions")
    sequence_by_id = {record["sequence_id"]: record for record in materialization["sequences"]}
    caption_by_identity = {record["identity_id"]: record for record in captions["records"]}
    input_dir = VLM_ROOT / "contact-sheets"
    VLM_ROOT.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    journal = VLM_ROOT / "records.jsonl"
    completed = _load_journal(journal)
    pending = [record for record in mixed if record["sequence_id"] not in completed]
    print(
        json.dumps(
            {"mixed": len(mixed), "completed": len(completed), "pending": len(pending)},
            sort_keys=True,
        )
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                _classify,
                record,
                sequence_by_id=sequence_by_id,
                caption_by_identity=caption_by_identity,
                input_dir=input_dir,
            ): record["sequence_id"]
            for record in pending
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            _append(journal, record)
            completed[record["sequence_id"]] = record
            if index == 1 or index % 10 == 0 or index == len(pending):
                print(
                    json.dumps(
                        {"classified_this_run": index, "total": len(completed)}, sort_keys=True
                    ),
                    flush=True,
                )
    ordered = [completed[record["sequence_id"]] for record in mixed]
    manifest_path = VLM_ROOT / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "artifact_kind": "mugen_subject_frame_qwen35_122b_decisions",
            "counts": {"records": len(ordered)},
            "endpoint": {
                "model": MODEL,
                "service": "llama_cpp_openai_compatible_on_user_spark",
            },
            "records": ordered,
            "schema_version": 1,
            "source": {
                "pixel_audit_file_sha256": hashlib.sha256(pixel_bytes).hexdigest(),
                "pixel_audit_path": str(PIXEL_AUDIT),
            },
        }
        _publish(manifest_path, canonical_json_bytes(manifest), guard=guard)
    manifest_bytes = manifest_path.read_bytes()
    manifest = _object(manifest_bytes, "VLM manifest")
    if (
        manifest.get("source", {}).get("pixel_audit_file_sha256")
        != hashlib.sha256(pixel_bytes).hexdigest()
    ):
        raise RuntimeError("VLM manifest pixel-audit binding differs")
    eligibility = merge_subject_frame_eligibility(pixel_audit, manifest["records"])
    eligibility["source"]["pixel_audit_file_sha256"] = hashlib.sha256(pixel_bytes).hexdigest()
    eligibility["source"]["vlm_manifest_file_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    if not ELIGIBILITY.exists():
        payload = canonical_json_bytes(eligibility)
        _publish(ELIGIBILITY, payload, guard=guard)
    print(
        json.dumps(
            {
                "eligibility": str(ELIGIBILITY),
                "sha256": hashlib.sha256(ELIGIBILITY.read_bytes()).hexdigest(),
                "counts": eligibility["counts"],
            },
            sort_keys=True,
        )
    )


def _classify(
    audit_record: dict[str, Any],
    *,
    sequence_by_id: dict[str, dict[str, Any]],
    caption_by_identity: dict[str, dict[str, Any]],
    input_dir: Path,
) -> dict[str, Any]:
    sequence_id = audit_record["sequence_id"]
    identity_id = audit_record["identity_id"]
    sequence = sequence_by_id[sequence_id]
    caption = caption_by_identity[identity_id]
    reference_sequence = sequence_by_id[caption["sequence_id"]]
    clip = _clip(sequence)
    reference = _clip(reference_sequence)[caption["frame_index"]]
    sheet = subject_contact_sheet(reference, clip)
    sheet_sha256 = hashlib.sha256(sheet).hexdigest()
    sheet_path = input_dir / f"{sequence_id}-{sheet_sha256[:12]}.png"
    if sheet_path.exists():
        if hashlib.sha256(sheet_path.read_bytes()).hexdigest() != sheet_sha256:
            raise RuntimeError(f"existing contact sheet differs: {sheet_path}")
    else:
        with sheet_path.open("xb") as handle:
            handle.write(sheet)
            handle.flush()
            os.fsync(handle.fileno())
    request = subject_frame_vlm_request(model=MODEL, sheet_png=sheet)
    request_payload = canonical_json_bytes(request)
    response_payload = _post(request_payload)
    response = _object(response_payload, "VLM response envelope")
    content = _response_content(response)
    decision = parse_subject_frame_vlm_response(content)
    return {
        **decision,
        "contact_sheet": {
            "file_sha256": sheet_sha256,
            "relative_path": sheet_path.relative_to(VLM_ROOT).as_posix(),
        },
        "identity_id": identity_id,
        "model_response": response,
        "model_response_file_sha256": hashlib.sha256(response_payload).hexdigest(),
        "pixel_gate_pass_indices": audit_record["pixel_gate_pass_indices"],
        "request_body_sha256": hashlib.sha256(request_payload).hexdigest(),
        "sequence_id": sequence_id,
    }


def _clip(sequence: dict[str, Any]) -> np.ndarray:
    output = sequence["output"]
    path = (MATERIALIZATION.parent / output["relative_path"]).resolve()
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise RuntimeError(f"clip geometry differs: {path}")
    return value


def _post(payload: bytes) -> bytes:
    for attempt in range(1, 4):
        request = urllib.request.Request(
            ENDPOINT,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
                return response.read()
        except (TimeoutError, urllib.error.HTTPError, urllib.error.URLError) as error:
            if attempt == 3:
                detail = ""
                if isinstance(error, urllib.error.HTTPError):
                    detail = error.read().decode("utf-8", errors="replace")[:2000]
                raise RuntimeError(f"Spark VLM request failed: {error}; {detail}") from error
            time.sleep(attempt * 2)
    raise AssertionError("unreachable retry loop")


def _response_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("VLM response must contain one choice")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("VLM response has no content")
    return content


def _load_journal(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = _object(line.encode(), f"journal line {line_number}")
            sequence_id = record.get("sequence_id")
            if not isinstance(sequence_id, str) or sequence_id in output:
                raise RuntimeError(f"journal sequence is invalid at line {line_number}")
            output[sequence_id] = record
    return output


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json_bytes(record).decode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _publish(path: Path, payload: bytes, *, guard: DiskGuard) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to replace artifact: {path}")
    guard.require_capacity(len(payload) + 16 * 1024**2, label=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


if __name__ == "__main__":
    main()
