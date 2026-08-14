"""Resume the bounded Spark/Qwen MUGEN motion-role precision audit."""

from __future__ import annotations

import argparse
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

from spritelab.mugen_motion_dataset import _array_sha256  # noqa: E402
from spritelab.mugen_motion_role import (  # noqa: E402
    motion_role_prompt_sha256,
    motion_role_vlm_request,
    parse_motion_role_vlm_response,
)
from spritelab.mugen_still_eligibility import subject_contact_sheet  # noqa: E402
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

SAMPLE = ROOT / "data/processed/mugen-mffa-motion-role-vlm-sample-v1.json"
PLAN = ROOT / "data/processed/mugen-mffa-reference-motion-plan-v2.json"
OUTPUT_ROOT = ROOT / "data/processed/mugen-mffa-motion-role-vlm-decisions-v2-prompt-json"
MODEL = "qwen3.5-122b"
USE_RESPONSE_FORMAT = False


def main(*, endpoint: str, workers: int, prepare_only: bool) -> None:
    if workers <= 0:
        raise ValueError("workers must be positive")
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    sample_bytes = SAMPLE.read_bytes()
    sample = _object(sample_bytes, "sample")
    plan_bytes = PLAN.read_bytes()
    plan = _object(plan_bytes, "plan")
    if (
        sample.get("source", {}).get("motion_plan_file_sha256")
        != hashlib.sha256(plan_bytes).hexdigest()
    ):
        raise RuntimeError("sample motion-plan hash differs")
    records = sample.get("records")
    plan_records = plan.get("records")
    if not isinstance(records, list) or sample.get("counts", {}).get("records") != len(records):
        raise RuntimeError("sample record count differs")
    if not isinstance(plan_records, list):
        raise RuntimeError("plan records are absent")
    sequence_by_id = {record["sequence_id"]: record for record in plan_records}
    materialization_path = Path(plan["source"]["materialization"]["path"]).resolve()
    if _file_sha256(materialization_path) != plan["source"]["materialization"]["file_sha256"]:
        raise RuntimeError("materialization hash differs")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    input_dir = OUTPUT_ROOT / "contact-sheets"
    input_dir.mkdir(parents=True, exist_ok=True)
    prepared = {
        record["sequence_id"]: _prepare(
            record,
            sequence_by_id=sequence_by_id,
            materialization_root=materialization_path.parent,
            input_dir=input_dir,
        )
        for record in records
    }
    if prepare_only:
        print(json.dumps({"prepared": len(prepared), "root": str(OUTPUT_ROOT)}, sort_keys=True))
        return
    journal = OUTPUT_ROOT / "records.jsonl"
    completed = _load_journal(journal)
    pending = [record for record in records if record["sequence_id"] not in completed]
    print(
        json.dumps(
            {"completed": len(completed), "pending": len(pending), "workers": workers},
            sort_keys=True,
        ),
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _classify,
                record,
                prepared=prepared[record["sequence_id"]],
                endpoint=endpoint,
            ): record["sequence_id"]
            for record in pending
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            decision = future.result()
            _append(journal, decision)
            completed[decision["sequence_id"]] = decision
            if index == 1 or index % 10 == 0 or index == len(pending):
                print(
                    json.dumps(
                        {"classified_this_run": index, "total": len(completed)}, sort_keys=True
                    ),
                    flush=True,
                )
    ordered = [completed[record["sequence_id"]] for record in records]
    manifest_path = OUTPUT_ROOT / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace role manifest: {manifest_path}")
    valid = [record for record in ordered if isinstance(record.get("decision"), dict)]
    invalid = [record for record in ordered if isinstance(record.get("decision_error"), dict)]
    if len(valid) + len(invalid) != len(ordered):
        raise RuntimeError("role records have an unknown decision status")
    accepted = sum(record["decision"]["conservative_same_subject_motion"] for record in valid)
    manifest = {
        "artifact_kind": "mugen_motion_role_qwen35_122b_decisions",
        "counts": {
            "conservative_same_subject_motion": accepted,
            "invalid_model_responses": len(invalid),
            "records": len(ordered),
            "rejected_or_ambiguous": len(valid) - accepted,
            "valid_decisions": len(valid),
        },
        "endpoint": {
            "model": MODEL,
            "service": "llama_cpp_openai_compatible_on_user_spark",
            "structured_output_mode": "prompt_constrained_json_without_response_format",
        },
        "records": ordered,
        "schema_version": 2,
        "source": {
            "motion_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "prompt_contract_sha256": motion_role_prompt_sha256(
                use_response_format=USE_RESPONSE_FORMAT
            ),
            "sample_file_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        },
    }
    payload = canonical_json_bytes(manifest)
    guard.require_capacity(len(payload) + 1024**2, label="MUGEN motion-role decisions")
    with manifest_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "counts": manifest["counts"],
                "manifest": str(manifest_path),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def _prepare(
    record: dict[str, Any],
    *,
    sequence_by_id: dict[str, dict[str, Any]],
    materialization_root: Path,
    input_dir: Path,
) -> dict[str, Any]:
    sequence_id = record["sequence_id"]
    plan_record = sequence_by_id[sequence_id]
    if plan_record["identity_id"] != record["identity_id"]:
        raise RuntimeError(f"sample identity differs for {sequence_id}")
    target = _clip(materialization_root, plan_record["target"]["source_pixels"])
    reference_record = sequence_by_id[plan_record["reference"]["sequence_id"]]
    reference_clip = _clip(materialization_root, reference_record["target"]["source_pixels"])
    frame_index = plan_record["reference"]["frame_index"]
    reference = np.ascontiguousarray(reference_clip[frame_index])
    expected_reference_sha = plan_record["reference"]["identity_reference_array_sha256"]
    if _array_sha256(reference) != expected_reference_sha:
        raise RuntimeError(f"reference frame hash differs for {sequence_id}")
    sheet = subject_contact_sheet(reference, target)
    sheet_sha256 = hashlib.sha256(sheet).hexdigest()
    sheet_path = input_dir / f"{sequence_id}-{sheet_sha256[:12]}.png"
    if sheet_path.exists():
        if _file_sha256(sheet_path) != sheet_sha256:
            raise RuntimeError(f"existing contact sheet differs: {sheet_path}")
    else:
        with sheet_path.open("xb") as handle:
            handle.write(sheet)
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "contact_sheet_file_sha256": sheet_sha256,
        "contact_sheet_relative_path": sheet_path.relative_to(OUTPUT_ROOT).as_posix(),
        "sheet_png": sheet,
    }


def _classify(record: dict[str, Any], *, prepared: dict[str, Any], endpoint: str) -> dict[str, Any]:
    request = motion_role_vlm_request(
        model=MODEL,
        sheet_png=prepared["sheet_png"],
        expected_verb=record["expected_verb"],
        use_response_format=USE_RESPONSE_FORMAT,
    )
    request_payload = canonical_json_bytes(request)
    response_payload = _post(endpoint, request_payload)
    response = _object(response_payload, "VLM response")
    result = {
        "contact_sheet": {
            "file_sha256": prepared["contact_sheet_file_sha256"],
            "relative_path": prepared["contact_sheet_relative_path"],
        },
        "expected_verb": record["expected_verb"],
        "identity_id": record["identity_id"],
        "model_response": response,
        "model_response_file_sha256": hashlib.sha256(response_payload).hexdigest(),
        "request_body_sha256": hashlib.sha256(request_payload).hexdigest(),
        "sequence_id": record["sequence_id"],
        "split": record["split"],
    }
    try:
        result["decision"] = parse_motion_role_vlm_response(_response_content(response))
    except ValueError as error:
        result["decision_error"] = {
            "error_type": type(error).__name__,
            "message": str(error),
            "policy": "quarantine_without_coercion",
        }
    return result


def _clip(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = (root / record["relative_path"]).resolve()
    if root != path and root not in path.parents:
        raise RuntimeError("clip path escapes materialization root")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record["file_sha256"]:
        raise RuntimeError(f"clip file hash differs: {path}")
    value = np.load(path, allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise RuntimeError(f"clip geometry differs: {path}")
    if _array_sha256(value) != record["array_content_sha256"]:
        raise RuntimeError(f"clip array hash differs: {path}")
    return np.ascontiguousarray(value)


def _post(endpoint: str, payload: bytes) -> bytes:
    for attempt in range(1, 4):
        request = urllib.request.Request(
            endpoint,
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


def _object(payload: bytes, label: str) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://spark:8080/v1/chat/completions")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    main(endpoint=args.endpoint, workers=args.workers, prepare_only=args.prepare_only)
