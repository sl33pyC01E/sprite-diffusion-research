"""Caption dense MUGEN identity references with Spark's installed Qwen service."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.spark_caption import (  # noqa: E402
    canonical_json_bytes,
    caption_prompt_sha256,
    openai_vision_request,
    parse_structured_caption,
    structured_training_prompt,
)
from spritelab.storage import DiskGuard  # noqa: E402

MODEL_ALIAS = "Qwen-Spark"
MODEL_FAMILY = "Qwen3.5-122B-A10B-MXFP4-MOE-GGUF"
SERVICE_UNIT = "qwen-122b.service"
MODEL_IDENTITY_REPORT = (
    ROOT / "data/index/reports/spark-qwen35-122b-existing-caption-service-v1.json"
)
MODEL_IDENTITY_REPORT_SHA256 = "26d512b8de876910163a35f919bab1fbf96a3b37943517f0d009e91db1d4e85f"


class CaptionFailure(ValueError):
    def __init__(self, message: str, record: dict[str, Any]) -> None:
        super().__init__(message)
        self.record = record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_manifest", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--base-url", default="http://spark:8080/v1")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    inputs, input_bytes = load_caption_inputs(args.input_manifest)
    model_identity_bytes = MODEL_IDENTITY_REPORT.read_bytes()
    if hashlib.sha256(model_identity_bytes).hexdigest() != MODEL_IDENTITY_REPORT_SHA256:
        raise RuntimeError("Spark model identity report SHA-256 differs")
    output = args.output_directory.resolve()
    preflight = {
        "base_url": args.base_url,
        "input_count": len(inputs),
        "input_manifest_file_sha256": hashlib.sha256(input_bytes).hexdigest(),
        "model_alias": MODEL_ALIAS,
        "model_family": MODEL_FAMILY,
        "model_identity_report_sha256": MODEL_IDENTITY_REPORT_SHA256,
        "output": str(output),
        "prompt_sha256": caption_prompt_sha256(),
        "service_unit": SERVICE_UNIT,
        "workers": args.workers,
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    DiskGuard(ROOT, 100 * 1024**3).require_capacity(
        512 * 1024**2, label="dense MUGEN Spark captions"
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"caption manifest already exists: {manifest_path}")
    journal_path = output / "caption-records.jsonl"
    failure_path = output / "caption-failures.jsonl"
    completed = _load_journal(journal_path)
    expected = {_text(row, "variant_id") for row in inputs}
    if set(completed) - expected:
        raise RuntimeError("caption journal contains variants outside the input manifest")
    pending = [row for row in inputs if row["variant_id"] not in completed]
    groups = _caption_request_groups(pending, completed)
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    with journal_path.open("a", encoding="utf-8", newline="\n") as journal:
        remote_groups = []
        finished = 0
        reused = 0
        for donor, sources in groups:
            if donor is None:
                remote_groups.append((sources[0], sources[1:]))
                continue
            for source in sources:
                record = _reuse_caption(donor, source)
                journal.write(canonical_json_bytes(record).decode("utf-8"))
                finished += 1
                reused += 1
        if reused:
            journal.flush()
            os.fsync(journal.fileno())
            print(
                json.dumps(
                    {
                        "captioned": len(completed) + finished,
                        "exact_render_reused": reused,
                        "remaining": len(pending) - finished,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        executor = ThreadPoolExecutor(max_workers=args.workers)
        iterator = iter(remote_groups)
        futures: dict[
            Future[dict[str, Any]], tuple[dict[str, Any], tuple[dict[str, Any], ...]]
        ] = {}

        def submit_next() -> bool:
            try:
                row, aliases = next(iterator)
            except StopIteration:
                return False
            futures[
                executor.submit(
                    _caption,
                    row,
                    endpoint=endpoint,
                    timeout=args.request_timeout,
                )
            ] = (row, aliases)
            return True

        for _ in range(args.workers):
            submit_next()
        finished = 0
        try:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    source, aliases = futures.pop(future)
                    try:
                        record = future.result()
                    except CaptionFailure as error:
                        _append(failure_path, error.record)
                        raise
                    journal.write(canonical_json_bytes(record).decode("utf-8"))
                    for alias in aliases:
                        journal.write(
                            canonical_json_bytes(_reuse_caption(record, alias)).decode("utf-8")
                        )
                    journal.flush()
                    os.fsync(journal.fileno())
                    finished += 1 + len(aliases)
                    reused += len(aliases)
                    print(
                        json.dumps(
                            {
                                "captioned": len(completed) + finished,
                                "exact_render_reused": reused,
                                "remaining": len(pending) - finished,
                                "subject_type": record["structured_caption"]["subject_type"],
                                "variant_id": source["variant_id"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    submit_next()
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    completed = _load_journal(journal_path)
    if set(completed) != expected:
        raise RuntimeError("caption journal is incomplete")
    records = [completed[row["variant_id"]] for row in inputs]
    manifest = {
        "artifact_kind": "mugen_dense_literal_visual_caption_dataset",
        "caption_contract": {
            "exact_render_duplicate_caption_reuse": True,
            "identity_and_franchise_hidden_from_model": True,
            "model_output_is_unverified": True,
            "pose_and_facing_excluded_from_identity_appearance_prompt": True,
            "prompt_sha256": caption_prompt_sha256(),
            "uncertain_features_excluded_from_training_prompt": True,
        },
        "model": {
            "alias": MODEL_ALIAS,
            "family": MODEL_FAMILY,
            "identity_report_file_sha256": MODEL_IDENTITY_REPORT_SHA256,
            "identity_report_path": str(MODEL_IDENTITY_REPORT),
            "service": "existing_user_managed_llama_cpp_on_spark",
            "service_unit": SERVICE_UNIT,
        },
        "record_count": len(records),
        "records": records,
        "schema_version": 1,
        "source": {
            "caption_input_manifest_file_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "caption_input_manifest_path": str(args.input_manifest.resolve()),
        },
    }
    payload = canonical_json_bytes(manifest)
    with manifest_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {"manifest": str(manifest_path), "sha256": hashlib.sha256(payload).hexdigest()},
            sort_keys=True,
        )
    )


def load_caption_inputs(path_value: Path | str) -> tuple[list[dict[str, Any]], bytes]:
    path = Path(path_value).resolve()
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("artifact_kind") != (
        "mugen_dense_literal_visual_caption_input_dataset"
    ):
        raise ValueError("caption input manifest has the wrong artifact kind")
    records = value.get("records")
    if (
        not isinstance(records, list)
        or value.get("record_count") != len(records)
        or any(not isinstance(row, dict) for row in records)
    ):
        raise ValueError("caption input record count differs")
    seen = set()
    for row in records:
        variant_id = _text(row, "variant_id")
        if variant_id in seen:
            raise ValueError(f"caption inputs duplicate variant: {variant_id}")
        seen.add(variant_id)
        input_record = _object(row.get("caption_input"), "caption input")
        relative = _safe_relative(_text(input_record, "relative_path"))
        image_path = path.parent.joinpath(*relative.parts).resolve()
        if path.parent != image_path and path.parent not in image_path.parents:
            raise ValueError(f"caption input escapes artifact: {variant_id}")
        image_payload = image_path.read_bytes()
        if hashlib.sha256(image_payload).hexdigest() != _digest(input_record, "file_sha256"):
            raise ValueError(f"caption input hash differs: {variant_id}")
        if len(image_payload) != input_record.get("size_bytes"):
            raise ValueError(f"caption input size differs: {variant_id}")
        row["_image_path"] = str(image_path)
    return records, payload


def _caption(source: dict[str, Any], *, endpoint: str, timeout: float) -> dict[str, Any]:
    image_path = Path(_text(source, "_image_path"))
    image_payload = image_path.read_bytes()
    data_url = "data:image/png;base64," + base64.b64encode(image_payload).decode("ascii")
    response = None
    response_payload = None
    request_payload = None
    structured = None
    last_error: Exception | None = None
    for max_tokens in (1024, 2048):
        request = openai_vision_request(
            model=MODEL_ALIAS,
            png_data_url=data_url,
            max_tokens=max_tokens,
            enable_thinking=False,
        )
        request_payload = canonical_json_bytes(request)
        response_payload = _post_json(endpoint, request_payload, timeout=timeout, attempts=3)
        response = _json_object(response_payload, "caption response")
        try:
            structured = parse_structured_caption(_response_content(response))
        except (RuntimeError, ValueError) as error:
            last_error = error
            continue
        break
    if structured is None:
        assert response is not None and response_payload is not None and request_payload is not None
        assert last_error is not None
        failure = {
            "error": str(last_error),
            "model_response": response,
            "model_response_file_sha256": hashlib.sha256(response_payload).hexdigest(),
            "request_body_sha256": hashlib.sha256(request_payload).hexdigest(),
            "variant_id": source["variant_id"],
        }
        raise CaptionFailure(str(last_error), failure) from last_error
    assert response is not None and response_payload is not None and request_payload is not None
    subject_type = structured["subject_type"]
    assert isinstance(subject_type, str)
    return {
        "caption_input": source["caption_input"],
        "caption_source": "remote_model_generated_unverified_literal_visual",
        "frame_index": source["frame_index"],
        "identity_id": source["identity_id"],
        "identity_label_provenance_only": source["identity_label_provenance_only"],
        "model_response": response,
        "model_response_file_sha256": hashlib.sha256(response_payload).hexdigest(),
        "reference_frame_array_content_sha256": source["reference_frame_array_content_sha256"],
        "request_body_sha256": hashlib.sha256(request_payload).hexdigest(),
        "split": source["split"],
        "structured_caption": structured,
        "training_appearance_prompt": structured_training_prompt(
            structured,
            entity_class=subject_type,
            include_pose_and_facing=False,
        ),
        "variant_id": source["variant_id"],
    }


def _caption_request_groups(
    pending: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, Any] | None, tuple[dict[str, Any], ...]]]:
    donors: dict[str, dict[str, Any]] = {}
    for variant_id in sorted(completed, key=str.encode):
        record = completed[variant_id]
        digest = _digest(_object(record.get("caption_input"), "caption input"), "file_sha256")
        existing = donors.get(digest)
        if existing is not None and (
            existing.get("structured_caption") != record.get("structured_caption")
            or existing.get("training_appearance_prompt")
            != record.get("training_appearance_prompt")
        ):
            raise RuntimeError("completed captions disagree for one exact rendered caption input")
        donors.setdefault(digest, record)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for source in pending:
        digest = _digest(_object(source.get("caption_input"), "caption input"), "file_sha256")
        grouped.setdefault(digest, []).append(source)
    return [
        (donors.get(digest), tuple(rows))
        for digest, rows in sorted(grouped.items(), key=lambda item: item[0])
    ]


def _reuse_caption(donor: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    donor_input = _object(donor.get("caption_input"), "donor caption input")
    source_input = _object(source.get("caption_input"), "source caption input")
    digest = _digest(source_input, "file_sha256")
    if _digest(donor_input, "file_sha256") != digest:
        raise ValueError("caption reuse requires byte-identical rendered inputs")
    record = dict(donor)
    for key in (
        "caption_input",
        "frame_index",
        "identity_id",
        "identity_label_provenance_only",
        "reference_frame_array_content_sha256",
        "split",
        "variant_id",
    ):
        record[key] = source[key]
    record["caption_reuse"] = {
        "caption_input_file_sha256": digest,
        "method": "exact_rendered_caption_input_sha256_v1",
        "source_variant_id": _text(donor, "variant_id"),
    }
    return record


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
                detail = ""
                if isinstance(error, urllib.error.HTTPError):
                    detail = "; " + error.read().decode("utf-8", errors="replace")[:2000]
                raise RuntimeError(
                    f"caption service failed after {attempts} attempts: {error}{detail}"
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
    output = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"caption journal has invalid line {line_number}") from error
            variant_id = record.get("variant_id") if isinstance(record, dict) else None
            if not isinstance(variant_id, str) or not variant_id or variant_id in output:
                raise RuntimeError(f"caption journal has invalid variant on line {line_number}")
            output[variant_id] = record
    return output


def _append(path: Path, record: dict[str, Any]) -> None:
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


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


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


if __name__ == "__main__":
    main()
