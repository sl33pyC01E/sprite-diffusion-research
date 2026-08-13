"""Frozen token-level CLIP text-state cache for the latent still DiT."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from spritelab.storage import DiskGuard


class TextTokenCacheError(ValueError):
    """Raised when a text model, prompt plan, or cached tensor violates its contract."""


def export_clip_text_token_cache(
    training_plan_path: Path | str,
    model_directory: Path | str,
    output_directory: Path | str,
    *,
    expected_source_index_sha256: str,
    batch_size: int = 64,
    device: str = "cuda",
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Encode every unique plan prompt to exact CLIP token hidden states."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    _validate_digest(expected_source_index_sha256, "expected_source_index_sha256")
    plan_path = Path(training_plan_path).resolve()
    model_root = Path(model_directory).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace text token cache: {output}")
    plan_bytes = plan_path.read_bytes()
    plan = _json_object(plan_bytes, "training plan")
    records = plan.get("records")
    if not isinstance(records, list) or not records:
        raise TextTokenCacheError("training plan records are empty")
    prompts = tuple(
        sorted(
            {
                record.get("prompt")
                for record in records
                if isinstance(record, dict) and isinstance(record.get("prompt"), str)
            },
            key=str.encode,
        )
    )
    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise TextTokenCacheError("training prompts must be non-empty strings")
    source_index_path = model_root / "source-index.json"
    source_index_bytes = source_index_path.read_bytes()
    if hashlib.sha256(source_index_bytes).hexdigest() != expected_source_index_sha256:
        raise TextTokenCacheError("Stable Diffusion source-index SHA-256 mismatch")
    source_index = _json_object(source_index_bytes, "Stable Diffusion source index")
    _verify_model_files(model_root, source_index)
    try:
        import torch
        import transformers
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("CLIP token export requires Torch and Transformers") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA text token export requested but unavailable")
    tokenizer = CLIPTokenizer.from_pretrained(model_root / "tokenizer", local_files_only=True)
    maximum_tokens = int(tokenizer.model_max_length)
    if maximum_tokens != 77:
        raise TextTokenCacheError(f"unexpected CLIP maximum token count: {maximum_tokens}")
    overlong = [
        {"prompt": prompt, "tokens": len(tokenizer.encode(prompt, add_special_tokens=True))}
        for prompt in prompts
        if len(tokenizer.encode(prompt, add_special_tokens=True)) > maximum_tokens
    ]
    if overlong:
        raise TextTokenCacheError(
            f"{len(overlong)} prompts exceed CLIP's 77-token contract; first={overlong[0]!r}"
        )
    model = (
        CLIPTextModel.from_pretrained(
            model_root / "text_encoder",
            local_files_only=True,
            use_safetensors=True,
        )
        .to(device)
        .eval()
    )
    hidden_width = int(model.config.hidden_size)
    if hidden_width != 768:
        raise TextTokenCacheError(f"unexpected CLIP hidden width: {hidden_width}")
    encoded_prompts = ("", *prompts)
    encoded_embeddings = np.empty(
        (len(encoded_prompts), maximum_tokens, hidden_width), dtype=np.float16
    )
    encoded_input_ids = np.empty((len(encoded_prompts), maximum_tokens), dtype=np.int32)
    encoded_attention_mask = np.empty((len(encoded_prompts), maximum_tokens), dtype=np.bool_)
    for start in range(0, len(encoded_prompts), batch_size):
        batch = encoded_prompts[start : start + batch_size]
        tokens = tokenizer(
            list(batch),
            padding="max_length",
            truncation=False,
            max_length=maximum_tokens,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = model(
                tokens.input_ids.to(device),
                attention_mask=tokens.attention_mask.to(device),
            ).last_hidden_state
        end = start + len(batch)
        encoded_embeddings[start:end] = hidden.float().cpu().numpy().astype(np.float16)
        encoded_input_ids[start:end] = tokens.input_ids.numpy().astype(np.int32)
        encoded_attention_mask[start:end] = tokens.attention_mask.numpy().astype(np.bool_)
    embeddings = np.ascontiguousarray(encoded_embeddings[1:])
    input_ids = np.ascontiguousarray(encoded_input_ids[1:])
    attention_mask = np.ascontiguousarray(encoded_attention_mask[1:])
    unconditional_embeddings = np.ascontiguousarray(encoded_embeddings[0])
    unconditional_input_ids = np.ascontiguousarray(encoded_input_ids[0])
    unconditional_attention_mask = np.ascontiguousarray(encoded_attention_mask[0])
    validate_text_token_arrays(embeddings, input_ids, attention_mask, row_count=len(prompts))
    guard = disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(
        embeddings.nbytes + input_ids.nbytes + attention_mask.nbytes + 128 * 1024**2,
        label="frozen CLIP token-state cache",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent)
    )
    arrays = {
        "attention_mask": attention_mask,
        "embeddings": embeddings,
        "input_ids": input_ids,
        "unconditional_attention_mask": unconditional_attention_mask,
        "unconditional_embeddings": unconditional_embeddings,
        "unconditional_input_ids": unconditional_input_ids,
    }
    array_records = {}
    for name, value in arrays.items():
        path = staging / f"{name}.npy"
        with path.open("xb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        array_records[name] = {
            "array_content_sha256": _array_sha256(value),
            "dtype": value.dtype.str,
            "file_sha256": _file_sha256(path),
            "path": path.name,
            "shape": list(value.shape),
        }
    rows = [
        {
            "embedding_row_sha256": _array_sha256(embeddings[index]),
            "prompt": prompt,
            "prompt_utf8_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "row_index": index,
        }
        for index, prompt in enumerate(prompts)
    ]
    manifest = {
        "arrays": array_records,
        "artifact_kind": "frozen_clip_token_hidden_state_cache",
        "classifier_free_unconditional": {
            "array_names": {
                "attention_mask": "unconditional_attention_mask",
                "embeddings": "unconditional_embeddings",
                "input_ids": "unconditional_input_ids",
            },
            "prompt": "",
            "prompt_utf8_sha256": hashlib.sha256(b"").hexdigest(),
        },
        "encoder": {
            "library": "transformers",
            "library_version": transformers.__version__,
            "model_id": source_index["model_id"],
            "model_revision": source_index["resolved_revision"],
            "output": "CLIPTextModel.last_hidden_state_float16",
            "source_index_file_sha256": expected_source_index_sha256,
        },
        "prompt_count": len(prompts),
        "rows": rows,
        "schema_version": 1,
        "source": {
            "training_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "training_plan_path": str(plan_path),
        },
        "tokenization": {
            "maximum_tokens": maximum_tokens,
            "overlong_prompt_count": 0,
            "padding": "max_length",
            "truncation": False,
        },
    }
    manifest_payload = _canonical_json(manifest)
    with (staging / "manifest.json").open("xb") as handle:
        handle.write(manifest_payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.rename(staging, output)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to replace text token cache: {output}") from error
    return output / "manifest.json", hashlib.sha256(manifest_payload).hexdigest()


def validate_text_token_arrays(
    embeddings: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    *,
    row_count: int,
) -> None:
    """Validate exact frozen-token tensor geometry and numeric semantics."""

    if embeddings.dtype != np.float16 or embeddings.shape != (row_count, 77, 768):
        raise TextTokenCacheError("embeddings must be float16 [N,77,768]")
    if input_ids.dtype != np.int32 or input_ids.shape != (row_count, 77):
        raise TextTokenCacheError("input_ids must be int32 [N,77]")
    if attention_mask.dtype != np.bool_ or attention_mask.shape != (row_count, 77):
        raise TextTokenCacheError("attention_mask must be bool [N,77]")
    if not bool(np.isfinite(embeddings).all()):
        raise TextTokenCacheError("embeddings contain non-finite values")
    if bool((input_ids < 0).any()):
        raise TextTokenCacheError("input IDs must be non-negative")
    if not bool(attention_mask[:, 0].all()) or not bool(attention_mask.any(axis=1).all()):
        raise TextTokenCacheError("every prompt must retain at least its first token")


def _verify_model_files(model_root: Path, source_index: dict[str, Any]) -> None:
    files = source_index.get("files")
    if not isinstance(files, list) or not files:
        raise TextTokenCacheError("Stable Diffusion source index files are invalid")
    for record in files:
        relative = record.get("relative_path") if isinstance(record, dict) else None
        expected = record.get("file_sha256") if isinstance(record, dict) else None
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TextTokenCacheError("Stable Diffusion source file record is invalid")
        path = (model_root / relative).resolve()
        if model_root not in path.parents or _file_sha256(path) != expected:
            raise TextTokenCacheError(f"Stable Diffusion source file mismatch: {relative}")


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TextTokenCacheError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise TextTokenCacheError(f"{label} must contain an object")
    return value


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_digest(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
