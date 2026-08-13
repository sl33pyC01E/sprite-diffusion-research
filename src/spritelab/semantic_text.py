"""Hash-bound, precomputed semantic text embeddings for sprite conditioning.

The byte-level condition encoder remains a useful plumbing baseline, but it cannot
provide semantic language transfer.  This module keeps the replacement boundary
reproducible: a frozen encoder produces one L2-normalized float32 vector per NFC
description, and an immutable table records the exact model snapshot, tokenizer,
input ordering, array bytes, and every row digest.

Loading a table never downloads code or weights.  The optional Transformers backend
also accepts only a local model snapshot so acquisition and inference cannot silently
change underneath a training run.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from spritelab.storage import DiskGuard

_DIGEST_CHARS = frozenset("0123456789abcdef")


class SemanticTextError(ValueError):
    """Raised when a semantic embedding artifact violates its exact contract."""


@dataclass(frozen=True, slots=True)
class SemanticEncoderFile:
    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.relative_path or "\\" in self.relative_path:
            raise ValueError("relative_path must be a non-empty POSIX path")
        if Path(self.relative_path).is_absolute() or ".." in Path(self.relative_path).parts:
            raise ValueError("relative_path must remain inside the encoder snapshot")
        _validate_digest(self.sha256, "sha256")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("size_bytes must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class SemanticEncoderDescriptor:
    """Portable identity of the exact frozen text encoder and tokenizer."""

    backend: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    library: str
    library_version: str
    embedding_dim: int
    maximum_tokens: int
    output_contract: str
    snapshot_tree_sha256: str
    files: tuple[SemanticEncoderFile, ...]

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "library",
            "library_version",
            "output_contract",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        for name in ("embedding_dim", "maximum_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _validate_digest(self.snapshot_tree_sha256, "snapshot_tree_sha256")
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(paths, key=str.encode)) or len(paths) != len(set(paths)):
            raise ValueError("encoder evidence files must be unique and UTF-8 sorted")


class SemanticTextBackend(Protocol):
    """Frozen backend accepted by :func:`export_semantic_embedding_table`."""

    @property
    def descriptor(self) -> SemanticEncoderDescriptor: ...

    def encode(self, descriptions: Sequence[str]) -> np.ndarray:
        """Return finite, L2-normalized float32 vectors in input order."""


@dataclass(frozen=True, slots=True)
class SemanticEmbeddingTable:
    artifact_directory: Path
    manifest_path: Path
    manifest_sha256: str
    embeddings_path: Path
    embeddings_file_sha256: str
    embeddings_array_sha256: str
    descriptor: SemanticEncoderDescriptor
    descriptions: tuple[str, ...]
    embeddings: np.ndarray

    def lookup(self, description: str) -> np.ndarray:
        normalized = normalize_description(description)
        try:
            index = self.descriptions.index(normalized)
        except ValueError as error:
            raise KeyError(f"description is absent from semantic table: {normalized!r}") from error
        return self.embeddings[index].copy()

    def lookup_many(self, descriptions: Sequence[str]) -> np.ndarray:
        if not descriptions:
            raise ValueError("descriptions cannot be empty")
        return np.stack(tuple(self.lookup(value) for value in descriptions), axis=0)


@dataclass(frozen=True, slots=True)
class SemanticEmbeddingExportResult:
    artifact_directory: Path
    manifest_path: Path
    manifest_sha256: str
    embeddings_path: Path
    embeddings_file_sha256: str
    embeddings_array_sha256: str
    description_count: int
    embedding_dim: int


def normalize_description(value: str) -> str:
    """Strip and NFC-normalize one non-empty semantic prompt."""

    if not isinstance(value, str):
        raise TypeError("description must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError("description cannot be empty")
    return normalized


def export_semantic_embedding_table(
    descriptions: Sequence[str],
    output_directory: Path | str,
    backend: SemanticTextBackend,
    *,
    batch_size: int = 64,
    disk_guard: DiskGuard | None = None,
) -> SemanticEmbeddingExportResult:
    """Encode, verify, and atomically publish a unique sorted prompt table."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not descriptions:
        raise ValueError("descriptions cannot be empty")
    normalized = tuple(
        sorted(
            {normalize_description(value) for value in descriptions},
            key=str.encode,
        )
    )
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace semantic embedding artifact: {output}")
    descriptor = backend.descriptor

    batches: list[np.ndarray] = []
    for start in range(0, len(normalized), batch_size):
        values = backend.encode(normalized[start : start + batch_size])
        _validate_vectors(
            values,
            row_count=min(batch_size, len(normalized) - start),
            embedding_dim=descriptor.embedding_dim,
        )
        batches.append(np.ascontiguousarray(values))
    embeddings = np.ascontiguousarray(np.concatenate(batches, axis=0))
    array_sha256 = _array_sha256(embeddings)
    array_payload = _npy_bytes(embeddings)
    array_file_sha256 = hashlib.sha256(array_payload).hexdigest()
    rows = [
        {
            "description": description,
            "description_utf8_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest(),
            "row_index": index,
            "vector_sha256": _array_sha256(embeddings[index]),
        }
        for index, description in enumerate(normalized)
    ]
    manifest = {
        "artifact_kind": "spritelab_semantic_text_embedding_table",
        "array": {
            "array_sha256": array_sha256,
            "dtype": "float32",
            "file": "embeddings.npy",
            "file_sha256": array_file_sha256,
            "l2_normalized": True,
            "shape": list(embeddings.shape),
        },
        "encoder": _descriptor_record(descriptor),
        "normalization": {
            "description": "strip_then_unicode_NFC",
            "duplicate_policy": "deduplicate_after_normalization",
            "order": "ascending_UTF-8_bytes",
            "vector": "float32_L2_unit_norm",
        },
        "rows": rows,
        "schema_version": 1,
    }
    manifest_payload = _canonical_json_bytes(manifest)
    estimate = len(array_payload) + len(manifest_payload)
    if disk_guard is not None:
        disk_guard.require_capacity(estimate, label="semantic embedding table publication")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_new_file(stage / "embeddings.npy", array_payload)
        _write_new_file(stage / "manifest.json", manifest_payload)
        if output.exists():
            raise FileExistsError(f"Refusing to replace semantic embedding artifact: {output}")
        os.rename(stage, output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    manifest_path = output / "manifest.json"
    return SemanticEmbeddingExportResult(
        artifact_directory=output,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        embeddings_path=output / "embeddings.npy",
        embeddings_file_sha256=array_file_sha256,
        embeddings_array_sha256=array_sha256,
        description_count=len(normalized),
        embedding_dim=descriptor.embedding_dim,
    )


def load_semantic_embedding_table(directory: Path | str) -> SemanticEmbeddingTable:
    """Strictly hash-verify and load a semantic table without backend access."""

    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    manifest_payload = manifest_path.read_bytes()
    try:
        raw = json.loads(manifest_payload)
    except json.JSONDecodeError as error:
        raise SemanticTextError(f"invalid semantic manifest JSON: {error}") from error
    if raw.get("artifact_kind") != "spritelab_semantic_text_embedding_table":
        raise SemanticTextError("unexpected semantic artifact_kind")
    if raw.get("schema_version") != 1:
        raise SemanticTextError("unsupported semantic embedding schema_version")
    descriptor = _descriptor_from_record(raw.get("encoder"))
    array = raw.get("array")
    if not isinstance(array, Mapping):
        raise SemanticTextError("array must be an object")
    if array.get("file") != "embeddings.npy" or array.get("dtype") != "float32":
        raise SemanticTextError("semantic array file/dtype contract mismatch")
    if array.get("l2_normalized") is not True:
        raise SemanticTextError("semantic array must declare L2 normalization")
    file_sha = _required_digest(array.get("file_sha256"), "array.file_sha256")
    array_sha = _required_digest(array.get("array_sha256"), "array.array_sha256")
    embeddings_path = root / "embeddings.npy"
    payload = embeddings_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != file_sha:
        raise SemanticTextError("semantic embeddings file SHA-256 mismatch")
    try:
        embeddings = np.load(io.BytesIO(payload), allow_pickle=False)
    except (ValueError, OSError) as error:
        raise SemanticTextError(f"invalid semantic embeddings NPY: {error}") from error
    declared_shape = array.get("shape")
    if declared_shape != list(embeddings.shape):
        raise SemanticTextError("semantic embeddings shape differs from manifest")
    _validate_vectors(
        embeddings,
        row_count=embeddings.shape[0],
        embedding_dim=descriptor.embedding_dim,
    )
    if _array_sha256(embeddings) != array_sha:
        raise SemanticTextError("semantic embeddings canonical array hash mismatch")
    rows = raw.get("rows")
    if not isinstance(rows, list) or len(rows) != embeddings.shape[0]:
        raise SemanticTextError("semantic row count differs from array")
    descriptions: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or row.get("row_index") != index:
            raise SemanticTextError("semantic rows must use contiguous source order")
        description = normalize_description(row.get("description"))
        if row.get("description") != description:
            raise SemanticTextError("semantic row description is not canonically normalized")
        expected_text_sha = hashlib.sha256(description.encode("utf-8")).hexdigest()
        if row.get("description_utf8_sha256") != expected_text_sha:
            raise SemanticTextError("semantic description UTF-8 hash mismatch")
        if row.get("vector_sha256") != _array_sha256(embeddings[index]):
            raise SemanticTextError("semantic row vector hash mismatch")
        descriptions.append(description)
    expected_order = sorted(descriptions, key=str.encode)
    if descriptions != expected_order or len(descriptions) != len(set(descriptions)):
        raise SemanticTextError("semantic descriptions must be unique and UTF-8 sorted")
    return SemanticEmbeddingTable(
        artifact_directory=root,
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        embeddings_path=embeddings_path,
        embeddings_file_sha256=file_sha,
        embeddings_array_sha256=array_sha,
        descriptor=descriptor,
        descriptions=tuple(descriptions),
        embeddings=np.ascontiguousarray(embeddings),
    )


class TransformersClipTextBackend:
    """Frozen CLIP text projection loaded exclusively from a local snapshot."""

    def __init__(
        self,
        snapshot_directory: Path | str,
        *,
        model_id: str,
        model_revision: str,
        device: str = "cpu",
    ) -> None:
        snapshot = Path(snapshot_directory).resolve()
        if not snapshot.is_dir():
            raise FileNotFoundError(f"CLIP snapshot directory does not exist: {snapshot}")
        try:
            import torch
            import transformers
            from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast
        except ImportError as error:  # pragma: no cover - optional dependency boundary
            raise RuntimeError(
                "TransformersClipTextBackend requires the project's ml dependencies"
            ) from error
        self._torch = torch
        self._tokenizer = CLIPTokenizerFast.from_pretrained(snapshot, local_files_only=True)
        self._model = CLIPTextModelWithProjection.from_pretrained(
            snapshot,
            local_files_only=True,
        ).to(device)
        self._model.eval()
        self._model.requires_grad_(False)
        self._device = device
        files, tree_sha = hash_snapshot_tree(snapshot)
        projection_dim = int(self._model.config.projection_dim)
        maximum_tokens = int(self._model.config.max_position_embeddings)
        self._descriptor = SemanticEncoderDescriptor(
            backend="transformers_clip_text_with_projection",
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_id=model_id,
            tokenizer_revision=model_revision,
            library="transformers",
            library_version=transformers.__version__,
            embedding_dim=projection_dim,
            maximum_tokens=maximum_tokens,
            output_contract="CLIP_text_embeds_float32_L2_unit_norm_v1",
            snapshot_tree_sha256=tree_sha,
            files=files,
        )

    @property
    def descriptor(self) -> SemanticEncoderDescriptor:
        return self._descriptor

    def encode(self, descriptions: Sequence[str]) -> np.ndarray:
        normalized = tuple(normalize_description(value) for value in descriptions)
        if not normalized:
            raise ValueError("descriptions cannot be empty")
        tokens = self._tokenizer(
            list(normalized),
            padding=True,
            truncation=True,
            max_length=self.descriptor.maximum_tokens,
            return_tensors="pt",
        )
        tokens = {name: value.to(self._device) for name, value in tokens.items()}
        with self._torch.inference_mode():
            vectors = self._model(**tokens).text_embeds.to(dtype=self._torch.float32)
            vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return np.ascontiguousarray(vectors.cpu().numpy(), dtype=np.float32)


def hash_snapshot_tree(directory: Path | str) -> tuple[tuple[SemanticEncoderFile, ...], str]:
    """Hash every regular file in a local model snapshot and its canonical tree."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    evidence: list[SemanticEncoderFile] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix().encode(),
    ):
        relative = path.relative_to(root).as_posix()
        payload_sha = _file_sha256(path)
        evidence.append(SemanticEncoderFile(relative, payload_sha, path.stat().st_size))
    if not evidence:
        raise SemanticTextError("encoder snapshot contains no files")
    tree_payload = _canonical_json_bytes([asdict(item) for item in evidence])
    return tuple(evidence), hashlib.sha256(tree_payload).hexdigest()


def _descriptor_record(value: SemanticEncoderDescriptor) -> dict[str, object]:
    record = asdict(value)
    record["files"] = [asdict(item) for item in value.files]
    return record


def _descriptor_from_record(value: object) -> SemanticEncoderDescriptor:
    if not isinstance(value, Mapping):
        raise SemanticTextError("encoder must be an object")
    expected = {
        "backend",
        "model_id",
        "model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "library",
        "library_version",
        "embedding_dim",
        "maximum_tokens",
        "output_contract",
        "snapshot_tree_sha256",
        "files",
    }
    if set(value) != expected:
        raise SemanticTextError("encoder fields do not match schema")
    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        raise SemanticTextError("encoder.files must be a list")
    try:
        files = tuple(SemanticEncoderFile(**dict(item)) for item in raw_files)
        kwargs = {name: value[name] for name in expected.difference({"files"})}
        return SemanticEncoderDescriptor(files=files, **kwargs)
    except (KeyError, TypeError, ValueError) as error:
        raise SemanticTextError(f"invalid encoder descriptor: {error}") from error


def _validate_vectors(value: np.ndarray, *, row_count: int, embedding_dim: int) -> None:
    if not isinstance(value, np.ndarray):
        raise SemanticTextError("semantic backend must return a NumPy array")
    if value.dtype != np.float32:
        raise SemanticTextError(f"semantic vectors must use float32; got {value.dtype}")
    if value.shape != (row_count, embedding_dim):
        raise SemanticTextError(
            f"semantic vectors must have shape {(row_count, embedding_dim)}; got {value.shape}"
        )
    if not bool(np.isfinite(value).all()):
        raise SemanticTextError("semantic vectors contain non-finite values")
    norms = np.linalg.norm(value.astype(np.float64), axis=1)
    if not bool(np.all(np.abs(norms - 1.0) <= 1e-5)):
        raise SemanticTextError("semantic vectors must be L2 unit normalized")


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _required_digest(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise SemanticTextError(f"{name} must be a SHA-256 string")
    try:
        _validate_digest(value, name)
    except ValueError as error:
        raise SemanticTextError(str(error)) from error
    return value


def _validate_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in _DIGEST_CHARS for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
