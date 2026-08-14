"""Complete character-by-verb rectangles for MUGEN motion training."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard


@dataclass(frozen=True, slots=True)
class MugenBalancedCorpusConfig:
    """One explicit complete identity-by-verb training rectangle."""

    name: str
    verbs: tuple[str, ...]
    deduplicate_exact_identity_references: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty text")
        if (
            not self.verbs
            or len(set(self.verbs)) != len(self.verbs)
            or any(not isinstance(verb, str) or not verb for verb in self.verbs)
        ):
            raise ValueError("verbs must be unique non-empty text")
        if not isinstance(self.deduplicate_exact_identity_references, bool):
            raise TypeError("deduplicate_exact_identity_references must be a boolean")


def build_mugen_verb_coverage_report(canonical_manifest_path: Path | str) -> dict[str, Any]:
    """Measure every exact shared-verb rectangle in a canonical MUGEN manifest."""

    path = Path(canonical_manifest_path).resolve()
    payload = path.read_bytes()
    manifest, records = _canonical_manifest(payload)
    by_identity, labels, splits = _identity_matrix(records)
    reference_hashes = _identity_reference_hashes(records)
    verbs = sorted({verb for values in by_identity.values() for verb in values}, key=str.encode)
    pareto = []
    for verb_count in range(1, len(verbs) + 1):
        maximum = 0
        winners: list[tuple[str, ...]] = []
        for combination in itertools.combinations(verbs, verb_count):
            required = set(combination)
            eligible = [
                identity_id
                for identity_id, values in by_identity.items()
                if required.issubset(values)
            ]
            count = len({reference_hashes[identity_id] for identity_id in eligible})
            if count > maximum:
                maximum = count
                winners = [combination]
            elif count == maximum:
                winners.append(combination)
        if maximum:
            pareto.append(
                {
                    "best_verb_sets": [list(value) for value in winners],
                    "cell_count": maximum * verb_count,
                    "identity_count": maximum,
                    "verb_count": verb_count,
                }
            )
    identity_rows = []
    for identity_id in sorted(by_identity, key=str.encode):
        identity_rows.append(
            {
                "identity_id": identity_id,
                "identity_label": labels[identity_id],
                "identity_reference_array_sha256": reference_hashes[identity_id],
                "split": splits[identity_id],
                "verbs": sorted(by_identity[identity_id], key=str.encode),
            }
        )
    return {
        "artifact_kind": "mugen_complete_identity_verb_coverage",
        "counts": {
            "exact_reference_identity_groups": len(set(reference_hashes.values())),
            "pack_identities": len(by_identity),
            "sequences": len(records),
            "verbs": len(verbs),
        },
        "identity_rows": identity_rows,
        "pareto_maxima": pareto,
        "schema_version": 1,
        "source": {
            "canonical_manifest_file_sha256": hashlib.sha256(payload).hexdigest(),
            "canonical_manifest_path": str(path),
            "canonical_manifest_schema_version": manifest.get("schema_version"),
        },
        "verb_identity_coverage": {
            verb: len(
                {
                    reference_hashes[identity_id]
                    for identity_id, values in by_identity.items()
                    if verb in values
                }
            )
            for verb in verbs
        },
        "verb_pack_identity_coverage": {
            verb: sum(verb in values for values in by_identity.values()) for verb in verbs
        },
    }


def build_mugen_balanced_corpus_manifest(
    canonical_manifest_path: Path | str,
    materialization_path: Path | str,
    *,
    config: MugenBalancedCorpusConfig,
) -> dict[str, Any]:
    """Select every identity having every requested verb, one exact clip per cell."""

    canonical_path = Path(canonical_manifest_path).resolve()
    materialized_path = Path(materialization_path).resolve()
    canonical_payload = canonical_path.read_bytes()
    materialized_payload = materialized_path.read_bytes()
    parent, records = _canonical_manifest(canonical_payload)
    materialization = _object(materialized_payload, "materialization")
    sequences = materialization.get("sequences")
    if (
        materialization.get("schema_version") != 1
        or not isinstance(sequences, list)
        or materialization.get("sequence_count") != len(sequences)
        or any(not isinstance(sequence, dict) for sequence in sequences)
    ):
        raise ValueError("materialization sequence count differs")
    materialized_by_id = _unique(sequences, "sequence_id", "materialization")
    by_identity, _, _ = _identity_matrix(records)
    available_verbs = {verb for values in by_identity.values() for verb in values}
    missing = set(config.verbs) - available_verbs
    if missing:
        raise ValueError(f"requested verbs are absent: {sorted(missing)!r}")
    required = set(config.verbs)
    eligible_identities = {
        identity_id for identity_id, values in by_identity.items() if required.issubset(values)
    }
    if not eligible_identities:
        raise ValueError("requested verb rectangle has no complete identities")
    reference_hashes = _identity_reference_hashes(records)
    selected_identities = set(eligible_identities)
    duplicate_identity_exclusions = []
    if config.deduplicate_exact_identity_references:
        by_reference: defaultdict[str, list[str]] = defaultdict(list)
        for identity_id in eligible_identities:
            by_reference[reference_hashes[identity_id]].append(identity_id)
        selected_identities = set()
        for reference_hash, identity_ids in sorted(by_reference.items()):
            ordered = sorted(
                identity_ids,
                key=lambda identity_id: (
                    _split_rank(_identity_split(records, identity_id)),
                    identity_id.encode(),
                ),
            )
            selected = ordered[0]
            selected_identities.add(selected)
            for excluded in ordered[1:]:
                duplicate_identity_exclusions.append(
                    {
                        "excluded_identity_id": excluded,
                        "excluded_split": _identity_split(records, excluded),
                        "identity_reference_array_sha256": reference_hash,
                        "selected_identity_id": selected,
                        "selected_split": _identity_split(records, selected),
                    }
                )
    selected = []
    cells: set[tuple[str, str]] = set()
    for record in records:
        identity_id = _text(record, "identity_id")
        verb = _record_verb(record)
        if identity_id not in selected_identities or verb not in required:
            continue
        cell = (identity_id, verb)
        if cell in cells:
            raise ValueError(f"canonical manifest duplicates identity/verb: {cell!r}")
        cells.add(cell)
        sequence_id = _text(record, "sequence_id")
        source_sequence = materialized_by_id.get(sequence_id)
        if source_sequence is None:
            raise ValueError(f"materialization omits selected sequence: {sequence_id}")
        if source_sequence.get("identity_id") != identity_id:
            raise ValueError(f"materialization identity differs: {sequence_id}")
        caption = _dict(source_sequence, "caption")
        provenance = _dict(source_sequence, "provenance")
        enriched = dict(record)
        enriched["identity"] = {
            "description": caption.get("description"),
            "label": _text(caption, "identity_label"),
        }
        enriched["source_evidence"] = {
            "air_member": provenance.get("air_member"),
            "archive_sha256": provenance.get("archive_sha256"),
            "sff_member": provenance.get("sff_member"),
            "sff_sha256": provenance.get("sff_sha256"),
            "source_action_index": provenance.get("source_action_index"),
            "source_action_number": provenance.get("source_action_number"),
            "source_id": provenance.get("source_id"),
            "source_meaning": provenance.get("source_meaning"),
        }
        selected.append(enriched)
    expected_cells = len(selected_identities) * len(config.verbs)
    if len(selected) != expected_cells or len(cells) != expected_cells:
        raise ValueError("selected corpus is not a complete identity/verb rectangle")
    selected.sort(
        key=lambda row: (
            _text(row, "identity_id").encode(),
            _record_verb(row).encode(),
            _text(row, "sequence_id").encode(),
        )
    )
    split_counts: Counter[str] = Counter()
    verb_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    identities_by_split: defaultdict[str, set[str]] = defaultdict(set)
    for record in selected:
        split = _text(record, "split")
        identity_id = _text(record, "identity_id")
        split_counts[split] += 1
        verb_counts[_record_verb(record)] += 1
        class_counts[_text(record, "entity_class")] += 1
        identities_by_split[split].add(identity_id)
    expected_per_verb = len(selected_identities)
    if any(verb_counts[verb] != expected_per_verb for verb in config.verbs):
        raise ValueError("selected verb columns are imbalanced")
    parent_source = _dict(parent, "source")
    output_config = dict(_dict(parent, "config"))
    output_config.update(
        {
            "complete_identity_verb_rectangle": True,
            "name": config.name,
            "verbs": list(config.verbs),
        }
    )
    parent_hash = hashlib.sha256(canonical_payload).hexdigest()
    corpus_id = (
        "mugen_balanced_"
        + hashlib.sha256(
            (
                parent_hash
                + "\0"
                + "\0".join(config.verbs)
                + "\0deduplicate_exact_identity_references="
                + str(config.deduplicate_exact_identity_references)
            ).encode()
        ).hexdigest()[:32]
    )
    return {
        "artifact_kind": "mugen_reference_conditioned_primary_motion_training_manifest",
        "balanced_corpus": {
            "cell_count": expected_cells,
            "complete_rectangle": True,
            "corpus_id": corpus_id,
            "duplicate_identity_exclusions": duplicate_identity_exclusions,
            "identity_count": len(selected_identities),
            "identity_ids": sorted(selected_identities, key=str.encode),
            "verb_count": len(config.verbs),
            "verbs": list(config.verbs),
        },
        "config": output_config,
        "counts": {
            "entity_classes": _sorted_counter(class_counts),
            "exclusions": {
                "identity_missing_one_or_more_required_verbs": len(by_identity)
                - len(eligible_identities),
                "pack_identity_exact_reference_duplicate": len(duplicate_identity_exclusions),
            },
            "identities": len(selected_identities),
            "sequences": len(selected),
            "split_identities": {
                split: len(values)
                for split, values in sorted(identities_by_split.items(), key=lambda item: item[0])
            },
            "splits": _sorted_counter(split_counts),
            "verbs": _sorted_counter(verb_counts),
        },
        "policy": {
            "action_balance": "exactly one source clip per identity and required verb",
            "admission": (
                "parent canonical all-frame subject pixel gate plus complete verb coverage"
            ),
            "identity_deduplication": (
                "one pack identity per exact canonical reference RGBA hash; "
                "training split preferred, then validation, then test, then stable identity ID"
                if config.deduplicate_exact_identity_references
                else "disabled"
            ),
            "rights_scope": "Unknown/unverified fan uploads; no permissive inference",
            "split": "inherits identity-disjoint parent split without reassignment",
            "target_cardinality": "one representative per identity and required verb",
        },
        "records": selected,
        "schema_version": 2,
        "source": {
            **parent_source,
            "canonical_manifest_file_sha256": parent_hash,
            "canonical_manifest_path": str(canonical_path),
            "materialization_file_sha256": hashlib.sha256(materialized_payload).hexdigest(),
            "materialization_path": str(materialized_path),
        },
    }


def export_mugen_json_artifact(
    artifact: dict[str, Any],
    output_path: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Publish one canonical JSON artifact with no-clobber semantics."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace MUGEN artifact: {output}")
    payload = canonical_json_bytes(artifact)
    (disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)).require_capacity(
        len(payload) + 1024**2, label="balanced MUGEN corpus artifact"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return output, hashlib.sha256(payload).hexdigest()


def export_mugen_balanced_gallery(
    manifest_path: Path | str,
    output_directory: Path | str,
    *,
    identities_per_page: int = 24,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Render all logical frames for every balanced cell into PNG-only pages."""

    if identities_per_page <= 0:
        raise ValueError("identities_per_page must be positive")
    manifest_file = Path(manifest_path).resolve()
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace MUGEN gallery: {output}")
    payload = manifest_file.read_bytes()
    manifest, records = _canonical_manifest(payload)
    balanced = _dict(manifest, "balanced_corpus")
    verbs = balanced.get("verbs")
    if not isinstance(verbs, list) or not verbs or any(not isinstance(v, str) for v in verbs):
        raise ValueError("balanced gallery verb list is invalid")
    source = _dict(manifest, "source")
    materialization_path = Path(_text(source, "materialization_path")).resolve()
    if _file_sha256(materialization_path) != source.get("materialization_file_sha256"):
        raise ValueError("balanced gallery materialization hash differs")
    root = materialization_path.parent
    grouped: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    labels: dict[str, str] = {}
    for record in records:
        identity_id = _text(record, "identity_id")
        verb = _record_verb(record)
        grouped[identity_id][verb] = record
        identity = _dict(record, "identity")
        labels[identity_id] = _text(identity, "label")
    if any(set(columns) != set(verbs) for columns in grouped.values()):
        raise ValueError("balanced gallery input is not rectangular")
    (disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)).require_capacity(
        128 * 1024**2, label="balanced MUGEN gallery"
    )
    staging = output.with_name(output.name + f".tmp-{uuid.uuid4().hex}")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        font = _gallery_font(15)
        small_font = _gallery_font(12)
        identity_ids = sorted(grouped, key=lambda value: (labels[value].casefold(), value))
        pages = []
        for page_index, start in enumerate(range(0, len(identity_ids), identities_per_page), 1):
            page_ids = identity_ids[start : start + identities_per_page]
            page = _render_gallery_page(
                page_ids,
                grouped,
                labels,
                verbs,
                root,
                font=font,
                small_font=small_font,
            )
            name = f"page-{page_index:02d}.png"
            path = staging / name
            page.save(path, format="PNG", optimize=False)
            pages.append(
                {
                    "file_sha256": _file_sha256(path),
                    "identities": page_ids,
                    "name": name,
                    "size_bytes": path.stat().st_size,
                }
            )
        index = {
            "artifact_kind": "mugen_balanced_corpus_gallery",
            "display_contract": {
                "background": "checkerboard display derivative; source alpha retained",
                "frame_order": "all eight logical frames left-to-right",
                "resampling": "positive-integer nearest-neighbor downscale for display only",
                "source_arrays_are_canonical": True,
            },
            "identity_count": len(identity_ids),
            "manifest_file_sha256": hashlib.sha256(payload).hexdigest(),
            "manifest_path": str(manifest_file),
            "pages": pages,
            "schema_version": 1,
            "verbs": verbs,
        }
        index_payload = canonical_json_bytes(index)
        index_path = staging / "index.json"
        with index_path.open("xb") as handle:
            handle.write(index_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, output)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return output / "index.json", hashlib.sha256(index_payload).hexdigest()


def _render_gallery_page(
    identity_ids: list[str],
    grouped: dict[str, dict[str, dict[str, Any]]],
    labels: dict[str, str],
    verbs: list[str],
    root: Path,
    *,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    label_width = 220
    strip_width = 8 * 32
    column_width = strip_width + 18
    header_height = 42
    row_height = 46
    width = label_width + len(verbs) * column_width + 10
    height = header_height + len(identity_ids) * row_height + 8
    page = Image.new("RGB", (width, height), (18, 21, 27))
    draw = ImageDraw.Draw(page)
    draw.text((10, 12), "CHARACTER", fill=(205, 212, 224), font=small_font)
    for column, verb in enumerate(verbs):
        draw.text(
            (label_width + column * column_width + 8, 12),
            verb.upper(),
            fill=(122, 226, 179),
            font=small_font,
        )
    for row, identity_id in enumerate(identity_ids):
        y = header_height + row * row_height
        if row % 2:
            draw.rectangle((0, y, width, y + row_height), fill=(23, 27, 34))
        label = labels[identity_id]
        if len(label) > 27:
            label = label[:26] + "…"
        draw.text((10, y + 7), label, fill=(238, 240, 244), font=font)
        draw.text((10, y + 25), identity_id[-12:], fill=(113, 123, 139), font=small_font)
        for column, verb in enumerate(verbs):
            record = grouped[identity_id][verb]
            target = _dict(record, "target")
            pixels = _dict(target, "source_pixels")
            array = _load_rgba(root, pixels, _text(record, "sequence_id"))
            strip = _frame_strip(array)
            page.paste(strip, (label_width + column * column_width + 8, y + 7))
    return page


def _frame_strip(array: np.ndarray) -> Image.Image:
    strip = Image.new("RGB", (8 * 32, 32), (0, 0, 0))
    for index, frame in enumerate(array):
        rgba = Image.fromarray(frame, mode="RGBA")
        rgba = rgba.resize((32, 32), resample=Image.Resampling.NEAREST)
        background = Image.new("RGB", (32, 32), (151, 151, 151))
        checker = Image.new("RGB", (32, 32), (151, 151, 151))
        checker_draw = ImageDraw.Draw(checker)
        for y in range(0, 32, 8):
            for x in range(0, 32, 8):
                if (x // 8 + y // 8) % 2:
                    checker_draw.rectangle((x, y, x + 7, y + 7), fill=(116, 116, 116))
        background.paste(checker)
        background.paste(rgba, mask=rgba.getchannel("A"))
        strip.paste(background, (index * 32, 0))
    return strip


def _load_rgba(root: Path, record: dict[str, Any], sequence_id: str) -> np.ndarray:
    relative = _text(record, "relative_path")
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"gallery source path escapes root: {sequence_id}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != record.get("file_sha256"):
        raise ValueError(f"gallery source file hash differs: {sequence_id}")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise ValueError(f"gallery source geometry differs: {sequence_id}")
    if _array_sha256(value) != record.get("array_content_sha256"):
        raise ValueError(f"gallery source array hash differs: {sequence_id}")
    return np.ascontiguousarray(value)


def _canonical_manifest(payload: bytes) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _object(payload, "canonical motion manifest")
    if manifest.get("artifact_kind") != (
        "mugen_reference_conditioned_primary_motion_training_manifest"
    ):
        raise ValueError("canonical manifest has the wrong artifact kind")
    config = manifest.get("config")
    counts = manifest.get("counts")
    records = manifest.get("records")
    if (
        not isinstance(config, dict)
        or config.get("one_sequence_per_identity_verb") is not True
        or not isinstance(counts, dict)
        or not isinstance(records, list)
        or counts.get("sequences") != len(records)
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ValueError("canonical manifest is not one-row-per-identity/verb data")
    _unique(records, "sequence_id", "canonical manifest")
    return manifest, records


def _identity_matrix(
    records: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, str], dict[str, str]]:
    by_identity: defaultdict[str, set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    splits: dict[str, str] = {}
    for record in records:
        identity_id = _text(record, "identity_id")
        verb = _record_verb(record)
        if verb in by_identity[identity_id]:
            raise ValueError(f"canonical manifest duplicates identity/verb: {identity_id}/{verb}")
        by_identity[identity_id].add(verb)
        split = _text(record, "split")
        prior = splits.setdefault(identity_id, split)
        if prior != split:
            raise ValueError(f"identity crosses splits: {identity_id}")
        identity = record.get("identity")
        labels[identity_id] = (
            _text(identity, "label")
            if isinstance(identity, dict) and isinstance(identity.get("label"), str)
            else identity_id
        )
    return dict(by_identity), labels, splits


def _identity_reference_hashes(records: list[dict[str, Any]]) -> dict[str, str]:
    output: dict[str, str] = {}
    for record in records:
        identity_id = _text(record, "identity_id")
        reference = _dict(record, "reference")
        digest = _text(reference, "identity_reference_array_sha256")
        prior = output.setdefault(identity_id, digest)
        if prior != digest:
            raise ValueError(f"identity has inconsistent canonical references: {identity_id}")
    return output


def _identity_split(records: list[dict[str, Any]], identity_id: str) -> str:
    values = {
        _text(record, "split") for record in records if record.get("identity_id") == identity_id
    }
    if len(values) != 1:
        raise ValueError(f"identity split is ambiguous: {identity_id}")
    return next(iter(values))


def _split_rank(split: str) -> int:
    ranks = {"train": 0, "validation": 1, "test": 2}
    if split not in ranks:
        raise ValueError(f"unsupported identity split: {split}")
    return ranks[split]


def _record_verb(record: dict[str, Any]) -> str:
    return _text(_dict(record, "conditioning"), "verb")


def _unique(records: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        value = _text(record, key)
        if value in output:
            raise ValueError(f"{label} duplicates {key}: {value}")
        output[value] = record
    return output


def _object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be non-empty text")
    return value


def _dict(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: item[0].encode()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


def _gallery_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/consola.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()
