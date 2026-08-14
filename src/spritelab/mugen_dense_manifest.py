"""Leakage-safe dense training manifest for streamed M.U.G.E.N core actions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from spritelab.storage import DiskGuard

MugenQualityTier = Literal["broad", "dense"]


def build_mugen_dense_manifest(
    materialization_roots: tuple[Path | str, ...],
    quality_audit_path: Path | str,
    *,
    tier: MugenQualityTier = "dense",
) -> dict[str, Any]:
    """Select an audited tier and group all exact SFF/array leakage components."""

    if tier not in {"broad", "dense"}:
        raise ValueError("tier must be 'broad' or 'dense'")
    quality_path = Path(quality_audit_path).resolve()
    quality_payload = quality_path.read_bytes()
    quality = _object(json.loads(quality_payload), "quality audit")
    if quality.get("artifact_kind") != "mugen_streamed_core_quality_audit":
        raise ValueError("quality audit kind differs")
    quality_rows = quality.get("quality_rows")
    if not isinstance(quality_rows, list) or any(not isinstance(row, dict) for row in quality_rows):
        raise ValueError("quality rows are invalid")
    quality_by_variant = _unique(quality_rows, "variant_id", "quality audit")
    source_facts = quality.get("source_materializations")
    if not isinstance(source_facts, list):
        raise ValueError("quality source materializations are invalid")
    audited_sources = {
        str(Path(_text(row, "manifest_path")).resolve()): _text(row, "manifest_file_sha256")
        for row in source_facts
        if isinstance(row, dict)
    }

    characters = []
    source_rows = []
    seen_variants: set[str] = set()
    for source_index, value in enumerate(materialization_roots):
        root = Path(value).resolve()
        manifest_path = root / "materialization.json"
        payload = manifest_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if audited_sources.get(str(manifest_path)) != digest:
            raise ValueError(f"quality audit does not bind materialization: {manifest_path}")
        materialization = _object(json.loads(payload), "materialization")
        if materialization.get("projection_version") != 2:
            raise ValueError("dense manifest requires MUGEN projection version 2")
        rows = materialization.get("characters")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("materialization characters are invalid")
        source_rows.append(
            {
                "manifest_file_sha256": digest,
                "manifest_path": str(manifest_path),
                "root": str(root),
                "source_index": source_index,
            }
        )
        for row in rows:
            variant_id = _text(row, "variant_id")
            if variant_id in seen_variants:
                raise ValueError(f"variant occurs in multiple materializations: {variant_id}")
            seen_variants.add(variant_id)
            audit = quality_by_variant.get(variant_id)
            if audit is None:
                raise ValueError(f"quality audit omits variant: {variant_id}")
            eligible = bool(audit[f"{tier}_eligible"])
            if not eligible:
                continue
            clips = row.get("clips")
            if not isinstance(clips, list) or any(not isinstance(clip, dict) for clip in clips):
                raise ValueError(f"variant clips are invalid: {variant_id}")
            by_slot = {_text(clip, "slot"): clip for clip in clips}
            if "idle" not in by_slot:
                continue
            characters.append((source_index, row, audit, by_slot))
    missing_quality = set(quality_by_variant) - seen_variants
    if missing_quality:
        raise ValueError(f"quality audit has unknown variants: {len(missing_quality)}")

    components = _leakage_components(characters)
    split_by_variant = {}
    component_rows = []
    for component in components:
        token_digest = hashlib.sha256(_canonical(component["tokens"]).rstrip(b"\n")).hexdigest()
        split = _split(token_digest)
        for variant_id in component["variant_ids"]:
            split_by_variant[variant_id] = split
        component_rows.append(
            {
                "component_sha256": token_digest,
                "split": split,
                "tokens": component["tokens"],
                "variant_ids": component["variant_ids"],
            }
        )

    records = []
    slot_counts: Counter[str] = Counter()
    for source_index, character, audit, by_slot in characters:
        variant_id = _text(character, "variant_id")
        definitions = character.get("definitions")
        if not isinstance(definitions, list):
            raise ValueError(f"definitions are invalid: {variant_id}")
        label = _identity_label(definitions, variant_id)
        actions = []
        for slot, clip in sorted(by_slot.items()):
            array = _object(clip.get("array"), "clip array")
            actions.append(
                {
                    "action_number": int(clip["action_number"]),
                    "array": array,
                    "loop_mode": _text(clip, "loop_mode"),
                    "record_id": _text(clip, "record_id"),
                    "schema_phase": clip.get("schema_phase"),
                    "schema_verb": clip.get("schema_verb"),
                    "slot": slot,
                    "source_action_index": int(clip["source_action_index"]),
                    "source_frame_count": int(
                        _object(clip.get("temporal_selection"), "temporal selection")[
                            "source_frame_count"
                        ]
                    ),
                    "temporal_selection": _object(
                        clip.get("temporal_selection"), "temporal selection"
                    ),
                }
            )
            slot_counts[slot] += 1
        idle = by_slot["idle"]
        idle_quality = next(row for row in audit["clip_metrics"] if row.get("slot") == "idle")
        reference_frame_index = int(idle_quality["medoid_frame_index"])
        if not 0 <= reference_frame_index < 8:
            raise ValueError(f"idle medoid index is invalid: {variant_id}")
        source = _object(character.get("source"), "character source")
        sff = _object(source.get("sff"), "character source SFF")
        records.append(
            {
                "actions": actions,
                "definitions": definitions,
                "identity": {
                    "description": None,
                    "label": label,
                    "text_source": "mugen_def_identity_only_uncaptioned",
                },
                "identity_id": _text(character, "identity_id"),
                "quality": {
                    "dense_exclusion_reasons": audit["dense_exclusion_reasons"],
                    "distinct_slot_arrays": int(audit["distinct_slot_arrays"]),
                    "dynamic_slots": int(audit["dynamic_slots"]),
                    "view_scale": float(audit["view_scale"]),
                },
                "reference": {
                    "array": _object(idle.get("array"), "idle array"),
                    "frame_index": reference_frame_index,
                    "frame_array_content_sha256": idle_quality["medoid_frame_array_content_sha256"],
                    "selection_method": "premultiplied_rgba_temporal_medoid_v1",
                    "slot": "idle",
                },
                "sff_sha256": _text(sff, "sha256"),
                "source": source,
                "source_index": source_index,
                "split": split_by_variant[variant_id],
                "variant_id": variant_id,
            }
        )
    records.sort(key=lambda row: row["variant_id"].encode("utf-8"))
    evaluation_probes = {
        split: _evaluation_probe(records, split=split, maximum_characters=32)
        for split in ("train", "validation", "test")
    }
    return {
        "artifact_kind": "mugen_dense_reference_motion_training_manifest",
        "components": component_rows,
        "counts": {
            "actions": sum(len(row["actions"]) for row in records),
            "characters": len(records),
            "components": len(component_rows),
            "slots": dict(sorted(slot_counts.items())),
            "splits": dict(sorted(Counter(row["split"] for row in records).items())),
            "unique_sff_identities": len({row["sff_sha256"] for row in records}),
        },
        "quality_audit": {
            "file_sha256": hashlib.sha256(quality_payload).hexdigest(),
            "path": str(quality_path),
            "policy": quality.get("policy"),
            "selected_tier": tier,
        },
        "records": records,
        "schema_version": 1,
        "source_materializations": source_rows,
        "evaluation_probes": evaluation_probes,
        "evaluation_policy": {
            "test": "identity-component-held-out generalization diagnostic",
            "train": "exact-training-member in-distribution reconstruction and steering",
            "validation": "identity-component-held-out model selection diagnostic",
        },
        "split_policy": {
            "grouping": (
                "transitive normalized DEF identity labels plus exact full-SFF, "
                "action-array, and nonempty-frame SHA-256 components"
            ),
            "identity_label_normalization": "Unicode NFKC, casefold, alphanumeric tokens",
            "test": "stable component digest buckets 950-999 of 1000",
            "train": "stable component digest buckets 0-899 of 1000",
            "validation": "stable component digest buckets 900-949 of 1000",
        },
    }


def _evaluation_probe(
    records: list[dict[str, Any]], *, split: str, maximum_characters: int
) -> list[dict[str, Any]]:
    candidates = [record for record in records if record["split"] == split]
    ordered = sorted(
        candidates,
        key=lambda record: (
            hashlib.sha256(f"mugen_dense_probe_v1\0{record['variant_id']}".encode()).digest(),
            record["variant_id"].encode(),
        ),
    )[:maximum_characters]
    return [
        {
            "actions": {
                action["slot"]: action["record_id"]
                for action in sorted(record["actions"], key=lambda row: row["slot"].encode())
            },
            "identity_id": record["identity_id"],
            "variant_id": record["variant_id"],
        }
        for record in ordered
    ]


def export_mugen_dense_manifest(
    materialization_roots: tuple[Path | str, ...],
    quality_audit_path: Path | str,
    output_path: Path | str,
    *,
    tier: MugenQualityTier = "dense",
    disk_guard: DiskGuard | None = None,
) -> str:
    """Publish a canonical no-clobber dense training manifest."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace dense manifest: {output}")
    artifact = build_mugen_dense_manifest(materialization_roots, quality_audit_path, tier=tier)
    payload = _canonical(artifact)
    guard = disk_guard or DiskGuard(output.anchor, 100 * 1024**3)
    guard.require_capacity(len(payload), label="MUGEN dense training manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace temporary dense manifest: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return hashlib.sha256(payload).hexdigest()


def _leakage_components(
    characters: list[tuple[int, dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]],
) -> list[dict[str, list[str]]]:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root.encode("utf-8") > right_root.encode("utf-8"):
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    tokens_by_variant = {}
    for _, character, audit, by_slot in characters:
        variant_id = _text(character, "variant_id")
        source = _object(character.get("source"), "character source")
        sff = _object(source.get("sff"), "character source SFF")
        tokens = {"sff:" + _sha256_text(sff, "sha256")}
        definitions = character.get("definitions")
        if not isinstance(definitions, list):
            raise ValueError(f"definitions are invalid: {variant_id}")
        identity_labels = {
            normalized
            for definition in definitions
            if isinstance(definition, dict)
            for key in ("display_name", "name")
            if isinstance(definition.get(key), str)
            if (normalized := _normalized_identity_label(definition[key]))
        }
        tokens.update("identity_label:" + value for value in identity_labels)
        for clip in by_slot.values():
            array = _object(clip.get("array"), "clip array")
            tokens.add("array:" + _sha256_text(array, "array_content_sha256"))
        metrics = audit.get("clip_metrics")
        if not isinstance(metrics, list):
            raise ValueError(f"quality clip metrics are invalid: {variant_id}")
        for metric in metrics:
            if not isinstance(metric, dict):
                raise ValueError(f"quality clip metric is invalid: {variant_id}")
            frame_hashes = metric.get("frame_array_content_sha256")
            frame_visible = metric.get("frame_visible_pixels")
            if (
                not isinstance(frame_hashes, list)
                or len(frame_hashes) != 8
                or not isinstance(frame_visible, list)
                or len(frame_visible) != 8
            ):
                raise ValueError(f"quality frame hashes are invalid: {variant_id}")
            tokens.update(
                "frame:" + _sha256_value(value, "frame_array_content_sha256")
                for value, visible in zip(frame_hashes, frame_visible, strict=True)
                if int(visible) > 0
            )
        ordered = sorted(tokens, key=str.encode)
        tokens_by_variant[variant_id] = ordered
        for token in ordered[1:]:
            union(ordered[0], token)
    variants_by_root: defaultdict[str, list[str]] = defaultdict(list)
    tokens_by_root: defaultdict[str, set[str]] = defaultdict(set)
    for variant_id, tokens in tokens_by_variant.items():
        root = find(tokens[0])
        variants_by_root[root].append(variant_id)
        tokens_by_root[root].update(tokens)
    return [
        {
            "tokens": sorted(tokens_by_root[root], key=str.encode),
            "variant_ids": sorted(variants, key=str.encode),
        }
        for root, variants in sorted(variants_by_root.items())
    ]


def _split(digest: str) -> str:
    bucket = int(digest[:16], 16) % 1000
    return "train" if bucket < 900 else "validation" if bucket < 950 else "test"


def _identity_label(definitions: list[Any], variant_id: str) -> str:
    for definition in definitions:
        if not isinstance(definition, dict):
            continue
        for key in ("display_name", "name"):
            value = definition.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return variant_id


def _normalized_identity_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


def _unique(rows: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        value = _text(row, key)
        if value in output:
            raise ValueError(f"{label} duplicates {key}: {value}")
        output[value] = row
    return output


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be non-empty text")
    return result


def _sha256_text(value: dict[str, Any], key: str) -> str:
    return _sha256_value(_text(value, key), key)


def _sha256_value(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
