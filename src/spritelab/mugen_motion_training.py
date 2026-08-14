"""Conservative multi-identity MUGEN motion training manifest selection."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spritelab.spark_caption import canonical_json_bytes
from spritelab.storage import DiskGuard

PRIMARY_MOTION_VERBS = (
    "backstep",
    "block",
    "crouch",
    "dizzy",
    "get_up",
    "hurt",
    "idle",
    "jump",
    "normal_attack",
    "run",
    "turn",
    "walk",
)


@dataclass(frozen=True, slots=True)
class MugenMotionTrainingSelectionConfig:
    verbs: tuple[str, ...] = PRIMARY_MOTION_VERBS
    required_pixel_gate_status: str = "all_pass"
    one_sequence_per_identity_verb: bool = False

    def __post_init__(self) -> None:
        if (
            not self.verbs
            or len(set(self.verbs)) != len(self.verbs)
            or any(not isinstance(verb, str) or not verb for verb in self.verbs)
        ):
            raise ValueError("verbs must be unique non-empty text")
        if self.required_pixel_gate_status not in {"all_pass", "mixed", "all_fail"}:
            raise ValueError("required_pixel_gate_status is invalid")
        if not isinstance(self.one_sequence_per_identity_verb, bool):
            raise TypeError("one_sequence_per_identity_verb must be a boolean")


def build_mugen_motion_training_manifest(
    motion_plan_path: Path | str,
    pixel_audit_path: Path | str,
    *,
    config: MugenMotionTrainingSelectionConfig | None = None,
) -> dict[str, Any]:
    """Select identity-disjoint, exact-subject motion clips for broad training."""

    selection = config or MugenMotionTrainingSelectionConfig()
    plan_path = Path(motion_plan_path).resolve()
    audit_path = Path(pixel_audit_path).resolve()
    plan_bytes = plan_path.read_bytes()
    audit_bytes = audit_path.read_bytes()
    plan = _object(plan_bytes, "motion plan")
    audit = _object(audit_bytes, "pixel audit")
    if plan.get("artifact_kind") != "mugen_reference_conditioned_latent_motion_plan":
        raise ValueError("motion plan has the wrong artifact kind")
    if audit.get("artifact_kind") != "mugen_subject_bearing_frame_pixel_gate":
        raise ValueError("pixel audit has the wrong artifact kind")
    records = _counted_records(plan, "records", "sequences", "motion plan")
    pixel_records = _counted_records(audit, "records", "sequences", "pixel audit")
    pixel_by_sequence = _unique(pixel_records, "sequence_id", "pixel audit")
    if {record["sequence_id"] for record in records} != set(pixel_by_sequence):
        raise ValueError("pixel-audit sequence closure differs from motion plan")

    output = []
    exclusions: Counter[str] = Counter()
    split_by_identity: dict[str, str] = {}
    selected_verbs = set(selection.verbs)
    for record in sorted(records, key=lambda row: _text(row, "sequence_id").encode()):
        sequence_id = _text(record, "sequence_id")
        conditioning = _dict(record, "conditioning")
        verb = _text(conditioning, "verb")
        pixel = pixel_by_sequence[sequence_id]
        if verb not in selected_verbs:
            exclusions[f"verb:{verb}"] += 1
            continue
        if pixel.get("pixel_gate_status") != selection.required_pixel_gate_status:
            exclusions[f"pixel_gate:{pixel.get('pixel_gate_status')}"] += 1
            continue
        indices = pixel.get("pixel_gate_pass_indices")
        if selection.required_pixel_gate_status == "all_pass" and indices != list(range(8)):
            raise ValueError(f"all-pass pixel indices differ for {sequence_id}")
        split = _text(record, "split")
        identity_id = _text(record, "identity_id")
        prior_split = split_by_identity.setdefault(identity_id, split)
        if prior_split != split:
            raise ValueError(f"identity crosses splits: {identity_id}")
        entity_class = _text(record, "entity_class")
        frame_metrics = pixel.get("frames")
        if not isinstance(frame_metrics, list) or len(frame_metrics) != 8:
            raise ValueError(f"pixel frame metrics differ for {sequence_id}")
        output.append(
            {
                "conditioning": conditioning,
                "eligibility": {
                    "method": "canonical_reference_pixel_gate_all_frames",
                    "pixel_gate_pass_indices": indices,
                    "pixel_gate_status": pixel["pixel_gate_status"],
                    "representative_quality": _representative_quality(frame_metrics),
                },
                "entity_class": entity_class,
                "identity_id": identity_id,
                "reference": _dict(record, "reference"),
                "reference_target_relation": _text(record, "reference_target_relation"),
                "sample_id": _text(record, "sample_id"),
                "sequence_id": sequence_id,
                "split": split,
                "target": _dict(record, "target"),
            }
        )
    if selection.one_sequence_per_identity_verb:
        grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for record in output:
            grouped[(record["identity_id"], record["conditioning"]["verb"])].append(record)
        representatives = []
        for candidates in grouped.values():
            ordered = sorted(candidates, key=_representative_key)
            representatives.append(ordered[0])
            exclusions["noncanonical_identity_verb_variant"] += len(ordered) - 1
        output = sorted(representatives, key=lambda row: row["sequence_id"].encode())
    if not output:
        raise ValueError("motion training selection is empty")
    split_counts: Counter[str] = Counter()
    verb_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    identities_by_split: defaultdict[str, set[str]] = defaultdict(set)
    for record in output:
        split_counts[record["split"]] += 1
        verb_counts[record["conditioning"]["verb"]] += 1
        class_counts[record["entity_class"]] += 1
        identities_by_split[record["split"]].add(record["identity_id"])
    missing_verbs = selected_verbs - set(verb_counts)
    if missing_verbs:
        raise ValueError(f"selected verbs have no records: {sorted(missing_verbs)!r}")
    return {
        "artifact_kind": "mugen_reference_conditioned_primary_motion_training_manifest",
        "config": asdict(selection),
        "counts": {
            "entity_classes": _sorted_counter(class_counts),
            "exclusions": _sorted_counter(exclusions),
            "identities": len(split_by_identity),
            "sequences": len(output),
            "split_identities": {
                split: len(values)
                for split, values in sorted(identities_by_split.items(), key=lambda item: item[0])
            },
            "splits": _sorted_counter(split_counts),
            "verbs": _sorted_counter(verb_counts),
        },
        "policy": {
            "action_balance": "hierarchical identity_then_verb_then_sequence at training time",
            "admission": "all eight frames pass exact canonical-reference pixel gate",
            "excluded_pending_role_vlm": [
                "intro",
                "special_attack",
                "super_attack",
                "victory",
            ],
            "scope": "primary-subject broad motion v1; effects may remain around visible subject",
            "target_cardinality": (
                "one representative per identity and verb"
                if selection.one_sequence_per_identity_verb
                else "all admitted source variants"
            ),
            "split": "inherits identity-disjoint motion plan split",
        },
        "records": output,
        "schema_version": 1,
        "source": {
            "motion_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "motion_plan_path": str(plan_path),
            "pixel_audit_file_sha256": hashlib.sha256(audit_bytes).hexdigest(),
            "pixel_audit_path": str(audit_path),
        },
    }


def export_mugen_motion_training_manifest(
    motion_plan_path: Path | str,
    pixel_audit_path: Path | str,
    output_path: Path | str,
    *,
    config: MugenMotionTrainingSelectionConfig | None = None,
    disk_guard: DiskGuard | None = None,
) -> tuple[Path, str]:
    """Publish one canonical no-clobber training manifest."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace motion training manifest: {output}")
    payload = canonical_json_bytes(
        build_mugen_motion_training_manifest(motion_plan_path, pixel_audit_path, config=config)
    )
    (disk_guard or DiskGuard(output.parent, min_free_bytes=100 * 1024**3)).require_capacity(
        len(payload) + 1024**2, label="MUGEN motion training manifest"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return output, hashlib.sha256(payload).hexdigest()


def _object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _counted_records(
    artifact: dict[str, Any], key: str, count_key: str, label: str
) -> list[dict[str, Any]]:
    records = artifact.get(key)
    counts = artifact.get("counts")
    if (
        not isinstance(records, list)
        or not isinstance(counts, dict)
        or counts.get(count_key) != len(records)
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ValueError(f"{label} record count differs")
    return records


def _unique(records: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    for record in records:
        value = _text(record, key)
        if value in output:
            raise ValueError(f"{label} has duplicate {key}: {value}")
        output[value] = record
    return output


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


def _representative_quality(frames: list[dict[str, Any]]) -> dict[str, float]:
    fields = (
        "anchored_overlap",
        "bbox_iou",
        "candidate_palette_coverage",
        "palette_histogram_intersection",
    )
    output = {}
    for field in fields:
        values = [frame.get(field) for frame in frames]
        if any(not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"pixel frame metric {field} is invalid")
        output[f"minimum_{field}"] = min(float(value) for value in values)
    occupancy = [frame.get("occupancy_ratio") for frame in frames]
    if any(not isinstance(value, (int, float)) for value in occupancy):
        raise ValueError("pixel frame metric occupancy_ratio is invalid")
    output["maximum_occupancy_deviation"] = max(abs(float(value) - 1.0) for value in occupancy)
    return output


def _representative_key(record: dict[str, Any]) -> tuple[float | bytes, ...]:
    quality = record["eligibility"]["representative_quality"]
    return (
        -quality["minimum_candidate_palette_coverage"],
        -quality["minimum_palette_histogram_intersection"],
        -quality["minimum_anchored_overlap"],
        -quality["minimum_bbox_iou"],
        quality["maximum_occupancy_deviation"],
        record["sequence_id"].encode(),
    )
