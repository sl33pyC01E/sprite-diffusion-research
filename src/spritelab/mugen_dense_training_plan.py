"""Captioned latent-still plan for the dense six-action M.U.G.E.N corpus."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from spritelab.storage import DiskGuard

_ACTION_PHRASES = {
    "attack_a": "performing standard attack A",
    "attack_b": "performing standard attack B",
    "block": "blocking in a defensive stance",
    "idle": "in an idle neutral stance",
    "jump": "jumping",
    "walk": "walking",
}


def build_mugen_dense_still_training_plan(
    captioned_materialization_path: Path | str,
) -> dict[str, Any]:
    """Build the exact prompt/target plan without changing or copying pixels."""

    path = Path(captioned_materialization_path).resolve()
    payload = path.read_bytes()
    materialization = _object(json.loads(payload), "captioned materialization")
    if materialization.get("artifact_kind") != "mugen_dense_captioned_materialization_bridge":
        raise ValueError("captioned materialization has the wrong artifact kind")
    eligibility = _object(materialization.get("model_eligibility"), "model eligibility")
    if eligibility.get("conditional_generation") is not True:
        raise ValueError("captioned materialization is not conditional-generation eligible")
    sequences = materialization.get("sequences")
    if (
        not isinstance(sequences, list)
        or materialization.get("sequence_count") != len(sequences)
        or any(not isinstance(row, dict) for row in sequences)
    ):
        raise ValueError("captioned materialization sequence count differs")
    output = []
    identities_by_split: dict[str, set[str]] = {}
    action_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    seen = set()
    for sequence in sequences:
        sequence_id = _text(sequence, "sequence_id")
        if sequence_id in seen:
            raise ValueError(f"duplicate sequence ID: {sequence_id}")
        seen.add(sequence_id)
        action = _text(sequence, "action")
        try:
            action_phrase = _ACTION_PHRASES[action]
        except KeyError as error:
            raise ValueError(f"unsupported dense action: {action}") from error
        split = _text(sequence, "split")
        identity_id = _text(sequence, "identity_id")
        identities_by_split.setdefault(split, set()).add(identity_id)
        caption = _object(sequence.get("caption"), "caption")
        appearance = _text(caption, "description").rstrip(" .;")
        prompt = f"{appearance}; {action_phrase}; side view"
        target = _object(sequence.get("output"), "output")
        for key in ("array_content_sha256", "file_sha256"):
            _digest(target, key)
        if target.get("shape") != [8, 128, 128, 4]:
            raise ValueError(f"target shape differs: {sequence_id}")
        output.append(
            {
                "conditioning": {
                    "action_phrase": action_phrase,
                    "direction": "unknown",
                    "verb": action,
                    "view": "side",
                },
                "entity_class": _text(sequence, "entity_class"),
                "identity_id": identity_id,
                "prompt": prompt,
                "sample_id": "still_sequence_"
                + hashlib.sha256(f"mugen_dense_still_v1\0{sequence_id}".encode()).hexdigest()[:32],
                "sequence_id": sequence_id,
                "split": split,
                "target": {
                    "array_content_sha256": target["array_content_sha256"],
                    "eligible_frame_indices": list(range(8)),
                    "file_sha256": target["file_sha256"],
                    "frame_count": 8,
                    "frame_sampling": "uniform_logical_frame_index",
                    "relative_path": _text(target, "relative_path"),
                    "shape": target["shape"],
                },
            }
        )
        action_counts[action] += 1
        prompt_counts[prompt] += 1
    output.sort(key=lambda row: row["sequence_id"].encode("utf-8"))
    return {
        "artifact_kind": "mugen_latent_still_sequence_training_plan",
        "counts": {
            "actions": dict(sorted(action_counts.items())),
            "identities": len({row["identity_id"] for row in output}),
            "prompts": len(prompt_counts),
            "sequences": len(output),
            "split_identities": {
                split: len(values) for split, values in sorted(identities_by_split.items())
            },
        },
        "records": output,
        "sampler_contract": {
            "frame": "uniform_logical_frame_within_selected_sequence",
            "hierarchy": ["identity", "action", "sequence", "frame"],
            "identity_component_split_disjoint": True,
            "raw_sequence_frequency_is_not_sampling_weight": True,
        },
        "schema_version": 3,
        "source": {
            "materialization_file_sha256": hashlib.sha256(payload).hexdigest(),
            "materialization_path": str(path),
        },
    }


def export_mugen_dense_still_training_plan(
    captioned_materialization_path: Path | str,
    output_path: Path | str,
    *,
    disk_guard: DiskGuard | None = None,
) -> str:
    """Publish the plan canonically and no-clobber."""

    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace dense still plan: {output}")
    artifact = build_mugen_dense_still_training_plan(captioned_materialization_path)
    payload = _canonical(artifact)
    (disk_guard or DiskGuard(Path(output.anchor), 100 * 1024**3)).require_capacity(
        len(payload), label="MUGEN dense still training plan"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to replace temporary dense still plan: {temporary}")
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


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
