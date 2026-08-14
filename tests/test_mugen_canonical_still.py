from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.mugen_canonical_still import (
    build_mugen_canonical_still_training_plan,
    export_mugen_canonical_still_training_plan,
)
from spritelab.mugen_stills import compose_caption_input
from spritelab.storage import DiskGuard


def test_canonical_plan_uses_one_exact_appearance_frame_per_identity(tmp_path: Path) -> None:
    sequence_plan, captions = _fixture(tmp_path)

    plan = build_mugen_canonical_still_training_plan(sequence_plan, captions)

    assert plan["artifact_kind"] == "mugen_canonical_appearance_still_training_plan"
    assert plan["counts"] == {
        "eligible_frames": 2,
        "identities": 2,
        "prompts": 2,
        "sequences": 2,
        "split_sequences": {"train": 1, "validation": 1},
    }
    assert [record["identity_id"] for record in plan["records"]] == [
        "identity_a",
        "identity_b",
    ]
    assert plan["records"][0]["conditioning"]["verb"] == "canonical_still"
    assert plan["records"][0]["conditioning"]["action_phrase"] is None
    assert plan["records"][0]["target"]["eligible_frame_indices"] == [2]
    assert "attacking" not in plan["records"][0]["prompt"]


def test_canonical_plan_detects_caption_input_pixel_drift(tmp_path: Path) -> None:
    sequence_plan, captions = _fixture(tmp_path)
    caption_manifest = json.loads(captions.read_text(encoding="utf-8"))
    relative = caption_manifest["records"][0]["caption_input"]["relative_path"]
    input_path = captions.parent / relative
    with Image.open(input_path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    rgb[0, 0] = (0, 0, 0)
    Image.fromarray(rgb, mode="RGB").save(input_path)
    caption_manifest["records"][0]["caption_input"]["file_sha256"] = _file_sha256(input_path)
    captions.write_bytes(_canonical_json(caption_manifest))

    with pytest.raises(ValueError, match="pixels differ"):
        build_mugen_canonical_still_training_plan(sequence_plan, captions)


def test_canonical_plan_export_is_hash_bound_and_no_clobber(tmp_path: Path) -> None:
    sequence_plan, captions = _fixture(tmp_path)
    output = tmp_path / "canonical-plan.json"

    path, sha256 = export_mugen_canonical_still_training_plan(
        sequence_plan,
        captions,
        output,
        disk_guard=DiskGuard(tmp_path, 0),
    )

    assert path == output.resolve()
    assert _file_sha256(path) == sha256
    with pytest.raises(FileExistsError):
        export_mugen_canonical_still_training_plan(
            sequence_plan,
            captions,
            output,
            disk_guard=DiskGuard(tmp_path, 0),
        )


def _fixture(root: Path) -> tuple[Path, Path]:
    materialized = root / "materialized"
    clips = materialized / "clips"
    clips.mkdir(parents=True)
    materialization = materialized / "materialization.json"
    materialization.write_bytes(b"{}\n")
    sequence_records = []
    caption_records = []
    caption_root = root / "captions"
    input_root = caption_root / "caption-inputs"
    input_root.mkdir(parents=True)
    for index, (identity, split, frame_index) in enumerate(
        (("identity_a", "train", 2), ("identity_b", "validation", 5))
    ):
        sequence_id = f"sequence_{index}"
        rgba = np.zeros((8, 128, 128, 4), dtype=np.uint8)
        rgba[frame_index, 32:96, 48:80] = (220, 40 + index * 30, 20, 255)
        clip_path = clips / f"{sequence_id}.npy"
        np.save(clip_path, rgba, allow_pickle=False)
        clip_file_sha256 = _file_sha256(clip_path)
        clip_array_sha256 = _array_sha256(rgba)
        reference = np.ascontiguousarray(rgba[frame_index])
        input_path = input_root / f"{identity}.png"
        Image.fromarray(compose_caption_input(reference), mode="RGB").resize(
            (512, 512), Image.Resampling.NEAREST
        ).save(input_path)
        sequence_records.append(
            {
                "entity_class": "humanoid",
                "identity_id": identity,
                "sequence_id": sequence_id,
                "split": split,
                "target": {
                    "array_content_sha256": clip_array_sha256,
                    "file_sha256": clip_file_sha256,
                    "relative_path": f"clips/{sequence_id}.npy",
                    "shape": [8, 128, 128, 4],
                },
            }
        )
        caption_records.append(
            {
                "caption_input": {
                    "file_sha256": _file_sha256(input_path),
                    "relative_path": f"caption-inputs/{identity}.png",
                },
                "entity_class": "humanoid",
                "frame_index": frame_index,
                "identity_id": identity,
                "reference_array_sha256": _array_sha256(reference),
                "request_body_sha256": hashlib.sha256(identity.encode()).hexdigest(),
                "sequence_id": sequence_id,
                "source_array_sha256": clip_array_sha256,
                "source_file_sha256": clip_file_sha256,
                "split": split,
                "structured_caption": {
                    "body_build": f"fighter build {index}",
                    "facing": "side",
                },
                "structured_verb": "idle",
                "training_prompt": (
                    f"2D sprite on a transparent background. humanoid fighter {index}. "
                    "full subject, centered, crisp hard edges."
                ),
            }
        )
    sequence_plan = root / "sequence-plan.json"
    sequence_plan.write_bytes(
        _canonical_json(
            {
                "counts": {"sequences": 2},
                "records": sequence_records,
                "source": {
                    "materialization_file_sha256": _file_sha256(materialization),
                    "materialization_path": str(materialization),
                },
            }
        )
    )
    captions = caption_root / "manifest.json"
    captions.write_bytes(
        _canonical_json(
            {
                "artifact_kind": "mugen_canonical_still_structured_caption_dataset",
                "caption_count": 2,
                "records": caption_records,
            }
        )
    )
    return sequence_plan, captions


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
