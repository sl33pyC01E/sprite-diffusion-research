from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.mugen_stills import compose_caption_input
from spritelab.qwen_still_dataset import export_qwen_image_lora_dataset
from spritelab.storage import DiskGuard


def test_qwen_export_is_exact_captioned_and_split_closed(tmp_path: Path) -> None:
    plan = _fixture(tmp_path)
    output = tmp_path / "qwen"

    manifest_path, digest = export_qwen_image_lora_dataset(
        plan, output, split="train", disk_guard=DiskGuard(tmp_path, 0)
    )

    assert _sha(manifest_path) == digest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"] == {"identities": 1, "images": 1, "prompts": 1}
    assert manifest["split"] == "train"
    assert manifest["training_contract"]["split_identity_disjoint_from"] == ["validation"]
    record = manifest["records"][0]
    assert "solid neutral gray preview canvas" in record["prompt"]
    assert "transparent background" in record["source_prompt"]
    metadata = [json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()]
    assert metadata == [{"file_name": record["image_relative_path"], "text": record["prompt"]}]
    assert _sha(output / record["image_relative_path"]) == record["image_file_sha256"]
    with pytest.raises(FileExistsError):
        export_qwen_image_lora_dataset(
            plan, output, split="train", disk_guard=DiskGuard(tmp_path, 0)
        )


def test_qwen_export_rejects_pixel_drift_and_leaves_evidence(tmp_path: Path) -> None:
    plan = _fixture(tmp_path)
    canonical = json.loads(plan.read_text(encoding="utf-8"))
    caption_path = Path(canonical["source"]["caption_manifest_path"])
    captions = json.loads(caption_path.read_text(encoding="utf-8"))
    input_path = caption_path.parent / captions["records"][0]["caption_input"]["relative_path"]
    with Image.open(input_path) as image:
        pixels = np.array(image.convert("RGB"), copy=True)
    pixels[0, 0] = (0, 0, 0)
    Image.fromarray(pixels, mode="RGB").save(input_path)
    changed = _sha(input_path)
    captions["records"][0]["caption_input"]["file_sha256"] = changed
    caption_path.write_bytes(_canonical(captions))
    canonical["source"]["caption_manifest_file_sha256"] = _sha(caption_path)
    canonical["records"][0]["caption_reference"]["caption_input_file_sha256"] = changed
    plan.write_bytes(_canonical(canonical))

    with pytest.raises(ValueError, match="pixels differ"):
        export_qwen_image_lora_dataset(
            plan, tmp_path / "failed-export", split="train", disk_guard=DiskGuard(tmp_path, 0)
        )

    assert not (tmp_path / "failed-export").exists()


def _fixture(root: Path) -> Path:
    materialized = root / "materialized"
    clips = materialized / "clips"
    clips.mkdir(parents=True)
    materialization = materialized / "materialization.json"
    materialization.write_bytes(b"{}\n")
    caption_root = root / "captions"
    inputs = caption_root / "inputs"
    inputs.mkdir(parents=True)
    plan_records = []
    caption_records = []
    for index, split in enumerate(("train", "validation")):
        identity = f"identity_{index}"
        sequence = f"sequence_{index}"
        sample = f"canonical_still_{index}"
        array = np.zeros((8, 128, 128, 4), dtype=np.uint8)
        array[index, 40:88, 52:76] = (220, 40 + index * 30, 20, 255)
        clip = clips / f"{sequence}.npy"
        np.save(clip, array, allow_pickle=False)
        reference = np.ascontiguousarray(array[index])
        caption_input = inputs / f"{identity}.png"
        Image.fromarray(compose_caption_input(reference), mode="RGB").resize(
            (512, 512), Image.Resampling.NEAREST
        ).save(caption_input)
        plan_records.append(
            {
                "caption_reference": {
                    "caption_input_file_sha256": _sha(caption_input),
                    "identity_reference_array_sha256": _array_sha(reference),
                },
                "entity_class": "humanoid",
                "identity_id": identity,
                "prompt": (
                    f"pixel art sprite; transparent background; humanoid fighter {index}; "
                    "full subject; centered; crisp hard edges"
                ),
                "sample_id": sample,
                "sequence_id": sequence,
                "split": split,
                "target": {
                    "array_content_sha256": _array_sha(array),
                    "eligible_frame_indices": [index],
                    "file_sha256": _sha(clip),
                    "relative_path": f"clips/{sequence}.npy",
                },
            }
        )
        caption_records.append(
            {
                "caption_input": {
                    "file_sha256": _sha(caption_input),
                    "relative_path": f"inputs/{identity}.png",
                },
                "frame_index": index,
                "identity_id": identity,
            }
        )
    captions = caption_root / "manifest.json"
    captions.write_bytes(
        _canonical(
            {
                "artifact_kind": "mugen_canonical_still_structured_caption_dataset",
                "caption_count": 2,
                "records": caption_records,
            }
        )
    )
    plan = root / "plan.json"
    plan.write_bytes(
        _canonical(
            {
                "artifact_kind": "mugen_canonical_appearance_still_training_plan",
                "counts": {
                    "sequences": 2,
                    "split_sequences": {"train": 1, "validation": 1},
                },
                "records": plan_records,
                "source": {
                    "caption_manifest_file_sha256": _sha(captions),
                    "caption_manifest_path": str(captions),
                    "materialization_file_sha256": _sha(materialization),
                    "materialization_path": str(materialization),
                },
            }
        )
    )
    return plan


def _array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
