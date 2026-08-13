"""Caption one canonical appearance reference per MFFA identity with pinned Florence-2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_stills import (  # noqa: E402
    MugenStillReference,
    compose_caption_input,
    detailed_training_prompt,
    filtered_appearance_caption,
    load_mugen_still_references,
)
from spritelab.storage import DiskGuard  # noqa: E402

MODEL_COMMIT = "1896e7c7f4a355c6c92be0dba8b1da35767eb75a"
MODEL_ID = "microsoft/Florence-2-base"
MODEL_DIR = ROOT / f"data/models/florence-2-base-{MODEL_COMMIT}"
MATERIALIZATION = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
TAXONOMY = ROOT / "data/index/reports/mugen-mffa-action-taxonomy-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-canonical-still-captions-v2"
TASK = "<MORE_DETAILED_CAPTION>"
EXPECTED_MODEL_FILES = {
    "LICENSE": "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383",
    "README.md": "e49de5527bd39688745ce20388bf3f31fdb891abc0e6950a3f64630114769d17",
    "config.json": "c666d0fe0172d46e115e8fba6cd93cd83714575b33a73005cab8d24ce2a3aa8f",
    "configuration_florence2.py": (
        "653bafddc9651eaff1583a16db4a2bb27d33ec7d541dfab7201aaa4ecaa1cfbf"
    ),
    "model.safetensors": "03075d2d2d2bbd3e180b9ba0afae4aa8563226e2d32911656966e05b2f2ee060",
    "modeling_florence2.py": "5bb7aa72c6ba62e96e1bbae6bc1aaf7b4e8e28cdfc62e670de3d5b67eeab1fdf",
    "preprocessor_config.json": "2f5921bbc53c7cc04251e1027b45b1cec726276be6db23d1bb40641bfbe2cf29",
    "processing_florence2.py": "4bd7158536cbf1c7891fc8efd94437d79fd09f07f539c7398fab8a885d7d8bca",
    "tokenizer.json": "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
    "tokenizer_config.json": "79ffcf43af8ebda99d165f61d243180da2e2639952e41e71e11611c18770489c",
    "vocab.json": "394fdc63c71aabe0a9b97117f5d62fb5fcc4d59b2b3ea929a3929e6a53217b3c",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_files = _verify_model_files()
    references = load_mugen_still_references(MATERIALIZATION, TAXONOMY)
    preflight = {
        "batch_size": args.batch_size,
        "device": args.device,
        "model_commit": MODEL_COMMIT,
        "model_file_count": len(model_files),
        "output": str(OUTPUT),
        "reference_count": len(references),
        "split_counts": {
            split: sum(reference.split == split for reference in references)
            for split in ("train", "validation", "test")
        },
    }
    if args.preflight_only:
        print(json.dumps(preflight, sort_keys=True))
        return
    DiskGuard(ROOT, min_free_bytes=100 * 1024**3).require_capacity(
        64 * 1024**2, label="MUGEN canonical still captions"
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    journal = OUTPUT / "caption-records.jsonl"
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"caption manifest already exists: {manifest_path}")
    completed = _load_journal(journal)
    pending = [reference for reference in references if reference.identity_id not in completed]
    if pending:
        runtime = _load_runtime(args.device)
        with journal.open("a", encoding="utf-8", newline="\n") as handle:
            for batch in _batches(pending, args.batch_size):
                records = _caption_batch(runtime, batch)
                for record in records:
                    handle.write(
                        json.dumps(
                            record, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                        )
                        + "\n"
                    )
                handle.flush()
                os.fsync(handle.fileno())
    completed = _load_journal(journal)
    expected_ids = {reference.identity_id for reference in references}
    if set(completed) != expected_ids:
        raise RuntimeError("caption journal is incomplete after generation")
    ordered = [completed[reference.identity_id] for reference in references]
    manifest = {
        "artifact_kind": "mugen_canonical_still_detailed_caption_dataset",
        "caption_contract": {
            "background_claim_filter": (
                "remove_period_sentences_containing_background_or_pixelated_appearance_v1"
            ),
            "caption_source": "model_generated_unverified",
            "composite_background_rgb": [127, 127, 127],
            "model_raw_caption_retained": True,
            "reference_selection": "neutral_verb_priority_then_sequence_id_medoid_frame_v1",
            "resize_for_captioner": "512x512_nearest_neighbor",
            "task": TASK,
        },
        "caption_count": len(ordered),
        "model": {
            "commit": MODEL_COMMIT,
            "files": model_files,
            "id": MODEL_ID,
            "local_path": str(MODEL_DIR),
            "remote_code_execution": "pinned_hash_verified_microsoft_python_only",
        },
        "records": ordered,
        "schema_version": 1,
        "source": {
            "materialization_file_sha256": _file_sha256(MATERIALIZATION),
            "materialization_path": str(MATERIALIZATION),
            "taxonomy_file_sha256": _file_sha256(TAXONOMY),
            "taxonomy_path": str(TAXONOMY),
        },
        "supersedes": {
            "file_sha256": "dbad188cbd0b92397cdf5381f4106b1427932aecf0b1bb60459e506b722c28dd",
            "path": str(
                ROOT / "data/processed/mugen-mffa-canonical-still-captions-v1/manifest.json"
            ),
            "reason": "literal_pad_tokens_and_intro_first_reference_priority",
        },
    }
    payload = _canonical_json(manifest)
    with manifest_path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print({"manifest": str(manifest_path), "sha256": hashlib.sha256(payload).hexdigest()})


def _caption_batch(
    runtime: dict[str, Any], batch: list[MugenStillReference]
) -> list[dict[str, Any]]:
    torch = runtime["torch"]
    processor = runtime["processor"]
    model = runtime["model"]
    device = runtime["device"]
    images = []
    composite_hashes = []
    for reference in batch:
        composite = compose_caption_input(reference.rgba)
        composite_hashes.append(_array_sha256(composite))
        images.append(Image.fromarray(composite).resize((512, 512), Image.Resampling.NEAREST))
    inputs = processor(text=[TASK] * len(batch), images=images, return_tensors="pt")
    inputs = {name: value.to(device=device) for name, value in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=128,
            num_beams=3,
            do_sample=False,
        )
    decoded_raw = processor.batch_decode(generated, skip_special_tokens=False)
    decoded_clean = processor.batch_decode(generated, skip_special_tokens=True)
    records = []
    for reference, composite_sha256, raw_tokens, clean_text in zip(
        batch, composite_hashes, decoded_raw, decoded_clean, strict=True
    ):
        processed = processor.post_process_generation(clean_text, task=TASK, image_size=(512, 512))
        raw_caption = processed[TASK]
        records.append(
            {
                "alpha_bbox_xywh": list(reference.alpha_bbox_xywh)
                if reference.alpha_bbox_xywh is not None
                else None,
                "caption_input_array_sha256": composite_sha256,
                "caption_model_raw": raw_caption,
                "caption_model_raw_tokens": raw_tokens,
                "caption_model_raw_filtered": filtered_appearance_caption(raw_caption),
                "caption_source": "model_generated_unverified",
                "entity_class": reference.entity_class,
                "frame_index": reference.frame_index,
                "identity_id": reference.identity_id,
                "identity_label": reference.identity_label,
                "legacy_action": reference.legacy_action,
                "palette_facts": [
                    {"fraction": fraction, "name": name}
                    for name, fraction in reference.palette_facts
                ],
                "reference_array_sha256": reference.reference_array_sha256,
                "sequence_id": reference.sequence_id,
                "source_array_sha256": reference.source_array_sha256,
                "source_file_sha256": reference.source_file_sha256,
                "split": reference.split,
                "structured_verb": reference.structured_verb,
                "training_prompt": detailed_training_prompt(reference, raw_caption),
                "visible_pixel_count": reference.visible_pixel_count,
            }
        )
    return records


def _load_runtime(device_name: str) -> dict[str, Any]:
    import torch
    from transformers import BartTokenizerFast, CLIPImageProcessor

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA captioning requested but unavailable")
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    # Import the exact hash-verified Microsoft modules directly. This bypasses
    # Transformers' remote-code downloader/import scanner; Hub/network access is
    # disabled, and no source outside MODEL_DIR can satisfy this package.
    package = types.ModuleType("spritelab_florence2_pinned")
    package.__path__ = [str(MODEL_DIR)]
    sys.modules[package.__name__] = package

    def load_module(name: str) -> Any:
        module_name = f"{package.__name__}.{name}"
        specification = importlib.util.spec_from_file_location(
            module_name, MODEL_DIR / f"{name}.py"
        )
        if specification is None or specification.loader is None:
            raise RuntimeError(f"cannot import pinned Florence module: {name}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        return module

    configuration = load_module("configuration_florence2")
    modeling = load_module("modeling_florence2")
    processing = load_module("processing_florence2")
    image_processor = CLIPImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)
    processor = processing.Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)
    config = configuration.Florence2Config.from_pretrained(MODEL_DIR, local_files_only=True)
    model = (
        modeling.Florence2ForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            config=config,
            local_files_only=True,
            torch_dtype=dtype,
        )
        .to(device=device)
        .eval()
    )
    return {"device": device, "model": model, "processor": processor, "torch": torch}


def _verify_model_files() -> list[dict[str, Any]]:
    actual_files = {
        path.name
        for path in MODEL_DIR.iterdir()
        if path.is_file() and path.name != ".gitattributes"
    }
    if actual_files != set(EXPECTED_MODEL_FILES):
        raise RuntimeError(
            "pinned Florence file set mismatch: "
            f"expected {sorted(EXPECTED_MODEL_FILES)!r}, got {sorted(actual_files)!r}"
        )
    records = []
    for name, expected_sha256 in sorted(
        EXPECTED_MODEL_FILES.items(), key=lambda item: item[0].encode()
    ):
        path = MODEL_DIR / name
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"pinned Florence file hash mismatch for {name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        records.append({"path": name, "sha256": actual_sha256, "size_bytes": path.stat().st_size})
    return records


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
            identity_id = record.get("identity_id") if isinstance(record, dict) else None
            if not isinstance(identity_id, str) or identity_id in output:
                raise RuntimeError(
                    f"caption journal has invalid/duplicate identity at line {line_number}"
                )
            output[identity_id] = record
    return output


def _batches(values: list[MugenStillReference], size: int) -> Iterator[list[MugenStillReference]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _array_sha256(value: np.ndarray) -> str:
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


if __name__ == "__main__":
    main()
