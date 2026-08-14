"""Generate matched-noise Qwen-Image stills for four held-out MUGEN identities."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from diffusers import QwenImagePipeline
from PIL import Image, ImageDraw

FIXED_IDENTITIES = (
    "mugen_6602b0aa83934ced_5950baeb1fad85d9",
    "mugen_303702787c067e97_27db79521ab654b5",
    "mugen_cf873e2ef7bdeb2e_35c02d5014d6f05d",
    "mugen_effe528cb5b4dde7_c408c72742ec46e1",
)
NEGATIVE_PROMPT = (
    "multiple subjects, sprite sheet, animation frames, text, labels, watermark, scenery, "
    "perspective floor, ground shadow, cropped body, blurry silhouette"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-dataset-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora", type=Path)
    parser.add_argument("--expected-lora-sha256")
    parser.add_argument("--lora-weight-name", default="pytorch_lora_weights.safetensors")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    args = parser.parse_args()
    if (args.lora is None) != (args.expected_lora_sha256 is None):
        raise ValueError("LoRA path and expected hash must be supplied together")
    if args.steps <= 0 or args.true_cfg_scale <= 1:
        raise ValueError("steps must be positive and true CFG must exceed one")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace Qwen held-out evaluation: {output}")

    model = args.model.resolve()
    model_manifest = model / "spritelab-snapshot-manifest.json"
    model_manifest_sha256 = _expect_hash(
        model_manifest, args.expected_model_manifest_sha256, "model snapshot manifest"
    )
    snapshot = _object(model_manifest.read_bytes(), "model snapshot manifest")
    if (
        snapshot.get("artifact_kind") != "spritelab_huggingface_model_snapshot"
        or snapshot.get("repo_id") != "Qwen/Qwen-Image"
        or snapshot.get("revision") != "75e0b4be04f60ec59a75f475837eced720f823b6"
    ):
        raise RuntimeError("model snapshot contract differs")

    dataset = args.dataset.resolve()
    dataset_manifest = dataset / "manifest.json"
    dataset_manifest_sha256 = _expect_hash(
        dataset_manifest, args.expected_dataset_manifest_sha256, "dataset manifest"
    )
    dataset_payload = _object(dataset_manifest.read_bytes(), "dataset manifest")
    if (
        dataset_payload.get("artifact_kind") != "mugen_qwen_image_lora_imagefolder"
        or dataset_payload.get("split") == "train"
    ):
        raise RuntimeError("evaluation dataset is not a held-out Qwen Image split")
    by_identity = _unique(dataset_payload.get("records"), "identity_id")
    selected = []
    for identity_id in FIXED_IDENTITIES:
        record = by_identity.get(identity_id)
        if record is None:
            raise RuntimeError(f"fixed held-out identity is absent: {identity_id}")
        target_path = _inside(dataset, _text(record, "image_relative_path"))
        _expect_hash(target_path, _text(record, "image_file_sha256"), "target image")
        selected.append((record, target_path))

    lora_evidence = None
    if args.lora is not None:
        lora = args.lora.resolve()
        lora_file = lora / args.lora_weight_name if lora.is_dir() else lora
        lora_sha256 = _expect_hash(lora_file, args.expected_lora_sha256, "LoRA weights")
        lora_evidence = {
            "file_sha256": lora_sha256,
            "path": str(lora_file),
            "weight_name": lora_file.name,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=str(output.parent.resolve()))
    )
    try:
        pipe = QwenImagePipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=False)
        if lora_evidence is not None:
            pipe.load_lora_weights(
                str(Path(lora_evidence["path"]).parent),
                weight_name=lora_evidence["weight_name"],
            )
        raw_root = staging / "raw-512"
        display_root = staging / "display-128-nearest"
        comparison_root = staging / "target-generated-comparisons"
        raw_root.mkdir()
        display_root.mkdir()
        comparison_root.mkdir()
        report_records = []
        comparison_images = []
        for ordinal, (record, target_path) in enumerate(selected):
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            latent = pipe.prepare_latents(
                1,
                pipe.transformer.config.in_channels // 4,
                512,
                512,
                torch.bfloat16,
                torch.device("cuda"),
                generator,
            )
            latent_sha256 = _tensor_sha256(latent)
            generated = (
                pipe(
                    prompt=_generation_prompt(_text(record, "prompt")),
                    negative_prompt=NEGATIVE_PROMPT,
                    width=512,
                    height=512,
                    num_inference_steps=args.steps,
                    true_cfg_scale=args.true_cfg_scale,
                    latents=latent,
                )
                .images[0]
                .convert("RGB")
            )
            prefix = f"{ordinal:02d}-{_text(record, 'sample_id')}"
            raw_path = raw_root / f"{prefix}.png"
            raw_bytes = _png_bytes(generated)
            _exclusive(raw_path, raw_bytes)
            display = generated.resize((128, 128), Image.Resampling.NEAREST)
            display_path = display_root / f"{prefix}.png"
            display_bytes = _png_bytes(display)
            _exclusive(display_path, display_bytes)
            with Image.open(target_path) as image:
                target_512 = image.convert("RGB")
            target_128 = target_512.resize((128, 128), Image.Resampling.NEAREST)
            comparison = _comparison(target_128, display, identity_id=record["identity_id"])
            comparison_path = comparison_root / f"{prefix}.png"
            comparison_bytes = _png_bytes(comparison)
            _exclusive(comparison_path, comparison_bytes)
            comparison_images.append(comparison)
            report_records.append(
                {
                    "generated_display": _file_record(staging, display_path, display_bytes),
                    "generated_raw": _file_record(staging, raw_path, raw_bytes),
                    "identity_id": record["identity_id"],
                    "latent": {
                        "dtype": "bfloat16",
                        "seed": args.seed,
                        "sha256": latent_sha256,
                        "shape": list(latent.shape),
                    },
                    "prompt": _generation_prompt(record["prompt"]),
                    "sample_id": record["sample_id"],
                    "sequence_id": record["sequence_id"],
                    "target_file_sha256": record["image_file_sha256"],
                    "target_generated_comparison": _file_record(
                        staging, comparison_path, comparison_bytes
                    ),
                }
            )
        sheet = Image.new("RGB", (272, 160 * len(comparison_images)), (22, 24, 28))
        for index, comparison in enumerate(comparison_images):
            sheet.paste(comparison, (0, index * 160))
        sheet_path = staging / "heldout-target-generated-contact-sheet.png"
        sheet_bytes = _png_bytes(sheet)
        _exclusive(sheet_path, sheet_bytes)
        report = {
            "artifact_kind": "mugen_qwen_image_heldout_matched_noise_evaluation",
            "claim_limits": [
                "identity-disjoint held-out prompts from the indexed MUGEN distribution",
                "raw 512x512 RGB generations are authoritative",
                "128x128 nearest-neighbor images are display derivatives",
                "no alpha inference, palette conversion, or pixel-art quality claim",
            ],
            "dataset": {
                "manifest_file_sha256": dataset_manifest_sha256,
                "path": str(dataset),
                "split": dataset_payload["split"],
            },
            "inference": {
                "height": 512,
                "negative_prompt": NEGATIVE_PROMPT,
                "seed": args.seed,
                "steps": args.steps,
                "true_cfg_scale": args.true_cfg_scale,
                "width": 512,
            },
            "lora": lora_evidence,
            "model": {
                "manifest_file_sha256": model_manifest_sha256,
                "path": str(model),
                "repo_id": snapshot["repo_id"],
                "revision": snapshot["revision"],
            },
            "records": report_records,
            "schema_version": 1,
            "sheet": _file_record(staging, sheet_path, sheet_bytes),
        }
        report_bytes = _canonical(report)
        report_path = staging / "report.json"
        _exclusive(report_path, report_bytes)
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        os.replace(staging, output)
    except Exception:
        print(f"Evaluation failed; retained staging directory: {staging}")
        raise
    print(
        json.dumps(
            {
                "output": str(output),
                "report_sha256": report_sha256,
                "sheet": str(output / "heldout-target-generated-contact-sheet.png"),
            },
            sort_keys=True,
        )
    )
    return 0


def _generation_prompt(prompt: str) -> str:
    return (
        f"{prompt}; one isolated full-body character only; generous empty margin; "
        "perfectly flat background; no scenery; no ground plane; no cast shadow"
    )


def _comparison(target: Image.Image, generated: Image.Image, *, identity_id: str) -> Image.Image:
    output = Image.new("RGB", (272, 160), (22, 24, 28))
    output.paste(target, (4, 24))
    output.paste(generated, (140, 24))
    draw = ImageDraw.Draw(output)
    draw.text((4, 6), "HELD-OUT TARGET", fill=(235, 235, 235))
    draw.text((140, 6), "QWEN OUTPUT", fill=(235, 235, 235))
    draw.text((4, 153), identity_id[-12:], fill=(155, 160, 170), anchor="ls")
    return output


def _tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    header = f"{contiguous.dtype}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    payload = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(header.encode() + payload).hexdigest()


def _unique(raw: object, key: str) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise RuntimeError("dataset records are absent")
    output = {}
    for record in raw:
        if not isinstance(record, dict):
            raise RuntimeError("dataset record is invalid")
        value = _text(record, key)
        if value in output:
            raise RuntimeError(f"dataset contains duplicate {key}: {value}")
        output[value] = record
    return output


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise RuntimeError(f"dataset path escapes root: {relative}")
    return path


def _expect_hash(path: Path, expected: str, label: str) -> str:
    actual = _file_sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 differs: expected {expected}, got {actual}")
    return actual


def _file_record(root: Path, path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "relative_path": path.relative_to(root).as_posix(),
        "size_bytes": len(payload),
    }


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


def _text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{key} must be non-empty text")
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
