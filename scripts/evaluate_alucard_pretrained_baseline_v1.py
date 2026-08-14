"""Evaluate the pinned pretrained Alucard model on exact held-out MUGEN prompts."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CODE_COMMIT = "02d1c60a16142015f7838a6a033da5e6ac9ce4f7"
MODEL_REVISION = "b8e7602fc8e676d0b0bc0abb11d2cda665c560d8"
CODE = ROOT / f"data/models/alucard-source-{CODE_COMMIT}"
MODEL = ROOT / f"data/models/alucard-{MODEL_REVISION}"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CODE))

from alucard import Alucard  # noqa: E402

from spritelab.sprite_postprocess import composite_rgba_on_checkerboard  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json"
SOURCE_INDEX = ROOT / "data/index/reports/alucard-pretrained-source-index-v1.json"
EXPECTED_SOURCE_INDEX_SHA256 = "39a9ce8e5f1866e4285e8c39a67c71b3a43267b9887fbd81551aa172bb511889"
UPSTREAM_OUTPUT = ROOT / "data/inference/alucard-pretrained-v1-mugen-heldout-canonical-prompts"
QUICKGELU_OUTPUT = (
    ROOT / "data/inference/alucard-pretrained-quickgelu-v2-mugen-heldout-canonical-prompts"
)
IDENTITIES = (
    "mugen_6602b0aa83934ced_5950baeb1fad85d9",
    "mugen_303702787c067e97_27db79521ab654b5",
    "mugen_cf873e2ef7bdeb2e_35c02d5014d6f05d",
    "mugen_effe528cb5b4dde7_c408c72742ec46e1",
)
CLIP_REVISION = "a6f597a30f7b82c51704746581f9a4e41421e878"
CLIP_PATH = (
    ROOT / f"data/models/openclip-vit-b32-openai-{CLIP_REVISION}/open_clip_model.safetensors"
)
CLIP_SHA256 = "e6d1bd7789aa45192b3bf90570a789b478bae1b74ebcce7eddd908e83a2b7c31"
SEED = 20260824


def main(*, clip_architecture: str) -> None:
    if clip_architecture not in {"upstream", "quickgelu"}:
        raise ValueError("clip_architecture must be upstream or quickgelu")
    output = UPSTREAM_OUTPUT if clip_architecture == "upstream" else QUICKGELU_OUTPUT
    if output.exists():
        raise FileExistsError(f"Refusing to replace Alucard evaluation: {output}")
    if _file_sha256(SOURCE_INDEX) != EXPECTED_SOURCE_INDEX_SHA256:
        raise RuntimeError("Alucard source index differs")
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    materialization = Path(plan["source"]["materialization_path"]).resolve()
    if _file_sha256(materialization) != plan["source"]["materialization_file_sha256"]:
        raise RuntimeError("MUGEN materialization differs")
    by_identity = {record["identity_id"]: record for record in plan["records"]}
    selected = []
    for identity in IDENTITIES:
        record = by_identity.get(identity)
        if record is None or record["split"] == "train":
            raise RuntimeError(f"held-out identity contract differs: {identity}")
        selected.append(record)
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    guard.require_capacity(1024**3, label="Alucard pretrained evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=output.parent)
    )
    try:
        model = _load_model(clip_architecture)
        if _file_sha256(CLIP_PATH) != CLIP_SHA256:
            raise RuntimeError("OpenAI CLIP ViT-B/32 weight hash differs")
        rows = []
        gallery_rows = []
        for index, record in enumerate(selected):
            target = record["target"]
            target_path = (materialization.parent / target["relative_path"]).resolve()
            target_bytes = target_path.read_bytes()
            if hashlib.sha256(target_bytes).hexdigest() != target["file_sha256"]:
                raise RuntimeError("exact MUGEN target bytes differ")
            target_array = np.load(io.BytesIO(target_bytes), allow_pickle=False)
            frame_index = target["eligible_frame_indices"][0]
            target_rgba = np.ascontiguousarray(target_array[frame_index])
            generated_image = model(
                record["prompt"],
                num_steps=40,
                cfg_text=5.0,
                seed=SEED,
            )
            generated_rgba = np.asarray(generated_image.convert("RGBA"), dtype=np.uint8)
            if generated_rgba.shape != (128, 128, 4):
                raise RuntimeError("Alucard output geometry differs")
            stem = f"{index:02d}-{identity_slug(record['identity_id'])}"
            raw_path = staging / f"{stem}-generated-rgba.png"
            target_path_out = staging / f"{stem}-exact-target-rgba.png"
            preview_path = staging / f"{stem}-generated-checker.png"
            Image.fromarray(generated_rgba, mode="RGBA").save(raw_path, optimize=False)
            Image.fromarray(target_rgba, mode="RGBA").save(target_path_out, optimize=False)
            Image.fromarray(composite_rgba_on_checkerboard(generated_rgba), mode="RGB").resize(
                (512, 512), Image.Resampling.NEAREST
            ).save(preview_path, optimize=False)
            rows.append(
                {
                    "generated": {
                        "array_content_sha256": _array_sha256(generated_rgba),
                        "file_sha256": _file_sha256(raw_path),
                        "path": raw_path.name,
                    },
                    "identity_id": record["identity_id"],
                    "prompt": record["prompt"],
                    "sequence_id": record["sequence_id"],
                    "split": record["split"],
                    "target": {
                        "array_content_sha256": _array_sha256(target_rgba),
                        "file_sha256": _file_sha256(target_path_out),
                        "frame_index": frame_index,
                        "path": target_path_out.name,
                    },
                }
            )
            gallery_rows.append((target_rgba, generated_rgba, record["prompt"]))
        gallery_path = staging / "exact-target-vs-alucard-gallery.png"
        _gallery(gallery_rows, gallery_path)
        report = {
            "artifact_kind": "alucard_pretrained_mugen_heldout_evaluation",
            "claim": (
                "external pretrained baseline on identity-held-out MUGEN prompts; no fine-tuning; "
                "target and generation share description but not pixel/noise alignment"
            ),
            "config": {
                "cfg_text": 5.0,
                "device": "cuda",
                "num_steps": 40,
                "seed_reused_per_prompt_for_matched_initial_noise": SEED,
            },
            "gallery": {
                "file_sha256": _file_sha256(gallery_path),
                "path": gallery_path.name,
            },
            "model": {
                "clip_architecture": clip_architecture,
                "clip_file_sha256": CLIP_SHA256,
                "clip_repository": "timm/vit_base_patch32_clip_224.openai",
                "clip_revision": CLIP_REVISION,
                "code_commit": CODE_COMMIT,
                "model_revision": MODEL_REVISION,
                "source_index_file_sha256": EXPECTED_SOURCE_INDEX_SHA256,
            },
            "plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "rows": rows,
            "schema_version": 1,
        }
        payload = _canonical_json(report)
        report_path = staging / "evaluation-report.json"
        with report_path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(staging, output)
    except BaseException:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise
    print(
        json.dumps(
            {
                "gallery": str(output / gallery_path.name),
                "report": str(output / report_path.name),
                "report_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def _load_model(clip_architecture: str) -> Alucard:
    if clip_architecture == "upstream":
        return Alucard.from_pretrained(str(MODEL), device="cuda")
    import open_clip
    from alucard.model import UNet
    from safetensors.torch import load_file

    unet = UNet()
    unet.load_state_dict(load_file(str(MODEL / "alucard_model.safetensors"), device="cpu"))
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32-quickgelu", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
    return Alucard(unet, clip_model, tokenizer, device="cuda")


def _gallery(rows: list[tuple[np.ndarray, np.ndarray, str]], path: Path) -> None:
    width = 1100
    row_height = 600
    canvas = Image.new("RGB", (width, row_height * len(rows)), (25, 27, 32))
    draw = ImageDraw.Draw(canvas)
    for index, (target, generated, prompt) in enumerate(rows):
        top = index * row_height
        target_display = Image.fromarray(composite_rgba_on_checkerboard(target), mode="RGB").resize(
            (480, 480), Image.Resampling.NEAREST
        )
        generated_display = Image.fromarray(
            composite_rgba_on_checkerboard(generated), mode="RGB"
        ).resize((480, 480), Image.Resampling.NEAREST)
        canvas.paste(target_display, (40, top + 40))
        canvas.paste(generated_display, (580, top + 40))
        draw.text((40, top + 16), "EXACT MUGEN TARGET", fill=(220, 226, 236))
        draw.text((580, top + 16), "ALUCARD PRETRAINED", fill=(220, 226, 236))
        for line_index, line in enumerate(_wrap(prompt, 130)[:3]):
            draw.text((24, top + 536 + line_index * 18), line, fill=(178, 188, 204))
    canvas.save(path, format="PNG", optimize=False)


def identity_slug(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:10]


def _wrap(value: str, maximum: int) -> list[str]:
    output = []
    line = ""
    for word in value.split():
        candidate = f"{line} {word}".strip()
        if line and len(candidate) > maximum:
            output.append(line)
            line = word
        else:
            line = candidate
    if line:
        output.append(line)
    return output


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip-architecture", choices=("upstream", "quickgelu"), default="upstream"
    )
    arguments = parser.parse_args()
    torch.set_grad_enabled(False)
    main(clip_architecture=arguments.clip_architecture)
