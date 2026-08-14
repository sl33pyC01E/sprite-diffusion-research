"""Run a hash-bound Qwen-Image text-to-sprite probe through local ComfyUI.

This script is intended to run on Spark after a separately managed ComfyUI
server is listening on localhost.  It preserves each raw RGB generation and
publishes only a clearly labelled 128x128 nearest-neighbour display derivative.
It does not infer transparency or claim that the derivative is canonical pixel
art.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ROOT = Path("/home/sleepy/ComfyUI/models")
MODEL_FILES = {
    "diffusion_model": MODEL_ROOT / "diffusion_models/qwen_image_fp8_e4m3fn.safetensors",
    "lightning_lora": MODEL_ROOT / "loras/Qwen-Image-Lightning-8steps-V1.0.safetensors",
    "text_encoder": MODEL_ROOT / "text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
    "vae": MODEL_ROOT / "vae/qwen_image_vae.safetensors",
}
MODEL_NAMES = {key: path.name for key, path in MODEL_FILES.items()}
FIXED_IDENTITIES = (
    "mugen_6602b0aa83934ced_5950baeb1fad85d9",
    "mugen_303702787c067e97_27db79521ab654b5",
    "mugen_cf873e2ef7bdeb2e_35c02d5014d6f05d",
    "mugen_effe528cb5b4dde7_c408c72742ec46e1",
)
NEGATIVE_PROMPT = (
    "sprite sheet, multiple characters, duplicate subject, cropped subject, text, labels, "
    "watermark, checkerboard, scenery, perspective floor, cast shadow, realistic photo, "
    "soft blurry edges"
)


@dataclass(frozen=True)
class ProbeRecord:
    identity_id: str
    sequence_id: str
    split: str
    source_prompt: str
    qwen_prompt: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="127.0.0.1:8188")
    parser.add_argument("--seed", type=int, default=20260902)
    arguments = parser.parse_args()

    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace Qwen sprite probe: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=str(output.parent.resolve()))
    )
    try:
        records, plan_sha256 = load_probe_records(arguments.plan, arguments.expected_plan_sha256)
        model_evidence = hash_model_files()
        raw_dir = staging / "raw"
        display_dir = staging / "display-128-nearest"
        raw_dir.mkdir()
        display_dir.mkdir()

        client_id = str(uuid.uuid4())
        generated = []
        for ordinal, record in enumerate(records):
            workflow = build_workflow(record.qwen_prompt, seed=arguments.seed)
            png, comfy_source = execute_workflow(arguments.server, client_id, workflow)
            raw_name = f"{ordinal:02d}-{record.identity_id}-raw-512.png"
            raw_path = raw_dir / raw_name
            _exclusive_write(raw_path, png)
            raw_image = Image.open(io.BytesIO(png)).convert("RGB")
            if raw_image.size != (512, 512):
                raise RuntimeError(
                    f"Qwen output has unexpected dimensions for {record.identity_id}: "
                    f"{raw_image.size!r}"
                )
            display = raw_image.resize((128, 128), Image.Resampling.NEAREST)
            display_name = f"{ordinal:02d}-{record.identity_id}-display-128-nearest.png"
            display_path = display_dir / display_name
            display_bytes = _png_bytes(display)
            _exclusive_write(display_path, display_bytes)
            generated.append(
                {
                    "identity_id": record.identity_id,
                    "sequence_id": record.sequence_id,
                    "split": record.split,
                    "source_prompt": record.source_prompt,
                    "qwen_prompt": record.qwen_prompt,
                    "comfyui_saved_output": comfy_source,
                    "raw": _file_record(staging, raw_path, png, [512, 512, 3]),
                    "display_derivative": {
                        **_file_record(staging, display_path, display_bytes, [128, 128, 3]),
                        "canonical_training_asset": False,
                        "operation": "Pillow RGB nearest-neighbour resize 512x512 to 128x128",
                        "purpose": "display-only zero-shot still-generator comparison",
                    },
                }
            )

        sheet_path = staging / "display-128-nearest-contact-sheet.png"
        sheet_bytes = build_contact_sheet(display_dir, len(records))
        _exclusive_write(sheet_path, sheet_bytes)
        report = {
            "artifact_kind": "mugen_qwen_image_zero_shot_sprite_probe",
            "schema_version": 1,
            "claim_limits": [
                "identity-disjoint prompts within the indexed MUGEN caption distribution",
                "zero-shot still-generator probe; no Qwen-Image fine-tuning",
                "raw RGB generations are authoritative",
                "128x128 nearest-neighbour images are display derivatives, not canonical RGBA",
                "no transparency inference, palette quantization, or background removal",
            ],
            "comfyui": {
                "server": arguments.server,
                "workflow": {
                    "cfg": 1.0,
                    "height": 512,
                    "lightning_lora_strength": 1.0,
                    "model_sampling_shift": 3.0,
                    "negative_prompt": NEGATIVE_PROMPT,
                    "sampler": "lcm",
                    "scheduler": "simple",
                    "seed": arguments.seed,
                    "steps": 8,
                    "width": 512,
                },
            },
            "generated": generated,
            "model_evidence": model_evidence,
            "prompt_policy": {
                "source": "exact Qwen-VLM canonical MUGEN appearance caption",
                "replacement": (
                    "replace the source's unsupported transparent-background request with an "
                    "explicitly flat neutral-gray RGB canvas and require one isolated full subject"
                ),
            },
            "source_plan": {
                "path": str(arguments.plan.resolve()),
                "file_sha256": plan_sha256,
            },
            "contact_sheet": _file_record(
                staging, sheet_path, sheet_bytes, [128, 128 * len(records), 3]
            ),
        }
        report_bytes = _canonical_json(report)
        report_path = staging / "report.json"
        _exclusive_write(report_path, report_bytes)
        report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        os.replace(staging, output)
    except Exception:
        print(f"Probe failed; retained staging directory: {staging}")
        raise

    print(
        json.dumps(
            {
                "output": str(output),
                "report": str(output / "report.json"),
                "report_sha256": report_sha256,
            },
            sort_keys=True,
        )
    )


def load_probe_records(plan_path: Path, expected_sha256: str) -> tuple[list[ProbeRecord], str]:
    payload = plan_path.resolve().read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Canonical still plan hash differs: expected {expected_sha256}, got {actual_sha256}"
        )
    plan = json.loads(payload)
    if plan.get("artifact_kind") != "mugen_canonical_appearance_still_training_plan":
        raise RuntimeError("Canonical still plan has the wrong artifact kind")
    by_identity = {record["identity_id"]: record for record in plan["records"]}
    selected = []
    for identity_id in FIXED_IDENTITIES:
        record = by_identity.get(identity_id)
        if record is None:
            raise RuntimeError(f"Fixed held-out identity is absent: {identity_id}")
        if record.get("split") == "train":
            raise RuntimeError(f"Fixed Qwen probe identity leaked into train: {identity_id}")
        source_prompt = record["prompt"]
        selected.append(
            ProbeRecord(
                identity_id=identity_id,
                sequence_id=record["sequence_id"],
                split=record["split"],
                source_prompt=source_prompt,
                qwen_prompt=_qwen_prompt(source_prompt),
            )
        )
    return selected, actual_sha256


def _qwen_prompt(source_prompt: str) -> str:
    description = source_prompt.replace("pixel art sprite;", "").replace(
        "transparent background;", ""
    )
    description = " ".join(description.split()).strip(" ;")
    return (
        "Create one isolated 128x128-style pixel art fighting-game character sprite, "
        "shown as a 4x nearest-neighbour enlargement on a perfectly flat neutral gray "
        "background. No ground plane and no shadow. One full-body subject only, centered "
        "with generous empty margin, readable silhouette, limited color palette, crisp "
        f"hard pixel clusters, no anti-aliased outline. Visible design: {description}."
    )


def build_workflow(prompt: str, *, seed: int) -> dict[str, Any]:
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 1.0,
                "denoise": 1.0,
                "latent_image": ["58", 0],
                "model": ["73", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": "lcm",
                "scheduler": "simple",
                "seed": seed,
                "steps": 8,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["38", 0], "text": prompt},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["38", 0], "text": NEGATIVE_PROMPT},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["39", 0]},
        },
        "37": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": MODEL_NAMES["diffusion_model"],
                "weight_dtype": "default",
            },
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": MODEL_NAMES["text_encoder"],
                "device": "default",
                "type": "qwen_image",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": MODEL_NAMES["vae"]},
        },
        "58": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"batch_size": 1, "height": 512, "width": 512},
        },
        "66": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["37", 0], "shift": 3.0},
        },
        "73": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": MODEL_NAMES["lightning_lora"],
                "model": ["66", 0],
                "strength_model": 1.0,
            },
        },
        "save_image_node": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "spritelab_qwen_probe/pending",
                "images": ["8", 0],
            },
        },
    }


def execute_workflow(
    server: str, client_id: str, workflow: dict[str, Any]
) -> tuple[bytes, dict[str, Any]]:
    workflow["save_image_node"]["inputs"]["filename_prefix"] = f"spritelab_qwen_probe/{client_id}"
    request = urllib.request.Request(
        f"http://{server}/prompt",
        data=json.dumps({"client_id": client_id, "prompt": workflow}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            queued = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode("utf-8", errors="replace")) from error
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + 30 * 60
    history_entry = None
    while time.monotonic() < deadline:
        with urllib.request.urlopen(f"http://{server}/history/{prompt_id}", timeout=30) as response:
            history = json.loads(response.read())
        history_entry = history.get(prompt_id)
        if history_entry is not None:
            status = history_entry.get("status", {})
            if status.get("completed"):
                break
            messages = status.get("messages", [])
            if any(message and message[0] == "execution_error" for message in messages):
                raise RuntimeError(json.dumps(status, sort_keys=True))
        time.sleep(2)
    else:
        raise TimeoutError(f"Timed out waiting for ComfyUI prompt {prompt_id}")
    output = history_entry.get("outputs", {}).get("save_image_node", {})
    images = output.get("images", [])
    if len(images) != 1:
        raise RuntimeError(f"Expected one Qwen output image, received {len(images)}")
    source = images[0]
    query = urllib.parse.urlencode(
        {
            "filename": source["filename"],
            "subfolder": source["subfolder"],
            "type": source["type"],
        }
    )
    with urllib.request.urlopen(f"http://{server}/view?{query}", timeout=30) as response:
        payload = response.read()
    return payload, {
        "prompt_id": prompt_id,
        "filename": source["filename"],
        "subfolder": source["subfolder"],
        "type": source["type"],
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def hash_model_files() -> dict[str, dict[str, Any]]:
    evidence = {}
    for role, path in MODEL_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required Qwen model file is absent: {path}")
        evidence[role] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "file_sha256": _file_sha256(path),
        }
    return evidence


def build_contact_sheet(display_dir: Path, count: int) -> bytes:
    images = []
    for path in sorted(display_dir.glob("*.png")):
        images.append(Image.open(path).convert("RGB"))
    if len(images) != count:
        raise RuntimeError(f"Contact-sheet input count differs: {len(images)} != {count}")
    sheet = Image.new("RGB", (128 * count, 128), (127, 127, 127))
    for index, image in enumerate(images):
        sheet.paste(image, (index * 128, 0))
    return _png_bytes(sheet)


def _file_record(root: Path, path: Path, payload: bytes, shape: list[int]) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "shape": shape,
    }


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _exclusive_write(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


if __name__ == "__main__":
    main()
