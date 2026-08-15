"""Render an exact six-action comparison from a latent key-pose checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from spritelab.latent_keypose_train import (  # noqa: E402
    LatentKeyposeTrainingConfig,
    _build_keypose_model,
    _keypose_batch,
    _keypose_bundle_metrics,
    _keypose_prediction_contract,
)
from spritelab.latent_motion_train import (  # noqa: E402
    _load_frozen_decoder,
    load_latent_motion_training_corpus,
)
from spritelab.models.latent_motion_dit import LatentMotionDiTConfig  # noqa: E402
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.sprite_postprocess import composite_rgba_on_checkerboard  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_from_checkpoint(raw: object) -> LatentKeyposeTrainingConfig:
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config must be an object")
    values = dict(raw)
    model = values.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint model config must be an object")
    values["model"] = LatentMotionDiTConfig(**model)
    # Checkpoints written before direct regression was added are endpoint-flow.
    values.setdefault("prediction_mode", "endpoint_flow")
    return LatentKeyposeTrainingConfig(**values)


def _rgba_uint8(chw: torch.Tensor) -> np.ndarray:
    return (
        chw.detach()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _display(rgba: np.ndarray, *, size: int) -> Image.Image:
    rgb = composite_rgba_on_checkerboard(rgba)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


def _sheet(
    actions: tuple[str, ...],
    targets: list[np.ndarray],
    generated: list[np.ndarray],
) -> Image.Image:
    cell = 256
    label = 28
    left = 92
    sheet = Image.new("RGB", (left + len(actions) * cell, label + 2 * cell), (20, 22, 27))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, label + cell // 2 - 6), "TARGET", fill=(235, 235, 235))
    draw.text((8, label + cell + cell // 2 - 6), "GENERATED", fill=(235, 235, 235))
    for index, action in enumerate(actions):
        x = left + index * cell
        draw.text((x + 8, 8), action, fill=(235, 235, 235))
        sheet.paste(_display(targets[index], size=cell), (x, label))
        sheet.paste(_display(generated[index], size=cell), (x, label + cell))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--state", choices=("ema", "raw"), default="ema")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha = _sha256(checkpoint_path)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace render output: {output}")

    selection_report_path = args.selection_report.resolve()
    selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    rows = selection_report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection report has no rows")
    seed_contract = selection_report.get("seed_contract")
    if not isinstance(seed_contract, dict):
        raise ValueError("selection report has no seed contract")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact_kind") != "mugen_fixed_middle_latent_keypose_resume_checkpoint":
        raise ValueError("checkpoint has the wrong artifact kind")
    config = _config_from_checkpoint(checkpoint.get("config"))
    corpus = load_latent_motion_training_corpus(
        args.manifest.resolve(), verify_hashes=True, array_loading="lazy"
    )
    if checkpoint.get("corpus") != corpus.contract:
        raise ValueError("checkpoint corpus contract differs")
    if checkpoint.get("action_vocabulary") != list(corpus.action_vocabulary):
        raise ValueError("checkpoint action vocabulary differs")

    model = _build_keypose_model(config, len(corpus.action_vocabulary)).cpu().eval()
    state_key = "ema_model" if args.state == "ema" else "raw_model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    decoder = _load_frozen_decoder(torch, corpus, device=torch.device("cpu")).eval()
    mean = torch.tensor(corpus.channel_mean).view(1, 8, 1, 1)
    std = torch.tensor(corpus.channel_standard_deviation).view(1, 8, 1, 1)
    sequence_index = {row.sequence_id: index for index, row in enumerate(corpus.rows)}
    output.mkdir(parents=True, exist_ok=False)
    report_rows = []

    with torch.no_grad():
        for selected in rows:
            split = selected["split"]
            sequence_ids = tuple(selected["sequence_ids"])
            selection = tuple(sequence_index[sequence_id] for sequence_id in sequence_ids)
            target, reference, target_rgba, phases, actions = _keypose_batch(
                torch,
                corpus,
                selection,
                frame_index=config.keypose_frame_index,
                device=torch.device("cpu"),
                mean=mean,
                std=std,
            )
            clean = target - reference
            generator = torch.Generator(device="cpu").manual_seed(int(seed_contract[split]))
            noise = torch.randn((1, *clean.shape[1:]), generator=generator).expand_as(clean)
            model_input, _ = _keypose_prediction_contract(
                torch,
                clean_residual=clean,
                noise=noise,
                prediction_mode=config.prediction_mode,
            )
            velocity = model(
                model_input.unsqueeze(1),
                reference,
                torch.ones((len(selection),)),
                actions,
                frame_phase=phases,
            )[:, 0]
            generated_latent = (reference + model_input - velocity) * std + mean
            generated_rgba = torch.sigmoid(decoder.decode_logits(generated_latent)).reshape(
                len(selection), 1, 4, 128, 128
            )
            target_rgba_5d = target_rgba.reshape(len(selection), 1, 4, 128, 128)
            metrics = {
                key: float(value.cpu())
                for key, value in _keypose_bundle_metrics(
                    torch,
                    predicted_rgba=generated_rgba.float(),
                    target_rgba=target_rgba_5d,
                ).items()
            }
            target_arrays = [_rgba_uint8(target_rgba[index]) for index in range(len(selection))]
            generated_arrays = [
                _rgba_uint8(generated_rgba[index, 0]) for index in range(len(selection))
            ]
            for index, action in enumerate(corpus.action_vocabulary):
                target_name = f"{split}-{index:02d}-{action}-target.png"
                generated_name = f"{split}-{index:02d}-{action}-generated.png"
                _display(target_arrays[index], size=128).save(output / target_name)
                _display(generated_arrays[index], size=128).save(output / generated_name)
            sheet_name = f"{split}-six-action-keyposes.png"
            sheet = _sheet(corpus.action_vocabulary, target_arrays, generated_arrays)
            sheet.save(output / sheet_name)
            report_rows.append(
                {
                    "identity_id": selected["identity_id"],
                    "metrics": metrics,
                    "sequence_ids": list(sequence_ids),
                    "sheet": sheet_name,
                    "sheet_sha256": _sha256(output / sheet_name),
                    "split": split,
                }
            )

    report = {
        "artifact_kind": "mugen_fixed_middle_keypose_checkpoint_visible_evaluation",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint["step"],
        "claim": (
            "fixed source frame 4; raw decoder RGBA on checkerboard; selected training "
            "and identity-disjoint validation examples"
        ),
        "config": asdict(config),
        "model_state": args.state,
        "rows": report_rows,
        "schema_version": 1,
        "seed_contract": seed_contract,
        "selection_report_path": str(selection_report_path),
        "selection_report_sha256": _sha256(selection_report_path),
    }
    (output / "report.json").write_bytes(canonical_json_bytes(report))
    print(
        json.dumps(
            {"output": str(output), "report_sha256": _sha256(output / "report.json")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
