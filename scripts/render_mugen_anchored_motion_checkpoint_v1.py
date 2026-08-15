"""Render exact teacher-forced MUGEN anchored-motion checkpoint comparisons."""

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

from spritelab.anchored_motion_train import (  # noqa: E402
    AnchoredMotionTrainingConfig,
    _anchored_batch,
    _trajectory_bundle_metrics,
    _validate,
    sample_anchored_motion_residual,
)
from spritelab.latent_keypose_train import (  # noqa: E402
    LatentKeyposeTrainingConfig,
    _build_keypose_model,
    _keypose_prediction_contract,
    build_keypose_action_bundles,
)
from spritelab.latent_motion_train import (  # noqa: E402
    _load_frozen_decoder,
    load_latent_motion_training_corpus,
)
from spritelab.models.anchored_latent_motion_dit import (  # noqa: E402
    AnchoredActionConditionedLatentMotionDiT,
)
from spritelab.models.latent_motion_dit import LatentMotionDiTConfig  # noqa: E402
from spritelab.previews import export_rgba_clip_preview  # noqa: E402
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.sprite_postprocess import composite_rgba_on_checkerboard  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_from_checkpoint(raw: object) -> AnchoredMotionTrainingConfig:
    if not isinstance(raw, dict):
        raise ValueError("checkpoint config must be an object")
    values = dict(raw)
    model = values.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint model config must be an object")
    values["model"] = LatentMotionDiTConfig(**model)
    return AnchoredMotionTrainingConfig(**values)


def _keypose_config_from_checkpoint(raw: object) -> LatentKeyposeTrainingConfig:
    if not isinstance(raw, dict):
        raise ValueError("key-pose checkpoint config must be an object")
    values = dict(raw)
    model = values.get("model")
    if not isinstance(model, dict):
        raise ValueError("key-pose checkpoint model config must be an object")
    values["model"] = LatentMotionDiTConfig(**model)
    values.setdefault("prediction_mode", "endpoint_flow")
    return LatentKeyposeTrainingConfig(**values)


def _rgba_uint8(btchw: torch.Tensor) -> np.ndarray:
    return (
        btchw.detach()
        .float()
        .clamp(0, 1)
        .permute(0, 1, 3, 4, 2)
        .mul(255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _comparison_sheet(target: np.ndarray, generated: np.ndarray) -> Image.Image:
    if target.shape != generated.shape or target.ndim != 4 or target.shape[-1] != 4:
        raise ValueError("comparison clips must share uint8 [T,H,W,4] geometry")
    frame_count = target.shape[0]
    cell = 256
    label_width = 92
    header = 28
    sheet = Image.new("RGB", (label_width + frame_count * cell, header + 2 * cell), (20, 22, 27))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, header + cell // 2 - 6), "TARGET", fill=(235, 235, 235))
    draw.text((8, header + cell + cell // 2 - 6), "GENERATED", fill=(235, 235, 235))
    for frame_index in range(frame_count):
        x = label_width + frame_index * cell
        draw.text((x + 8, 8), f"frame {frame_index}", fill=(235, 235, 235))
        for row, clip in enumerate((target, generated)):
            display = Image.fromarray(composite_rgba_on_checkerboard(clip[frame_index]))
            display = display.resize((cell, cell), Image.Resampling.NEAREST)
            sheet.paste(display, (x, header + row * cell))
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--state", choices=("ema", "raw"), default="ema")
    parser.add_argument("--aggregate-identities", type=int, default=0)
    parser.add_argument("--keypose-checkpoint", type=Path)
    parser.add_argument("--expected-keypose-sha256")
    parser.add_argument("--keypose-state", choices=("ema", "raw"), default="raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.aggregate_identities < 0:
        raise ValueError("aggregate identities cannot be negative")
    if (args.keypose_checkpoint is None) != (args.expected_keypose_sha256 is None):
        parser.error("key-pose checkpoint and expected SHA-256 must be supplied together")
    if args.keypose_checkpoint is not None and args.aggregate_identities:
        parser.error("predicted-keypose aggregate evaluation is not implemented in v1")

    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha256 = _sha256(checkpoint_path)
    if checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace render output: {output}")

    selection_report_path = args.selection_report.resolve()
    selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    rows = selection_report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection report has no rows")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("artifact_kind") != (
        "mugen_start_middle_start_anchored_motion_resume_checkpoint"
    ):
        raise ValueError("checkpoint has the wrong artifact kind")
    config = _config_from_checkpoint(checkpoint.get("config"))
    corpus = load_latent_motion_training_corpus(
        args.manifest.resolve(), verify_hashes=True, array_loading="lazy"
    )
    if checkpoint.get("corpus") != corpus.contract:
        raise ValueError("checkpoint corpus contract differs")
    if checkpoint.get("action_vocabulary") != list(corpus.action_vocabulary):
        raise ValueError("checkpoint action vocabulary differs")

    model = (
        AnchoredActionConditionedLatentMotionDiT(
            config.model,
            len(corpus.action_vocabulary),
            action_token_count=config.action_token_count,
            action_condition_scale=config.action_condition_scale,
        )
        .cpu()
        .eval()
    )
    state_key = "ema_model" if args.state == "ema" else "raw_model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    decoder = _load_frozen_decoder(torch, corpus, device=torch.device("cpu")).eval()
    keypose_model = None
    keypose_config = None
    keypose_checkpoint_path = None
    keypose_checkpoint_sha256 = None
    if args.keypose_checkpoint is not None:
        keypose_checkpoint_path = args.keypose_checkpoint.resolve()
        keypose_checkpoint_sha256 = _sha256(keypose_checkpoint_path)
        if keypose_checkpoint_sha256 != args.expected_keypose_sha256:
            raise ValueError("key-pose checkpoint SHA-256 differs")
        keypose_checkpoint = torch.load(
            keypose_checkpoint_path, map_location="cpu", weights_only=True
        )
        if keypose_checkpoint.get("artifact_kind") != (
            "mugen_fixed_middle_latent_keypose_resume_checkpoint"
        ):
            raise ValueError("key-pose checkpoint has the wrong artifact kind")
        if keypose_checkpoint.get("corpus") != corpus.contract:
            raise ValueError("key-pose checkpoint corpus contract differs")
        if keypose_checkpoint.get("action_vocabulary") != list(corpus.action_vocabulary):
            raise ValueError("key-pose checkpoint action vocabulary differs")
        keypose_config = _keypose_config_from_checkpoint(keypose_checkpoint.get("config"))
        if keypose_config.keypose_frame_index != config.canonical_middle_frame_index:
            raise ValueError("key-pose frame differs from the motion middle anchor")
        keypose_model = (
            _build_keypose_model(keypose_config, len(corpus.action_vocabulary)).cpu().eval()
        )
        keypose_state_key = "ema_model" if args.keypose_state == "ema" else "raw_model"
        keypose_model.load_state_dict(keypose_checkpoint[keypose_state_key], strict=True)
    mean = torch.tensor(corpus.channel_mean).view(1, 1, 8, 1, 1)
    std = torch.tensor(corpus.channel_standard_deviation).view(1, 1, 8, 1, 1)
    sequence_index = {row.sequence_id: index for index, row in enumerate(corpus.rows)}
    guard = DiskGuard(output.parent, min_free_bytes=100 * 1024**3)
    guard.require_capacity(512 * 1024**2, label="anchored-motion checkpoint render")
    output.mkdir(parents=True, exist_ok=False)
    report_rows = []
    split_seeds = {"training": config.seed + 20_000, "validation": config.seed + 20_001}

    with torch.no_grad():
        for selected in rows:
            split = selected["split"]
            if split not in split_seeds:
                raise ValueError(f"unsupported selected split: {split!r}")
            sequence_ids = tuple(selected["sequence_ids"])
            selection = tuple(sequence_index[sequence_id] for sequence_id in sequence_ids)
            clean, reference, target_rgba, phases, actions, anchors, mask = _anchored_batch(
                torch,
                corpus,
                selection,
                config=config,
                device=torch.device("cpu"),
                mean=mean,
                std=std,
            )
            true_middle_anchor = anchors[:, config.canonical_middle_frame_index].clone()
            if keypose_model is not None:
                assert keypose_config is not None
                keypose_generator = torch.Generator(device="cpu").manual_seed(
                    keypose_config.seed + 20_000 + (0 if split == "training" else 1)
                )
                keypose_noise = torch.randn(
                    (1, *true_middle_anchor.shape[1:]), generator=keypose_generator
                ).expand_as(true_middle_anchor)
                keypose_input, _ = _keypose_prediction_contract(
                    torch,
                    clean_residual=true_middle_anchor,
                    noise=keypose_noise,
                    prediction_mode=keypose_config.prediction_mode,
                )
                keypose_phase = torch.full(
                    (len(selection), 1),
                    config.canonical_middle_frame_index / config.model.num_frames,
                )
                keypose_velocity = keypose_model(
                    keypose_input.unsqueeze(1),
                    reference,
                    torch.ones((len(selection),)),
                    actions,
                    frame_phase=keypose_phase,
                )[:, 0]
                anchors = anchors.clone()
                anchors[:, config.canonical_middle_frame_index] = keypose_input - keypose_velocity
            generator = torch.Generator(device="cpu").manual_seed(split_seeds[split])
            noise = torch.randn((1, *clean.shape[1:]), generator=generator).expand_as(clean)
            residual = sample_anchored_motion_residual(
                torch,
                model,
                noise=noise,
                reference=reference,
                actions=actions,
                phases=phases,
                anchor_residuals=anchors,
                anchor_mask=mask,
            )
            generated_latent = (reference.unsqueeze(1) + residual) * std + mean
            generated_rgba = torch.sigmoid(
                decoder.decode_logits(generated_latent.reshape(-1, 8, 64, 64))
            ).reshape(len(selection), 8, 4, 128, 128)
            evaluation_frames = (1, 2, 3, 4, 5, 6) if keypose_model is not None else (1, 2, 3, 5, 6)
            metrics = {
                key: float(value.cpu())
                for key, value in _trajectory_bundle_metrics(
                    torch,
                    predicted_rgba=generated_rgba[:, evaluation_frames].float(),
                    target_rgba=target_rgba[:, evaluation_frames],
                ).items()
            }
            metrics["anchor_latent_max_abs_error"] = float(
                ((residual - anchors).abs() * mask.view(len(selection), 8, 1, 1, 1)).max().cpu()
            )
            predicted_middle = generated_rgba[:, config.canonical_middle_frame_index].float()
            target_middle = target_rgba[:, config.canonical_middle_frame_index]
            predicted_middle_alpha = predicted_middle[:, 3:4]
            target_middle_alpha = target_middle[:, 3:4]
            predicted_middle_pm = torch.cat(
                (predicted_middle[:, :3] * predicted_middle_alpha, predicted_middle_alpha), dim=1
            )
            target_middle_pm = torch.cat(
                (target_middle[:, :3] * target_middle_alpha, target_middle_alpha), dim=1
            )
            middle_anchor_metrics = {
                "latent_residual_mae": float(
                    (anchors[:, config.canonical_middle_frame_index] - true_middle_anchor)
                    .abs()
                    .mean()
                    .cpu()
                ),
                "premultiplied_rgba_mae": float(
                    torch.nn.functional.l1_loss(predicted_middle_pm, target_middle_pm).cpu()
                ),
            }
            target_arrays = _rgba_uint8(target_rgba)
            generated_arrays = _rgba_uint8(generated_rgba)
            action_rows = []
            for index, action in enumerate(corpus.action_vocabulary):
                stem = f"{split}-{index:02d}-{action}"
                row = corpus.rows[selection[index]]
                target_preview = export_rgba_clip_preview(
                    target_arrays[index],
                    output,
                    artifact_stem=f"{stem}-target",
                    duration_ms=row.duration_ms,
                    loop_mode="loop",
                    integer_scale=2,
                    preserve_frame_slots=True,
                    disk_guard=guard,
                )
                generated_preview = export_rgba_clip_preview(
                    generated_arrays[index],
                    output,
                    artifact_stem=f"{stem}-generated",
                    duration_ms=row.duration_ms,
                    loop_mode="loop",
                    integer_scale=2,
                    preserve_frame_slots=True,
                    disk_guard=guard,
                )
                comparison_path = output / f"{stem}-comparison.png"
                _comparison_sheet(target_arrays[index], generated_arrays[index]).save(
                    comparison_path, optimize=False
                )
                action_rows.append(
                    {
                        "action": action,
                        "comparison_path": comparison_path.name,
                        "comparison_sha256": _sha256(comparison_path),
                        "generated_animated_path": generated_preview.animated_png_path.name,
                        "generated_animated_sha256": generated_preview.animated_png_sha256,
                        "generated_sheet_path": generated_preview.contact_sheet_path.name,
                        "generated_sheet_sha256": generated_preview.contact_sheet_sha256,
                        "sequence_id": row.sequence_id,
                        "target_animated_path": target_preview.animated_png_path.name,
                        "target_animated_sha256": target_preview.animated_png_sha256,
                        "target_sheet_path": target_preview.contact_sheet_path.name,
                        "target_sheet_sha256": target_preview.contact_sheet_sha256,
                    }
                )
            report_rows.append(
                {
                    "actions": action_rows,
                    "identity_id": selected["identity_id"],
                    "middle_anchor_metrics": middle_anchor_metrics,
                    "metrics": metrics,
                    "sequence_ids": list(sequence_ids),
                    "split": split,
                }
            )

    report = {
        "anchor_contract": {
            "anchor_frames": list(config.anchor_frame_indices),
            "canonical_middle_frame": config.canonical_middle_frame_index,
            "predicted_frames": [1, 2, 3, 5, 6],
        },
        "artifact_kind": "mugen_anchored_motion_checkpoint_visible_evaluation",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": checkpoint["step"],
        "claim": (
            (
                "predicted frame-4 key-pose anchor; exact reference at frames 0 and 7; "
                "raw decoder RGBA; no text-to-image claim"
            )
            if keypose_model is not None
            else (
                "teacher-forced true frame-4 anchor; exact reference at frames 0 and 7; "
                "raw decoder RGBA; no predicted-keypose or text-to-image claim"
            )
        ),
        "config": asdict(config),
        "middle_anchor": (
            {
                "checkpoint_path": str(keypose_checkpoint_path),
                "checkpoint_sha256": keypose_checkpoint_sha256,
                "model_state": args.keypose_state,
                "source": "predicted_keypose",
            }
            if keypose_model is not None
            else {"source": "teacher_forced_true_frame_4"}
        ),
        "model_state": args.state,
        "rows": report_rows,
        "schema_version": 1,
        "seed_contract": split_seeds,
        "selection_report_path": str(selection_report_path),
        "selection_report_sha256": _sha256(selection_report_path),
    }
    if args.aggregate_identities:
        train_bundles = build_keypose_action_bundles(corpus, corpus.train_indices)[
            : args.aggregate_identities
        ]
        validation_bundles = build_keypose_action_bundles(corpus, corpus.validation_indices)[
            : args.aggregate_identities
        ]
        report["aggregate_metrics"] = {
            "training": {
                "identity_count": len(train_bundles),
                "metrics": _validate(
                    torch,
                    corpus,
                    train_bundles,
                    model,
                    decoder,
                    config=config,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    autocast=False,
                    mean=mean,
                    std=std,
                ),
            },
            "validation": {
                "identity_count": len(validation_bundles),
                "metrics": _validate(
                    torch,
                    corpus,
                    validation_bundles,
                    model,
                    decoder,
                    config=config,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    autocast=False,
                    mean=mean,
                    std=std,
                ),
            },
        }
    report_path = output / "report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    print(
        json.dumps({"output": str(output), "report_sha256": _sha256(report_path)}, sort_keys=True)
    )


if __name__ == "__main__":
    main()
