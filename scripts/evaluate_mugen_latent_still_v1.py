"""Generate a deterministic train/validation/test gallery for the scratch still DiT."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.evaluation import compare_matched_sequences  # noqa: E402
from spritelab.latent_still_inference import (  # noqa: E402
    LatentStillInferenceConfig,
    run_latent_still_inference,
)
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

DEFAULT_TEXT_MODEL = ROOT / "data/models/stable-diffusion-v1-4-eb7ecef2ce03-training-components"
TEXT_SOURCE_INDEX_SHA256 = "6c02b65f1d657f8db316c4976248b0ca6d2406b3396025e801b45c3ef6a91b47"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    if args.per_split <= 0:
        raise ValueError("--per-split must be positive")
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if not isinstance(plan, dict) or plan.get("artifact_kind") != (
        "mugen_latent_still_sequence_training_plan"
    ):
        raise ValueError("still plan has the wrong artifact kind")
    records = plan.get("records")
    if (
        not isinstance(records, list)
        or not records
        or any(not isinstance(record, dict) for record in records)
    ):
        raise ValueError("still plan records are invalid")
    selected = []
    for split in ("train", "validation", "test"):
        candidates = [record for record in records if record.get("split") == split]
        candidates.sort(
            key=lambda record: hashlib.sha256(
                (str(record.get("identity_id")) + "\0" + str(record.get("prompt"))).encode()
            ).digest()
        )
        selected.extend(candidates[: args.per_split])
    if not selected:
        raise ValueError("still plan has no supported split records")
    prompts = [str(record["prompt"]) for record in selected]
    if len(set(prompts)) != len(prompts):
        raise ValueError("selected still prompts are not unique")
    report_path, report_sha256 = run_latent_still_inference(
        args.checkpoint,
        args.codec_checkpoint,
        args.text_model,
        prompts,
        args.output,
        expected_checkpoint_sha256=_file_sha256(args.checkpoint),
        expected_codec_checkpoint_sha256=_file_sha256(args.codec_checkpoint),
        expected_text_source_index_sha256=TEXT_SOURCE_INDEX_SHA256,
        config=LatentStillInferenceConfig(
            seed=args.seed,
            sample_steps=args.sample_steps,
            guidance_scale=args.guidance_scale,
        ),
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    target_previews = [
        _export_target_preview(record, args.output, index=index, plan_root=args.plan.parent)
        for index, record in enumerate(selected)
    ]
    inference_report = json.loads(report_path.read_bytes())
    samples = inference_report.get("samples") if isinstance(inference_report, dict) else None
    if not isinstance(samples, list) or len(samples) != len(selected):
        raise ValueError("inference report sample count differs from the selected targets")
    metric_rows = []
    for record, sample in zip(selected, samples, strict=True):
        predicted = _load_generated_frame(sample, args.output)
        target, _evidence = _load_target_frame(record, args.plan.parent)
        metric_rows.append(
            {
                "alpha_threshold_0": _matched_metrics(predicted, target, alpha_threshold=0),
                "alpha_threshold_127": _matched_metrics(predicted, target, alpha_threshold=127),
            }
        )
    selection = {
        "aggregate_metrics": {
            threshold: {
                split: _aggregate_metrics(
                    [
                        metric_rows[index][threshold]
                        for index, record in enumerate(selected)
                        if split == "all" or record.get("split") == split
                    ]
                )
                for split in ("all", "train", "validation", "test")
            }
            for threshold in ("alpha_threshold_0", "alpha_threshold_127")
        },
        "artifact_kind": "mugen_latent_still_split_inference_selection",
        "inference_report": {
            "file_sha256": report_sha256,
            "path": report_path.name,
        },
        "plan": {
            "file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
            "path": str(args.plan.resolve()),
        },
        "records": [
            {
                "identity_id": str(record["identity_id"]),
                "prompt": str(record["prompt"]),
                "sample_index": index,
                "split": str(record["split"]),
                "target": target_previews[index],
                "metrics": metric_rows[index],
            }
            for index, record in enumerate(selected)
        ],
        "schema_version": 2,
    }
    selection_path = args.output / "selection.json"
    with selection_path.open("xb") as handle:
        handle.write(canonical_json_bytes(selection))
    print(
        json.dumps(
            {
                "inference_report": str(report_path),
                "inference_report_sha256": report_sha256,
                "selection": str(selection_path),
                "selection_sha256": _file_sha256(selection_path),
            },
            sort_keys=True,
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_target_preview(
    record: dict[str, object],
    output: Path,
    *,
    index: int,
    plan_root: Path,
) -> dict[str, object]:
    frame, evidence = _load_target_frame(record, plan_root)
    frame_index = int(evidence["frame_index"])
    array_sha256 = str(evidence["source_array_content_sha256"])
    file_sha256 = str(evidence["source_file_sha256"])
    prompt = str(record["prompt"])
    stem = f"{index:03d}-{hashlib.sha256(prompt.encode()).hexdigest()[:10]}-target"
    native = output / f"{stem}-native.png"
    preview = output / f"{stem}-preview.png"
    image = Image.fromarray(frame)
    image.save(native, format="PNG", optimize=False)
    image.resize((512, 512), resample=Image.Resampling.NEAREST).save(
        preview, format="PNG", optimize=False
    )
    return {
        "array_content_sha256": _array_sha256(frame),
        "frame_index": frame_index,
        "native_png": {"file_sha256": _file_sha256(native), "path": native.name},
        "preview_png": {
            "display_only_nearest_neighbor_scale": 4,
            "file_sha256": _file_sha256(preview),
            "path": preview.name,
        },
        "sequence_id": str(record["sequence_id"]),
        "source_array_content_sha256": array_sha256,
        "source_file_sha256": file_sha256,
    }


def _load_target_frame(
    record: dict[str, object], plan_root: Path
) -> tuple[np.ndarray, dict[str, object]]:
    target = record.get("target")
    if not isinstance(target, dict):
        raise ValueError("still target must be an object")
    relative = target.get("relative_path")
    file_sha256 = target.get("file_sha256")
    array_sha256 = target.get("array_content_sha256")
    eligible = target.get("eligible_frame_indices")
    reference_index = target.get("reference_frame_index")
    reference_sha256 = target.get("reference_frame_array_content_sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(file_sha256, str)
        or not isinstance(array_sha256, str)
        or not isinstance(eligible, list)
        or not eligible
        or any(
            isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < 8
            for frame in eligible
        )
        or eligible != sorted(set(eligible))
    ):
        raise ValueError("still target evidence is invalid")
    if reference_index is None and len(eligible) == 1:
        reference_index = eligible[0]
    if (
        isinstance(reference_index, bool)
        or not isinstance(reference_index, int)
        or reference_index not in eligible
    ):
        raise ValueError("still target reference frame is invalid")
    if reference_sha256 is not None and (
        not isinstance(reference_sha256, str)
        or len(reference_sha256) != 64
        or any(character not in "0123456789abcdef" for character in reference_sha256)
    ):
        raise ValueError("still target reference frame hash is invalid")
    source = (plan_root / relative).resolve()
    plan_root_resolved = plan_root.resolve()
    if plan_root_resolved != source and plan_root_resolved not in source.parents:
        raise ValueError("still target escapes the plan root")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != file_sha256:
        raise ValueError("still target file SHA-256 differs")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (8, 128, 128, 4):
        raise ValueError("still target array geometry differs")
    value = np.ascontiguousarray(value)
    if _array_sha256(value) != array_sha256:
        raise ValueError("still target array SHA-256 differs")
    frame_index = reference_index
    frame = np.ascontiguousarray(value[frame_index])
    if reference_sha256 is not None and _array_sha256(frame) != reference_sha256:
        raise ValueError("still target reference frame SHA-256 differs")
    return frame, {
        "frame_index": frame_index,
        "source_array_content_sha256": array_sha256,
        "source_file_sha256": file_sha256,
    }


def _load_generated_frame(sample: object, output: Path) -> np.ndarray:
    if not isinstance(sample, dict) or not isinstance(sample.get("array"), dict):
        raise ValueError("inference sample array evidence is invalid")
    evidence = sample["array"]
    relative = evidence.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).name != relative:
        raise ValueError("inference sample array path is invalid")
    path = output / relative
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != evidence.get("file_sha256"):
        raise ValueError("inference sample file SHA-256 differs")
    value = np.load(io.BytesIO(payload), allow_pickle=False)
    if value.dtype != np.uint8 or value.shape != (128, 128, 4):
        raise ValueError("inference sample array geometry differs")
    value = np.ascontiguousarray(value)
    if _array_sha256(value) != evidence.get("array_content_sha256"):
        raise ValueError("inference sample array SHA-256 differs")
    return value


def _matched_metrics(
    prediction: np.ndarray, target: np.ndarray, *, alpha_threshold: int
) -> dict[str, object]:
    metrics = compare_matched_sequences(
        [Image.fromarray(prediction)],
        [Image.fromarray(target)],
        loop_mode="one_shot",
        alpha_threshold=alpha_threshold,
    )
    return asdict(metrics)


def _aggregate_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"count": 0, "mean": {}}
    fields = (
        "premultiplied_rgba_mae",
        "alpha_mae",
        "alpha_iou",
        "alpha_precision",
        "alpha_recall",
        "composite_black_mae",
        "composite_white_mae",
        "alpha_centroid_error_px",
        "alpha_bbox_edge_mae_px",
        "target_visible_premultiplied_rgba_mae",
        "target_background_premultiplied_rgba_mae",
        "predicted_visible_canvas_fraction",
        "target_visible_canvas_fraction",
        "predicted_to_target_visible_canvas_ratio",
    )
    return {
        "count": len(rows),
        "mean": {field: float(np.mean([float(row[field]) for row in rows])) for field in fields},
    }


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()


if __name__ == "__main__":
    main()
