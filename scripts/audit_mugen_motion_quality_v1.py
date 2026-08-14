"""Publish the exact MUGEN six-action latent-motion quality ablation."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from spritelab.evaluation import compare_matched_sequences  # noqa: E402
from spritelab.latent_motion_overfit import (  # noqa: E402
    _decode_latents,
    _images,
    load_motion_overfit_corpus,
)
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-reference-motion-plan-v2.json"
IDENTITY = "mugen_736300dbce136df7_5d0d3dd2a2377512"
VERBS = ("idle", "walk", "run", "block", "normal_attack", "hurt")
OUTPUT = ROOT / "data/index/reports/mugen-reference-motion-quality-ablation-v1.json"
RUNS = (
    (
        "mixed_flow_6000",
        ROOT / "data/experiments/mugen-reference-motion-736300-six-action-v1-step6000",
    ),
    (
        "endpoint_only_6000",
        ROOT
        / "data/experiments/mugen-reference-motion-736300-six-action-endpoint-only-v1-step6000",
    ),
    (
        "pixel_refined_parent6000_plus3000",
        ROOT / "data/experiments/mugen-reference-motion-736300-pixel-refine-v1-parent6000-plus3000",
    ),
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(metrics: list[dict[str, object]]) -> dict[str, float]:
    fields = ("premultiplied_rgba_mae", "alpha_iou", "temporal_delta_mae")
    return {field: sum(float(row[field]) for row in metrics) / len(metrics) for field in fields}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace quality audit: {OUTPUT}")
    DiskGuard(ROOT, min_free_bytes=100 * 1024**3).require_capacity(
        1024**2, label="MUGEN motion quality audit"
    )
    corpus = load_motion_overfit_corpus(PLAN, identity_id=IDENTITY, verbs=VERBS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec_decoded = _decode_latents(
        torch,
        corpus,
        torch.from_numpy(corpus.target_latents).to(device),
        device=device,
    )
    codec_metrics = []
    for index, verb in enumerate(VERBS):
        metric = compare_matched_sequences(
            _images(codec_decoded[index]),
            _images(corpus.target_rgba[index]),
            loop_mode=corpus.loop_modes[index],
        )
        codec_metrics.append({"verb": verb, **asdict(metric)})

    runs = []
    for label, directory in RUNS:
        report_path = directory / "report.json"
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        if report.get("artifact_kind") != "mugen_reference_latent_motion_overfit_report":
            raise ValueError(f"{label} report has the wrong artifact kind")
        if report.get("plan_file_sha256") != corpus.plan_file_sha256:
            raise ValueError(f"{label} report uses a different plan")
        checkpoint_path = directory / str(report["checkpoint"]["path"])
        checkpoint_sha256 = _file_sha256(checkpoint_path)
        if checkpoint_sha256 != report["checkpoint"]["file_sha256"]:
            raise ValueError(f"{label} checkpoint hash differs")
        metrics = report.get("metrics")
        if not isinstance(metrics, list) or [row.get("verb") for row in metrics] != list(VERBS):
            raise ValueError(f"{label} metrics differ from the requested verbs")
        runs.append(
            {
                "checkpoint_file_sha256": checkpoint_sha256,
                "endpoint": report["endpoint"],
                "initialization": report.get("initialization"),
                "label": label,
                "metrics": metrics,
                "report_file_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "summary": _summary(metrics),
            }
        )

    artifact = {
        "artifact_kind": "mugen_reference_latent_motion_quality_ablation",
        "claim": "same-identity in-sample reconstruction and action-token sensitivity only",
        "codec_floor": {
            "autoencoder_checkpoint_sha256": corpus.autoencoder_checkpoint_sha256,
            "metrics": codec_metrics,
            "summary": _summary(codec_metrics),
        },
        "identity_id": IDENTITY,
        "plan_file_sha256": corpus.plan_file_sha256,
        "runs": runs,
        "schema_version": 1,
        "verbs": list(VERBS),
    }
    payload = canonical_json_bytes(artifact)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {"path": str(OUTPUT), "sha256": hashlib.sha256(payload).hexdigest()},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
