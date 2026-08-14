"""Audit the frozen sprite codec on every untouched broad-MUGEN test clip."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from spritelab.evaluation import compare_matched_sequences  # noqa: E402
from spritelab.latent_motion_overfit import _images  # noqa: E402
from spritelab.latent_motion_train import (  # noqa: E402
    _load_frozen_decoder,
    load_latent_motion_training_corpus,
)
from spritelab.previews import export_rgba_clip_preview  # noqa: E402
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

MANIFEST = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"
OUTPUT = ROOT / "data/index/reports/mugen-primary-motion-broad-v2-codec-floor-test-v1.json"
PREVIEW_ROOT = ROOT / "data/inference/mugen-primary-motion-broad-v2-codec-floor-test-v1"
PREVIEW_SEQUENCE_IDS = (
    "sequence_c2dffc899e9b325631fcc87990ac538c",
    "sequence_f8c12f02790ecb8ddcb24a2972113256",
)
SUMMARY_FIELDS = ("premultiplied_rgba_mae", "alpha_iou", "temporal_delta_mae")


def main() -> None:
    if OUTPUT.exists() or PREVIEW_ROOT.exists():
        raise FileExistsError("Refusing to replace broad codec-floor artifacts")
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    guard.require_capacity(256 * 1024**2, label="broad MUGEN codec-floor audit")
    corpus = load_latent_motion_training_corpus(MANIFEST, verify_hashes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = _load_frozen_decoder(torch, corpus, device=device)
    decoded_by_index: dict[int, np.ndarray] = {}
    with torch.no_grad():
        for start in range(0, len(corpus.test_indices), 8):
            indices = corpus.test_indices[start : start + 8]
            latent = torch.from_numpy(corpus.target_latents[list(indices)]).to(
                device=device, dtype=torch.float32
            )
            flat = latent.reshape(-1, 8, 64, 64)
            decoded = (
                decoder.decode(flat)
                .clamp(0, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .cpu()
                .numpy()
                .transpose(0, 2, 3, 1)
                .reshape(len(indices), 8, 128, 128, 4)
            )
            for offset, row_index in enumerate(indices):
                decoded_by_index[row_index] = np.ascontiguousarray(decoded[offset])

    metrics = []
    per_verb: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row_index in corpus.test_indices:
        row = corpus.rows[row_index]
        metric = asdict(
            compare_matched_sequences(
                _images(decoded_by_index[row_index]),
                _images(corpus.target_rgba[row_index]),
                loop_mode=row.loop_mode,
            )
        )
        record = {
            "identity_id": row.identity_id,
            "metrics": metric,
            "sequence_id": row.sequence_id,
            "verb": row.verb,
        }
        metrics.append(record)
        per_verb[row.verb].append(metric)

    PREVIEW_ROOT.mkdir(parents=True, exist_ok=False)
    previews = []
    row_by_sequence = {corpus.rows[index].sequence_id: index for index in corpus.test_indices}
    for sequence_id in PREVIEW_SEQUENCE_IDS:
        row_index = row_by_sequence[sequence_id]
        row = corpus.rows[row_index]
        target = export_rgba_clip_preview(
            corpus.target_rgba[row_index],
            PREVIEW_ROOT,
            artifact_stem=f"{row.verb}-{sequence_id}-target",
            duration_ms=row.duration_ms,
            loop_mode=row.loop_mode,
            integer_scale=2,
            preserve_frame_slots=True,
            disk_guard=guard,
        )
        codec = export_rgba_clip_preview(
            decoded_by_index[row_index],
            PREVIEW_ROOT,
            artifact_stem=f"{row.verb}-{sequence_id}-codec",
            duration_ms=row.duration_ms,
            loop_mode=row.loop_mode,
            integer_scale=2,
            preserve_frame_slots=True,
            disk_guard=guard,
        )
        previews.append(
            {
                "codec_animated_sha256": codec.animated_png_sha256,
                "codec_sheet_sha256": codec.contact_sheet_sha256,
                "sequence_id": sequence_id,
                "target_animated_sha256": target.animated_png_sha256,
                "target_sheet_sha256": target.contact_sheet_sha256,
                "verb": row.verb,
            }
        )

    manifest_bytes = MANIFEST.read_bytes()
    artifact = {
        "artifact_kind": "mugen_primary_motion_broad_codec_floor_test_audit",
        "autoencoder_checkpoint_sha256": corpus.contract["autoencoder_checkpoint_sha256"],
        "claim": "frozen-codec reconstruction on all untouched identity-disjoint test clips",
        "corpus": corpus.contract,
        "metrics": metrics,
        "previews": previews,
        "schema_version": 1,
        "source_manifest_file_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "summary": _summary([record["metrics"] for record in metrics]),
        "test_clips": len(metrics),
        "verb_summary": {
            verb: {"clips": len(rows), **_summary(rows)}
            for verb, rows in sorted(per_verb.items(), key=lambda item: item[0].encode())
        },
    }
    payload = canonical_json_bytes(artifact)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {
                "path": str(OUTPUT),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "summary": artifact["summary"],
            },
            sort_keys=True,
        )
    )


def _summary(rows: list[dict[str, object]]) -> dict[str, float]:
    return {field: sum(float(row[field]) for row in rows) / len(rows) for field in SUMMARY_FIELDS}


if __name__ == "__main__":
    main()
