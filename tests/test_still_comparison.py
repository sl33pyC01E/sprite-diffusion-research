from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.sprite_postprocess import SpriteDisplayDecodeConfig
from spritelab.still_comparison import (
    StillComparisonError,
    _eligible_indices,
    _wrap,
    build_sd_lora_ablation_comparison,
)
from spritelab.storage import DiskGuard


def test_comparison_prompt_wrap_preserves_words() -> None:
    value = "detailed pixel art sprite with silver armor and a red scarf"
    lines = _wrap(value, 20)
    assert " ".join(lines) == value
    assert all(len(line) <= 20 for line in lines)


def test_subject_bearing_target_indices_fail_closed() -> None:
    assert _eligible_indices({"eligible_frame_indices": [2, 5]}) == [2, 5]
    assert _eligible_indices({}) == [0]
    with pytest.raises(StillComparisonError, match="eligible"):
        _eligible_indices({"eligible_frame_indices": []})
    with pytest.raises(StillComparisonError, match="eligible"):
        _eligible_indices({"eligible_frame_indices": [2, 2]})


def test_ablation_comparison_binds_exact_sequence_and_writes_display_gallery(
    tmp_path: Path,
) -> None:
    materialized = tmp_path / "materialized"
    clips = materialized / "clips"
    clips.mkdir(parents=True)
    rgba = np.zeros((8, 128, 128, 4), dtype=np.uint8)
    rgba[:, 40:88, 48:80] = (230, 60, 20, 255)
    clip_path = clips / "sequence_exact.npy"
    np.save(clip_path, rgba, allow_pickle=False)
    materialization_path = materialized / "materialization.json"
    materialization_path.write_bytes(b"{}\n")
    prompt = "pixel art orange fighter blocking"
    plan = {
        "counts": {"sequences": 1},
        "records": [
            {
                "identity_id": "fighter_one",
                "prompt": prompt,
                "sequence_id": "sequence_exact",
                "target": {
                    "array_content_sha256": "0" * 64,
                    "eligible_frame_indices": [3],
                    "file_sha256": _file_sha256(clip_path),
                    "relative_path": "clips/sequence_exact.npy",
                },
            }
        ],
        "source": {
            "materialization_file_sha256": _file_sha256(materialization_path),
            "materialization_path": str(materialization_path),
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(_canonical_json(plan))
    reports = []
    for index, label in enumerate(("dirty", "clean")):
        inference = tmp_path / label
        inference.mkdir()
        rgb = np.full((128, 128, 3), 135, dtype=np.uint8)
        rgb[40:88, 48:80] = (220, 50 + index * 10, 20)
        image_path = inference / "sample.png"
        Image.fromarray(rgb, mode="RGB").save(image_path)
        report = {
            "artifact_kind": "mugen_sd14_attention_lora_rgb_inference",
            "noise_batch_sha256": "1" * 64,
            "samples": [
                {
                    "downsample_128": {
                        "file_sha256": _file_sha256(image_path),
                        "path": image_path.name,
                    },
                    "prompt": prompt,
                }
            ],
        }
        report_path = inference / "inference-report.json"
        report_path.write_bytes(_canonical_json(report))
        reports.append((label, report_path, _file_sha256(report_path)))

    report_path, report_sha256 = build_sd_lora_ablation_comparison(
        reports,
        plan_path,
        tmp_path / "comparison",
        target_sequence_ids=["sequence_exact"],
        display_decode_config=SpriteDisplayDecodeConfig(
            background_rgb_distance=0,
            minimum_component_pixels=4,
            palette_colors=8,
        ),
        disk_guard=DiskGuard(tmp_path, 0),
    )

    assert _file_sha256(report_path) == report_sha256
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["target_selection"] == "explicit_sequence_id"
    assert report["rows"][0]["target_frame_index"] == 3
    assert report["display_decode"]["claim"].startswith("exact target alpha")
    assert (report_path.parent / report["display_decode"]["gallery"]["path"]).is_file()

    with pytest.raises(StillComparisonError, match="no target uses"):
        build_sd_lora_ablation_comparison(
            reports,
            plan_path,
            tmp_path / "wrong-sequence",
            target_sequence_ids=["sequence_missing"],
            disk_guard=DiskGuard(tmp_path, 0),
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
