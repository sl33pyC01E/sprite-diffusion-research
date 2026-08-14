from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from spritelab.mugen_still_eligibility import (
    SubjectFramePixelGateConfig,
    frame_pixel_metrics,
    merge_subject_frame_eligibility,
    parse_subject_frame_vlm_response,
    subject_contact_sheet,
    subject_frame_vlm_request,
)


def _frame(*, color: tuple[int, int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    value = np.zeros((16, 16, 4), dtype=np.uint8)
    left, top, right, bottom = box
    value[top:bottom, left:right, :3] = color
    value[top:bottom, left:right, 3] = 255
    return value


def test_pixel_gate_accepts_same_subject_and_rejects_effect_palette() -> None:
    reference = _frame(color=(200, 40, 20), box=(5, 3, 11, 15))
    pose = _frame(color=(200, 40, 20), box=(4, 4, 12, 15))
    effect = _frame(color=(20, 100, 255), box=(1, 10, 15, 14))

    accepted = frame_pixel_metrics(pose, reference)
    rejected = frame_pixel_metrics(effect, reference)

    assert accepted["passes_pixel_gate"] is True
    assert accepted["candidate_palette_coverage"] == 1
    assert rejected["passes_pixel_gate"] is False
    assert rejected["palette_histogram_intersection"] == 0


def test_gate_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="odd"):
        SubjectFramePixelGateConfig(dilation_size=4)
    with pytest.raises(ValueError, match="minimum_anchored_overlap"):
        SubjectFramePixelGateConfig(minimum_anchored_overlap=1.1)
    with pytest.raises(ValueError, match="maximum_occupancy_ratio"):
        SubjectFramePixelGateConfig(minimum_occupancy_ratio=2, maximum_occupancy_ratio=1)


def test_contact_sheet_and_request_have_fixed_panel_contract() -> None:
    reference = _frame(color=(200, 40, 20), box=(5, 3, 11, 15))
    clip = np.stack([reference] * 8)
    payload = subject_contact_sheet(reference, clip)
    image = Image.open(__import__("io").BytesIO(payload))
    request = subject_frame_vlm_request(model="qwen", sheet_png=payload)

    assert image.size == (768, 768)
    assert request["temperature"] == 0
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert request["response_format"]["json_schema"]["strict"] is True


def test_vlm_response_is_strict_and_disjoint() -> None:
    value = parse_subject_frame_vlm_response(
        '```json\n{"same_primary_subject_indices":[7,0],"ambiguous_indices":[3]}\n```'
    )
    assert value == {
        "ambiguous_indices": [3],
        "same_primary_subject_indices": [0, 7],
    }
    with pytest.raises(ValueError, match="overlap"):
        parse_subject_frame_vlm_response(
            json.dumps(
                {
                    "same_primary_subject_indices": [1],
                    "ambiguous_indices": [1],
                }
            )
        )


def test_merge_intersects_mixed_and_fails_closed() -> None:
    pixel = {
        "artifact_kind": "mugen_subject_bearing_frame_pixel_gate",
        "counts": {"sequences": 3},
        "records": [
            {
                "identity_id": "a",
                "pixel_gate_pass_indices": list(range(8)),
                "pixel_gate_status": "all_pass",
                "sequence_id": "all-pass",
                "split": "train",
            },
            {
                "identity_id": "b",
                "pixel_gate_pass_indices": [1, 2, 3],
                "pixel_gate_status": "mixed",
                "sequence_id": "mixed",
                "split": "train",
            },
            {
                "identity_id": "c",
                "pixel_gate_pass_indices": [],
                "pixel_gate_status": "all_fail",
                "sequence_id": "all-fail",
                "split": "validation",
            },
        ],
        "source": {
            "caption_manifest_file_sha256": "a" * 64,
            "materialization_file_sha256": "b" * 64,
        },
    }
    merged = merge_subject_frame_eligibility(
        pixel,
        [
            {
                "ambiguous_indices": [2],
                "same_primary_subject_indices": [0, 1, 4],
                "sequence_id": "mixed",
            }
        ],
    )
    by_id = {record["sequence_id"]: record for record in merged["records"]}
    assert by_id["all-pass"]["eligible_frame_indices"] == list(range(8))
    assert by_id["mixed"]["eligible_frame_indices"] == [1]
    assert by_id["all-fail"]["eligible_frame_indices"] == []
    assert merged["counts"] == {
        "ambiguous_frames_excluded": 1,
        "eligible_frames": 9,
        "excluded_frames": 15,
        "excluded_sequences": 1,
        "retained_sequences": 2,
        "sequences": 3,
    }
    with pytest.raises(ValueError, match="closure"):
        merge_subject_frame_eligibility(pixel, [])
