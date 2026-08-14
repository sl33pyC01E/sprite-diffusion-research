from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.sprite_postprocess import (
    SpriteDisplayDecodeConfig,
    composite_rgba_on_checkerboard,
    decode_generated_rgb_sprite,
    export_inference_sprite_display_bundle,
)
from spritelab.storage import DiskGuard


def test_decode_generated_rgb_sprite_extracts_border_connected_background() -> None:
    rgb = np.full((16, 16, 3), 140, dtype=np.uint8)
    rgb[4:12, 5:11] = (220, 40, 20)

    rgba, metadata = decode_generated_rgb_sprite(
        rgb,
        config=SpriteDisplayDecodeConfig(
            background_rgb_distance=2,
            minimum_component_pixels=4,
            palette_colors=8,
        ),
    )

    assert rgba.dtype == np.uint8
    assert rgba.shape == (16, 16, 4)
    assert np.all(rgba[4:12, 5:11, 3] == 255)
    assert np.all(rgba[:4, :, 3] == 0)
    assert metadata["foreground_pixel_count"] == 48
    assert metadata["reference_or_target_pixels_used"] is False


def test_decode_keeps_enclosed_background_colored_pixels_as_subject() -> None:
    rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
    rgb[2:14, 2:14] = (20, 80, 220)
    rgb[6:10, 6:10] = 100

    rgba, _ = decode_generated_rgb_sprite(
        rgb,
        config=SpriteDisplayDecodeConfig(
            background_rgb_distance=0,
            minimum_component_pixels=1,
            palette_colors=8,
        ),
    )

    assert np.all(rgba[6:10, 6:10, 3] == 255)


def test_decode_removes_small_disconnected_foreground_speckles() -> None:
    rgb = np.full((16, 16, 3), 120, dtype=np.uint8)
    rgb[4:12, 4:12] = (20, 180, 80)
    rgb[1, 1] = (255, 0, 255)

    rgba, _ = decode_generated_rgb_sprite(
        rgb,
        config=SpriteDisplayDecodeConfig(
            background_rgb_distance=0,
            minimum_component_pixels=4,
            palette_colors=8,
        ),
    )

    assert rgba[1, 1, 3] == 0
    assert rgba[5, 5, 3] == 255


@pytest.mark.parametrize(
    "arguments",
    [
        {"background_rgb_distance": -1},
        {"background_rgb_distance": float("nan")},
        {"minimum_component_pixels": 0},
        {"palette_colors": 1},
        {"palette_colors": 257},
    ],
)
def test_decode_config_rejects_invalid_values(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SpriteDisplayDecodeConfig(**arguments)


def test_checkerboard_composite_preserves_opaque_and_reveals_transparent() -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[1:3, 1:3] = (240, 20, 10, 255)

    composite = composite_rgba_on_checkerboard(rgba, tile_pixels=1)

    assert np.array_equal(composite[1, 1], np.array([240, 20, 10], dtype=np.uint8))
    assert tuple(composite[0, 0]) == (96, 96, 96)
    assert tuple(composite[0, 1]) == (128, 128, 128)


def test_export_display_bundle_hash_verifies_and_refuses_clobber(tmp_path: Path) -> None:
    rgb = np.full((16, 16, 3), 130, dtype=np.uint8)
    rgb[3:13, 5:11] = (230, 60, 30)
    source_path = tmp_path / "sample.png"
    Image.fromarray(rgb, mode="RGB").save(source_path)
    source_sha256 = _file_sha256(source_path)
    report = {
        "artifact_kind": "mugen_sd14_attention_lora_rgb_inference",
        "samples": [
            {
                "downsample_128": {
                    "file_sha256": source_sha256,
                    "path": source_path.name,
                },
                "prompt": "pixel art orange fighter idle",
            }
        ],
    }
    report_path = tmp_path / "inference-report.json"
    report_path.write_bytes(_canonical_json(report))
    report_sha256 = _file_sha256(report_path)
    output = tmp_path / "decoded"

    manifest_path, manifest_sha256 = export_inference_sprite_display_bundle(
        report_path,
        output,
        expected_inference_report_sha256=report_sha256,
        config=SpriteDisplayDecodeConfig(
            background_rgb_distance=0,
            minimum_component_pixels=4,
            palette_colors=8,
        ),
        disk_guard=DiskGuard(tmp_path, 0),
    )

    assert _file_sha256(manifest_path) == manifest_sha256
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["record_count"] == 1
    assert manifest["records"][0]["decode"]["foreground_pixel_count"] == 60
    transparent_path = output / manifest["records"][0]["transparent_rgba"]["path"]
    with Image.open(transparent_path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    assert np.all(rgba[3:13, 5:11, 3] == 255)
    assert np.all(rgba[:3, :, 3] == 0)
    with pytest.raises(FileExistsError):
        export_inference_sprite_display_bundle(
            report_path,
            output,
            expected_inference_report_sha256=report_sha256,
            disk_guard=DiskGuard(tmp_path, 0),
        )

    report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        export_inference_sprite_display_bundle(
            report_path,
            tmp_path / "tampered-output",
            expected_inference_report_sha256=report_sha256,
            disk_guard=DiskGuard(tmp_path, 0),
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
