from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.decode import (
    GlobalPaletteDecodeConfig,
    HardAlphaDecodeConfig,
    global_palette_decode_rgba,
    hard_alpha_decode_rgba,
)
from spritelab.decode_bundle import (
    DecodeBundleArtifactRef,
    DecodeBundleClipRef,
    export_decode_preview_bundle,
)
from spritelab.storage import HashMismatch


def _write_npy(path: Path, rgba: np.ndarray) -> Path:
    with path.open("wb") as handle:
        np.save(handle, rgba, allow_pickle=False)
    return path


def _write_json(path: Path, document: dict[str, object]) -> Path:
    path.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clip(frame_count: int = 2) -> np.ndarray:
    rgba = np.zeros((frame_count, 2, 3, 4), dtype=np.uint8)
    rgba[:, 0, 0] = (250, 10, 20, 230)
    rgba[:, 0, 1] = (15, 245, 30, 140)
    rgba[:, 1, 1] = (20, 30, 240, 255)
    rgba[1:, 1, 2] = (240, 220, 10, 250)
    return rgba


def _artifact(path: Path, artifact_id: str) -> DecodeBundleArtifactRef:
    return DecodeBundleArtifactRef(
        artifact_id=artifact_id,
        path=path,
        file_sha256=_sha256(path),
    )


def _provenance(
    tmp_path: Path,
    *,
    threshold: int = 128,
    palette_sizes: list[int] | None = None,
) -> tuple[
    DecodeBundleArtifactRef,
    tuple[DecodeBundleArtifactRef, ...],
    tuple[DecodeBundleArtifactRef, ...],
]:
    sizes = palette_sizes or [2, 8]
    source_report = _write_json(
        tmp_path / "source-report.json",
        {"artifact_kind": "test_inference_report", "schema_version": 1},
    )
    alpha = _write_json(
        tmp_path / "hard-alpha-calibration.json",
        {
            "artifact_kind": "hard_alpha_threshold_calibration",
            "estimate": {"held_out": False, "kind": "training_target_estimate"},
            "thresholds": [threshold],
        },
    )
    palette = _write_json(
        tmp_path / "palette-calibration.json",
        {
            "artifact_kind": "clip_global_palette_size_calibration",
            "estimate": {"held_out": False, "kind": "training_target_estimate"},
            "palette_sizes": sizes,
            "parameters": {"alpha_threshold": threshold},
        },
    )
    return (
        _artifact(source_report, "source-report"),
        (_artifact(alpha, "hard-alpha-calibration"),),
        (_artifact(palette, "palette-calibration"),),
    )


def _clip_ref(
    path: Path,
    sample_id: str,
    *,
    duration_ms: tuple[float, ...] = (100.0, 250.0),
    loop_mode: str = "loop",
) -> DecodeBundleClipRef:
    return DecodeBundleClipRef(
        sample_id=sample_id,
        source_path=path,
        source_file_sha256=_sha256(path),
        duration_ms=duration_ms,
        loop_mode=loop_mode,
    )


def test_bundle_publishes_exact_derivatives_previews_and_checksum_index(
    tmp_path: Path,
) -> None:
    source_a = _write_npy(tmp_path / "a.npy", _clip())
    source_b_array = np.flip(_clip(), axis=2).copy()
    source_b = _write_npy(tmp_path / "b.npy", source_b_array)
    source_a_before = source_a.read_bytes()
    source_b_before = source_b.read_bytes()
    source_report, alpha_calibrations, palette_calibrations = _provenance(tmp_path)
    output = tmp_path / "derived-bundle"

    result = export_decode_preview_bundle(
        [
            _clip_ref(source_a, "sprite-a"),
            _clip_ref(
                source_b,
                "sprite-b",
                duration_ms=(75.0, 125.0),
                loop_mode="one_shot",
            ),
        ],
        output,
        hard_alpha_threshold=128,
        palette_sizes=[8, 2],
        source_report=source_report,
        hard_alpha_calibrations=alpha_calibrations,
        palette_calibrations=palette_calibrations,
        integer_scale=2,
    )

    assert result.bundle_path == output.resolve()
    assert result.clip_count == 2
    assert result.palette_sizes == (2, 8)
    assert result.payload_file_count == 30
    assert source_a.read_bytes() == source_a_before
    assert source_b.read_bytes() == source_b_before

    hard_alpha = np.load(output / "hard-alpha" / "sprite-a.npy", allow_pickle=False)
    palette_two = np.load(output / "palette-2" / "sprite-a.npy", allow_pickle=False)
    palette_eight = np.load(output / "palette-8" / "sprite-a.npy", allow_pickle=False)
    assert np.array_equal(
        hard_alpha,
        hard_alpha_decode_rgba(
            _clip(),
            config=HardAlphaDecodeConfig(threshold=128),
        ),
    )
    assert np.array_equal(
        palette_two,
        global_palette_decode_rgba(
            _clip(),
            config=GlobalPaletteDecodeConfig(alpha_threshold=128, maximum_colors=2),
        ),
    )
    assert np.array_equal(
        palette_eight,
        global_palette_decode_rgba(
            _clip(),
            config=GlobalPaletteDecodeConfig(alpha_threshold=128, maximum_colors=8),
        ),
    )

    decode_metadata_path = output / "palette-2" / "sprite-a.npy.decode.json"
    decode_metadata = json.loads(decode_metadata_path.read_text(encoding="utf-8"))
    assert decode_metadata["derivative_status"] == {
        "canonical_model_output": False,
        "display_only": True,
        "evaluation_authority": "raw source array",
        "raw_source_mutated": False,
    }
    assert decode_metadata["operation"]["reference_or_target_palette_used"] is False
    assert decode_metadata["source"]["path"] == str(source_a.resolve())
    assert decode_metadata["source"]["file_sha256"] == _sha256(source_a)
    assert decode_metadata["decoded"]["path"] == str(
        (output / "palette-2" / "sprite-a.npy").resolve()
    )
    assert ".partial" not in decode_metadata_path.read_text(encoding="utf-8")

    preview_metadata_path = output / "palette-2" / "sprite-a-preview.json"
    preview_metadata = json.loads(preview_metadata_path.read_text(encoding="utf-8"))
    assert preview_metadata["source_sample_path"] == str(
        (output / "palette-2" / "sprite-a.npy").resolve()
    )
    assert preview_metadata["source_report_sha256"] == source_report.file_sha256
    with Image.open(output / "palette-2" / "sprite-a-animated.png") as animation:
        assert animation.n_frames == 2
        assert animation.size == (6, 4)
        animation.seek(0)
        assert animation.info["duration"] == pytest.approx(100)
        animation.seek(1)
        assert animation.info["duration"] == pytest.approx(250)
    with Image.open(output / "palette-2" / "sprite-a-sheet.png") as sheet:
        assert sheet.size == (12, 4)

    index_payload = result.index_path.read_bytes()
    index = json.loads(index_payload)
    assert index_payload == (
        json.dumps(index, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")
    assert result.index_sha256 == hashlib.sha256(index_payload).hexdigest()
    assert index["inputs"]["clips"][0]["sample_id"] == "sprite-a"
    assert index["inputs"]["clips"][1]["sample_id"] == "sprite-b"
    assert index["parameters"]["palette_sizes"] == [2, 8]
    assert index["derivative_policy"]["canonical_raw_outputs_mutated"] is False
    assert index["derivative_policy"]["reference_or_target_palette_used"] is False
    assert index["provenance"]["hard_alpha_calibrations"][0]["file_sha256"] == (
        alpha_calibrations[0].file_sha256
    )
    assert index["provenance"]["palette_calibrations"][0]["file_sha256"] == (
        palette_calibrations[0].file_sha256
    )
    indexed_paths = [record["path"] for record in index["payload_files"]]
    assert indexed_paths == sorted(indexed_paths)
    assert "bundle-index.json" not in indexed_paths
    for record in index["payload_files"]:
        path = output / record["path"]
        assert record["size_bytes"] == path.stat().st_size
        assert record["file_sha256"] == _sha256(path)


def test_bundle_is_no_clobber_and_rejects_hash_mismatch(tmp_path: Path) -> None:
    source = _write_npy(tmp_path / "source.npy", _clip())
    source_report, alpha_calibrations, palette_calibrations = _provenance(tmp_path)
    output = tmp_path / "bundle"
    arguments = {
        "hard_alpha_threshold": 128,
        "palette_sizes": [2],
        "source_report": source_report,
        "hard_alpha_calibrations": alpha_calibrations,
        "palette_calibrations": palette_calibrations,
    }

    bad_ref = DecodeBundleClipRef(
        sample_id="sprite",
        source_path=source,
        source_file_sha256="0" * 64,
        duration_ms=(100.0, 100.0),
        loop_mode="loop",
    )
    with pytest.raises(HashMismatch, match="expected SHA-256"):
        export_decode_preview_bundle([bad_ref], output, **arguments)
    assert not output.exists()

    reference = _clip_ref(source, "sprite", duration_ms=(100.0, 100.0))
    export_decode_preview_bundle([reference], output, **arguments)
    index_before = (output / "bundle-index.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing"):
        export_decode_preview_bundle([reference], output, **arguments)
    assert (output / "bundle-index.json").read_bytes() == index_before


def test_bundle_failure_never_publishes_partial_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_npy(tmp_path / "source.npy", _clip())
    source_report, alpha_calibrations, palette_calibrations = _provenance(tmp_path)
    output = tmp_path / "bundle"

    def fail_preview(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected preview failure")

    monkeypatch.setattr("spritelab.decode_bundle.export_rgba_clip_preview", fail_preview)
    with pytest.raises(RuntimeError, match="injected"):
        export_decode_preview_bundle(
            [_clip_ref(source, "sprite", duration_ms=(100.0, 100.0))],
            output,
            hard_alpha_threshold=128,
            palette_sizes=[2],
            source_report=source_report,
            hard_alpha_calibrations=alpha_calibrations,
            palette_calibrations=palette_calibrations,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".bundle.*.partial")) == []


@pytest.mark.parametrize(
    ("sample_id", "duration_ms", "error"),
    [
        ("../escape", (100.0, 100.0), ValueError),
        ("sprite", (100.0,), ValueError),
        ("sprite", [100.0, 100.0], TypeError),
    ],
)
def test_bundle_rejects_unsafe_ids_or_implicit_timing(
    tmp_path: Path,
    sample_id: str,
    duration_ms: tuple[float, ...],
    error: type[Exception],
) -> None:
    source = _write_npy(tmp_path / "source.npy", _clip())
    source_report, alpha_calibrations, palette_calibrations = _provenance(tmp_path)
    reference = DecodeBundleClipRef(
        sample_id=sample_id,
        source_path=source,
        source_file_sha256=_sha256(source),
        duration_ms=duration_ms,
        loop_mode="loop",
    )

    with pytest.raises(error):
        export_decode_preview_bundle(
            [reference],
            tmp_path / "bundle",
            hard_alpha_threshold=128,
            palette_sizes=[2],
            source_report=source_report,
            hard_alpha_calibrations=alpha_calibrations,
            palette_calibrations=palette_calibrations,
        )


def test_bundle_requires_calibration_coverage_and_matching_alpha(tmp_path: Path) -> None:
    source = _write_npy(tmp_path / "source.npy", _clip())
    source_report, alpha_calibrations, palette_calibrations = _provenance(
        tmp_path,
        threshold=96,
        palette_sizes=[8],
    )

    with pytest.raises(ValueError, match="hard_alpha_threshold 128 is absent"):
        export_decode_preview_bundle(
            [_clip_ref(source, "sprite", duration_ms=(100.0, 100.0))],
            tmp_path / "bundle",
            hard_alpha_threshold=128,
            palette_sizes=[8],
            source_report=source_report,
            hard_alpha_calibrations=alpha_calibrations,
            palette_calibrations=palette_calibrations,
        )


def test_bundle_calls_disk_guard_for_staged_payloads(tmp_path: Path) -> None:
    source = _write_npy(tmp_path / "source.npy", _clip())
    source_report, alpha_calibrations, palette_calibrations = _provenance(tmp_path)

    class RecordingGuard:
        def __init__(self) -> None:
            self.calls: list[tuple[int, str]] = []

        def require_capacity(self, additional_bytes: int = 0, *, label: str = "write") -> None:
            self.calls.append((additional_bytes, label))

    guard = RecordingGuard()
    export_decode_preview_bundle(
        [_clip_ref(source, "sprite", duration_ms=(100.0, 100.0))],
        tmp_path / "bundle",
        hard_alpha_threshold=128,
        palette_sizes=[2],
        source_report=source_report,
        hard_alpha_calibrations=alpha_calibrations,
        palette_calibrations=palette_calibrations,
        disk_guard=guard,  # type: ignore[arg-type]
    )

    labels = [label for _additional_bytes, label in guard.calls]
    assert labels[0] == "decode preview bundle staging"
    assert "hard-alpha decoded array" in labels
    assert "animated sprite preview" in labels
    assert "decode preview bundle checksum index" in labels
