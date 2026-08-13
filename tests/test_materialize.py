from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spritelab.dataset import (
    DatasetManifest,
    SequenceSample,
    SplitAssignment,
    SplitPolicy,
    coverage_report,
)
from spritelab.materialize import (
    PIXEL_TRANSFORM_OP,
    PIXEL_TRANSFORM_SCHEMA,
    ExistingOutputError,
    NoLosslessBucketError,
    SnapshotHashMismatch,
    SourceBlobHashMismatch,
    UnsupportedPixelTransformError,
    UnsupportedSheetCoordinatesError,
    materialize_snapshot,
)
from spritelab.snapshot import (
    TEMPORAL_KNOWN_CONTRACT,
    SnapshotArtifact,
    SnapshotFilters,
    write_snapshot,
)

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
YELLOW = (255, 255, 0, 255)


def _image_bytes(
    color: tuple[int, int, int, int],
    *,
    size: tuple[int, int] = (3, 2),
) -> bytes:
    output = BytesIO()
    Image.new("RGBA", size, color).save(output, "PNG")
    return output.getvalue()


def _gif_bytes(*colors: tuple[int, int, int, int]) -> bytes:
    frames = [Image.new("RGBA", (3, 2), color) for color in colors]
    output = BytesIO()
    frames[0].save(
        output,
        "GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[20 + 10 * index for index in range(len(frames))],
        loop=0,
        disposal=[1] * len(frames),
        optimize=False,
    )
    return output.getvalue()


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def _pixel_transform() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": PIXEL_TRANSFORM_SCHEMA,
        "op": PIXEL_TRANSFORM_OP,
        "rgb": [255, 0, 255],
        "evidence": [
            {
                "member_path": "fixture/src/core/color.c",
                "sha256": "a" * 64,
                "line_numbers": [190],
                "scope": "engine_exact_color_key_predicate",
                "claim": "exact_uint8_rgb_is_transparent",
            }
        ],
    }
    payload["transform_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _blob(root: Path, payload: bytes, name: str) -> tuple[str, Path, int]:
    digest = hashlib.sha256(payload).hexdigest()
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return digest, path, len(payload)


def _sample(
    sequence_id: str,
    blobs: tuple[tuple[str, Path, int], ...],
    frames: list[dict[str, object]],
    *,
    identity: str = "fox_red",
) -> SequenceSample:
    return SequenceSample(
        sequence_id=sequence_id,
        identity_id=f"entity-{identity}",
        source_id="fixture-source",
        source_pack_id="fixture-pack",
        entity_class="animal",
        action="run",
        view="side",
        direction="right",
        loop_mode="loop",
        frame_count=len(frames),
        source_blob_sha256=tuple(sorted(blob[0] for blob in blobs)),
        quality_tier="F0",
        metadata={
            "archive_occurrences": [],
            "blob_records": [
                {
                    "mime_type": "image/gif" if path.suffix == ".gif" else "image/png",
                    "sha256": digest,
                    "size_bytes": size,
                    "storage_path": str(path),
                }
                for digest, path, size in sorted(blobs)
            ],
            "frame_provenance": frames,
            "item": {"id": "fixture-pack"},
            "item_blob_occurrence_ids": ["item-blob-1"],
            "retrieval_ids": ["retrieval-1"],
            "rights_observation_ids": ["rights-1"],
            "sequence_metadata": {},
            "sequence_source_keys": [
                {
                    "external_sequence_key": sequence_id,
                    "source_id": "fixture-source",
                }
            ],
            "source": {"id": "fixture-source", "name": "Fixture"},
            "source_ids": ["fixture-source"],
            "subjects": [
                {
                    "entity_id": f"entity-{identity}",
                    "external_identity_key": identity,
                    "role": "primary",
                }
            ],
            "temporal_evidence": {"known": True},
        },
    )


def _write_snapshot(path: Path, samples: tuple[SequenceSample, ...]) -> Path:
    assignments = tuple(
        SplitAssignment(
            sequence_id=sample.sequence_id,
            split="train",
            component_id=f"component-{index}",
        )
        for index, sample in enumerate(samples)
    )
    manifest = DatasetManifest(
        schema_version=1,
        policy=SplitPolicy(seed="materialize-fixture"),
        samples=samples,
        assignments=assignments,
    )
    snapshot = SnapshotArtifact(
        schema_version=1,
        filters=SnapshotFilters(minimum_frame_count=1),
        temporal_known_contract=TEMPORAL_KNOWN_CONTRACT,
        index_schema_versions=(1,),
        manifest=manifest,
        coverage=coverage_report(manifest),
        timing_counts={"known": len(samples), "pose_only": 0},
    )
    return write_snapshot(snapshot, path)


def _frame(
    ordinal: int,
    digest: str,
    source_index: int,
    duration: int,
    phase: float,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "direction": "right",
        "duration_ms": duration,
        "metadata": metadata or {},
        "ordinal": ordinal,
        "phase": phase,
        "source_blob_sha256": digest,
        "source_frame_index": source_index,
        "view": "side",
    }


def test_materializes_exact_repeated_frames_across_multiple_carriers(tmp_path: Path) -> None:
    gif = _blob(tmp_path / "blobs", _gif_bytes(RED, GREEN, BLUE), "motion.gif")
    png = _blob(tmp_path / "blobs", _image_bytes(YELLOW), "insert.png")
    frames = [
        _frame(0, gif[0], 2, 60, 0.0),
        _frame(1, png[0], 0, 40, 0.25),
        _frame(2, gif[0], 0, 50, 0.5),
        _frame(3, gif[0], 2, 70, 0.75),
    ]
    snapshot = _write_snapshot(
        tmp_path / "snapshot.json", (_sample("fox-run", (gif, png), frames),)
    )

    result = materialize_snapshot(snapshot, tmp_path / "output", bucket_sizes=(4, 8))

    array = np.load(result.clip_paths[0], allow_pickle=False)
    assert array.dtype == np.uint8
    assert array.shape == (4, 4, 4, 4)
    assert [tuple(array[index, 2, 0]) for index in range(4)] == [BLUE, YELLOW, RED, BLUE]
    record = result.manifest["sequences"][0]
    assert record["timing"] == {
        "duration_ms": (60, 40, 50, 70),
        "phase": (0.0, 0.25, 0.5, 0.75),
        "temporal_evidence": {"known": True},
        "total_duration_ms": 220.0,
    }
    assert [row["source_frame_index"] for row in record["frame_provenance"]] == [2, 0, 0, 2]
    assert (
        record["frame_provenance"][0]["source_frame_pixel_sha256"]
        == record["frame_provenance"][3]["source_frame_pixel_sha256"]
    )
    assert record["caption"]["description"] == "fox red"
    assert record["caption"]["description_basis"] == "external_identity_key"
    assert record["split"] == "train"
    assert record["target_bucket"] == (4, 4)
    assert record["normalization"]["transform"]["integer_scale"] == 1
    assert (
        hashlib.sha256(result.clip_paths[0].read_bytes()).hexdigest()
        == record["output"]["file_sha256"]
    )


def test_chooses_smallest_fitting_bucket_and_never_downsamples(tmp_path: Path) -> None:
    png = _blob(tmp_path / "blobs", _image_bytes(RED, size=(5, 3)), "wide.png")
    sample = _sample("wide", (png,), [_frame(0, png[0], 0, 100, 0.0)])
    snapshot = _write_snapshot(tmp_path / "snapshot.json", (sample,))

    result = materialize_snapshot(
        snapshot,
        tmp_path / "fit",
        bucket_sizes=(16, 4, 8, 8),
    )

    record = result.manifest["sequences"][0]
    assert record["target_bucket"] == (8, 8)
    assert record["normalization"]["transform"]["aligned_size"] == (5, 3)
    assert record["normalization"]["transform"]["integer_scale"] == 1
    assert np.load(result.clip_paths[0], allow_pickle=False).shape == (1, 8, 8, 4)

    with pytest.raises(NoLosslessBucketError, match="without downsampling"):
        materialize_snapshot(snapshot, tmp_path / "too-small", bucket_sizes=(4,))
    assert not (tmp_path / "too-small" / "materialization.json").exists()


def test_rejects_snapshot_and_source_blob_hash_mismatches(tmp_path: Path) -> None:
    png = _blob(tmp_path / "blobs", _image_bytes(GREEN), "green.png")
    sample = _sample("green", (png,), [_frame(0, png[0], 0, 100, 0.0)])
    snapshot = _write_snapshot(tmp_path / "snapshot.json", (sample,))

    payload = json.loads(snapshot.read_text(encoding="utf-8"))
    payload["manifest"]["samples"][0]["action"] = "attack"
    corrupted_snapshot = tmp_path / "snapshot-bad-hash.json"
    corrupted_snapshot.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SnapshotHashMismatch, match="manifest SHA-256 mismatch"):
        materialize_snapshot(corrupted_snapshot, tmp_path / "snapshot-failure")

    original = png[1].read_bytes()
    png[1].write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(SourceBlobHashMismatch, match="SHA-256 mismatch"):
        materialize_snapshot(snapshot, tmp_path / "blob-failure")
    assert not (tmp_path / "blob-failure" / "materialization.json").exists()


def test_outputs_are_deterministic_and_existing_artifacts_are_preserved(tmp_path: Path) -> None:
    png = _blob(tmp_path / "blobs", _image_bytes(BLUE), "blue.png")
    sample = _sample("blue-idle", (png,), [_frame(0, png[0], 0, 90, 0.0)])
    snapshot = _write_snapshot(tmp_path / "snapshot.json", (sample,))

    first = materialize_snapshot(snapshot, tmp_path / "one", bucket_sizes=(4,))
    second = materialize_snapshot(snapshot, tmp_path / "two", bucket_sizes=(4,))

    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert first.clip_paths[0].read_bytes() == second.clip_paths[0].read_bytes()
    assert first.sha256 == second.sha256
    before_manifest = first.manifest_path.read_bytes()
    before_clip = first.clip_paths[0].read_bytes()
    with pytest.raises(ExistingOutputError, match="Refusing to replace"):
        materialize_snapshot(snapshot, tmp_path / "one", bucket_sizes=(4,))
    assert first.manifest_path.read_bytes() == before_manifest
    assert first.clip_paths[0].read_bytes() == before_clip
    assert not tuple((tmp_path / "one").rglob("*.tmp"))


def test_rejects_unsupported_sheet_coordinates_instead_of_guessing_a_crop(
    tmp_path: Path,
) -> None:
    sheet = _blob(tmp_path / "blobs", _image_bytes(RED, size=(8, 8)), "sheet.png")
    sample = _sample(
        "sheet-run",
        (sheet,),
        [
            _frame(
                0,
                sheet[0],
                0,
                100,
                0.0,
                metadata={"direct_uv_rect_variants": [[0, 0, 4, 4]]},
            )
        ],
    )
    snapshot = _write_snapshot(tmp_path / "snapshot.json", (sample,))

    with pytest.raises(UnsupportedSheetCoordinatesError, match="direct_uv_rect_variants"):
        materialize_snapshot(snapshot, tmp_path / "output", bucket_sizes=(8,))
    assert not tuple((tmp_path / "output").rglob("*.npy"))


def test_executes_complete_audited_source_sheet_rectangle(tmp_path: Path) -> None:
    sheet_image = Image.new("RGBA", (8, 4), RED)
    sheet_image.paste(GREEN, (4, 0, 8, 4))
    output = BytesIO()
    sheet_image.save(output, "PNG")
    sheet = _blob(tmp_path / "blobs", output.getvalue(), "sheet.png")
    sample = _sample(
        "sheet-run",
        (sheet,),
        [
            _frame(
                0,
                sheet[0],
                7,
                100,
                0.0,
                metadata={
                    "within_declared_source_rect": True,
                    "frame_rect": {
                        "bottom": 4,
                        "coordinate_space": "source_sheet",
                        "height": 4,
                        "left": 4,
                        "right": 8,
                        "top": 0,
                        "width": 4,
                    },
                },
            )
        ],
    )
    snapshot = _write_snapshot(tmp_path / "snapshot.json", (sample,))

    result = materialize_snapshot(snapshot, tmp_path / "output", bucket_sizes=(4,))

    array = np.load(result.clip_paths[0], allow_pickle=False)
    assert array.shape == (1, 4, 4, 4)
    assert np.all(array == np.asarray(GREEN, dtype=np.uint8))
    provenance = result.manifest["sequences"][0]["frame_provenance"][0]
    assert provenance["source_frame_index"] == 7
    assert provenance["source_rect"] == (4, 0, 8, 4)
    assert provenance["source_rect_coordinate_space"] == "source_sheet"
    assert provenance["source_carrier_size"] == (8, 4)
    assert provenance["source_sheet_size"] == (8, 4)
    assert provenance["reconstruction_method"] == "audited_source_sheet_rectangle_v1"


def test_executes_complete_audited_source_image_rectangle_alias(tmp_path: Path) -> None:
    source_image = Image.new("RGBA", (8, 4), RED)
    source_image.paste(BLUE, (0, 0, 4, 4))
    output = BytesIO()
    source_image.save(output, "PNG")
    source = _blob(tmp_path / "blobs", output.getvalue(), "source.png")
    sample = _sample(
        "source-image-run",
        (source,),
        [
            _frame(
                0,
                source[0],
                3,
                100,
                0.0,
                metadata={
                    "frame_rect": {
                        "bottom": 4,
                        "coordinate_space": "source_image",
                        "height": 4,
                        "left": 0,
                        "right": 4,
                        "top": 0,
                        "width": 4,
                    }
                },
            )
        ],
    )
    snapshot = _write_snapshot(tmp_path / "snapshot.json", (sample,))

    result = materialize_snapshot(snapshot, tmp_path / "output", bucket_sizes=(4,))

    array = np.load(result.clip_paths[0], allow_pickle=False)
    assert array.shape == (1, 4, 4, 4)
    assert np.all(array == np.asarray(BLUE, dtype=np.uint8))
    provenance = result.manifest["sequences"][0]["frame_provenance"][0]
    assert provenance["source_frame_index"] == 3
    assert provenance["source_rect"] == (0, 0, 4, 4)
    assert provenance["source_rect_coordinate_space"] == "source_image"
    assert provenance["source_carrier_size"] == (8, 4)
    assert provenance["source_sheet_size"] == (8, 4)
    assert provenance["reconstruction_method"] == "audited_source_sheet_rectangle_v1"


@pytest.mark.parametrize("coordinate_space", ("atlas_texture", ["source_image"]))
def test_rejects_unknown_frame_rectangle_coordinate_space(
    tmp_path: Path,
    coordinate_space: object,
) -> None:
    source = _blob(tmp_path / "blobs", _image_bytes(RED, size=(8, 4)), "source.png")
    sample = _sample(
        "unknown-coordinate-space",
        (source,),
        [
            _frame(
                0,
                source[0],
                0,
                100,
                0.0,
                metadata={
                    "frame_rect": {
                        "bottom": 4,
                        "coordinate_space": coordinate_space,
                        "height": 4,
                        "left": 0,
                        "right": 4,
                        "top": 0,
                        "width": 4,
                    }
                },
            )
        ],
    )
    snapshot = _write_snapshot(tmp_path / "snapshot.json", (sample,))

    with pytest.raises(
        UnsupportedSheetCoordinatesError,
        match="source_image.*source_sheet",
    ):
        materialize_snapshot(snapshot, tmp_path / "output", bucket_sizes=(4,))
    assert not tuple((tmp_path / "output").rglob("*.npy"))


def test_executes_exact_audited_color_key_without_touching_near_magenta(
    tmp_path: Path,
) -> None:
    source_image = Image.new("RGBA", (4, 4), GREEN)
    source_image.putpixel((0, 0), (255, 0, 255, 255))
    source_image.putpixel((1, 0), (254, 0, 255, 255))
    sheet = _blob(tmp_path / "blobs", _png_bytes(source_image), "magenta.png")
    frame_rect = {
        "bottom": 4,
        "coordinate_space": "source_sheet",
        "height": 4,
        "left": 0,
        "right": 4,
        "top": 0,
        "width": 4,
    }
    transformed_sample = _sample(
        "opensurge-color-key",
        (sheet,),
        [
            _frame(
                0,
                sheet[0],
                0,
                100,
                0.0,
                metadata={
                    "frame_rect": frame_rect,
                    "pixel_transforms": [_pixel_transform()],
                },
            )
        ],
    )
    untouched_sample = _sample(
        "non-opensurge-no-transform",
        (sheet,),
        [_frame(0, sheet[0], 0, 100, 0.0, metadata={"frame_rect": frame_rect})],
    )

    transformed = materialize_snapshot(
        _write_snapshot(tmp_path / "transformed.json", (transformed_sample,)),
        tmp_path / "transformed",
        bucket_sizes=(4,),
    )
    untouched = materialize_snapshot(
        _write_snapshot(tmp_path / "untouched.json", (untouched_sample,)),
        tmp_path / "untouched",
        bucket_sizes=(4,),
    )

    transformed_pixels = np.load(transformed.clip_paths[0], allow_pickle=False)
    untouched_pixels = np.load(untouched.clip_paths[0], allow_pickle=False)
    assert tuple(transformed_pixels[0, 0, 0]) == (0, 0, 0, 0)
    assert tuple(transformed_pixels[0, 0, 1]) == (254, 0, 255, 255)
    assert tuple(transformed_pixels[0, 0, 2]) == GREEN
    assert tuple(untouched_pixels[0, 0, 0]) == (255, 0, 255, 255)
    assert tuple(untouched_pixels[0, 0, 1]) == (254, 0, 255, 255)

    provenance = transformed.manifest["sequences"][0]["frame_provenance"][0]
    assert provenance["source_frame_pixel_sha256"] == provenance["pre_transform_pixel_sha256"]
    assert provenance["pre_transform_pixel_sha256"] != provenance["post_transform_pixel_sha256"]
    assert provenance["pixel_transform_results"] == (
        {
            "matched_pixel_count": 1,
            "transform_sha256": _pixel_transform()["transform_sha256"],
        },
    )
    transform_manifest = transformed.manifest["sequences"][0]["pixel_transform"]
    assert transform_manifest["frames_with_declared_transform"] == 1
    assert transform_manifest["total_matched_pixel_count"] == 1
    assert (
        transform_manifest["frame_results"][0]["pre_transform_pixel_sha256"]
        == provenance["pre_transform_pixel_sha256"]
    )
    assert transformed.manifest["config"]["pixel_transform_contract"] == (PIXEL_TRANSFORM_SCHEMA)

    untouched_provenance = untouched.manifest["sequences"][0]["frame_provenance"][0]
    assert untouched_provenance["pixel_transforms"] == ()
    assert (
        untouched_provenance["pre_transform_pixel_sha256"]
        == (untouched_provenance["post_transform_pixel_sha256"])
    )


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    (
        ("schema", "spritelab.pixel_transform.v999", r"\.schema must equal"),
        ("op", "fuzzy_color_distance", r"\.op must equal"),
        ("rgb", [255, 0, 255, 0], r"\.rgb must be exactly three uint8"),
        ("evidence", [], r"\.evidence must be a non-empty list"),
        ("transform_sha256", "0" * 64, r"\.transform_sha256 mismatch"),
    ),
)
def test_rejects_invalid_pixel_transform_metadata(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    transform = _pixel_transform()
    transform[field] = invalid_value
    png = _blob(tmp_path / "blobs", _image_bytes(GREEN), "source.png")
    sample = _sample(
        f"invalid-transform-{field}",
        (png,),
        [
            _frame(
                0,
                png[0],
                0,
                100,
                0.0,
                metadata={"pixel_transforms": [transform]},
            )
        ],
    )
    snapshot = _write_snapshot(tmp_path / f"{field}.json", (sample,))

    with pytest.raises(UnsupportedPixelTransformError, match=message):
        materialize_snapshot(snapshot, tmp_path / f"output-{field}", bucket_sizes=(4,))
    assert not tuple((tmp_path / f"output-{field}").rglob("*.npy"))


def test_rejects_inconsistent_or_out_of_bounds_audited_rectangle(tmp_path: Path) -> None:
    sheet = _blob(tmp_path / "blobs", _image_bytes(RED, size=(8, 8)), "sheet.png")
    for sequence_id, frame_rect, match in (
        (
            "bad-width",
            {
                "left": 0,
                "top": 0,
                "right": 4,
                "bottom": 4,
                "width": 3,
                "height": 4,
                "coordinate_space": "source_sheet",
            },
            "width must equal",
        ),
        (
            "out-of-bounds",
            {
                "left": 4,
                "top": 4,
                "right": 12,
                "bottom": 12,
                "width": 8,
                "height": 8,
                "coordinate_space": "source_sheet",
            },
            "exceeds source sheet",
        ),
    ):
        sample = _sample(
            sequence_id,
            (sheet,),
            [
                _frame(
                    0,
                    sheet[0],
                    3,
                    100,
                    0.0,
                    metadata={"frame_rect": frame_rect},
                )
            ],
        )
        snapshot = _write_snapshot(tmp_path / f"{sequence_id}.json", (sample,))
        with pytest.raises((UnsupportedSheetCoordinatesError, RuntimeError), match=match):
            materialize_snapshot(snapshot, tmp_path / sequence_id, bucket_sizes=(16,))


def test_later_failure_keeps_an_earlier_atomically_published_clip(tmp_path: Path) -> None:
    small = _blob(tmp_path / "blobs", _image_bytes(GREEN, size=(2, 2)), "small.png")
    large = _blob(tmp_path / "blobs", _image_bytes(RED, size=(9, 9)), "large.png")
    first = _sample("a-valid", (small,), [_frame(0, small[0], 0, 100, 0.0)])
    second = _sample("z-oversized", (large,), [_frame(0, large[0], 0, 100, 0.0)])
    snapshot = _write_snapshot(tmp_path / "snapshot.json", (first, second))

    with pytest.raises(NoLosslessBucketError):
        materialize_snapshot(snapshot, tmp_path / "output", bucket_sizes=(8,))

    clips = tuple((tmp_path / "output").rglob("*.npy"))
    assert len(clips) == 1
    assert np.load(clips[0], allow_pickle=False).shape == (1, 8, 8, 4)
    assert not (tmp_path / "output" / "materialization.json").exists()
    assert not tuple((tmp_path / "output").rglob("*.tmp"))
