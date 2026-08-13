from __future__ import annotations

from pathlib import Path

import pytest

from spritelab.cli import (
    command_dataset_export,
    command_dataset_materialize,
    command_export_lpc_manifest,
    command_ingest_flare,
    command_ingest_open_surge,
    command_ingest_shattered_pixel_dungeon,
    command_ingest_ss14,
    command_ingest_tmwa,
    command_ingest_wesnoth,
    command_ingest_widelands,
    parser,
)


def test_dataset_export_parser_defaults_are_evaluation_safe(tmp_path: Path) -> None:
    output = tmp_path / "snapshot.json"
    args = parser().parse_args(["dataset", "export", str(output)])

    assert args.func is command_dataset_export
    assert args.output == str(output)
    assert args.assignment_strategy == "balanced"
    assert args.temporal_mode == "known"
    assert args.minimum_frame_count == 2
    assert args.include_source == []
    assert args.exclude_source == []
    assert args.group_source_pack is False
    assert args.overwrite is False


def test_dataset_export_parser_accepts_spatial_action_snapshot_options(tmp_path: Path) -> None:
    args = parser().parse_args(
        [
            "dataset",
            "export",
            str(tmp_path / "spatial.json"),
            "--seed",
            "spatial-v1",
            "--minimum-frame-count",
            "1",
            "--temporal-mode",
            "all",
            "--action",
            "idle",
            "--action",
            "run",
            "--include-source",
            "open_surge",
            "--include-source",
            "shattered_pixel_dungeon",
            "--exclude-source",
            "freedoom",
            "--group-source-pack",
            "--overwrite",
        ]
    )

    assert args.seed == "spatial-v1"
    assert args.minimum_frame_count == 1
    assert args.temporal_mode == "all"
    assert args.action == ["idle", "run"]
    assert args.include_source == ["open_surge", "shattered_pixel_dungeon"]
    assert args.exclude_source == ["freedoom"]
    assert args.group_source_pack is True
    assert args.overwrite is True


def test_dataset_export_parser_accepts_model_ready_temporal_mode(tmp_path: Path) -> None:
    args = parser().parse_args(
        [
            "dataset",
            "export",
            str(tmp_path / "model-ready.json"),
            "--temporal-mode",
            "model_ready",
        ]
    )

    assert args.temporal_mode == "model_ready"


def test_dataset_export_refuses_implicit_snapshot_replacement(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve me", encoding="utf-8")
    args = parser().parse_args(["dataset", "export", str(output)])

    with pytest.raises(FileExistsError, match="--overwrite"):
        command_dataset_export(args)

    assert output.read_text(encoding="utf-8") == "preserve me"


def test_lpc_manifest_parser_requires_exact_archive_and_output(tmp_path: Path) -> None:
    output = tmp_path / "lpc.jsonl.gz"
    args = parser().parse_args(["corpus", "lpc-manifest", "a" * 64, str(output), "--overwrite"])

    assert args.func is command_export_lpc_manifest
    assert args.sha256 == "a" * 64
    assert args.output == str(output)
    assert args.overwrite is True


def test_dataset_materialize_parser_exposes_lossless_controls(tmp_path: Path) -> None:
    args = parser().parse_args(
        [
            "dataset",
            "materialize",
            str(tmp_path / "snapshot.json"),
            str(tmp_path / "clips"),
            "--bucket",
            "32",
            "--bucket",
            "64",
            "--no-upscale",
        ]
    )

    assert args.func is command_dataset_materialize
    assert args.bucket == [32, 64]
    assert args.anchor == "bottom_center"
    assert args.no_upscale is True
    assert args.overwrite is False


def test_exact_source_projection_parsers_require_archive_digest() -> None:
    spd = parser().parse_args(["corpus", "shattered-pixel-dungeon", "a" * 64])
    surge = parser().parse_args(["corpus", "opensurge", "b" * 64])
    wesnoth = parser().parse_args(["corpus", "wesnoth", "c" * 64])
    flare = parser().parse_args(["corpus", "flare", "d" * 64])
    widelands = parser().parse_args(["corpus", "widelands", "e" * 64])
    ss14 = parser().parse_args(["corpus", "ss14", "f" * 64])
    tmwa = parser().parse_args(["corpus", "tmwa", "a" * 64])

    assert spd.func is command_ingest_shattered_pixel_dungeon
    assert spd.sha256 == "a" * 64
    assert surge.func is command_ingest_open_surge
    assert surge.sha256 == "b" * 64
    assert wesnoth.func is command_ingest_wesnoth
    assert wesnoth.sha256 == "c" * 64
    assert flare.func is command_ingest_flare
    assert flare.sha256 == "d" * 64
    assert widelands.func is command_ingest_widelands
    assert widelands.sha256 == "e" * 64
    assert ss14.func is command_ingest_ss14
    assert ss14.sha256 == "f" * 64
    assert tmwa.func is command_ingest_tmwa
    assert tmwa.sha256 == "a" * 64
