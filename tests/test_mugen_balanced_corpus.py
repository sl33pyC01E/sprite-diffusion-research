from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spritelab.mugen_balanced_corpus import (
    MugenBalancedCorpusConfig,
    build_mugen_balanced_corpus_manifest,
    build_mugen_verb_coverage_report,
    export_mugen_balanced_gallery,
    export_mugen_json_artifact,
)


def test_balanced_manifest_is_an_exact_complete_rectangle(tmp_path: Path) -> None:
    canonical, materialization = _fixtures(tmp_path)
    artifact = build_mugen_balanced_corpus_manifest(
        canonical,
        materialization,
        config=MugenBalancedCorpusConfig(name="idle-walk", verbs=("idle", "walk")),
    )
    assert artifact["counts"]["identities"] == 2
    assert artifact["counts"]["sequences"] == 4
    assert artifact["counts"]["verbs"] == {"idle": 2, "walk": 2}
    assert artifact["counts"]["split_identities"] == {"test": 1, "train": 1}
    assert artifact["balanced_corpus"]["complete_rectangle"] is True
    assert {
        (record["identity_id"], record["conditioning"]["verb"]) for record in artifact["records"]
    } == {("a", "idle"), ("a", "walk"), ("b", "idle"), ("b", "walk")}
    assert all(record["identity"]["label"] for record in artifact["records"])
    assert all(record["source_evidence"]["source_action_number"] for record in artifact["records"])


def test_coverage_report_finds_largest_rectangles(tmp_path: Path) -> None:
    canonical, _ = _fixtures(tmp_path)
    report = build_mugen_verb_coverage_report(canonical)
    assert report["verb_identity_coverage"] == {"idle": 3, "run": 1, "walk": 2}
    assert report["pareto_maxima"] == [
        {
            "best_verb_sets": [["idle"]],
            "cell_count": 3,
            "identity_count": 3,
            "verb_count": 1,
        },
        {
            "best_verb_sets": [["idle", "walk"]],
            "cell_count": 4,
            "identity_count": 2,
            "verb_count": 2,
        },
        {
            "best_verb_sets": [["idle", "run", "walk"]],
            "cell_count": 3,
            "identity_count": 1,
            "verb_count": 3,
        },
    ]


def test_json_export_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    path, digest = export_mugen_json_artifact({"a": 1}, output)
    assert path == output.resolve()
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        export_mugen_json_artifact({"a": 2}, output)


def test_gallery_renders_all_balanced_cells_as_png(tmp_path: Path) -> None:
    materialization_root = tmp_path / "materialized"
    clips = materialization_root / "clips"
    clips.mkdir(parents=True)
    records = []
    sequences = []
    for identity_index, identity_id in enumerate(("a", "b")):
        for verb_index, verb in enumerate(("idle", "walk")):
            sequence_id = f"{identity_id}-{verb}"
            array = np.zeros((8, 128, 128, 4), dtype=np.uint8)
            array[:, 16:112, 24 + verb_index * 8 : 56 + verb_index * 8, :3] = (
                40 + identity_index * 100,
                80 + verb_index * 100,
                140,
            )
            array[:, 16:112, 24 + verb_index * 8 : 56 + verb_index * 8, 3] = 255
            relative = f"clips/{sequence_id}.npy"
            path = materialization_root / relative
            np.save(path, array, allow_pickle=False)
            payload = path.read_bytes()
            output = {
                "array_content_sha256": _array_sha256(array),
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "relative_path": relative,
            }
            records.append(
                {
                    "conditioning": {"verb": verb},
                    "entity_class": "humanoid",
                    "identity": {"label": f"Fighter {identity_id.upper()}"},
                    "identity_id": identity_id,
                    "reference": {
                        "identity_reference_array_sha256": hashlib.sha256(
                            identity_id.encode()
                        ).hexdigest()
                    },
                    "sequence_id": sequence_id,
                    "split": "train",
                    "target": {"source_pixels": output},
                }
            )
            sequences.append({"sequence_id": sequence_id})
    materialization = materialization_root / "materialization.json"
    materialization.write_text(
        json.dumps({"schema_version": 1, "sequence_count": 4, "sequences": sequences}),
        encoding="utf-8",
    )
    manifest = tmp_path / "balanced.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_kind": "mugen_reference_conditioned_primary_motion_training_manifest",
                "balanced_corpus": {
                    "complete_rectangle": True,
                    "identity_count": 2,
                    "verb_count": 2,
                    "verbs": ["idle", "walk"],
                },
                "config": {"one_sequence_per_identity_verb": True},
                "counts": {"sequences": 4},
                "records": records,
                "source": {
                    "materialization_file_sha256": hashlib.sha256(
                        materialization.read_bytes()
                    ).hexdigest(),
                    "materialization_path": str(materialization),
                },
            }
        ),
        encoding="utf-8",
    )
    index_path, digest = export_mugen_balanced_gallery(
        manifest, tmp_path / "gallery", identities_per_page=1
    )
    index = json.loads(index_path.read_bytes())
    assert hashlib.sha256(index_path.read_bytes()).hexdigest() == digest
    assert len(index["pages"]) == 2
    assert all((index_path.parent / page["name"]).suffix == ".png" for page in index["pages"])


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    records = []
    sequences = []
    identity_verbs = {"a": ("idle", "walk", "run"), "b": ("idle", "walk"), "c": ("idle",)}
    for identity_id, verbs in identity_verbs.items():
        for index, verb in enumerate(verbs, 1):
            sequence_id = f"{identity_id}-{verb}"
            records.append(
                {
                    "conditioning": {"verb": verb},
                    "entity_class": "humanoid",
                    "identity_id": identity_id,
                    "reference": {
                        "identity_reference_array_sha256": hashlib.sha256(
                            identity_id.encode()
                        ).hexdigest()
                    },
                    "sequence_id": sequence_id,
                    "split": "train" if identity_id != "b" else "test",
                    "target": {},
                }
            )
            sequences.append(
                {
                    "caption": {
                        "identity_label": f"Fighter {identity_id.upper()}",
                        "description": f"Fighter {identity_id.upper()}",
                    },
                    "identity_id": identity_id,
                    "provenance": {
                        "air_member": f"{identity_id}.air",
                        "archive_sha256": identity_id * 64,
                        "sff_member": f"{identity_id}.sff",
                        "sff_sha256": identity_id * 64,
                        "source_action_index": index,
                        "source_action_number": index,
                        "source_id": "fixture",
                        "source_meaning": verb,
                    },
                    "sequence_id": sequence_id,
                }
            )
    canonical = tmp_path / "canonical.json"
    canonical.write_text(
        json.dumps(
            {
                "artifact_kind": "mugen_reference_conditioned_primary_motion_training_manifest",
                "config": {"one_sequence_per_identity_verb": True},
                "counts": {"sequences": len(records)},
                "records": records,
                "schema_version": 1,
                "source": {
                    "motion_plan_file_sha256": "1" * 64,
                    "motion_plan_path": str(tmp_path / "plan.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    materialization = tmp_path / "materialization.json"
    materialization.write_text(
        json.dumps({"schema_version": 1, "sequence_count": len(sequences), "sequences": sequences}),
        encoding="utf-8",
    )
    return canonical, materialization


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = (
        f"{contiguous.dtype.str}\0{'x'.join(str(item) for item in contiguous.shape)}\0"
    ).encode()
    return hashlib.sha256(header + contiguous.tobytes(order="C")).hexdigest()
