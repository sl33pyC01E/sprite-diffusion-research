"""Publish complete shared-verb MUGEN corpus tiers and the primary target gallery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_balanced_corpus import (  # noqa: E402
    MugenBalancedCorpusConfig,
    build_mugen_balanced_corpus_manifest,
    build_mugen_verb_coverage_report,
    export_mugen_balanced_gallery,
    export_mugen_json_artifact,
)
from spritelab.storage import DiskGuard  # noqa: E402

CANONICAL = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"
MATERIALIZATION = ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
REPORT = ROOT / "data/index/reports/mugen-mffa-balanced-verb-coverage-v1.json"
TIERS = (
    (
        MugenBalancedCorpusConfig(
            name="primary_idle_walk_normal_attack",
            verbs=("idle", "normal_attack", "walk"),
        ),
        ROOT / "data/processed/mugen-mffa-balanced-idle-walk-attack-v1.json",
    ),
    (
        MugenBalancedCorpusConfig(
            name="core_idle_crouch_walk_normal_attack",
            verbs=("crouch", "idle", "normal_attack", "walk"),
        ),
        ROOT / "data/processed/mugen-mffa-balanced-idle-crouch-walk-attack-v1.json",
    ),
    (
        MugenBalancedCorpusConfig(
            name="dynamic_idle_walk_run_normal_attack",
            verbs=("idle", "normal_attack", "run", "walk"),
        ),
        ROOT / "data/processed/mugen-mffa-balanced-idle-walk-run-attack-v1.json",
    ),
    (
        MugenBalancedCorpusConfig(
            name="rich_idle_crouch_turn_walk_normal_attack",
            verbs=("crouch", "idle", "normal_attack", "turn", "walk"),
        ),
        ROOT / "data/processed/mugen-mffa-balanced-idle-crouch-turn-walk-attack-v1.json",
    ),
)
GALLERY = ROOT / "data/previews/mugen-mffa-balanced-idle-walk-attack-v1"


def main() -> None:
    guard = DiskGuard(ROOT, min_free_bytes=100 * 1024**3)
    outputs = []
    report_path, report_sha256 = export_mugen_json_artifact(
        build_mugen_verb_coverage_report(CANONICAL), REPORT, disk_guard=guard
    )
    outputs.append({"kind": "coverage", "path": str(report_path), "sha256": report_sha256})
    primary_path = None
    for config, path in TIERS:
        artifact = build_mugen_balanced_corpus_manifest(
            CANONICAL,
            MATERIALIZATION,
            config=config,
        )
        output, digest = export_mugen_json_artifact(artifact, path, disk_guard=guard)
        outputs.append(
            {
                "counts": artifact["counts"],
                "kind": "balanced_corpus",
                "name": config.name,
                "path": str(output),
                "sha256": digest,
            }
        )
        if config.name == "primary_idle_walk_normal_attack":
            primary_path = output
    if primary_path is None:
        raise RuntimeError("primary balanced corpus was not created")
    gallery_index, gallery_sha256 = export_mugen_balanced_gallery(
        primary_path,
        GALLERY,
        disk_guard=guard,
    )
    outputs.append({"kind": "gallery", "path": str(gallery_index), "sha256": gallery_sha256})
    print(json.dumps({"outputs": outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
