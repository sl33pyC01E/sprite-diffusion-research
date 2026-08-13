"""Hash-pinned launcher for the TMWA down-facing idle/walk causal baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spritelab.overfit import (  # noqa: E402
    TinyOverfitConfig,
    _build_endpoint_contrast_plan,
    run_tiny_overfit,
)
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import (  # noqa: E402
    collate_materialized_clips,
    load_materialized_training_clips,
)

MANIFEST = (
    REPOSITORY / "data/processed/tmwa-model-ready-action-v1/"
    "tmwa-causal-down-idle-walk-16-materialization-v1.json"
)
TRAINING_AUDIT = (
    REPOSITORY / "data/index/reports/tmwa-causal-down-idle-walk-16-training-audit-v1.json"
)
PIXEL_AUDIT = REPOSITORY / "data/index/reports/tmwa-causal-down-idle-walk-16-pixel-quality-v1.json"
DESIGN_AUDIT = REPOSITORY / "data/index/reports/tmwa-model-ready-training-design-v1.json"
OUTPUT = REPOSITORY / "data/experiments/tmwa-down-idle-walk-causal16-b64-f8-baseline-v1-1000"
EXPECTED_SHA256 = {
    MANIFEST: "095205d49811a0102368d607a5bcb8dd9a0c2b057ebe33ddaeeda893bede45ca",
    TRAINING_AUDIT: "ccac5cd70963aaded8fa2265b53c8fe6efe945a6a9992aa0148278ac949beec0",
    PIXEL_AUDIT: "cc77d5167f0eb8cd6eb9046673f8f623183c140480fa088c2d2f3197c27f42bc",
    DESIGN_AUDIT: "6d0d3d459a98d804fba32b6f158f26f90a656db65ee08b444c8329445e511582",
}
CONFIG = TinyOverfitConfig(
    target_bucket=64,
    target_frames=8,
    patch_size=4,
    model_dim=128,
    depth=4,
    num_heads=4,
    condition_dim=128,
    max_text_bytes=48,
    learning_rate=3e-4,
    weight_decay=0.0,
    foreground_weight=2.0,
    alpha_channel_weight=1.0,
    matched_endpoint_weight=1.0,
    steps=1_000,
    log_every=25,
    sample_steps=32,
    seed=0,
    device="cuda",
    precision="float32",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _preflight() -> dict[str, object]:
    for path, expected in EXPECTED_SHA256.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {observed}")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace experiment output: {OUTPUT}")

    disk_guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    disk_guard.require_capacity(2 * 1024**3, label="TMWA causal16 baseline reserve")
    clips = load_materialized_training_clips(
        MANIFEST,
        split="train",
        target_bucket=CONFIG.target_bucket,
        target_frames=CONFIG.target_frames,
    )
    batch = collate_materialized_clips(clips)
    endpoint_plan = _build_endpoint_contrast_plan(clips, batch.clean)
    identities = {clip.identity_id for clip in clips}
    action_counts = Counter(clip.request.action for clip in clips)
    direction_counts = Counter(clip.request.direction for clip in clips)
    entity_counts = Counter(clip.request.entity_class for clip in clips)
    loop_counts = Counter(clip.request.loop_mode for clip in clips)
    if len(clips) != 16 or len(identities) != 8:
        raise ValueError("expected exactly sixteen clips from eight identities")
    if action_counts != {"idle": 8, "walk": 8}:
        raise ValueError(f"unexpected action balance: {dict(action_counts)}")
    if direction_counts != {"down": 16}:
        raise ValueError(f"unexpected direction balance: {dict(direction_counts)}")
    if entity_counts != {"animal": 8, "humanoid": 2, "monster": 6}:
        raise ValueError(f"unexpected entity balance: {dict(entity_counts)}")
    if loop_counts != {"loop": 16}:
        raise ValueError(f"unexpected loop-mode balance: {dict(loop_counts)}")
    if len(endpoint_plan.groups) != 8 or len(endpoint_plan.selected_indices) != 16:
        raise ValueError("expected eight target-distinct action groups spanning all rows")
    if endpoint_plan.exclusions:
        raise ValueError("causal16 subset unexpectedly contains endpoint exclusions")
    if any(group.actions != ("idle", "walk") for group in endpoint_plan.groups):
        raise ValueError("every endpoint group must be an idle/walk pair")

    design = json.loads(DESIGN_AUDIT.read_text(encoding="utf-8"))
    selected_groups = design.get("selection", {}).get("groups")
    if not isinstance(selected_groups, list) or len(selected_groups) != 8:
        raise ValueError("design audit must declare eight selected groups")
    expected_targets = {
        sequence_id: target_sha256
        for group in selected_groups
        for sequence_id, target_sha256 in zip(
            group["sequence_ids"],
            group["fixed_model_target_sha256"],
            strict=True,
        )
    }
    observed_targets = {
        sequence_id: target_sha256
        for group in endpoint_plan.groups
        for sequence_id, target_sha256 in zip(
            group.sequence_ids,
            group.target_sha256,
            strict=True,
        )
    }
    if observed_targets != expected_targets:
        raise ValueError("endpoint target hashes differ from the pinned design audit")
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "config": asdict(CONFIG),
        "design_audit_sha256": EXPECTED_SHA256[DESIGN_AUDIT],
        "direction_counts": dict(sorted(direction_counts.items())),
        "disk_free_bytes": disk_guard.status().free_bytes,
        "endpoint_contrast_group_count": len(endpoint_plan.groups),
        "entity_class_counts": dict(sorted(entity_counts.items())),
        "gpu_idle_attestation_required_for_launch": True,
        "identity_ids": sorted(identities),
        "manifest_path": str(MANIFEST),
        "manifest_sha256": EXPECTED_SHA256[MANIFEST],
        "output_directory": str(OUTPUT),
        "sequence_count": len(clips),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-gpu-idle",
        action="store_true",
        help="Attest that an external process audit found no concurrent GPU workload.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify every input and print the exact plan without training.",
    )
    arguments = parser.parse_args()
    facts = _preflight()
    if arguments.preflight_only:
        print(json.dumps({"status": "preflight_ok", **facts}, sort_keys=True))
        return
    if not arguments.confirm_gpu_idle:
        raise SystemExit(
            "Refusing GPU launch without --confirm-gpu-idle after an external process audit"
        )
    result = run_tiny_overfit(
        MANIFEST,
        OUTPUT,
        config=CONFIG,
        overwrite=False,
        disk_guard=DiskGuard(REPOSITORY, 100 * 1024**3),
    )
    print(
        json.dumps(
            {
                "checkpoint_path": str(result.checkpoint_path),
                "final_loss": result.final_loss,
                "initial_loss": result.initial_loss,
                "minimum_loss": result.minimum_loss,
                "output_directory": str(result.output_directory),
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "sample_count": len(result.sample_paths),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
