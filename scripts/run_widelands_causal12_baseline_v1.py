"""Hash-pinned launcher for the Widelands carry/walk memorization baseline."""

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
    REPOSITORY / "data/processed/widelands-ss14-model-ready-v1/"
    "widelands-causal-carry-walk-12-materialization-v1.json"
)
TRAINING_AUDIT = (
    REPOSITORY / "data/index/reports/widelands-causal-carry-walk-12-training-audit-v2.json"
)
PIXEL_AUDIT = REPOSITORY / "data/index/reports/widelands-causal-carry-walk-12-pixel-quality-v1.json"
OUTPUT = REPOSITORY / "data/experiments/widelands-spiderbreeder-carry-walk-b64-f8-baseline-v1-1000"
EXPECTED_SHA256 = {
    MANIFEST: "25b4fc9dd3e58ab2336ca0201419bb8c01c6bfdbc94e89f02ba5a6eb818fb505",
    TRAINING_AUDIT: "ed7d9985e3923d70f9cf0c2c5f98f525e97d2279cb3f6c64143c825b4ee8b460",
    PIXEL_AUDIT: "96a6fa87628e7b7b2bec6772e1b0e4724a6ef87f0bd1e6270700ca918cf2926c",
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
    disk_guard.require_capacity(2 * 1024**3, label="Widelands causal12 baseline reserve")
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
    if len(clips) != 12 or len(identities) != 1:
        raise ValueError("expected exactly twelve clips from one identity")
    if action_counts != {"carry": 6, "walk": 6}:
        raise ValueError(f"unexpected action balance: {dict(action_counts)}")
    if set(direction_counts.values()) != {2} or len(direction_counts) != 6:
        raise ValueError(f"unexpected direction balance: {dict(direction_counts)}")
    if len(endpoint_plan.groups) != 6 or len(endpoint_plan.selected_indices) != 12:
        raise ValueError("expected six target-distinct action groups spanning all twelve rows")
    if endpoint_plan.exclusions:
        raise ValueError("causal12 subset unexpectedly contains endpoint exclusions")
    return {
        "action_counts": dict(sorted(action_counts.items())),
        "config": asdict(CONFIG),
        "direction_counts": dict(sorted(direction_counts.items())),
        "disk_free_bytes": disk_guard.status().free_bytes,
        "endpoint_contrast_group_count": len(endpoint_plan.groups),
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
