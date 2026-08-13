"""Run the hash-pinned TMWA causal16 alpha-weight-four quality ablation."""

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
OUTPUT = REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-3000"
EXPECTED_MANIFEST_SHA256 = "095205d49811a0102368d607a5bcb8dd9a0c2b057ebe33ddaeeda893bede45ca"
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
    alpha_channel_weight=4.0,
    matched_endpoint_weight=1.0,
    steps=3_000,
    log_every=50,
    sample_steps=32,
    seed=0,
    device="cuda",
    precision="float32",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _preflight() -> dict[str, object]:
    if _sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("materialization manifest SHA-256 mismatch")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace experiment output: {OUTPUT}")
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    guard.require_capacity(2 * 1024**3, label="TMWA causal16 alpha4 reserve")
    clips = load_materialized_training_clips(
        MANIFEST,
        split="train",
        target_bucket=CONFIG.target_bucket,
        target_frames=CONFIG.target_frames,
    )
    batch = collate_materialized_clips(clips)
    plan = _build_endpoint_contrast_plan(clips, batch.clean)
    action_counts = Counter(clip.request.action for clip in clips)
    if len(clips) != 16 or action_counts != {"idle": 8, "walk": 8}:
        raise ValueError("expected the exact balanced sixteen-clip causal subset")
    if len(plan.groups) != 8 or len(plan.selected_indices) != 16 or plan.exclusions:
        raise ValueError("expected eight conflict-free endpoint contrast groups")
    return {
        "config": asdict(CONFIG),
        "disk_free_bytes": guard.status().free_bytes,
        "endpoint_contrast_group_count": len(plan.groups),
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "output_directory": str(OUTPUT),
        "sequence_count": len(clips),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-gpu-idle", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    facts = _preflight()
    if arguments.preflight_only:
        print(json.dumps({"status": "preflight_ok", **facts}, sort_keys=True))
        return
    if not arguments.confirm_gpu_idle:
        raise SystemExit("Refusing GPU launch without --confirm-gpu-idle")
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
                "minimum_loss": result.minimum_loss,
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
                "sample_count": len(result.sample_paths),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
