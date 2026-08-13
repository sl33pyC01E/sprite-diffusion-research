"""Run the TMWA causal16 alpha4, foreground-weight-four quality ablation."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from run_tmwa_causal16_alpha4_v1 import (  # noqa: E402
    CONFIG,
    EXPECTED_MANIFEST_SHA256,
    MANIFEST,
    _sha256,
)

from spritelab.overfit import _build_endpoint_contrast_plan, run_tiny_overfit  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import (  # noqa: E402
    collate_materialized_clips,
    load_materialized_training_clips,
)

OUTPUT = REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-fg4-ew1-v1-3000"
ABLATION_CONFIG = replace(CONFIG, foreground_weight=4.0)


def main() -> None:
    if _sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("materialization manifest SHA-256 mismatch")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace experiment output: {OUTPUT}")
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    guard.require_capacity(2 * 1024**3, label="TMWA alpha4/foreground4 reserve")
    clips = load_materialized_training_clips(
        MANIFEST,
        split="train",
        target_bucket=ABLATION_CONFIG.target_bucket,
        target_frames=ABLATION_CONFIG.target_frames,
    )
    plan = _build_endpoint_contrast_plan(clips, collate_materialized_clips(clips).clean)
    if len(clips) != 16 or len(plan.groups) != 8 or plan.exclusions:
        raise ValueError("expected the exact conflict-free causal16 subset")
    result = run_tiny_overfit(
        MANIFEST,
        OUTPUT,
        config=ABLATION_CONFIG,
        overwrite=False,
        disk_guard=guard,
    )
    print(
        json.dumps(
            {
                "checkpoint_path": str(result.checkpoint_path),
                "config": asdict(ABLATION_CONFIG),
                "final_loss": result.final_loss,
                "minimum_loss": result.minimum_loss,
                "report_path": str(result.report_path),
                "report_sha256": result.report_sha256,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
