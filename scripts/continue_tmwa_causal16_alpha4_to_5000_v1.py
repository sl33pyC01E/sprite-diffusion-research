"""Continue the verified TMWA alpha4 quality run immutably to step 5,000."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from continue_tmwa_causal16_alpha4_to_4000_v1 import (  # noqa: E402
    EXPECTED_MANIFEST_SHA256,
    MANIFEST,
    _append_log,
    _sha256,
)

from spritelab.overfit import TinyOverfitConfig, continue_tiny_overfit  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

PARENT = REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-4000"
PARENT_CHECKPOINT = PARENT / "checkpoint.pt"
PARENT_REPORT = PARENT / "overfit-report.json"
GATE_REPORT = REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-ew1-4000-matched-v1.json"
OUTPUT = REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-5000"
LOG = REPOSITORY / "data/logs/tmwa-causal16-b64-f8-alpha4-ew1-v1-5000.jsonl"
EXPECTED_PARENT_CHECKPOINT_SHA256 = (
    "6d63446f22444ff36632d6f9869003883ba65c1b2171968c5b657fd77a78624f"
)
EXPECTED_PARENT_REPORT_SHA256 = "5365f324cf49ff1db0d421ce2145b0820d964a262f4b3f21f4874d35ea36e8ea"
EXPECTED_GATE_REPORT_SHA256 = "1b9d4baf8026ccc57508a73a412424766d79a0a481a551393400258cee7005d4"
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
    steps=5_000,
    log_every=50,
    sample_steps=32,
    seed=0,
    device="cuda",
    precision="float32",
)


def main() -> None:
    for path, expected in (
        (MANIFEST, EXPECTED_MANIFEST_SHA256),
        (PARENT_CHECKPOINT, EXPECTED_PARENT_CHECKPOINT_SHA256),
        (PARENT_REPORT, EXPECTED_PARENT_REPORT_SHA256),
        (GATE_REPORT, EXPECTED_GATE_REPORT_SHA256),
    ):
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {observed}")
    if OUTPUT.exists() or LOG.exists():
        raise FileExistsError("Refusing to replace alpha4 continuation output or log")
    gate = json.loads(GATE_REPORT.read_text(encoding="utf-8"))["samplers"]["endpoint"]
    metrics = gate["matched_metrics_by_alpha_threshold"]["127"]["aggregate_mean"]
    causal = gate["causal_aggregate"]
    if not (
        metrics["premultiplied_rgba_mae"] < 0.026
        and metrics["alpha_iou"] > 0.94
        and causal["mean_generated_to_target_separation_ratio"] > 0.9
    ):
        raise ValueError("alpha4 step-4,000 continuation gates did not pass")
    parent = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    sequence_ids = tuple(str(value) for value in parent["sequence_ids"])
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    guard.require_capacity(2 * 1024**3, label="TMWA alpha4 continuation reserve")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("x", encoding="utf-8", newline="\n") as log:
        _append_log(
            log,
            "launch",
            additional_steps=1_000,
            config=asdict(CONFIG),
            disk_free_bytes=guard.status().free_bytes,
            gate_report_sha256=EXPECTED_GATE_REPORT_SHA256,
            output_directory=str(OUTPUT),
            parent_checkpoint_sha256=EXPECTED_PARENT_CHECKPOINT_SHA256,
            parent_report_sha256=EXPECTED_PARENT_REPORT_SHA256,
            sequence_ids=list(sequence_ids),
        )
        result = continue_tiny_overfit(
            MANIFEST,
            PARENT_CHECKPOINT,
            PARENT_REPORT,
            OUTPUT,
            expected_parent_checkpoint_sha256=EXPECTED_PARENT_CHECKPOINT_SHA256,
            expected_parent_report_sha256=EXPECTED_PARENT_REPORT_SHA256,
            additional_steps=1_000,
            config=CONFIG,
            sequence_ids=sequence_ids,
            disk_guard=guard,
        )
        _append_log(
            log,
            "complete",
            checkpoint_path=str(result.checkpoint_path),
            checkpoint_sha256=_sha256(result.checkpoint_path),
            final_loss=result.final_loss,
            minimum_loss=result.minimum_loss,
            report_path=str(result.report_path),
            report_sha256=result.report_sha256,
            sample_count=len(result.sample_paths),
        )


if __name__ == "__main__":
    main()
