"""Continue the verified TMWA alpha4 quality run immutably to step 6,000."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from continue_tmwa_causal16_alpha4_to_5000_v1 import (  # noqa: E402
    CONFIG,
    EXPECTED_MANIFEST_SHA256,
    MANIFEST,
    _append_log,
    _sha256,
)

from spritelab.overfit import continue_tiny_overfit  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

PARENT = REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-5000"
PARENT_CHECKPOINT = PARENT / "checkpoint.pt"
PARENT_REPORT = PARENT / "overfit-report.json"
GATE_REPORT = REPOSITORY / "data/index/reports/tmwa-causal16-alpha4-ew1-5000-matched-v1.json"
OUTPUT = REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-6000"
LOG = REPOSITORY / "data/logs/tmwa-causal16-b64-f8-alpha4-ew1-v1-6000.jsonl"
EXPECTED_PARENT_CHECKPOINT_SHA256 = (
    "dc2e528372e2a0e17042fa582c6435052297462c7128ec435a86769324bd444a"
)
EXPECTED_PARENT_REPORT_SHA256 = "3635c59396ff4f6e44f0e7623bdfa84e75213bdf56034e4a1a90c098897307d7"
EXPECTED_GATE_REPORT_SHA256 = "3fd894d77e6bb24a20c14f5060f3f9e913eb935a9c616471202c252fe5716cf7"
CONTINUATION_CONFIG = replace(CONFIG, steps=6_000)


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
        metrics["premultiplied_rgba_mae"] < 0.023
        and metrics["alpha_iou"] > 0.96
        and causal["mean_generated_to_target_separation_ratio"] > 0.95
        and causal["idle_to_walk_moves_toward_replacement_count"] == 8
    ):
        raise ValueError("alpha4 step-5,000 continuation gates did not pass")
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
            config=asdict(CONTINUATION_CONFIG),
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
            config=CONTINUATION_CONFIG,
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
