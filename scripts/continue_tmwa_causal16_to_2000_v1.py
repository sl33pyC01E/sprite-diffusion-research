"""Hash-pinned immutable continuation of the TMWA causal16 baseline to step 2,000."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spritelab.overfit import TinyOverfitConfig, continue_tiny_overfit  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

MANIFEST = (
    REPOSITORY / "data/processed/tmwa-model-ready-action-v1/"
    "tmwa-causal-down-idle-walk-16-materialization-v1.json"
)
PARENT = REPOSITORY / "data/experiments/tmwa-down-idle-walk-causal16-b64-f8-baseline-v1-1000"
PARENT_CHECKPOINT = PARENT / "checkpoint.pt"
PARENT_REPORT = PARENT / "overfit-report.json"
OUTPUT = REPOSITORY / "data/experiments/tmwa-down-idle-walk-causal16-b64-f8-baseline-v1-2000"
LOG = REPOSITORY / "data/logs/tmwa-down-idle-walk-causal16-b64-f8-baseline-v1-2000.jsonl"
EXPECTED_PARENT_CHECKPOINT_SHA256 = (
    "7b844bb14276c64c16bc4bd723e8ac55d0d9cdd6b01079e957df0fc5aac01f7c"
)
EXPECTED_PARENT_REPORT_SHA256 = "147acc50a41b9bbffd905d0636520aeaebfc73a45755982a64d235dc5ac9fdbf"
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
    alpha_channel_weight=1.0,
    matched_endpoint_weight=1.0,
    steps=2_000,
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


def _append_log(handle: object, event: str, **fields: object) -> None:
    payload = {
        "event": event,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **fields,
    }
    line = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    handle.write(line + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    print(line, flush=True)


def main() -> None:
    expected = {
        MANIFEST: EXPECTED_MANIFEST_SHA256,
        PARENT_CHECKPOINT: EXPECTED_PARENT_CHECKPOINT_SHA256,
        PARENT_REPORT: EXPECTED_PARENT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        observed = _sha256_file(path)
        if observed != digest:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {digest}, got {observed}")
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace continuation output: {OUTPUT}")
    if LOG.exists():
        raise FileExistsError(f"Refusing to replace continuation log: {LOG}")
    parent_report = json.loads(PARENT_REPORT.read_text(encoding="utf-8"))
    sequence_ids = tuple(parent_report["sequence_ids"])
    if len(sequence_ids) != 16:
        raise ValueError("parent report must contain exactly sixteen ordered sequence IDs")
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    guard.require_capacity(2 * 1024**3, label="TMWA causal16 continuation reserve")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("x", encoding="utf-8", newline="\n") as log:
        _append_log(
            log,
            "launch",
            additional_steps=1_000,
            config=asdict(CONFIG),
            disk_free_bytes=guard.status().free_bytes,
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
            checkpoint_sha256=_sha256_file(result.checkpoint_path),
            final_loss=result.final_loss,
            initial_loss=result.initial_loss,
            minimum_loss=result.minimum_loss,
            report_path=str(result.report_path),
            report_sha256=result.report_sha256,
            sample_count=len(result.sample_paths),
        )


if __name__ == "__main__":
    main()
