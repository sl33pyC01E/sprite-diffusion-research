"""Run exact/swap, Euler/endpoint matched inference for the TMWA step-3,000 child."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from spritelab.inference import (  # noqa: E402
    CheckpointInferenceConfig,
    run_checkpoint_inference,
)
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.training_data import load_materialized_training_clips  # noqa: E402

MANIFEST = (
    REPOSITORY / "data/processed/tmwa-model-ready-action-v1/"
    "tmwa-causal-down-idle-walk-16-materialization-v1.json"
)
EXPERIMENT = REPOSITORY / "data/experiments/tmwa-down-idle-walk-causal16-b64-f8-baseline-v1-3000"
CHECKPOINT = EXPERIMENT / "checkpoint.pt"
SOURCE_REPORT = EXPERIMENT / "overfit-report.json"
EXPECTED_CHECKPOINT_SHA256 = "032b6de0c41c11b5dfa38c4c8e2e4f01ec231863096931d24cf6a676b934bf67"
EXPECTED_SOURCE_REPORT_SHA256 = "0ea6ca38c1d5fff6d8680efa5e65540b3df7e6526b52e5b42ffaabaf929fc415"
EXPECTED_MANIFEST_SHA256 = "095205d49811a0102368d607a5bcb8dd9a0c2b057ebe33ddaeeda893bede45ca"
SEED = 20_260_812


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path, expected in (
        (MANIFEST, EXPECTED_MANIFEST_SHA256),
        (CHECKPOINT, EXPECTED_CHECKPOINT_SHA256),
        (SOURCE_REPORT, EXPECTED_SOURCE_REPORT_SHA256),
    ):
        observed = _sha256_file(path)
        if observed != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {observed}")

    report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    sequence_ids = tuple(report["sequence_ids"])
    if len(sequence_ids) != 16 or len(set(sequence_ids)) != 16:
        raise ValueError("step-3,000 report must contain sixteen unique ordered sequence IDs")
    clips = load_materialized_training_clips(
        MANIFEST,
        sequence_ids=sequence_ids,
        split="train",
        target_bucket=64,
        target_frames=8,
    )
    by_sequence = {clip.sequence_id: clip for clip in clips}
    ordered = tuple(by_sequence[sequence_id] for sequence_id in sequence_ids)
    exact_requests = tuple(clip.request for clip in ordered)
    phase_rows = tuple(clip.frame_phases for clip in ordered)

    swapped_action_by_sequence: dict[str, str] = {}
    plan = report["matched_endpoint_contrast_plan"]
    for group in plan["groups"]:
        group_ids = tuple(group["sequence_ids"])
        group_actions = tuple(group["actions"])
        if len(group_ids) != 2 or set(group_actions) != {"idle", "walk"}:
            raise ValueError("every causal16 contrast group must be one idle/walk pair")
        swapped_action_by_sequence[group_ids[0]] = group_actions[1]
        swapped_action_by_sequence[group_ids[1]] = group_actions[0]
    if set(swapped_action_by_sequence) != set(sequence_ids):
        raise ValueError("contrast plan does not cover every causal16 sequence")
    swap_requests = tuple(
        replace(clip.request, action=swapped_action_by_sequence[clip.sequence_id])
        for clip in ordered
    )

    outputs = (
        (
            "endpoint_exact",
            exact_requests,
            CheckpointInferenceConfig(
                seed=SEED,
                sample_steps=1,
                sampler_algorithm="endpoint",
                noise_strategy="shared",
                device="cuda",
                deterministic_algorithms=True,
            ),
            REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-endpoint-exact-v1",
        ),
        (
            "endpoint_swap",
            swap_requests,
            CheckpointInferenceConfig(
                seed=SEED,
                sample_steps=1,
                sampler_algorithm="endpoint",
                noise_strategy="shared",
                device="cuda",
                deterministic_algorithms=True,
            ),
            REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-endpoint-action-swap-v1",
        ),
        (
            "euler_exact",
            exact_requests,
            CheckpointInferenceConfig(
                seed=SEED,
                sample_steps=32,
                sampler_algorithm="euler",
                noise_strategy="shared",
                device="cuda",
                deterministic_algorithms=True,
            ),
            REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-euler32-exact-v1",
        ),
        (
            "euler_swap",
            swap_requests,
            CheckpointInferenceConfig(
                seed=SEED,
                sample_steps=32,
                sampler_algorithm="euler",
                noise_strategy="shared",
                device="cuda",
                deterministic_algorithms=True,
            ),
            REPOSITORY / "data/inference/tmwa-causal16-b64-f8-baseline3000-euler32-action-swap-v1",
        ),
    )
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    guard.require_capacity(2 * 1024**3, label="TMWA step-3,000 matched inference reserve")
    noise_sha256: str | None = None
    for label, requests, config, output in outputs:
        if output.exists():
            raise FileExistsError(f"Refusing to replace existing inference output: {output}")
        result = run_checkpoint_inference(
            CHECKPOINT,
            output,
            requests,
            phase_rows,
            expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
            config=config,
            source_report_path=SOURCE_REPORT,
            expected_source_report_sha256=EXPECTED_SOURCE_REPORT_SHA256,
            disk_guard=guard,
        )
        if noise_sha256 is None:
            noise_sha256 = result.noise_sha256
        elif result.noise_sha256 != noise_sha256:
            raise ValueError("matched inference runs did not share one exact noise tensor")
        print(
            json.dumps(
                {
                    "label": label,
                    "noise_sha256": result.noise_sha256,
                    "report_path": str(result.report_path),
                    "report_sha256": result.report_sha256,
                    "sample_count": len(result.sample_paths),
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
