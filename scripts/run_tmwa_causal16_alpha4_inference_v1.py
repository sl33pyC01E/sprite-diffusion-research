"""Run exact/swap, Euler/endpoint matched inference for TMWA alpha4 step 3,000."""

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
EXPERIMENT = REPOSITORY / "data/experiments/tmwa-causal16-b64-f8-alpha4-ew1-v1-3000"
CHECKPOINT = EXPERIMENT / "checkpoint.pt"
SOURCE_REPORT = EXPERIMENT / "overfit-report.json"
EXPECTED_CHECKPOINT_SHA256 = "804e4077e0e8d6a20237cd078b97b8069058fd7635390c8d0764ec9eea847826"
EXPECTED_SOURCE_REPORT_SHA256 = "1c471220c2b47cdbd821d8921520db97722abe808f07f02a50f07a499fe035c5"
EXPECTED_MANIFEST_SHA256 = "095205d49811a0102368d607a5bcb8dd9a0c2b057ebe33ddaeeda893bede45ca"
SEED = 20_260_812


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run_variant(
    *,
    experiment: Path,
    expected_checkpoint_sha256: str,
    expected_source_report_sha256: str,
    output_prefix: str,
) -> None:
    checkpoint = experiment / "checkpoint.pt"
    source_report = experiment / "overfit-report.json"
    for path, expected in (
        (MANIFEST, EXPECTED_MANIFEST_SHA256),
        (checkpoint, expected_checkpoint_sha256),
        (source_report, expected_source_report_sha256),
    ):
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {observed}")

    report = json.loads(source_report.read_text(encoding="utf-8"))
    sequence_ids = tuple(str(value) for value in report["sequence_ids"])
    if len(sequence_ids) != 16 or len(set(sequence_ids)) != 16:
        raise ValueError("alpha4 report must contain sixteen unique ordered sequence IDs")
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
    for group in report["matched_endpoint_contrast_plan"]["groups"]:
        group_ids = tuple(str(value) for value in group["sequence_ids"])
        group_actions = tuple(str(value) for value in group["actions"])
        if len(group_ids) != 2 or set(group_actions) != {"idle", "walk"}:
            raise ValueError("every contrast group must be one idle/walk pair")
        swapped_action_by_sequence[group_ids[0]] = group_actions[1]
        swapped_action_by_sequence[group_ids[1]] = group_actions[0]
    if set(swapped_action_by_sequence) != set(sequence_ids):
        raise ValueError("contrast plan does not cover every sequence")
    swap_requests = tuple(
        replace(clip.request, action=swapped_action_by_sequence[clip.sequence_id])
        for clip in ordered
    )

    output_root = REPOSITORY / "data/inference"
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
            output_root / f"{output_prefix}-endpoint-exact-v1",
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
            output_root / f"{output_prefix}-endpoint-action-swap-v1",
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
            output_root / f"{output_prefix}-euler32-exact-v1",
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
            output_root / f"{output_prefix}-euler32-action-swap-v1",
        ),
    )
    guard = DiskGuard(REPOSITORY, 100 * 1024**3)
    guard.require_capacity(2 * 1024**3, label="TMWA alpha4 matched inference reserve")
    noise_sha256: str | None = None
    for label, requests, config, output in outputs:
        if output.exists():
            raise FileExistsError(f"Refusing to replace inference output: {output}")
        result = run_checkpoint_inference(
            checkpoint,
            output,
            requests,
            phase_rows,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            config=config,
            source_report_path=source_report,
            expected_source_report_sha256=expected_source_report_sha256,
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


def main() -> None:
    run_variant(
        experiment=EXPERIMENT,
        expected_checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        expected_source_report_sha256=EXPECTED_SOURCE_REPORT_SHA256,
        output_prefix="tmwa-causal16-alpha4-ew1-3000",
    )


if __name__ == "__main__":
    main()
