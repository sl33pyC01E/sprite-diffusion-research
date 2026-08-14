"""Export the selected 2x MUGEN RGBA latent cache."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.latent_cache import export_mugen_latent_cache  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

LEGACY_MATERIALIZATION = (
    ROOT / "data/processed/mugen-mffa-anime-combined-action-v1/materialization.json"
)
LEGACY_CHECKPOINT = (
    ROOT / "data/experiments/mugen-mffa-rgba-autoencoder-2x-v1-10000/training-step-0010000.pt"
)
LEGACY_CHECKPOINT_SHA256 = "f8993d4b7dcc0b53526f0b4f1fae15a90dc942cc1c2f3c07e74029fc26c6be85"
LEGACY_OUTPUT = ROOT / "data/processed/mugen-mffa-rgba-latents-2x-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--materialization", type=Path, default=LEGACY_MATERIALIZATION)
    parser.add_argument("--checkpoint", type=Path, default=LEGACY_CHECKPOINT)
    parser.add_argument("--expected-checkpoint-sha256", default=LEGACY_CHECKPOINT_SHA256)
    parser.add_argument("--output", type=Path, default=LEGACY_OUTPUT)
    parser.add_argument("--batch-sequences", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda":
        _wait_for_cuda_headroom()
    path, sha256 = export_mugen_latent_cache(
        args.materialization,
        args.checkpoint,
        args.output,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        batch_sequences=args.batch_sequences,
        device=args.device,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(json.dumps({"manifest": str(path), "sha256": sha256}, sort_keys=True))


def _wait_for_cuda_headroom(*, maximum_used_memory_mib: int = 4096) -> None:
    """Wait before initializing CUDA when another local workload owns the GPU."""

    while True:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
        first = result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else ""
        try:
            used_memory_mib = int(first)
        except ValueError as error:
            raise RuntimeError(f"could not parse nvidia-smi memory use: {first!r}") from error
        if used_memory_mib < maximum_used_memory_mib:
            return
        print(
            json.dumps(
                {
                    "maximum_used_memory_mib": maximum_used_memory_mib,
                    "status": "waiting_for_cuda_headroom",
                    "used_memory_mib": used_memory_mib,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        time.sleep(30)


if __name__ == "__main__":
    main()
