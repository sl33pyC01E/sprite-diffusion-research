"""Export the canonical train split for Qwen-Image LoRA fine-tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.qwen_still_dataset import export_qwen_image_lora_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"data/processed/mugen-qwen-image-lora-{args.split}-v1"
    manifest, sha256 = export_qwen_image_lora_dataset(
        ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json",
        output,
        split=args.split,
    )
    print(json.dumps({"manifest": str(manifest), "sha256": sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
