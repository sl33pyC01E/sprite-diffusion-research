"""Export the canonical train split for Qwen-Image LoRA fine-tuning."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.qwen_still_dataset import export_qwen_image_lora_dataset  # noqa: E402


def main() -> int:
    manifest, sha256 = export_qwen_image_lora_dataset(
        ROOT / "data/processed/mugen-mffa-canonical-appearance-still-plan-v4.json",
        ROOT / "data/processed/mugen-qwen-image-lora-train-v1",
        split="train",
    )
    print(json.dumps({"manifest": str(manifest), "sha256": sha256}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
