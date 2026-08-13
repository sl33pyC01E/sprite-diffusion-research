"""Export a hash-bound CLIP embedding table for every TMWA identity description."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.semantic_text import (  # noqa: E402
    TransformersClipTextBackend,
    export_semantic_embedding_table,
)
from spritelab.storage import DiskGuard  # noqa: E402

REVISION = "c7244be81152024ce0e99ac8d2e373a8953d9f9a"
SNAPSHOT = ROOT / f"data/models/openai-clip-vit-base-patch32-{REVISION[:12]}"
MATERIALIZATION = ROOT / "data/processed/tmwa-model-ready-action-v1/materialization.json"
OUTPUT = ROOT / "data/processed/semantic-text/tmwa-openai-clip-vit-b32-c7244be-v1"


def main() -> None:
    raw = json.loads(MATERIALIZATION.read_text(encoding="utf-8"))
    descriptions = [row["caption"]["description"] for row in raw["sequences"]]
    backend = TransformersClipTextBackend(
        SNAPSHOT,
        model_id="openai/clip-vit-base-patch32",
        model_revision=REVISION,
        device="cpu",
    )
    result = export_semantic_embedding_table(
        descriptions,
        OUTPUT,
        backend,
        batch_size=64,
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(result)


if __name__ == "__main__":
    main()
