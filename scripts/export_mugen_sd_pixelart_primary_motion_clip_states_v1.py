"""Export action-aware CLIP states for the canonical MUGEN motion corpus."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402
from spritelab.text_token_cache import export_clip_text_token_cache  # noqa: E402

REVISION = "8229c9b6e928103f0e657cfe6b14d902cb2101d6"
MODEL = ROOT / f"data/models/sd-pixelart-spritesheet-{REVISION}"
SOURCE_INDEX_SHA256 = "5f7eea291d7831ccb4d6bb07b011669f532ebd4371dee35b8743ea90fbc926df"
CANONICAL = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"
PLAN = ROOT / "data/processed/mugen-mffa-sd-primary-motion-text-plan-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-sd-pixelart-clip-token-states-primary-motion-v1"

ACTION_PHRASES = {
    "backstep": "stepping backward",
    "block": "blocking",
    "crouch": "crouching",
    "dizzy": "staggering dizzy",
    "get_up": "getting up",
    "hurt": "recoiling hurt",
    "idle": "standing idle",
    "jump": "jumping",
    "normal_attack": "light attacking",
    "run": "running",
    "turn": "turning around",
    "walk": "walking",
}
REDUNDANT_LAYOUT_SEGMENTS = {"centered", "crisp hard edges", "full subject"}


def motion_prompt(appearance_prompt: str, verb: str) -> str:
    """Compose a compact appearance+action prompt without truncation."""

    if verb not in ACTION_PHRASES:
        raise ValueError(f"unsupported canonical verb: {verb}")
    segments = [
        segment.strip()
        for segment in appearance_prompt.split(";")
        if segment.strip() and segment.strip() not in REDUNDANT_LAYOUT_SEGMENTS
    ]
    if not segments:
        raise ValueError("appearance prompt is empty")
    return "; ".join((*segments, "single fighter", ACTION_PHRASES[verb]))


def main() -> None:
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError("Refusing to replace the SD primary-motion text plan/cache")
    canonical_bytes = CANONICAL.read_bytes()
    canonical = json.loads(canonical_bytes)
    source_records = canonical.get("records")
    count = canonical.get("counts", {}).get("sequences")
    if not isinstance(source_records, list) or count != len(source_records):
        raise RuntimeError("canonical primary-motion count differs")
    records = []
    for record in source_records:
        reference = record.get("reference") if isinstance(record, dict) else None
        conditioning = record.get("conditioning") if isinstance(record, dict) else None
        if not isinstance(reference, dict) or not isinstance(conditioning, dict):
            raise RuntimeError("canonical prompt fields differ")
        prompt = motion_prompt(reference.get("appearance_prompt", ""), conditioning.get("verb", ""))
        records.append(
            {
                "identity_id": record.get("identity_id"),
                "prompt": prompt,
                "sample_id": record.get("sample_id"),
                "sequence_id": record.get("sequence_id"),
                "split": record.get("split"),
                "verb": conditioning.get("verb"),
            }
        )
    plan = {
        "artifact_kind": "mugen_sd14_primary_motion_text_cache_plan",
        "claim": "appearance and canonical action conditioning for pretrained temporal adaptation",
        "counts": {
            "prompts": len({record["prompt"] for record in records}),
            "sequences": len(records),
        },
        "prompt_policy": {
            "action_phrases": ACTION_PHRASES,
            "removed_redundant_layout_segments": sorted(REDUNDANT_LAYOUT_SEGMENTS),
            "suffix": "single fighter; {action_phrase}",
        },
        "records": records,
        "schema_version": 1,
        "source": {
            "canonical_manifest_file_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
            "canonical_manifest_path": str(CANONICAL.resolve()),
        },
    }
    payload = canonical_json_bytes(plan)
    PLAN.write_bytes(payload)
    manifest, digest = export_clip_text_token_cache(
        PLAN,
        MODEL,
        OUTPUT,
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        batch_size=64,
        device="cpu",
        disk_guard=DiskGuard(ROOT, min_free_bytes=100 * 1024**3),
    )
    print(
        json.dumps(
            {
                "cache_manifest": str(manifest),
                "cache_manifest_sha256": digest,
                "plan_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
