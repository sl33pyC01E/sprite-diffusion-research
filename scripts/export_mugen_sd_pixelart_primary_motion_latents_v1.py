"""Build and encode the canonical primary-motion corpus in the sprite SD VAE."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.sd_control_cache import export_sd14_rgb_latent_cache  # noqa: E402
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

REVISION = "8229c9b6e928103f0e657cfe6b14d902cb2101d6"
MODEL = ROOT / f"data/models/sd-pixelart-spritesheet-{REVISION}"
SOURCE_INDEX_SHA256 = "5f7eea291d7831ccb4d6bb07b011669f532ebd4371dee35b8743ea90fbc926df"
CANONICAL = ROOT / "data/processed/mugen-mffa-reference-primary-motion-canonical-v2.json"
PLAN = ROOT / "data/processed/mugen-mffa-sd-primary-motion-cache-plan-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-sd-pixelart-rgb-vae-latents-primary-motion-v1"


def main() -> None:
    if PLAN.exists() or OUTPUT.exists():
        raise FileExistsError("Refusing to replace the SD primary-motion cache plan/output")
    canonical_bytes = CANONICAL.read_bytes()
    canonical = json.loads(canonical_bytes)
    canonical_records = canonical.get("records")
    count = canonical.get("counts", {}).get("sequences")
    if not isinstance(canonical_records, list) or count != len(canonical_records):
        raise RuntimeError("canonical primary-motion count differs")
    records = []
    for record in canonical_records:
        target = record.get("target") if isinstance(record, dict) else None
        source_pixels = target.get("source_pixels") if isinstance(target, dict) else None
        if not isinstance(source_pixels, dict):
            raise RuntimeError("canonical primary-motion source pixels differ")
        records.append({**record, "target": source_pixels})
    source = canonical.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("canonical primary-motion source differs")
    motion_plan_path = Path(source["motion_plan_path"]).resolve()
    motion_plan_bytes = motion_plan_path.read_bytes()
    if hashlib.sha256(motion_plan_bytes).hexdigest() != source["motion_plan_file_sha256"]:
        raise RuntimeError("canonical motion-plan hash differs")
    motion_plan = json.loads(motion_plan_bytes)
    materialization = motion_plan["source"]["materialization"]
    plan = {
        "artifact_kind": "mugen_sd14_primary_motion_rgb_cache_plan",
        "claim": "noncanonical RGB latent bridge for pretrained motion training",
        "counts": {"sequences": len(records)},
        "records": records,
        "schema_version": 1,
        "source": {
            "canonical_manifest_file_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
            "canonical_manifest_path": str(CANONICAL.resolve()),
            "materialization_file_sha256": materialization["file_sha256"],
            "materialization_path": materialization["path"],
            "motion_plan_file_sha256": hashlib.sha256(motion_plan_bytes).hexdigest(),
            "motion_plan_path": str(motion_plan_path),
        },
    }
    payload = canonical_json_bytes(plan)
    PLAN.write_bytes(payload)
    manifest, digest = export_sd14_rgb_latent_cache(
        PLAN,
        MODEL,
        OUTPUT,
        expected_source_index_sha256=SOURCE_INDEX_SHA256,
        device="cuda",
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
