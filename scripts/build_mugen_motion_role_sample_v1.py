"""Publish the deterministic all-pass MUGEN motion-role VLM sample."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_motion_role import (  # noqa: E402
    MotionRoleSampleConfig,
    stratified_motion_role_sample,
)
from spritelab.spark_caption import canonical_json_bytes  # noqa: E402
from spritelab.storage import DiskGuard  # noqa: E402

PLAN = ROOT / "data/processed/mugen-mffa-reference-motion-plan-v2.json"
PIXEL_AUDIT = ROOT / "data/index/reports/mugen-mffa-subject-frame-pixel-gate-v1.json"
OUTPUT = ROOT / "data/processed/mugen-mffa-motion-role-vlm-sample-v1.json"


def _object(path: Path) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value, payload


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace role sample: {OUTPUT}")
    plan, plan_bytes = _object(PLAN)
    pixel, pixel_bytes = _object(PIXEL_AUDIT)
    artifact = stratified_motion_role_sample(plan, pixel, config=MotionRoleSampleConfig(per_verb=4))
    artifact["source"] = {
        "motion_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "pixel_audit_file_sha256": hashlib.sha256(pixel_bytes).hexdigest(),
    }
    payload = canonical_json_bytes(artifact)
    DiskGuard(ROOT, min_free_bytes=100 * 1024**3).require_capacity(
        len(payload) + 1024**2, label="MUGEN motion-role sample"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        handle.write(payload)
    print(
        json.dumps(
            {
                "counts": artifact["counts"],
                "path": str(OUTPUT),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
