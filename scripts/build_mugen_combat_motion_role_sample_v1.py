"""Publish a larger VLM audit sample for MUGEN combat-action vocabulary."""

from __future__ import annotations

import hashlib
import json
import os
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
OUTPUT = ROOT / "data/processed/mugen-mffa-combat-motion-role-vlm-sample-v1.json"
VERBS = ("block", "normal_attack", "special_attack", "super_attack")


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace combat role sample: {OUTPUT}")
    plan_bytes = PLAN.read_bytes()
    pixel_bytes = PIXEL_AUDIT.read_bytes()
    plan = _object(plan_bytes, "motion plan")
    pixel = _object(pixel_bytes, "pixel audit")
    records = plan.get("records")
    if not isinstance(records, list) or plan.get("counts", {}).get("sequences") != len(records):
        raise RuntimeError("motion plan record count differs")
    selected = [
        record
        for record in records
        if isinstance(record, dict)
        and isinstance(record.get("conditioning"), dict)
        and record["conditioning"].get("verb") in VERBS
    ]
    filtered = dict(plan)
    filtered["records"] = selected
    filtered["counts"] = {**plan["counts"], "sequences": len(selected)}
    artifact = stratified_motion_role_sample(
        filtered,
        pixel,
        config=MotionRoleSampleConfig(per_verb=32, include_pixel_statuses=("all_pass",)),
    )
    actual_verbs = sorted(
        {record["expected_verb"] for record in artifact["records"]},
        key=lambda value: value.encode(),
    )
    if actual_verbs != sorted(VERBS):
        raise RuntimeError(f"combat sample verb closure differs: {actual_verbs!r}")
    artifact["scope"] = {
        "claim": "bounded visual precision audit before admitting advanced combat verbs",
        "verbs": list(VERBS),
    }
    artifact["source"] = {
        "motion_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "motion_plan_path": str(PLAN.resolve()),
        "pixel_audit_file_sha256": hashlib.sha256(pixel_bytes).hexdigest(),
        "pixel_audit_path": str(PIXEL_AUDIT.resolve()),
    }
    payload = canonical_json_bytes(artifact)
    DiskGuard(ROOT, min_free_bytes=100 * 1024**3).require_capacity(
        len(payload) + 1024**2, label="MUGEN combat motion-role sample"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
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
    return 0


def _object(payload: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
