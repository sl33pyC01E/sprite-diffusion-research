"""Remove redundant resume checkpoints after a verified final training report.

The earliest milestone and the report-bound final checkpoint are retained.  A
hash-bound plan is published before any unlink so interruption is resumable and
every removed byte remains auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (ROOT / "data/experiments").resolve()
_CHECKPOINT = re.compile(r"training-step-(\d{7})\.pt\Z")
_REPORT_KINDS = frozenset(
    {
        "mugen_latent_still_dit_training",
        "mugen_reference_latent_motion_training",
        "sprite_rgba_autoencoder_training",
    }
)


def compact_completed_training_checkpoints(run_directory: Path | str) -> dict[str, Any]:
    run = Path(run_directory).resolve()
    if run != EXPERIMENTS and EXPERIMENTS not in run.parents:
        raise ValueError("training run must remain under data/experiments")
    report_path = run / "training-report.json"
    report_payload = report_path.read_bytes()
    report = _object(json.loads(report_payload), "training report")
    report_kind = report.get("artifact_kind")
    if report_kind not in _REPORT_KINDS:
        raise ValueError("training report kind does not permit checkpoint compaction")
    checkpoint_key = (
        "checkpoint" if report_kind == "sprite_rgba_autoencoder_training" else "training_checkpoint"
    )
    final = _object(report.get(checkpoint_key), "final training checkpoint")
    final_name = _safe_checkpoint_name(final.get("path"))
    final_path = run / final_name
    if _file_sha256(final_path) != _digest(final, "file_sha256"):
        raise ValueError("final training checkpoint hash differs")
    plan_path = run / "checkpoint-compaction-plan.json"
    report_sha256 = hashlib.sha256(report_payload).hexdigest()
    if plan_path.exists():
        plan = _validated_existing_plan(
            plan_path,
            run=run,
            final_name=final_name,
            report_sha256=report_sha256,
        )
    else:
        checkpoints = sorted(
            (path for path in run.iterdir() if path.is_file() and _CHECKPOINT.fullmatch(path.name)),
            key=lambda path: path.name,
        )
        if not checkpoints or final_path not in checkpoints:
            raise ValueError("training checkpoint set does not contain the report-bound final file")
        keep = {checkpoints[0].name, final_name}
        targets = [_checkpoint_fact(path) for path in checkpoints if path.name not in keep]
        plan = {
            "artifact_kind": "completed_training_checkpoint_compaction_plan",
            "keep": sorted(keep),
            "kept_files": [_checkpoint_fact(run / name) for name in sorted(keep)],
            "remove": targets,
            "run_directory": str(run),
            "schema_version": 1,
            "training_report_file_sha256": report_sha256,
        }
        _publish_or_verify(plan_path, plan)
    keep = set(plan["keep"])
    targets = plan["remove"]
    current_names = {
        path.name for path in run.iterdir() if path.is_file() and _CHECKPOINT.fullmatch(path.name)
    }
    planned_names = keep | {target["path"] for target in targets}
    unexpected = sorted(current_names - planned_names)
    if unexpected:
        raise ValueError(f"unplanned training checkpoints appeared: {unexpected}")
    missing_kept = sorted(keep - current_names)
    if missing_kept:
        raise ValueError(f"retained training checkpoints are absent: {missing_kept}")
    for kept in plan["kept_files"]:
        _verify_checkpoint_fact(run / kept["path"], kept, label="retained")
    for target in targets:
        path = run / target["path"]
        if not path.exists():
            continue
        _verify_checkpoint_fact(path, target, label="removal")
        path.unlink()
    result = {
        "artifact_kind": "completed_training_checkpoint_compaction_result",
        "kept": sorted(keep),
        "plan_file_sha256": _file_sha256(plan_path),
        "removed": targets,
        "removed_bytes": sum(int(row["size_bytes"]) for row in targets),
        "schema_version": 1,
    }
    result_path = run / "checkpoint-compaction-result.json"
    _publish_or_verify(result_path, result)
    return result


def _validated_existing_plan(
    plan_path: Path,
    *,
    run: Path,
    final_name: str,
    report_sha256: str,
) -> dict[str, Any]:
    plan = _object(json.loads(plan_path.read_bytes()), "checkpoint compaction plan")
    if plan.get("artifact_kind") != "completed_training_checkpoint_compaction_plan":
        raise ValueError("existing checkpoint compaction plan kind differs")
    if plan.get("schema_version") != 1:
        raise ValueError("existing checkpoint compaction plan schema differs")
    if plan.get("run_directory") != str(run):
        raise ValueError("existing checkpoint compaction plan run differs")
    if plan.get("training_report_file_sha256") != report_sha256:
        raise ValueError("training report changed after checkpoint compaction plan")
    keep_value = plan.get("keep")
    if not isinstance(keep_value, list) or not keep_value:
        raise ValueError("checkpoint compaction plan keep must be a nonempty list")
    keep = [_safe_checkpoint_name(value) for value in keep_value]
    if keep != sorted(set(keep)) or final_name not in keep or len(keep) > 2:
        raise ValueError("checkpoint compaction plan keep set is invalid")
    kept_files = _checkpoint_facts(plan.get("kept_files"), "kept_files")
    if [fact["path"] for fact in kept_files] != keep:
        raise ValueError("checkpoint compaction kept_files do not match keep")
    targets = _checkpoint_facts(plan.get("remove"), "remove")
    target_names = [target["path"] for target in targets]
    if target_names != sorted(set(target_names)) or set(target_names) & set(keep):
        raise ValueError("checkpoint compaction removal set is invalid")
    return plan


def _checkpoint_facts(value: object, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"checkpoint compaction {label} must be a list")
    facts = []
    for raw in value:
        fact = _object(raw, f"checkpoint compaction {label} row")
        path = _safe_checkpoint_name(fact.get("path"))
        size = fact.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("checkpoint size_bytes must be a nonnegative integer")
        facts.append(
            {
                "file_sha256": _digest(fact, "file_sha256"),
                "path": path,
                "size_bytes": size,
            }
        )
    return facts


def _checkpoint_fact(path: Path) -> dict[str, Any]:
    return {
        "file_sha256": _file_sha256(path),
        "path": path.name,
        "size_bytes": path.stat().st_size,
    }


def _verify_checkpoint_fact(path: Path, fact: dict[str, Any], *, label: str) -> None:
    if path.stat().st_size != fact["size_bytes"] or _file_sha256(path) != fact["file_sha256"]:
        raise ValueError(f"{label} checkpoint differs from compaction plan: {path.name}")


def _publish_or_verify(path: Path, value: dict[str, Any]) -> None:
    payload = _canonical(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing compaction artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.rename(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _safe_checkpoint_name(value: object) -> str:
    if not isinstance(value, str) or _CHECKPOINT.fullmatch(value) is None:
        raise ValueError("final training checkpoint path is not a safe checkpoint filename")
    return value


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _digest(value: dict[str, Any], key: str) -> str:
    result = value.get(key)
    if (
        not isinstance(result, str)
        or len(result) != 64
        or any(character not in "0123456789abcdef" for character in result)
    ):
        raise ValueError(f"{key} must be a lowercase SHA-256")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    result = compact_completed_training_checkpoints(args.run_directory)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
