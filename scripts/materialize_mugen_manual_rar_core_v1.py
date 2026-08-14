"""Run resumable isolated workers over one large manual MUGEN RAR catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.storage import DiskGuard  # noqa: E402

PROFILES = {
    "iidx-jus-chibi-2000": {
        "catalog": ROOT / "data/index/reports/mugen-iidx-jus-chibi-air-schema-catalog-v1.json",
        "catalog_sha256": "4b4b9532dbf4df38d55210659fe7748ffa51503204c353aa031373dfb2defdab",
        "output": ROOT / "data/processed/mugen-iidx-jus-chibi-schema-core-b128-f8-v2",
        "stage": ROOT / "data/processed/.mugen-iidx-jus-chibi-schema-core-b128-f8-v2.partial",
        "duplicate_sources": (),
    },
    "anime-ascension-4000": {
        "catalog": ROOT / "data/index/reports/mugen-anime-ascension-air-schema-catalog-v1.json",
        "catalog_sha256": "5602a57b867b74324e2908fd19fcc316c2d73dfc0c8dbab58e69e1c51c7e7938",
        "output": ROOT / "data/processed/mugen-anime-ascension-schema-core-b128-f8-v2",
        "stage": ROOT / "data/processed/.mugen-anime-ascension-schema-core-b128-f8-v2.partial",
        "duplicate_sources": (
            ROOT / "data/processed/mugen-iidx-jus-chibi-schema-core-b128-f8-v2",
            ROOT / "data/processed/.mugen-iidx-jus-chibi-schema-core-b128-f8-v2.partial",
        ),
    },
}
EXCLUDED_NO_RENDERABLE_CORE_EXIT_CODE = 3
EXACT_DUPLICATE_EXIT_CODE = 4
PROJECTION_VERSION = 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    parser.add_argument("--max-new-variants", type=int)
    parser.add_argument("--worker-timeout-seconds", type=int, default=900)
    parser.add_argument("--worker-attempts", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.max_new_variants is not None and args.max_new_variants <= 0:
        raise ValueError("--max-new-variants must be positive")
    if args.worker_attempts <= 0:
        raise ValueError("--worker-attempts must be positive")
    profile = PROFILES[args.profile]
    output: Path = profile["output"]
    stage: Path = profile["stage"]
    if output.exists():
        raise FileExistsError(f"Refusing to replace materialization: {output}")
    catalog: Path = profile["catalog"]
    catalog_bytes = catalog.read_bytes()
    if hashlib.sha256(catalog_bytes).hexdigest() != profile["catalog_sha256"]:
        raise ValueError("schema catalog SHA-256 differs")
    source = json.loads(catalog_bytes)
    stage.mkdir(parents=True, exist_ok=True)
    character_journal = stage / "character-records.jsonl"
    status_journal = stage / "status-records.jsonl"
    characters = _load_journal(character_journal, "variant_id")
    statuses = _load_journal(status_journal, "variant_id", allow_revisions=True)
    known_pairs = _known_exact_pairs(characters, profile["duplicate_sources"])
    guard = DiskGuard(ROOT, 100 * 1024**3)
    records = sorted(
        source["characters"],
        key=lambda row: (row["sff"]["size_bytes"], row["variant_id"].encode("utf-8")),
    )
    new_count = 0
    worker = ROOT / "scripts/materialize_mugen_manual_rar_variant_worker_v1.py"
    for position, record in enumerate(records, 1):
        previous = statuses.get(record["variant_id"])
        if previous and not (args.retry_failed and previous["status"] == "failed"):
            continue
        if args.max_new_variants is not None and new_count >= args.max_new_variants:
            break
        guard.require_capacity(8 * 1024**2, label="next streamed MUGEN variant")
        worker_record = dict(record)
        fingerprint = _catalog_fingerprint(record)
        worker_record["known_exact_pair_candidates"] = known_pairs.get(fingerprint, [])
        result = None
        attempt_count = 0
        accepted_codes = {0, EXCLUDED_NO_RENDERABLE_CORE_EXIT_CODE, EXACT_DUPLICATE_EXIT_CODE}
        for _attempt_count in range(1, args.worker_attempts + 1):
            attempt_count = _attempt_count
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(worker),
                        "--profile",
                        args.profile,
                        "--stage-root",
                        str(stage),
                    ],
                    input=_canonical(worker_record),
                    capture_output=True,
                    check=False,
                    timeout=args.worker_timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode in accepted_codes:
                break
        if result is None:
            status = {
                "status": "failed",
                "timeout_seconds": args.worker_timeout_seconds,
                "variant_id": record["variant_id"],
                "worker_attempts": attempt_count,
            }
            _append(status_journal, status)
            statuses[record["variant_id"]] = status
            new_count += 1
            print(
                f"[{position}/{len(records)}] {record['variant_id']}: timed out "
                f"after {attempt_count} attempts",
                flush=True,
            )
            continue
        if result.returncode == 0:
            character = json.loads(result.stdout)
            _append(character_journal, character)
            characters[record["variant_id"]] = character
            status = {
                "clip_count": len(character["clips"]),
                "status": "materialized",
                "variant_id": record["variant_id"],
                "worker_attempts": attempt_count,
            }
            known_pairs.setdefault(fingerprint, []).append(
                {
                    "identity_id": character["identity_id"],
                    "sff_sha256": character["source"]["sff"]["sha256"],
                    "variant_id": character["variant_id"],
                }
            )
        elif result.returncode == EXCLUDED_NO_RENDERABLE_CORE_EXIT_CODE:
            exclusion = json.loads(result.stdout)
            if (
                exclusion.get("status") != "excluded"
                or exclusion.get("variant_id") != record["variant_id"]
                or exclusion.get("reason") != "no_canonical_core_action_rendered"
            ):
                raise ValueError("worker returned an invalid exclusion record")
            status = exclusion
        elif result.returncode == EXACT_DUPLICATE_EXIT_CODE:
            duplicate = json.loads(result.stdout)
            if (
                duplicate.get("status") != "duplicate"
                or duplicate.get("variant_id") != record["variant_id"]
                or duplicate.get("reason") != "exact_air_and_sff_duplicate"
            ):
                raise ValueError("worker returned an invalid duplicate record")
            status = duplicate
        else:
            status = {
                "returncode": result.returncode,
                "status": "failed",
                "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
                "stderr_tail": result.stderr.decode("utf-8", "replace")[-4000:],
                "variant_id": record["variant_id"],
                "worker_attempts": attempt_count,
            }
        status["worker_attempts"] = attempt_count
        _append(status_journal, status)
        statuses[record["variant_id"]] = status
        new_count += 1
        print(
            f"[{position}/{len(records)}] {record['variant_id']}: "
            f"{status['status']} {status.get('clip_count', 0)}",
            flush=True,
        )

    failure_count = sum(row["status"] == "failed" for row in statuses.values())
    complete_run = len(statuses) == len(records) and failure_count == 0
    if not complete_run:
        print(
            json.dumps(
                {
                    "catalog_variants": len(records),
                    "materialized": sum(
                        row["status"] == "materialized" for row in statuses.values()
                    ),
                    "failed": failure_count,
                    "new_variants": new_count,
                    "stage": str(stage),
                    "status": "partial_resumable",
                },
                sort_keys=True,
            )
        )
        return 0

    materialized_ids = sorted(
        key for key, row in statuses.items() if row["status"] == "materialized"
    )
    if set(characters) != set(materialized_ids):
        raise ValueError("character journal differs from materialized status rows")
    character_rows = [
        {**characters[key], "projection_version": PROJECTION_VERSION} for key in materialized_ids
    ]
    status_rows = [
        {**statuses[key], "projection_version": PROJECTION_VERSION} for key in sorted(statuses)
    ]
    clips = [clip for character in character_rows for clip in character["clips"]]
    artifact = {
        "artifact_kind": "mugen_manual_rar_fixed_schema_core_materialization",
        "characters": character_rows,
        "clips": clips,
        "counts": {
            "catalog_variants": len(records),
            "clips": len(clips),
            "complete_six_slot_characters": sum(
                row["complete_six_slot_core"] for row in character_rows
            ),
            "materialized_characters": len(character_rows),
            "excluded_characters": sum(row["status"] == "excluded" for row in status_rows),
            "duplicate_character_occurrences": sum(
                row["status"] == "duplicate" for row in status_rows
            ),
            "slots": dict(sorted(Counter(row["slot"] for row in clips).items())),
            "status": dict(sorted(Counter(row["status"] for row in status_rows).items())),
        },
        "policy": {
            "geometry": (
                "all admitted core actions share one world-origin view fitted without "
                "visible-pixel clipping; 8x128x128 RGBA"
            ),
            "isolation": "one source SFF per subprocess; failures are variant-local",
            "rights_scope": "unknown/unverified fan collection; no permissive inference",
            "runtime": "no CMD/CNS/ST or executable content interpreted or executed",
            "split": "unset until cross-corpus exact SFF/array duplicate components are built",
        },
        "projection_version": PROJECTION_VERSION,
        "schema_version": 1,
        "source": {
            "catalog_path": str(catalog),
            "catalog_sha256": profile["catalog_sha256"],
        },
        "status_rows": status_rows,
    }
    payload = _canonical(artifact)
    manifest = stage / "materialization.json"
    if manifest.exists():
        raise FileExistsError(f"partial manifest already exists: {manifest}")
    with manifest.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.rename(stage, output)
    print(
        json.dumps(
            {
                "counts": artifact["counts"],
                "path": str(output / "materialization.json"),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_journal(
    path: Path, key: str, *, allow_revisions: bool = False
) -> dict[str, dict[str, object]]:
    output = {}
    if not path.exists():
        return output
    for number, line in enumerate(path.read_bytes().splitlines(), 1):
        row = json.loads(line)
        value = row[key]
        if value in output and not allow_revisions:
            raise ValueError(f"duplicate journal key at line {number}: {value}")
        output[value] = row
    return output


def _catalog_fingerprint(record: dict[str, object]) -> tuple[str, int, str | None]:
    air = record["air"]
    sff = record["sff"]
    if not isinstance(air, dict) or not isinstance(sff, dict):
        raise TypeError("catalog record AIR/SFF metadata must be objects")
    return str(air["sha256"]), int(sff["size_bytes"]), sff.get("crc32")


def _known_exact_pairs(
    current: dict[str, dict[str, object]], duplicate_sources: tuple[Path, ...]
) -> dict[tuple[str, int, str | None], list[dict[str, str]]]:
    rows = list(current.values())
    for root in duplicate_sources:
        journal = root / "character-records.jsonl"
        manifest = root / "materialization.json"
        if journal.exists():
            rows.extend(_load_journal(journal, "variant_id").values())
        elif manifest.exists():
            payload = json.loads(manifest.read_bytes())
            rows.extend(payload.get("characters", []))
    output: dict[tuple[str, int, str | None], list[dict[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        source = row.get("source")
        if not isinstance(source, dict):
            continue
        air = source.get("air")
        sff = source.get("sff")
        if not isinstance(air, dict) or not isinstance(sff, dict):
            continue
        candidate = {
            "identity_id": str(row["identity_id"]),
            "sff_sha256": str(sff["sha256"]),
            "variant_id": str(row["variant_id"]),
        }
        dedup_key = candidate["variant_id"], candidate["sff_sha256"]
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        fingerprint = str(air["sha256"]), int(sff["size_bytes"]), sff.get("crc32")
        output.setdefault(fingerprint, []).append(candidate)
    for candidates in output.values():
        candidates.sort(key=lambda row: row["variant_id"].encode("utf-8"))
    return output


def _append(path: Path, row: dict[str, object]) -> None:
    payload = _canonical(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
