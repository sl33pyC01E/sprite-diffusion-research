"""Publish compact schema catalogs for the manually acquired large MUGEN RARs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.mugen_archive_directory import (  # noqa: E402
    audit_mugen_archive_metadata_directory,
    parse_7z_slt_members,
)
from spritelab.mugen_schema import (  # noqa: E402
    canonical_six_slot_action_numbers,
    measure_core_schema_coverage,
    schema_verb,
)
from spritelab.storage import DiskGuard  # noqa: E402

SEVEN_ZIP = Path(r"C:\Program Files\7-Zip\7z.exe")
PROFILES = {
    "iidx-jus-chibi-2000": {
        "acquisition": ROOT / "data/raw/acquisitions/mugen-iidx-jus-chibi-2000-manual-v1.json",
        "archive": Path(
            "C:/Users/forre/Downloads/"
            "IIDX Distortion K-Shoot Mania - JUS & Chibi Edition w 2000+ Chars.rar"
        ),
        "archive_sha256": "eb9983574ebc441f44d668693c402befde62aac6eaa604e615652b660e4a596a",
        "archive_size_bytes": 41_804_753_407,
        "expected_listing_exit_codes": (0,),
        "output": ROOT / "data/index/reports/mugen-iidx-jus-chibi-air-schema-catalog-v1.json",
        "root": ROOT / "data/staging/mugen-iidx-jus-chibi-2000-v1",
    },
    "anime-ascension-4000": {
        "acquisition": ROOT
        / "data/raw/acquisitions/mugen-anime-ascension-4000-manual-trim-v1.json",
        "archive": ROOT
        / "data/raw/manual/mugen-anime-ascension-4000-v1/anime-ascension-4000-physical-rar-v1.rar",
        "archive_sha256": "0a16a93be8971843ea1822cffd95942364e2b9f6ce05a1dd921ce490f1a71294",
        "archive_size_bytes": 71_736_537_088,
        "expected_listing_exit_codes": (2,),
        "output": ROOT / "data/index/reports/mugen-anime-ascension-air-schema-catalog-v1.json",
        "root": ROOT / "data/staging/mugen-anime-ascension-4000-v1",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), required=True)
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    output: Path = profile["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to replace schema catalog: {output}")
    archive: Path = profile["archive"]
    _verify_file(
        archive,
        size=int(profile["archive_size_bytes"]),
        sha256=str(profile["archive_sha256"]),
    )
    acquisition: Path = profile["acquisition"]
    acquisition_bytes = acquisition.read_bytes()
    json.loads(acquisition_bytes)
    listing = subprocess.run(
        [
            str(SEVEN_ZIP),
            "l",
            "-slt",
            "-ba",
            "-sccUTF-8",
            "-ir!*.sff",
            "--",
            str(archive),
        ],
        check=False,
        capture_output=True,
    )
    if listing.returncode not in profile["expected_listing_exit_codes"]:
        raise ValueError(f"unexpected 7-Zip listing exit code: {listing.returncode}")
    members = parse_7z_slt_members(listing.stdout.decode("utf-8", "replace"))
    audit = audit_mugen_archive_metadata_directory(profile["root"], members)

    records = []
    slot_counts: Counter[str] = Counter()
    verb_variant_counts: Counter[str] = Counter()
    verb_action_counts: Counter[str] = Counter()
    loop_mode_counts: Counter[str] = Counter()
    complete = 0
    total_actions = 0
    recovered_rows = 0
    for variant in audit.variants:
        coverage = measure_core_schema_coverage(variant.actions)
        canonical = canonical_six_slot_action_numbers(coverage)
        complete += int(coverage.complete_six_slot_core)
        total_actions += len(variant.actions)
        recovered_rows += len(variant.air_parse_exclusions)
        for slot in ("idle", "walk", "jump", "block", "attack"):
            if getattr(coverage, f"{slot}_action_numbers"):
                slot_counts[slot] += 1
        per_variant_verbs: set[str] = set()
        per_variant_action_counts: Counter[str] = Counter()
        for action in variant.actions:
            verb = schema_verb(action.action_number) or "unmapped"
            per_variant_verbs.add(verb)
            per_variant_action_counts[verb] += 1
            verb_action_counts[verb] += 1
            loop_mode_counts[action.loop_mode] += 1
        verb_variant_counts.update(per_variant_verbs)
        stable = {
            "air_sha256": variant.air_sha256,
            "archive_sha256": profile["archive_sha256"],
            "sff_crc32": variant.sff_member.crc32,
            "sff_member": variant.sff_member.path,
            "sff_size_bytes": variant.sff_member.size_bytes,
        }
        variant_id = (
            "mugen_archive_variant_"
            + hashlib.sha256(_canonical(stable).rstrip(b"\n")).hexdigest()[:32]
        )
        records.append(
            {
                "action_count": len(variant.actions),
                "action_counts_by_schema_verb": dict(sorted(per_variant_action_counts.items())),
                "action_number_occurrences": [row.action_number for row in variant.actions],
                "air": {
                    "path": variant.air_path,
                    "sha256": variant.air_sha256,
                },
                "air_parse_exclusions": [asdict(row) for row in variant.air_parse_exclusions],
                "canonical_six_slot_action_numbers": canonical,
                "core_coverage": asdict(coverage),
                "definitions": [
                    {
                        "author": definition.author,
                        "display_name": definition.display_name,
                        "file_sha256": _file_sha256(profile["root"] / path),
                        "name": definition.name,
                        "path": path,
                        "source_comments": list(definition.source_comments),
                    }
                    for path, definition in zip(
                        variant.definition_paths, variant.definitions, strict=True
                    )
                ],
                "provisional_identity_id": "mugen_archive_member_"
                + hashlib.sha256(_canonical(stable).rstrip(b"\n")).hexdigest()[:32],
                "sff": asdict(variant.sff_member),
                "variant_id": variant_id,
            }
        )
    records.sort(key=lambda row: row["variant_id"].encode("utf-8"))
    member_rows = [asdict(row) for row in members]
    member_payload = _canonical(member_rows)
    artifact = {
        "artifact_kind": "mugen_manual_rar_air_schema_catalog",
        "characters": records,
        "counts": {
            "actions": total_actions,
            "complete_six_slot_variants": complete,
            "definition_failures": len(audit.failures),
            "definitions": audit.definition_count,
            "incomplete_six_slot_variants": len(records) - complete,
            "recovered_air_element_rows": recovered_rows,
            "slot_variant_coverage": dict(sorted(slot_counts.items())),
            "sff_inventory_members": len(members),
            "unique_variants": len(records),
            "verb_action_counts": dict(sorted(verb_action_counts.items())),
            "verb_variant_coverage": dict(sorted(verb_variant_counts.items())),
        },
        "definition_failures": [asdict(row) for row in audit.failures],
        "policy": {
            "admission": "every DEF-resolved AIR/SFF pair; SFF pixels remain uninspected",
            "core_view": "idle, walk, jump, block, attack_a, attack_b",
            "identity": (
                "archive-member identity is provisional until SFF payload SHA-256 is streamed"
            ),
            "rights_scope": "unknown/unverified fan collection; no permissive inference",
            "runtime": "CMD/CNS/ST and executable content are never interpreted or executed",
            "timing": "AIR is authoritative; malformed rows are omitted only with exact evidence",
        },
        "schema_version": 1,
        "source": {
            "acquisition_path": str(acquisition),
            "acquisition_sha256": hashlib.sha256(acquisition_bytes).hexdigest(),
            "archive_path": str(archive),
            "archive_sha256": profile["archive_sha256"],
            "archive_size_bytes": profile["archive_size_bytes"],
            "seven_zip_listing_exit_code": listing.returncode,
            "seven_zip_stderr_sha256": hashlib.sha256(listing.stderr).hexdigest(),
            "sff_inventory_sha256": hashlib.sha256(member_payload).hexdigest(),
        },
    }
    payload = _canonical(artifact)
    guard = DiskGuard(ROOT, 100 * 1024**3)
    guard.require_capacity(len(payload), label=f"{args.profile} schema catalog")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print(
        json.dumps(
            {
                "counts": artifact["counts"],
                "path": str(output),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


def _verify_file(path: Path, *, size: int, sha256: str) -> None:
    if path.stat().st_size != size or _file_sha256(path) != sha256:
        raise ValueError(f"source archive identity differs: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
