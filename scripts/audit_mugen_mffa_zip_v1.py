"""Power-loss-resumable exact audit of acquired MFFA anime ZIP characters."""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.adapters.mugen import (  # noqa: E402
    audit_character_zip,
    decode_sff_v1,
    decode_sff_v2,
    materialize_actions,
)
from spritelab.storage import DiskGuard  # noqa: E402

SOURCE = ROOT / "data/index/reports/mugen-mffa-anime-zip-acquisition-v1.json"
JOURNAL = ROOT / "data/index/reports/mugen-mffa-anime-zip-corpus-audit-v3.jsonl"
OUTPUT = ROOT / "data/index/reports/mugen-mffa-anime-zip-corpus-audit-v3.json"


def _console(message: str) -> None:
    print(message.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _load_journal() -> dict[str, dict[str, object]]:
    if not JOURNAL.exists():
        return {}
    rows: dict[str, dict[str, object]] = {}
    with JOURNAL.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"journal line {number} is not terminated")
            row = json.loads(line)
            digest = row["archive_sha256"]
            if digest in rows and rows[digest] != row:
                raise ValueError(f"conflicting journal rows for {digest}")
            rows[digest] = row
    return rows


def _append(row: dict[str, object]) -> None:
    payload = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    with JOURNAL.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _member_bytes(archive_payload: bytes, normalized_name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
        matches = [
            info
            for info in archive.infolist()
            if PurePosixPath(info.filename.replace("\\", "/")).as_posix().lstrip("/").casefold()
            == normalized_name.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one SFF member {normalized_name!r}, found {len(matches)}")
        return archive.read(matches[0])


def _initial_act_palette(
    archive_payload: bytes, definition_member: str, reference: str | None
) -> bytes | None:
    if not reference:
        return None
    normalized = str(PurePosixPath(definition_member).parent / reference.replace("\\", "/")).lstrip(
        "/"
    )
    try:
        payload = _member_bytes(archive_payload, normalized)
    except ValueError:
        return None
    return payload[:768] if len(payload) >= 768 else None


def _audit(item: dict[str, object]) -> dict[str, object]:
    archive = item["archive"]
    payload = Path(archive["cas_path"]).read_bytes()
    if hashlib.sha256(payload).hexdigest() != archive["sha256"]:
        raise ValueError("CAS archive hash mismatch")
    audit = audit_character_zip(payload)
    base: dict[str, object] = {
        "archive_bytes": audit.archive_bytes,
        "archive_sha256": audit.archive_sha256,
        "inventory_sha256": audit.inventory_sha256,
        "member_count": audit.member_count,
        "resource": item["resource"],
        "definition_members": audit.definition_members,
        "internal_names": [value.name for value in audit.definition_variants],
        "internal_display_names": [value.display_name for value in audit.definition_variants],
        "internal_authors": [value.author for value in audit.definition_variants],
        "air_member": audit.air_member,
        "sff_member": audit.sff_member,
        "sff_format_family": audit.sff_header.format_family,
        "sff_sha256": audit.sff_header.sha256,
        "air_actions": len(audit.actions),
        "air_frame_occurrences": sum(len(action.elements) for action in audit.actions),
        "executable_members": audit.executable_members,
        "runtime_logic_members": audit.runtime_logic_members,
    }
    sff_payload = _member_bytes(payload, audit.sff_member)
    palette_count: int | None = None
    pixel_formats: list[int] | None = None
    if audit.sff_header.format_family == "sff_v1":
        sprites = decode_sff_v1(
            sff_payload,
            initial_palette_rgb=_initial_act_palette(
                payload,
                audit.definition_member,
                audit.definition.file("pal1"),
            ),
        )
        decode_status = "decoded_sff_v1"
    else:
        sprites, palettes = decode_sff_v2(sff_payload)
        palette_count = len(palettes)
        pixel_formats = sorted({sprite.pixel_format for sprite in sprites})
        decode_status = "decoded_sff_v2"
    plan = materialize_actions(audit.actions, sprites)
    labels = Counter(value.normalized_action or "unknown" for value in plan.admitted)
    loops = Counter(value.loop_mode for value in plan.admitted)
    exclusions = Counter(value.reason for value in plan.excluded)
    return {
        **base,
        "decode_status": decode_status,
        "palette_count": palette_count,
        "pixel_formats": pixel_formats,
        "sprite_occurrences": len(sprites),
        "unique_rgba_payloads": len({value.rgba_sha256 for value in sprites}),
        "linked_sprite_occurrences": sum(
            value.linked_sprite_index is not None for value in sprites
        ),
        "admitted_actions": len(plan.admitted),
        "admitted_frame_occurrences": sum(len(value.frames) for value in plan.admitted),
        "action_labels": dict(sorted(labels.items())),
        "loop_modes": dict(sorted(loops.items())),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "exclusions": [
            {"action_number": value.action_number, "detail": value.detail, "reason": value.reason}
            for value in plan.excluded
        ],
    }


def _atomic_no_clobber(payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{OUTPUT.name}.", suffix=".tmp", dir=OUTPUT.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if OUTPUT.exists():
            raise FileExistsError(f"refusing to replace audit: {OUTPUT}")
        temporary.replace(OUTPUT)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace audit: {OUTPUT}")
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    items = sorted(source["items"], key=lambda value: value["archive"]["sha256"].encode())
    rows = _load_journal()
    for position, item in enumerate(items, 1):
        digest = item["archive"]["sha256"]
        if digest in rows:
            _console(f"[{position}/{len(items)}] resume {item['resource']['title']}")
            continue
        try:
            row = _audit(item)
        except Exception as error:
            row = {
                "archive_bytes": item["archive"]["bytes"],
                "archive_sha256": digest,
                "decode_status": "audit_failed",
                "error": f"{type(error).__name__}: {error}",
                "resource": item["resource"],
            }
        _append(row)
        rows[digest] = row
        _console(f"[{position}/{len(items)}] {item['resource']['title']}: {row['decode_status']}")

    totals: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    loops: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    for row in rows.values():
        totals.update(
            {
                "archive_bytes": int(row["archive_bytes"]),
                "packs": 1,
                f"status_{row['decode_status']}": 1,
            }
        )
        for name in (
            "air_actions",
            "air_frame_occurrences",
            "sprite_occurrences",
            "unique_rgba_payloads",
            "linked_sprite_occurrences",
            "admitted_actions",
            "admitted_frame_occurrences",
        ):
            totals[name] += int(row.get(name, 0))
        labels.update(row.get("action_labels", {}))
        loops.update(row.get("loop_modes", {}))
        exclusions.update(row.get("exclusion_reasons", {}))
    artifact = {
        "schema_version": 1,
        "artifact_kind": "mugen_mffa_anime_zip_corpus_audit",
        "claim": "Exact inert archive/AIR/SFF audit; source art, not generated output.",
        "rights_scope": "Uploader/internal author claims only; no permissive-license inference.",
        "source_acquisition_index": {
            "path": str(SOURCE),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "totals": dict(sorted(totals.items())),
        "action_labels": dict(sorted(labels.items())),
        "loop_modes": dict(sorted(loops.items())),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "packs": [rows[key] for key in sorted(rows)],
    }
    payload = (
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    DiskGuard(ROOT, 100 * 1024**3).require_capacity(len(payload), label="MFFA ZIP audit")
    _atomic_no_clobber(payload)
    _console(
        json.dumps(
            {"audit_sha256": hashlib.sha256(payload).hexdigest(), **artifact["totals"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
