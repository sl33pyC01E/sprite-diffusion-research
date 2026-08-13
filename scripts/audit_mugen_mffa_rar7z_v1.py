"""Power-loss-resumable inert audit of acquired MFFA RAR and 7z characters."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
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
    parse_character_def,
)
from spritelab.storage import DiskGuard  # noqa: E402

SOURCE = ROOT / "data/index/reports/mugen-mffa-anime-rar7z-acquisition-v1.json"
JOURNAL = ROOT / "data/index/reports/mugen-mffa-anime-rar7z-corpus-audit-v1.jsonl"
OUTPUT = ROOT / "data/index/reports/mugen-mffa-anime-rar7z-corpus-audit-v1.json"
TAR = "tar.exe"


def _console(message: str) -> None:
    print(message.encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append(row: dict[str, object]) -> None:
    payload = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    with JOURNAL.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


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


def _tar(path: Path, *arguments: str, timeout: int = 300) -> bytes:
    command = [TAR, *arguments, str(path)] if arguments == ("-tf",) else [TAR, *arguments]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[-1000:]
        raise ValueError(f"libarchive failed ({result.returncode}): {detail}")
    return result.stdout


def _inventory(path: Path) -> tuple[str, ...]:
    raw = _tar(path, "-tf")
    names = raw.decode("utf-8", "surrogateescape").splitlines()
    normalized: list[str] = []
    folded: set[str] = set()
    for name in names:
        slash_name = name.replace("\\", "/")
        if slash_name.endswith("/"):
            continue
        while slash_name.startswith("./"):
            slash_name = slash_name[2:]
        clean = PurePosixPath(slash_name).as_posix()
        parts = PurePosixPath(clean).parts
        if not clean or clean.startswith("/") or ".." in parts or ":" in parts[0]:
            raise ValueError(f"unsafe archive member path {name!r}")
        key = clean.casefold()
        if key in folded:
            raise ValueError(f"case-colliding archive member {clean!r}")
        folded.add(key)
        normalized.append(clean)
    return tuple(normalized)


def _extract(path: Path, member: str) -> bytes:
    result = subprocess.run(
        [TAR, "-xOf", str(path), member],
        capture_output=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[-1000:]
        raise ValueError(f"cannot stream {member!r} ({result.returncode}): {detail}")
    return result.stdout


def _resolve(names: tuple[str, ...], owner: str, reference: str | None) -> str:
    if not reference:
        raise ValueError("missing DEF media reference")
    wanted = str(PurePosixPath(owner).parent / reference.replace("\\", "/")).lstrip("/")
    matches = [name for name in names if name.casefold() == wanted.casefold()]
    if len(matches) != 1:
        raise ValueError(f"DEF reference {reference!r} resolves to {len(matches)} members")
    return matches[0]


def _synthetic_character(path: Path, names: tuple[str, ...]) -> tuple[bytes, int]:
    definitions: list[tuple[str, object, str, str, str | None]] = []
    for name in names:
        if PurePosixPath(name).suffix.casefold() != ".def":
            continue
        try:
            definition = parse_character_def(_extract(path, name))
            air = _resolve(names, name, definition.file("anim"))
            sff = _resolve(names, name, definition.file("sprite"))
            pal = (
                _resolve(names, name, definition.file("pal1")) if definition.file("pal1") else None
            )
        except ValueError:
            continue
        definitions.append((name, definition, air, sff, pal))
    if not definitions:
        raise ValueError("no character DEF with resolvable AIR and SFF")
    media_pairs = {(row[2].casefold(), row[3].casefold()) for row in definitions}
    if len(media_pairs) != 1:
        raise ValueError(f"character definitions resolve to {len(media_pairs)} media pairs")
    selected_names = {row[0] for row in definitions}
    selected_names.update(row[2] for row in definitions)
    selected_names.update(row[3] for row in definitions)
    selected_names.update(row[4] for row in definitions if row[4] is not None)
    output = io.BytesIO()
    total_bytes = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(selected_names, key=lambda value: value.encode("utf-8")):
            payload = _extract(path, name)
            total_bytes += len(payload)
            if total_bytes > 1024**3:
                raise ValueError("selected media closure exceeds 1 GiB")
            archive.writestr(name, payload)
    return output.getvalue(), total_bytes


def _audit(item: dict[str, object]) -> dict[str, object]:
    archive = item["archive"]
    path = Path(archive["cas_path"])
    if path.stat().st_size != archive["bytes"] or _sha_file(path) != archive["sha256"]:
        raise ValueError("archive CAS identity mismatch")
    names = _inventory(path)
    synthetic, selected_bytes = _synthetic_character(path, names)
    audit = audit_character_zip(synthetic)
    with zipfile.ZipFile(io.BytesIO(synthetic)) as selected:
        sff_payload = selected.read(audit.sff_member)
        if audit.sff_header.format_family == "sff_v1":
            reference = audit.definition.file("pal1")
            initial_palette = None
            if reference:
                palette_name = _resolve(
                    tuple(selected.namelist()), audit.definition_member, reference
                )
                palette = selected.read(palette_name)
                initial_palette = palette[:768] if len(palette) >= 768 else None
            sprites = decode_sff_v1(sff_payload, initial_palette_rgb=initial_palette)
            palette_count = None
            pixel_formats = None
            status = "decoded_sff_v1"
        else:
            sprites, palettes = decode_sff_v2(sff_payload)
            palette_count = len(palettes)
            pixel_formats = sorted({sprite.pixel_format for sprite in sprites})
            status = "decoded_sff_v2"
    plan = materialize_actions(audit.actions, sprites)
    labels = Counter(value.normalized_action or "unknown" for value in plan.admitted)
    loops = Counter(value.loop_mode for value in plan.admitted)
    exclusions = Counter(value.reason for value in plan.excluded)
    return {
        "action_labels": dict(sorted(labels.items())),
        "admitted_actions": len(plan.admitted),
        "admitted_frame_occurrences": sum(len(value.frames) for value in plan.admitted),
        "air_actions": len(audit.actions),
        "air_frame_occurrences": sum(len(value.elements) for value in audit.actions),
        "air_member": audit.air_member,
        "archive_bytes": archive["bytes"],
        "archive_format": PurePosixPath(item["candidate"]["filename"]).suffix.casefold(),
        "archive_sha256": archive["sha256"],
        "decode_status": status,
        "definition_members": list(audit.definition_members),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "internal_authors": [value.author for value in audit.definition_variants],
        "internal_display_names": [value.display_name for value in audit.definition_variants],
        "internal_names": [value.name for value in audit.definition_variants],
        "loop_modes": dict(sorted(loops.items())),
        "member_count": len(names),
        "palette_count": palette_count,
        "pixel_formats": pixel_formats,
        "resource": item["resource"],
        "selected_media_bytes": selected_bytes,
        "sff_format_family": audit.sff_header.format_family,
        "sff_member": audit.sff_member,
        "sff_sha256": audit.sff_header.sha256,
        "sprite_occurrences": len(sprites),
        "unique_rgba_payloads": len({value.rgba_sha256 for value in sprites}),
    }


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace {OUTPUT}")
    DiskGuard(ROOT, 100 * 1024**3).require_capacity(1024**2, label="RAR/7z audit")
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    rows = _load_journal()
    items = sorted(source["items"], key=lambda value: value["archive"]["sha256"])
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

    packs = [rows[item["archive"]["sha256"]] for item in items]
    totals: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    loops: Counter[str] = Counter()
    exclusions: Counter[str] = Counter()
    for row in packs:
        totals["packs"] += 1
        totals[f"status_{row['decode_status']}"] += 1
        totals["archive_bytes"] += int(row["archive_bytes"])
        for key in (
            "sprite_occurrences",
            "unique_rgba_payloads",
            "air_actions",
            "air_frame_occurrences",
            "admitted_actions",
            "admitted_frame_occurrences",
        ):
            totals[key] += int(row.get(key, 0))
        actions.update(row.get("action_labels", {}))
        loops.update(row.get("loop_modes", {}))
        exclusions.update(row.get("exclusion_reasons", {}))
    artifact = {
        "schema_version": 1,
        "artifact_kind": "mugen_mffa_anime_rar7z_corpus_audit",
        "claim": "Exact inert libarchive/AIR/SFF audit; source art, not generated output.",
        "source_acquisition_index": {
            "path": str(SOURCE),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "trust_boundary": "No member paths unpacked and no character code executed.",
        "rights_scope": "Uploader/internal author claims only; no permissive-license inference.",
        "totals": dict(sorted(totals.items())),
        "action_labels": dict(sorted(actions.items())),
        "loop_modes": dict(sorted(loops.items())),
        "exclusion_reasons": dict(sorted(exclusions.items())),
        "packs": packs,
    }
    payload = (
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    with OUTPUT.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    print({"path": str(OUTPUT), "sha256": hashlib.sha256(payload).hexdigest()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
