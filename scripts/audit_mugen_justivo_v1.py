from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spritelab.adapters.mugen import (  # noqa: E402
    audit_character_zip,
    decode_sff_v1,
    materialize_actions,
)
from spritelab.storage import DiskGuard  # noqa: E402

SOURCE = ROOT / "data/index/reports/mugen-justivo-acquisition-v2.json"
OUTPUT = ROOT / "data/index/reports/mugen-justivo-corpus-audit-v1.json"


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to replace corpus audit: {OUTPUT}")
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    packs: list[dict[str, object]] = []
    totals: Counter[str] = Counter()
    label_totals: Counter[str] = Counter()
    loop_totals: Counter[str] = Counter()
    exclusion_totals: Counter[str] = Counter()
    for item in source["items"]:
        archive_path = Path(item["archive"]["cas_path"])
        archive_bytes = archive_path.read_bytes()
        audit = audit_character_zip(archive_bytes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            sff_bytes = archive.read(audit.sff_member)
        sprites = decode_sff_v1(sff_bytes)
        plan = materialize_actions(audit.actions, sprites)
        labels = Counter(action.normalized_action or "unknown" for action in plan.admitted)
        loops = Counter(action.loop_mode for action in plan.admitted)
        exclusions = Counter(exclusion.reason for exclusion in plan.excluded)
        admitted_frames = sum(len(action.frames) for action in plan.admitted)
        rgba_hashes = {sprite.rgba_sha256 for sprite in sprites}
        packs.append(
            {
                "landing_claim": item["landing_claim"],
                "archive_sha256": audit.archive_sha256,
                "archive_bytes": audit.archive_bytes,
                "inventory_sha256": audit.inventory_sha256,
                "definition_members": audit.definition_members,
                "internal_names": [definition.name for definition in audit.definition_variants],
                "internal_display_names": [
                    definition.display_name for definition in audit.definition_variants
                ],
                "internal_authors": [definition.author for definition in audit.definition_variants],
                "air_member": audit.air_member,
                "sff_member": audit.sff_member,
                "sff_sha256": audit.sff_header.sha256,
                "sff_format_family": audit.sff_header.format_family,
                "sprite_occurrences": len(sprites),
                "sprite_keys": len(
                    {(sprite.group_number, sprite.image_number) for sprite in sprites}
                ),
                "unique_rgba_payloads": len(rgba_hashes),
                "linked_sprite_occurrences": sum(
                    sprite.linked_sprite_index is not None for sprite in sprites
                ),
                "air_actions": len(audit.actions),
                "air_frame_occurrences": sum(len(action.elements) for action in audit.actions),
                "admitted_actions": len(plan.admitted),
                "admitted_frame_occurrences": admitted_frames,
                "action_labels": dict(sorted(labels.items())),
                "loop_modes": dict(sorted(loops.items())),
                "exclusions": [
                    {
                        "action_number": exclusion.action_number,
                        "reason": exclusion.reason,
                        "detail": exclusion.detail,
                    }
                    for exclusion in plan.excluded
                ],
                "exclusion_reasons": dict(sorted(exclusions.items())),
                "executable_members": audit.executable_members,
                "runtime_logic_members": audit.runtime_logic_members,
            }
        )
        totals.update(
            {
                "packs": 1,
                "archive_bytes": audit.archive_bytes,
                "sprite_occurrences": len(sprites),
                "unique_rgba_payloads_pack_sum": len(rgba_hashes),
                "linked_sprite_occurrences": sum(
                    sprite.linked_sprite_index is not None for sprite in sprites
                ),
                "air_actions": len(audit.actions),
                "air_frame_occurrences": sum(len(action.elements) for action in audit.actions),
                "admitted_actions": len(plan.admitted),
                "admitted_frame_occurrences": admitted_frames,
                "excluded_actions": len(plan.excluded),
            }
        )
        label_totals.update(labels)
        loop_totals.update(loops)
        exclusion_totals.update(exclusions)
        print(item["landing_claim"]["name"], len(plan.admitted), admitted_frames)

    artifact = {
        "schema_version": 1,
        "artifact_kind": "mugen_character_corpus_audit",
        "source_acquisition_index": {
            "path": str(SOURCE),
            "file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "embedded_index_sha256": (
                "dd891a7daa4627b449b9a43f359bc80b99dd27d89fa85428879468150ddbe836"
            ),
        },
        "decoder": {
            "formats": ["SFF v1 PCX and linked sprites", "AIR", "character DEF"],
            "transparency": "palette index 0 -> alpha 0; all other indices -> alpha 255",
            "alignment": "AIR offset minus flip-adjusted SFF axis; union canvas; no resize",
            "unsupported": ["SFF v2 pixel decoding", "AIR scale/angle/blend", "runtime CMD/CNS"],
        },
        "claim": "Exact public-mirror format audit; source art, not generated output.",
        "rights_scope": "Unknown/unverified per pack; no permissive inference.",
        "totals": dict(sorted(totals.items())),
        "action_labels": dict(sorted(label_totals.items())),
        "loop_modes": dict(sorted(loop_totals.items())),
        "exclusion_reasons": dict(sorted(exclusion_totals.items())),
        "packs": packs,
    }
    payload = (
        json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    guard = DiskGuard(ROOT, 100 * 1024**3)
    guard.require_capacity(len(payload), label="MUGEN corpus audit")
    _atomic_no_clobber(OUTPUT, payload)
    print({"audit_sha256": hashlib.sha256(payload).hexdigest(), **artifact["totals"]})
    return 0


def _atomic_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to replace corpus audit: {path}")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
