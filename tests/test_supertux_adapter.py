from __future__ import annotations

import hashlib
import io
import json
import stat
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from PIL import Image

from spritelab.adapters.supertux import (
    EXPECTED_SUPERTUX_ARCHIVE_SHA256,
    SUPERTUX_COMMIT,
    SuperTuxArchiveError,
    audit_known_supertux_archive,
    audit_supertux_archive,
    known_supertux_cas_path,
)

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "98/ea/98ea15f57224ab3374fb5a3a1bfc538fa33790eecf60c5f2193d782e96b1abc5"
)

_EVIDENCE = (
    "LICENSE.txt",
    "README.md",
    "data/AUTHORS",
    "data/credits.stxt",
    "src/sprite/sprite_data.cpp",
    "src/sprite/sprite.cpp",
)


def _png(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _base_archive(tmp_path: Path, *, broken: bool = False) -> Path:
    path = tmp_path / ("supertux-broken.zip" if broken else "supertux-fixture.zip")
    root = "supertux-fixture"
    hero_manifest = r"""
(supertux-sprite
  (action
    (name "walk-left")
    (fps 20)
    (loops 1)
    (loop-frame 2)
    (hitbox 1 2 8 6)
    (images "a.png" "b.png" "a.png"))
  (action
    (name "walk-right")
    (fps 20)
    (hitbox 2 2 8 6)
    (mirror-action "walk-left"))
  (action
    (name "roof-left")
    (fps 20)
    (hitbox 1 2 8 6)
    (flip-action "walk-left"))
  (action
    (name "slow-left")
    (fps 5)
    (hitbox 9 9 1 1)
    (clone-action "walk-left"))
  (action
    (name "idle")
    (images "a.png")))
"""
    broken_manifest = r"""
(supertux-sprite
  (action (name "broken") (images "missing.png"))
  (action (name "same") (images "a.png"))
  (action (name "same") (mirror-action "same")))
"""
    effect_manifest = r"""
(supertux-sprite
  (action (name "default") (hitbox 0 0 4 4) (images "glow.png")))
"""
    module_manifest = r"""
(supertux-sprite
  (action (name "default") (hitbox 0 0 4 4) (images "hat.png")))
"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for evidence in _EVIDENCE:
            archive.writestr(f"{root}/{evidence}", f"evidence:{evidence}\n")
        hero_path = "data/images/creatures/tux/tux.sprite"
        archive.writestr(f"{root}/{hero_path}", broken_manifest if broken else hero_manifest)
        archive.writestr(
            f"{root}/data/images/creatures/crystallo/crystallo-overlay.sprite",
            effect_manifest,
        )
        archive.writestr(f"{root}/data/images/creatures/tux/santahat.sprite", module_manifest)
        for logical_path, payload in {
            "data/images/creatures/tux/a.png": _png((12, 8), (20, 40, 60, 255)),
            "data/images/creatures/tux/b.png": _png((10, 7), (70, 80, 90, 180)),
            "data/images/creatures/tux/unreferenced.png": _png((3, 5), (1, 2, 3, 255)),
            "data/images/creatures/crystallo/glow.png": _png((4, 4), (0, 200, 255, 90)),
            "data/images/creatures/tux/hat.png": _png((4, 4), (200, 10, 10, 255)),
        }.items():
            archive.writestr(f"{root}/{logical_path}", payload)
    return path


def _manifest(audit: object, path: str) -> object:
    matches = [item for item in audit.manifests if item.logical_path == path]  # type: ignore[attr-defined]
    assert len(matches) == 1
    return matches[0]


def _action(manifest: object, name: str) -> object:
    matches = [item for item in manifest.actions if item.name == name]  # type: ignore[attr-defined]
    assert len(matches) == 1
    return matches[0]


def test_resolves_frame_order_timing_aliases_geometry_and_roles(tmp_path: Path) -> None:
    audit = audit_supertux_archive(_base_archive(tmp_path))

    hero = _manifest(audit, "data/images/creatures/tux/tux.sprite")
    assert hero.role == "complete_entity"
    assert hero.entity_class == "humanoid"
    walk_left = _action(hero, "walk-left")
    assert walk_left.normalized_action == "walk"
    assert walk_left.direction == "left"
    assert walk_left.effective_fps == 20
    assert walk_left.frame_duration_milliseconds == 50
    assert walk_left.has_custom_loops
    assert walk_left.effective_loops == 1
    assert walk_left.effective_loop_frame == 2
    assert walk_left.hitbox == (1.0, 2.0, 8.0, 6.0)
    assert [frame.requested_path for frame in walk_left.frames] == ["a.png", "b.png", "a.png"]
    assert [frame.sha256 for frame in walk_left.frames][0] == [
        frame.sha256 for frame in walk_left.frames
    ][2]

    walk_right = _action(hero, "walk-right")
    assert walk_right.alias_chain == ("mirror:walk-left",)
    assert {frame.transform for frame in walk_right.frames} == {"horizontal_flip"}
    roof = _action(hero, "roof-left")
    assert roof.alias_chain == ("flip:walk-left",)
    assert {frame.transform for frame in roof.frames} == {"vertical_flip"}

    clone = _action(hero, "slow-left")
    assert clone.declared_fps == 5
    assert clone.effective_fps == 20
    assert clone.hitbox == walk_left.hitbox
    assert clone.effective_loop_frame == 2
    assert clone.alias_chain == ("clone:walk-left",)

    idle = _action(hero, "idle")
    assert idle.hitbox == (0.0, 0.0, 12.0, 8.0)
    effect = _manifest(audit, "data/images/creatures/crystallo/crystallo-overlay.sprite")
    module = _manifest(audit, "data/images/creatures/tux/santahat.sprite")
    assert effect.role == "effect_layer"
    assert module.role == "modular_component"
    assert not effect.complete_entity
    assert audit.counts.unreferenced_creature_tree_pngs == 1


def test_missing_and_duplicate_self_alias_are_preserved_as_quarantine(tmp_path: Path) -> None:
    audit = audit_supertux_archive(_base_archive(tmp_path, broken=True))
    hero = _manifest(audit, "data/images/creatures/tux/tux.sprite")
    broken = _action(hero, "broken")
    assert broken.frames[0].logical_path.endswith("/missing.png")
    assert not broken.frames[0].exists
    assert "missing_source_image" in broken.quarantine_reasons

    same = [action for action in hero.actions if action.name == "same"]
    assert len(same) == 2
    assert not same[0].effective_declaration
    assert "superseded_duplicate_action" in same[0].quarantine_reasons
    assert same[1].effective_declaration
    assert not same[1].frames
    assert "self_alias_clears_source_frames" in same[1].quarantine_reasons
    assert audit.counts.duplicate_action_name_excess == 1
    assert audit.counts.empty_effective_actions == 1


def test_audit_hash_is_canonical_and_excludes_self_hash(tmp_path: Path) -> None:
    audit = audit_supertux_archive(_base_archive(tmp_path))
    payload = audit.to_dict()
    stored_hash = payload.pop("audit_record_sha256")
    expected = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    assert stored_hash == expected
    assert audit.canonical_json() == json.dumps(
        audit.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def test_archive_validation_rejects_path_traversal(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("root/../escape.txt", "no")
    with pytest.raises(SuperTuxArchiveError, match="unsafe archive member"):
        audit_supertux_archive(path)


def test_archive_validation_rejects_escaping_symlink(tmp_path: Path) -> None:
    path = tmp_path / "unsafe-link.zip"
    with ZipFile(path, "w") as archive:
        link = ZipInfo("root/a/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(link, "../../../outside")
    with pytest.raises(SuperTuxArchiveError, match="escaping symlink target"):
        audit_supertux_archive(path)


def test_known_archive_rejects_unpinned_fixture(tmp_path: Path) -> None:
    with pytest.raises(SuperTuxArchiveError, match="SHA-256 mismatch"):
        audit_known_supertux_archive(_base_archive(tmp_path))


def test_known_cas_path_uses_digest_sharding(tmp_path: Path) -> None:
    assert known_supertux_cas_path(tmp_path) == (
        tmp_path
        / "objects"
        / "sha256"
        / EXPECTED_SUPERTUX_ARCHIVE_SHA256[:2]
        / EXPECTED_SUPERTUX_ARCHIVE_SHA256[2:4]
        / EXPECTED_SUPERTUX_ARCHIVE_SHA256
    )


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact SuperTux CAS archive is absent")
def test_exact_cas_archive_audit() -> None:
    audit = audit_known_supertux_archive(EXACT_ARCHIVE)

    assert audit.archive_sha256 == EXPECTED_SUPERTUX_ARCHIVE_SHA256
    assert audit.commit == SUPERTUX_COMMIT
    assert audit.archive_size_bytes == 290_571_350
    assert audit.inventory_sha256 == (
        "2da2740e59deeb960db9d24505171e7a97ab2cc5b3968b82d353f643927c48d2"
    )
    assert audit.counts.archive_members == 6_708
    assert audit.counts.archive_files == 6_320
    assert audit.counts.archive_directories == 387
    assert audit.counts.archive_symlinks == 1
    assert audit.counts.creature_manifests == 135
    assert audit.counts.complete_entity_manifests == 96
    assert audit.counts.modular_component_manifests == 23
    assert audit.counts.effect_layer_manifests == 15
    assert audit.counts.deprecated_manifests == 1
    assert audit.counts.action_declarations == 1_151
    assert audit.counts.effective_actions == 1_150
    assert audit.counts.direct_image_actions == 662
    assert audit.counts.mirror_alias_actions == 461
    assert audit.counts.flip_alias_actions == 16
    assert audit.counts.clone_alias_actions == 12
    assert audit.counts.exact_complete_tracks == 1_010
    assert audit.counts.quarantined_effective_tracks == 12
    assert audit.counts.resolved_frame_occurrences == 7_687
    assert audit.counts.direct_image_occurrences == 4_134
    assert audit.counts.unique_referenced_images == 1_841
    assert audit.counts.missing_image_reference_occurrences == 25
    assert audit.counts.unique_missing_images == 23
    assert audit.counts.creature_tree_pngs == 1_940
    assert audit.counts.referenced_creature_tree_pngs == 1_825
    assert audit.counts.referenced_external_pngs == 16
    assert audit.counts.unreferenced_creature_tree_pngs == 115
    assert audit.counts.duplicate_creature_image_hash_groups == 30
    assert audit.counts.duplicate_creature_image_hash_excess == 33
    assert audit.counts.duplicate_action_name_excess == 1
    assert audit.counts.empty_effective_actions == 1
    assert dict(audit.counts.entity_class_counts) == {
        "animal": 20,
        "construct": 32,
        "elemental": 7,
        "humanoid": 3,
        "monster": 21,
        "plant": 13,
    }
    assert dict(audit.counts.direction_counts) == {
        "down": 7,
        "down_left": 7,
        "down_right": 7,
        "left": 430,
        "none": 110,
        "right": 429,
        "up": 4,
        "up_left": 8,
        "up_right": 8,
    }
    assert dict(audit.counts.transform_counts) == {
        "horizontal_flip": 3_303,
        "horizontal_vertical_flip": 16,
        "identity": 3_752,
        "vertical_flip": 32,
    }
    assert audit.rights.root_license.sha256 == (
        "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"
    )
    assert audit.rights.authors.sha256 == (
        "bcfb4d94d5bcdae86923e0896788251674ffdea80311954dd00cf05e1871f527"
    )
    assert audit.audit_record_sha256 == (
        "1b5fd92ffbfe2dc7fbd9ca7f53d0c7fd2b540b84f8a3da6f0fbe722f09183703"
    )
