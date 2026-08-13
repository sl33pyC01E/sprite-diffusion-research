from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.widelands import (
    EXPECTED_WIDELANDS_ARCHIVE_SHA256,
    WIDELANDS_COMMIT,
    WidelandsArchiveError,
    WidelandsParseError,
    audit_known_widelands_archive,
    audit_widelands_archive,
    known_widelands_cas_path,
)

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "51/09/51093e130b2f4098716bc442ce3d8566d94b4590cfa233ba1c74e5d173a48186"
)

_ENGINE_FILES = (
    "src/logic/map_objects/map_object.cc",
    "src/graphic/animation/animation.cc",
    "src/graphic/animation/nonpacked_animation.cc",
    "src/graphic/animation/spritesheet_animation.cc",
    "src/io/filesystem/filesystem.cc",
)


def _png(size: tuple[int, int], color: tuple[int, int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", size, color).save(output, format="PNG")
    return output.getvalue()


def _write_png(
    archive: ZipFile,
    root: str,
    logical_path: str,
    size: tuple[int, int],
    ordinal: int,
) -> None:
    archive.writestr(
        f"{root}/{logical_path}",
        _png(size, ((ordinal * 31) % 255, (ordinal * 47) % 255, 80, 255)),
    )


def _fixture_archive(tmp_path: Path) -> Path:
    path = tmp_path / "widelands-fixture.zip"
    root = "widelands-fixture"
    worker_manifest = r"""
-- A commented constructor and table must not be audited.
-- wl.Descriptions():new_worker_type { name = "phantom" }
local dirname = path.dirname(__file__)

spritesheets = {
   walk = {
      fps = 20,
      frames = 3,
      rows = 2,
      columns = 2,
      directional = true,
      hotspot = { -1, 6 },
   },
   working = {
      directory = dirname .. "gear",
      fps = 4,
      frames = 2,
      rows = 1,
      columns = 2,
      hotspot = { 3, 7 },
      play_once = true,
   },
}

wl.Descriptions():new_worker_type {
   name = "fixture_worker",
   animation_directory = dirname,
   animations = {
      idle = {
         hotspot = { 2, 8 },
      },
   },
   spritesheets = spritesheets,
}
"""
    critter_manifest = r"""
local dirname = path.dirname(__file__)
wl.Descriptions():new_critter_type{
   name = "fixture_wolf",
   animation_directory = dirname,
   spritesheets = {
      idle = {
         frames = 1,
         rows = 1,
         columns = 1,
         hotspot = { 2, 4 },
      },
      eating = {
         basename = "idle",
         frames = 1,
         rows = 1,
         columns = 1,
         hotspot = { 2, 4 },
      },
      walk = {
         fps = 10,
         frames = 2,
         rows = 1,
         columns = 2,
         directional = true,
         hotspot = { 3, 5 },
      },
   },
}
"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for logical_path, payload in {
            "COPYING": b"GNU GENERAL PUBLIC LICENSE Version 2\n",
            "CREDITS": b"Fixture contributors\n",
            "data/txts/LICENSE.lua": b"GPL V2.0 or any later version\n",
            "data/txts/developers.json": b'{"developers": []}\n',
        }.items():
            archive.writestr(f"{root}/{logical_path}", payload)
        for logical_path in _ENGINE_FILES:
            archive.writestr(f"{root}/{logical_path}", f"evidence:{logical_path}\n")
        worker_root = "data/tribes/workers/test/fixture_worker"
        critter_root = "data/world/critters/fixture_wolf"
        archive.writestr(f"{root}/{worker_root}/init.lua", worker_manifest)
        archive.writestr(f"{root}/{critter_root}/init.lua", critter_manifest)

        ordinal = 1
        for stem in ("idle_00", "idle_01"):
            _write_png(archive, root, f"{worker_root}/{stem}.png", (5, 9), ordinal)
            ordinal += 1
            _write_png(archive, root, f"{worker_root}/{stem}_pc.png", (5, 9), ordinal)
            ordinal += 1
        for direction in ("ne", "e", "se", "sw", "w", "nw"):
            stem = f"walk_{direction}_1"
            _write_png(archive, root, f"{worker_root}/{stem}.png", (8, 8), ordinal)
            ordinal += 1
            _write_png(archive, root, f"{worker_root}/{stem}_pc.png", (8, 8), ordinal)
            ordinal += 1
        _write_png(archive, root, f"{worker_root}/gear/working_1.png", (10, 5), ordinal)
        ordinal += 1
        _write_png(archive, root, f"{worker_root}/gear/working_1_pc.png", (10, 5), ordinal)
        ordinal += 1
        _write_png(archive, root, f"{worker_root}/menu.png", (4, 4), ordinal)
        ordinal += 1
        _write_png(archive, root, f"{worker_root}/health_level0.png", (3, 3), ordinal)
        ordinal += 1
        _write_png(archive, root, f"{worker_root}/shadow.png", (6, 2), ordinal)
        ordinal += 1

        _write_png(archive, root, f"{critter_root}/idle_1.png", (4, 6), ordinal)
        ordinal += 1
        for direction in ("ne", "e", "se", "sw", "w", "nw"):
            _write_png(
                archive,
                root,
                f"{critter_root}/walk_{direction}_1.png",
                (8, 6),
                ordinal,
            )
            ordinal += 1
        _write_png(archive, root, f"{critter_root}/menu.png", (4, 4), ordinal)
    return path


def _entity(audit: object, entity_id: str) -> object:
    matches = [entity for entity in audit.entities if entity.entity_id == entity_id]  # type: ignore[attr-defined]
    assert len(matches) == 1
    return matches[0]


def _tracks(entity: object, name: str) -> list[object]:
    return [track for track in entity.animations if track.declared_name == name]  # type: ignore[attr-defined]


def test_synthetic_archive_preserves_frames_directions_timing_and_roles(tmp_path: Path) -> None:
    archive_path = _fixture_archive(tmp_path)
    audit = audit_widelands_archive(archive_path)

    assert audit.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert audit.counts.worker_manifests == 1
    assert audit.counts.critter_manifests == 1
    assert audit.counts.entities == 2
    assert dict(audit.counts.constructor_role_counts) == {"critter": 1, "worker": 1}
    assert audit.counts.animation_declarations == 16
    assert audit.counts.direction_tracks == 12
    assert audit.counts.exact_tracks == 16
    assert audit.counts.quarantined_tracks == 0
    assert audit.counts.primary_frames == 36
    assert audit.counts.primary_animation_images == 16
    assert audit.counts.playercolor_mask_images == 9
    assert audit.counts.ui_icon_images == 2
    assert audit.counts.equipment_or_status_images == 1
    assert audit.counts.unreferenced_layer_or_effect_images == 1
    assert audit.counts.worker_tree_pngs == 21
    assert audit.counts.critter_tree_pngs == 8
    assert audit.counts.surplus_spritesheet_cells == 6

    worker = _entity(audit, "fixture_worker")
    assert worker.entity_class == "humanoid"
    assert worker.animation_directory == "data/tribes/workers/test/fixture_worker"
    idle = _tracks(worker, "idle")
    assert len(idle) == 1
    assert idle[0].representation == "numbered_files"
    assert idle[0].frame_duration_milliseconds == 250
    assert [frame.source_logical_path for frame in idle[0].frames] == [
        "data/tribes/workers/test/fixture_worker/idle_00.png",
        "data/tribes/workers/test/fixture_worker/idle_01.png",
    ]

    walk = _tracks(worker, "walk")
    assert [track.direction for track in walk] == [
        "northeast",
        "east",
        "southeast",
        "southwest",
        "west",
        "northwest",
    ]
    northeast = next(track for track in walk if track.direction == "northeast")
    assert northeast.frame_duration_milliseconds == 50
    assert northeast.hotspot == (-1, 6)
    assert [(frame.x, frame.y, frame.width, frame.height) for frame in northeast.frames] == [
        (0, 0, 4, 4),
        (4, 0, 4, 4),
        (0, 4, 4, 4),
    ]
    assert northeast.source_images[0].playercolor_mask_path is not None

    working = _tracks(worker, "working")
    assert len(working) == 1
    assert working[0].normalized_action == "work"
    assert working[0].source_directory.endswith("/gear")
    assert working[0].loop_mode == "one_shot"
    assert working[0].frame_duration_milliseconds == 250

    wolf = _entity(audit, "fixture_wolf")
    assert wolf.entity_class == "animal"
    eating = _tracks(wolf, "eating")
    assert len(eating) == 1
    assert eating[0].normalized_action == "eat"
    assert eating[0].source_images[0].logical_path.endswith("/idle_1.png")

    roles = {image.logical_path: image.role for image in audit.auxiliary_images}
    assert roles["data/tribes/workers/test/fixture_worker/menu.png"] == "ui_icon"
    assert (
        roles["data/tribes/workers/test/fixture_worker/health_level0.png"]
        == "equipment_or_status_icon"
    )
    assert (
        roles["data/tribes/workers/test/fixture_worker/shadow.png"]
        == "unreferenced_layer_or_effect"
    )


def test_audit_hash_is_canonical_and_excludes_self_hash(tmp_path: Path) -> None:
    audit = audit_widelands_archive(_fixture_archive(tmp_path))
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
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("root/../escape.txt", "no")
    with pytest.raises(WidelandsArchiveError, match="unsafe archive member"):
        audit_widelands_archive(archive_path)


def test_known_archive_rejects_unpinned_fixture(tmp_path: Path) -> None:
    with pytest.raises(WidelandsArchiveError, match="SHA-256 mismatch"):
        audit_known_widelands_archive(_fixture_archive(tmp_path))


def test_dynamic_animation_table_is_rejected_instead_of_executed(tmp_path: Path) -> None:
    archive_path = tmp_path / "dynamic.zip"
    root = "widelands-dynamic"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        for logical_path in (
            "COPYING",
            "CREDITS",
            "data/txts/LICENSE.lua",
            "data/txts/developers.json",
            *_ENGINE_FILES,
        ):
            archive.writestr(f"{root}/{logical_path}", "evidence\n")
        archive.writestr(
            f"{root}/data/tribes/workers/test/dynamic/init.lua",
            """
local dirname = path.dirname(__file__)
wl.Descriptions():new_worker_type {
   name = "dynamic_worker",
   animation_directory = dirname,
   spritesheets = make_spritesheets(),
}
""",
        )
    with pytest.raises(WidelandsParseError, match="not a resolvable literal table"):
        audit_widelands_archive(archive_path)


def test_known_cas_path_uses_digest_sharding(tmp_path: Path) -> None:
    expected = (
        tmp_path
        / "objects"
        / "sha256"
        / EXPECTED_WIDELANDS_ARCHIVE_SHA256[:2]
        / EXPECTED_WIDELANDS_ARCHIVE_SHA256[2:4]
        / EXPECTED_WIDELANDS_ARCHIVE_SHA256
    )
    assert known_widelands_cas_path(tmp_path) == expected


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact Widelands CAS archive is absent")
def test_exact_cas_archive_audit() -> None:
    audit = audit_known_widelands_archive(EXACT_ARCHIVE)

    assert audit.archive_sha256 == EXPECTED_WIDELANDS_ARCHIVE_SHA256
    assert audit.commit == WIDELANDS_COMMIT
    assert audit.archive_size_bytes == 497_242_680
    assert audit.counts.archive_members == 26_840
    assert audit.counts.archive_files == 25_166
    assert audit.counts.archive_directories == 1_674
    assert audit.counts.worker_manifests == 161
    assert audit.counts.critter_manifests == 16
    assert dict(audit.counts.constructor_role_counts) == {
        "carrier": 10,
        "critter": 16,
        "ferry": 5,
        "soldier": 5,
        "worker": 141,
    }
    assert audit.counts.entities == 177
    assert audit.counts.animation_declarations == 2_275
    assert audit.counts.direction_tracks == 1_890
    assert audit.counts.exact_tracks == 2_275
    assert audit.counts.quarantined_tracks == 0
    assert audit.counts.primary_frames == 34_414
    assert audit.counts.primary_animation_images == 5_335
    assert audit.counts.playercolor_mask_images == 5_160
    assert audit.counts.ui_icon_images == 177
    assert audit.counts.equipment_or_status_images == 69
    assert audit.counts.unreferenced_layer_or_effect_images == 0
    assert audit.counts.worker_tree_pngs == 10_613
    assert audit.counts.critter_tree_pngs == 128
    assert dict(audit.counts.action_counts) == {
        "attack": 55,
        "carry": 810,
        "death": 20,
        "dodge": 32,
        "eat": 15,
        "idle": 179,
        "walk": 1_074,
        "work": 90,
    }
    assert dict(audit.counts.entity_class_counts) == {
        "animal": 21,
        "humanoid": 151,
        "vehicle": 5,
    }
    assert dict(audit.counts.representation_counts) == {
        "numbered_files": 107,
        "spritesheet": 2_168,
    }
    assert dict(audit.counts.scale_counts) == {
        "0.5": 1_087,
        "1": 2_074,
        "2": 1_087,
        "4": 1_087,
    }
    assert audit.counts.surplus_spritesheet_cells == 2_074
    assert audit.counts.duplicate_primary_image_hash_groups == 13
    assert audit.counts.duplicate_primary_image_hash_excess == 15
    assert audit.rights.root_license.sha256 == (
        "8177f97513213526df2cf6184d8ff986c675afb514d4e68a404010521b880643"
    )
    assert audit.rights.in_game_license.sha256 == (
        "081c8efe81ea36b1280f2ddfcb8f21642bdfcdf8473493fd1052b5bcb7a67fa5"
    )
    assert audit.audit_record_sha256 == (
        "2208e5ef94bbe6adbe80e2c668336ef04bcd79997e5794d9b81df5ded9ad9a86"
    )
