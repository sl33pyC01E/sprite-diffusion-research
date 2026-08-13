from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from spritelab.adapters.tgstation import (
    EXPECTED_TGSTATION_ARCHIVE_SHA256,
    EXPECTED_TGSTATION_AUDIT_RECORD_SHA256,
    EXPECTED_TGSTATION_INVENTORY_SHA256,
    TGSTATION_COMMIT,
    TgstationArchiveError,
    audit_known_tgstation_archive,
    audit_tgstation_archive,
    known_tgstation_cas_path,
)

EXACT_ARCHIVE = known_tgstation_cas_path(Path(__file__).resolve().parents[1] / "data" / "raw")


def _dmi_bytes(
    description: str,
    *,
    cell_size: tuple[int, int],
    cell_count: int,
    columns: int,
) -> bytes:
    width, height = cell_size
    rows = (cell_count + columns - 1) // columns
    image = Image.new("RGBA", (columns * width, rows * height), (0, 0, 0, 0))
    for index in range(cell_count):
        color = (
            (index * 37 + 11) % 256,
            (index * 71 + 23) % 256,
            (index * 109 + 41) % 256,
            255,
        )
        cell = Image.new("RGBA", cell_size, color)
        image.paste(cell, ((index % columns) * width, (index // columns) * height))
    metadata = PngInfo()
    metadata.add_text("Description", description, zip=True)
    output = BytesIO()
    image.save(output, format="PNG", pnginfo=metadata)
    return output.getvalue()


def _fixture_archive(tmp_path: Path) -> Path:
    root = "tgstation-fixture"
    path = tmp_path / "tgstation.zip"
    animal_description = """# BEGIN DMI
version = 4.0
	width = 32
	height = 32
state = "wolf"
	dirs = 4
	frames = 1
state = "wolf"
	dirs = 4
	frames = 3
	delay = 1,2,3
	movement = 1
state = "wolf_attack"
	dirs = 4
	frames = 2
	delay = 0.5,1
	loop = 1
	rewind = 1
state = "wolf_dead"
	dirs = 1
	frames = 1
# END DMI"""
    parts_description = """# BEGIN DMI
version = 4.0
	width = 32
	height = 32
state = "head"
	dirs = 4
	frames = 1
# END DMI"""
    bad_timing_description = """# BEGIN DMI
version = 4.0
	width = 16
	height = 16
state = "blob_attack"
	dirs = 1
	frames = 2
	delay = 1,2,3
# END DMI"""
    malformed_description = """# BEGIN DMI
version = 4.0
	width = 16
	height = 16
state = "mystery"
	dirs = 1
	frames = 1
	unsupported = 7
# END DMI"""
    animal = _dmi_bytes(animal_description, cell_size=(32, 32), cell_count=25, columns=5)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/LICENSE", "GNU AFFERO GENERAL PUBLIC LICENSE\n")
        archive.writestr(f"{root}/GPLv3.txt", "GNU GENERAL PUBLIC LICENSE\n")
        archive.writestr(
            f"{root}/README.md",
            "All assets including icons are under a Creative Commons 3.0 BY-SA "
            "license unless otherwise indicated.\n",
        )
        archive.writestr(f"{root}/tools/dmi/__init__.py", "DIR_ORDER = [2, 1, 4, 8]\n")
        archive.writestr(f"{root}/icons/mob/simple/animal.dmi", animal)
        archive.writestr(f"{root}/icons/mob/simple/animal_copy.dmi", animal)
        archive.writestr(
            f"{root}/icons/mob/human/bodyparts.dmi",
            _dmi_bytes(parts_description, cell_size=(32, 32), cell_count=4, columns=2),
        )
        archive.writestr(
            f"{root}/icons/mob/simple/bad.dmi",
            _dmi_bytes(bad_timing_description, cell_size=(16, 16), cell_count=2, columns=2),
        )
        archive.writestr(
            f"{root}/icons/mob/simple/malformed.dmi",
            _dmi_bytes(malformed_description, cell_size=(16, 16), cell_count=1, columns=1),
        )
    return path


def _pack(audit, logical_path: str):
    return next(pack for pack in audit.packs if pack.logical_path == logical_path)


def _state(pack, name: str, *, movement: bool = False):
    return next(state for state in pack.states if state.name == name and state.movement == movement)


def test_synthetic_audit_preserves_ztxt_geometry_timing_roles_and_lineage(
    tmp_path: Path,
) -> None:
    archive_path = _fixture_archive(tmp_path)
    before = archive_path.stat()
    audit = audit_tgstation_archive(archive_path)
    after = archive_path.stat()

    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert audit.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert audit.counts.mob_dmi_files == 5
    assert audit.counts.parsed_dmi_files == 4
    assert audit.counts.malformed_dmi_files == 1
    assert audit.malformed_dmis[0].logical_path.endswith("malformed.dmi")
    assert "unsupported DMI metadata key" in audit.malformed_dmis[0].error

    animal = _pack(audit, "icons/mob/simple/animal.dmi")
    assert animal.pack_role == "complete_entity_candidate"
    assert animal.entity_class == "animal"
    assert animal.metadata.chunk_type == "zTXt"
    assert animal.metadata.keyword == "Description"
    assert animal.metadata.compression_method == 0
    assert animal.description_verbatim.startswith("# BEGIN DMI\nversion = 4.0")
    assert animal.declared_source_cells == animal.grid_capacity == 25
    assert animal.lineage_key == f"github:tgstation/tgstation@{TGSTATION_COMMIT}"
    assert animal.blob_url.endswith(f"/blob/{TGSTATION_COMMIT}/icons/mob/simple/animal.dmi")

    idle = _state(animal, "wolf")
    walk = _state(animal, "wolf", movement=True)
    attack = _state(animal, "wolf_attack")
    dead = _state(animal, "wolf_dead")
    assert (idle.normalized_action, walk.normalized_action) == ("idle", "walk")
    assert (attack.normalized_action, dead.normalized_action) == ("attack", "death")
    assert walk.delay_decisecond_literals == ("1", "2", "3")
    assert walk.durations_milliseconds == (100, 200, 300)
    assert attack.durations_milliseconds == (50, 100)
    assert attack.loop_count == 1 and attack.rewind
    assert attack.playback_semantics == "finite_declared_loop_count_with_rewind"
    assert attack.eligible_animated_action_sequence

    assert walk.source_cell_start == 4
    assert [frame.source_cell_index for frame in walk.frames[:8]] == list(range(4, 12))
    assert [frame.direction for frame in walk.frames[:4]] == [
        "south",
        "north",
        "east",
        "west",
    ]
    assert [(frame.left, frame.top) for frame in walk.frames[:4]] == [
        (128, 0),
        (0, 32),
        (32, 32),
        (64, 32),
    ]
    assert len({frame.rgba_sha256 for frame in walk.frames}) == len(walk.frames)

    action_set = next(
        item
        for item in audit.entity_action_sets
        if item.logical_path == animal.logical_path and item.entity_cue == "wolf"
    )
    assert action_set.actions == ("attack", "death", "idle", "walk")
    assert action_set.steerable and action_set.has_animated_action
    assert action_set.animated_action_sequence_count == 2

    parts = _pack(audit, "icons/mob/human/bodyparts.dmi")
    assert parts.pack_role == "modular_component_pack"
    assert parts.states[0].role == "modular_component"
    assert not parts.states[0].eligible_complete_entity_sequence

    bad = _state(_pack(audit, "icons/mob/simple/bad.dmi"), "blob_attack")
    assert bad.durations_milliseconds == ()
    assert "delay_count_mismatch" in bad.quarantine_reasons
    assert all(frame.duration_milliseconds is None for frame in bad.frames)

    assert len(audit.duplicate_dmi_groups) == 1
    assert audit.duplicate_dmi_groups[0].logical_paths == (
        "icons/mob/simple/animal.dmi",
        "icons/mob/simple/animal_copy.dmi",
    )
    assert audit.rights.asset_license_expression == "CC-BY-SA-3.0"
    assert not audit.rights.per_file_author_manifest_present
    assert (
        audit.audit_record_sha256
        == hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in audit.to_dict().items()
                    if key != "audit_record_sha256"
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
    )


def test_archive_validation_rejects_path_traversal(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("root/../escape.dmi", b"no")

    with pytest.raises(TgstationArchiveError, match="unsafe archive member"):
        audit_tgstation_archive(path)


def test_known_archive_rejects_unpinned_fixture(tmp_path: Path) -> None:
    with pytest.raises(TgstationArchiveError, match="archive size mismatch"):
        audit_known_tgstation_archive(_fixture_archive(tmp_path))


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact /tg/station CAS archive absent")
def test_exact_cas_archive_identity_semantics_rights_and_counts() -> None:
    audit = audit_known_tgstation_archive(EXACT_ARCHIVE)
    counts = audit.counts

    assert audit.archive_sha256 == EXPECTED_TGSTATION_ARCHIVE_SHA256
    assert audit.inventory_sha256 == EXPECTED_TGSTATION_INVENTORY_SHA256
    assert audit.audit_record_sha256 == EXPECTED_TGSTATION_AUDIT_RECORD_SHA256
    assert audit.commit == TGSTATION_COMMIT
    assert audit.archive_size_bytes == 193_871_729
    assert (
        counts.archive_members,
        counts.archive_files,
        counts.archive_directories,
        counts.archive_symlinks,
    ) == (19_584, 17_796, 1_788, 0)
    assert (counts.mob_dmi_files, counts.parsed_dmi_files, counts.malformed_dmi_files) == (
        401,
        401,
        0,
    )
    assert (counts.dmi_states, counts.declared_source_cells) == (11_862, 60_051)
    assert (
        counts.temporally_animated_states,
        counts.directional_states,
        counts.movement_states,
        counts.rewind_states,
        counts.finite_loop_states,
    ) == (1_314, 9_338, 107, 127, 174)
    assert (
        counts.delay_count_mismatch_states,
        counts.invalid_hotspot_states,
        counts.duplicate_runtime_key_excess,
    ) == (23, 32, 13)
    assert (
        counts.exact_capacity_dmis,
        counts.surplus_capacity_dmis,
        counts.unused_source_cells,
    ) == (127, 274, 1_457)
    assert (
        counts.complete_entity_candidate_states,
        counts.eligible_complete_entity_sequences,
        counts.eligible_action_sequences,
        counts.eligible_animated_action_sequences,
    ) == (1_686, 1_680, 1_209, 260)
    assert (
        counts.entity_action_sets,
        counts.steerable_entity_action_sets,
        counts.steerable_entity_action_sets_with_animation,
    ) == (1_150, 283, 140)
    assert dict(counts.entity_class_counts) == {
        "animal": 480,
        "creature": 146,
        "humanoid": 30,
        "monster": 550,
        "object": 16,
        "robot": 458,
    }
    assert dict(counts.image_mode_counts) == {"P": 148, "RGBA": 253}
    assert {pack.metadata.chunk_type for pack in audit.packs} == {"zTXt"}
    assert {pack.metadata.keyword for pack in audit.packs} == {"Description"}
    assert audit.rights.asset_license_expression == "CC-BY-SA-3.0"
    assert audit.rights.readme.sha256 == (
        "c785d87bb165d1d7d29d78a6e285dbf7875ffa4efd589d9b1d7256135f264420"
    )
    assert audit.engine_semantics.implementation.sha256 == (
        "ee64b893a87dd08a55942900c422053c906cbf72e5b4bcbd293a7d0e0dbe9d63"
    )
    assert not audit.rights.path_local_rights_documents

    animal = _pack(audit, "icons/mob/simple/animal.dmi")
    chicken_idle = _state(animal, "chicken_brown")
    chicken_walk = _state(animal, "chicken_brown", movement=True)
    parrot = _state(animal, "parrot_fly")
    assert (chicken_idle.normalized_action, chicken_walk.normalized_action) == (
        "idle",
        "walk",
    )
    assert chicken_walk.eligible_animated_action_sequence
    assert parrot.normalized_action == "fly"
    assert parrot.entity_cue == "parrot"
    assert animal.declared_source_cells == 989 and animal.unused_source_cells == 3

    assert [(group.sha256, group.logical_paths) for group in audit.duplicate_dmi_groups] == [
        (
            "c32afa45c7f2399a6ad42ffcdd58c543725d4f3258b2d5d7432cf61dc5b59cb2",
            ("icons/mob/cows.dmi", "icons/mob/simple/cows.dmi"),
        )
    ]
