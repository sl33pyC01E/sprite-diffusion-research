import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.ss14 import (
    EXPECTED_SS14_ARCHIVE_SHA256,
    ROBUST_TOOLBOX_COMMIT,
    SS14_COMMIT,
    RsiMetadataError,
    Ss14ArchiveError,
    audit_known_ss14_archive,
    audit_ss14_archive,
    classify_state_role,
    extract_upstream_references,
    fold_direction_delays,
    known_ss14_cas_path,
    normalize_state_action,
)

EXACT_ARCHIVE = known_ss14_cas_path(Path(__file__).resolve().parents[1] / "data" / "raw")


def _image_bytes(size: tuple[int, int], *, image_format: str = "PNG") -> bytes:
    image = Image.new("RGBA", size, (25, 50, 75, 255))
    output = io.BytesIO()
    image.save(output, format=image_format)
    return output.getvalue()


def _write_rsi(
    archive: ZipFile,
    *,
    root: str,
    path: str,
    metadata: dict[str, object],
    images: dict[str, tuple[tuple[int, int], str]],
) -> None:
    archive.writestr(f"{root}/{path}/meta.json", json.dumps(metadata))
    for name, (size, image_format) in images.items():
        archive.writestr(
            f"{root}/{path}/{name}.png",
            _image_bytes(size, image_format=image_format),
        )


def _fixture_archive(tmp_path: Path) -> Path:
    path = tmp_path / "ss14.zip"
    root = "space-station-14-fixture"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/LICENSE.TXT", "MIT License\n")
        archive.writestr(f"{root}/.github/rsi-schema.json", '{"type":"object"}\n')
        _write_rsi(
            archive,
            root=root,
            path="Resources/Textures/Mobs/Animals/rat.rsi",
            metadata={
                "version": 1,
                "license": "CC-BY-SA-3.0",
                "copyright": (
                    "Taken from https://github.com/tgstation/tgstation/blob/"
                    "53d1f1477d22a11a99c6c6924977cd431075761b/icons/mob/animal.dmi"
                ),
                "size": {"x": 8, "y": 8},
                "states": [
                    {"name": "rat", "directions": 4},
                    {
                        "name": "rat-running",
                        "directions": 4,
                        "delays": [[0.2, 0.3], [0.1, 0.4], [0.25, 0.25], [0.5]],
                    },
                    {
                        "name": "eyes-moving",
                        "directions": 4,
                        "delays": [[0.2, 0.2], [0.2, 0.2], [0.2, 0.2], [0.2, 0.2]],
                    },
                ],
            },
            images={
                "rat": ((16, 16), "PNG"),
                "rat-running": ((32, 16), "WEBP"),
                "eyes-moving": ((32, 16), "PNG"),
            },
        )
        _write_rsi(
            archive,
            root=root,
            path="Resources/Textures/Mobs/Pets/cat.rsi",
            metadata={
                "version": 1,
                "license": "CC-BY-NC-SA-3.0",
                "copyright": "Fixture artist",
                "size": {"x": 8, "y": 8},
                "states": [{"name": "cat-running", "delays": [[0.2, 0.2]]}],
            },
            images={"cat-running": ((16, 8), "PNG")},
        )
        _write_rsi(
            archive,
            root=root,
            path="Resources/Textures/Mobs/Species/Human/parts.rsi",
            metadata={
                "version": 1,
                "license": "CC0-1.0",
                "copyright": "Fixture artist",
                "size": {"x": 8, "y": 8},
                "states": [{"name": "full", "directions": 4}],
            },
            images={"full": ((16, 16), "PNG")},
        )
        _write_rsi(
            archive,
            root=root,
            path="Resources/Textures/Mobs/Effects/stunned.rsi",
            metadata={
                "version": 1,
                "license": "CC-BY-4.0",
                "copyright": "Fixture artist",
                "size": {"x": 8, "y": 8},
                "states": [{"name": "stunned", "delays": [[0.1, 0.1]]}],
            },
            images={"stunned": ((16, 8), "PNG")},
        )
    return path


def _pack(audit: object, logical_path: str) -> object:
    matches = [pack for pack in audit.packs if pack.logical_path == logical_path]  # type: ignore[attr-defined]
    assert len(matches) == 1
    return matches[0]


def _state(pack: object, name: str) -> object:
    matches = [state for state in pack.states if state.name == name]  # type: ignore[attr-defined]
    assert len(matches) == 1
    return matches[0]


def test_fold_direction_delays_matches_engine_fixed_point_timeline() -> None:
    delays, indices = fold_direction_delays(((0.2, 0.3), (0.1, 0.4), (0.25, 0.25), (0.5,)))

    assert delays == pytest.approx((0.1, 0.1, 0.05, 0.25))
    assert indices == (
        (0, 0, 1, 1),
        (2, 3, 3, 3),
        (4, 4, 4, 5),
        (6, 6, 6, 6),
    )

    with pytest.raises(RsiMetadataError, match="finite and positive"):
        fold_direction_delays(((0.0,),))


def test_action_and_role_hints_are_explicit_and_conservative() -> None:
    assert normalize_state_action("rat-moving") == ("move", "explicit_state_name_token")
    assert normalize_state_action("cat_deadcollar")[0] == "death"
    assert normalize_state_action("alive") == (None, "state_name_unmapped")

    assert (
        classify_state_role("Resources/Textures/Mobs/Animals/rat.rsi", "rat-moving")[0]
        == "complete_entity_candidate"
    )
    assert (
        classify_state_role("Resources/Textures/Mobs/Animals/rat.rsi", "eyes-moving")[0]
        == "modular_component"
    )
    assert (
        classify_state_role("Resources/Textures/Mobs/Species/Human/parts.rsi", "full")[0]
        == "modular_component"
    )


def test_upstream_parser_retains_immutable_tgstation_asset_key() -> None:
    commit = "53d1f1477d22a11a99c6c6924977cd431075761b"
    references = extract_upstream_references(
        f"Taken from https://github.com/tgstation/tgstation/blob/{commit}/icons/mob/animal.dmi"
    )

    assert len(references) == 1
    reference = references[0]
    assert reference.repository == "tgstation/tgstation"
    assert reference.revision == commit
    assert reference.revision_is_immutable
    assert reference.asset_path == "icons/mob/animal.dmi"
    assert reference.lineage_key == f"github:tgstation/tgstation@{commit}"
    assert reference.asset_deduplication_key == (
        f"github:tgstation/tgstation@{commit}:icons/mob/animal.dmi"
    )


def test_synthetic_archive_preserves_geometry_rights_roles_and_timing(tmp_path: Path) -> None:
    archive_path = _fixture_archive(tmp_path)
    audit = audit_ss14_archive(archive_path)

    assert audit.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert audit.robust_toolbox_commit == ROBUST_TOOLBOX_COMMIT
    assert audit.counts.mob_rsi_packs == 4
    assert audit.counts.states == 6
    assert audit.counts.expected_source_cells == 27
    assert audit.counts.decoded_source_cells == 27
    assert audit.counts.animated_states == 4
    assert audit.counts.eligible_complete_entity_sequences == 2
    assert audit.counts.eligible_animated_action_sequences == 1
    assert audit.counts.noncommercial_packs == 1
    assert audit.counts.noncommercial_states == 1
    assert audit.counts.tgstation_family_packs == 1
    assert audit.counts.tgstation_immutable_revision_packs == 1
    assert audit.counts.exact_capacity_images == 5
    assert audit.counts.surplus_capacity_images == 1
    assert audit.counts.unused_source_cells == 1
    assert dict(audit.counts.image_format_counts) == {"PNG": 5, "WEBP": 1}

    rat_pack = _pack(audit, "Resources/Textures/Mobs/Animals/rat.rsi")
    running = _state(rat_pack, "rat-running")
    assert rat_pack.role_summary == "mixed_roles"
    assert running.entity_cue == "rat"
    assert running.normalized_action == "run"
    assert running.role == "complete_entity_candidate"
    assert running.direction_names == ("south", "north", "east", "west")
    assert running.expected_source_cell_count == 7
    assert running.image.detected_format == "WEBP"
    assert running.image.grid_capacity == 8
    assert running.image.unused_cell_count == 1
    assert [frame.source_cell_index for frame in running.source_frames] == list(range(7))
    assert [frame.direction for frame in running.source_frames] == [
        "south",
        "south",
        "north",
        "north",
        "east",
        "east",
        "west",
    ]
    assert running.engine_delays_seconds == pytest.approx((0.1, 0.1, 0.05, 0.25))
    assert running.loop_semantics == "not_encoded_in_rsi_caller_controls_playback"

    cat_pack = _pack(audit, "Resources/Textures/Mobs/Pets/cat.rsi")
    cat = _state(cat_pack, "cat-running")
    assert cat_pack.rights_status == "quarantine_noncommercial"
    assert cat_pack.quarantine_reasons == ("noncommercial_asset_license",)
    assert "noncommercial_asset_license" in cat.quarantine_reasons
    assert not cat.eligible_complete_entity_sequence

    parts = _pack(audit, "Resources/Textures/Mobs/Species/Human/parts.rsi")
    assert parts.role_summary == "modular_component_pack"
    assert parts.entity_class_candidates == ("humanoid",)
    assert audit.rights.root_license.sha256 == hashlib.sha256(b"MIT License\n").hexdigest()
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
    archive_path = tmp_path / "unsafe.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("root/../escape.txt", "no")

    with pytest.raises(Ss14ArchiveError, match="unsafe archive member"):
        audit_ss14_archive(archive_path)


def test_known_archive_rejects_unpinned_fixture(tmp_path: Path) -> None:
    with pytest.raises(Ss14ArchiveError, match="SHA-256 mismatch"):
        audit_known_ss14_archive(_fixture_archive(tmp_path))


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact SS14 CAS archive is not present")
def test_exact_cas_archive_audit() -> None:
    audit = audit_known_ss14_archive(EXACT_ARCHIVE)

    assert audit.archive_sha256 == EXPECTED_SS14_ARCHIVE_SHA256
    assert audit.commit == SS14_COMMIT
    assert audit.archive_size_bytes == 234_732_657
    assert audit.counts.archive_members == 49_472
    assert audit.counts.archive_files == 43_004
    assert audit.counts.mob_rsi_packs == 184
    assert audit.counts.rsi_directories_without_meta == 0
    assert audit.counts.states == 1_980
    assert audit.counts.expected_source_cells == 9_849
    assert audit.counts.decoded_source_cells == 9_849
    assert audit.counts.engine_timeline_occurrences == 9_902
    assert audit.counts.animated_states == 209
    assert audit.counts.directional_states == 1_563
    assert audit.counts.directional_animated_states == 134
    assert audit.counts.normalized_action_states == 254
    assert audit.counts.eligible_complete_entity_sequences == 496
    assert audit.counts.eligible_animated_action_sequences == 46
    assert audit.counts.noncommercial_packs == 5
    assert audit.counts.noncommercial_states == 13
    assert audit.counts.tgstation_family_packs == 69
    assert audit.counts.tgstation_immutable_revision_packs == 63
    assert audit.counts.exact_capacity_images == 1_933
    assert audit.counts.surplus_capacity_images == 47
    assert audit.counts.unused_source_cells == 112
    assert audit.counts.missing_images == 0
    assert audit.counts.invalid_or_short_images == 0
    assert audit.counts.undeclared_state_images == 0
    assert audit.counts.srgb_false_packs == 10
    assert audit.counts.meta_atlas_false_packs == 3
    assert audit.counts.duplicate_image_hash_groups == 42
    assert audit.counts.duplicate_image_hash_excess == 53
    assert dict(audit.counts.image_format_counts) == {"PNG": 1_978, "WEBP": 2}
    assert dict(audit.counts.role_counts) == {
        "complete_entity_candidate": 507,
        "effect_or_overlay": 147,
        "icon_or_item_view": 65,
        "modular_component": 1_261,
    }
    assert dict(audit.counts.license_counts) == {
        "CC-BY-3.0": 1,
        "CC-BY-4.0": 1,
        "CC-BY-NC-SA-3.0": 4,
        "CC-BY-NC-SA-4.0": 1,
        "CC-BY-SA-3.0": 173,
        "CC-BY-SA-4.0": 2,
        "CC0-1.0": 2,
    }
    assert audit.rights.root_license.sha256 == (
        "0ac4d87483582bfec5500d39df7889a513730deeaf434c64f44a7975c3b82381"
    )
    assert audit.rights.rsi_schema.sha256 == (
        "befd549dbaafb13cb720a3891ec66ae352e0c3a997096c608a6fc1927244e44c"
    )
    assert {
        document.logical_path: document.sha256 for document in audit.classification_evidence
    } == {
        "Resources/Prototypes/AppearanceCustomization/station_ai.yml": (
            "cd4acb3709d73078879bdbd7250a15a8d0a3c34faaf717423a0ec9e6c58863f5"
        ),
        "Resources/Prototypes/Entities/Mobs/Cyborgs/base_borg_chassis.yml": (
            "c7c74497348d27798f211c07976d8e7d943a4c01e5e191db51a125a206e8ab2c"
        ),
        "Resources/Prototypes/Entities/Mobs/Cyborgs/borg_chassis.yml": (
            "d4959a976d8e2c451efdc240ee27181c78a8b6f9c6eb0203bb67f92052e94c70"
        ),
        "Resources/Prototypes/Entities/Mobs/Cyborgs/xenoborgs.yml": (
            "cfcbe73796459198337d06f1d5805cd66fd15286046517f60b64f6efbc847731"
        ),
        "Resources/Prototypes/Entities/Mobs/NPCs/animals.yml": (
            "d49ed27cc162a04a89f663725816032cd93d14f77c05ccb723c95e4d5d41bdbb"
        ),
        "Resources/Prototypes/Entities/Mobs/NPCs/pets.yml": (
            "72ffc01061fcbb2a784cd67ff5f6d31f5bd1418dc7a612c5b07beba057805067"
        ),
        "Resources/Prototypes/Entities/Mobs/Player/silicon.yml": (
            "e4341699b7ad760d15e6a2d0f6be34a9954a17810988f26a8c9ab9ae897580f5"
        ),
    }
    assert audit.audit_record_sha256 == (
        "0804bd63eed162bd09c42716f7a1ef46f712e2aa702f6a75c38552ec18fac973"
    )

    rat = _pack(audit, "Resources/Textures/Mobs/Animals/rat.rsi")
    moving = _state(rat, "rat-moving")
    eyes = _state(rat, "eyes-moving")
    assert moving.normalized_action == "move"
    assert moving.role == "complete_entity_candidate"
    assert moving.direction_count == 4
    assert moving.expected_source_cell_count == 12
    assert moving.eligible_animated_action_sequence
    assert eyes.role == "modular_component"
    assert not eyes.eligible_complete_entity_sequence

    station_ai = _pack(audit, "Resources/Textures/Mobs/Silicon/station_ai.rsi")
    assert {_state(station_ai, name).role for name in ("base", "ai", "ai_angel_dead")} == {
        "modular_component"
    }
    chassis = _pack(audit, "Resources/Textures/Mobs/Silicon/chassis.rsi")
    assert _state(chassis, "robot").role == "complete_entity_candidate"
    assert _state(chassis, "robot_e").role == "effect_or_overlay"
    displacement = _pack(audit, "Resources/Textures/Mobs/Animals/mothroach/displacement.rsi")
    assert {state.role for state in displacement.states} == {"modular_component"}
