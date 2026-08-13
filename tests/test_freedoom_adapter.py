from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.freedoom import (
    EXPECTED_FREEDOOM_ARCHIVE_SHA256,
    FREEDOOM_COMMIT,
    DoomSpriteNameError,
    action_hint_for_frame,
    audit_freedoom_archive,
    audit_known_freedoom_archive,
    parse_dehacked_cc_labels,
    parse_doom_sprite_name,
)

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "49/62/4962902bfe9fa921c6ecb4419c55dcd40ca2b93c2d2e3b77c9fc3e89561aec78"
)


def _png_bytes(size: tuple[int, int], *, mode: str = "RGBA") -> bytes:
    color: int | tuple[int, ...] = (20, 40, 60, 128) if mode == "RGBA" else 1
    image = Image.new(mode, size, color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _synthetic_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / "freedoom.zip"
    root = f"freedoom-{FREEDOOM_COMMIT}"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/sprites/README", "Doom sprite sources")
        archive.writestr(f"{root}/sprites/possa1.png", _png_bytes((10, 20)))
        archive.writestr(f"{root}/sprites/possa2a8.png", _png_bytes((11, 20)))
        archive.writestr(f"{root}/sprites/bspia1d1.png", _png_bytes((30, 30)))
        archive.writestr(f"{root}/sprites/vile^0.png", _png_bytes((40, 50)))
        archive.writestr(f"{root}/sprites/blank.png", _png_bytes((1, 1)))
        archive.writestr(
            f"{root}/lumps/dehacked/dehacked.txt",
            """Patch File for DeHackEd v3.0

# Zombie flash
Frame 185
Sprite subnumber = 32773

[STRINGS]
CC_ZOMBIE = research zombie
CC_ARACH = audit spider
""",
        )
        archive.writestr(
            f"{root}/buildcfg.txt",
            """[sprites]
POSSA1 0 0 ; former human
POSSA2A8
BSPIA1D1
VILE\\0

[patches]
""",
        )
        archive.writestr(
            f"{root}/COPYING.adoc",
            "Redistribution and use in source and binary forms are permitted.\n"
            "Neither the name of the Freedoom project may be used to endorse.\n",
        )
        archive.writestr(
            f"{root}/CREDITS",
            "N: Sprite Artist\nS: pixels\nD: sprites and graphics\n\nN: Level Author\nD: levels\n",
        )
        archive.writestr(f"{root}/dist/COPYING.CC0", "CC0 1.0 Universal\n")
    return archive_path


def _conflicting_rotation_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / "conflicting-rotations.zip"
    root = "freedoom-2222222222222222222222222222222222222222"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/sprites/README", "Doom sprite sources")
        archive.writestr(f"{root}/sprites/possa0a1.png", _png_bytes((8, 8)))
        archive.writestr(f"{root}/sprites/trooa1a1.png", _png_bytes((8, 8)))
        archive.writestr(f"{root}/sprites/skula1b2.png", _png_bytes((8, 8)))
    return archive_path


def test_doom_name_parser_preserves_dual_pairs_and_raw_names() -> None:
    parsed = parse_doom_sprite_name("sprites/POSSA2A8.png")

    assert parsed.raw_filename == "POSSA2A8.png"
    assert parsed.raw_stem == "POSSA2A8"
    assert parsed.family == "POSS"
    assert [(ref.frame_token, ref.rotation, ref.mirrored) for ref in parsed.references] == [
        ("A", 2, False),
        ("A", 8, True),
    ]
    assert [ref.canonical_transform for ref in parsed.references] == [
        "identity",
        "horizontal_flip",
    ]

    cross_frame = parse_doom_sprite_name("bspia1d1.png")
    assert [(ref.frame_token, ref.rotation, ref.mirrored) for ref in cross_frame.references] == [
        ("A", 1, False),
        ("D", 1, True),
    ]
    assert cross_frame.references[1].canonical_transform == "horizontal_flip"


def test_parser_preserves_conflicting_pairs_for_later_audit() -> None:
    mixed = parse_doom_sprite_name("POSSA0A1.png")
    duplicate = parse_doom_sprite_name("POSSA1A1.png")

    assert [(ref.rotation, ref.canonical_transform) for ref in mixed.references] == [
        (0, "identity"),
        (1, "horizontal_flip"),
    ]
    assert [ref.rotation for ref in duplicate.references] == [1, 1]


def test_doom_name_parser_preserves_extended_frame_tokens() -> None:
    frame_26 = parse_doom_sprite_name("VILE[0.PNG")
    manifest_frame = parse_doom_sprite_name(r"VILE\0.png")
    unmatched_frame = parse_doom_sprite_name("vile^0.png")

    assert (frame_26.references[0].frame_token, frame_26.references[0].frame_index) == ("[", 26)
    assert frame_26.references[0].vanilla_frame_range_valid is True
    assert (manifest_frame.references[0].frame_token, manifest_frame.references[0].frame_index) == (
        "\\",
        27,
    )
    unmatched_reference = unmatched_frame.references[0]
    assert (unmatched_reference.frame_token, unmatched_reference.frame_index) == (
        "^",
        29,
    )
    assert unmatched_reference.vanilla_frame_range_valid is False


@pytest.mark.parametrize(
    "filename",
    ["blank.png", "POSSAA.png", "POSSA9.png", "POSSA1A.png", "POSSA1.jpg"],
)
def test_doom_name_parser_rejects_non_lump_names(filename: str) -> None:
    with pytest.raises(DoomSpriteNameError):
        parse_doom_sprite_name(filename)


def test_cc_labels_and_action_hints_are_evidence_scoped() -> None:
    labels = parse_dehacked_cc_labels(
        "CC_ZOMBIE = zombie label\nCC_CYBER = tripod label\nCC_FUTURE = unknown slot\n"
    )
    assert [(label.cc_key, label.family, label.label) for label in labels] == [
        ("CC_ZOMBIE", "POSS", "zombie label"),
        ("CC_CYBER", "CYBR", "tripod label"),
        ("CC_FUTURE", None, "unknown slot"),
    ]

    reused = action_hint_for_frame("POSS", "A")
    assert reused.candidate_actions == ("idle", "run")
    assert reused.ambiguous is True
    assert reused.unknown is False
    assert action_hint_for_frame("POSS", "E").candidate_actions == ("attack",)
    assert action_hint_for_frame("POSS", "H").candidate_actions == ("death", "resurrect")
    assert action_hint_for_frame("KEEN", "A").candidate_actions == ("idle", "death")
    assert action_hint_for_frame("KEEN", "B").candidate_actions == ("death",)
    assert action_hint_for_frame("KEEN", "L").candidate_actions == ("death",)

    unknown = action_hint_for_frame("VILE", "^")
    assert unknown.candidate_actions == ()
    assert unknown.unknown is True
    assert unknown.reason == "frame_not_present_in_canonical_action_groups"


def test_rotation_audit_retains_mixed_duplicate_and_cross_frame_pairs(tmp_path: Path) -> None:
    audit = audit_freedoom_archive(_conflicting_rotation_archive(tmp_path))
    families = {family.family: family for family in audit.families}

    poss_a = families["POSS"].frames[0]
    assert poss_a.rotation_scheme == "mixed_all_views_and_directional"
    assert poss_a.rotation_complete is False

    troo_a = families["TROO"].frames[0]
    assert troo_a.duplicate_rotation_references == (1,)
    assert troo_a.rotation_complete is False

    assert families["SKUL"].frame_tokens == ("A", "B")
    issue = next(
        issue for issue in audit.issues if issue.code == "incomplete_or_conflicting_rotation_sets"
    )
    assert "POSS:A:mixed_all_views_and_directional" in issue.related_names
    assert "TROO:A:incomplete_directional" in issue.related_names


def test_synthetic_archive_groups_families_sequences_and_evidence(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    audit = audit_freedoom_archive(archive_path)

    assert audit.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert audit.repository_commit == FREEDOOM_COMMIT
    assert audit.counts.sprite_png_file_count == 5
    assert audit.counts.parsed_sprite_file_count == 4
    assert audit.counts.unparsed_sprite_file_count == 1
    assert audit.counts.dual_pair_file_count == 2
    assert audit.counts.frame_reference_count == 6
    assert audit.counts.family_count == 3
    assert {file.raw_filename for file in audit.sprite_files if file.parse_error} == {"blank.png"}

    families = {family.family: family for family in audit.families}
    poss = families["POSS"]
    assert poss.identity_key == "doom_sprite_family:POSS"
    assert poss.label_hints[0].label == "research zombie"
    assert poss.frame_tokens == ("A",)
    assert poss.frames[0].reference_count == 3
    assert poss.frames[0].rotations == (1, 2, 8)
    assert poss.frames[0].rotation_scheme == "incomplete_directional"
    assert poss.frames[0].rotation_complete is False
    assert poss.frames[0].action_hint.candidate_actions == ("idle", "run")
    assert {sequence.action for sequence in poss.sequences} == {"idle", "run"}
    assert all(sequence.overlaps_other_action_groups for sequence in poss.sequences)

    bspi = families["BSPI"]
    assert bspi.frame_tokens == ("A", "D")
    frame_d = next(frame for frame in bspi.frames if frame.frame_token == "D")
    assert frame_d.direct_reference_count == 0
    assert frame_d.mirrored_reference_count == 1
    assert frame_d.source_references[0].raw_filename == "bspia1d1.png"
    assert frame_d.source_references[0].frame_token == "D"
    assert frame_d.source_references[0].canonical_transform == "horizontal_flip"
    run_sequence = next(sequence for sequence in bspi.sequences if sequence.action == "run")
    rotation_one = next(track for track in run_sequence.rotation_tracks if track.rotation == 1)
    assert rotation_one.frame_tokens == ("A", "D")
    assert rotation_one.complete_for_action_group is True
    assert [reference.mirrored for reference in rotation_one.source_references] == [False, True]
    assert [reference.canonical_transform for reference in rotation_one.source_references] == [
        "identity",
        "horizontal_flip",
    ]
    assert run_sequence.loop_hint is None
    assert run_sequence.sequence_semantics == "ordered_unique_artwork_projection_not_state_cycle"
    assert run_sequence.state_occurrence_order_preserved is False
    assert run_sequence.timing_preserved is False

    vile = families["VILE"]
    assert vile.unknown_action_frame_tokens == ()
    assert [sequence.action for sequence in vile.sequences] == ["revive"]
    assert vile.frames[0].action_hint.frame_token == "^"
    assert vile.frames[0].action_hint.interpreted_frame_token == "\\"
    assert vile.frames[0].action_hint.confidence == "probable_commit_scoped_manifest_alias"
    assert vile.frames[0].action_hint.alias_hint is not None

    rotation_counts = {entry.rotation: entry for entry in audit.rotations}
    assert rotation_counts[0].reference_count == 1
    assert rotation_counts[1].reference_count == 3
    assert rotation_counts[1].physical_file_count == 2
    assert rotation_counts[2].reference_count == 1
    assert rotation_counts[8].reference_count == 1

    assert audit.build_manifest is not None
    assert audit.build_manifest.expected_lump_count == 4
    assert audit.build_manifest.missing_from_archive == (r"VILE\0",)
    assert audit.build_manifest.extra_in_archive == ("BLANK", "VILE^0")
    assert len(audit.build_manifest.alias_hints) == 1
    alias = audit.build_manifest.alias_hints[0]
    assert (alias.archive_lump_name, alias.manifest_lump_name) == ("VILE^0", r"VILE\0")
    assert alias.evidence_member_path.endswith("/buildcfg.txt")
    assert audit.dehacked_frame_patches[0].frame_number == 185
    assert audit.dehacked_frame_patches[0].fields == (("Sprite subnumber", "32773"),)

    evidence = {document.relative_path: document for document in audit.evidence_documents}
    assert evidence["COPYING.adoc"].detected_license_identifiers == ("BSD-3-Clause",)
    assert evidence["dist/COPYING.CC0"].detected_license_identifiers == ("CC0-1.0",)
    assert evidence["dist/COPYING.CC0"].scope.endswith("not_inherited_by_sprites")
    assert audit.credits is not None
    assert audit.credits.record_count == 2
    assert audit.credits.sprite_related_record_count == 1
    assert audit.credits.sprite_related_records[0].display_name == "Sprite Artist"
    assert audit.to_dict()["counts"]["frame_reference_count"] == 6


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact CAS archive is not present")
def test_exact_cas_archive_audit() -> None:
    audit = audit_known_freedoom_archive(EXACT_ARCHIVE)

    assert audit.archive_sha256 == EXPECTED_FREEDOOM_ARCHIVE_SHA256
    assert audit.repository_commit == "d14dbbee3b6fbfb2c11cdb65eb61216e86d4ee85"
    assert audit.counts.zip_member_count == 3880
    assert audit.counts.sprite_png_file_count == 1328
    assert audit.counts.parsed_sprite_file_count == 1325
    assert audit.counts.unparsed_sprite_file_count == 3
    assert audit.counts.dual_pair_file_count == 398
    assert audit.counts.frame_reference_count == 1723
    assert audit.counts.family_count == 140
    assert audit.counts.unique_dimension_count == 730
    assert audit.counts.cc_label_hint_count == 17

    assert {file.raw_filename for file in audit.sprite_files if file.parse_error} == {
        "blank.png",
        "dummy.png",
        "nomonst.png",
    }
    rotations = {entry.rotation: entry.reference_count for entry in audit.rotations}
    assert rotations == {0: 499, 1: 153, 2: 153, 3: 153, 4: 153, 5: 153, 6: 153, 7: 153, 8: 153}

    labels = {hint.family: hint.label for hint in audit.cc_label_hints if hint.family}
    assert labels["POSS"] == "zombie"
    assert labels["CYBR"] == "assault tripod"
    assert labels["BSPI"] == "technospider"

    families = {family.family: family for family in audit.families}
    assert families["POSS"].frame_tokens == tuple("ABCDEFGHIJKLMNOPQRSTU")
    assert families["KEEN"].frames[0].action_hint.candidate_actions == ("idle", "death")
    assert families["KEEN"].frames[1].action_hint.candidate_actions == ("death",)
    assert families["VILE"].unknown_action_frame_tokens == ()
    assert families["VILE"].frames[-1].frame_token == "^"
    vile_alias_hint = families["VILE"].frames[-1].action_hint
    assert vile_alias_hint.unknown is False
    assert vile_alias_hint.ambiguous is True
    assert vile_alias_hint.candidate_actions == ("revive",)
    assert vile_alias_hint.interpreted_frame_token == "\\"
    assert vile_alias_hint.alias_hint is not None
    range_issue = next(
        issue for issue in audit.issues if issue.code == "frames_outside_vanilla_range"
    )
    assert range_issue.related_names == ("vile^0.png:^",)
    assert all(frame.rotation_complete for family in audit.families for frame in family.frames)

    assert audit.build_manifest is not None
    assert audit.build_manifest.expected_lump_count == 1325
    assert audit.build_manifest.missing_from_archive == (r"VILE\0",)
    assert audit.build_manifest.extra_in_archive == ("BLANK", "DUMMY", "NOMONST", "VILE^0")
    assert len(audit.build_manifest.alias_hints) == 1
    alias_issue = next(
        issue for issue in audit.issues if issue.code == "probable_manifest_frame_aliases"
    )
    assert alias_issue.related_names == (r"VILE^0->VILE\0",)

    evidence = {document.relative_path: document for document in audit.evidence_documents}
    assert evidence["COPYING.adoc"].detected_license_identifiers == ("BSD-3-Clause",)
    assert evidence["COPYING.adoc"].scope == "project_root_license_evidence"
    assert evidence["dist/COPYING.CC0"].scope.endswith("not_inherited_by_sprites")
    assert evidence["lumps/colormap/COPYING"].detected_license_identifiers == ("GPL-2.0-only",)
    assert audit.credits is not None
    assert audit.credits.record_count == 251
    assert audit.credits.sprite_related_record_count == 66
