from __future__ import annotations

from pathlib import Path

import pytest

from spritelab.mugen_archive_directory import (
    MugenArchiveMember,
    audit_mugen_archive_metadata_directory,
    parse_7z_slt_members,
)


def test_parse_7z_slt_members_normalizes_paths_and_retains_crc() -> None:
    rows = parse_7z_slt_members(
        "Path = Pack\\Hero.SFF\nSize = 123\nCRC = A1B2C3D4\nAttributes = A\n"
        "\nPath = Pack/Empty.sff\nSize = 0\nCRC = \n"
    )

    assert rows == (
        MugenArchiveMember("Pack/Hero.SFF", 123, "a1b2c3d4"),
        MugenArchiveMember("Pack/Empty.sff", 0, None),
    )


def test_parse_7z_slt_members_rejects_traversal() -> None:
    with pytest.raises(ValueError, match="traversal"):
        parse_7z_slt_members("Path = ../Hero.sff\nSize = 1\n")


def test_metadata_audit_resolves_archive_sff_without_extracting_it(tmp_path: Path) -> None:
    fighter = tmp_path / "Pack" / "Hero"
    fighter.mkdir(parents=True)
    (fighter / "Hero.def").write_text(
        "[Info]\nname=Hero\n[Files]\nanim=hero.air\nsprite=HERO.sff\n",
        encoding="utf-8",
    )
    (fighter / "Hero.air").write_text(
        "[Begin Action 0]\n0,0,0,0,4\n0,1,0,broken,4\n",
        encoding="utf-8",
    )
    inventory = (MugenArchiveMember("Pack/Hero/Hero.SFF", 999, "12345678"),)

    audit = audit_mugen_archive_metadata_directory(tmp_path, inventory)

    assert audit.definition_count == 1
    assert len(audit.variants) == 1
    variant = audit.variants[0]
    assert variant.sff_member == inventory[0]
    assert variant.actions[0].action_number == 0
    assert variant.air_parse_exclusions[0].raw_line == "0,1,0,broken,4"
    assert audit.failures == ()


def test_metadata_audit_retains_missing_sff_as_failure(tmp_path: Path) -> None:
    (tmp_path / "Hero.def").write_text(
        "[Files]\nanim=Hero.air\nsprite=missing.sff\n", encoding="utf-8"
    )
    (tmp_path / "Hero.air").write_text("[Begin Action 0]\n0,0,0,0,4\n", encoding="utf-8")

    audit = audit_mugen_archive_metadata_directory(tmp_path, ())

    assert audit.variants == ()
    assert len(audit.failures) == 1
    assert audit.failures[0].reason == "unresolved_media_reference"
