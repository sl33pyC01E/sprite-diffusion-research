from __future__ import annotations

from pathlib import Path

from spritelab.mugen_directory import audit_mugen_directory


def _sff_header() -> bytes:
    payload = bytearray(16)
    payload[:12] = b"ElecbyteSpr\x00"
    payload[12:16] = bytes((0, 0, 1, 0))
    return bytes(payload)


def test_directory_audit_resolves_case_and_groups_definition_variants(tmp_path: Path) -> None:
    fighter = tmp_path / "Fighter"
    fighter.mkdir()
    (fighter / "Hero.AIR").write_text("[Begin Action 0]\n0,0,0,0,4\n", encoding="utf-8")
    (fighter / "Hero.SFF").write_bytes(_sff_header())
    for name in ("hero.def", "alternate.def"):
        (fighter / name).write_text(
            '[Info]\nname="Hero"\n[Files]\nanim=hero.air\nsprite=HERO.sff\n',
            encoding="utf-8",
        )

    audit = audit_mugen_directory(tmp_path)

    assert audit.definition_count == 2
    assert len(audit.variants) == 1
    assert audit.variants[0].definition_paths == (
        "Fighter/alternate.def",
        "Fighter/hero.def",
    )
    assert audit.variants[0].actions[0].action_number == 0
    assert audit.failures == ()


def test_directory_audit_retains_non_character_and_unresolved_definitions(tmp_path: Path) -> None:
    (tmp_path / "ending.def").write_text("[Info]\nname=ending\n", encoding="utf-8")
    (tmp_path / "broken.def").write_text(
        "[Files]\nanim=missing.air\nsprite=missing.sff\n", encoding="utf-8"
    )

    audit = audit_mugen_directory(tmp_path)

    assert audit.variants == ()
    assert [row.reason for row in audit.failures] == [
        "unresolved_media_reference",
        "missing_media_reference",
    ]


def test_directory_audit_forbids_reference_escape(tmp_path: Path) -> None:
    fighter = tmp_path / "fighter"
    fighter.mkdir()
    (fighter / "bad.def").write_text(
        "[Files]\nanim=../../outside.air\nsprite=../../outside.sff\n", encoding="utf-8"
    )

    audit = audit_mugen_directory(tmp_path)

    assert len(audit.failures) == 1
    assert "escapes collection root" in audit.failures[0].detail
