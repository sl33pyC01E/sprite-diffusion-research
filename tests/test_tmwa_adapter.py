from __future__ import annotations

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.tmwa import (
    EXPECTED_INVENTORY_SHA256,
    EXPECTED_TMWA_ARCHIVE_SHA256,
    MANAPLUS_ENGINE_COMMIT,
    TMWA_CLIENT_DATA_COMMIT,
    TmwaArchiveError,
    audit_known_tmwa_archive,
    audit_tmwa_archive,
)

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "7b/7a/7b7a8c61eca284b35ddedaa1af4210a1694933e36ebdc4299bd73395a6532152"
)


def _png_bytes(size: tuple[int, int] = (24, 8)) -> bytes:
    output = BytesIO()
    Image.new("RGBA", size, (25, 50, 75, 128)).save(output, format="PNG")
    return output.getvalue()


def _synthetic_archive(tmp_path: Path) -> Path:
    root = "tmwa-client-data-fixture"
    path = tmp_path / "tmwa.zip"
    wolf = """<?xml version="1.0"?>
<sprite variants="2" variant_offset="3">
  <!-- Fixture Artist; GPLv2. -->
  <imageset name="base" src="graphics/sprites/monsters/wolf.png" width="8"
            height="8" offsetX="2" offsetY="3"/>
  <action name="walk" imageset="base">
    <animation direction="down">
      <sequence start="0" end="2" delay="75" offsetX="1" offsetY="-2"/>
      <end/>
    </animation>
  </action>
</sprite>
"""
    include = """<sprite>
  <include file="monsters/wolf.xml"/>
</sprite>
"""
    dyed = """<sprite>
  <imageset name="base" src="graphics/sprites/monsters/wolf.png|W" width="8" height="8"/>
  <action name="stand" imageset="base">
    <animation><frame index="0"/></animation>
  </action>
</sprite>
"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/", b"")
        archive.writestr(f"{root}/COPYING", "GNU GENERAL PUBLIC LICENSE Version 2\n")
        archive.writestr(
            f"{root}/license.md",
            "## Graphic Licenses\nFile | Artists | Licenses\n--- | --- | ---\n"
            "`graphics/sprites/monsters/wolf.png` | Fixture Artist | GPLv2\n",
        )
        archive.writestr(f"{root}/license-missing", "")
        archive.writestr(f"{root}/graphics/sprites/monsters/wolf.png", _png_bytes())
        archive.writestr(f"{root}/graphics/sprites/monsters/wolf.xml", wolf)
        archive.writestr(f"{root}/graphics/sprites/monsters/include.xml", include)
        archive.writestr(f"{root}/graphics/sprites/monsters/dyed.xml", dyed)
        archive.writestr(
            f"{root}/monsters.xml",
            '<monsters><monster id="7" name="Wolf">'
            "<sprite>monsters/wolf.xml</sprite></monster></monsters>",
        )
    return path


@lru_cache(maxsize=1)
def _known_audit():
    return audit_known_tmwa_archive(EXACT_ARCHIVE)


def test_synthetic_audit_preserves_engine_geometry_timing_includes_and_palette(
    tmp_path: Path,
) -> None:
    path = _synthetic_archive(tmp_path)
    before = path.stat()
    audit = audit_tmwa_archive(path)
    after = path.stat()

    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert audit.counts.sprite_document_count == 3
    assert audit.counts.inspected_png_count == 1
    assert audit.counts.xml_comment_count == 1
    assert audit.xml_comments[0].verbatim == "<!-- Fixture Artist; GPLv2. -->"

    wolf = next(
        item
        for item in audit.sprite_documents
        if item.logical_path == "graphics/sprites/monsters/wolf.xml"
    )
    assert (wolf.variant_count, wolf.variant_offset) == (2, 3)
    assert wolf.imagesets[0].complete_cell_count == 3
    assert (wolf.imagesets[0].remainder_x, wolf.imagesets[0].remainder_y) == (0, 0)

    track = next(
        item
        for item in audit.effective_tracks
        if item.definition_logical_path == "graphics/sprites/monsters/include.xml"
    )
    assert track.source_documents == (
        "tmwa-client-data-fixture/graphics/sprites/monsters/include.xml",
        "tmwa-client-data-fixture/graphics/sprites/monsters/wolf.xml",
    )
    assert [frame.source_frame_index for frame in track.frames] == [0, 1, 2]
    assert [frame.duration_ms for frame in track.frames] == [75, 75, 75]
    assert [frame.rectangle.x for frame in track.frames] == [0, 8, 16]
    assert {(frame.engine_offset_x, frame.engine_offset_y) for frame in track.frames} == {(15, 25)}
    assert track.loop_mode == "one_shot_return_to_stand"
    assert not track.issues

    dyed_track = next(
        item
        for item in audit.effective_tracks
        if item.definition_logical_path == "graphics/sprites/monsters/dyed.xml"
    )
    assert dyed_track.frames[0].imageset_source_literal.endswith("|W")
    assert dyed_track.frames[0].palette_expression == "W"


def test_archive_path_traversal_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("root/../escape.xml", "<sprite/>")
    with pytest.raises(TmwaArchiveError, match="Unsafe ZIP member path"):
        audit_tmwa_archive(path)


def test_exact_archive_identity_inventory_media_and_sprite_regression() -> None:
    audit = _known_audit()
    counts = audit.counts

    assert audit.archive_sha256 == EXPECTED_TMWA_ARCHIVE_SHA256
    assert audit.repository_commit == TMWA_CLIENT_DATA_COMMIT
    assert audit.engine_semantics_commit == MANAPLUS_ENGINE_COMMIT
    assert audit.fix_dead_animation
    assert audit.fix_dead_animation_basis == "manaplus_default_true_features_xml_has_no_override"
    assert audit.inventory_sha256 == EXPECTED_INVENTORY_SHA256
    assert (
        counts.zip_member_count,
        counts.non_directory_member_count,
        counts.regular_file_member_count,
        counts.directory_member_count,
        counts.symlink_member_count,
        counts.expanded_member_bytes,
        counts.compressed_member_bytes,
    ) == (5_082, 4_913, 4_912, 169, 1, 193_431_213, 64_110_574)
    assert (
        counts.xml_member_count,
        counts.png_member_count,
        counts.inspected_png_count,
        counts.relevant_extracted_record_count,
    ) == (2_521, 1_636, 1_636, 4_169)
    assert all(
        image.media_format == "PNG" and image.mode == "RGBA" and image.has_alpha
        for image in audit.images
    )
    assert len({(image.width, image.height) for image in audit.images}) == 396
    assert (
        counts.sprite_document_count,
        counts.physical_imageset_count,
        counts.physical_include_count,
        counts.physical_action_count,
        counts.physical_animation_count,
    ) == (756, 762, 209, 3_501, 12_744)
    assert (
        counts.physical_frame_command_count,
        counts.physical_sequence_command_count,
        counts.physical_end_command_count,
        counts.physical_jump_command_count,
        counts.physical_label_command_count,
        counts.physical_goto_command_count,
    ) == (31_395, 1_601, 7_062, 6, 3, 5)
    assert (counts.effective_track_count, counts.effective_resolved_frame_count) == (
        20_718,
        65_366,
    )


def test_exact_archive_bindings_palette_geometry_and_rights_regression() -> None:
    audit = _known_audit()
    corpora = {item.corpus: item for item in audit.semantic_corpora}
    monsters = corpora["monsters"]
    assert (
        monsters.document_count,
        monsters.entity_count,
        monsters.sprite_layer_reference_count,
        monsters.unique_definition_path_count,
        monsters.zero_layer_entity_count,
        monsters.single_layer_entity_count,
        monsters.multi_layer_entity_count,
        monsters.palette_reference_count,
    ) == (1, 233, 442, 224, 0, 167, 66, 241)
    assert [(item.target_logical_path, item.reason) for item in monsters.include_issues] == [
        ("mods/monsters.xml", "included_document_unavailable")
    ]
    assert (
        corpora["npcs"].document_count,
        corpora["npcs"].entity_count,
        corpora["npcs"].sprite_layer_reference_count,
        corpora["npcs"].zero_layer_entity_count,
        corpora["npcs"].single_layer_entity_count,
        corpora["npcs"].multi_layer_entity_count,
    ) == (259, 260, 908, 18, 93, 149)
    assert (
        corpora["items"].document_count,
        corpora["items"].entity_count,
        corpora["items"].sprite_layer_reference_count,
        corpora["items"].resolved_reference_count,
        corpora["items"].unresolved_reference_count,
    ) == (1_171, 1_159, 1_401, 1_397, 4)
    assert (
        corpora["effects"].entity_count,
        corpora["effects"].zero_layer_entity_count,
        corpora["emotes"].entity_count,
        corpora["emotes"].sprite_layer_reference_count,
        corpora["emotes"].single_layer_entity_count,
    ) == (179, 179, 43, 43, 43)

    imagesets = [item for document in audit.sprite_documents for item in document.imagesets]
    assert sum(item.image is None for item in imagesets) == 4
    assert (
        sum(
            item.image is not None and (item.remainder_x != 0 or item.remainder_y != 0)
            for item in imagesets
        )
        == 5
    )
    assert sum(item.palette_expression is not None for item in imagesets) == 357
    assert len({item.source_literal for item in imagesets}) == 587
    assert len({item.image_logical_path for item in imagesets}) == 586

    rights = audit.rights
    assert (
        rights.table_claim_count,
        rights.unique_table_path_count,
        rights.missing_claim_count,
        rights.unique_missing_path_count,
        rights.contradictory_path_count,
        rights.inconsistent_duplicate_path_count,
    ) == (1_534, 1_532, 309, 309, 4, 1)
    statuses = {item.status for item in rights.image_assessments}
    assert statuses == {
        "contradictory",
        "documented_path_claim",
        "license_missing",
        "unclaimed",
        "unresolved_contributor_or_license",
    }
    assert audit.counts.xml_comment_count == 587
    assert len(audit.engine_evidence) == 8
