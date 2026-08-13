from __future__ import annotations

import hashlib
import json
import plistlib
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.openduelyst import (
    EXPECTED_OPENDUELYST_ARCHIVE_SHA256,
    OPENDUELYST_COMMIT,
    OpenDuelystArchiveError,
    audit_known_openduelyst_archive,
    audit_openduelyst_archive,
    parse_card_lookup,
    parse_entity_animation_mappings,
    parse_resource_descriptors,
    parse_texture_packer_plist,
    resolve_animation_sequence,
    runtime_frame_keys,
)

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "9d/90/9d907a2d299b0f1598984192e3d4832aeb770e75fa2507370ff8e66428282f8e"
)


def _png_bytes(size: tuple[int, int] = (16, 16)) -> bytes:
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _gif_bytes() -> bytes:
    frames = [Image.new("RGBA", (2, 2), color) for color in ((255, 0, 0, 0), (0, 0, 0, 0))]
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=[60, 70],
        loop=0,
        transparency=0,
    )
    return output.getvalue()


def _frame(
    rect: str,
    *,
    offset: str = "{0,0}",
    rotated: bool = False,
    source_rect: str | None = None,
    source_size: str = "{4,4}",
) -> dict[str, object]:
    return {
        "frame": rect,
        "offset": offset,
        "rotated": rotated,
        "sourceColorRect": source_rect or "{{0,0},{4,4}}",
        "sourceSize": source_size,
    }


def _plist_bytes() -> bytes:
    return plistlib.dumps(
        {
            "frames": {
                "wolf_run_10.png": _frame("{{0,0},{4,4}}"),
                "wolf_run_2.png": _frame(
                    "{{4,0},{3,4}}",
                    offset="{0.5,-0.5}",
                    source_rect="{{1,0},{3,4}}",
                    rotated=True,
                ),
                "wolf_run_2.5.png": _frame("{{7,0},{4,4}}"),
                "unmapped.png": _frame("{{11,0},{4,4}}"),
            },
            "metadata": {
                "format": 2,
                "size": "{16,16}",
                "textureFileName": "wolf.png",
                "realTextureFileName": "wolf.png",
            },
        },
        sort_keys=False,
    )


def _resources_source() -> str:
    return """
const RSX = {
  ordinaryImage: { name: 'ordinaryImage', img: 'resources/plain.png' },
  wolfRun: {
    name: 'wolfRuntimeRun',
    img: 'resources/units/wolf.png',
    is16Bit: true,
    plist: 'resources/units/wolf.plist',
    framePrefix: 'wolf_run_',
    frameDelay: .10,
  },
};
"""


def _factory_source() -> str:
    return """
class CardFactory_Test
  @cardForIdentifier: (identifier,gameSession) ->
    if (identifier == Cards.Neutral.Wolf)
      card = new Unit(gameSession)
      card.factionId = Factions.Neutral
      card.raceId = Races.Vespyr
      card.name = i18next.t("cards.wolf_name")
      card.setBaseAnimResource(
        walk : RSX.wolfRun.name
        attackDelay: 0.4
      )
"""


def _synthetic_archive(tmp_path: Path) -> Path:
    root = f"duelyst-{OPENDUELYST_COMMIT}"
    archive_path = tmp_path / "duelyst.zip"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/LICENSE", "CC0 1.0 Universal\n")
        archive.writestr(f"{root}/app/data/resources.js", _resources_source())
        archive.writestr(
            f"{root}/app/sdk/cards/cardsLookup.coffee",
            "class Cards\n  @Neutral:{\n    Wolf: 900\n  }\n",
        )
        archive.writestr(f"{root}/app/sdk/cards/factory/test/neutral.coffee", _factory_source())
        archive.writestr(
            f"{root}/app/localization/locales/en/cards.json",
            json.dumps({"wolf_name": "Snow Wolf"}),
        )
        archive.writestr(f"{root}/app/resources/units/wolf.plist", _plist_bytes())
        archive.writestr(f"{root}/app/resources/units/wolf.png", _png_bytes())
        archive.writestr(f"{root}/app/resources/unit_gifs/wolf_run.gif", _gif_bytes())
    return archive_path


def test_resource_parser_keeps_raw_literals_and_does_not_promote_plain_images() -> None:
    (declaration,) = parse_resource_descriptors(_resources_source())

    assert declaration.alias == "wolfRun"
    assert declaration.name == "wolfRuntimeRun"
    assert declaration.frame_prefix == "wolf_run_"
    assert declaration.frame_delay == 0.1
    assert declaration.frame_delay_expression == ".10"
    assert declaration.effective_frame_delay_seconds == pytest.approx(0.08)
    assert {field.name: field.expression for field in declaration.raw_fields}["is16Bit"] == "true"


def test_texturepacker_parser_preserves_order_geometry_trim_rotation_and_offset() -> None:
    atlas = parse_texture_packer_plist(
        _plist_bytes(),
        relative_path="resources/units/wolf.plist",
        image_size=(16, 16),
    )

    assert [frame.key for frame in atlas.frames] == [
        "wolf_run_10.png",
        "wolf_run_2.png",
        "wolf_run_2.5.png",
        "unmapped.png",
    ]
    special = atlas.frames[1]
    assert special.rotated is True
    assert special.is_trimmed is True
    assert (special.offset.x, special.offset.y) == (0.5, -0.5)
    assert special.offset.raw == "{0.5,-0.5}"
    assert special.within_image_bounds is True
    assert atlas.metadata_matches_image_size is True


def test_runtime_sequence_uses_numeric_last_token_stable_sort_and_role_semantics() -> None:
    (declaration,) = parse_resource_descriptors(_resources_source())
    atlas = parse_texture_packer_plist(
        _plist_bytes(),
        relative_path="resources/units/wolf.plist",
        image_size=(16, 16),
    )
    sequence = resolve_animation_sequence(declaration, atlas, source_roles=("walk",))

    assert runtime_frame_keys(
        ["wolf_run_10.png", "wolf_run_2.png", "wolf_run_2.5.png", "other.png"],
        "wolf_run_",
    ) == ("wolf_run_2.png", "wolf_run_2.5.png", "wolf_run_10.png")
    assert [frame.key for frame in sequence.frames] == [
        "wolf_run_2.png",
        "wolf_run_2.5.png",
        "wolf_run_10.png",
    ]
    assert sequence.declared_frame_delay_expression == ".10"
    assert sequence.effective_frame_delay_seconds == pytest.approx(0.08)
    assert sequence.total_duration_seconds == pytest.approx(0.24)
    assert sequence.normalized_action == "walk"
    assert sequence.loop_mode == "loop"
    assert sequence.direction is None
    assert "horizontally flips at runtime" in sequence.direction_semantics


def test_card_mapping_preserves_identity_name_race_roles_and_numeric_fields() -> None:
    card_ids = parse_card_lookup("class Cards\n  @Neutral:{\n    Wolf: 900\n  }\n")
    (mapping,) = parse_entity_animation_mappings(
        _factory_source(),
        relative_path="app/sdk/cards/factory/test/neutral.coffee",
        card_ids=card_ids,
        localization={"cards.wolf_name": "Snow Wolf"},
    )

    assert mapping.identifier_token == "Cards.Neutral.Wolf"
    assert mapping.card_id == 900
    assert mapping.card_kind == "Unit"
    assert mapping.display_name == "Snow Wolf"
    assert mapping.race_expression == "Races.Vespyr"
    assert [
        (reference.role, reference.resource_alias) for reference in mapping.animation_references
    ] == [("walk", "wolfRun")]
    numeric = {field.role: field.numeric_value for field in mapping.animation_fields}
    assert numeric["attackDelay"] == 0.4


def test_synthetic_archive_audit_is_read_only_and_joins_source_spaces(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    before = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    audit = audit_openduelyst_archive(archive_path)

    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == before
    assert audit.counts.animation_descriptor_count == 1
    assert audit.counts.unique_png_payload_count == 1
    assert audit.counts.duplicate_png_payload_group_count == 0
    assert audit.counts.texture_atlas_count == 1
    assert audit.counts.atlas_frame_count == 4
    assert audit.counts.descriptor_frame_occurrence_count == 3
    assert audit.counts.atlas_frame_unmatched_by_descriptor_count == 1
    assert audit.counts.entity_mapping_count == 1
    assert audit.counts.entity_animation_reference_count == 1
    assert audit.counts.gif_animation_count == 1
    assert audit.counts.unique_gif_payload_count == 1
    assert audit.atlases[0].relative_path == "resources/units/wolf.plist"
    assert audit.sequences[0].entity_mapping_indices == (0,)
    assert audit.entity_mappings[0].display_name == "Snow Wolf"
    assert [frame.duration_milliseconds for frame in audit.gifs[0].frames] == [60, 70]
    assert audit.gifs[0].loop_value == 0
    assert audit.evidence_documents[0].scope == "repository_project"
    assert audit.to_dict()["sequences"][0]["frames"][0]["key"] == "wolf_run_2.png"


def test_known_archive_rejects_an_unpinned_payload(tmp_path: Path) -> None:
    with pytest.raises(OpenDuelystArchiveError, match="digest mismatch"):
        audit_known_openduelyst_archive(_synthetic_archive(tmp_path))


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact CAS archive is not present")
def test_exact_cas_archive_audit() -> None:
    audit = audit_known_openduelyst_archive(EXACT_ARCHIVE)
    counts = audit.counts

    assert audit.archive_sha256 == EXPECTED_OPENDUELYST_ARCHIVE_SHA256
    assert audit.repository_commit == OPENDUELYST_COMMIT
    assert audit.root_prefix == f"duelyst-{OPENDUELYST_COMMIT}"
    assert audit.archive_size_bytes == 1_173_967_743
    assert (counts.zip_member_count, counts.file_member_count) == (13_364, 12_807)
    assert counts.directory_member_count == 557
    assert (
        counts.archive_png_file_count,
        counts.archive_gif_file_count,
        counts.archive_plist_file_count,
        counts.archive_coffeescript_file_count,
    ) == (6_303, 82, 1_385, 1_578)
    assert counts.unique_png_payload_count == 5_398
    assert counts.duplicate_png_payload_group_count == 852
    assert counts.resource_entry_count == 7_320
    assert counts.animation_descriptor_count == 5_312
    assert counts.unique_descriptor_plist_count == 1_277
    assert counts.unique_descriptor_image_count == 1_277
    assert counts.unique_descriptor_plist_prefix_count == 5_310
    assert counts.texture_atlas_count == 1_294
    assert counts.atlas_frame_count == 69_564
    assert counts.rotated_atlas_frame_count == 21
    assert counts.trimmed_atlas_frame_count == 202
    assert counts.nonzero_offset_atlas_frame_count == 110
    assert counts.descriptor_frame_occurrence_count == 69_091
    assert counts.descriptor_unique_frame_count == 69_072
    assert counts.resolved_animation_descriptor_count == 5_304
    assert counts.empty_animation_descriptor_count == 8
    assert counts.atlas_frame_unmatched_by_descriptor_count == 167
    assert counts.unreferenced_atlas_count == 17
    assert counts.unreferenced_atlas_frame_count == 325

    assert counts.entity_mapping_count == 1_076
    assert counts.entity_animation_reference_count == 4_988
    assert counts.unique_referenced_resource_alias_count == 4_708
    assert counts.mapped_resource_alias_multiple_role_count == 29
    assert counts.resource_alias_name_mismatch_count == 3
    assert counts.runtime_name_collision_count == 2
    assert counts.shared_physical_frame_key_count == 19
    assert counts.exact_timeline_alias_group_count == 1

    categories = Counter(sequence.category for sequence in audit.sequences)
    assert categories == {
        "unit": 4_319,
        "icon_animation": 742,
        "effect": 238,
        "tile": 6,
        "rune": 6,
        "arena_effect": 1,
    }
    empty_aliases = {
        "iconSuperMaliceActive",
        "iconThoughtExchangeActive",
        "f2TwilightFoxHit",
        "f1ThirdGeneralCast",
        "f3DuplicatorObelyskRun",
        "f4AbominationDeath",
        "f5OrphanAspectDeath",
        "f5OrphanAspectHit",
    }
    assert {
        sequence.resource_alias for sequence in audit.sequences if not sequence.frames
    } == empty_aliases
    aliases = {declaration.alias: declaration for declaration in audit.declarations}
    assert aliases["neutralSkywingHit"].name == "neutralSkywingiHit"
    assert aliases["f3BBZephyrHit"].name == "f3ZephyrHit"
    assert aliases["f4FallenAspectAttack"].name == "f2OrizuruAttack"
    exact_aliases = {
        group.keys
        for group in audit.duplicate_groups
        if group.kind == "exact_nonempty_timeline_alias"
    }
    assert exact_aliases == {("f6ExplodingWallDamage", "f6ExplodingWallHit")}

    assert counts.gif_animation_count == 82
    assert counts.unique_gif_payload_count == 70
    assert counts.duplicate_gif_payload_group_count == 12
    assert counts.gif_frame_count == 1_119
    assert Counter(gif.frame_count for gif in audit.gifs) == {
        14: 67,
        12: 8,
        10: 3,
        11: 1,
        20: 1,
        16: 1,
        8: 1,
    }
    assert all(gif.loop_value == 0 and gif.has_transparency for gif in audit.gifs)
    assert sum(group.kind == "byte_identical_gif" for group in audit.duplicate_groups) == 12

    assert counts.evidence_document_count == 11
    root_license = next(
        item for item in audit.evidence_documents if item.relative_path == "LICENSE"
    )
    assert root_license.sha256 == "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499"
    assert root_license.detected_license_identifiers == ("CC0-1.0",)
    assert root_license.scope == "repository_project"
    assert audit.embedded_metadata.png_with_xmp_count == 2_289
    assert audit.embedded_metadata.duplicate_png_path_excess_count == 905
    assert audit.embedded_metadata.png_with_comment_count == 17
    assert audit.embedded_metadata.gimp_comment_count == 17
    assert audit.embedded_metadata.png_with_asset_attribution_field_count == 0
    assert counts.plist_parse_error_count == 4
    assert all(
        frame.within_image_bounds is True for atlas in audit.atlases for frame in atlas.frames
    )
