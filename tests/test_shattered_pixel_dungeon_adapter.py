from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from PIL import Image

from spritelab.adapters.shattered_pixel_dungeon import (
    EXPECTED_SHATTERED_PIXEL_DUNGEON_ARCHIVE_SHA256,
    SHATTERED_PIXEL_DUNGEON_COMMIT,
    JavaSpriteParseError,
    ShatteredPixelDungeonArchiveError,
    audit_known_shattered_pixel_dungeon_archive,
    audit_shattered_pixel_dungeon_archive,
    parse_assets_sprite_mappings,
)

EXACT_ARCHIVE = Path(
    "C:/Users/forre/Documents/sprite-diffusion-research/data/raw/objects/sha256/"
    "de/ed/deed31527c24b2a500f6c79e1cf2502c4c9d8b4391529033da7015cd73c97544"
)


def _png_bytes(size: tuple[int, int]) -> bytes:
    image = Image.new("P", size)
    image.putpalette([0, 0, 0, 255, 255, 255] + [0, 0, 0] * 254)
    output = BytesIO()
    image.save(output, format="PNG", transparency=0)
    return output.getvalue()


def _synthetic_archive(tmp_path: Path) -> Path:
    archive_path = tmp_path / "shattered-pixel-dungeon.zip"
    root = f"shattered-pixel-dungeon-{SHATTERED_PIXEL_DUNGEON_COMMIT}"
    assets = """
public class Assets {
    public static class Sprites {
        public static final String RAT = "sprites/rat.png";
        public static final String WARDS = "sprites/wards.png";
        public static final String WARRIOR = "sprites/warrior.png";
        public static final String MAGE = "sprites/mage.png";
    }
}
"""
    hero_class = """
public enum HeroClass {
    WARRIOR, MAGE;
    public String spritesheet() {
        switch (this) {
            case WARRIOR: default: return Assets.Sprites.WARRIOR;
            case MAGE: return Assets.Sprites.MAGE;
        }
    }
}
"""
    rat_sprite = """
public class RatSprite extends MobSprite {
    public RatSprite() {
        texture( Assets.Sprites.RAT );
        TextureFilm frames = new TextureFilm( texture, 8, 8 );
        idle = new Animation( 4, true );
        idle.frames( frames, 0, 0, 1 );
        run = new Animation( 8, true );
        run.frames( frames, 2, 3 );
        attack = new Animation( 12, false );
        attack.frames( frames, 4, 5, 4 );
        zap = attack.clone();
        die = new Animation( 6, false );
        die.frames( frames, 6, 7 );
    }
}
"""
    variant_sprite = """
public abstract class VariantSprite extends MobSprite {
    protected abstract int texOffset();
    public VariantSprite() {
        texture( Assets.Sprites.RAT );
        int c = texOffset();
        TextureFilm frames = new TextureFilm( texture, 8, 8 );
        idle = new Animation( 2, true );
        idle.frames( frames, c+0, c+1 );
        die = new Animation( 5, false );
        die.frames( frames, c+2, c+3 );
    }
    public static class Blue extends VariantSprite {
        protected int texOffset() { return 4; }
    }
}
"""
    ward_sprite = """
public class WardSprite extends MobSprite {
    private Animation[] tierIdles = new Animation[2];
    public WardSprite() {
        texture( Assets.Sprites.WARDS );
        tierIdles[1] = new Animation( 1, true );
        tierIdles[1].frames( texture.uvRect( 1, 2, 7, 9 ) );
    }
}
"""
    hero_sprite = """
public class HeroSprite extends CharSprite {
    private static final int FRAME_WIDTH = 8;
    private static final int FRAME_HEIGHT = 8;
    private static final int RUN_FRAMERATE = 20;
    public HeroSprite() {
        texture( Dungeon.hero.heroClass.spritesheet() );
        TextureFilm film = new TextureFilm(
            tiers(), Dungeon.hero.tier(), FRAME_WIDTH, FRAME_HEIGHT );
        idle = new Animation( 1, true );
        idle.frames( film, 0, 0, 1 );
        run = new Animation( RUN_FRAMERATE, true );
        run.frames( film, 2, 3 );
    }
}
"""
    java_root = f"{root}/core/src/main/java/com/shatteredpixel/shatteredpixeldungeon"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(f"{root}/LICENSE.txt", "GNU GENERAL PUBLIC LICENSE\nVersion 3\n")
        archive.writestr(
            f"{root}/README.md",
            "Shattered Pixel Dungeon is based on Pixel Dungeon, by Watabou.\n",
        )
        archive.writestr(f"{java_root}/Assets.java", assets)
        archive.writestr(f"{java_root}/actors/hero/HeroClass.java", hero_class)
        archive.writestr(f"{java_root}/sprites/RatSprite.java", rat_sprite)
        archive.writestr(f"{java_root}/sprites/VariantSprite.java", variant_sprite)
        archive.writestr(f"{java_root}/sprites/WardSprite.java", ward_sprite)
        archive.writestr(f"{java_root}/sprites/HeroSprite.java", hero_sprite)
        for filename, size in {
            "rat.png": (32, 16),
            "wards.png": (16, 16),
            "warrior.png": (32, 16),
            "mage.png": (32, 16),
        }.items():
            archive.writestr(f"{root}/core/src/main/assets/sprites/{filename}", _png_bytes(size))
    return archive_path


def _class_map(audit: object) -> dict[str, object]:
    return {item.class_name: item for item in audit.sprite_classes}  # type: ignore[attr-defined]


def _animation(sprite_class: object, action: str) -> object:
    matches = [
        item
        for item in sprite_class.animations
        if item.source_action == action  # type: ignore[attr-defined]
    ]
    assert len(matches) == 1
    return matches[0]


def test_assets_mapping_parser_is_scoped_and_rejects_duplicate_paths() -> None:
    mappings = parse_assets_sprite_mappings(
        """
class Assets {
    static class Sounds { static final String RAT = "sounds/rat.mp3"; }
    static class Sprites {
        // static final String OLD = "sprites/old.png";
        static final String RAT = "sprites/rat.png";
        static final String BAT = "sprites/bat.png";
    }
}
"""
    )
    assert [(item.key, item.relative_path) for item in mappings] == [
        ("RAT", "sprites/rat.png"),
        ("BAT", "sprites/bat.png"),
    ]

    with pytest.raises(JavaSpriteParseError, match="duplicate Assets.Sprites path"):
        parse_assets_sprite_mappings(
            """
class Assets { static class Sprites {
    static final String RAT = "sprites/rat.png";
    static final String COPY = "sprites/rat.png";
} }
"""
        )


def test_synthetic_audit_preserves_repeats_clones_inheritance_and_uv_rects(
    tmp_path: Path,
) -> None:
    archive_path = _synthetic_archive(tmp_path)
    audit = audit_shattered_pixel_dungeon_archive(archive_path)
    classes = _class_map(audit)

    assert audit.archive_sha256 == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    assert audit.repository_commit == SHATTERED_PIXEL_DUNGEON_COMMIT
    assert audit.counts.assets_sprite_mapping_count == 4
    assert audit.counts.hero_class_sheet_mapping_count == 2
    assert audit.counts.sprite_png_file_count == 4
    assert audit.counts.missing_mapped_sprite_png_count == 0
    assert audit.counts.invalid_geometry_animation_count == 0

    rat = classes["RatSprite"]
    assert rat.entity_class == "animal"
    assert rat.morphology_tags == ("quadruped",)
    idle = _animation(rat, "idle")
    assert idle.fps_values == (4,)
    assert idle.looping_values == (True,)
    assert idle.frame_index_variants == ((0, 0, 1),)
    assert idle.deliberate_repeats_preserved
    assert idle.film.frame_width == 8
    assert idle.film.capacities == (8,)

    zap = _animation(rat, "zap")
    assert zap.clone_of == "attack"
    assert zap.frame_index_variants == ((4, 5, 4),)
    assert zap.fps_values == (12,)
    assert zap.looping_values == (False,)

    inherited = classes["VariantSprite.Blue"]
    assert inherited.resolved_parent_class == "VariantSprite"
    blue_idle = _animation(inherited, "idle")
    assert blue_idle.inherited
    assert blue_idle.defined_in_class == "VariantSprite"
    assert blue_idle.frame_index_variants == ((4, 5),)
    assert blue_idle.frame_variable_expressions == (("c", ("texOffset()",)),)

    ward_idle = _animation(classes["WardSprite"], "tierIdles[1]")
    assert ward_idle.normalized_action == "idle"
    assert ward_idle.frame_index_variants == ()
    assert [
        (cell.left, cell.top, cell.right, cell.bottom) for cell in ward_idle.direct_uv_rect_variants
    ] == [(1, 2, 7, 9)]

    hero = classes["HeroSprite"]
    assert hero.source_asset_keys == ("WARRIOR", "MAGE")
    hero_run = _animation(hero, "run")
    assert hero_run.frame_index_variants == ((2, 3),)
    assert hero_run.film.layout_kind == "dynamic_hero_armor_tier_patch_row_major_grid"
    assert hero_run.film.columns == (4,)
    assert hero_run.film.source_sheet_grid_rows == (2,)
    assert "runtime_selects_one_of_multiple_source_sheets" in hero_run.ambiguity_reasons


def test_known_archive_rejects_an_unpinned_payload(tmp_path: Path) -> None:
    archive_path = _synthetic_archive(tmp_path)
    with pytest.raises(ShatteredPixelDungeonArchiveError, match="digest mismatch"):
        audit_known_shattered_pixel_dungeon_archive(archive_path)


@pytest.mark.skipif(not EXACT_ARCHIVE.is_file(), reason="exact CAS archive is not present")
def test_exact_cas_archive_audit() -> None:
    audit = audit_known_shattered_pixel_dungeon_archive(EXACT_ARCHIVE)
    classes = _class_map(audit)

    assert audit.archive_sha256 == EXPECTED_SHATTERED_PIXEL_DUNGEON_ARCHIVE_SHA256
    assert audit.repository_commit == SHATTERED_PIXEL_DUNGEON_COMMIT
    assert audit.counts.zip_member_count == 2111
    assert audit.counts.archive_png_file_count == 202
    assert audit.counts.java_file_count == 1289
    assert audit.counts.parsed_java_class_count == 1792
    assert audit.counts.assets_sprite_mapping_count == 75
    assert audit.counts.hero_class_sheet_mapping_count == 6
    assert audit.counts.sprite_png_file_count == 75
    assert audit.counts.mapped_sprite_png_file_count == 75
    assert audit.counts.unmapped_sprite_png_file_count == 0
    assert audit.counts.missing_mapped_sprite_png_count == 0
    assert audit.counts.sprite_definition_class_count == 115
    assert audit.counts.concrete_sprite_class_count == 109
    assert audit.counts.abstract_sprite_class_count == 6
    assert audit.counts.source_animation_frame_call_count == 348
    assert audit.counts.source_animation_clone_assignment_count == 46
    assert audit.counts.concrete_action_slot_count == 506
    assert audit.counts.resolved_sequence_variant_count == 659
    assert audit.counts.unresolved_animation_count == 0
    assert audit.counts.frame_occurrence_count == 2577
    assert audit.counts.invalid_geometry_animation_count == 0
    assert audit.counts.animal_concrete_class_count == 20
    assert audit.counts.quadruped_concrete_class_count == 6
    assert audit.counts.sprite_png_with_embedded_attribution_count == 0

    snake_idle = _animation(classes["SnakeSprite"], "idle")
    assert snake_idle.fps_values == (10,)
    assert snake_idle.looping_values == (True,)
    assert snake_idle.frame_index_variants == (
        (
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            3,
            2,
            1,
            1,
        ),
    )

    dm300_idle = _animation(classes["DM300Sprite"], "idle")
    assert dm300_idle.fps_values == (10, 15)
    assert dm300_idle.frame_index_variants == ((0, 1), (10, 11))
    assert dm300_idle.frame_variable_expressions == (("c", ("enraged ? 10 : 0",)),)
    assert "runtime_branch_produces_multiple_fps_values" in dm300_idle.ambiguity_reasons

    rat_king_run = _animation(classes["RatKingSprite"], "run")
    assert rat_king_run.frame_index_variants == (
        (2, 3, 4, 5, 6),
        (10, 11, 12, 13, 14),
        (18, 19, 20, 21, 22),
        (26, 27, 28, 29, 30),
    )
    assert rat_king_run.frame_variable_expressions == (("c", ("0", "8", "16", "24")),)

    fist_attack = _animation(classes["FistSprite.Burning"], "attack")
    assert fist_attack.fps_expression == "Math.round(1 / SLAM_TIME)"
    assert fist_attack.fps_values == (3,)
    assert fist_attack.frame_index_variants == ((0,),)

    tengu_run = _animation(classes["TenguSprite"], "run")
    assert tengu_run.looping_values == (False,)
    assert tengu_run.frame_index_variants == ((2, 3, 4, 5, 0),)

    hero = classes["HeroSprite"]
    assert hero.source_asset_keys == (
        "WARRIOR",
        "MAGE",
        "ROGUE",
        "HUNTRESS",
        "DUELIST",
        "CLERIC",
    )
    hero_run = _animation(hero, "run")
    assert hero_run.fps_values == (20,)
    assert hero_run.frame_index_variants == ((2, 3, 4, 5, 6, 7),)
    assert hero_run.film.frame_width == 12
    assert hero_run.film.frame_height == 15
    assert hero_run.film.columns == (21,)
    assert hero_run.film.capacities == (21,)
    assert hero_run.film.source_sheet_grid_rows == (8,)
    assert hero_run.film.source_sheet_grid_capacities == (168,)

    sheep = classes["SheepSprite"]
    assert sheep.entity_class == "animal"
    assert sheep.morphology_tags == ("quadruped",)
    sheep_idle = _animation(sheep, "idle")
    assert sheep_idle.frame_index_variants == ((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0),)

    evidence = {document.relative_path: document for document in audit.evidence_documents}
    assert evidence["LICENSE.txt"].sha256 == (
        "d0495053051967ebe76fb1facd287d79d1ed800da1be75cf501a556bc39a0472"
    )
    assert evidence["LICENSE.txt"].detected_license_identifiers == ("GPL-3.0-only",)
    assert evidence[
        "core/src/main/java/com/shatteredpixel/shatteredpixeldungeon/Assets.java"
    ].detected_license_identifiers == ("GPL-3.0-or-later",)
    assert {item.name for item in audit.attributions} == {
        "Pixel Dungeon / Watabou",
        "Evan Debenham",
        "Oleg Dolya",
    }
    assert audit.to_dict()["counts"]["resolved_sequence_variant_count"] == 659
