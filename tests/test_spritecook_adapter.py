import json

import pytest

from spritelab.adapters.spritecook import (
    candidate_media_roles,
    classify_member,
    normalize_member_path,
    parse_index_metadata,
    repository_commit_hint,
)

ROOT = "spritecook-free-game-assets-cd0db67d6f849c8d84e406a436aac76b733a7dba"


def _index():
    return parse_index_metadata(
        json.dumps(
            {
                "generatedAt": "2026-05-23T10:49:14.073Z",
                "source": "https://www.spritecook.ai/examples",
                "license": "CC0-1.0",
                "examples": [
                    {
                        "slug": "pixel-art-astronaut-squirrel",
                        "title": "Pixel Art Astronaut Squirrel",
                        "category": "character",
                        "prompt": "An astronaut squirrel",
                        "settings": [
                            {"label": "Style", "value": "Pixel"},
                            {"label": "Output", "value": "Sheet + frames"},
                        ],
                        "animationCount": 3,
                        "previewPath": ("examples/pixel-art-astronaut-squirrel/idle.webp"),
                        "folder": "examples/pixel-art-astronaut-squirrel",
                        "sourceUrl": (
                            "https://www.spritecook.ai/examples/pixel-art-astronaut-squirrel"
                        ),
                    },
                    {
                        "slug": "pixel-art-characters",
                        "title": "Pixel Art Characters",
                        "category": "character",
                        "prompt": "A tiny mushroom druid",
                        "settings": [],
                        "animationCount": None,
                        "previewPath": (
                            "examples/pixel-art-characters/astronaut-squirrel/idle.webp"
                        ),
                        "folder": "examples/pixel-art-characters",
                        "sourceUrl": ("https://www.spritecook.ai/examples/pixel-art-characters"),
                    },
                ],
            }
        )
    )


def test_parse_index_preserves_source_fields_and_nullable_count() -> None:
    index = _index()

    assert index.generated_at == "2026-05-23T10:49:14.073Z"
    assert index.license_expression == "CC0-1.0"
    assert index.example("PIXEL ART ASTRONAUT SQUIRREL").animation_count == 3
    assert index.example("pixel-art-characters").animation_count is None
    assert [(setting.label, setting.value) for setting in index.examples[0].settings] == [
        ("Style", "Pixel"),
        ("Output", "Sheet + frames"),
    ]


def test_duplicate_collection_and_single_example_share_identity_key() -> None:
    index = _index()
    aggregate = classify_member(
        f"{ROOT}/examples/pixel-art-characters/astronaut-squirrel/idle.webp",
        index=index,
    )
    singleton = classify_member(
        "examples/pixel-art-astronaut-squirrel/idle_sheet.png",
        index=index,
    )

    assert aggregate.identity_key == singleton.identity_key
    assert aggregate.identity_key == "spritecook:pixel-character:astronaut-squirrel"
    assert aggregate.raw_action_hint == "idle"
    assert aggregate.normalized_action_candidates == ("idle",)
    assert aggregate.media_role_candidates == ("animation_container", "preview")
    assert singleton.media_role_candidates == (
        "sprite_sheet",
        "horizontal_animation_frames_candidate",
    )


def test_tiny_wiz_and_wizard_filename_variants_share_identity() -> None:
    animation = classify_member("examples/tiny-pixel-art/tiny_wiz_jump.webp")
    sheet = classify_member("examples/tiny-pixel-art/tiny_wizard_jump_spritesheet.png")

    assert animation.identity_key == sheet.identity_key
    assert animation.identity_key == "spritecook:tiny-pixel-art:tiny-wizard"
    assert animation.raw_entity_hint == "tiny_wiz"
    assert sheet.raw_entity_hint == "tiny_wizard"
    assert animation.normalized_action_candidates == ("jump",)


def test_static_state_variants_group_without_guessing_normalized_action() -> None:
    closed = classify_member("examples/game-asset-pack/chest_closed.png")
    opened = classify_member("examples/game-asset-pack/chest_open.png")

    assert closed.identity_key == opened.identity_key == "spritecook:game-asset-pack:chest"
    assert closed.raw_action_hint == "closed"
    assert opened.raw_action_hint == "open"
    assert closed.normalized_action_candidates == ()
    assert closed.action_basis == "filename_state_token_unmapped"
    assert closed.normalized_entity_class_candidates == ("object",)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "examples/detailed-characters-anime/frog_paladin.png",
            ("animal", "humanoid"),
        ),
        (
            "examples/detailed-characters-anime/clockwork_owl.png",
            ("robot", "animal"),
        ),
        (
            "examples/pixel-art-royal-goblin/idle.webp",
            ("monster", "humanoid"),
        ),
        (
            "examples/game-asset-pack/skeleton.png",
            ("monster",),
        ),
    ],
)
def test_entity_classification_keeps_ambiguous_candidates(
    path: str,
    expected: tuple[str, ...],
) -> None:
    assert classify_member(path).normalized_entity_class_candidates == expected


def test_animation_container_does_not_imply_semantic_action() -> None:
    hint = classify_member("examples/isometric-buildings/animated_house_1.webp")

    assert hint.identity_key == "spritecook:isometric-building:animated-house-1"
    assert hint.raw_action_hint is None
    assert hint.normalized_action_candidates == ()
    assert hint.normalized_entity_class_candidates == ("environment", "object")
    assert hint.media_role_candidates == ("animation_container", "preview")


def test_provenance_preserves_prompt_scope_and_license_evidence() -> None:
    index = _index()
    singleton = classify_member(
        "examples/pixel-art-astronaut-squirrel/walk.webp",
        index=index,
    ).provenance
    collection_member = classify_member(
        "examples/pixel-art-characters/astronaut-squirrel/walk.webp",
        index=index,
    ).provenance

    assert singleton.prompt == "An astronaut squirrel"
    assert singleton.prompt_scope == "identity"
    assert collection_member.prompt == "A tiny mushroom druid"
    assert collection_member.prompt_scope == "collection"
    assert singleton.license_expression == "CC0-1.0"
    assert singleton.declared_folder == "examples/pixel-art-astronaut-squirrel"
    assert singleton.license_evidence_members == (
        "examples/pixel-art-astronaut-squirrel/LICENSE.txt",
        "examples/pixel-art-astronaut-squirrel/README.md",
        "LICENSE",
        "README.md",
        "index.json",
    )


def test_non_media_evidence_has_role_but_no_identity() -> None:
    hint = classify_member(f"{ROOT}\\examples\\tiny-pixel-art\\LICENSE.txt")

    assert hint.archive_member == "examples/tiny-pixel-art/LICENSE.txt"
    assert hint.identity_key is None
    assert hint.media_role_candidates == ("license_evidence",)


def test_commit_root_is_preserved_as_provenance_but_not_identity() -> None:
    hint = classify_member(
        f"{ROOT}/examples/pixel-art-astronaut-squirrel/idle.webp",
        index=_index(),
    )

    assert hint.provenance.original_archive_member.startswith(f"{ROOT}/")
    assert hint.provenance.repository_commit == ROOT.rsplit("-", 1)[-1]
    assert repository_commit_hint(hint.archive_member) is None


@pytest.mark.parametrize(
    "path",
    ["../examples/tiny-pixel-art/tiny_wiz.png", "/examples/x.png", "C:\\x.png"],
)
def test_member_normalization_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError):
        normalize_member_path(path)


def test_candidate_roles_do_not_treat_misleading_extension_as_proof_of_pixels() -> None:
    assert candidate_media_roles(
        "examples/stylized-seamless-textures/seamless_stylized_dirt.png"
    ) == ("seamless_texture", "environment_asset")


def test_parse_index_rejects_duplicate_slugs() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        parse_index_metadata(
            {
                "examples": [
                    {"slug": "same", "settings": []},
                    {"slug": "SAME", "settings": []},
                ]
            }
        )
