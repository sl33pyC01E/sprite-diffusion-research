from __future__ import annotations

from dataclasses import replace

from spritelab.captions import SpriteGenerationRequest, build_sprite_caption
from spritelab.dataset import SequenceSample


def _sample(*, prompt: str, scope: str, identity: str = "warrior-cat") -> SequenceSample:
    return SequenceSample(
        sequence_id="sequence-1",
        identity_id="entity-1",
        source_id="fixture",
        source_pack_id="pack-1",
        entity_class="animal",
        action="run",
        view="side",
        direction="right",
        loop_mode="loop",
        frame_count=8,
        source_blob_sha256=("a" * 64,),
        metadata={
            "sequence_metadata": {"prompt": prompt, "prompt_scope": scope},
            "subjects": [
                {
                    "role": "primary",
                    "entity_id": "entity-1",
                    "external_identity_key": f"source:collection:{identity}",
                }
            ],
        },
    )


def test_collection_prompt_is_preserved_but_not_used_as_identity_caption() -> None:
    caption = build_sprite_caption(_sample(prompt="A tiny mushroom druid", scope="collection"))

    assert caption.description == "warrior cat"
    assert caption.description_basis == "external_identity_key"
    assert caption.source_prompt == "A tiny mushroom druid"
    assert "mushroom" not in caption.text
    assert caption.text == (
        "warrior cat, animal entity, run action, side view, facing right, "
        "seamless loop, transparent background, pixel art animated sprite"
    )


def test_identity_prompt_is_used_when_source_scopes_it_to_identity() -> None:
    caption = build_sprite_caption(
        _sample(prompt="An astronaut squirrel", scope="identity", identity="squirrel-v2")
    )

    assert caption.description == "An astronaut squirrel"
    assert caption.description_basis == "source_identity_prompt"
    assert caption.identity_label == "squirrel v2"
    assert caption.text.startswith("An astronaut squirrel, animal entity")


def test_generation_request_renders_text_and_keeps_structured_controls() -> None:
    request = SpriteGenerationRequest(
        description="a copper clockwork fox",
        entity_class="robot",
        action="emote",
        view="three_quarter",
        direction="left",
        loop_mode="one_shot",
    )

    assert request.text == (
        "a copper clockwork fox, robot entity, emote action, three quarter view, "
        "facing left, one-shot animation, transparent background, pixel art animated sprite"
    )


def test_source_class_beats_opaque_identity_and_splits_camel_case() -> None:
    sample = replace(
        _sample(prompt="", scope=""),
        identity_id="entity_f4238330c8954e66852eba1c3c503c49",
        metadata={
            "sequence_metadata": {"source_class": "SpiritHawkSprite"},
            "subjects": [],
        },
    )

    caption = build_sprite_caption(sample)

    assert caption.identity_label == "spirit hawk sprite"
    assert caption.description_basis == "external_identity_key"


def test_humanizer_removes_known_image_extensions() -> None:
    sample = replace(
        _sample(prompt="", scope=""),
        metadata={
            "sequence_metadata": {"source_class": "piranha.png"},
            "subjects": [],
        },
    )

    assert build_sprite_caption(sample).identity_label == "piranha"
