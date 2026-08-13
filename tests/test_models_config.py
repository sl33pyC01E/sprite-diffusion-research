from __future__ import annotations

from dataclasses import replace

import pytest

from spritelab.models import (
    DEFAULT_ACTION_CLASSES,
    DEFAULT_ENTITY_CLASSES,
    ConditioningSchema,
    FactorizedSpriteDiT,
    MissingTorchError,
    PixelDiTConfig,
    SpriteClipCondition,
    torch_available,
    validate_conditioning_shape,
    validate_identity_action_groups,
    validate_model_input_shapes,
    validate_phase_shape,
    validate_video_shape,
)


def _clip(identity_id: str, action: str) -> SpriteClipCondition:
    return SpriteClipCondition(
        identity_id=identity_id,
        entity_class="humanoid",
        action=action,
        view="side",
        direction="right",
        loop_mode="loop",
        frame_phases=tuple(index / 8 for index in range(8)),
        sequence_id=f"{identity_id}:{action}",
    )


def test_default_config_describes_native_rgba_patch_grid() -> None:
    config = PixelDiTConfig()
    grid = config.patch_grid

    assert config.expected_video_shape(3) == (3, 8, 4, 64, 64)
    assert (grid.rows, grid.columns, grid.frames) == (32, 32, 8)
    assert grid.tokens_per_frame == 1_024
    assert grid.total_tokens == 8_192
    assert grid.patch_vector_width == 16


def test_default_schema_covers_broad_entities_and_steerable_actions() -> None:
    assert {
        "humanoid",
        "animal",
        "creature",
        "monster",
        "robot",
        "vehicle",
        "object",
        "prop",
        "effect",
        "unknown",
    }.issubset(DEFAULT_ENTITY_CLASSES)
    assert {"idle", "run", "emote", "action"}.issubset(DEFAULT_ACTION_CLASSES)

    schema = ConditioningSchema()
    assert schema.condition_tokens == (
        "identity",
        "entity",
        "action",
        "view",
        "direction",
        "loop",
    )
    assert schema.phase_bins == 8
    assert schema.min_actions_per_identity == 2


def test_default_schema_accepts_indexed_unknown_and_robot_labels() -> None:
    schema = ConditioningSchema()
    schema.validate_clip(
        SpriteClipCondition(
            identity_id="clockwork-robot",
            entity_class="robot",
            action="unknown",
            view="unknown",
            direction="unknown",
            loop_mode="unknown",
            frame_phases=(0.0, 1.0),
        ),
        expected_frames=2,
    )


def test_shape_validators_are_torch_free() -> None:
    config = PixelDiTConfig()

    assert validate_video_shape((2, 8, 4, 64, 64), config) == config.patch_grid
    assert validate_conditioning_shape((2, 13, 768), config, batch_size=2).tokens == 13
    assert validate_conditioning_shape((2, 768), config, batch_size=2).tokens == 1
    assert validate_phase_shape((2, 8), config, batch_size=2) == (2, 8)
    validate_model_input_shapes((2, 8, 4, 64, 64), (2, 13, 768), config)


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((8, 4, 64, 64), "rank 5"),
        ((1, 7, 4, 64, 64), "tail"),
        ((1, 8, 3, 64, 64), "tail"),
        ((1, 8, 4, 32, 64), "tail"),
    ],
)
def test_video_shape_rejects_wrong_layout(shape: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_video_shape(shape, PixelDiTConfig())


def test_config_rejects_non_rgba_and_invalid_patch_geometry() -> None:
    with pytest.raises(ValueError, match="exactly 4 RGBA"):
        PixelDiTConfig(channels=3)
    with pytest.raises(ValueError, match="divisible by patch_size"):
        PixelDiTConfig(width=63)
    with pytest.raises(ValueError, match="model_dim must be divisible"):
        PixelDiTConfig(model_dim=385)


def test_config_requires_one_phase_bin_per_frame() -> None:
    with pytest.raises(ValueError, match="phase_bins must equal num_frames"):
        PixelDiTConfig(num_frames=4)

    schema = replace(ConditioningSchema(), phase_bins=4)
    config = PixelDiTConfig(num_frames=4, conditioning=schema)
    assert config.patch_grid.frames == 4


def test_schema_rejects_narrow_entity_or_action_vocabulary() -> None:
    narrow_entities = tuple(label for label in DEFAULT_ENTITY_CLASSES if label != "effect")
    with pytest.raises(ValueError, match="missing required labels"):
        ConditioningSchema(entity_classes=narrow_entities)

    narrow_actions = tuple(label for label in DEFAULT_ACTION_CLASSES if label != "emote")
    with pytest.raises(ValueError, match="missing required labels"):
        ConditioningSchema(action_classes=narrow_actions)


def test_clip_validation_checks_phase_and_labels() -> None:
    schema = ConditioningSchema()
    schema.validate_clip(_clip("knight_017", "run"), expected_frames=8)

    bad_phase = replace(_clip("knight_017", "run"), frame_phases=(0.0, 0.25, 0.5, 1.0))
    with pytest.raises(ValueError, match="length must match"):
        schema.validate_clip(bad_phase, expected_frames=8)

    phase_at_loop_endpoint = replace(
        _clip("knight_017", "run"),
        frame_phases=(0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0),
    )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        schema.validate_clip(phase_at_loop_endpoint, expected_frames=8)


def test_identity_groups_require_multiple_distinct_actions() -> None:
    schema = ConditioningSchema()
    grouped = validate_identity_action_groups(
        [_clip("knight_017", "idle"), _clip("knight_017", "run")], schema
    )
    assert grouped == {"knight_017": ("idle", "run")}

    with pytest.raises(ValueError, match="at least 2 distinct actions"):
        validate_identity_action_groups([_clip("knight_017", "idle")], schema)


def test_import_and_validation_work_without_torch() -> None:
    config = PixelDiTConfig()
    if torch_available():
        assert FactorizedSpriteDiT(config).config == config
    else:
        with pytest.raises(MissingTorchError, match="requires.*PyTorch"):
            FactorizedSpriteDiT(config)
