from pathlib import Path

from spritelab.taxonomy import load_taxonomy


def test_taxonomy_preserves_source_action_and_normalizes_conditioning() -> None:
    root = Path(__file__).resolve().parents[1]
    taxonomy = load_taxonomy(root / "configs" / "taxonomy.toml")

    condition = taxonomy.motion_condition(
        action="Blue_Knight-running-east",
        direction="east",
        view="profile",
    )

    assert condition.source_action == "Blue_Knight-running-east"
    assert condition.normalized_action == "run"
    assert condition.action_family == "locomotion"
    assert condition.direction == "right"
    assert condition.view == "side"
    assert condition.loopable_default is True


def test_taxonomy_has_broad_entity_classes() -> None:
    root = Path(__file__).resolve().parents[1]
    taxonomy = load_taxonomy(root / "configs" / "taxonomy.toml")

    assert taxonomy.normalize_entity_class("quadruped").value == "animal"
    assert taxonomy.normalize_entity_class("mech").value == "robot"
    assert taxonomy.normalize_entity_class("unseen category").value == "unknown"
