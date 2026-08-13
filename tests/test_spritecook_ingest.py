from spritelab.ingest.spritecook import _phase, _primary_entity_class


def test_spritecook_phase_respects_loop_and_one_shot() -> None:
    assert [_phase(index, 4, "loop") for index in range(4)] == [0.0, 0.25, 0.5, 0.75]
    assert [_phase(index, 4, "one_shot") for index in range(4)] == [
        0.0,
        1 / 3,
        2 / 3,
        1.0,
    ]


def test_spritecook_primary_entity_preserves_broad_class_priority() -> None:
    assert _primary_entity_class(("animal", "humanoid")) == "animal"
    assert _primary_entity_class(("environment", "object")) == "environment"
    assert _primary_entity_class(()) == "unknown"
