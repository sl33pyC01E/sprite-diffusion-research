from spritelab.ingest.freedoom import _external_sequence_key, _phase


def test_freedoom_phase_distinguishes_loop_and_one_shot() -> None:
    assert [_phase(index, 4, True) for index in range(4)] == [0.0, 0.25, 0.5, 0.75]
    assert [_phase(index, 3, False) for index in range(3)] == [0.0, 0.5, 1.0]


def test_freedoom_sequence_key_is_stable_json() -> None:
    assert (
        _external_sequence_key(family="POSS", action="run", rotation=3, commit="abc")
        == '{"action":"run","commit":"abc","family":"POSS","rotation":3}'
    )
