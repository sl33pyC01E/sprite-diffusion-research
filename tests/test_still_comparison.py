from __future__ import annotations

from spritelab.still_comparison import _wrap


def test_comparison_prompt_wrap_preserves_words() -> None:
    value = "detailed pixel art sprite with silver armor and a red scarf"
    lines = _wrap(value, 20)
    assert " ".join(lines) == value
    assert all(len(line) <= 20 for line in lines)
