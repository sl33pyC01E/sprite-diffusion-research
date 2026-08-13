from __future__ import annotations

from spritelab.caption_audit import _balanced_review_selection, _wrap_text


def test_review_selection_round_robins_subject_types_deterministically() -> None:
    records = [{"identity_id": f"a-{index}", "subject_type": "human"} for index in range(5)] + [
        {"identity_id": "robot-1", "subject_type": "robot"}
    ]
    selected = _balanced_review_selection(records, 3)
    assert {record["subject_type"] for record in selected[:2]} == {"human", "robot"}
    assert len(selected) == 3


def test_wrap_text_preserves_words() -> None:
    value = "a compact literal sprite description with several visible details"
    lines = _wrap_text(value, 18)
    assert " ".join(lines) == value
    assert all(len(line) <= 18 for line in lines)
