from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mugen_dense_spark_captions_v1.py"
    spec = importlib.util.spec_from_file_location("run_mugen_dense_spark_captions_v1", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source(variant: str, digest: str = "a" * 64) -> dict[str, object]:
    return {
        "caption_input": {"file_sha256": digest},
        "frame_index": 2,
        "identity_id": f"identity-{variant}",
        "identity_label_provenance_only": variant,
        "reference_frame_array_content_sha256": "b" * 64,
        "split": "train",
        "variant_id": variant,
    }


def _caption(variant: str, digest: str = "a" * 64) -> dict[str, object]:
    return {
        **_source(variant, digest),
        "structured_caption": {"subject_type": "humanoid"},
        "training_appearance_prompt": "pixel humanoid",
    }


def test_exact_render_groups_reuse_completed_and_pending_donors() -> None:
    module = _module()
    completed = {"first": _caption("first")}
    pending = [_source("second"), _source("third"), _source("unique", "c" * 64)]

    groups = module._caption_request_groups(pending, completed)

    by_digest = {rows[0]["caption_input"]["file_sha256"]: (donor, rows) for donor, rows in groups}
    donor, aliases = by_digest["a" * 64]
    assert donor["variant_id"] == "first"
    assert [row["variant_id"] for row in aliases] == ["second", "third"]
    assert by_digest["c" * 64][0] is None

    reused = module._reuse_caption(donor, aliases[0])
    assert reused["variant_id"] == "second"
    assert reused["identity_id"] == "identity-second"
    assert reused["structured_caption"] == donor["structured_caption"]
    assert reused["caption_reuse"]["source_variant_id"] == "first"


def test_caption_reuse_rejects_different_render_and_conflicting_donors() -> None:
    module = _module()
    with pytest.raises(ValueError, match="byte-identical"):
        module._reuse_caption(_caption("first"), _source("second", "c" * 64))
    conflict = _caption("conflict")
    conflict["training_appearance_prompt"] = "different"
    with pytest.raises(RuntimeError, match="disagree"):
        module._caption_request_groups([], {"first": _caption("first"), "conflict": conflict})
