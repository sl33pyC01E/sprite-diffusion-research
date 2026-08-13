from __future__ import annotations

import hashlib
import json

import pytest

from spritelab.mugen_taxonomy import build_mugen_action_taxonomy


def _sequence(identifier: str, number: int, meaning: str, action: str) -> dict:
    return {
        "action": action,
        "identity_id": f"identity-{identifier}",
        "provenance": {
            "source_action_number": number,
            "source_id": "fixture",
            "source_meaning": meaning,
        },
        "sequence_id": f"sequence-{identifier}",
        "split": "train",
    }


def test_taxonomy_projects_structured_verbs_and_retains_evidence_gap(tmp_path) -> None:
    manifest = {
        "schema_version": 1,
        "sequence_count": 3,
        "sequences": [
            _sequence("a", 200, "recommended_attack_range", "attack"),
            _sequence("b", 1000, "special_attack", "attack"),
            _sequence("c", 130, "guard_standing", "defend"),
        ],
    }
    source = tmp_path / "materialization.json"
    source.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "taxonomy.json"

    digest = build_mugen_action_taxonomy(source, output)

    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["counts"]["verb"] == {
        "block": 1,
        "normal_attack": 1,
        "special_attack": 1,
    }
    assert report["counts"]["attack_strength"] == {"unresolved": 3}
    assert report["evidence_gaps"] == {
        "attack_form_requires_cns_hitdef_or_air_comment_join": 2,
        "attack_strength_requires_cns_hitdef_or_air_comment_join": 2,
    }
    with pytest.raises(FileExistsError, match="replace"):
        build_mugen_action_taxonomy(source, output)
