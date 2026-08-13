from __future__ import annotations

import hashlib
import io
import json

import numpy as np

from spritelab.mugen_stills import (
    compose_caption_input,
    detailed_training_prompt,
    filtered_appearance_caption,
    load_mugen_still_references,
)


def _save_array(path, value: np.ndarray) -> tuple[str, str]:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    payload = buffer.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    header = f"{value.dtype.str}\0{'x'.join(str(item) for item in value.shape)}\0".encode()
    return hashlib.sha256(payload).hexdigest(), hashlib.sha256(
        header + value.tobytes(order="C")
    ).hexdigest()


def test_reference_plan_prefers_idle_medoid_and_builds_caption_facts(tmp_path) -> None:
    frames = np.zeros((3, 8, 8, 4), dtype=np.uint8)
    frames[:, 2:6, 2:6, 3] = 255
    frames[0, 2:6, 2:6, 0] = 255
    frames[1, 2:6, 2:6, 1] = 255
    frames[2] = frames[0]
    idle_file, idle_array = _save_array(tmp_path / "clips/idle.npy", frames)
    attack = frames[:, ::-1].copy()
    attack_file, attack_array = _save_array(tmp_path / "clips/attack.npy", attack)

    def sequence(identifier: str, action: str, output_path: str, file_hash: str, array_hash: str):
        return {
            "action": action,
            "caption": {"identity_label": "Fixture Knight"},
            "entity_class": "humanoid",
            "identity_id": "identity-a",
            "output": {
                "array_content_sha256": array_hash,
                "file_sha256": file_hash,
                "relative_path": output_path,
            },
            "sequence_id": identifier,
            "split": "train",
        }

    materialization = {
        "sequence_count": 2,
        "sequences": [
            sequence("sequence-attack", "attack", "clips/attack.npy", attack_file, attack_array),
            sequence("sequence-idle", "idle", "clips/idle.npy", idle_file, idle_array),
        ],
    }
    taxonomy = {
        "sequence_count": 2,
        "records": [
            {"sequence_id": "sequence-attack", "verb": "normal_attack"},
            {"sequence_id": "sequence-idle", "verb": "idle"},
        ],
    }
    materialization_path = tmp_path / "materialization.json"
    taxonomy_path = tmp_path / "taxonomy.json"
    materialization_path.write_text(json.dumps(materialization), encoding="utf-8")
    taxonomy_path.write_text(json.dumps(taxonomy), encoding="utf-8")

    reference = load_mugen_still_references(materialization_path, taxonomy_path)[0]

    assert reference.sequence_id == "sequence-idle"
    assert reference.frame_index == 0
    assert reference.alpha_bbox_xywh == (2, 2, 4, 4)
    assert reference.visible_pixel_count == 16
    assert reference.palette_facts[0] == ("red", 1.0)
    composite = compose_caption_input(reference.rgba)
    assert composite[0, 0].tolist() == [127, 127, 127]
    raw = "A red armored knight. The background is grey. Pixelated appearance."
    assert filtered_appearance_caption(raw) == "A red armored knight."
    prompt = detailed_training_prompt(reference, raw)
    assert "Fixture Knight" in prompt
    assert "transparent background" in prompt
    assert "background is grey" not in prompt
