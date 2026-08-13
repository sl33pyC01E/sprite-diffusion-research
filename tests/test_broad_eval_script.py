from types import SimpleNamespace

import numpy as np

from scripts.evaluate_tmwa_broad_inference_v1 import _identity_retrieval, _Target


def _clip(rgb: tuple[int, int, int], x: int) -> np.ndarray:
    value = np.zeros((8, 16, 16, 4), dtype=np.uint8)
    value[:, 4:12, x : x + 4, :3] = rgb
    value[:, 4:12, x : x + 4, 3] = 255
    return value


def test_identity_retrieval_ranks_same_control_identity() -> None:
    request = SimpleNamespace(action="idle", direction="down", view="side", loop_mode="loop")
    red = _clip((255, 0, 0), 2)
    blue = _clip((0, 0, 255), 10)
    targets = {
        "red": _Target("red", "identity-red", "idle", request, red),
        "blue": _Target("blue", "identity-blue", "idle", request, blue),
    }

    exact = _identity_retrieval({"red": red, "blue": blue}, targets)
    swapped = _identity_retrieval({"red": blue, "blue": red}, targets)

    assert exact["eligible"] == 2
    assert exact["top1_correct"] == 2
    assert exact["mean_reciprocal_rank"] == 1.0
    assert swapped["top1_correct"] == 0
    assert swapped["mean_reciprocal_rank"] == 0.5
