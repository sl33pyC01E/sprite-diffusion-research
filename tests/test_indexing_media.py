from pathlib import Path

from PIL import Image

from spritelab.indexing import inspect_media_observation


def test_pure_media_observation_uses_original_member_extension(tmp_path: Path) -> None:
    path = tmp_path / ("a" * 64)
    Image.new("RGBA", (7, 9), (1, 2, 3, 4)).save(path, "PNG")

    observation = inspect_media_observation(
        blob_sha256="a" * 64,
        path=path,
        original_name="sprites/hero.png",
    )

    assert observation["media_format"] == "PNG"
    assert observation["width"] == 7
    assert observation["height"] == 9
    assert observation["has_alpha"] is True
