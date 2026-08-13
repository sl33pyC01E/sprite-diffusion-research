from pathlib import Path

import pytest

from spritelab.adapters.github import _github_coordinates


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://github.com/example/sprites", ("example", "sprites")),
        ("https://github.com/example/sprites.git", ("example", "sprites")),
        ("https://github.com/example/sprites/tree/main/assets", ("example", "sprites")),
    ],
)
def test_github_coordinates(url: str, expected: tuple[str, str]) -> None:
    assert _github_coordinates(url) == expected


def test_github_coordinates_reject_non_github(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _github_coordinates(tmp_path.as_uri())
