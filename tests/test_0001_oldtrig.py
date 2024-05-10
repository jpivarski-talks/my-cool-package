from __future__ import annotations

import math

import pytest

from my_cool_package import oldtrig as old


@pytest.mark.parametrize(
    ("arg", "expected"),
    [
        (-10, -0.17364817766693033),
        (0, 0),
        (10, 0.17364817766693033),
        (30, 0.5),
        (45, 1 / math.sqrt(2)),
        (60, math.sqrt(3) / 2),
        (80, 0.984807753012208),
        (90, 1),
        (100, 0.984807753012208),
        (180, 0),
        (270, -1),
        (360, 0),
        (370, 0.17364817766693033),
    ],
)
def test_values(arg, expected):
    assert old.sine(arg) == pytest.approx(expected)


def test_random(random_angles):
    for x in random_angles:
        assert old.sine(x) ** 2 + old.cosine(x) ** 2 == pytest.approx(1)
