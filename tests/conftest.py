from __future__ import annotations

import random

import pytest


@pytest.fixture
def random_angles():
    random.seed(12345)
    out = []
    for _ in range(1000):
        out.append(random.uniform(-370, 370))
    return out
