from __future__ import annotations

import sys
import math

import pytest
from hypothesis import given, strategies

from my_cool_package import oldtrig as old


@given(x=strategies.floats(
    min_value=-sys.float_info.max / 4,
    max_value=sys.float_info.max / 4,
    allow_infinity=False,
))
def test_property(x):
    assert old.sine(x)**2 + old.cosine(x)**2 == pytest.approx(1)
