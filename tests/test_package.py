from __future__ import annotations

import importlib.metadata

import my_cool_package as m


def test_version():
    assert importlib.metadata.version("my_cool_package") == m.__version__
