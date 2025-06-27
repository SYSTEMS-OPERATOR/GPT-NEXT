import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from minimal_yaml import safe_load


def test_alias_resolution():
    text = """
key1: &val 123
key2: *val
"""
    data = safe_load(text)
    assert data["key1"] == 123
    assert data["key2"] == 123


def test_unknown_alias_error():
    text = "key: *missing"
    with pytest.raises(ValueError):
        safe_load(text)
