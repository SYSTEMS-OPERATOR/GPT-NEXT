import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from model.utils import RollingCounter


def test_rolling_counter_basic():
    counter = RollingCounter(limit=3)
    for val in [1, 2, 3, 4]:
        counter.add(val)
    assert counter.total_average() == (1 + 2 + 3 + 4) / 4
    assert counter.rolling_average() == (2 + 3 + 4) / 3
