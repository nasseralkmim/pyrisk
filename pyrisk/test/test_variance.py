import pytest
from ..variance import variance


def test_variance():
    def func(x):
        return 2*(x-1)
    var = variance(func, interval=[1, 2], num_division=100)
    assert pytest.approx(var, 1e-4) == 1/18
