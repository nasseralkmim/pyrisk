import pytest
from ..mean import mean


def test_mean():
    def func(x):
        return 2*(x-1)
    E = mean(func, interval=[1, 2], num_division=100)
    assert pytest.approx(E, 1e-4) == 5/3


def test_mean_inf():

    def func(x):
        return (x > 100) * 20000 / x**3

    E = mean(func, interval=[100, 'inf'], num_division=100)
    assert pytest.approx(E, 1) == 200
