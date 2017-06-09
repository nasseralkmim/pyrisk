import pytest
from ..derivative import derivative


def test_derivative():
    def f(x1, x2):
        return 2*x1+5*x2+10

    d = derivative(f, points=[1, 1])
    assert pytest.approx(d) == [2, 5]


