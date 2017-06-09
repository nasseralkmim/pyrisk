import pytest
from ..integral import trapezoidal


def test_trapezoidal():
    def func(x):
        return 2*(x-1)
    integral = trapezoidal(func, interval=[1, 2], num_division=100)
    assert pytest.approx(integral) == 1


def test_trapezoidal_inf2():
    import numpy as np
    mu, sig = 1, 0.2

    def func(x):
        return 1/(np.sqrt(2*np.pi) * sig)*(np.exp(-1/(2 * sig**2) *
                                                  (x - mu)**2))

    P = trapezoidal(func, interval=['-inf', 'inf'], num_division=100)
    assert pytest.approx(P, 1e-4) == 1
