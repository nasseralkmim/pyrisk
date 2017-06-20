import numpy as np
from scipy import stats
import pytest
from ..form_hlrf_correlation import form_hlrf_correlation
from ..stochastic_model import StochasticModel
from .. import montecarlo


def test_form_hlrf_correlation2():
    def limit_state(x1, x2, x3):
        """From choi 2007 p. 224
        x1 = Ma
        x2 = P1
        x3 = P2
        """
        return x1 - x2 - 2*x3

    X = StochasticModel(['norm', 50, 5],
                        ['norm', 10, 2],
                        ['norm', 15, 3])

    x, beta, i = form_hlrf_correlation(limit_state, X, tol=1e-5)
    pf = stats.norm.cdf(-beta)
    assert pytest.approx(pf, rel=1e-2) == 0.1073


def test_form_hlrf_correlation3():
    def limit_state(x1, x2, x3):
        """From choi 2007 p. 224
        x1 = Ma
        x2 = P1
        x3 = P2
        """
        return x1 - x2 - 2*x3

    X = StochasticModel(['norm', 50, 5],
                        ['norm', 10, 2],
                        ['norm', 15, 3])

    X.add_correlation(2, 3, .25)

    x, beta, i = form_hlrf_correlation(limit_state, X, tol=1e-5)
    pf = stats.norm.cdf(-beta)
    assert pytest.approx(pf, rel=1e-2) == 0.1171


def test_form_hlrf_correlation4():
    def limit_state(x1, x2):
        return 3*x1 - 2*x2

    X = StochasticModel(['norm', 16.6, 2.45],
                        ['norm', 18.8, 2.83])

    X.add_correlation(1, 2, 2/(2.45*2.83))

    x, beta, i = form_hlrf_correlation(limit_state, X, tol=1e-5)
    print(beta)
    assert pytest.approx(beta, rel=1e-2) == 1.55

