import pytest
from ..form_hlrf_nowak import form_hlrf_nowak
from ..stochastic_model import StochasticModel


def test_form_hlrf_nowak():
    def limit_state(x1, x2):
        return x1 - x2

    def var_func(x1, x2):
        return x1

    X = StochasticModel(['lognorm', 200, 20],
                        ['gumbel_r', 100, 12])

    x, beta, i = form_hlrf_nowak(limit_state, X, var_func)
    assert pytest.approx(beta, rel=1e-2) == 3.76


def test_form_hlrf_nowak2():
    def limit_state(x1, x2, x3):
        return x1*x2 - x3

    def var_func(x1, x2, x3):
        return x1*x2

    X = StochasticModel(['norm', 100, 4/100*100],
                        ['lognorm', 40, 10/100*40],
                        ['gumbel_r', 2000, 10/100*2000])

    x, beta, i = form_hlrf_nowak(limit_state, X, var_func)
    assert pytest.approx(beta, rel=1e-2) == 4.03
