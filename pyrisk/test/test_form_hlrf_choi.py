import pytest
from ..form_hlrf_choi import form_hlrf_choi
from ..stochastic_model import StochasticModel


def test_form_hlrf_choi():
    def limit_state(x1, x2):
        return x1**3 + x2**3 - 18

    X = StochasticModel(['norm', 10, 5],
                        ['norm', 10, 5])

    x, beta, i = form_hlrf_choi(limit_state, X)
    assert pytest.approx(beta, rel=1e-2) == 2.24


def test_form_hlrf_choi2():
    def limit_state(x1, x2):
        return x1**3 + x2**3 - 18

    X = StochasticModel(['norm', 10, 5],
                        ['norm', 9.9, 5])

    x, beta, i = form_hlrf_choi(limit_state, X)
    assert pytest.approx(beta, rel=1e-2) == 1.16


def test_form_hlrf_choi3():
    def limit_state(x1, x2, x3):
        return x2*x3 - 78.12*x1

    X = StochasticModel(['gumbel_r', 4, 1],
                        ['norm', 2e7, .5e7],
                        ['norm', 1e-4, .2e-4])

    x, beta, i = form_hlrf_choi(limit_state, X)
    assert pytest.approx(beta, rel=1e-2) == 3.322

