import pytest
import numpy as np
from ..form_hl import form_hl


def test_form_hl():
    def limit_state(x1, x2, x3):
        L = 5
        return 1/360 - 0.00694*x3*L**4/(x2*x1)
    def var_func(x1, x2, x3):
        L = 5
        return x2*x1/(L**3 * 360 * 0.00694)

    mean = np.array([8e-4, 2e7, 10])
    std = np.array([1.5e-4, .5e7, .4])
    x, beta, i = form_hl(limit_state, mean, std, var_func)
    assert pytest.approx(beta, rel=1e-2) == 3.175
