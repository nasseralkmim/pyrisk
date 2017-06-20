from scipy import stats
import numpy as np
import pytest
from .. import montecarlo


np.random.seed(987654321)


def test_mc_normal():
    """kstest returns:

              1. KS statistic
              2. pvalue

    if pvalue > 0.05 (5%) we accept the null hypothesis:

    H0: our random variable from simulation follow the distribution with
        the parameters obtained from 'stats.dist.fit'.             

    """
    x = montecarlo.normal(1000, 0, 1)
    ks = stats.kstest(x, 'norm', stats.norm.fit(x))
    assert ks[1] > .05


def test_mc_lognormal():
    x = montecarlo.lognormal(1000, 1, .1)
    ks = stats.kstest(x, 'lognorm', stats.lognorm.fit(x))
    assert ks[1] > .05


def test_mc_gumbelr():
    x = montecarlo.gumbel_r(1000, 1, .1)
    ks = stats.kstest(x, 'gumbel_r', stats.gumbel_r.fit(x))
    assert ks[1] > .05


def test_correlated():
    cov = np.array([[1, .5*3*1],
                    [.5*3*1, 3**2]])
    x = montecarlo.correlated(['norm', 10, 1],
                              ['norm', 15, 3],
                              cov=cov,
                              num_simulations=1000)
    numpy_cov = np.cov(x, rowvar=False)
    assert np.allclose(numpy_cov, cov, rtol=1, atol=1)
