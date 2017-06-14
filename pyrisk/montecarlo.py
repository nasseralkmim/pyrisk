"""Generates random numbers from Monte Carlo simulation"""
import numpy as np
from scipy import stats


def normal(num_simulations, mu, sig):
    """Perform Monte Carlo simulation to generate rv following normal distribution

    Args:
        num_simulations: number of simulations
        mu: mean of RV following normal distribution
        sig: standard deviation of distribution

    Returns:
        rv: rv ~ N(mu, sig) as a numpy ndarray

    """
    u = np.random.rand(num_simulations)  # uniform [0, 1]
    z = stats.norm.ppf(u)              # standard normal [0, 1]
    x = z * sig + mu                   # normal [mu, sig]
    return x


def lognormal(num_simulations, mu, sig):
    """Generate lognormal random variable

    In order to generate a RV X ~ lN(mu, sig), we use the following relation

        z = (ln(x) - mu_ln) / sig_ln

    where the RV ln(X) ~ N(mu_ln, sig_ln)

    Args:
        num_simulations: number of simulations
        mu: mean of the RV following lognormal distribution
        sig: standard deviation of RV distribution

    Returns:
        rv: rv ~ lN(mu, sig) as a numpy ndarray

    """
    u = np.random.rand(num_simulations)  # uniform [0, 1]
    z = stats.norm.ppf(u)              # standard normal [0, 1]

    sig_ln = np.log(sig**2/mu**2 + 1)
    mu_ln = np.log(mu) - 1/2 * sig_ln**2

    x = np.exp(mu_ln + z*sig_ln)
    return x
