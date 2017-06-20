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
        rv: rv -> N(mu, sig) as a numpy ndarray

    """
    u = np.random.rand(num_simulations)  # uniform [0, 1]
    z = stats.norm.ppf(u)              # standard normal [0, 1]
    x = z * sig + mu                   # normal [mu, sig]
    return x


def lognormal(num_simulations, mu, sig):
    """Generate lognormal random variable

    In order to generate a RV X -> lN(mu, sig), we use the following relation

        z = (ln(x) - mu_ln) / sig_ln

    where the RV ln(X) -> N(mu_ln, sig_ln)

    Args:
        num_simulations: number of simulations
        mu: mean of the RV following lognormal distribution
        sig: standard deviation of RV distribution

    Returns:
        rv: rv -> lN(mu, sig) as a numpy ndarray

    """
    u = np.random.rand(num_simulations)  # uniform [0, 1]
    z = stats.norm.ppf(u)              # standard normal [0, 1]

    sig_ln = np.sqrt(np.log(sig**2/mu**2 + 1))
    mu_ln = np.log(mu) - 1/2 * sig_ln**2

    x = np.exp(mu_ln + z*sig_ln)
    return x


def gumbel_r(num_simulations, mu, sig):
    """Generates Gumbel right skewed random variable

    Gumbel right skewed, also knwon as extreme Type I,
    distribution is given, using scipy notation, as:

        f(x) = exp(-(x-loc)1/scale) exp(-exp(-(x-loc)1/scale))

    where the scale and loc parameters are related to the mean and
    standard deviation by,

        1/scale = a = sqrt(pi**2/(6*sig**2))
        loc = u = mu - 0.5772/a

    Montecarlo procedure:
       1. Generate u -> U[0, 1]
       2. Generate x -> F[mu, sig] with

          xi = Fx^{-1}(ui), Fx is CDF of F

    Args:
        num_simulations: number of simulations
        mu: mean of the RV following lognormal distribution
        sig: standard deviation of RV distribution

    Returns:
        x: rv -> Gumbel Right (mu, sig) as a numpy ndarray

    """
    u = np.random.rand(num_simulations)  # uniform [0, 1]

    a = np.sqrt(np.pi**2/(6*sig**2))
    loc = mu - 0.5772/a
    scale = 1/a

    x = stats.gumbel_r.ppf(u, loc=loc, scale=scale)
    return x


def correlated(*args, cov=None, num_simulations=1000):
    """generate X_N correlated random variables

    Args:
        *args (list): [dist, mu, sig] of each random variable
        cov (array): 2d array with covariance elements

    Returns:
        [X_N]: array with random variables each column is a variables
               each row is a sample
    """
    dist, mu_x, sig_x = [], [], []
    for d, m, s in args:
        dist.append(d)
        mu_x.append(m)
        sig_x.append(s)

    Cx = cov
    _, T = np.linalg.eig(Cx)
    Cy = T.T @ Cx @ T

    mu_x = np.array(mu_x)
    mu_y = T.T @ mu_x
    sig_y = [np.sqrt(Cy[i, i]) for i in range(Cy.shape[0])]

    y = np.zeros((num_simulations, len(dist)))
    for i, d in enumerate(dist):
        if d is 'norm':
            y[:, i] = normal(num_simulations, mu_y[i], sig_y[i])
        elif d is 'lognorm':
            y[:, i] = lognormal(num_simulations, mu_y[i], sig_y[i])
        else:
            print('Dist not implemented!')

    x = []
    for yi in y:
        x.append(T @ yi)
    x = np.array(x)

    return x


if __name__ == '__main__':
    x = correlated(['norm', 10, 1],
                   ['norm', 15, 3],
                   cov=np.array([[1, .5*3*1],
                                 [.5*3*1, 3**2]]),
                   num_simulations=100000)
