"""Creates a stochastic model class with random variable attributes"""
from scipy import stats
import numpy as np


class StochasticModel(object):
    """Creates an object with random variable attributes

    Args:
        list with: [distribution, mean, std]
    """
    def __init__(self, *args):
        self.dist_func = []
        mean = []
        std = []
        self.dist_name = []

        for dist, mu, sig in args:

            if dist is 'lognorm':
                """scipy lognormal

                Y -> LN(mu_Y, sig_Y)
                ln(Y) -> N(mu_lnY, sig_lnY)

                mu_lnY = ln(mu_Y) - 0.5(sig_lnY**2)
                sig_lnY = sqrt(ln(1 + sig_Y**2/mu_Y**2))

                s = sig_lnY
                scale = exp(mu_lnY)
                """
                s = np.sqrt(np.log(1 + (sig**2)/mu**2))
                scale = np.exp(np.log(mu) - .5 * s**2)
                self.dist_func.append(stats.lognorm(s=s, scale=scale))

            elif dist is 'gumbel_r':
                """scipy gumbel right skw aka extreme type I

                f(x) = exp(-(x-loc)1/scale) exp(-exp(-(x-loc)1/scale))

                1/scale = a = sqrt(pi**2/(6*sig**2))
                loc = u = mu - 0.5772/a
                """
                a = np.sqrt(np.pi**2/(6 * sig**2))
                u = mu - 0.5772/a
                self.dist_func.append(stats.gumbel_r(loc=u, scale=1/a))

            else:
                self.dist_func.append(getattr(stats, dist)(loc=mu, scale=sig))

            self.dist_name.append(dist)
            mean.append(mu)
            std.append(sig)

        self.mean = np.array(mean)
        self.std = np.array(std)
