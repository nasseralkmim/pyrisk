"""This module performs the FORM - Hasofer and Lind algorithm from Nowak book"""
import numpy as np
from scipy import stats
from .derivative import derivative
from .stochastic_model import StochasticModel


def form_hlrf_nowak(limit_state, X, var_func, tol=1e-3):
    """Performs the FORM-HL algorithm

    The goal is to find the reliability index for a nonlinear
    limit state function g(Xi).

    The Nowak algorithm uses the g(x)=0 written as a function of one 
    of the variables to ensure g(x)=0. 

    Args:
        limit_state (function): g(X_1, ...,X_n)
        X (object): random variable attributes of the stochastic model
        var_func: inverse function to ensure g(x)=0

    Returns:
        x (array): converged design point
        beta (float): reliability index
    """
    i = 1
    convergence = False

    # Initialize design points
    x = np.copy(X.mean)
    # change last variable to garantee g(Xi)=0
    x[-1] = var_func(*x)

    while not convergence:

        for i, dist in enumerate(X.dist_name):
            if dist is not 'norm':
                # equivalent values
                X.std[i] = (1/(X.dist_func[i].pdf(x[i]))
                            * stats.norm.pdf(stats.norm.ppf(X.dist_func[i].cdf(x[i]))))
                X.mean[i] = x[i] - X.std[i] * (stats.norm.ppf(X.dist_func[i].cdf(x[i])))

        # transform to standard space
        z = (x - X.mean)/X.std

        # compute gradient of g with respect to z
        grad_g_x = derivative(limit_state, x)
        J_z_x = X.std
        grad_g_z = - grad_g_x * J_z_x

        beta = grad_g_z @ z / np.linalg.norm(grad_g_z)
        alpha = grad_g_z / np.linalg.norm(grad_g_z)

        # update design points in standard space
        z = alpha * beta
        # transform to physical space
        x_updt = X.mean + z * X.std
        # # update last variable to guarantee g(Xi)=0
        x_updt[-1] = var_func(*x_updt)

        if np.linalg.norm(x_updt - x)/np.linalg.norm(x_updt) < tol:
            convergence = True
        if not convergence:
            x = x_updt
            i += 1
    return x, beta, i


if __name__ == '__main__':
    def limit_state(x1, x2, x3):
        L = 5
        return 1/360 - 0.00694*x3*L**4/(x2*x1)

    def var_func(x1, x2, x3):
        L = 5
        return x2*x1/(L**3 * 360 * 0.00694)

    X = StochasticModel(['norm', 8e-4, 1.4e-4],
                        ['norm', 2e7, .5e7],
                        ['norm', 10, .4])
    x, beta, i = form_hlrf_nowak(limit_state, X, var_func)
    print(beta, i)
