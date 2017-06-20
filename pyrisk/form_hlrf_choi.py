"""This module performs the FORM - Hasofer and Lind algorithm from Choi book"""
import numpy as np
from scipy import stats
from .derivative import derivative
from .stochastic_model import StochasticModel
from .std_linear import std_linear


def form_hlrf_choi(limit_state, X, tol=1e-3):
    """Performs the FORM-HL algorithm

    The goal is to find the reliability index for a nonlinear
    limit state function g(Xi).

    The reliability index is the shortest distance from g(Xi)=0 to the 
    origin.

    For a line given by ax + by + c = 0, the distance from (xp, yp) to 
    this line is,

       beta = |a*xp + b*yp + c| / ||[a, b]||

    because (xp, yp) = (0, 0) and using a linear approximation for g(Xi),
    which generates :math: c = g(z*) - grad_u_z @ z*.

       beta = (g(z*) - grad_g_z @ z) / || grad_g_z ||

    where z* is the vector with the design points in the standard normalized
    space

    Args:
        limit_state (function): g(X_1, ...,X_n)
        X (object): random variables attributes of the stochastic model

    Returns:
        x (array): converged design point
        beta (float): reliability index
    """
    iteration = 1
    convergence = False

    # initial beta
    mu_g = limit_state(*X.mean)
    grad_g_x = derivative(limit_state, X.mean)
    sig_g = std_linear(grad_g_x, X.std)
    beta = mu_g/sig_g
    alpha = - grad_g_x * X.std / sig_g

    # Initialize design points
    x = X.mean + beta * X.std * alpha

    while not convergence:

        for i, dist in enumerate(X.dist_name):
            if dist is not 'norm':
                # equivalent values
                X.std[i] = (1/(X.dist_func[i].pdf(x[i]))
                            * stats.norm.pdf(
                                stats.norm.ppf(
                                    X.dist_func[i].cdf(x[i]))))
                X.mean[i] = x[i] - X.std[i] * (stats.norm.ppf(
                    X.dist_func[i].cdf(x[i])))

        # transform to standard space
        z = (x - X.mean)/X.std

        # compute gradient of g with respect to z
        grad_g_x = derivative(limit_state, x)
        grad_g_z = grad_g_x * X.std

        beta_previous = beta
        beta = (limit_state(*x) - grad_g_z @ z) / np.linalg.norm(grad_g_z)
        alpha = - grad_g_z / np.linalg.norm(grad_g_z)

        # update design points in standard space
        z = alpha * beta
        # transform to physical space
        x_updt = X.mean + z * X.std

        condition1 = np.linalg.norm(x_updt - x)/np.linalg.norm(x_updt) < tol
        condition2 = abs(np.round(beta, 3) - np.round(beta_previous, 3)) < tol
        if condition2 or condition1:
            convergence = True
        if not convergence:
            x = x_updt
            iteration += 1
    return x, beta, iteration


if __name__ == '__main__':
    def limit_state(x1, x2):
        return x1**3 + x2**3 - 18

    X = StochasticModel(['norm', 10, 5],
                        ['norm', 9.9, 5])

    x, beta, i = form_hlrf_choi(limit_state, X, tol=1e-4)
    print(beta, i)
