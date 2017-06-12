"""This module performs the FORM - Hasofer and Lind algorithm"""
import numpy as np
from .derivative import derivative


def form_hl(limit_state, mean, std, var_func, tol=1e-3):
    """Performs the FORM-HL algorithm

    The goal is to find the reliability index for a nonlinear
    limit state function g(Xi).

    Args:
        limit_state (function): g(X_1, ..., X_n)
        mean (array): mean of random variables (RV)
        std (array): standard deviation of RV

    Returns:
        x (array): converged design point
        beta (float): reliability index
    """
    i = 1
    convergence = False
    # Initialize design points
    x = np.copy(mean)
    # change last variable to garantee g(Xi)=0
    x[-1] = var_func(*mean)

    while not convergence:
        # transform to standard space
        z = (x - mean)/std

        # compute gradient of g with respect to z
        grad_g_x = derivative(limit_state, x)
        J_z_x = std
        grad_g_z = - grad_g_x * J_z_x

        beta = grad_g_z @ z / np.linalg.norm(grad_g_z)
        alpha = grad_g_z / np.linalg.norm(grad_g_z)

        # update design points in standard space
        z = alpha * beta
        # transform to physical space
        x_updt = mean + z*std
        # update last variable to guarantee g(Xi)=0
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

    mean = np.array([8e-4, 2e7, 10])
    std = np.array([1.5e-4, .5e7, .4])
    x, beta, i = form_hl(limit_state, mean, std, var_func)
    print(beta, i)
