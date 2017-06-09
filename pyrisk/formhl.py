"""This module performs the FORM - Hasofer and Lind algorithm"""
import numpy as np


def form_hl(perform_func, mean, std, var_func, derivative_x):
    """performs the FORM-HL algorithm

    Args:
        perform_func (function): g(X_i)
        mean (array): mean of random variables (RV)
        std (array): standard deviation of RV
    """
    x_last = var_func(*mean[:-1])

    dp = np.copy(mean)
    dp[-1] = x_last

    z = (dp - mean)/std

    ddx = derivative_x(*dp)
    ddz = - ddx * std
    print(ddz)
    beta = ddz @ z / np.sqrt(ddz.T @ ddz)

    alpha = ddz  / np.sqrt(ddz.T @ ddz)
    print(alpha)
    z = alpha * beta

    x = mean + z * std
    x[-1] = var_func(*x[:-1])


if __name__ == '__main__':
    def perform_func(x1, x2, x3):
        L = 5
        return 1/360 - 0.00694*x3*L**4/(x2*x1)
    def var_func(x1, x2):
        L = 5
        return x2*x1/(L**3 * 360 * 0.00694)
    def derivative_x(x1, x2, x3):
        L = 5
        ddx1 = 0.00694 * x3 * L**4 / (x2 * x1**2)
        ddx2 = 0.00694 * x3 * L**4 / (x2**2 * x1)
        ddx3 = - 0.00694 * L**4 / (x2 * x1)
        return np.array([ddx1, ddx2, ddx3])
        

    mean = np.array([8e-4, 2e7, 10])
    std = np.array([1.5e-4, .5e7, .4])
    form_hl(perform_func, mean, std, var_func, derivative_x)

