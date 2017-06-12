"""Compute the standard deviation of g(Xi) using linear approximation"""


def std_linear(grad_g_x, sig):
    """Compute the standard deviation of g(Xi) using

    .. math::

       sig_g = (sum_i^n (grad_g_xi * sig_Xi)^2)^(1/2)

    Args:
        grad_g_x (array): partial derivative of g with respect xi
        sig (array): standard deviation of variables

    Returns:
        std_g (float): standard deviation of g(Xi)
    """
    std_g = 0
    for dg_dx, s in zip(grad_g_x, sig):
        std_g += (dg_dx**2 * s**2)
    return (std_g)**(1/2)
