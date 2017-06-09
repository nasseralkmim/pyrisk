from .integral import trapezoidal
from .mean import mean


def variance(func, interval=[0, 1], num_division=10, g=None):
    """Computes variance of distribution

    Returns:
        var (float): variance result
    """

    def xx_func(x):
        if g is not None:
            Eg = mean(func, interval, num_division, g)
            return (g(x) - Eg)**2 * func(x)
        else:
            E = mean(func, interval, num_division)
            return (x - E)**2 * func(x)

    var = trapezoidal(xx_func, interval, num_division)
    return var
