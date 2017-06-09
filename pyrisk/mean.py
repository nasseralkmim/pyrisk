from .integral import trapezoidal


def mean(func, interval=[0, 1], num_division=10, g=None):
    """Computes the expected value of a pdf using trapezoidal rule

    Returns:
        E (float): expected value result
    """

    def x_func(x):
        if g is not None:
            return func(x) * g(x)
        else:
            return func(x) * x

    E = trapezoidal(x_func, interval, num_division)
    return E


if __name__ == '__main__':
    def func(x):
        return 2*(x-1)
    E = mean(func, interval=[1, 2], num_division=100)

