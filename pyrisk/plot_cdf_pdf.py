# Author: Nasser S. Alkmim 2017
import numpy as np
import matplotlib.pyplot as plt
from mean import mean
from variance import variance
from integral import trapezoidal


def cdf_from_pdf(func, interval, num_points=20):
    """Computes a discrete comulative distribution function

    Returns:
        cdf (list): [x, F(x)]
    """
    a, b = interval[0], interval[1]

    x = np.linspace(a, b, num_points)
    F = [0]
    for xi in x[1:]:
        F.append(trapezoidal(func, interval=[a, xi], num_division=100))

    return [list(x), F]


def plot_pdf_and_cdf(func,
                     interval,
                     CDF=None,
                     ax_cdf=None,
                     ax_pdf=None,
                     **kwargs):
    """Plot PDF and CDF
    """
    if CDF is None:
        CDF = cdf_from_pdf(func, interval, num_points=100)

    fig = plt.figure()
    ax_pdf = fig.add_subplot(121)
    ax_cdf = fig.add_subplot(122)

    a, b = interval[0], interval[1]
    x = np.linspace(a, b)

    ax_pdf.plot(x, func(x), label='pdf', **kwargs)
    ax_cdf.plot(CDF[0], CDF[1], '-g', label='cdf', **kwargs)

    ax_cdf.set_xlabel('X')
    ax_cdf.set_ylabel(r'CDF $F(x)$')

    ax_pdf.set_xlabel('X')
    ax_pdf.set_ylabel(r'PDF $f(x)$')

    ax_pdf.legend()
    ax_cdf.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Test 1
    def func(x):
        return 2 * (x - 1)

    P1 = trapezoidal(func, interval=[1, 2], num_division=100)
    print('P={}'.format(P1))
    E1 = mean(func, interval=[1, 2], num_division=100)
    print('E={}'.format(E1))
    var = variance(func, interval=[1, 2], num_division=100)
    print('var={}'.format(var))
    CDF = cdf_from_pdf(func, interval=[1, 2], num_points=100)
    plot_pdf_and_cdf(func, interval=[1, 2], CDF=CDF)
