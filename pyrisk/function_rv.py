from .derivative import derivative


def mean_variance(func, mean_vector, cov_matrix):
    """Computes mean and variance of a function that depends on other variables

    Args:
        func: function of variables
        mean_vector: array with means of variable
        cov_matrix: covariance matrix
    """
    mu = func(*mean_vector)

    d = derivative(func, points=mean_vector)

    var = 0
    for i in range(len(mean_vector)):
        for j in range(len(mean_vector)):
            var += d[i]*d[j]*(cov_matrix[i][j])

    return mu, var
