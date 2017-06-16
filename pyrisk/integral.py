"""Computes the integral of a function"""
def trapezoidal(func, interval=[0, 1], num_division=10):
    """computes integral using trapezoidal rule

    Returns:
        integral (float): approximated integral result
    """
    a, b = interval[0], interval[1]

    if b == 'inf' and a != '-inf':
        if a > 0:
            # int from 1/b to 1/a 1/t**2 f(1/t) dt
            b = 1 / a  # upper limit
            a = 0  # 1/inf lower limit
            f = func  # copy func

            def func_varchange(x):
                return 1 / x**2 * f(1 / x)

            func = func_varchange  # redefine func
        else:
            integral_right = trapezoidal(func, [1, 'inf'], num_division)
            integral_before = trapezoidal(func, [a, 1], num_division)
            return integral_before + integral_right

    if a == '-inf' and b != 'inf':
        if b < 0:
            # int from 1/b to 1/a 1/t**2 f(1/t) dt
            a = 1 / b  # lower limit
            b = 0  # upper limit
            f = func  # copy func

            def func_varchange(x):
                return 1 / x**2 * f(1 / x)

            func = func_varchange  # redefine func
        else:
            integral_left = trapezoidal(func, ['-inf', -1], num_division)
            integral_after = trapezoidal(func, [-1, b], num_division)
            return integral_left + integral_after

    if a == '-inf' and b == 'inf':
        integral_left = trapezoidal(func, ['-inf', -1], num_division)
        integral_center = trapezoidal(func, [-1, 1], num_division)
        integral_right = trapezoidal(func, [1, 'inf'], num_division)
        return integral_left + integral_right + integral_center

    integral = 0

    dx = (b - a) / num_division

    xi = a
    for i in range(num_division):
        xj = xi + dx

        try:
            integral += (func(xi) + func(xj)) * dx / 2
        except ZeroDivisionError:
            xi += 1e-8

        xi += dx

    return integral
