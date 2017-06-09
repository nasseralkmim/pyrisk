def derivative(func, points, eps=1e-8):
    """Compute derivative of func at points

    Returns:
        derivative: list with derivative, for instance
        derivate[0] = d/d X1 func evaluated at points[0]
        and derivative[1] = d/d X2 func evaluated at points[1]
    """
    d = []
    for i, p in enumerate(points):
        step_up, step_down = list(points), list(points)  # copy list

        # approximate derivative by tangent line with eps distance at
        step_up[i] = p + eps
        step_down[i] = p - eps
        d.append((func(*step_up) - func(*step_down))/(2*eps))

    return d
