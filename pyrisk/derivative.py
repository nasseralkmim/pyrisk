"""Computes derivative with finite differences"""
import numpy as np


def derivative(func, points, eps=1e-8):
    """Compute derivative of func at points using finite differences
    
       ddx = \\frac{func(points + eps) - func(points - eps)}{2 * eps}
    
    Args:
        func (function): function with N parameters
        points (array): array with N-dimension

    Returns:
        derivative: list with derivative
        for instance:
            derivative[0] = d/d X1 func evaluated at points[0]
            derivative[1] = d/d X2 func evaluated at points[1]
    """
    d = []
    for i, p in enumerate(points):
        step_up, step_down = list(points), list(points)  # copy list

        # approximate derivative by tangent line with eps distance at
        step_up[i] = p + eps
        step_down[i] = p - eps
        d.append((func(*step_up) - func(*step_down))/(2*eps))

    return np.array(d)
