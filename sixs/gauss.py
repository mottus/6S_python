"""
gauss.py
--------
Translated from GAUSS.f

Computes abscissas and weights for Gauss-Legendre quadrature
on the interval [x1, x2] with n points.
"""

import math


def gauss(x1, x2, n):
    """
    Gauss-Legendre quadrature nodes and weights on [x1, x2].

    Parameters
    ----------
    x1, x2 : float  – integration limits
    n       : int    – number of quadrature points

    Returns
    -------
    x : list of float – abscissas (length n)
    w : list of float – weights   (length n)
    """
    EPS = 3.0e-14

    x = [0.0] * n
    w = [0.0] * n

    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)
    m  = (n + 1) // 2

    for i in range(1, m + 1):
        z = math.cos(math.pi * (i - 0.25) / (n + 0.5))

        while True:
            p1 = 1.0
            p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z  = z1 - p1 / pp
            if abs(z - z1) <= EPS:
                break

        if abs(z) < EPS:
            z = 0.0

        x[i - 1]     = xm - xl * z
        x[n - i]     = xm + xl * z
        w[i - 1]     = 2.0 * xl / ((1.0 - z * z) * pp * pp)
        w[n - i]     = w[i - 1]

    return x, w
