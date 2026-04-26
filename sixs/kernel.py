"""
kernel.py
---------
Translated from KERNEL.f

Computes the phase-function kernel expansion used in the successive
orders of scattering (ISO/OS) calculations.
"""

import math
import numpy as np

from .commons import trunc


def kernel_func(is_, mu, rm):
    """
    Compute generalised spherical-harmonic kernel for order is_.

    Parameters
    ----------
    is_ : int   – scattering order (0, 1, or higher)
    mu  : int   – number of Gauss half-angles
    rm  : array – Gauss nodes, indexed -mu..+mu (0-offset by +mu internally)

    Returns
    -------
    xpl : array (size 2*mu+1) – second-order Legendre values, indexed -mu..+mu
    psl : 2-D array           – generalised spherical harmonics (l, j)
    bp  : 2-D array           – phase-function kernel, shape (mu+1, 2*mu+1)
    """
    betal = trunc.betal

    ip1   = 80
    rac3  = math.sqrt(3.0)
    size_j = 2 * mu + 1

    def ji(k):   # map signed index -mu..+mu → 0-based
        return k + mu

    # psl[l][j]  l = 0..ip1,  j = -mu..+mu (stored 0-based)
    psl = np.zeros((ip1 + 2, size_j))

    if is_ == 0:
        for j in range(mu + 1):
            c = float(rm[ji(j)])
            psl[0, ji(-j)] = 1.0
            psl[0, ji(j)]  = 1.0
            psl[1, ji(j)]  = c
            psl[1, ji(-j)] = -c
            xdb = (3.0 * c * c - 1.0) * 0.5
            if abs(xdb) < 1.0e-30:
                xdb = 0.0
            psl[2, ji(-j)] = xdb
            psl[2, ji(j)]  = xdb
        psl[1, ji(0)] = rm[ji(0)]

    elif is_ == 1:
        for j in range(mu + 1):
            c  = float(rm[ji(j)])
            x  = 1.0 - c * c
            psl[0, ji(j)]  = 0.0
            psl[0, ji(-j)] = 0.0
            v = math.sqrt(x * 0.5)
            psl[1, ji(-j)] = v
            psl[1, ji(j)]  = v
            psl[2, ji(j)]  = c * v * rac3
            psl[2, ji(-j)] = -psl[2, ji(j)]
        psl[2, ji(0)] = -psl[2, ji(0)]

    else:
        a = 1.0
        for i in range(1, is_ + 1):
            a *= math.sqrt((i + is_) / float(i)) * 0.5
        # b = a * sqrt(is/(is+1)) * sqrt((is-1)/(is+2))   (not needed here)
        for j in range(mu + 1):
            c   = float(rm[ji(j)])
            xx  = 1.0 - c * c
            psl[is_ - 1, ji(j)] = 0.0
            xdb = a * (xx ** (is_ * 0.5))
            if abs(xdb) < 1.0e-30:
                xdb = 0.0
            psl[is_, ji(-j)] = xdb
            psl[is_, ji(j)]  = xdb

    # Recurrence for higher orders
    k_start = 2 if is_ == 0 else is_
    ip = ip1
    if k_start < ip:
        ig = -1 if is_ != 1 else 1
        for l in range(k_start, ip):
            lp = l + 1
            lm = l - 1
            a  = (2 * l + 1.0) / math.sqrt((l + is_ + 1.0) * (l - is_ + 1.0))
            b  = math.sqrt(float((l + is_) * (l - is_))) / (2.0 * l + 1.0)
            for j in range(mu + 1):
                c   = float(rm[ji(j)])
                xdb = a * (c * psl[l, ji(j)] - b * psl[lm, ji(j)])
                if abs(xdb) < 1.0e-30:
                    xdb = 0.0
                psl[lp, ji(j)] = xdb
                if j != 0:
                    psl[lp, ji(-j)] = ig * xdb
            ig = -ig

    # xpl = psl[2, :]
    xpl = psl[2, :].copy()

    # bp[j, k] = sum_{l=is_}^{ip1} betal[l] * psl[l,j] * psl[l,k]
    # shape: (mu+1) × (2*mu+1)
    bp = np.zeros((mu + 1, size_j))
    for j in range(mu + 1):
        for k in range(-mu, mu + 1):
            sbp = 0.0
            if is_ <= ip1:
                for l in range(is_, ip1 + 1):
                    sbp += psl[l, ji(j)] * psl[l, ji(k)] * betal[l]
            if abs(sbp) < 1.0e-30:
                sbp = 0.0
            bp[j, ji(k)] = sbp

    return xpl, psl, bp
