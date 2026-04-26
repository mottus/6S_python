"""
oda550.py
---------
Translated from ODA550.f

Aerosol optical depth at 550 nm from visibility.
"""

import math
import numpy as np

from .commons import atm, delta_sigma


# Vertical aerosol number density profiles (particles cm-3)
# for standard visibilities V=23 km and V=5 km (34 levels)
_AN23 = np.array([
    2.828e+03, 1.244e+03, 5.371e+02, 2.256e+02, 1.192e+02,
    8.987e+01, 6.337e+01, 5.890e+01, 6.069e+01, 5.818e+01,
    5.675e+01, 5.317e+01, 5.585e+01, 5.156e+01, 5.048e+01,
    4.744e+01, 4.511e+01, 4.458e+01, 4.314e+01, 3.634e+01,
    2.667e+01, 1.933e+01, 1.455e+01, 1.113e+01, 8.826e+00,
    7.429e+00, 2.238e+00, 5.890e-01, 1.550e-01, 4.082e-02,
    1.078e-02, 5.550e-05, 1.969e-08, 0.000e+00,
], dtype=np.float64)

_AN5 = np.array([
    1.378e+04, 5.030e+03, 1.844e+03, 6.731e+02, 2.453e+02,
    8.987e+01, 6.337e+01, 5.890e+01, 6.069e+01, 5.818e+01,
    5.675e+01, 5.317e+01, 5.585e+01, 5.156e+01, 5.048e+01,
    4.744e+01, 4.511e+01, 4.458e+01, 4.314e+01, 3.634e+01,
    2.667e+01, 1.933e+01, 1.455e+01, 1.113e+01, 8.826e+00,
    7.429e+00, 2.238e+00, 5.890e-01, 1.550e-01, 4.082e-02,
    1.078e-02, 5.550e-05, 1.969e-08, 0.000e+00,
], dtype=np.float64)


def oda550(iaer, v):
    """
    Aerosol optical depth at 550 nm given visibility.

    Parameters
    ----------
    iaer : int   – aerosol type index (0 = no aerosol)
    v    : float – meteorological visibility (km)

    Returns
    -------
    taer55 : float – aerosol optical depth at 550 nm
    """
    taer55 = 0.0

    if abs(v) <= 0.0 or iaer == 0:
        return taer55

    z     = atm.z
    sigma = delta_sigma.sigma

    for k in range(32):              # Fortran k=1..32
        dz    = z[k + 1] - z[k]
        bn5   = _AN5[k]
        bn51  = _AN5[k + 1]
        bn23  = _AN23[k]
        bn231 = _AN23[k + 1]

        az   = (115.0 / 18.0) * (bn5  - bn23)
        az1  = (115.0 / 18.0) * (bn51 - bn231)
        bz   = (5.0 * bn5  / 18.0) - (23.0 * bn23  / 18.0)
        bz1  = (5.0 * bn51 / 18.0) - (23.0 * bn231 / 18.0)

        bnz  = az  / v - bz
        bnz1 = az1 / v - bz1

        ev = dz * math.exp((math.log(bnz) + math.log(bnz1)) * 0.5)
        taer55 += ev * sigma * 1.0e-3

    return taer55
