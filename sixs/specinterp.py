"""
specinterp.py
-------------
Translated from SPECINTERP.f and ENVIRO.f

Spectral interpolation of aerosol properties and environmental function.
"""

import math
import numpy as np

from .commons import aer, disc, trunc
from .interp  import trunca


# ---------------------------------------------------------------------------
# SPECINTERP  (SPECINTERP.f)
# ---------------------------------------------------------------------------

def specinterp(wl, taer55, taer55p):
    """
    Interpolate aerosol optical properties to wavelength wl and apply
    phase-function truncation.

    Parameters
    ----------
    wl           : float – wavelength (µm)
    taer55       : float – aerosol OD at 550 nm (full column)
    taer55p      : float – aerosol OD at 550 nm (above plane)

    Returns
    -------
    tamoy   : float – truncated aerosol OD (full column)
    tamoyp  : float – truncated aerosol OD (above plane)
    pizmoy  : float – truncated single-scatter albedo
    pizmoyp : float – same as pizmoy (plane level)
    """
    ext    = aer.ext
    ome    = aer.ome
    wldis  = disc.wldis
    pha    = trunc.pha

    # Phase function for SOS (stored separately in commons)
    # Here we use the phasel from the /sixs_sos/ common
    # (not yet modelled as a separate Python object — we read from trunc)

    linf = 0
    for ll in range(9):
        if wldis[ll] <= wl <= wldis[ll + 1]:
            linf = ll
    if wl > wldis[9]:
        linf = 8
    lsup = linf + 1

    coef  = math.log(wldis[lsup] / wldis[linf])
    wlinf = wldis[linf]

    # Single-scatter albedo × extinction → scattering coeff
    v0 = ext[linf] * ome[linf]
    v1 = ext[lsup] * ome[lsup]
    alphaa = math.log(v1 / v0) / coef
    betaa  = v0 / (wlinf ** alphaa)
    tsca   = taer55 * betaa * (wl ** alphaa) / ext[3]

    # Extinction coefficient
    alphaa = math.log(ext[lsup] / ext[linf]) / coef
    betaa  = ext[linf] / (wlinf ** alphaa)
    tamoy  = taer55  * betaa * (wl ** alphaa) / ext[3]
    tamoyp = taer55p * betaa * (wl ** alphaa) / ext[3]

    pizmoy  = tsca / tamoy if tamoy > 0.0 else 0.0
    pizmoyp = pizmoy

    # Interpolate phase function — uses trunc.pha which holds phasel at linf/lsup
    # (set by aeroso); simplified: keep as-is (trunca will recompute betal)
    coeff  = trunca()
    tamoy  = tamoy  * (1.0 - pizmoy  * coeff)
    tamoyp = tamoyp * (1.0 - pizmoyp * coeff)
    denom  = 1.0 - pizmoy * coeff
    pizmoy  = pizmoy  * (1.0 - coeff) / denom if denom > 0.0 else 0.0

    return tamoy, tamoyp, pizmoy, pizmoyp


# ---------------------------------------------------------------------------
# ENVIRO  (ENVIRO.f)
# ---------------------------------------------------------------------------

_ALT  = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
         10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 60.0]
_CFR1 = [0.730, 0.710, 0.656, 0.606, 0.560, 0.516, 0.473,
         0.433, 0.395, 0.323, 0.258, 0.209, 0.171, 0.142, 0.122, 0.070]
_CFR2 = [2.8, 1.51, 0.845, 0.634, 0.524, 0.465, 0.429,
         0.405, 0.390, 0.386, 0.409, 0.445, 0.488, 0.545, 0.608, 0.868]
_CFA1 = [0.239, 0.396, 0.588, 0.626, 0.612, 0.505, 0.454,
         0.448, 0.444, 0.445, 0.444, 0.448, 0.448, 0.448, 0.448, 0.448]
_CFA2 = [1.40, 1.20, 1.02, 0.86, 0.74, 0.56, 0.46, 0.42,
         0.38, 0.34, 0.30, 0.28, 0.27, 0.27, 0.27, 0.27]
_CFA3 = [9.17, 6.26, 5.48, 5.16, 4.74, 3.65, 3.24, 3.15,
         3.07, 2.97, 2.88, 2.83, 2.83, 2.83, 2.83, 2.83]


def enviro(difr, difa, r, palt, xmuv):
    """
    Environmental (adjacency) correction function.

    Parameters
    ----------
    difr : float – diffuse Rayleigh irradiance fraction
    difa : float – diffuse aerosol irradiance fraction
    r    : float – target–background radius (km)
    palt : float – plane altitude (km; ≥60 → satellite)
    xmuv : float – cos(view zenith)

    Returns
    -------
    fra : float – Rayleigh environmental function
    fae : float – aerosol environmental function
    fr  : float – combined environmental function
    """
    a0 = 1.3347; b0 = 0.57757
    a1 = -1.479; b1 = -1.5275

    if palt >= 60.0:
        fae0 = 1.0 - 0.448 * math.exp(-r * 0.27) - 0.552 * math.exp(-r * 2.83)
        fra0 = 1.0 - 0.930 * math.exp(-r * 0.080) - 0.070 * math.exp(-r * 1.100)
        xcfr1 = xcfr2 = xcfa1 = xcfa2 = xcfa3 = None   # not used
    else:
        i = 0
        while i < 16 and palt >= _ALT[i]:
            i += 1
        if i > 0:
            zmin = _ALT[i - 1]; zmax = _ALT[i]
            frac = (palt - zmin) / (zmax - zmin)
            xcfr1 = _CFR1[i - 1] + (_CFR1[i] - _CFR1[i - 1]) * frac
            xcfr2 = _CFR2[i - 1] + (_CFR2[i] - _CFR2[i - 1]) * frac
            xcfa1 = _CFA1[i - 1] + (_CFA1[i] - _CFA1[i - 1]) * frac
            xcfa2 = _CFA2[i - 1] + (_CFA2[i] - _CFA2[i - 1]) * frac
            xcfa3 = _CFA3[i - 1] + (_CFA3[i] - _CFA3[i - 1]) * frac
        else:
            xcfr1 = _CFR1[0]; xcfr2 = _CFR2[0]
            xcfa1 = _CFA1[0]; xcfa2 = _CFA2[0]; xcfa3 = _CFA3[0]
        fra0 = 1.0 - xcfr1 * math.exp(-r * xcfr2) - (1.0 - xcfr1) * math.exp(-r * 0.08)
        fae0 = 1.0 - xcfa1 * math.exp(-r * xcfa2) - (1.0 - xcfa1) * math.exp(-r * xcfa3)

    # View-zenith correction
    xlnv = math.log(xmuv)
    fra  = fra0 * (xlnv * (1.0 - fra0) + 1.0)
    fae  = fae0 * ((1.0 + a0 * xlnv + b0 * xlnv**2)
                   + fae0 * (a1 * xlnv + b1 * xlnv**2)
                   + fae0**2 * ((-a1 - a0) * xlnv + (-b1 - b0) * xlnv**2))

    if difa + difr > 1.0e-3:
        fr = (fae * difa + fra * difr) / (difa + difr)
    else:
        fr = 1.0

    return fra, fae, fr
