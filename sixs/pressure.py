"""
pressure.py
-----------
Translated from PRESSURE.f and PRESPLANE.f

Atmospheric profile adjustments for surface pressure (ground level)
and aircraft/plane-level simulations.
"""

import math
import numpy as np

from .commons import atm, planesim


# ---------------------------------------------------------------------------
# PRESSURE  (PRESSURE.f)
# ---------------------------------------------------------------------------

def pressure(xps):
    """
    Update the atmospheric profile for a given surface altitude/pressure
    and compute integrated water vapour (uw) and ozone (uo3) columns.

    The input xps is the surface altitude in km (stored negative in the
    original Fortran; sign flip done inside this function to match the
    Fortran convention).

    Bug fix (2024): the original Python translation had an off-by-one error
    in the index expression of the profile-shift loop.  The Fortran code
    (1-based) reads:
        do i=2, 33-iinf+1
            z(i) = z(i+iinf-1)
    which in 0-based Python is:
        for i in range(1, 34-iinf):
            z[i] = z[i+iinf]        # correct: shift by iinf positions
    The original translation wrote z[i+iinf-1] instead of z[i+iinf],
    which for iinf=0 reduces to z[i]=z[i-1] — a read-after-write hazard
    that zeroed the entire profile.  odrayl() then returned near-zero (or
    negative) Rayleigh optical depths, making xa ≈ 0.012 instead of the
    physically correct ≈ 0.18 at 427 nm and effectively eliminating the
    Rayleigh path contribution from the atmospheric correction.

    Parameters
    ----------
    xps : float – surface altitude (km, positive up)

    Returns
    -------
    uw  : float – integrated water-vapour column (g cm-2)
    uo3 : float – integrated ozone column (Dobson units)
    """
    z  = atm.z
    p  = atm.p
    t  = atm.t
    wh = atm.wh
    wo = atm.wo

    xps = -xps                          # Fortran sign convention
    if xps >= 100.0:
        xps = 99.99

    # Log-linear interpolation to find pressure at altitude xps
    i = 0
    while z[i] <= xps:
        i += 1
    isup = i
    iinf = i - 1

    xa = (z[isup] - z[iinf]) / math.log(p[isup] / p[iinf])
    xb = z[isup] - xa * math.log(p[isup])
    ps = math.exp((xps - xb) / xa)

    # Interpolate T, wh, wo at xps
    xalt  = xps
    xtemp = (t[isup]  - t[iinf])  / (z[isup] - z[iinf]) * (xalt - z[iinf]) + t[iinf]
    xwo   = (wo[isup] - wo[iinf]) / (z[isup] - z[iinf]) * (xalt - z[iinf]) + wo[iinf]
    xwh   = (wh[isup] - wh[iinf]) / (z[isup] - z[iinf]) * (xalt - z[iinf]) + wh[iinf]

    # Update profile: set level 0 to the target surface; shift levels above
    # xps into positions 1..33-iinf; pad any remaining levels.
    #
    # Faithful translation of the Fortran loop (1-based Fortran → 0-based Python):
    #   Fortran: z(1)=xalt first, then do i=2,33-iinf+1: z(i)=z(i+iinf-1)
    #   Python:  z[0]=xalt first, then for i in range(1,33-iinf): z[i]=z[i+iinf]
    # For the typical case iinf=0 (surface below z[1]): the loop is a no-op
    # (z[i]=z[i+0]=z[i]) and only z[0] is updated.  For iinf>0 (surface above
    # the first profile level), the loop shifts levels up by iinf positions.
    # Note: the loop must run BEFORE setting z[0] so that z[i+iinf] still
    # refers to the original (unmodified) profile values.
    for i in range(1, 34 - iinf):
        z[i]  = z[i + iinf]
        p[i]  = p[i + iinf]
        t[i]  = t[i + iinf]
        wh[i] = wh[i + iinf]
        wo[i] = wo[i + iinf]

    z[0]  = xalt;  p[0]  = ps;   t[0]  = xtemp
    wh[0] = xwh;   wo[0] = xwo

    # Pad levels above 33-iinf by linear interpolation between l and level 34
    l = 34 - iinf   # first index that needs padding (0-based)
    for i in range(l, 34):
        frac  = (i - l) / (34 - l) if (34 - l) != 0 else 0.0
        z[i]  = (z[33] - z[l - 1]) * frac + z[l - 1]
        p[i]  = (p[33] - p[l - 1]) * frac + p[l - 1]
        t[i]  = (t[33] - t[l - 1]) * frac + t[l - 1]
        wh[i] = (wh[33] - wh[l - 1]) * frac + wh[l - 1]
        wo[i] = (wo[33] - wo[l - 1]) * frac + wo[l - 1]

    # Compute integrated columns
    g    = 98.1
    air  = 0.028964 / 0.0224
    ro3  = 0.048    / 0.0224

    rmwh = np.zeros(34)
    rmo3 = np.zeros(34)
    for k in range(33):
        roair   = air * 273.16 * p[k] / (1013.25 * t[k])
        rmwh[k] = wh[k] / (roair * 1000.0)
        rmo3[k] = wo[k] / (roair * 1000.0)

    uw  = 0.0
    uo3 = 0.0
    for k in range(1, 33):
        ds   = (p[k - 1] - p[k]) / p[0]
        uw  += ((rmwh[k] + rmwh[k - 1]) / 2.0) * ds
        uo3 += ((rmo3[k] + rmo3[k - 1]) / 2.0) * ds

    uw  = uw  * p[0] * 100.0 / g
    uo3 = uo3 * p[0] * 100.0 / g
    uo3 = 1000.0 * uo3 / ro3

    return uw, uo3


# ---------------------------------------------------------------------------
# PRESPLANE  (PRESPLANE.f)
# ---------------------------------------------------------------------------

def presplane(xpp):
    """
    Build a plane-level atmospheric profile and compute integrated columns
    plus the Rayleigh optical depth conversion factor ftray.

    Parameters
    ----------
    xpp : float – plane altitude above ground (km)

    Returns
    -------
    uw    : float – integrated water-vapour column (g cm-2)
    uo3   : float – integrated ozone column (Dobson units)
    xpp   : float – updated plane altitude (absolute km)
    ftray : float – Rayleigh OD conversion factor (rp/rt)
    """
    z  = atm.z
    p  = atm.p
    t  = atm.t
    wh = atm.wh
    wo = atm.wo

    zpl  = planesim.zpl
    ppl  = planesim.ppl
    tpl  = planesim.tpl
    whpl = planesim.whpl
    wopl = planesim.wopl

    xpp = xpp + z[0]
    if xpp >= 100.0:
        xpp = 1000.0

    # Log-linear interpolation
    i = 0
    while i < len(z) - 1 and z[i] <= xpp:
        i += 1
    isup = min(i, len(z) - 1)
    iinf = max(isup - 1, 0)

    dz = z[isup] - z[iinf]
    dp = p[isup] / p[iinf] if p[iinf] > 0 else 1.0
    xa = dz / math.log(dp) if (dz != 0 and dp > 0 and dp != 1.0) else 1.0
    xb = z[isup] - xa * math.log(max(p[isup], 1e-10))
    ps = math.exp((xpp - xb) / xa) if xa != 0 else p[isup]

    xalt  = xpp
    interp = (xalt - z[iinf]) / dz if dz != 0 else 0.0
    xtemp = (t[isup]  - t[iinf])  * interp + t[iinf]
    xwo   = (wo[isup] - wo[iinf]) * interp + wo[iinf]
    xwh   = (wh[isup] - wh[iinf]) * interp + wh[iinf]

    # Build plane-level profile
    for i in range(iinf):
        zpl[i]  = z[i]
        ppl[i]  = p[i]
        tpl[i]  = t[i]
        whpl[i] = wh[i]
        wopl[i] = wo[i]

    zpl[iinf]  = xalt;  ppl[iinf]  = ps;    tpl[iinf]  = xtemp
    whpl[iinf] = xwh;   wopl[iinf] = xwo

    for i in range(iinf + 1, 34):
        zpl[i]  = zpl[iinf]
        ppl[i]  = ppl[iinf]
        tpl[i]  = tpl[iinf]
        whpl[i] = whpl[iinf]
        wopl[i] = wopl[iinf]

    # Compute ftray and integrated columns
    g    = 98.1
    air  = 0.028964 / 0.0224
    ro3  = 0.048    / 0.0224

    rmwh = np.zeros(34)
    rmo3 = np.zeros(34)
    rt   = 0.0
    rp   = 0.0

    for k in range(33):
        roair   = air * 273.16 * ppl[k] / (1013.25 * tpl[k])
        rmwh[k] = wh[k] / (roair * 1000.0)
        rmo3[k] = wo[k] / (roair * 1000.0)
        rt += (p[k + 1] / t[k + 1] + p[k] / t[k]) * (z[k + 1] - z[k])
        rp += (ppl[k + 1] / tpl[k + 1] + ppl[k] / tpl[k]) * (zpl[k + 1] - zpl[k])

    ftray = rp / rt

    uw  = 0.0
    uo3 = 0.0
    for k in range(1, 33):
        ds   = (ppl[k - 1] - ppl[k]) / ppl[0]
        uw  += ((rmwh[k] + rmwh[k - 1]) / 2.0) * ds
        uo3 += ((rmo3[k] + rmo3[k - 1]) / 2.0) * ds

    uw  = uw  * ppl[0] * 100.0 / g
    uo3 = uo3 * ppl[0] * 100.0 / g
    uo3 = 1000.0 * uo3 / ro3

    return uw, uo3, xpp, ftray