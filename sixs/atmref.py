"""
atmref.py
---------
Translated from ATMREF.f and CHAND.f

Atmospheric reflectance computation.
"""

import math
from .commons import delta_sigma


def chand(xphi, xmuv, xmus, xtau):
    """
    Analytical approximation to molecular (Rayleigh) reflectance.
    (Chandrasekhar / successive orders method)

    Parameters
    ----------
    xphi : float – relative azimuth angle (degrees, 0=backscatter)
    xmuv : float – cos(view zenith)
    xmus : float – cos(solar zenith)
    xtau : float – Rayleigh optical depth

    Returns
    -------
    xrray : float – molecular reflectance
    """
    pi  = 3.1415927
    fac = pi / 180.0

    phios  = 180.0 - xphi
    xcosf1 = 1.0
    xcosf2 = math.cos(phios * fac)
    xcosf3 = math.cos(2.0 * phios * fac)

    xdep   = 0.0279
    xfd    = xdep / (2.0 - xdep)
    xfd    = (1.0 - xfd) / (1.0 + 2.0 * xfd)

    xph1 = 1.0 + (3.0 * xmus**2 - 1.0) * (3.0 * xmuv**2 - 1.0) * xfd / 8.0
    xph2 = (-xmus * xmuv * math.sqrt(1.0 - xmus**2) * math.sqrt(1.0 - xmuv**2)
            * xfd * 0.5 * 1.5)
    xph3 = (1.0 - xmus**2) * (1.0 - xmuv**2) * xfd * 0.5 * 0.375

    xitm = (1.0 - math.exp(-xtau * (1.0/xmus + 1.0/xmuv))) * xmus / (4.0 * (xmus + xmuv))
    xp1  = xph1 * xitm
    xp2  = xph2 * xitm
    xp3  = xph3 * xitm

    xitm   = (1.0 - math.exp(-xtau / xmus)) * (1.0 - math.exp(-xtau / xmuv))
    cfonc1 = xph1 * xitm
    cfonc2 = xph2 * xitm
    cfonc3 = xph3 * xitm

    xlntau = math.log(max(xtau, 1e-10))
    pl = [0.0] * 11   # 1-based: pl[1]..pl[10]
    pl[1]  = 1.0
    pl[2]  = xlntau
    pl[3]  = xmus + xmuv
    pl[4]  = xlntau * pl[3]
    pl[5]  = xmus * xmuv
    pl[6]  = xlntau * pl[5]
    pl[7]  = xmus**2 + xmuv**2
    pl[8]  = xlntau * pl[7]
    pl[9]  = xmus**2 * xmuv**2
    pl[10] = xlntau * pl[9]

    as0 = [0.0,  # index 0 unused
           0.33243832, -6.777104e-2, 0.16285370,  1.577425e-3,
          -0.30924818, -1.240906e-2,-0.10324388,  3.241678e-2,
           0.11493334, -3.503695e-2]
    as1 = [0.0, 0.19666292, -5.439061e-2]
    as2 = [0.0, 0.14545937, -2.910845e-2]

    fs0 = sum(pl[i] * as0[i] for i in range(1, 11))
    fs1 = pl[1] * as1[1] + pl[2] * as1[2]
    fs2 = pl[1] * as2[1] + pl[2] * as2[2]

    xitot1 = xp1 + cfonc1 * fs0 * xmus
    xitot2 = xp2 + cfonc2 * fs1 * xmus
    xitot3 = xp3 + cfonc3 * fs2 * xmus

    xrray = (xitot1 * xcosf1
             + xitot2 * xcosf2 * 2.0
             + xitot3 * xcosf3 * 2.0) / xmus
    return xrray


def atmref(iaer, tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
           phi, xmus, xmuv, phirad, nt, mu, np_, rm, gb, rp, xlm1, xlm2):
    """
    Compute atmospheric reflectances (Rayleigh, aerosol, mixed).

    Parameters
    ----------
    iaer                  : int   – aerosol flag
    tamoy, trmoy, pizmoy  : float – aerosol OD, Rayleigh OD, SSA
    tamoyp, trmoyp        : float – ODs above plane
    palt                  : float – plane altitude (km; >900 = satellite)
    phi, xmus, xmuv       : float – geometry
    phirad                : float – azimuth (radians)
    nt, mu, np_           : int   – discretization parameters
    rm, gb, rp            : arrays
    xlm1, xlm2            : arrays – radiance fields (modified in place)

    Returns
    -------
    rorayl : float – Rayleigh reflectance
    roaero : float – aerosol reflectance
    romix  : float – mixed reflectance
    """
    from .os_module import os_sos   # deferred import to avoid circularity

    def ji(k): return k + mu   # convert signed index to 0-based offset

    rorayl = 0.0
    roaero = 0.0

    if palt < 900.0 and palt > 0.0:
        rm[ji(-mu)] = -xmuv
        rm[ji(mu)]  =  xmuv
        rm[ji(0)]   = -xmus
        tamol  = 0.0; tamolp = 0.0
        os_sos(tamol, trmoy, pizmoy, tamolp, trmoyp, palt,
               phirad, nt, mu, np_, rm, gb, rp, xlm1)
        rorayl = xlm1[ji(-mu), 0] / xmus   # xlm1(-mu,1) in Fortran → 0-indexed col 0
    elif palt > 0.0:
        rorayl = chand(phi, xmuv, xmus, trmoy)
    # else rorayl = 0 (ground level)

    if iaer == 0:
        romix = rorayl
        return rorayl, roaero, romix

    # Mixed and aerosol reflectances
    if palt > 0.0:
        rm[ji(-mu)] = -xmuv
        rm[ji(mu)]  =  xmuv
        rm[ji(0)]   = -xmus
        os_sos(tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
               phirad, nt, mu, np_, rm, gb, rp, xlm2)
        romix = xlm2[ji(-mu), 0] / xmus

        tamol = 0.0; tamolp = 0.0
        os_sos(tamoy, tamol, pizmoy, tamoyp, tamolp, palt,
               phirad, nt, mu, np_, rm, gb, rp, xlm2)
        roaero = xlm2[ji(-mu), 0] / xmus
    else:
        roaero = 0.0
        romix  = 0.0

    return rorayl, roaero, romix
