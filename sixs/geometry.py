"""
geometry.py
-----------
Translated from POSGE.f, POSGW.f, POSMTO.f, POSNOA.f, POSLAN.f, POSSPO.f

Satellite geometry: compute solar and viewing angles from pixel coordinates
for various satellite platforms.
"""

import math
from .possol  import possol
from .commons import ier as ier_common
from .utils   import print_error


def _geos_common(month, jday, tu, nc, nl, alti, deltax, deltay,
                 sublon_offset):
    """
    Common logic for geostationary satellites (GOES-E, GOES-W, Meteosat).

    Parameters
    ----------
    alti          : float – orbit altitude above Earth surface (km)
    deltax, deltay: float – pixel size in radians
    sublon_offset : float – sub-satellite longitude (degrees)

    Returns
    -------
    asol, phi0, avis, phiv, xlon, xlat : floats
    """
    yr = float(nl) - 8665.5
    xr = float(nc) - 6498.5
    re  = 6378.155
    aaa = 1.0 / 297.0
    rp  = re / (1.0 + aaa)
    pi  = 3.1415926
    cdr = pi / 180.0
    crd = 180.0 / pi
    rs  = re + alti

    x = xr * deltax * cdr
    y = yr * deltay * cdr

    tanx = math.tan(x)
    tany = math.tan(y)
    val1 = 1.0 + tanx**2
    val2 = 1.0 + (tany * (1.0 + aaa))**2
    yk   = rs / re
    cosx2 = 1.0 / (val1 * val2)

    if 1.0 / cosx2 > yk**2 / (yk**2 - 1.0):
        print_error('no possibility to compute lat. and long.')
        return None

    sn   = (rs - re * math.sqrt(yk**2 - (yk**2 - 1.0) / cosx2)) / (1.0 / cosx2)
    zt   = rs - sn
    xt   = -(sn * tanx)
    yt   = sn * tany / math.cos(x)
    teta = math.asin(yt / rp)
    ylat = math.atan(math.tan(teta) * rp / re)
    ylon = math.atan(xt / zt)

    xlat = ylat * crd
    xlon = ylon * crd + sublon_offset

    asol, phi0 = possol(month, jday, tu, xlon, xlat)
    if ier_common.ier:
        return None

    ylon = xlon * pi / 180.0 - sublon_offset * cdr
    ylat = xlat * pi / 180.0
    gam  = math.sqrt((1.0 / cosx2 - 1.0) * cosx2)
    avis = math.degrees(math.asin((1.0 + alti / re) * gam))
    phiv = math.degrees(math.atan2(math.tan(ylon), math.sin(ylat)) + pi)

    return asol, phi0, avis, phiv, xlon, xlat


def posge(month, jday, tu, nc, nl):
    """
    GOES-East geometry.

    Returns
    -------
    asol, phi0, avis, phiv, xlon, xlat : floats
    """
    alti   = 42107.0 - 6378.155
    deltax = 18.0 / 12997.0
    deltay = 20.0 / 17331.0
    return _geos_common(month, jday, tu, nc, nl, alti, deltax, deltay, -75.0)


def posgw(month, jday, tu, nc, nl):
    """
    GOES-West geometry.

    Returns
    -------
    asol, phi0, avis, phiv, xlon, xlat : floats
    """
    alti   = 42147.0 - 6378.155
    deltax = 18.0 / 12997.0
    deltay = 20.0 / 17331.0
    return _geos_common(month, jday, tu, nc, nl, alti, deltax, deltay, -135.0)


def posmto(month, jday, tu, nc, nl):
    """
    Meteosat geometry.

    Returns
    -------
    asol, phi0, avis, phiv, xlon, xlat : floats
    """
    yr    = float(nl) - 1250.5
    xr    = float(nc) - 2500.5
    alti  = 42164.0 - 6378.155
    re    = 6378.155
    aaa   = 1.0 / 297.0
    rp    = re / (1.0 + aaa)
    pi    = 3.1415926
    cdr   = pi / 180.0
    crd   = 180.0 / pi
    rs    = re + alti
    deltax = 18.0 / 5000.0
    deltay = 18.0 / 2500.0

    x = xr * deltax * cdr
    y = yr * deltay * cdr

    tanx  = math.tan(x)
    tany  = math.tan(y)
    val1  = 1.0 + tanx**2
    val2  = 1.0 + (tany * (1.0 + aaa))**2
    yk    = rs / re
    cosx2 = 1.0 / (val1 * val2)

    if 1.0 / cosx2 > yk**2 / (yk**2 - 1.0):
        print_error('no possibility to compute lat. and long.')
        return None

    sn   = (rs - re * math.sqrt(yk**2 - (yk**2 - 1.0) / cosx2)) / (1.0 / cosx2)
    zt   = rs - sn
    xt   = -(sn * tanx)
    yt   = sn * tany / math.cos(x)
    teta = math.asin(yt / rp)
    ylat = math.atan(math.tan(teta) * rp / re)
    ylon = math.atan(xt / zt)

    xlat = ylat * crd
    xlon = ylon * crd    # sub-satellite at 0°E

    asol, phi0 = possol(month, jday, tu, xlon, xlat)
    if ier_common.ier:
        return None

    ylon  = xlon * pi / 180.0
    ylat  = xlat * pi / 180.0
    gam   = math.sqrt((1.0 / cosx2 - 1.0) * cosx2)
    avis  = math.degrees(math.asin((1.0 + alti / re) * gam))
    phiv  = math.degrees(math.atan2(math.tan(ylon), math.sin(ylat)) + pi)

    return asol, phi0, avis, phiv, xlon, xlat


def posnoa(month, jday, tu, nc, xlonan, hna, campm):
    """
    NOAA polar orbiter geometry.

    Parameters
    ----------
    nc     : int   – pixel column (1..2048)
    xlonan : float – longitude of ascending node (degrees)
    hna    : float – time of ascending node (decimal hours UT)
    campm  : float – +1 for AM platform, -1 for PM platform

    Returns
    -------
    asol, phi0, avis, phiv, xlon, xlat : floats
    """
    pi     = 3.1415926
    r      = 860.0 / 6378.155
    ai     = 98.96 * pi / 180.0
    an     = 360.0 * pi / (6119.0 * 180.0)
    ylonan = xlonan * pi / 180.0

    t    = tu * 3600.0
    hnam = hna * 3600.0
    u    = campm * (t - hnam) * an

    delt = (nc - (2048 + 1) / 2.0) * 55.385 / ((2048.0 - 1.0) / 2.0)
    delt = campm * delt * pi / 180.0

    avis_r = math.asin((1.0 + r) * math.sin(delt))
    d      = avis_r - delt
    y      = math.cos(d) * math.cos(ai) * math.sin(u) - math.sin(ai) * math.sin(d)
    z      = math.cos(d) * math.sin(ai) * math.sin(u) + math.cos(ai) * math.sin(d)
    ylat   = math.asin(z)
    cosy   = math.cos(d) * math.cos(u) / math.cos(ylat)
    siny   = y / math.cos(ylat)
    ylon   = math.asin(siny)

    if cosy <= 0.0:
        if siny > 0.0:
            ylon = pi - ylon
        else:
            ylon = -(pi + ylon)

    ylo1 = ylon + ylonan - (t - hnam) * 2.0 * pi / 86400.0
    xlat = math.degrees(ylat)
    xlon = math.degrees(ylo1)

    asol, phi0 = possol(month, jday, tu, xlon, xlat)
    if ier_common.ier:
        return None

    zlat = math.asin(math.sin(ai) * math.sin(u))
    zlon = math.atan2(math.cos(ai) * math.sin(u), math.cos(u))

    if nc != 1024:
        xnum = math.sin(zlon - ylon) * math.cos(zlat) / math.sin(abs(d))
        xden = (math.sin(zlat) - math.sin(ylat) * math.cos(d)) / math.cos(ylat) / math.sin(abs(d))
        phiv = math.degrees(math.atan2(xnum, xden))
    else:
        phiv = 0.0

    avis = abs(math.degrees(avis_r))
    return asol, phi0, avis, phiv, xlon, xlat


def poslan(month, jday, tu, xlon, xlat):
    """
    Landsat-5 geometry (nadir viewing, avis=phiv=0).

    Returns
    -------
    asol, phi0, avis, phiv : floats
    """
    asol, phi0 = possol(month, jday, tu, xlon, xlat)
    return asol, phi0, 0.0, 0.0


def posspo(month, jday, tu, xlon, xlat):
    """
    SPOT geometry (nadir viewing, avis=phiv=0).

    Returns
    -------
    asol, phi0, avis, phiv : floats
    """
    asol, phi0 = possol(month, jday, tu, xlon, xlat)
    return asol, phi0, 0.0, 0.0
