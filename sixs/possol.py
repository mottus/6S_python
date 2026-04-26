"""
possol.py
---------
Translated from POSSOL.f

Solar position calculation: zenith angle and azimuth.
"""

import math


def day_number(jday, month, ia=0):
    """
    Convert day-of-month + month to day-of-year.

    Parameters
    ----------
    jday  : int – day of month
    month : int – month (1-12)
    ia    : int – year (used only for leap-year check when non-zero)

    Returns
    -------
    j : int – day of year
    """
    if month <= 2:
        j = 31 * (month - 1) + jday
    elif month > 8:
        j = 31 * (month - 1) - ((month - 2) // 2) - 2 + jday
    else:
        j = 31 * (month - 1) - ((month - 1) // 2) - 2 + jday

    if ia != 0 and (ia % 4) == 0:
        j += 1
    return j


def pos_fft(j, tu, xlon, xlat):
    """
    Compute solar zenith and azimuth angles.

    Parameters
    ----------
    j    : int   – day of year
    tu   : float – universal time (decimal hours)
    xlon : float – longitude (degrees, East positive)
    xlat : float – latitude  (degrees, North positive)

    Returns
    -------
    asol : float – solar zenith angle  (degrees)
    phi0 : float – solar azimuth angle (degrees)
    """
    pi  = 3.14159265
    fac = pi / 180.0

    # Mean solar time (decimal hours)
    tsm = tu + xlon / 15.0
    xla = xlat * fac
    xj  = float(j)
    tet = 2.0 * pi * xj / 365.0

    # Time equation (minutes)
    a1 = 0.000075
    a2 = 0.001868
    a3 = 0.032077
    a4 = 0.014615
    a5 = 0.040849
    et = (a1 + a2 * math.cos(tet) - a3 * math.sin(tet)
          - a4 * math.cos(2.0 * tet) - a5 * math.sin(2.0 * tet))
    et = et * 12.0 * 60.0 / pi

    # True solar time
    tsv = tsm + et / 60.0 - 12.0

    # Hour angle (radians)
    ah = tsv * 15.0 * fac

    # Solar declination (radians)
    b1 = 0.006918
    b2 = 0.399912
    b3 = 0.070257
    b4 = 0.006758
    b5 = 0.000907
    b6 = 0.002697
    b7 = 0.001480
    delta = (b1 - b2 * math.cos(tet) + b3 * math.sin(tet)
             - b4 * math.cos(2.0 * tet) + b5 * math.sin(2.0 * tet)
             - b6 * math.cos(3.0 * tet) + b7 * math.sin(3.0 * tet))

    # Elevation and azimuth
    amuzero = (math.sin(xla) * math.sin(delta)
               + math.cos(xla) * math.cos(delta) * math.cos(ah))
    elev = math.asin(amuzero)
    az   = math.cos(delta) * math.sin(ah) / math.cos(elev)

    if abs(az) - 1.0 > 0.0:
        az = math.copysign(1.0, az)

    caz  = ((-math.cos(xla) * math.sin(delta)
              + math.sin(xla) * math.cos(delta) * math.cos(ah))
            / math.cos(elev))
    azim = math.asin(az)

    if caz <= 0.0:
        azim = pi - azim
    elif az <= 0.0:
        azim = 2.0 * pi + azim

    azim += pi
    pi2   = 2.0 * pi
    if azim > pi2:
        azim -= pi2

    elev = elev * 180.0 / pi
    asol = 90.0 - elev
    phi0 = azim / fac
    return asol, phi0


def possol(month, jday, tu, xlon, xlat):
    """
    Solar position: zenith angle and azimuth.

    Parameters
    ----------
    month : int   – month (1-12)
    jday  : int   – day of month
    tu    : float – universal time (decimal hours)
    xlon  : float – longitude (degrees)
    xlat  : float – latitude  (degrees)

    Returns
    -------
    asol : float – solar zenith angle  (degrees)
    phi0 : float – solar azimuth angle (degrees)

    Raises
    ------
    ValueError – if the sun is below the horizon (asol > 90°)
    """
    nojour       = day_number(jday, month)
    asol, phi0   = pos_fft(nojour, tu, xlon, xlat)
    if asol > 90.0:
        raise ValueError("The sun is not raised")
    return asol, phi0
