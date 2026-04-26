"""
interp.py
---------
Translated from SPLINE.f, SPLINT.f, SPLIE2.f, SPLIN2.f,
               INTERP.f, EQUIVWL.f, TRUNCA.f

Interpolation and spectral utilities.
"""

import math
import numpy as np

from .commons import aer, disc, ffu, delta_sigma, trunc
from .utils   import solirr
from .gauss   import gauss


# ---------------------------------------------------------------------------
# SPLINE / SPLINT  (cubic spline)
# ---------------------------------------------------------------------------

def spline(x, y, yp1=1.0e31, ypn=1.0e31):
    """
    Compute the second derivatives of a cubic spline interpolant.

    Parameters
    ----------
    x, y : array-like – tabulated (x, y) data (length n)
    yp1  : float – first derivative at x[0]  (>0.99e30 → natural spline)
    ypn  : float – first derivative at x[-1] (>0.99e30 → natural spline)

    Returns
    -------
    y2 : list – second derivatives at each node
    """
    n  = len(x)
    y2 = [0.0] * n
    u  = [0.0] * n

    if yp1 > 0.99e30:
        y2[0] = 0.0
        u[0]  = 0.0
    else:
        y2[0] = -0.5
        u[0]  = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1)

    for i in range(1, n - 1):
        sig    = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p      = sig * y2[i - 1] + 2.0
        y2[i]  = (sig - 1.0) / p
        u[i]   = ((6.0 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i])
                           - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
                   / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p)

    if ypn > 0.99e30:
        qn = 0.0
        un = 0.0
    else:
        qn = 0.5
        un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))

    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0)

    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]

    return y2


def splint(xa, ya, y2a, x):
    """
    Cubic spline interpolation.

    Parameters
    ----------
    xa, ya : array-like – tabulated (x, y)
    y2a    : array-like – second derivatives from spline()
    x      : float – interpolation point

    Returns
    -------
    y : float – interpolated value
    """
    n   = len(xa)
    klo = 0
    khi = n - 1
    while khi - klo > 1:
        k = (khi + klo) // 2
        if xa[k] > x:
            khi = k
        else:
            klo = k

    h = xa[khi] - xa[klo]
    if h == 0.0:
        raise ValueError("splint: duplicate xa values")

    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h
    y = (a * ya[klo] + b * ya[khi]
         + ((a**3 - a) * y2a[klo] + (b**3 - b) * y2a[khi]) * (h**2) / 6.0)
    return y


# ---------------------------------------------------------------------------
# SPLIE2 / SPLIN2  (2-D spline)
# ---------------------------------------------------------------------------

def splie2(x2a, ya):
    """
    Compute 2-D spline second derivatives (row by row).

    Parameters
    ----------
    x2a : array-like – column coordinates (length n)
    ya  : 2-D array  – data, shape (m, n)

    Returns
    -------
    y2a : 2-D list – second derivatives, shape (m, n)
    """
    m   = len(ya)
    n   = len(x2a)
    y2a = [[0.0] * n for _ in range(m)]
    for j in range(m):
        y2a[j] = spline(x2a, ya[j])
    return y2a


def splin2(x1a, x2a, ya, y2a, x1, x2):
    """
    2-D cubic spline interpolation.

    Parameters
    ----------
    x1a  : array-like – row coordinates    (length m)
    x2a  : array-like – column coordinates (length n)
    ya   : 2-D array  – data, shape (m, n)
    y2a  : 2-D array  – second derivs from splie2(), shape (m, n)
    x1   : float – first  coordinate of interpolation point
    x2   : float – second coordinate of interpolation point

    Returns
    -------
    y : float – interpolated value
    """
    m      = len(x1a)
    yytmp  = [0.0] * m
    for j in range(m):
        yytmp[j] = splint(x2a, ya[j], y2a[j], x2)

    y2tmp = spline(x1a, yytmp)
    return splint(x1a, yytmp, y2tmp, x1)


# ---------------------------------------------------------------------------
# INTERP  (INTERP.f)
# ---------------------------------------------------------------------------

def interp(iaer, idatmp, wl, taer55, taer55p, xmud):
    """
    Interpolate atmospheric radiative-transfer quantities to wavelength wl.

    Uses global commons: aer, disc, delta_sigma.

    Parameters
    ----------
    iaer    : int   – aerosol type (0 = no aerosol)
    idatmp  : int   – atmospheric profile type (0 = no atmosphere)
    wl      : float – wavelength (µm)
    taer55  : float – aerosol optical depth at 550 nm (ground to TOA)
    taer55p : float – aerosol optical depth at 550 nm (ground to plane)
    xmud    : float – cosine of scattering angle

    Returns
    -------
    dict with keys:
        romix, rorayl, roaero, phaa, phar,
        tsca, tray, trayp, taer, taerp,
        dtott, utott, astot, asray, asaer,
        utotr, utota, dtotr, dtota
    """
    ext   = aer.ext
    ome   = aer.ome
    phase = aer.phase

    roatm  = disc.roatm
    dtdir  = disc.dtdir
    dtdif  = disc.dtdif
    utdir  = disc.utdir
    utdif  = disc.utdif
    sphal  = disc.sphal
    wldis  = disc.wldis
    trayl  = disc.trayl
    traypl = disc.traypl

    delta  = delta_sigma.delta

    # Find bracketing wavelength indices
    linf = 0
    for ll in range(9):
        if wl > wldis[ll] and wl <= wldis[ll + 1]:
            linf = ll
    if wl > wldis[9]:
        linf = 8
    lsup = linf + 1

    # Initialise outputs
    romix = rorayl = roaero = phaa = phar = 0.0
    tsca  = tray  = trayp  = taer = taerp = 0.0
    dtott = utott = astot  = asray = asaer = 0.0
    utotr = utota = dtotr  = dtota = 0.0

    coef  = math.log(wldis[lsup] / wldis[linf])
    wlinf = wldis[linf]

    # Phase functions
    phaa  = 0.0
    roaero = 0.0
    dtota  = 1.0
    utota  = 1.0
    asaer  = 0.0
    taer   = 0.0
    taerp  = 0.0

    if iaer != 0 and phase[linf] > 0 and phase[lsup] > 0:
        alphaa = math.log(phase[lsup] / phase[linf]) / coef
        betaa  = phase[linf] / (wlinf ** alphaa)
        phaa   = betaa * (wl ** alphaa)

    d2   = 2.0 + delta
    phar = ((2.0 * (1.0 - delta) / d2) * 0.75 * (1.0 + xmud * xmud)
            + 3.0 * delta / d2)

    if idatmp == 0:
        # No atmosphere
        tray  = 0.0
        trayp = 0.0
        dtotr = 1.0
        utotr = 1.0
        dtott = 1.0
        utott = 1.0
        asray = 0.0
        astot = 0.0
    else:
        def _interp_log(a_lo, a_hi, wl_lo, coef, wl):
            if a_lo < 0.001 or a_hi < 0.001 or a_lo <= 0.0:
                return a_lo + (a_hi - a_lo) * (wl - wl_lo) / (wldis[lsup] - wl_lo) if coef else a_lo
            try:
                alpha = math.log(a_hi / a_lo) / coef
                beta  = a_lo / (wl_lo ** alpha)
                result = beta * (wl ** alpha)
                return result if math.isfinite(result) else a_lo
            except (OverflowError, ValueError, ZeroDivisionError):
                return a_lo

        rorayl = _interp_log(roatm[0, linf], roatm[0, lsup], wlinf, coef, wl)
        romix  = _interp_log(roatm[1, linf], roatm[1, lsup], wlinf, coef, wl)

        if iaer != 0:
            roaero = _interp_log(roatm[2, linf], roatm[2, lsup], wlinf, coef, wl)

        alphar = math.log(trayl[lsup] / trayl[linf]) / coef
        betar  = trayl[linf] / (wlinf ** alphar)
        tray   = betar * (wl ** alphar)

        alphar  = math.log(traypl[lsup] / traypl[linf]) / coef
        betar   = traypl[linf] / (wlinf ** alphar)
        trayp   = betar * (wl ** alphar)

        if iaer != 0:
            # tsca
            v1 = ext[lsup] * ome[lsup]
            v0 = ext[linf] * ome[linf]
            alphaa = math.log(v1 / v0) / coef
            betaa  = v0 / (wlinf ** alphaa)
            tsca   = taer55 * betaa * (wl ** alphaa) / ext[3]
            # taer
            alphaa = math.log(ext[lsup] / ext[linf]) / coef
            betaa  = ext[linf] / (wlinf ** alphaa)
            taerp  = taer55p * betaa * (wl ** alphaa) / ext[3]
            taer   = taer55  * betaa * (wl ** alphaa) / ext[3]

        # Downward transmittances
        drinf = dtdif[0, linf] + dtdir[0, linf]
        drsup = dtdif[0, lsup] + dtdir[0, lsup]
        alphar = math.log(drsup / drinf) / coef
        betar  = drinf / (wlinf ** alphar)
        dtotr  = betar * (wl ** alphar)

        dtinf = dtdif[1, linf] + dtdir[1, linf]
        dtsup = dtdif[1, lsup] + dtdir[1, lsup]
        alphac = math.log((dtsup * drinf) / (dtinf * drsup)) / coef
        betac  = (dtinf / drinf) / (wlinf ** alphac)
        dtotc  = betac * (wl ** alphac)

        if iaer != 0:
            dainf = dtdif[2, linf] + dtdir[2, linf]
            dasup = dtdif[2, lsup] + dtdir[2, lsup]
            alphaa = math.log(dasup / dainf) / coef
            betaa  = dainf / (wlinf ** alphaa)
            dtota  = betaa * (wl ** alphaa)

        dtott = dtotc * dtotr

        # Upward transmittances
        urinf = utdif[0, linf] + utdir[0, linf]
        ursup = utdif[0, lsup] + utdir[0, lsup]
        alphar = math.log(ursup / urinf) / coef
        betar  = urinf / (wlinf ** alphar)
        utotr  = betar * (wl ** alphar)

        utinf = utdif[1, linf] + utdir[1, linf]
        utsup = utdif[1, lsup] + utdir[1, lsup]
        alphac = math.log((utsup * urinf) / (utinf * ursup)) / coef
        betac  = (utinf / urinf) / (wlinf ** alphac)
        utotc  = betac * (wl ** alphac)

        if iaer != 0:
            uainf = utdif[2, linf] + utdir[2, linf]
            uasup = utdif[2, lsup] + utdir[2, lsup]
            try:
                alphaa = math.log(uasup / uainf) / coef
                betaa  = uainf / (wlinf ** alphaa)
                utota  = betaa * (wl ** alphaa)
                if not math.isfinite(utota):
                    utota = uainf
            except (OverflowError, ValueError, ZeroDivisionError):
                utota  = uainf

        utott = utotc * utotr

        # Spherical albedos
        arinf  = sphal[0, linf];  arsup = sphal[0, lsup]
        alphar = math.log(arsup / arinf) / coef
        betar  = arinf / (wlinf ** alphar)
        asray  = betar * (wl ** alphar)

        atinf  = sphal[1, linf];  atsup = sphal[1, lsup]
        alphac = math.log(atsup / atinf) / coef
        betac  = atinf / (wlinf ** alphac)
        astot  = betac * (wl ** alphac)

        if iaer != 0:
            aainf  = sphal[2, linf];  aasup = sphal[2, lsup]
            alphaa = math.log(aasup / aainf) / coef
            betaa  = aainf / (wlinf ** alphaa)
            asaer  = betaa * (wl ** alphaa)

    return dict(
        romix=romix, rorayl=rorayl, roaero=roaero,
        phaa=phaa,   phar=phar,     tsca=tsca,
        tray=tray,   trayp=trayp,   taer=taer,   taerp=taerp,
        dtott=dtott, utott=utott,   astot=astot,
        asray=asray, asaer=asaer,   utotr=utotr,
        utota=utota, dtotr=dtotr,   dtota=dtota,
    )


# ---------------------------------------------------------------------------
# EQUIVWL  (EQUIVWL.f)
# ---------------------------------------------------------------------------

def equivwl(iinf, isup, step):
    """
    Compute the equivalent (solar-weighted) wavelength over a spectral band.

    Uses global common: ffu (s, wlinf, wlsup).

    Parameters
    ----------
    iinf, isup : int   – indices into the filter array (1-based, Fortran style)
    step       : float – wavelength step (µm)

    Returns
    -------
    wlmoy : float – equivalent wavelength (µm)
    """
    s = ffu.s

    seb    = 0.0
    wlwave = 0.0

    for l in range(iinf, isup + 1):
        sbor = s[l - 1]                           # convert to 0-based
        if l == iinf or l == isup:
            sbor *= 0.5
        wl   = 0.25 + (l - 1) * step
        swl  = solirr(wl)
        coef = sbor * step * swl
        seb    += coef
        wlwave += wl * coef

    wlmoy = wlwave / seb
    return wlmoy


# ---------------------------------------------------------------------------
# TRUNCA  (TRUNCA.f)
# ---------------------------------------------------------------------------

def trunca():
    """
    Phase function truncation and Legendre expansion.

    Uses / updates global common: trunc (pha, betal).

    Returns
    -------
    coeff : float – truncation coefficient (1 - betal[0] before normalisation)
    """
    pha   = trunc.pha
    betal = trunc.betal

    nbmu = 83
    nang = 80

    ptemp = pha.copy()

    cosang, weight = gauss(-1.0, 1.0, nang)

    rmu = np.zeros(nbmu)
    ga  = np.zeros(nbmu)

    for j in range(40):
        rmu[j + 1] = cosang[j]
        ga[j + 1]  = weight[j]

    rmu[0]  = -1.0;  ga[0]  = 0.0
    rmu[42] =  0.0;  ga[42] = 0.0

    for j in range(40, 80):
        rmu[j + 2] = cosang[j]
        ga[j + 2]  = weight[j]

    rmu[82] = 1.0;  ga[82] = 0.0

    # Find k: last index with rmu <= 0.8
    k = 0
    for j in range(nbmu):
        if rmu[j] <= 0.8:
            k = j

    # Find kk: last index with rmu <= 0.94
    kk = 0
    for j in range(nbmu):
        if rmu[j] <= 0.94:
            kk = j

    aa = ((math.log10(pha[kk]) - math.log10(pha[k]))
          / (math.acos(rmu[kk]) - math.acos(rmu[k])))
    x1 = math.log10(pha[kk])
    x2 = math.acos(rmu[kk])

    for j in range(kk + 1, nbmu):
        if abs(rmu[j] - 1.0) <= 1e-8:
            a = x1 - aa * x2
        else:
            a = x1 + aa * (math.acos(rmu[j]) - x2)
        ptemp[j] = 10.0 ** a

    pha[:] = ptemp

    # Legendre expansion
    betal[:] = 0.0
    for j in range(nbmu):
        x  = pha[j] * ga[j]
        rm = rmu[j]
        pl = np.zeros(83)           # pl[-1..81] → use pl[0]=pl_(-1), pl[1]=pl_(0) ...
        # Use direct recursion with offset: index k → pl_k stored at pl[k+1]
        pl_prev2 = 0.0   # P_{-1}
        pl_prev1 = 1.0   # P_0
        betal[0] += x * pl_prev1
        for kk2 in range(1, 81):
            pl_curr = ((2 * kk2 - 1.0) * rm * pl_prev1 - (kk2 - 1.0) * pl_prev2) / kk2
            betal[kk2] += x * pl_curr
            pl_prev2 = pl_prev1
            pl_prev1 = pl_curr

    for k2 in range(81):
        betal[k2] *= (2 * k2 + 1) * 0.5

    z1    = betal[0]
    coeff = 1.0 - z1
    betal[:] /= z1

    trunc.betal[:] = betal
    return coeff
