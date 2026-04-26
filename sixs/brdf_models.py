"""
brdf_models.py
--------------
Translated from WALTALBE.f, WALTBRDF.f, RAHMALBE.f, RAHMBRDF.f,
               ROUJALBE.f, ROUJBRDF.f, HAPKALBE.f, HAPKBRDF.f,
               MINNALBE.f, MINNBRDF.f, BRDFGRID.f, VERSALBE.f,
               VERSBRDF.f, VERSTOOLS.f

BRDF (Bidirectional Reflectance Distribution Function) models.
Each model comes in two variants:
  *albe  – hemispherical (bi-hemispherical) albedo via Gauss integration
  *brdf  – directional BRDF at Gauss quadrature points
"""

import math
import numpy as np

from .gauss  import gauss
from .interp import splie2, splin2


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _gauss_hemisphere(nta=24, nfa=48):
    """Return (ta, wta, fa, wfa) for 2-D hemisphere Gauss integration."""
    pi   = math.acos(-1.0)
    ta, wta = gauss(0.0, pi / 2.0, nta)
    fa, wfa = gauss(0.0, 2.0 * pi, nfa)
    return ta, wta, fa, wfa


# ---------------------------------------------------------------------------
# Walt (polynomial) BRDF model  (WALTBRDF.f, WALTALBE.f)
# ---------------------------------------------------------------------------

def waltbrdf(a, ap, b, c, mu, np_, rm, rp):
    """
    Walt polynomial BRDF at quadrature angles.

    Parameters
    ----------
    a, ap, b, c : float  – model parameters
    mu, np_     : int    – number of Gauss angles / azimuth points
    rm          : array  – Gauss nodes (indexed -mu..+mu)
    rp          : array  – azimuth points (length np_)

    Returns
    -------
    brdfint : 2-D array, shape (2*mu+1, np_)
    """
    def ji(k): return k + mu
    brdfint = np.zeros((2 * mu + 1, np_))
    xmu = rm[ji(0)]
    ts  = math.acos(xmu)
    for k in range(np_):
        for j in range(1, mu + 1):
            view = rm[ji(j)]
            tv   = math.acos(view)
            fi   = rm[ji(-mu)] if j == mu else rp[k] + rm[ji(-mu)]
            phi  = fi
            brdfint[ji(j), k] = a * (ts**2 * tv**2) + ap * (ts**2 + tv**2) + b * ts * tv * math.cos(phi) + c
    return brdfint


def waltalbe(a, ap, b, c):
    """Walt polynomial hemispherical albedo."""
    nta, nfa = 24, 48
    ta, wta, fa, wfa = _gauss_hemisphere(nta, nfa)
    brdfalb = summ = 0.0
    for k in range(nfa):
        for j in range(nta):
            for l in range(nta):
                mu2 = math.cos(ta[j]); mu1 = math.cos(ta[l])
                si2 = math.sin(ta[j]); si1 = math.sin(ta[l])
                ts  = ta[j]; tv = ta[l]; phi = fa[k]
                pond = mu1 * mu2 * si1 * si2 * wfa[k] * wta[j] * wta[l]
                brdfv = a * ts**2 * tv**2 + ap * (ts**2 + tv**2) + b * ts * tv * math.cos(phi) + c
                brdfalb += brdfv * pond
                summ    += pond
    return brdfalb / summ


# ---------------------------------------------------------------------------
# Rahman–Pinty–Verstraete (RPV) BRDF  (RAHMBRDF.f, RAHMALBE.f)
# ---------------------------------------------------------------------------

def rahmbrdf(rho0, af, xk, mu, np_, rm, rp):
    """Rahman–Pinty–Verstraete BRDF at quadrature angles."""
    def ji(k): return k + mu
    pi = math.acos(-1.0)
    brdfint = np.zeros((2 * mu + 1, np_))
    mu1 = rm[ji(0)]
    for k in range(np_):
        for j in range(1, mu + 1):
            mu2 = rm[ji(j)]
            fi  = rm[ji(-mu)] if j == mu else rp[k] + rm[ji(-mu)]
            cospha = mu1 * mu2 + math.sqrt(1.0 - mu1**2) * math.sqrt(1.0 - mu2**2) * math.cos(fi)
            cospha = max(-1.0, min(1.0, cospha))
            phaang = math.acos(cospha)
            tante1 = math.sqrt(1.0 - mu1**2) / mu1
            tante2 = math.sqrt(1.0 - mu2**2) / mu2
            geofac = math.sqrt(tante1**2 + tante2**2 - 2.0 * tante1 * tante2 * math.cos(fi))
            coef1  = (mu1**(xk - 1.0)) * (mu2**(xk - 1.0)) / ((mu1 + mu2)**(1.0 - xk))
            phafun = (1.0 - af**2) / ((1.0 + af**2 - 2.0 * af * math.cos(pi - phaang))**1.5)
            coef2  = 1.0 + (1.0 - rho0) / (1.0 + geofac)
            brdfint[ji(j), k] = rho0 * coef1 * phafun * coef2
    return brdfint


def rahmalbe(rho0, af, xk):
    """Rahman–Pinty–Verstraete hemispherical albedo."""
    nta, nfa = 24, 48
    pi = math.acos(-1.0)
    ta, wta, fa, wfa = _gauss_hemisphere(nta, nfa)
    brdfalb = summ = 0.0
    for k in range(nfa):
        for j in range(nta):
            for l in range(nta):
                mu2 = math.cos(ta[j]); mu1 = math.cos(ta[l])
                si2 = math.sin(ta[j]); si1 = math.sin(ta[l])
                fi  = fa[k]
                cospha = mu1 * mu2 + math.sqrt(1.0 - mu1**2) * math.sqrt(1.0 - mu2**2) * math.cos(fi)
                cospha = max(-1.0, min(1.0, cospha))
                phaang = math.acos(cospha)
                tante1 = math.sqrt(1.0 - mu1**2) / mu1
                tante2 = math.sqrt(1.0 - mu2**2) / mu2
                geofac = math.sqrt(tante1**2 + tante2**2 - 2.0 * tante1 * tante2 * math.cos(fi))
                coef1  = (mu1**(xk - 1.0)) * (mu2**(xk - 1.0)) / ((mu1 + mu2)**(1.0 - xk))
                phafun = (1.0 - af**2) / ((1.0 + af**2 - 2.0 * af * math.cos(pi - phaang))**1.5)
                coef2  = 1.0 + (1.0 - rho0) / (1.0 + geofac)
                pond   = mu1 * mu2 * si1 * si2 * wfa[k] * wta[j] * wta[l]
                brdfalb += rho0 * coef1 * phafun * coef2 * pond
                summ    += pond
    return brdfalb / summ


# ---------------------------------------------------------------------------
# Roujean kernel BRDF  (ROUJBRDF.f, ROUJALBE.f)
# ---------------------------------------------------------------------------

def roujbrdf(k0, k1, k2, mu, np_, rm, rp):
    """Roujean geometric-optical BRDF at quadrature angles."""
    def ji(k): return k + mu
    pi = math.atan(1.0) * 4.0
    brdfint = np.zeros((2 * mu + 1, np_))
    xmus = rm[ji(0)]
    for k in range(np_):
        for j in range(1, mu + 1):
            xmuv = rm[ji(j)]
            fi   = rm[ji(-mu)] if j == mu else rp[k] + rm[ji(-mu)]
            fr   = math.acos(math.cos(fi))
            tts  = math.tan(math.acos(xmus))
            ttv  = math.tan(math.acos(xmuv))
            cpsi = xmus * xmuv + math.sqrt(1.0 - xmus**2) * math.sqrt(1.0 - xmuv**2) * math.cos(fi)
            cpsi = max(-1.0, min(1.0, cpsi))
            psi  = math.acos(cpsi)
            f2   = (4.0 / (3.0 * pi * (xmus + xmuv))) * ((pi / 2.0 - psi) * cpsi + math.sin(psi)) - 1.0 / 3.0
            ft   = tts**2 + ttv**2 - 2.0 * tts * ttv * math.cos(fr)
            f1   = (0.5 * ((pi - fr) * math.cos(fr) + math.sin(fr)) * tts * ttv
                    - tts - ttv - math.sqrt(ft)) / pi
            brdfint[ji(j), k] = k0 + k1 * f1 + k2 * f2
    return brdfint


def roujalbe(k0, k1, k2):
    """Roujean geometric-optical hemispherical albedo."""
    nta, nfa = 24, 48
    pi = math.atan(1.0) * 4.0
    ta, wta, fa, wfa = _gauss_hemisphere(nta, nfa)
    brdfalb = summ = 0.0
    for k in range(nfa):
        for j in range(nta):
            for l in range(nta):
                mu2 = math.cos(ta[j]); mu1 = math.cos(ta[l])
                si2 = math.sin(ta[j]); si1 = math.sin(ta[l])
                ts  = ta[j]; tv = ta[l]
                fr  = math.acos(math.cos(fa[k]))
                tts = math.tan(ts); ttv = math.tan(tv)
                xmus = math.cos(ts); xmuv = math.cos(tv)
                cpsi = xmus * xmuv + math.sin(ts) * math.sin(tv) * math.cos(fr)
                cpsi = max(-1.0, min(1.0, cpsi))
                psi  = math.acos(cpsi)
                f2   = (4.0 / (3.0 * pi * (xmus + xmuv))) * ((pi / 2.0 - psi) * cpsi + math.sin(psi)) - 1.0 / 3.0
                ft   = tts**2 + ttv**2 - 2.0 * tts * ttv * math.cos(fr)
                f1   = (0.5 * ((pi - fr) * math.cos(fr) + math.sin(fr)) * tts * ttv
                        - tts - ttv - math.sqrt(ft)) / pi
                pond = mu1 * mu2 * si1 * si2 * wfa[k] * wta[j] * wta[l]
                brdfalb += (k0 + k1 * f1 + k2 * f2) * pond
                summ    += pond
    return brdfalb / summ


# ---------------------------------------------------------------------------
# Hapke BRDF  (HAPKBRDF.f, HAPKALBE.f)
# ---------------------------------------------------------------------------

def hapkbrdf(om, af, s0, h, mu, np_, rm, rp):
    """Hapke photometric BRDF at quadrature angles."""
    def ji(k): return k + mu
    brdfint = np.zeros((2 * mu + 1, np_))
    mu1 = rm[ji(0)]
    for k in range(np_):
        for j in range(1, mu + 1):
            mu2 = rm[ji(j)]
            fi  = rm[ji(-mu)] if j == mu else rp[k] + rm[ji(-mu)]
            cg  = mu1 * mu2 + math.sqrt(1.0 - mu1**2) * math.sqrt(1.0 - mu2**2) * math.cos(fi)
            cg  = max(-1.0, min(1.0, cg))
            f   = om / 4.0 / (mu2 + mu1)
            h1  = (1.0 + 2.0 * mu1) / (1.0 + 2.0 * math.sqrt(1.0 - om) * mu1)
            h2  = (1.0 + 2.0 * mu2) / (1.0 + 2.0 * math.sqrt(1.0 - om) * mu2)
            h1h2 = h1 * h2
            pg   = (1.0 - af**2) / ((1.0 + af**2 + 2.0 * af * cg)**1.5)
            p0   = (1.0 - af**2) / ((1.0 + af**2 + 2.0 * af)**1.5)
            g    = math.acos(cg)
            bg   = (s0 / (om * p0)) / (1.0 + math.tan(g / 2.0) / h)
            brdfint[ji(j), k] = f * ((1.0 + bg) * pg + h1h2 - 1.0)
    return brdfint


def hapkalbe(om, af, s0, h):
    """Hapke photometric hemispherical albedo."""
    nta, nfa = 24, 48
    ta, wta, fa, wfa = _gauss_hemisphere(nta, nfa)
    brdfalb = summ = 0.0
    for k in range(nfa):
        for j in range(nta):
            for l in range(nta):
                mu2 = math.cos(ta[j]); mu1 = math.cos(ta[l])
                si2 = math.sin(ta[j]); si1 = math.sin(ta[l])
                fi  = fa[k]
                cg  = mu1 * mu2 + math.sqrt(1.0 - mu1**2) * math.sqrt(1.0 - mu2**2) * math.cos(fi)
                cg  = max(-1.0, min(1.0, cg))
                f   = om / 4.0 / (mu2 + mu1)
                h1  = (1.0 + 2.0 * mu1) / (1.0 + 2.0 * math.sqrt(1.0 - om) * mu1)
                h2  = (1.0 + 2.0 * mu2) / (1.0 + 2.0 * math.sqrt(1.0 - om) * mu2)
                pg  = (1.0 - af**2) / ((1.0 + af**2 + 2.0 * af * cg)**1.5)
                p0  = (1.0 - af**2) / ((1.0 + af**2 + 2.0 * af)**1.5)
                g   = math.acos(cg)
                bg  = (s0 / (om * p0)) / (1.0 + math.tan(g / 2.0) / h)
                pond = mu1 * mu2 * si1 * si2 * wfa[k] * wta[j] * wta[l]
                brdfalb += f * ((1.0 + bg) * pg + h1 * h2 - 1.0) * pond
                summ    += pond
    return brdfalb / summ


# ---------------------------------------------------------------------------
# Minnaert BRDF  (MINNBRDF.f, MINNALBE.f)
# ---------------------------------------------------------------------------

def minnbrdf(par1, par2, mu, np_, rm):
    """Minnaert limb-darkening BRDF at quadrature angles."""
    def ji(k): return k + mu
    brdfint = np.zeros((2 * mu + 1, np_))
    xmu = rm[ji(0)]
    for k in range(np_):
        for j in range(1, mu + 1):
            view = rm[ji(j)]
            brdfint[ji(j), k] = 0.5 * par2 * (par1 + 1.0) * ((xmu * view)**(par1 - 1.0))
    return brdfint


def minnalbe(par1, par2):
    """Minnaert hemispherical albedo (analytical)."""
    return 2.0 * par2 / (par1 + 1.0)


# ---------------------------------------------------------------------------
# BRDF grid interpolation  (BRDFGRID.f)
# ---------------------------------------------------------------------------

def brdfgrid(mu, np_, rm, rp, brdfdat, angmu, angphi):
    """
    Interpolate tabulated BRDF data onto quadrature angles using 2-D splines.

    Parameters
    ----------
    mu, np_         : int    – Gauss angle counts
    rm, rp          : arrays – Gauss nodes / azimuth points
    brdfdat         : 2-D array, shape (10, 13)
    angmu, angphi   : arrays – angle grids (length 10, 13)

    Returns
    -------
    brdfint : 2-D array, shape (2*mu+1, np_)
    """
    def ji(k): return k + mu
    brdfint  = np.zeros((2 * mu + 1, np_))
    brdftemp = splie2(angphi, brdfdat)
    for j in range(1, mu + 1):
        for k in range(np_):
            gaussmu  = rm[ji(j)]
            gaussphi = rp[k]
            brdfint[ji(j), k] = splin2(angmu, angphi, brdfdat, brdftemp,
                                        gaussmu, gaussphi)
    return brdfint


# ---------------------------------------------------------------------------
# Verstraete–Pinty (VP) BRDF  (VERSTOOLS.f → mvbp1)
# ---------------------------------------------------------------------------

def mvbp1(option, angles, optics, struct):
    """
    Verstraete–Pinty vegetation canopy BRDF (single value).

    Parameters
    ----------
    option : list of 5 ints   – model options
    angles : list of 3 floats – (theta_i, theta_v, phi) in radians
    optics : list of 3 floats – (omega, g, g2)
    struct : list of 4 floats – (LAD, r_sun, kappa1_or_chil, kappa2)

    Returns
    -------
    refl   : float – BRDF value; -1 on error
    error  : int   – error code (0 = OK)
    """
    pi    = 3.14159265
    pitwo = 6.28318531
    pio2me = 1.57
    coef1 = 0.42441318
    coef2 = 7.957747e-2
    error = 0

    # Validate options
    for i in range(5):
        if option[i] < 0 or option[i] > (1 if i != 2 else 2) and (i != 3 or option[i] > 2):
            option[i] = 1
            error = 101 + i

    # Validate angles
    if not (0.0 <= angles[0] <= pio2me): error = 201; return -1.0, error
    if not (0.0 <= angles[1] <= pio2me): error = 202; return -1.0, error
    if not (0.0 <= angles[2] <= pitwo):  error = 203; return -1.0, error
    if not (0.0 <= optics[0] <= 1.0):    error = 301; return -1.0, error

    mu1    = math.cos(angles[0])
    mu2    = math.cos(angles[1])
    cosphi = math.cos(angles[2])
    cospha = mu1 * mu2 + math.sin(angles[0]) * math.sin(angles[1]) * cosphi
    cospha = max(-1.0, min(1.0, cospha))
    phaang = math.acos(cospha)
    tante1 = math.tan(angles[0])
    tante2 = math.tan(angles[1])
    geofac = math.sqrt(tante1**2 + tante2**2 - 2.0 * tante1 * tante2 * cosphi)

    # Leaf orientation kappa
    if option[2] == 0:
        kappa1 = struct[2]; kappa2 = struct[3]
    elif option[2] == 1:
        psi1 = 0.5 - (0.6333 + 0.33 * struct[2]) * struct[2]
        psi2 = 0.877 * (1.0 - 2.0 * psi1)
        kappa1 = psi1 + psi2 * mu1; kappa2 = psi1 + psi2 * mu2
    else:
        psi1 = 0.5 - (0.489 + 0.11 * struct[2]) * struct[2]
        psi2 = 1.0 * (1.0 - 2.0 * psi1)
        kappa1 = psi1 + psi2 * mu1; kappa2 = psi1 + psi2 * mu2

    # Phase function
    if option[3] == 0:
        phafun = 1.0
    elif option[3] == 1:
        asyf2  = optics[1]**2
        x1     = 1.0 + asyf2 - 2.0 * optics[1] * math.cos(pi - phaang)
        phafun = (1.0 - asyf2) / (x1**1.5)
    else:
        phafun = 1.0 + optics[1] * cospha + optics[2] * (0.5 * (3.0 * cospha**2 - 1.0))

    # Hot spot (parameterised)
    kmkm   = kappa1 * mu2 + kappa2 * mu1
    gotwor = geofac / (2.0 * struct[1])
    vpg    = 4.0 * (1.0 - coef1) * mu2 * gotwor / (struct[0] * kappa2)
    pvg    = 1.0 + 1.0 / (1.0 + vpg)

    # Multiple scattering
    if option[4] == 0:
        mulsca = 0.0
    else:
        x1     = mu1 / kappa1; x2 = mu2 / kappa2
        sq     = math.sqrt(1.0 - optics[0])
        hfun1  = (1.0 + x1) / (1.0 + x1 * sq)
        hfun2  = (1.0 + x2) / (1.0 + x2 * sq)
        mulsca = hfun1 * hfun2 - 1.0

    # Reflectance
    if option[1] == 0:
        coef4 = coef2 * optics[0] * kappa1 * mu1 / kmkm
    else:
        coef4 = 0.25 * optics[0] * kappa1 / kmkm

    refl = coef4 * (pvg * phafun + mulsca)
    return refl, error


def versbrdf(option, optics, struct, mu, np_, rm, rp):
    """
    Verstraete–Pinty BRDF at all quadrature angle combinations.

    Returns
    -------
    brdfint : 2-D array, shape (2*mu+1, np_)
    """
    def ji(k): return k + mu
    brdfint = np.zeros((2 * mu + 1, np_))
    theta_i = math.acos(rm[ji(0)])
    for k in range(np_):
        for j in range(1, mu + 1):
            theta_v = math.acos(rm[ji(j)])
            fi      = rm[ji(-mu)] if j == mu else rp[k] + rm[ji(-mu)]
            angles  = [theta_i, theta_v, fi]
            val, _  = mvbp1(list(option), angles, list(optics), list(struct))
            brdfint[ji(j), k] = max(0.0, val)
    return brdfint


def versalbe(option, optics, struct):
    """Verstraete–Pinty hemispherical albedo via Gauss integration."""
    nta, nfa = 24, 48
    ta, wta, fa, wfa = _gauss_hemisphere(nta, nfa)
    brdfalb = summ = 0.0
    for k in range(nfa):
        for j in range(nta):
            for l in range(nta):
                mu2  = math.cos(ta[j]); mu1 = math.cos(ta[l])
                si2  = math.sin(ta[j]); si1 = math.sin(ta[l])
                fi   = fa[k]
                angles = [ta[l], ta[j], fi]
                val, err = mvbp1(list(option), angles, list(optics), list(struct))
                if err < 200 and val >= 0.0:
                    pond     = mu1 * mu2 * si1 * si2 * wfa[k] * wta[j] * wta[l]
                    brdfalb += val * pond
                    summ    += pond
    return brdfalb / summ if summ > 0.0 else 0.0
