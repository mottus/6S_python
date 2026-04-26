"""
iapi_brdf.py
------------
Translated from IAPIBRDF.f, IAPIALBE.f, IAPITOOLS.f

Iaquinta and Pinty canopy radiative transfer BRDF model (ibrdf=7).
Reference: Iaquinta J. and Pinty B. (courtesy Jean Iaquinta).

Supports 5 leaf angle distributions (LAD):
    1 = planophile
    2 = erectophile
    3 = plagiophile
    4 = extremophile
    5 = uniform

Parameters:
    pild : int   – leaf angle distribution (1–5)
    pxLt : float – leaf area index [1–15]
    pRl  : float – leaf reflectance [0–0.99]
    pTl  : float – leaf transmittance [0–0.99]  (Rl+Tl < 0.99)
    pRs  : float – soil albedo [0–0.99]
    pihs : int   – hot-spot (0=off, 1=on)
    pc   : float – hot-spot parameter 2*r*Lambda [0–2]
"""

import math
import numpy as np

from .gauss import gauss

_PI = math.acos(-1.0)


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature (internal, matches Fortran's gauleg)
# ---------------------------------------------------------------------------
def _gauleg(x1, x2, n):
    """Gauss-Legendre quadrature points and weights."""
    eps  = 3e-14
    m    = (n + 1) // 2
    xm   = 0.5 * (x2 + x1)
    xl   = 0.5 * (x2 - x1)
    x    = [0.0] * n
    w    = [0.0] * n
    for i in range(1, m + 1):
        z = math.cos(_PI * (i - 0.25) / (n + 0.5))
        while True:
            p1 = 1.0; p2 = 0.0
            for j in range(1, n + 1):
                p3 = p2; p2 = p1
                p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z; z = z1 - p1 / pp
            if abs(z - z1) <= eps:
                break
        x[i - 1]     = xm - xl * z
        x[n - i]     = xm + xl * z
        w[i - 1]     = 2.0 * xl / ((1.0 - z * z) * pp * pp)
        w[n - i]     = w[i - 1]
    return x, w


# ---------------------------------------------------------------------------
# Leaf angle distribution
# ---------------------------------------------------------------------------
def _lad(ild):
    """Return (a, b, c, d) coefficients for gl(Theta) = a + b*cos(2T) + c*cos(4T) + d*sin(T)."""
    pi = _PI
    if   ild == 1: return  2/pi,  2/pi,  0.0,  0.0   # planophile
    elif ild == 2: return  2/pi, -2/pi,  0.0,  0.0   # erectophile
    elif ild == 3: return  2/pi,  0.0,  -2/pi, 0.0   # plagiophile
    elif ild == 4: return  2/pi,  0.0,   2/pi, 0.0   # extremophile
    else:          return  0.0,   0.0,   0.0,  1.0   # uniform


def _gl(Theta, ild):
    """Leaf angle density function."""
    a, b, c, d = _lad(ild)
    return a + b * math.cos(2 * Theta) + c * math.cos(4 * Theta) + d * math.sin(Theta)


# ---------------------------------------------------------------------------
# G-function (mean projection of leaf area)
# ---------------------------------------------------------------------------
def _G_f(Theta, ild, xgm, wgm, n):
    """G function: projected leaf area in direction Theta."""
    pi  = _PI
    xmm = 0.5 * pi / 2.0
    xrm = 0.5 * pi / 2.0
    val = 0.0
    for j in range(n):
        xt  = xmm + xrm * xgm[j]
        val += wgm[j] * xrm * _Psi(Theta, xt) * _gl(xt, ild)
    return val


def _Psi(Theta, xt):
    """Element of the G function."""
    pi  = _PI
    xmu = math.cos(xt); smu = math.sin(xt)
    if abs(xmu - 1.0) < 1e-10:
        return abs(math.cos(Theta))
    if abs(math.sin(Theta)) < 1e-10:
        return xmu
    if abs(smu) < 1e-10:
        cpt = 0.0
    else:
        cpt = xmu / smu * math.cos(Theta) / math.sin(Theta)
    if abs(cpt) > 1.0:
        return abs(xmu * math.cos(Theta))
    pt  = math.acos(-cpt)
    return abs(xmu * math.cos(Theta) * (2 / pi * pt - 1.0)
               + 2 / pi * smu * math.sin(Theta) * math.sin(pt))


# ---------------------------------------------------------------------------
# Geo function (geometric factor for hot-spot)
# ---------------------------------------------------------------------------
def _Geo(Ti, Pi_, Te, Pe):
    val = math.sqrt(abs(math.tan(Ti)**2 + math.tan(Te)**2
                        - 2 * math.tan(Ti) * math.tan(Te) * math.cos(Pi_ - Pe)))
    return max(val, 1e-35)


# ---------------------------------------------------------------------------
# H function (hot-spot)
# ---------------------------------------------------------------------------
def _h(xL, xLi):
    pi = _PI
    if xL < xLi:
        return (1.0 - 4.0 / (3.0 * pi)) / xLi * xL
    else:
        return 1.0 - 4.0 / (3.0 * pi) * xLi / xL


# ---------------------------------------------------------------------------
# Gamma_f: bi-Lambertian scattering function
# ---------------------------------------------------------------------------
def _Gamma_f(Theta_p, Phi_p, Theta, Phi, xLt, Rl, Tl, ild, xgm, wgm, n):
    pi  = _PI
    xmm = 0.5 * pi / 2.0; xrm = 0.5 * pi / 2.0
    ymm = 0.5 * 2 * pi;   yrm = 0.5 * 2 * pi
    total = 0.0
    for j in range(n):
        xt  = xmm + xrm * xgm[j]
        s   = 0.0
        for i in range(n):
            yt  = ymm + yrm * xgm[i]
            dpp = (math.cos(Theta_p) * math.cos(xt)
                   + math.sin(Theta_p) * math.sin(xt) * math.cos(Phi_p - yt))
            dp  = (math.cos(Theta) * math.cos(xt)
                   + math.sin(Theta) * math.sin(xt) * math.cos(Phi - yt))
            f = Rl * abs(dp) / pi if dp * dpp < 0.0 else Tl * abs(dp) / pi
            s += wgm[i] * yrm * _gl(xt, ild) * f * abs(dpp)
        total += wgm[j] * xrm * s
    return total / 2.0


# ---------------------------------------------------------------------------
# Solve: multiple-scattering contribution (Ro_mult)
# ---------------------------------------------------------------------------
def _solve(Theta_i, xLt, Rl, Tl, Rs, xgm, wgm, n):
    """Compute Ro_mult for given illumination angle."""
    pi = _PI; m = 20
    dL = xLt / m
    xmm = 0.0; xrm = 1.0   # integration from -1 to 1
    xtmu = 1e-5
    xmui = max(abs(math.cos(Theta_i)), xtmu)
    G_list = []
    for j in range(n):
        xmu = xmm + xrm * xgm[j]
        G_list.append(_G_f(math.acos(xmu), 0, xgm, wgm, n))
    Gi = _G_f(Theta_i, 0, xgm, wgm, n)

    # Down-welling 0th-order source
    Q0d = []
    for k in range(1, m + 1):
        xL  = (k - 0.5) * dL
        xdb = Gi / xmui * xL
        val = (Rl + Tl) / 2.0 * Gi * math.exp(-min(xdb, 20.0)) if xdb < 20.0 else 0.0
        Q0d.append(val)

    # Boundary
    xdb = Gi / xmui * xLt
    xI0t = 2 * Rs * xmui * math.exp(-min(xdb, 20.0)) if xdb < 20.0 else 0.0

    # Up-welling 0th-order source
    Q0u = []
    for k in range(1, m + 1):
        xL  = (k - 0.5) * dL
        s   = 0.0
        for j in range(n // 2, n):
            xmu = xmm + xrm * xgm[j]
            xdb = G_list[j] / xmu * (xLt - xL) if xmu > 0 else 50.0
            if xdb < 20.0:
                s += wgm[j] * xrm * xI0t * (Rl + Tl) / 2.0 * G_list[j] * math.exp(-xdb)
        Q0u.append(s)

    # Single scattering (xI): shape (m+2, n)
    xI = [[0.0] * n for _ in range(m + 2)]
    for k in range(m):
        for j in range(n // 2):
            xmu = xmm + xrm * xgm[j]
            denom = G_list[j] / 2.0 - xmu / dL
            if abs(denom) < 1e-30: continue
            xI[k + 1][j] = (Q0d[k] - xI[k][j] * (G_list[j] / 2.0 + xmu / dL)) / denom

    xI1t = 0.0
    for j in range(n // 2):
        xmu = xmm + xrm * xgm[j]
        xI1t += wgm[j] * xrm * 2 * Rs * abs(xmu) * xI[m][j]
    for j in range(n // 2, n):
        xI[m][j] = 0.0

    for k in range(m - 1, -1, -1):
        for j in range(n // 2, n):
            xmu = xmm + xrm * xgm[j]
            denom = G_list[j] / 2.0 + xmu / dL
            if abs(denom) < 1e-30: continue
            xI[k][j] = (Q0d[k] - xI[k + 1][j] * (G_list[j] / 2.0 - xmu / dL)) / denom

    Q1 = []
    for k in range(m):
        s = 0.0
        for j in range(n):
            s += wgm[j] * xrm * (Rl + Tl) / 2.0 * G_list[j] * (xI[k + 1][j] + xI[k][j]) / 2.0
        Q1.append(s)

    # Multiple scattering iterations
    S    = [0.0] * m
    xIf  = [[0.0] * n for _ in range(m + 2)]
    eps_ = 1e-4

    for _ in range(50):
        for k in range(m):
            for j in range(n // 2):
                xmu   = xmm + xrm * xgm[j]
                denom = G_list[j] / 2.0 - xmu / dL
                if abs(denom) < 1e-30: continue
                xI[k + 1][j] = (S[k] + Q0u[k] + Q1[k]
                                 - xI[k][j] * (G_list[j] / 2.0 + xmu / dL)) / denom

        xImt = 0.0
        for j in range(n // 2):
            xmu = xmm + xrm * xgm[j]
            xImt += wgm[j] * xrm * 2 * Rs * abs(xmu) * xI[m][j]
        for j in range(n // 2, n):
            xI[m][j] = xImt + xI1t

        for k in range(m - 1, -1, -1):
            for j in range(n // 2, n):
                xmu   = xmm + xrm * xgm[j]
                denom = G_list[j] / 2.0 + xmu / dL
                if abs(denom) < 1e-30: continue
                xI[k][j] = (S[k] + Q0u[k] + Q1[k]
                             - xI[k + 1][j] * (G_list[j] / 2.0 - xmu / dL)) / denom

        nc = sum(1 for k in range(m + 1) for j in range(n)
                 if abs(xIf[k][j] - xI[k][j]) < eps_)
        for k in range(m + 1):
            xIf[k][:] = xI[k][:]

        if nc == (m + 1) * n:
            break

        for k in range(m):
            s = 0.0
            for j in range(n):
                s += wgm[j] * xrm * (Rl + Tl) / 2.0 * G_list[j] * (xI[k + 1][j] + xI[k][j]) / 2.0
            S[k] = s

    # Ro_mult: upward diffuse at canopy top
    Ro_mult = 0.0
    for j in range(n // 2, n):
        xmu = xmm + xrm * xgm[j]
        Ro_mult += wgm[j] * xrm * xI[0][j] * xmu / xmui

    return Ro_mult


# ---------------------------------------------------------------------------
# Ro_1: first-order BRDF
# ---------------------------------------------------------------------------
def _Ro_1(Theta_i, Phi_i, Theta_e, Phi_e, xLt, Rl, Tl, Rs, c, ild, xgm, wgm, n):
    xtmu = 1e-5
    xmui = abs(math.cos(Theta_i))
    xmu  = max(math.cos(Theta_e), xtmu)
    Gi   = _G_f(Theta_i, ild, xgm, wgm, n)
    Ge   = _G_f(Theta_e, ild, xgm, wgm, n)
    Ki   = Gi / xmui
    Ke   = Ge / xmu
    xLi  = c / _Geo(Theta_i, Phi_i, Theta_e, Phi_e)

    xmm = 0.5 * xLt; xrm = 0.5 * xLt
    Ro_1_c = 0.0
    for j in range(n):
        xL  = xmm + xrm * xgm[j]
        xdb = (Ki + Ke * _h(xL, xLi)) * xL
        if xdb < 20.0:
            Ro_1_c += wgm[j] * xrm * math.exp(-xdb)
    Ro_1_c *= _Gamma_f(Theta_i, Phi_i, Theta_e, Phi_e, xLt, Rl, Tl, ild, xgm, wgm, n) / xmui / xmu

    xdb   = (Ki + Ke * _h(xLt, xLi)) * xLt
    Ro_1_s = Rs * math.exp(-min(xdb, 20.0)) if xdb < 20.0 else 0.0

    return Ro_1_c + Ro_1_s


# ---------------------------------------------------------------------------
# Public interface: IAPIBRDF and IAPIALBE
# ---------------------------------------------------------------------------

def iapibrdf(pild, pxLt, pRl, pTl, pRs, pihs, pc, mu, np_, rm, rp):
    """
    Iaquinta-Pinty canopy BRDF at Gauss quadrature angles.

    Parameters
    ----------
    pild : int   – leaf angle distribution (1–5)
    pxLt : float – leaf area index
    pRl  : float – leaf reflectance
    pTl  : float – leaf transmittance
    pRs  : float – soil albedo
    pihs : int   – hot-spot flag (0=off, 1=on)
    pc   : float – hot-spot parameter
    mu, np_ : int – Gauss angles / azimuth points
    rm, rp  : arrays

    Returns
    -------
    brdfint : 2-D array, shape (2*mu+1, np_)
    """
    # Validate
    pxLt = max(0.01, min(15.0, pxLt))
    pRl  = max(0.0,  min(0.99, pRl))
    pTl  = max(0.0,  min(0.99 - pRl, pTl))
    pRs  = max(0.0,  min(0.99, pRs))
    pc   = max(0.0,  min(2.0,  pc)) if pihs == 1 else 1e-20

    def ji(k): return k + mu
    brdfint = np.zeros((2 * mu + 1, np_))

    n       = 10
    xgm, wgm = _gauleg(-1.0, 1.0, n)

    mu1    = -rm[ji(0)]           # cos(solar zenith) [rm(0) = -xmus]
    Theta_i = math.acos(max(-1.0, min(1.0, mu1)))
    Theta_i = _PI - Theta_i

    Ro_mult = _solve(Theta_i, pxLt, pRl, pTl, pRs, xgm, wgm, n)

    for k in range(np_):
        for j in range(1, mu + 1):
            mu2     = rm[ji(j)]
            Theta_e = math.acos(max(-1.0, min(1.0, mu2)))
            fi      = rm[ji(-mu)] if j == mu else rp[k] + rm[ji(-mu)]
            while fi < 0.0:      fi += 2.0 * _PI
            while fi > 2.0 * _PI: fi -= 2.0 * _PI
            Phi_i = fi; Phi_v = 0.0
            y = _Ro_1(Theta_i, Phi_i, Theta_e, Phi_v,
                      pxLt, pRl, pTl, pRs, pc, pild, xgm, wgm, n) + Ro_mult
            brdfint[ji(j), k] = max(0.0, y)

    return brdfint


def iapialbe(pild, pxLt, pRl, pTl, pRs, pihs, pc):
    """
    Iaquinta-Pinty canopy hemispherical albedo.

    Parameters
    ----------
    (same as iapibrdf minus mu, np_, rm, rp)

    Returns
    -------
    brdfalb : float – hemispherical albedo
    """
    pi   = _PI
    pxLt = max(0.01, min(15.0, pxLt))
    pRl  = max(0.0,  min(0.99, pRl))
    pTl  = max(0.0,  min(0.99 - pRl, pTl))
    pRs  = max(0.0,  min(0.99, pRs))
    pc   = max(0.0,  min(2.0,  pc)) if pihs == 1 else 1e-20

    nta = 24; nfa = 48
    ta, wta = gauss(0.0, pi / 2.0, nta)
    fa, wfa = gauss(0.0, 2.0 * pi, nfa)

    n       = 10
    xgm, wgm = _gauleg(-1.0, 1.0, n)

    brdfalb = summ = 0.0

    for l in range(nta):
        mu1     = math.cos(ta[l]); si1 = math.sin(ta[l])
        Theta_i = math.acos(mu1)
        Theta_i = pi - Theta_i
        Ro_mult = _solve(Theta_i, pxLt, pRl, pTl, pRs, xgm, wgm, n)

        for k in range(nfa):
            for j in range(nta):
                mu2     = math.cos(ta[j]); si2 = math.sin(ta[j])
                Theta_e = math.acos(mu2)
                fi      = fa[k]
                while fi < 0.0:      fi += 2.0 * pi
                while fi > 2.0 * pi: fi -= 2.0 * pi
                Phi_i = fi; Phi_v = 0.0
                y = (_Ro_1(Theta_i, Phi_i, Theta_e, Phi_v,
                           pxLt, pRl, pTl, pRs, pc, pild, xgm, wgm, n)
                     + Ro_mult)
                pond       = mu1 * mu2 * si1 * si2 * wfa[k] * wta[j] * wta[l]
                brdfalb   += pond * y
                summ       += pond

    return brdfalb / summ if summ > 0 else 0.0
