"""
os_module.py
------------
Translated from OS.f

Full successive-orders-of-scattering radiative transfer (OS subroutine).
Runs for Fourier azimuth harmonic is=0 (isotropic component).
For nadir viewing (xmuv=1), this is exact since higher harmonics have zero contribution.
"""

import math
import numpy as np

from .commons    import delta_sigma, trunc, ier as ier_common
from .scattering import discre
from .kernel     import kernel_func


def os_sos(tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
           phirad, nt, mu, np_, rm, gb, rp, xl):
    """
    Successive Orders of Scattering — fills xl(-mu:mu, np) with
    the radiance field.

    For satellite nadir view (xmuv≈1), the is=0 Fourier harmonic dominates.
    """
    delta  = delta_sigma.delta
    _ier   = ier_common

    snt  = nt
    accu = 1.0e-20
    accu2= 1.0e-3
    hr   = 8.0

    ta  = tamoy
    tr  = trmoy
    piz = pizmoy

    if palt <= 900.0 and palt > 0.0:
        ha  = 2.0
        ntp = nt - 1
    else:
        ha  = 2.0
        ntp = nt

    h    = np.zeros(nt + 2)
    xdel = np.zeros(nt + 2)
    ydel = np.zeros(nt + 2)

    if ta <= accu2 and tr > ta:
        for j in range(ntp + 1):
            h[j]    = j * tr / ntp if ntp else 0.0
            ydel[j] = 1.0; xdel[j] = 0.0
    elif tr <= accu2 and ta > tr:
        for j in range(ntp + 1):
            h[j]    = j * ta / ntp if ntp else 0.0
            ydel[j] = 0.0; xdel[j] = piz
    else:
        h[0] = 0.0; ydel[0] = 1.0; xdel[0] = 0.0
        for it in range(1, ntp + 1):
            zx = discre(ta, ha, tr, hr, it, ntp, h[it-1], ydel[it-1], 300.0, 0.0)
            if _ier.ier: return
            xxx = -zx / ha
            ca  = ta * math.exp(max(-87.0, xxx))
            cr  = tr * math.exp(-zx / hr)
            h[it] = cr + ca
            cr2 = cr / hr; ca2 = ca / ha
            denom = cr2 + ca2
            ratio = cr2 / denom if denom > 0 else 0.5
            xdel[it] = (1.0 - ratio) * piz
            ydel[it] = ratio

    # Insert plane layer
    ipl = 0
    if ntp == nt - 1:
        taup = (tamoy - tamoyp) + (trmoy - trmoyp)
        ipl  = -1
        for i in range(ntp + 1):
            if taup >= h[i]: ipl = i
        if ipl < 0: ipl = 0
        xt1 = abs(h[ipl] - taup)
        xt2 = abs(h[min(ipl+1, nt)] - taup)
        th  = 0.005
        if xt1 > th and xt2 > th:
            for i in range(nt, ipl, -1):
                xdel[i] = xdel[i-1]; ydel[i] = ydel[i-1]; h[i] = h[i-1]
        elif xt2 < xt1:
            ipl += 1
        h[ipl] = taup
        if tr > accu2 and ta > accu2:
            ca = ta * math.exp(max(-87.0, -palt/ha))
            cr = tr * math.exp(-palt/hr)
            cr2=cr/hr; ca2=ca/ha; denom=cr2+ca2
            ratio=cr2/denom if denom else 0.5
            xdel[ipl]=(1-ratio)*piz; ydel[ipl]=ratio
        elif tr > accu2:
            ydel[ipl]=1.0; xdel[ipl]=0.0
        else:
            ydel[ipl]=0.0; xdel[ipl]=piz

    aaaa  = delta / (2.0 - delta)
    ron   = (1.0 - aaaa) / (1.0 + 2.0 * aaaa)
    beta0 = 1.0; beta2 = 0.5 * ron

    size_j = 2 * mu + 1
    def ji(k): return k + mu

    xl[:] = 0.0
    xpl, psl, bp = kernel_func(0, mu, rm)

    i1   = np.zeros((nt + 1, size_j))
    i2   = np.zeros((nt + 1, size_j))
    i3   = np.zeros(size_j)
    i4   = np.zeros(size_j)
    inm1 = np.zeros(size_j)
    inm2 = np.zeros(size_j)
    _in  = np.zeros(size_j)

    roavion0 = roavion1 = roavion2 = roavion = 0.0

    # Primary upward radiation
    for k in range(1, mu + 1):
        yy = rm[ji(k)]
        if abs(yy) < 1e-10: continue
        i1[nt, ji(k)] = 1.0
        for i in range(nt - 1, -1, -1):
            i1[i, ji(k)] = math.exp(max(-87.0, -(ta + tr - h[i]) / yy))

    # Primary downward = 0
    for k in range(-mu, 0):
        i1[:, ji(k)] = 0.0

    # Initialise
    for k in range(-mu, mu + 1):
        idx = nt if k < 0 else 0
        inm1[ji(k)] = i1[idx, ji(k)]
        inm2[ji(k)] = i1[idx, ji(k)]
        i3[ji(k)]   = i1[idx, ji(k)]

    roavion2 = i1[ipl, ji(mu)]
    roavion  = i1[ipl, ji(mu)]
    ig = 1

    while True:
        ig += 1
        # Source function
        for k in range(1, mu + 1):
            xpk = xpl[ji(k)]; ypk = xpl[ji(-k)]
            for i in range(nt + 1):
                ii1 = ii2 = 0.0
                x = xdel[i]; y = ydel[i]
                for j in range(1, mu + 1):
                    xpj = xpl[ji(j)]; z = gb[ji(j)]
                    xi1 = i1[i, ji(j)]; xi2 = i1[i, ji(-j)]
                    bpjk  = bp[j, ji(k)]  * x + y * (beta0 + beta2 * xpj * xpk)
                    bpjmk = bp[j, ji(-k)] * x + y * (beta0 + beta2 * xpj * ypk)
                    ii2 += z * (xi1 * bpjk  + xi2 * bpjmk)
                    ii1 += z * (xi1 * bpjmk + xi2 * bpjk)
                i2[i, ji(k)] = ii2; i2[i, ji(-k)] = ii1

        # Upward integration
        for k in range(1, mu + 1):
            yy = rm[ji(k)]
            if abs(yy) < 1e-10: continue
            i1[nt, ji(k)] = 0.0; zi1 = 0.0
            for i in range(nt - 1, -1, -1):
                jj = i + 1; f = h[jj] - h[i]
                a = (i2[jj, ji(k)] - i2[i, ji(k)]) / f if f != 0 else 0.0
                b = i2[i, ji(k)] - a * h[i]
                c = math.exp(max(-87.0, -f / yy)); d = 1.0 - c; xx = h[i] - h[jj] * c
                zi1 = c * zi1 + (d * (b + a * yy) + a * xx) * 0.5
                i1[i, ji(k)] = zi1

        # Downward integration
        for k in range(-mu, 0):
            yy = rm[ji(k)]
            if abs(yy) < 1e-10: continue
            i1[0, ji(k)] = 0.0; zi1 = 0.0
            for i in range(1, nt + 1):
                jj = i - 1; f = h[i] - h[jj]
                a = (i2[i, ji(k)] - i2[jj, ji(k)]) / f if f != 0 else 0.0
                b = i2[i, ji(k)] - a * h[i]
                c = math.exp(min(87.0, f / yy)); d = 1.0 - c; xx = h[i] - h[jj] * c
                zi1 = c * zi1 + (d * (b + a * yy) + a * xx) * 0.5
                i1[i, ji(k)] = zi1

        for k in range(-mu, mu + 1):
            idx = nt if k < 0 else 0
            _in[ji(k)] = i1[idx, ji(k)]
        roavion0 = i1[ipl, ji(mu)]

        # Convergence test (geometric series)
        if ig > 2:
            z = 0.0
            a1 = roavion2; d1 = roavion1; g1 = roavion0
            if a1 >= accu and d1 >= accu and roavion >= accu:
                denom = 1.0 - g1 / d1
                if abs(denom) > accu:
                    z = max(z, abs(((g1/d1 - d1/a1) / (denom**2)) * (g1/roavion)))
            for l in range(-mu, mu + 1):
                if l == 0: continue
                a1 = inm2[ji(l)]; d1 = inm1[ji(l)]; g1 = _in[ji(l)]
                if a1 == 0.0 or d1 == 0.0 or i3[ji(l)] == 0.0: continue
                denom = 1.0 - g1 / d1
                if abs(denom) > accu:
                    z = max(z, abs(((g1/d1 - d1/a1) / (denom**2)) * (g1/i3[ji(l)])))
            if z < 0.0001:
                for l in range(-mu, mu + 1):
                    if l == 0: continue
                    d1 = inm1[ji(l)]; g1 = _in[ji(l)]
                    if d1 == 0.0: continue
                    y1 = 1.0 - g1/d1
                    if abs(y1) > accu:
                        i3[ji(l)] += g1 / y1
                d1 = roavion1; g1 = roavion0
                if d1 >= accu:
                    y1 = 1.0 - g1/d1
                    roavion += g1 / y1 if abs(y1) > accu else g1
                else:
                    roavion += g1
                break
            inm2[:] = inm1; roavion2 = roavion1

        inm1[:] = _in; roavion1 = roavion0
        i3 += _in; i4 += _in; roavion += roavion0

        z = 0.0
        for l in range(-mu, mu + 1):
            if i4[ji(l)] != 0.0:
                z = max(z, abs(_in[ji(l)] / i4[ji(l)]))
        if z < 0.001 or ig >= 20:
            break

    # Output assignment
    # For satellite (palt>900): xl(-mu,1) = i3[ji(-mu)] = accumulated TOA reflectance
    # For plane/ground: xl(-mu,1) = roavion (upward at plane level)
    if palt > 900.0:
        xl[ji(-mu), 0] = i3[ji(-mu)]
    else:
        xl[ji(-mu), 0] = roavion
    xl[ji(mu), 0]  = i3[ji(mu)]
    for k in range(1, mu):
        xl[ji(0), 0] += rm[ji(k)] * gb[ji(k)] * i3[ji(-k)]

    mum1 = mu - 1
    for k in range(-mum1, mum1 + 1):
        for p in range(np_):
            xl[ji(k), p] = i3[ji(k)]

    nt = snt
