"""
os_module.py
------------
Successive Orders of Scattering (OS subroutine), faithfully translated
from OS.f (6SV1.1 / 6S Fortran 77 source).

Key fix (2025): the original Python translation was missing:
  - xmus = -rm[ji(0)]  (solar zenith cosine extracted from the rm array)
  - ch[] array:  ch[j] = exp(-h[j]/xmus) / 2   (direct-beam attenuation per layer)
  - The primary source function used ch[k] as the multiplicative factor:
      i2[k, j] = ch[k] * (sa2*xdel[k] + sa1*ydel[k])
  - The Fourier azimuth decomposition loop (is = 0..iborm);
    the old code only computed is=0 which is exact only for nadir viewing.
Without ch[], the source function was always zero → the SOS converged to a
non-physical solution giving path reflectance 1.6x too high in the blue.
"""

import math
import numpy as np

from .commons    import delta_sigma, trunc, ier as ier_common
from .scattering import discre
from .kernel     import kernel_func


def os_sos(tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt,
           phirad, nt, mu, np_, rm, gb, rp, xl):
    """
    Successive Orders of Scattering — fills xl(-mu:mu, np) with the
    upward radiance field (Fourier sum over azimuth harmonics).

    Parameters match the Fortran OS subroutine exactly.
    """
    delta  = delta_sigma.delta
    _ier   = ier_common

    snt  = nt
    accu  = 1.0e-20
    accu2 = 1.0e-3
    hr    = 8.0

    ta  = tamoy
    tr  = trmoy
    piz = pizmoy

    # Optical depth above plane (aircraft) vs satellite
    if palt <= 900.0 and palt > 0.0:
        ha  = 2.0
        ntp = nt - 1
    else:
        ha  = 2.0
        ntp = nt

    # ── Extract solar zenith cosine from the rm array ────────────────────────
    # In the Fortran: xmus = -rm(0)   [rm(0) = -cos(SZA) by convention]
    def ji(k): return k + mu
    xmus = -rm[ji(0)]

    # ── Allocate layer arrays ────────────────────────────────────────────────
    h    = np.zeros(nt + 2)
    ch   = np.zeros(nt + 2)   # direct solar beam attenuation per layer
    xdel = np.zeros(nt + 2)
    ydel = np.zeros(nt + 2)

    # ── Build the optical depth / layer arrays (3 cases) ────────────────────
    # Case 1: pure Rayleigh (ta ≈ 0)
    if ta <= accu2 and tr > ta:
        for j in range(ntp + 1):
            h[j]    = j * tr / ntp if ntp else 0.0
            ch[j]   = math.exp(max(-87.0, -h[j] / xmus)) / 2.0
            ydel[j] = 1.0
            xdel[j] = 0.0

    # Case 2: pure aerosol (tr ≈ 0)
    elif tr <= accu2 and ta > tr:
        for j in range(ntp + 1):
            h[j]    = j * ta / ntp if ntp else 0.0
            ch[j]   = math.exp(max(-87.0, -h[j] / xmus)) / 2.0
            ydel[j] = 0.0
            xdel[j] = piz

    # Case 3: mixed Rayleigh + aerosol
    else:
        h[0]    = 0.0
        ch[0]   = 0.5          # exp(0/xmus)/2 = 0.5
        ydel[0] = 1.0
        xdel[0] = 0.0
        for it in range(1, ntp + 1):
            yy = h[it-1]
            dd = ydel[it-1]
            zx = discre(ta, ha, tr, hr, it, ntp, yy, dd, 300.0, 0.0)
            if _ier.ier: return
            ca  = ta * math.exp(max(-87.0, -zx / ha))
            cr  = tr * math.exp(max(-87.0, -zx / hr))
            h[it]  = cr + ca
            ch[it] = math.exp(max(-87.0, -h[it] / xmus)) / 2.0
            cr2 = cr / hr; ca2 = ca / ha
            denom = cr2 + ca2
            ratio = cr2 / denom if denom > 0 else 0.5
            xdel[it] = (1.0 - ratio) * piz
            ydel[it] = ratio

    # ── Insert plane layer (aircraft sensor) ─────────────────────────────────
    ipl = 0
    if ntp == nt - 1:
        taup = (tamoy - tamoyp) + (trmoy - trmoyp)
        ipl  = -1
        for i in range(ntp + 1):
            if taup >= h[i]: ipl = i
        if ipl < 0: ipl = 0
        xt1 = abs(h[ipl]           - taup)
        xt2 = abs(h[min(ipl+1,nt)] - taup)
        th  = 0.0005
        if xt1 > th and xt2 > th:
            for i in range(nt, ipl, -1):
                xdel[i] = xdel[i-1]
                ydel[i] = ydel[i-1]
                h[i]    = h[i-1]
                ch[i]   = ch[i-1]
        elif xt2 < xt1:
            ipl += 1
        h[ipl] = taup
        if tr > accu2 and ta > accu2:
            ca = ta * math.exp(max(-87.0, -palt / ha))
            cr = tr * math.exp(max(-87.0, -palt / hr))
            h[ipl]  = ca + cr
            cr2=cr/hr; ca2=ca/ha; denom=cr2+ca2
            ratio = cr2/denom if denom else 0.5
            xdel[ipl] = (1.0-ratio)*piz; ydel[ipl] = ratio
            ch[ipl] = math.exp(max(-87.0, -h[ipl]/xmus)) / 2.0
        elif tr > accu2:
            ydel[ipl]=1.0; xdel[ipl]=0.0
        else:
            ydel[ipl]=0.0; xdel[ipl]=piz

    # ── Rayleigh phase function parameters ───────────────────────────────────
    aaaa  = delta / (2.0 - delta)
    ron   = (1.0 - aaaa) / (1.0 + 2.0 * aaaa)
    beta0 = 1.0
    beta2 = 0.5 * ron

    size_j = 2 * mu + 1
    mum1   = mu - 1
    pi     = math.acos(-1.0)

    xl[:] = 0.0

    i2   = np.zeros((nt + 1, size_j))
    i3   = np.zeros(size_j)
    i4   = np.zeros(size_j)
    inm1 = np.zeros(size_j)
    inm2 = np.zeros(size_j)
    _in  = np.zeros(size_j)

    roavion0 = roavion1 = roavion2 = roavion = 0.0

    # Number of Fourier azimuth harmonics
    # For xmus ≈ 1 (nadir sun) only is=0 contributes
    iborm = 0 if abs(xmus - 1.0) < 1.0e-6 else 80

    # ── Fourier azimuth decomposition loop ───────────────────────────────────
    for is_ in range(iborm + 1):

        ig = 1
        roavion0 = roavion1 = roavion2 = roavion = 0.0
        i3[:] = 0.0

        beta0 = 0.0 if is_ > 0 else 1.0

        # kernel for this harmonic
        xpl, psl, bp = kernel_func(is_, mu, rm)

        # ── Primary scattering source function i2[k, j] = ch[k]*(sa2*b+sa1*a) ─
        for j in range(-mu, mu + 1):
            if is_ <= 2:
                spl = xpl[ji(0)]
                sa1 = beta0 + beta2 * xpl[ji(j)] * spl
                sa2 = bp[0, ji(j)]
            else:
                sa2 = bp[0, ji(j)]
                sa1 = 0.0

            for k in range(ntp + 1):
                c = ch[k]
                a = ydel[k]
                b = xdel[k]
                i2[k, ji(j)] = c * (sa2 * b + sa1 * a)

        # ── Vertical integration of primary scattering → i1 ─────────────────
        i1 = np.zeros((nt + 1, size_j))

        # Upward (k > 0)
        for k in range(1, mu + 1):
            yy = rm[ji(k)]
            if abs(yy) < 1.0e-10: continue
            i1[nt, ji(k)] = 0.0
            zi1 = 0.0
            for i in range(nt - 1, -1, -1):
                jj = i + 1
                f  = h[jj] - h[i]
                if f == 0.0: continue
                a  = (i2[jj, ji(k)] - i2[i, ji(k)]) / f
                b  = i2[i,  ji(k)]  - a * h[i]
                c  = math.exp(max(-87.0, -f / yy))
                d  = 1.0 - c
                xx = h[i] - h[jj] * c
                zi1 = c*zi1 + (d*(b + a*yy) + a*xx) * 0.5
                i1[i, ji(k)] = zi1

        # Downward (k < 0)
        for k in range(-mu, 0):
            yy = rm[ji(k)]
            if abs(yy) < 1.0e-10: continue
            i1[0, ji(k)] = 0.0
            zi1 = 0.0
            for i in range(1, nt + 1):
                jj = i - 1
                f  = h[i] - h[jj]
                if f == 0.0: continue
                c  = math.exp(max(-87.0, f / yy))
                d  = 1.0 - c
                a  = (i2[i, ji(k)] - i2[jj, ji(k)]) / f
                b  = i2[i, ji(k)]  - a * h[i]
                xx = h[i] - h[jj] * c
                zi1 = c*zi1 + (d*(b + a*yy) + a*xx) * 0.5
                i1[i, ji(k)] = zi1

        # Initialise accumulators from first-order i1
        for k in range(-mu, mu + 1):
            idx = nt if k < 0 else 0
            inm1[ji(k)] = i1[idx, ji(k)]
            inm2[ji(k)] = i1[idx, ji(k)]
            i3[ji(k)]   = i1[idx, ji(k)]

        roavion2 = i1[ipl, ji(mu)]
        roavion  = i1[ipl, ji(mu)]

        # ── Successive orders loop ───────────────────────────────────────────
        while True:
            ig += 1

            # Multiple-scattering source function
            if is_ <= 2:
                for k in range(1, mu + 1):
                    xpk = xpl[ji(k)]; ypk = xpl[ji(-k)]
                    for i in range(nt + 1):
                        ii1 = ii2 = 0.0
                        x = xdel[i]; y = ydel[i]
                        for j in range(1, mu + 1):
                            xpj = xpl[ji(j)]; z = gb[ji(j)]
                            xi1 = i1[i, ji(j)]; xi2 = i1[i, ji(-j)]
                            bpjk  = bp[j, ji(k)]  * x + y*(beta0 + beta2*xpj*xpk)
                            bpjmk = bp[j, ji(-k)] * x + y*(beta0 + beta2*xpj*ypk)
                            ii2 += z * (xi1*bpjk  + xi2*bpjmk)
                            ii1 += z * (xi1*bpjmk + xi2*bpjk)
                        if ii2 < 1.0e-30: ii2 = 0.0
                        if ii1 < 1.0e-30: ii1 = 0.0
                        i2[i, ji(k)]  = ii2
                        i2[i, ji(-k)] = ii1
            else:
                for k in range(1, mu + 1):
                    for i in range(nt + 1):
                        ii1 = ii2 = 0.0
                        x = xdel[i]
                        for j in range(1, mu + 1):
                            z  = gb[ji(j)]
                            xi1 = i1[i, ji(j)]; xi2 = i1[i, ji(-j)]
                            bpjk  = bp[j, ji(k)]  * x
                            bpjmk = bp[j, ji(-k)] * x
                            ii2 += z * (xi1*bpjk  + xi2*bpjmk)
                            ii1 += z * (xi1*bpjmk + xi2*bpjk)
                        if ii2 < 1.0e-30: ii2 = 0.0
                        if ii1 < 1.0e-30: ii1 = 0.0
                        i2[i, ji(k)]  = ii2
                        i2[i, ji(-k)] = ii1

            # Upward integration
            for k in range(1, mu + 1):
                yy = rm[ji(k)]
                if abs(yy) < 1.0e-10: continue
                i1[nt, ji(k)] = 0.0; zi1 = 0.0
                for i in range(nt - 1, -1, -1):
                    jj = i + 1
                    f  = h[jj] - h[i]
                    if f == 0.0: continue
                    a  = (i2[jj, ji(k)] - i2[i, ji(k)]) / f
                    b  = i2[i,  ji(k)]  - a * h[i]
                    c  = math.exp(max(-87.0, -f / yy))
                    d  = 1.0 - c
                    xx = h[i] - h[jj] * c
                    zi1 = c*zi1 + (d*(b + a*yy) + a*xx) * 0.5
                    if abs(zi1) <= 1.0e-20: zi1 = 0.0
                    i1[i, ji(k)] = zi1

            # Downward integration
            for k in range(-mu, 0):
                yy = rm[ji(k)]
                if abs(yy) < 1.0e-10: continue
                i1[0, ji(k)] = 0.0; zi1 = 0.0
                for i in range(1, nt + 1):
                    jj = i - 1
                    f  = h[i] - h[jj]
                    if f == 0.0: continue
                    c  = math.exp(max(-87.0, f / yy))
                    d  = 1.0 - c
                    a  = (i2[i, ji(k)] - i2[jj, ji(k)]) / f
                    b  = i2[i, ji(k)]  - a * h[i]
                    xx = h[i] - h[jj] * c
                    zi1 = c*zi1 + (d*(b + a*yy) + a*xx) * 0.5
                    if abs(zi1) <= 1.0e-20: zi1 = 0.0
                    i1[i, ji(k)] = zi1

            # Collect this order
            for k in range(-mu, mu + 1):
                idx = nt if k < 0 else 0
                _in[ji(k)] = i1[idx, ji(k)]
            roavion0 = i1[ipl, ji(mu)]

            # Convergence test (geometric series acceleration, ig > 2)
            if ig > 2:
                z = 0.0
                a1=roavion2; d1=roavion1; g1=roavion0
                if a1 >= accu and d1 >= accu and roavion >= accu:
                    y = abs(((g1/d1 - d1/a1) / (1.0 - g1/d1)**2) * (g1/roavion))
                    z = max(z, y)
                for l in range(-mu, mu + 1):
                    if l == 0: continue
                    a1=inm2[ji(l)]; d1=inm1[ji(l)]; g1=_in[ji(l)]
                    if a1<=accu or d1<=accu or i3[ji(l)]<=accu: continue
                    y = abs(((g1/d1 - d1/a1) / (1.0 - g1/d1)**2) * (g1/i3[ji(l)]))
                    z = max(z, y)
                if z < 0.0001:
                    # geometric series sum
                    for l in range(-mu, mu + 1):
                        d1=inm1[ji(l)]; g1=_in[ji(l)]
                        y1=1.0
                        if d1 > accu:
                            if abs(g1-d1) > accu:
                                y1 = 1.0 - g1/d1
                                g1 = g1/y1
                        i3[ji(l)] += g1
                    d1=roavion1; g1=roavion0; y1=1.0
                    if d1 >= accu:
                        if abs(g1-d1) >= accu:
                            y1 = 1.0 - g1/d1
                            g1 = g1/y1
                        roavion += g1
                    break

            # Update inm2/inm1
            inm2[:] = inm1[:]
            roavion2 = roavion1
            inm1[:]  = _in[:]
            roavion1 = roavion0

            # Accumulate
            i3[:]    += _in[:]
            roavion  += roavion0

            # Stop if order n < 0.001% of sum
            z = 0.0
            for l in range(-mu, mu + 1):
                if abs(i3[ji(l)]) >= accu:
                    z = max(z, abs(_in[ji(l)] / i3[ji(l)]))
            if z < 0.00001:
                break

            # Stop at order 20
            if ig >= 20:
                break

        # ── Accumulate Fourier components into xl ────────────────────────────
        delta0s = 1 if is_ == 0 else 2
        i4[:] += delta0s * i3[:]

        for l in range(1, np_ + 1):
            phi_l = rp[l-1]
            for m in range(-mum1, mum1 + 1):
                if m > 0:
                    xl[ji(m), l-1] += delta0s * i3[ji(m)] * math.cos(is_*(phi_l + pi))
                else:
                    xl[ji(m), l-1] += delta0s * i3[ji(m)] * math.cos(is_ * phi_l)

        if is_ == 0:
            for k in range(1, mum1 + 1):
                xl[ji(0), 0] += rm[ji(k)] * gb[ji(k)] * i3[ji(-k)]

        xl[ji(mu),  0] += delta0s * i3[ji(mu)]  * math.cos(is_*(phirad + pi))
        xl[ji(-mu), 0] += delta0s * roavion       * math.cos(is_*(phirad + pi))

        # Stop Fourier loop if higher harmonics are negligible
        z = 0.0
        for l in range(-mu, mu + 1):
            if abs(i4[ji(l)]) >= accu:
                z = max(z, abs(i3[ji(l)] / i4[ji(l)]))
        if z <= 0.001:
            break

    nt = snt