"""
iso.py
------
Translated from ISO.f

Successive Orders of Scattering (ISO) — computes diffuse transmittances.
"""

import math
import numpy as np

from .commons   import delta_sigma, trunc, ier as ier_common
from .scattering import discre   # avoid circular import


def iso(tamoy, trmoy, pizmoy, tamoyp, trmoyp, palt, nt, mu, rm, gb):
    """
    Successive orders of scattering — scalar version.
    Returns xf dict with keys -1, 0, 1.
    """
    from .scattering import discre

    delta  = delta_sigma.delta
    _ier   = ier_common

    accu  = 1.0e-20
    accu2 = 1.0e-3
    hr    = 8.0
    ta = tamoy; tr = trmoy; piz = pizmoy
    trp = trmoy - trmoyp
    tap = tamoy - tamoyp

    # Scale height for aerosols
    if palt <= 900.0 and palt > 0.0:
        ha  = -palt / math.log(tap / ta) if tap > 1.0e-3 else 2.0
        ntp = nt - 1
    else:
        ha  = 2.0
        ntp = nt

    h    = [0.0] * (nt + 2)
    xdel = [0.0] * (nt + 2)
    ydel = [0.0] * (nt + 2)

    # Build layer structure
    if ta <= accu2 and tr > ta:
        for j in range(ntp + 1):
            h[j]    = j * tr / ntp if ntp > 0 else 0.0
            ydel[j] = 1.0; xdel[j] = 0.0
    elif tr <= accu2 and ta > tr:
        for j in range(ntp + 1):
            h[j]    = j * ta / ntp if ntp > 0 else 0.0
            ydel[j] = 0.0; xdel[j] = piz
    else:
        ydel[0] = 1.0; xdel[0] = 0.0; h[0] = 0.0
        for it in range(ntp + 1):
            yy = h[it-1] if it > 0 else 0.0
            dd = ydel[it-1] if it > 0 else 0.0
            zx = discre(ta, ha, tr, hr, it, ntp, yy, dd, 300.0, 0.0)
            if _ier.ier:
                return {-1: 0.0, 0: 0.0, 1: 0.0}
            xxx = -zx / ha
            ca  = ta * math.exp(xxx) if xxx >= -18.0 else 0.0
            cr  = tr * math.exp(-zx / hr)
            h[it] = cr + ca
            denom = (cr/hr + ca/ha)
            ratio = (cr/hr) / denom if denom > 0 else 1.0
            xdel[it] = (1.0 - ratio) * piz
            ydel[it] = ratio

    # Plane layer insertion
    iplane = 0
    if ntp == nt - 1:
        taup = tap + trp
        iplane = 0
        for i in range(ntp + 1):
            if taup >= h[i]:
                iplane = i
        th = 0.005
        xt1 = abs(h[iplane] - taup)
        xt2 = abs(h[iplane+1] - taup) if iplane+1 <= nt else 1e9
        if xt1 > th and xt2 > th:
            for i in range(nt, iplane, -1):
                xdel[i]=xdel[i-1]; ydel[i]=ydel[i-1]; h[i]=h[i-1]
        else:
            if xt2 < xt1: iplane += 1
        h[iplane] = taup
        if tr > accu2 and ta > accu2:
            ca = ta*math.exp(-palt/ha); cr = tr*math.exp(-palt/hr)
            cr/=hr; ca/=ha
            ratio = cr/(cr+ca) if (cr+ca)>0 else 1.0
            xdel[iplane] = (1.0-ratio)*piz; ydel[iplane] = ratio
        elif tr > accu2:
            ydel[iplane]=1.0; xdel[iplane]=0.0
        else:
            ydel[iplane]=0.0; xdel[iplane]=piz

    # Rayleigh phase function coefficients
    aaaa  = delta / (2.0 - delta)
    ron   = (1.0 - aaaa) / (1.0 + 2.0 * aaaa)
    beta0 = 1.0
    beta2 = 0.5 * ron

    # Array dimensions: signed indices -mu..+mu stored at offset +mu
    N = 2*mu+1
    def ji(k): return k+mu   # signed → 0-based

    # Kernel (xpl=psl[2,:], bp=phase kernel matrix)
    from .kernel import kernel_func
    xpl, psl, bp = kernel_func(0, mu, rm)

    # i1[layer, angle] primary radiation field
    i1 = np.zeros((nt+1, N))
    i2 = np.zeros((nt+1, N))
    i3 = np.zeros(N)
    inm1 = np.zeros(N)
    inm2 = np.zeros(N)
    _in  = np.zeros(N)

    # Primary upward radiation: i1(nt,k>0) = exp(-tau_total/rm(k))
    for k in range(1, mu+1):
        yy = rm[ji(k)]
        if abs(yy) < 1e-10: continue
        i1[nt, ji(k)] = 1.0
        for i in range(nt-1, -1, -1):
            arg = -(ta+tr-h[i]) / yy
            i1[i, ji(k)] = math.exp(max(-87.0, arg))

    # Primary downward: zero (no solar downward source in this formulation)
    # (downward comes from scattering of primary upward)

    # Initialise accumulators with primary field
    for k in range(-mu, mu+1):
        idx = nt if k < 0 else 0
        inm1[ji(k)] = i1[idx, ji(k)]
        inm2[ji(k)] = i1[idx, ji(k)]
        i3[ji(k)]   = i1[idx, ji(k)]

    # For plane: tavion = upward flux at plane level
    ipl = iplane if iplane >= 0 else 0
    tavion  = i1[ipl, ji(mu)]
    tavion2 = tavion
    tavion1 = 0.0
    ig = 1

    # Successive orders loop
    for _iter in range(20):
        ig += 1

        # Source function i2 at each level
        for k in range(1, mu+1):
            xpk = xpl[ji(k)]; ypk = xpl[ji(-k)]
            for i in range(nt+1):
                ii1 = 0.0; ii2 = 0.0
                x = xdel[i]; y = ydel[i]
                for j in range(1, mu+1):
                    xpj = xpl[ji(j)]
                    z   = gb[ji(j)]
                    xi1 = i1[i, ji(j)]; xi2 = i1[i, ji(-j)]
                    bpjk  = bp[j, ji(k)]  * x + y*(beta0 + beta2*xpj*xpk)
                    bpjmk = bp[j, ji(-k)] * x + y*(beta0 + beta2*xpj*ypk)
                    ii2  += z*(xi1*bpjk  + xi2*bpjmk)
                    ii1  += z*(xi1*bpjmk + xi2*bpjk)
                i2[i, ji(k)]  = ii2
                i2[i, ji(-k)] = ii1

        # Upward integration
        for k in range(1, mu+1):
            yy = rm[ji(k)]
            if abs(yy) < 1e-10: continue
            zi1 = 0.0; i1[nt, ji(k)] = 0.0
            for i in range(nt-1, -1, -1):
                jj = i+1; f = h[jj]-h[i]
                if f == 0: continue
                a = (i2[jj,ji(k)]-i2[i,ji(k)])/f
                b = i2[i,ji(k)] - a*h[i]
                c = math.exp(max(-87.0,-f/yy)); d = 1.0-c
                xx = h[i]-h[jj]*c
                zi1 = c*zi1 + (d*(b+a*yy)+a*xx)*0.5
                i1[i, ji(k)] = zi1

        # Downward integration
        for k in range(-mu, 0):
            yy = rm[ji(k)]
            if abs(yy) < 1e-10: continue
            zi1 = 0.0; i1[0, ji(k)] = 0.0
            for i in range(1, nt+1):
                jj = i-1; f = h[i]-h[jj]
                if f == 0: continue
                c = math.exp(min(87.0, f/yy)); d = 1.0-c
                a = (i2[i,ji(k)]-i2[jj,ji(k)])/f
                b = i2[i,ji(k)]-a*h[i]
                xx = h[i]-h[jj]*c
                zi1 = c*zi1 + (d*(b+a*yy)+a*xx)*0.5
                i1[i, ji(k)] = zi1

        # Current order contribution
        for k in range(-mu, mu+1):
            idx = nt if k < 0 else 0
            _in[ji(k)] = i1[idx, ji(k)]
        tavion0 = i1[ipl, ji(mu)]

        # Convergence test (geometric series extrapolation)
        if ig > 2:
            z = 0.0
            a1=tavion2; d1=tavion1; g1=tavion0
            if a1>=accu and d1>=accu and tavion>=accu:
                if abs(1.0-g1/d1) > 1e-14:
                    z = max(z, abs(((g1/d1-d1/a1)/((1.0-g1/d1)**2))*(g1/tavion)))
            for l in range(-mu, mu+1):
                if l==0: continue
                a1=inm2[ji(l)]; d1=inm1[ji(l)]; g1=_in[ji(l)]
                if a1==0 or d1==0 or i3[ji(l)]==0: continue
                if abs(1.0-g1/d1) > 1e-14:
                    z = max(z, abs(((g1/d1-d1/a1)/((1.0-g1/d1)**2))*(g1/i3[ji(l)])))

            if z < 0.0001:
                # Geometric series sum
                for l in range(-mu, mu+1):
                    if l==0: continue
                    d1=inm1[ji(l)]; g1=_in[ji(l)]
                    if d1 == 0: continue
                    y1 = 1.0 - g1/d1
                    if abs(y1) > 1e-14:
                        i3[ji(l)] += g1/y1
                d1=tavion1; g1=tavion0
                if d1>=accu and abs(g1-d1)>=accu:
                    y1=1.0-g1/d1
                    if abs(y1)>1e-14: tavion += g1/y1
                    else: tavion += g1
                else:
                    tavion += g1
                break

            inm2[:] = inm1; tavion2 = tavion1

        inm1[:] = _in; tavion1 = tavion0
        i3 += _in; tavion += tavion0

        # Stop if nth order < 1e-5 of sum
        z = 0.0
        for l in range(-mu, mu+1):
            if i3[ji(l)] != 0:
                z = max(z, abs(_in[ji(l)]/i3[ji(l)]))
        if z < 0.00001:
            break

    # Build output
    xf_1 = i3[ji(mu)]           # downward diffuse at surface
    xf_m1 = tavion               # upward at plane/TOA
    xf_0  = sum(rm[ji(k)]*gb[ji(k)]*i3[ji(-k)] for k in range(1,mu+1))

    return {1: xf_1, -1: xf_m1, 0: xf_0}
