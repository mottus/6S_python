"""
mie.py
------
Translated from MIE.f

Mie scattering computation for arbitrary aerosol size distributions.
Computes extinction coefficients, scattering coefficients, asymmetry parameters,
and phase functions for the aerosol SOS common block.

Supported size distributions (iaer):
    8  – multimodal log-normal (up to 4 modes)
    9  – modified gamma distribution
    10 – Junge power-law distribution
    11 – sun photometer measurements (50 points max)
"""

import math
import numpy as np


def mie(iaer, wldis, ex_in, sc_in, asy_in, icp, rmin, rmax,
        rn, ri, x1, x2, x3, cij, irsunph, rsunph, nrsunph,
        cgaus, pdgs):
    """
    Mie scattering: compute bulk optical properties for a particle mixture.

    Parameters
    ----------
    iaer     : int         – distribution type (8=LogNorm, 9=ModGamma, 10=Junge, 11=SunPhot)
    wldis    : array (10,) – wavelengths (µm)
    ex_in    : array (4,10) – output extinction (modified in-place)
    sc_in    : array (4,10) – output scattering (modified in-place)
    asy_in   : array (4,10) – output asymmetry (modified in-place)
    icp      : int         – number of particle components (1..4)
    rmin,rmax: float       – radius integration limits (µm)
    rn,ri    : (10,4) arrays – real and imaginary refractive indices
    x1,x2,x3: arrays (4,)  – distribution parameters
    cij      : array (4,)  – component volume fractions
    irsunph  : int         – number of sun-photometer radius points
    rsunph   : array (50,) – sun-photometer radii
    nrsunph  : array (50,) – sun-photometer dV/d(logr) values
    cgaus    : array (83,) – Gauss angles (cos values) for phase function
    pdgs     : array (83,) – Gauss weights for phase function

    Side-effects
    ------------
    Fills the /sixs_aerbas/ ph(10,83) array via return value.

    Returns
    -------
    ph : array (10,83) – mixed phase function
    ex_out,sc_out,asy_out : arrays (4,10) – mixed optical properties
    """
    pi      = 4.0 * math.atan(1.0)
    rlogpas = 0.030
    nbmu    = 83
    ldexp   = -300.0

    # Output arrays (component-wise, then mixed into [0,:])
    ext = np.zeros((10, 4))
    sca = np.zeros((10, 4))
    np_ = np.zeros(4)
    p1  = np.zeros((10, 4, nbmu))
    ph  = np.zeros((10, nbmu))

    def size_dist(r, i):
        """Number density dn/dr for component i at radius r."""
        typ = iaer - 7
        if   typ == 1:   # Log-Normal
            sigma = float(x2[i])
            if sigma <= 0: return 0.0
            arg = -0.5 * (math.log10(r / float(x1[i])) / math.log10(sigma))**2
            return (math.exp(arg) / (math.sqrt(2 * pi) * math.log10(sigma) * math.log(10) * r))
        elif typ == 2:   # Modified Gamma
            r0  = 1.0
            arg = -float(x2[i]) * ((r / r0)**float(x3[i]))
            if arg < ldexp: return 0.0
            return ((r / r0)**float(x1[i])) * math.exp(arg)
        elif typ == 3:   # Junge power-law
            r0 = 0.1
            return (r**(-float(x1[i])) if r > r0 else r0**(-float(x1[i])))
        else:            # Sun photometer (iaer=11)
            for j in range(1, irsunph):
                if r < rsunph[j]:
                    frac = (r - rsunph[j-1]) / (rsunph[j] - rsunph[j-1])
                    return float(nrsunph[j-1]) + frac * float(nrsunph[j] - nrsunph[j-1])
            return 0.0

    # Integration over radius for each component
    for i in range(icp):
        r  = float(rmin)
        dr = r * (10**rlogpas - 1.0)
        while r < float(rmax):
            nr_dens = size_dist(r, i)
            xndpr2  = nr_dens * dr * pi * r**2
            np_[i] += nr_dens * dr

            for l in range(10):
                if xndpr2 * float(cij[i]) < 1e-8 / math.sqrt(float(wldis[l])):
                    continue
                alpha = 2.0 * pi * r / float(wldis[l])
                nr_val = float(rn[l, i]); ni_val = float(ri[l, i])
                Qext, Qsca, p11 = exscphase(alpha, nr_val, ni_val, cgaus, nbmu)
                ext[l, i] += xndpr2 * Qext
                sca[l, i] += xndpr2 * Qsca
                for k in range(nbmu):
                    p1[l, i, k] += p11[k] * xndpr2

            r  += dr
            dr  = r * (10**rlogpas - 1.0)

    # Mix components (normalize by np_, convert to km-1)
    ex_out  = np.zeros((4, 10), dtype=np.float32)
    sc_out  = np.zeros((4, 10), dtype=np.float32)
    asy_out = np.zeros((4, 10), dtype=np.float32)

    for l in range(10):
        for i in range(icp):
            if np_[i] > 0:
                ext[l, i] /= np_[i] * 1e3
                sca[l, i] /= np_[i] * 1e3
            ex_out[0, l] += float(cij[i]) * float(ext[l, i])
            sc_out[0, l] += float(cij[i]) * float(sca[l, i])

    # Phase function and asymmetry
    for l in range(10):
        asy_n = asy_d = 0.0
        sc_l = sc_out[0, l]
        for k in range(nbmu):
            ph[l, k] = 0.0
            for i in range(icp):
                if np_[i] > 0:
                    ph[l, k] += float(cij[i]) * float(p1[l, i, k]) / (np_[i] * 1e3)
            if sc_l > 0:
                ph[l, k] /= sc_l
            asy_n += cgaus[k] * ph[l, k] * pdgs[k] / 10.0
            asy_d += ph[l, k] * pdgs[k] / 10.0
        asy_out[0, l] = asy_n / asy_d if asy_d > 0 else 0.0

    return ph, ex_out, sc_out, asy_out


def exscphase(X, nr, ni, cgaus, nbmu=83):
    """
    Mie theory: compute Qext, Qsca, and phase function p11 at nbmu angles.

    Parameters
    ----------
    X    : float – size parameter (2π r / λ)
    nr   : float – real part of refractive index
    ni   : float – imaginary part
    cgaus: array (83,) – cos(angle) values
    nbmu : int   – number of angles

    Returns
    -------
    Qext : float
    Qsca : float
    p11  : array (83,) – phase function values
    """
    nser = 10000
    Ren  = nr / (nr**2 + ni**2)
    Imn  = ni / (nr**2 + ni**2)

    # Determine truncation order
    N    = int(0.5 * (-1.0 + math.sqrt(1.0 + 4.0 * X**2))) + 1
    N    = max(N, 2)
    mu2  = 1000000
    Np   = N
    Up   = 2.0 * X / (2.0 * Np + 1.0)
    mu1  = int(Np + 30.0 * (0.10 + 0.35 * Up * (2 - Up**2) / 2.0 / max(1 - Up, 1e-10)))
    Np2  = int(X - 0.5 + math.sqrt(30.0 * 0.35 * X))
    if Np2 > N:
        Up2 = 2.0 * X / (2.0 * Np2 + 1.0)
        mu2 = int(Np2 + 30.0 * (0.10 + 0.35 * Up2 * (2 - Up2**2) / 2.0 / max(1 - Up2, 1e-10)))
    mu  = min(mu1, mu2)
    mu  = min(mu, nser)

    # Downward recursion for Rn and xj (Bessel functions)
    Rn  = [0.0] * (mu + 2)
    xj  = [0.0] * (mu + 2)
    mub = mu
    k   = mu + 1
    while k > 1:
        k -= 1
        xj[k] = 0.0
        denom = 2.0 * k + 1.0 - X * Rn[k]
        Rn[k-1] = X / denom if abs(denom) > 1e-30 else 0.0
        if k == 2:
            mub = mu; xj[mub+1] = Rn[mub]; xj[mub] = 1.0; break
        if Rn[k-1] > 1.0:
            mub = k - 1; xj[mub+1] = Rn[mub]; xj[mub] = 1.0; break

    for k in range(mub, 0, -1):
        xj[k-1] = (2.0 * k + 1.0) * xj[k] / X - xj[k+1]

    coxj = (xj[0] - X * xj[1]) * math.cos(X) + X * xj[0] * math.sin(X)
    if abs(coxj) < 1e-300:
        return 0.0, 0.0, np.zeros(nbmu)

    # Dn(X) and Dn(X*m): downward recursion
    RDnX = [0.0] * (mu + 1)
    RDnY = [0.0] * (mu + 1)
    IDnY = [0.0] * (mu + 1)
    for k in range(mu, 0, -1):
        denom = RDnX[k] + k / X
        RDnX[k-1] = k / X - 1.0 / denom if abs(denom) > 1e-300 else 0.0
        xnR = RDnY[k] + Ren * k / X
        xnI = IDnY[k] + Imn * k / X
        den = xnR**2 + xnI**2
        if den < 1e-300: den = 1e-300
        RDnY[k-1] = k * Ren / X - xnR / den
        IDnY[k-1] = k * Imn / X + xnI / den

    # Upward recursion
    xy  = [0.0] * (mu + 2)
    xy[-1] = math.sin(X) / X
    xy[0]  = -math.cos(X) / X
    RGnX = [0.0] * (mu + 1)
    IGnX = [0.0] * (mu + 1)
    IGnX[0] = -1.0
    RAn = [0.0] * (mu + 1)
    IAn = [0.0] * (mu + 1)
    RBn = [0.0] * (mu + 1)
    IBn = [0.0] * (mu + 1)

    Qsca = 0.0; Qext = 0.0
    mu_eff = mu

    for k in range(1, mu + 1):
        if k <= mub:
            xj[k] = xj[k] / coxj
        else:
            xj[k] = Rn[k-1] * xj[k-1]

        xy[k] = (2.0*k - 1.0) * xy[k-1] / X - xy[k-2]
        h2 = xj[k]**2 + xy[k]**2
        xJonH = xj[k] / h2 if h2 > 1e-300 else 0.0

        den = (RGnX[k-1] - k/X)**2 + IGnX[k-1]**2
        if den < 1e-300: den = 1e-300
        RGnX[k] = (k/X - RGnX[k-1]) / den - k/X
        IGnX[k] = IGnX[k-1] / den

        # An coefficients
        n1A = RDnY[k] - nr * RDnX[k]
        n2A = IDnY[k] + ni * RDnX[k]
        d1A = RDnY[k] - nr * RGnX[k] - ni * IGnX[k]
        d2A = IDnY[k] + ni * RGnX[k] - nr * IGnX[k]
        dA  = d1A**2 + d2A**2
        if dA < 1e-300: dA = 1e-300
        RAnb = (n1A * d1A + n2A * d2A) / dA
        IAnb = (-n1A * d2A + n2A * d1A) / dA
        RAn[k] = xJonH * (xj[k] * RAnb - xy[k] * IAnb)
        IAn[k] = xJonH * (xy[k] * RAnb + xj[k] * IAnb)

        # Bn coefficients
        n1B = nr * RDnY[k] + ni * IDnY[k] - RDnX[k]
        n2B = nr * IDnY[k] - ni * RDnY[k]
        d1B = nr * RDnY[k] + ni * IDnY[k] - RGnX[k]
        d2B = nr * IDnY[k] - ni * RDnY[k] - IGnX[k]
        dB  = d1B**2 + d2B**2
        if dB < 1e-300: dB = 1e-300
        RBnb = (n1B * d1B + n2B * d2B) / dB
        IBnb = (-n1B * d2B + n2B * d1B) / dB
        RBn[k] = xJonH * (xj[k] * RBnb - xy[k] * IBnb)
        IBn[k] = xJonH * (xy[k] * RBnb + xj[k] * IBnb)

        test = (RAn[k]**2 + IAn[k]**2 + RBn[k]**2 + IBn[k]**2) / k
        if test < 1e-14:
            mu_eff = k; break

        xpond = 2.0 / X**2 * (2.0 * k + 1.0)
        Qsca += xpond * (RAn[k]**2 + IAn[k]**2 + RBn[k]**2 + IBn[k]**2)
        Qext += xpond * (RAn[k] + RBn[k])

    # Phase function
    p11 = np.zeros(nbmu)
    for j in range(nbmu):
        xmud = cgaus[j]
        RS1 = RS2 = IS1 = IS2 = 0.0
        PIn  = [0.0] * (mu_eff + 2)
        TAUn = [0.0] * (mu_eff + 2)
        PIn[0]  = 0.0; PIn[1]  = 1.0
        TAUn[1] = xmud
        for k in range(1, mu_eff + 1):
            cn  = (2.0 * k + 1.0) / (k * (k + 1.0))
            RS1 += cn * (RAn[k] * PIn[k]  + RBn[k] * TAUn[k])
            RS2 += cn * (RAn[k] * TAUn[k] + RBn[k] * PIn[k])
            IS1 += cn * (IAn[k] * PIn[k]  + IBn[k] * TAUn[k])
            IS2 += cn * (IAn[k] * TAUn[k] + IBn[k] * PIn[k])
            if k < mu_eff:
                PIn[k+1]  = ((2*k+1) * xmud * PIn[k] - (k+1) * PIn[k-1]) / k
                TAUn[k+1] = (k+1) * xmud * PIn[k+1] - (k+2) * PIn[k]
        p11[j] = 2.0 * (RS1**2 + IS1**2 + RS2**2 + IS2**2) / X**2

    return Qext, Qsca, p11
