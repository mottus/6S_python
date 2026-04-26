"""
ocean_brdf.py
-------------
Translated from OCEABRDF.f, OCEAALBE.f, OCEATOOLS.f

Ocean surface BRDF model including:
- Cox-Munk sunglint (anisotropic Gaussian wave-facet distribution)
- Fresnel reflection
- Morel Case-I water body reflectance
- Whitecap reflectance (Koepke 1984)
"""

import math
import numpy as np

from .gauss import gauss

# ---------------------------------------------------------------------------
# Whitecap effective reflectance (Koepke 1984), 39 values, 0.2–4.0 µm
# ---------------------------------------------------------------------------
_REF_WHITECAP = [
    0.220, 0.220, 0.220, 0.220, 0.220, 0.220, 0.215, 0.210, 0.200, 0.190,
    0.175, 0.155, 0.130, 0.080, 0.100, 0.105, 0.100, 0.080, 0.045, 0.055,
    0.065, 0.060, 0.055, 0.040, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
    0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000,
]

# ---------------------------------------------------------------------------
# Water refractive index (Hale & Querry 1973)
# ---------------------------------------------------------------------------
_TWL = [
    0.250, 0.275, 0.300, 0.325, 0.345, 0.375, 0.400, 0.425, 0.445, 0.475,
    0.500, 0.525, 0.550, 0.575, 0.600, 0.625, 0.650, 0.675, 0.700, 0.725,
    0.750, 0.775, 0.800, 0.825, 0.850, 0.875, 0.900, 0.925, 0.950, 0.975,
    1.000, 1.200, 1.400, 1.600, 1.800, 2.000, 2.200, 2.400, 2.600, 2.650,
    2.700, 2.750, 2.800, 2.850, 2.900, 2.950, 3.000, 3.050, 3.100, 3.150,
    3.200, 3.250, 3.300, 3.350, 3.400, 3.450, 3.500, 3.600, 3.700, 3.800,
    3.900, 4.000,
]
_TNR = [
    1.362, 1.354, 1.349, 1.346, 1.343, 1.341, 1.339, 1.338, 1.337, 1.336,
    1.335, 1.334, 1.333, 1.333, 1.332, 1.332, 1.331, 1.331, 1.331, 1.330,
    1.330, 1.330, 1.329, 1.329, 1.329, 1.328, 1.328, 1.328, 1.327, 1.327,
    1.327, 1.324, 1.321, 1.317, 1.312, 1.306, 1.296, 1.279, 1.242, 1.219,
    1.188, 1.157, 1.142, 1.149, 1.201, 1.292, 1.371, 1.426, 1.467, 1.483,
    1.478, 1.467, 1.450, 1.432, 1.420, 1.410, 1.400, 1.385, 1.374, 1.364,
    1.357, 1.351,
]
_TNI = [
    3.35e-8, 2.35e-8, 1.60e-8, 1.08e-8, 6.50e-9,
    3.50e-9, 1.86e-9, 1.30e-9, 1.02e-9, 9.35e-10,
    1.00e-9, 1.32e-9, 1.96e-9, 3.60e-9, 1.09e-8,
    1.39e-8, 1.64e-8, 2.23e-8, 3.35e-8, 9.15e-8,
    1.56e-7, 1.48e-7, 1.25e-7, 1.82e-7, 2.93e-7,
    3.91e-7, 4.86e-7, 1.06e-6, 2.93e-6, 3.48e-6,
    2.89e-6, 9.89e-6, 1.38e-4, 8.55e-5, 1.15e-4,
    1.10e-3, 2.89e-4, 9.56e-4, 3.17e-3, 6.70e-3,
    1.90e-2, 5.90e-2, 1.15e-1, 1.85e-1, 2.68e-1,
    2.98e-1, 2.72e-1, 2.40e-1, 1.92e-1, 1.35e-1,
    9.24e-2, 6.10e-2, 3.68e-2, 2.61e-2, 1.95e-2,
    1.32e-2, 9.40e-3, 5.15e-3, 3.60e-3, 3.40e-3,
    3.80e-3, 4.60e-3,
]

# Morel Case-I water attenuation tables (400–700 nm, 5 nm steps, 61 points)
_TKW = [0.0209,0.0200,0.0196,0.0189,0.0183,0.0182,0.0171,0.0170,0.0168,0.0166,
        0.0168,0.0170,0.0173,0.0174,0.0175,0.0184,0.0194,0.0203,0.0217,0.0240,
        0.0271,0.0320,0.0384,0.0445,0.0490,0.0505,0.0518,0.0543,0.0568,0.0615,
        0.0640,0.0640,0.0717,0.0762,0.0807,0.0940,0.1070,0.1280,0.1570,0.2000,
        0.2530,0.2790,0.2960,0.3030,0.3100,0.3150,0.3200,0.3250,0.3300,0.3400,
        0.3500,0.3700,0.4050,0.4180,0.4300,0.4400,0.4500,0.4700,0.5000,0.5500,
        0.6500]
_TXC = [0.1100,0.1111,0.1125,0.1135,0.1126,0.1104,0.1078,0.1065,0.1041,0.0996,
        0.0971,0.0939,0.0896,0.0859,0.0823,0.0788,0.0746,0.0726,0.0690,0.0660,
        0.0636,0.0600,0.0578,0.0540,0.0498,0.0475,0.0467,0.0450,0.0440,0.0426,
        0.0410,0.0400,0.0390,0.0375,0.0360,0.0340,0.0330,0.0328,0.0325,0.0330,
        0.0340,0.0350,0.0360,0.0375,0.0385,0.0400,0.0420,0.0430,0.0440,0.0445,
        0.0450,0.0460,0.0475,0.0490,0.0515,0.0520,0.0505,0.0440,0.0390,0.0340,
        0.0300]
_TE  = [0.668,0.672,0.680,0.687,0.693,0.701,0.707,0.708,0.707,0.704,
        0.701,0.699,0.700,0.703,0.703,0.703,0.703,0.704,0.702,0.700,
        0.700,0.695,0.690,0.685,0.680,0.675,0.670,0.665,0.660,0.655,
        0.650,0.645,0.640,0.630,0.623,0.615,0.610,0.614,0.618,0.622,
        0.626,0.630,0.634,0.638,0.642,0.647,0.653,0.658,0.663,0.667,
        0.672,0.677,0.682,0.687,0.695,0.697,0.693,0.665,0.640,0.620,
        0.600]
_TBW = [0.0076,0.0072,0.0068,0.0064,0.0061,0.0058,0.0055,0.0052,0.0049,0.0047,
        0.0045,0.0043,0.0041,0.0039,0.0037,0.0036,0.0034,0.0033,0.0031,0.0030,
        0.0029,0.0027,0.0026,0.0025,0.0024,0.0023,0.0022,0.0022,0.0021,0.0020,
        0.0019,0.0018,0.0018,0.0017,0.0017,0.0016,0.0016,0.0015,0.0015,0.0014,
        0.0014,0.0013,0.0013,0.0012,0.0012,0.0011,0.0011,0.0010,0.0010,0.0010,
        0.0010,0.0009,0.0008,0.0008,0.0008,0.0007,0.0007,0.0007,0.0007,0.0007,
        0.0007]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def indwat(wl, xsal):
    """
    Compute sea water refractive index at wavelength wl (µm).

    Parameters
    ----------
    wl   : float – wavelength (µm)
    xsal : float – salinity (ppt); if <0 uses 34.3 ppt

    Returns
    -------
    nr, ni : floats – real and imaginary refractive index
    """
    i = 1
    while i < 61 and wl >= _TWL[i]:
        i += 1
    xwl = _TWL[i] - _TWL[i - 1]
    yr  = _TNR[i] - _TNR[i - 1]
    yi  = _TNI[i] - _TNI[i - 1]
    nr  = _TNR[i - 1] + (wl - _TWL[i - 1]) * yr / xwl
    ni  = _TNI[i - 1] + (wl - _TWL[i - 1]) * yi / xwl
    # Salinity correction
    nrc = 0.006
    nic = 0.000
    sal = max(xsal, 34.3) if xsal < 0 else xsal
    nr += nrc * (sal / 34.3)
    ni += nic * (sal / 34.3)
    return nr, ni


def morcasiwat(wl, C):
    """
    Morel Case-I water sub-surface reflectance.

    Parameters
    ----------
    wl : float – wavelength (µm), valid 0.40–0.70
    C  : float – pigment concentration (mg/m³)

    Returns
    -------
    R2 : float – sub-surface reflectance
    """
    if wl < 0.400 or wl > 0.700:
        return 0.0
    iwl = 1 + round((wl - 0.400) / 0.005)
    iwl = max(0, min(60, iwl))
    Kw  = _TKW[iwl]
    Xc  = _TXC[iwl]
    e   = _TE[iwl]
    bw  = _TBW[iwl]
    if abs(C) < 0.0001:
        bb = 0.5 * bw
        Kd = Kw
    else:
        b   = 0.30 * C ** 0.62
        bbt = 0.002 + 0.02 * (0.5 - 0.25 * math.log10(C)) * 0.550 / wl
        bb  = 0.5 * bw + bbt * b
        Kd  = Kw + Xc * C ** e
    # Iterative solution for R2
    u1 = 0.75
    R1 = 0.33 * bb / u1 / Kd
    for _ in range(50):
        u2 = 0.90 * (1.0 - R1) / (1.0 + 2.25 * R1)
        R2 = 0.33 * bb / u2 / Kd
        if abs((R2 - R1) / R2) < 0.0001:
            return R2
        R1 = R2
    return R2


def _fresnel(nr, ni, coschi, sinchi):
    """Fresnel reflection coefficient."""
    a1  = abs(nr**2 - ni**2 - sinchi**2)
    a2  = math.sqrt((nr**2 - ni**2 - sinchi**2)**2 + 4 * nr**2 * ni**2)
    u   = math.sqrt(0.5 * (a1 + a2))
    v   = math.sqrt(max(0.0, 0.5 * (-a1 + a2)))
    Rr2 = ((coschi - u)**2 + v**2) / ((coschi + u)**2 + v**2)
    b1  = (nr**2 - ni**2) * coschi
    b2  = 2 * nr * ni * coschi
    Rl2 = ((b1 - u)**2 + (b2 + v)**2) / ((b1 + u)**2 + (b2 - v)**2)
    return (Rr2 + Rl2) / 2.0


def sunglint(wspd, nr, ni, azw, ts, tv, fi):
    """
    Sun-glint reflectance using Cox-Munk anisotropic Gaussian wave statistics.

    Parameters
    ----------
    wspd : float – wind speed (m/s)
    nr   : float – real refractive index of sea water
    ni   : float – imaginary refractive index
    azw  : float – azimuth of sun minus azimuth of wind (degrees)
    ts   : float – solar zenith angle (degrees)
    tv   : float – view zenith angle (degrees)
    fi   : float – relative azimuth sun–satellite (degrees)

    Returns
    -------
    rog : float – sun-glint reflectance
    """
    pi  = math.acos(-1.0)
    fac = pi / 180.0
    phw = azw * fac
    cs  = math.cos(ts * fac);  ss = math.sin(ts * fac)
    cv  = math.cos(tv * fac);  sv = math.sin(tv * fac)
    phi = fi * fac

    Zx = -sv * math.sin(phi) / (cs + cv)
    Zy =  (ss + sv * math.cos(phi)) / (cs + cv)
    tantilt = math.sqrt(Zx**2 + Zy**2)
    tilt    = math.atan(tantilt)

    sigmaC = 0.003 + 0.00192 * wspd
    sigmaU = 0.00316 * wspd
    C21 = 0.01 - 0.0086 * wspd
    C03 = 0.04 - 0.033  * wspd
    C40 = 0.40;  C22 = 0.12;  C04 = 0.23

    xe  = (math.cos(phw) * Zx + math.sin(phw) * Zy) / math.sqrt(sigmaC)
    xn  = (-math.sin(phw) * Zx + math.cos(phw) * Zy) / math.sqrt(sigmaU)
    xe2 = xe**2;  xn2 = xn**2

    coef = (1 - C21 / 2.0 * (xe2 - 1) * xn
              - C03 / 6.0 * (xn2 - 3) * xn
              + C40 / 24.0 * (xe2**2 - 6 * xe2 + 3)
              + C04 / 24.0 * (xn2**2 - 6 * xn2 + 3)
              + C22 / 4.0  * (xe2 - 1) * (xn2 - 1))

    proba = coef / (2.0 * pi * math.sqrt(sigmaU) * math.sqrt(sigmaC)) \
            * math.exp(-(xe2 + xn2) / 2.0)

    cos2chi = cv * cs + sv * ss * math.cos(phi)
    cos2chi = max(-1.0, min(1.0, cos2chi))
    coschi  = math.sqrt(0.5 * (1.0 + cos2chi))
    sinchi  = math.sqrt(max(0.0, 0.5 * (1.0 - cos2chi)))

    R1  = _fresnel(nr, ni, coschi, sinchi)
    cos_tilt4 = math.cos(tilt)**4
    if abs(cs) < 1e-10 or abs(cv) < 1e-10 or abs(cos_tilt4) < 1e-30:
        return 0.0
    rog = pi * R1 * proba / (4.0 * cs * cv * cos_tilt4)
    return max(0.0, rog)


def glitalbe(wspd, nr, ni, azw):
    """
    Spherical albedo of the ocean surface (sun-glint component).

    Parameters
    ----------
    wspd : float – wind speed (m/s)
    nr   : float – real refractive index
    ni   : float – imaginary refractive index
    azw  : float – azimuth of sun minus azimuth of wind (degrees)

    Returns
    -------
    rge : float – glint spherical albedo
    """
    pi  = math.acos(-1.0)
    fac = pi / 180.0
    sigma  = 0.003 + 0.00512 * wspd
    sigmaC = 0.003 + 0.00192 * wspd
    sigmaU = 0.00316 * wspd
    C21 = 0.01 - 0.0086 * wspd
    C03 = 0.04 - 0.033  * wspd
    C40 = 0.40;  C22 = 0.12;  C04 = 0.23

    q     = 50.0
    costt = 1.0 / math.sqrt(1.0 + q * sigma / 4.0)
    phw   = azw * fac

    prefl = 0.0
    proba = 0.0
    ntb   = 31
    htb   = 1.0 / (ntb - 1)

    for km in range(1, ntb + 1):
        costet = (km - 1) * htb
        tet    = math.acos(costet)
        sintet = math.sin(tet)
        tet_d  = tet / fac

        # Simpson weight
        cotb = 2.0
        if km % 2 == 0: cotb = 4.0
        if km == 1 or km == ntb: cotb = 1.0

        if   tet_d < 65: nta = 31
        elif tet_d < 75: nta = 101
        elif tet_d < 81: nta = 301
        else:            nta = 801
        nfa = nta
        hta = (1.0 - costt) / (nta - 1)
        hfa = pi / (nfa - 1)

        pr = pp = 0.0
        for i in range(1, nfa + 1):
            phin    = (i - 1) * hfa
            cosphin = math.cos(phin)
            sinphin = math.sin(phin)
            cofa = 2.0
            if i % 2 == 0: cofa = 4.0
            if i == 1 or i == nfa: cofa = 1.0

            for j in range(1, nta + 1):
                costetn = costt + (j - 1) * hta
                sintetn = math.sqrt(max(0.0, 1.0 - costetn**2))
                tantetn = sintetn / costetn if costetn > 1e-10 else 1e10
                cota = 2.0
                if j % 2 == 0: cota = 4.0
                if j == 1 or j == nta: cota = 1.0

                coschi = costet * costetn + sintet * sintetn * cosphin
                coschi = max(-1.0, min(1.0, coschi))
                sinchi = math.sqrt(max(0.0, 1.0 - coschi**2))
                if coschi < 0.0:
                    r1   = 0.0
                    cota = 0.0
                else:
                    r1 = _fresnel(nr, ni, coschi, sinchi)

                Zx  = -tantetn * cosphin
                Zy  = -tantetn * sinphin
                xe  = (math.cos(phw) * Zx + math.sin(phw) * Zy) / math.sqrt(sigmaC)
                xn  = (-math.sin(phw) * Zx + math.cos(phw) * Zy) / math.sqrt(sigmaU)
                xe2 = xe**2;  xn2 = xn**2
                coef = (1 - C21/2*(xe2-1)*xn - C03/6*(xn2-3)*xn
                          + C40/24*(xe2**2-6*xe2+3)
                          + C04/24*(xn2**2-6*xn2+3)
                          + C22/4*(xe2-1)*(xn2-1))
                fonc0 = (0.5 * coschi * coef
                         * math.exp(-(xe2 + xn2) / 2.0)
                         / max(costetn**4, 1e-30))
                pr += r1   * fonc0 * cofa * cota * cotb
                pp += fonc0 * cofa * cota * cotb

        pond = 2.0 * hta * hfa * htb / pi / math.sqrt(sigmaC) / math.sqrt(sigmaU) / 27.0
        prefl += pr * pond
        proba += pp * pond

    return prefl / proba if proba > 0 else 0.0


# ---------------------------------------------------------------------------
# OCEABRDF  — directional BRDF at Gauss angles
# ---------------------------------------------------------------------------

def oceabrdf(pws, paw, xsal, pcl, pwl, mu, np_, rm, rp):
    """
    Ocean surface BRDF at Gauss quadrature angles.

    Parameters
    ----------
    pws  : float – wind speed (m/s)
    paw  : float – azimuth of sun minus azimuth of wind (degrees)
    xsal : float – salinity (ppt); <0 → 34.3 ppt
    pcl  : float – pigment concentration (mg/m³)
    pwl  : float – wavelength (µm)
    mu   : int   – number of Gauss zenith angles
    np_  : int   – number of azimuth points
    rm   : array – Gauss nodes, signed index -mu..+mu (offset by +mu)
    rp   : array – azimuth points (length np_)

    Returns
    -------
    brdfint : 2-D array, shape (2*mu+1, np_)
    """
    pi  = math.atan(1.0) * 4.0
    fac = pi / 180.0
    C   = pcl;  wspd = pws;  azw = paw;  wl = pwl

    def ji(k): return k + mu
    brdfint = np.zeros((2 * mu + 1, np_))

    # Solar zenith from rm[ji(0)]
    tetas = math.acos(max(-1.0, min(1.0, -rm[ji(0)]))) / fac   # rm(0) = -xmus

    # Whitecap reflectance
    W    = 2.95e-6 * wspd**3.52
    iwl  = 1 + int((wl - 0.2) / 0.1)
    iwl  = max(0, min(37, iwl))
    wlp  = 0.5 + (iwl - 1) * 0.1
    Ref_i = (_REF_WHITECAP[iwl + 1]
             + (wl - wlp) / 0.1 * (_REF_WHITECAP[iwl] - _REF_WHITECAP[iwl + 1]))
    Rwc  = W * Ref_i

    # Water properties
    nr, ni = indwat(wl, xsal)
    n12    = math.sqrt(nr**2 + ni**2)
    Rw     = morcasiwat(wl, C)

    # Downward transmission through glint
    tds = 1.0
    if Rw > 0.0001:
        nta = 24; nfa = 48
        ta, wta = gauss(0.0, pi / 2.0, nta)
        fa, wfa = gauss(0.0, 2.0 * pi, nfa)
        tds_sum = summ = 0.0
        for k in range(nfa):
            for j in range(nta):
                tp   = ta[j] / fac
                fip  = fa[k] / fac
                rogp = sunglint(wspd, nr, ni, azw, tetas, tp, fip)
                pond = math.cos(ta[j]) * math.sin(ta[j]) * wfa[k] * wta[j]
                summ    += pond
                tds_sum += rogp * pond
        tds = 1.0 - tds_sum / summ if summ > 0 else 1.0

    a = 0.485   # sub-surface albedo (Austin 1974)

    for j in range(1, mu + 1):
        tetav = math.acos(max(-1.0, min(1.0, rm[ji(j)]))) / fac

        # Upward transmission through glint
        tdv = 1.0
        if Rw > 0.0001:
            nta = 24; nfa = 48
            ta, wta = gauss(0.0, pi / 2.0, nta)
            fa, wfa = gauss(0.0, 2.0 * pi, nfa)
            tw = math.asin(max(-1.0, min(1.0, math.sin(tetav * fac) / nr))) / fac
            tdv_sum = summ = 0.0
            for n in range(nfa):
                for m in range(nta):
                    tp   = ta[m] / fac
                    fip  = fa[n] / fac
                    rogp = sunglint(wspd, 1.0 / nr, 0.0, azw, tw, tp, fip)
                    pond = math.cos(ta[m]) * math.sin(ta[m]) * wfa[n] * wta[m]
                    summ    += pond
                    tdv_sum += rogp * pond
            tdv = 1.0 - tdv_sum / summ if summ > 0 else 1.0

        for k in range(np_):
            fi = rm[ji(-mu)] if j == mu else (rp[k] + rm[ji(-mu)])
            while fi < 0.0:   fi += 2.0 * pi
            while fi > 2.0 * pi: fi -= 2.0 * pi
            fi_d = fi / fac

            # Sun-glint
            rog = sunglint(wspd, nr, ni, azw, tetas, tetav, fi_d)

            # Water body reflectance above surface
            Rwb = (1.0 / n12**2) * tds * tdv * Rw / (1.0 - a * Rw) if n12 > 0 else 0.0

            # Total ocean reflectance
            brdfint[ji(j), k] = Rwc + (1 - W) * rog + (1 - Rwc) * Rwb

    return brdfint


# ---------------------------------------------------------------------------
# OCEAALBE  — hemispherical albedo of the ocean
# ---------------------------------------------------------------------------

def oceaalbe(pws, paw, xsal, pcl, pwl):
    """
    Hemispherical (spherical) albedo of the ocean surface.

    Parameters
    ----------
    pws  : float – wind speed (m/s)
    paw  : float – azimuth of sun minus azimuth of wind (degrees)
    xsal : float – salinity (ppt); <0 → 34.3 ppt
    pcl  : float – pigment concentration (mg/m³)
    pwl  : float – wavelength (µm)

    Returns
    -------
    brdfalbe : float – spherical albedo
    """
    pi   = math.atan(1.0) * 4.0
    wl   = pwl;  wspd = pws;  azw = paw;  C = pcl

    # Whitecap
    W    = 2.95e-6 * wspd**3.52
    iwl  = 1 + int((wl - 0.2) / 0.1)
    iwl  = max(0, min(37, iwl))
    wlp  = 0.5 + (iwl - 1) * 0.1
    Ref_i = (_REF_WHITECAP[iwl + 1]
             + (wl - wlp) / 0.1 * (_REF_WHITECAP[iwl] - _REF_WHITECAP[iwl + 1]))
    Rwc  = W * Ref_i

    # Water
    nr, ni = indwat(wl, xsal)
    Rw     = morcasiwat(wl, C)

    # Glint spherical albedo
    rogalbe = glitalbe(wspd, nr, ni, azw)

    # Water body albedo above surface
    a    = 0.485
    Rwb  = (1.0 - rogalbe) * (1.0 - a) * Rw / (1.0 - a * Rw) if abs(1.0 - a * Rw) > 1e-10 else 0.0

    return Rwc + (1 - W) * rogalbe + (1 - Rwc) * Rwb
