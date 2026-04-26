"""
scattering.py
-------------
Translated from SCATRA.f, DISCOM.f, DISCRE.f

Scattering transmittance computation and discrete wavelength
atmospheric optical property tables.
"""

import math
import numpy as np

from .commons    import aer, disc, ffu, delta_sigma, trunc
from .utils      import odrayl
from .interp     import trunca
from .utils      import csalbr


# ---------------------------------------------------------------------------
# Forward declarations (filled in by importing modules)
# ---------------------------------------------------------------------------
# iso()     → defined in iso.py (imported at call time to avoid circular)
# atmref()  → defined in atmref.py
# ---------------------------------------------------------------------------


def discre(ta, ha, tr, hr, it, nt, yy, dd, ppp2, ppp1):
    """
    Find the altitude level zx such that the optical thickness from
    the surface to zx equals yy (used for plane-level layering).

    Parameters
    ----------
    ta, ha : float – aerosol OD and scale height
    tr, hr : float – Rayleigh OD and scale height
    it     : int   – iteration counter
    nt     : int   – total number of layers
    yy     : float – target cumulative OD
    dd     : float – fractional Rayleigh contribution
    ppp2   : float – lower altitude bound
    ppp1   : float – upper altitude bound

    Returns
    -------
    zx : float – altitude (km) where cumulative OD = yy
    """
    from .utils import print_error

    if ha >= 7.0:
        print_error('check aerosol measurements or plane altitude')
        return ppp1

    if it == 0:
        dt = 1.0e-17
    else:
        dt = 2.0 * (ta + tr - yy) / (nt - it + 1.0)

    while True:
        dt /= 2.0
        ti = yy + dt
        y1 = ppp2
        y3 = ppp1

        _niter = 0
        while True:
            _niter += 1
            y2 = (y1 + y3) * 0.5
            xx = -y2 / ha
            if xx < -18.0:
                x2 = tr * math.exp(-y2 / hr)
            else:
                x2 = ta * math.exp(xx) + tr * math.exp(-y2 / hr)

            xd = abs(ti - x2)
            if xd < 0.00001 or _niter >= 100:
                break
            if ti - x2 < 0:
                y3 = y2
            else:
                y1 = y2

        zx    = y2
        denom = 1.0 + ta * hr / tr / ha * math.exp((zx - ppp1) * (1.0 / hr - 1.0 / ha))
        delta = 1.0 / denom
        ecart = 0.0
        if dd != 0.0:
            ecart = abs((dd - delta) / dd)
        if ecart <= 0.75 or it == 0:
            break

    return zx


def scatra(taer, taerp, tray, trayp, piza,
           palt, nt, mu, rm, gb, xmus, xmuv):
    """
    Compute direct and diffuse transmittances (downward/upward) and
    spherical albedo for Rayleigh, aerosol, and combined atmosphere.

    Parameters
    ----------
    taer, taerp : float – aerosol OD (full atmosphere, above plane)
    tray, trayp : float – Rayleigh OD (full atmosphere, above plane)
    piza        : float – single-scatter albedo
    palt        : float – plane altitude (km; >900 = ground)
    nt, mu      : int   – number of layers / Gauss angles
    rm, gb      : array – Gauss nodes and weights
    xmus, xmuv  : float – cos(solar zenith), cos(view zenith)

    Returns
    -------
    dict with keys:
        ddirtt, ddiftt, udirtt, udiftt, sphalbt  (total)
        ddirtr, ddiftr, udirtr, udiftr, sphalbr  (Rayleigh)
        ddirta, ddifta, udirta, udifta, sphalba  (aerosol)
    """
    def ji(k): return k + mu
    from .iso import iso    # deferred import

    ddirtt = ddiftt = udirtt = udiftt = sphalbt = 1.0
    ddiftt = udiftt = sphalbt = 0.0
    ddirtr = ddiftr = udirtr = udiftr = sphalbr = 0.0
    ddirta = ddifta = udirta = udifta = sphalba = 0.0

    ddirtt = 1.0; ddiftt = 0.0; udirtt = 1.0; udiftt = 0.0; sphalbt = 0.0
    ddirtr = 1.0; ddiftr = 0.0; udirtr = 1.0; udiftr = 0.0; sphalbr = 0.0
    ddirta = 1.0; ddifta = 0.0; udirta = 1.0; udifta = 0.0; sphalba = 0.0

    for it in range(1, 4):
        if it == 2 and taer <= 0.0:
            continue

        if it == 1:
            if palt > 900.0:
                # Rayleigh-only, full atmosphere
                udiftt = ((2.0/3.0 + xmuv) + (2.0/3.0 - xmuv) * math.exp(-tray/xmuv)) \
                         / (4.0/3.0 + tray) - math.exp(-tray/xmuv)
                ddiftt = ((2.0/3.0 + xmus) + (2.0/3.0 - xmus) * math.exp(-tray/xmus)) \
                         / (4.0/3.0 + tray) - math.exp(-tray/xmus)
                ddirtt = math.exp(-tray / xmus)
                udirtt = math.exp(-tray / xmuv)
                sphalbt = csalbr(tray)
            else:
                tamol = 0.0; tamolp = 0.0
                rm_copy = rm.copy()
                rm_copy[ji(-mu)] = -xmuv; rm_copy[ji(mu)] = xmuv; rm_copy[ji(0)] = -xmus
                xtrans = iso(tamol, tray, piza, tamolp, trayp, palt, nt, mu, rm_copy, gb)
                udiftt = xtrans[-1] - math.exp(-trayp / xmuv)
                udirtt = math.exp(-trayp / xmuv)
                rm_copy[ji(-mu)] = -xmus; rm_copy[ji(mu)] = xmus; rm_copy[ji(0)] = -xmus
                ddiftt = ((2.0/3.0 + xmus) + (2.0/3.0 - xmus) * math.exp(-tray/xmus)) \
                         / (4.0/3.0 + tray) - math.exp(-tray/xmus)
                ddirtt = math.exp(-tray / xmus)
                udirtt = math.exp(-tray / xmuv)
                sphalbt = csalbr(tray)
            if palt <= 0.0:
                udiftt = 0.0; udirtt = 1.0
            ddirtr = ddirtt; ddiftr = ddiftt
            udirtr = udirtt; udiftr = udiftt; sphalbr = sphalbt

        elif it == 2:
            tamol = 0.0; tamolp = 0.0
            rm_copy = rm.copy()
            rm_copy[ji(-mu)] = -xmuv; rm_copy[ji(mu)] = xmuv; rm_copy[ji(0)] = -xmus
            xtrans = iso(taer, tamol, piza, taerp, tamolp, palt, nt, mu, rm_copy, gb)
            udiftt = xtrans[-1] - math.exp(-taerp / xmuv)
            udirtt = math.exp(-taerp / xmuv)
            rm_copy[ji(-mu)] = -xmus; rm_copy[ji(mu)] = xmus; rm_copy[ji(0)] = -xmus
            xtrans2 = iso(taer, tamol, piza, taerp, tamolp, 999.0, nt, mu, rm_copy, gb)
            ddirtt = math.exp(-taer / xmus)
            ddiftt = xtrans2[1] - math.exp(-taer / xmus)
            sphalbt = xtrans2[0] * 2.0
            if palt <= 0.0:
                udiftt = 0.0; udirtt = 1.0
            ddirta = ddirtt; ddifta = ddiftt
            udirta = udirtt; udifta = udiftt; sphalba = sphalbt

        else:  # it == 3
            rm_copy = rm.copy()
            rm_copy[ji(-mu)] = -xmuv; rm_copy[ji(mu)] = xmuv; rm_copy[ji(0)] = -xmus
            xtrans = iso(taer, tray, piza, taerp, trayp, palt, nt, mu, rm_copy, gb)
            udirtt = math.exp(-(taerp + trayp) / xmuv)
            udiftt = xtrans[-1] - math.exp(-(taerp + trayp) / xmuv)
            rm_copy[ji(-mu)] = -xmus; rm_copy[ji(mu)] = xmus; rm_copy[ji(0)] = -xmus
            xtrans2 = iso(taer, tray, piza, taerp, trayp, 999.0, nt, mu, rm_copy, gb)
            ddiftt = xtrans2[1] - math.exp(-(taer + tray) / xmus)
            ddirtt = math.exp(-(taer + tray) / xmus)
            sphalbt = xtrans2[0] * 2.0
            if palt <= 0.0:
                udiftt = 0.0; udirtt = 1.0

    return dict(
        ddirtt=ddirtt, ddiftt=ddiftt, udirtt=udirtt, udiftt=udiftt, sphalbt=sphalbt,
        ddirtr=ddirtr, ddiftr=ddiftr, udirtr=udirtr, udiftr=udiftr, sphalbr=sphalbr,
        ddirta=ddirta, ddifta=ddifta, udirta=udirta, udifta=udifta, sphalba=sphalba,
    )


def discom(idatmp, iaer, xmus, xmuv, phi,
           taer55, taer55p, palt,
           phirad, nt, mu, np_, rm, gb, rp,
           ftray, xlm1, xlm2):
    """
    Compute atmospheric optical properties at discrete wavelengths and
    store them in the /sixs_disc/ common for later interpolation.

    Parameters
    ----------
    idatmp           : int   – atmospheric profile flag
    iaer             : int   – aerosol model flag
    xmus, xmuv, phi  : float – geometry
    taer55, taer55p  : float – aerosol OD at 550 nm
    palt             : float – plane altitude (km)
    phirad           : float – azimuth (radians)
    nt, mu, np_      : int   – RT discretization parameters
    rm, gb, rp       : arrays
    ftray            : float – Rayleigh OD ratio
    xlm1, xlm2       : arrays – radiance fields (modified in place)
    """
    from .atmref import atmref   # deferred import

    ext   = aer.ext
    ome   = aer.ome

    roatm  = disc.roatm
    dtdir  = disc.dtdir
    dtdif  = disc.dtdif
    utdir  = disc.utdir
    utdif  = disc.utdif
    sphal  = disc.sphal
    wldis  = disc.wldis
    trayl  = disc.trayl
    traypl = disc.traypl

    s     = ffu.s
    wlinf = ffu.wlinf
    wlsup = ffu.wlsup

    pha   = trunc.pha
    betal = trunc.betal

    for l in range(10):        # l = 0..9  (Fortran l=1..10)
        wl = wldis[l]

        # Skip wavelengths outside the spectral band
        if wlsup < wldis[0] and l <= 1:
            pass   # always compute first two
        elif wlinf > wldis[9] and l >= 8:
            pass
        elif l < 9 and wldis[l] < wlinf and wldis[l+1] < wlinf:
            continue
        elif l > 0 and wldis[l] > wlsup and wldis[l-1] > wlsup:
            continue

        # Rayleigh optical depth
        tray  = odrayl(wl)
        if idatmp == 0:
            trayp = 0.0
        elif idatmp == 4:
            trayp = tray
        else:
            trayp = tray * ftray

        trayl[l]  = tray
        traypl[l] = trayp

        # Aerosol OD at this wavelength
        taer  = taer55  * ext[l] / ext[3]
        taerp = taer55p * ext[l] / ext[3]
        piza  = ome[l]

        # Truncation of phase function - copy phasel[l,:] into trunc.pha
        coeff = 0.0
        if iaer != 0:
            if hasattr(disc, 'phasel'):
                trunc.pha[:] = disc.phasel[l, :]
            coeff  = trunca()

        tamoy  = taer  * (1.0 - piza * coeff)
        tamoyp = taerp * (1.0 - piza * coeff)
        pizmoy = piza  * (1.0 - coeff) / (1.0 - piza * coeff) if piza * coeff < 1.0 else 0.0

        # Atmospheric reflectances
        rorayl, roaero, romix = atmref(
            iaer, tamoy, tray, pizmoy, tamoyp, trayp, palt,
            phi, xmus, xmuv, phirad, nt, mu, np_, rm, gb, rp, xlm1, xlm2
        )

        # Scattering transmittances
        st = scatra(tamoy, tamoyp, tray, trayp, pizmoy,
                    palt, nt, mu, rm, gb, xmus, xmuv)

        roatm[0, l] = rorayl
        roatm[1, l] = romix
        roatm[2, l] = roaero
        dtdir[0, l] = st['ddirtr'];  dtdif[0, l] = st['ddiftr']
        dtdir[1, l] = st['ddirtt'];  dtdif[1, l] = st['ddiftt']
        dtdir[2, l] = st['ddirta'];  dtdif[2, l] = st['ddifta']
        utdir[0, l] = st['udirtr'];  utdif[0, l] = st['udiftr']
        utdir[1, l] = st['udirtt'];  utdif[1, l] = st['udiftt']
        utdir[2, l] = st['udirta'];  utdif[2, l] = st['udifta']
        sphal[0, l] = st['sphalbr']
        sphal[1, l] = st['sphalbt']
        sphal[2, l] = st['sphalba']
