"""
sixs_main.py
------------
Translated from main.f  (program ssssss)

Second Simulation of the Satellite Signal in the Solar Spectrum (6S)
Version 4.1 — Python translation.

NOTE: This module preserves the Estonian variable comments from the
original customised main.f (m_korgus, m_h2o, m_o, m_aot, m_dir, m_dif,
m_dir2, m_dif2, m_seb, m_sb, m_paev, m_kuu — Estonian for altitude,
water, ozone, AOT, direct, diffuse, date/day/month).
"""

import math
import sys
import numpy as np

from .commons           import atm, delta_sigma, aer, disc, ffu, ier as ier_common, trunc
from .gauss             import gauss
from .utils             import varsol, odrayl, csalbr, solirr
from .possol            import possol
from .atm_profiles      import tropic, midsum, midwin, subsum, subwin, us62
from .pressure          import pressure, presplane
from .aeroso            import aeroso
from .oda550            import oda550
from .abstra            import abstra
from .interp            import equivwl, interp as interp_wl, trunca
from .scattering        import discom
from .specinterp        import specinterp, enviro
from .os_module         import os_sos
from .surface_reflectance import vegeta, sand, clearw, lakew
from .brdf_models       import (hapkbrdf, hapkalbe, versbrdf, versalbe,
                                 roujbrdf, roujalbe, waltbrdf, waltalbe,
                                 minnbrdf, minnalbe, rahmbrdf, rahmalbe,
                                 brdfgrid)
from .ocean_brdf        import oceabrdf, oceaalbe
from .iapi_brdf         import iapibrdf, iapialbe
from .aktool            import akbrdf, akalbe
from .sensors           import meteo, goes, avhrr, hrv, tm, mss, mas, modis, polder
from .geometry          import posge, posgw, posmto, posnoa, poslan, posspo


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NT_P  = 26
MU_P  = 25
MU2_P = 48
NP_P  = 49

_WLDISC = np.array([0.400, 0.488, 0.515, 0.550, 0.633,
                     0.694, 0.860, 1.536, 2.250, 3.750], dtype=np.float32)

_ANGMU  = np.array([85.0, 80.0, 70.0, 60.0, 50.0,
                     40.0, 30.0, 20.0, 10.0,  0.0], dtype=np.float64)
_ANGPHI = np.array([0., 30., 60., 90., 120., 150., 180.,
                     210., 240., 270., 300., 330., 360.], dtype=np.float64)


def _setup_gauss(mu, mu2, np_):
    """Set up Gauss quadrature nodes/weights for the RT integration."""
    pi   = math.acos(-1.0)
    pi2  = 2.0 * pi
    anglem, weightm = gauss(-1.0, 1.0, mu2)
    rp, gp          = gauss(0.0, pi2, np_)

    size  = 2 * mu + 1
    rm    = np.zeros(size)   # indexed -mu..+mu, stored offset by +mu
    gb    = np.zeros(size)

    def ji(k): return k + mu

    mum1 = mu - 1
    for j in range(-mum1, 0):
        k = mu + j
        rm[ji(-j - mu)] = anglem[k - 1]
        gb[ji(-j - mu)] = weightm[k - 1]
    for j in range(1, mum1 + 1):
        k = mum1 + j
        rm[ji(mu - j)] = anglem[k - 1]
        gb[ji(mu - j)] = weightm[k - 1]

    gb[ji(-mu)] = 0.0
    gb[ji(0)]   = 0.0
    gb[ji(mu)]  = 0.0

    return rm, gb, np.array(rp), np.array(gp)


def run6S(input_stream=None, output_stream=None):
    """
    Run the 6S radiative transfer model.

    Parameters
    ----------
    input_stream  : file-like or None – source of input parameters (default: stdin)
    output_stream : file-like or None – destination for output (default: stdout)

    Returns
    -------
    dict – see example_run.py for a full worked example of every key.

    Input metadata:
        day, month       acquisition date
        sza              solar zenith angle (degrees)
        h2o              precipitable water vapour (g cm⁻²)
        o3               ozone column (cm-atm)
        aot550           aerosol optical depth at 550 nm

    Geometry:
        asol/phi0/avis/phiv    solar/view zenith and azimuth (degrees)
        xmus/xmuv/xmud         cos(solar zenith)/cos(view zenith)/cos(scattering angle)

    Apparent reflectance (band-integrated, dimensionless):
        apparent_reflectance          TOA reflectance = π L_toa / (E₀ cos SZA)
        apparent_reflectance_atm_only path contribution only (black surface)
        apparent_reflectance_Rayleigh Rayleigh-only path contribution
        apparent_radiance             TOA radiance (W m⁻² sr⁻¹ µm⁻¹)

    Surface reflectance (retrieval mode only, rapp < 0):
        rog    retrieved surface reflectance

    Path reflectance (SOS decomposition):
        srotot  total = sroray + sroaer. Use as xa in retrieval formula.
                NOT the physical path refl. as seen by sensor (which is
                approximately chand(tau_R) + sroaer ~ 0.18 at 427 nm).
                Self-consistent with sdtott/sutott/sast.
        sroray  Rayleigh-aerosol coupling correction; negative in blue because
                aerosol forward-scattering reduces Rayleigh backscatter.
        sroaer  aerosol contribution to path reflectance

    Optical depths (band-integrated):
        sodray / sodaer / sodtot    Rayleigh / aerosol / total

    Spherical albedo (band-integrated):
        spherical_albedo_tot (sast)  TOTAL atmospheric spherical albedo.
                                     Use as 's' in retrieval denominator:
                                     rho_s=(rho_toa-xa)/(T_d*T_u+s*(rho_toa-xa))
        spherical_albedo_ray (sasr)  Rayleigh component
        spherical_albedo_aer (sasa)  aerosol component
        pizera   aerosol single-scatter albedo ω₀. NOT spherical albedo.

    Transmittances (band-integrated):
        sdtott/sutott         total downward/upward (Rayleigh + aerosol + gas)
        sdtotr/sutotr         Rayleigh component
        sdtota/sutota         aerosol component
        sdwava/suwava/stwava  water vapour down/up/total
        sdozon/suozon/stozon  ozone down/up/total
        tgasm/dgasm/ugasm     total/downward/upward gas transmittance

    Ground irradiances (W m⁻² µm⁻¹ at target surface):
        ground_direct_irr / ground_diffuse_irr / ground_env_irr
        ground_direct_fraction / ground_diffuse_fraction / ground_env_fraction

    Satellite radiances (W m⁻² sr⁻¹ µm⁻¹ and W m⁻²):
        atm_radiance / atm_radiance_wm2       atmospheric path
        env_radiance / env_radiance_wm2       environment
        target_radiance / target_radiance_wm2 target surface

    Band-integrated irradiances (W m⁻² µm⁻¹):
        direct_irr   direct beam (= ground_direct_irr)
        diffuse_irr  diffuse sky (= ground_diffuse_irr)

    Per-wavelength arrays:
        spec_wl / spec_dir / spec_dif  wavelength, direct, diffuse irradiances
    """
    _opened = False
    if input_stream is None:
        input_stream = sys.stdin
    elif isinstance(input_stream, str):
        input_stream = open(input_stream)
        _opened = True
    if output_stream is None:
        output_stream = sys.stdout

    def read(*args):
        """Read next non-blank, non-comment line and return parsed values.
        Strips full-line and inline comments: '#' (highest priority),
        then '(' and '!' (6S legacy comment styles)."""
        while True:
            line = input_stream.readline()
            if not line:
                raise EOFError("Unexpected end of input")
            # Strip comments using find() so '#' in a value is not mis-parsed
            for marker in ('#', '(', '!'):
                idx = line.find(marker)
                if idx >= 0:
                    line = line[:idx]
            vals = line.split()
            if vals:  # skip blank/comment-only lines
                break
        if len(args) == 0:
            return [_parse(v) for v in vals]
        # coerce to requested types
        out = []
        for v, t in zip(vals, args):
            out.append(t(v))
        return out if len(out) > 1 else out[0]

    def _parse(v):
        """Auto-parse a string to int or float."""
        try:
            return int(v)
        except ValueError:
            return float(v)

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------
    nt  = NT_P;  mu  = MU_P;  mu2 = MU2_P;  np_ = NP_P
    pi  = math.acos(-1.0);   pi2 = 2.0 * pi
    step  = 0.0025
    accu2 = 1.0e-3
    accu3 = 1.0e-7

    ier_common.ier = False
    ier_common.iwr = 6

    delta_sigma.sigma = 0.056032
    delta_sigma.delta = 0.0279

    # Discrete wavelengths
    wldis = _WLDISC.copy()
    disc.wldis[:] = wldis

    # Angle grids for BRDF interpolation
    angmu  = np.cos(_ANGMU  * pi / 180.0)
    angphi = _ANGPHI * pi / 180.0

    # Gauss quadrature
    rm, gb, rp, gp = _setup_gauss(mu, mu2, np_)

    # Radiance field arrays
    def ji(k): return k + mu
    xlm1      = np.zeros((2 * mu + 1, np_))
    xlm2      = np.zeros((2 * mu + 1, np_))
    xlmus     = np.zeros((2 * mu + 1, np_))
    xlmuv     = np.zeros((2 * mu + 1, np_))
    brdfints  = np.zeros((2 * mu + 1, np_))
    brdfintv  = np.zeros((2 * mu + 1, np_))
    brdfdats  = np.zeros((10, 13))
    brdfdatv  = np.zeros((10, 13))
    robar     = np.zeros(1501)
    robarp    = np.zeros(1501)
    robard    = np.zeros(1501)
    rocl      = np.zeros(1501)
    roel      = np.zeros(1501)
    s         = ffu.s
    s[:]      = 1.0

    iinf = 1;   isup = 1501
    idirec = 0; inhomo = 0; idatmp = 0

    # -----------------------------------------------------------------------
    # 1. Geometry
    # -----------------------------------------------------------------------
    igeom = read(int)
    asol = phi0 = avis = phiv = 0.0
    xlon = xlat = tu = 0.0
    month = jday = 1

    if igeom == 0:
        vals = read()
        asol, phi0, avis, phiv, month, jday = (float(vals[0]), float(vals[1]),
            float(vals[2]), float(vals[3]), int(vals[4]), int(vals[5]))
    elif igeom == 1:
        month, jday, tu, nc, nl = read(int, int, float, int, int)
        result = posmto(month, jday, tu, nc, nl)
        if result: asol, phi0, avis, phiv, xlon, xlat = result
    elif igeom == 2:
        month, jday, tu, nc, nl = read(int, int, float, int, int)
        result = posge(month, jday, tu, nc, nl)
        if result: asol, phi0, avis, phiv, xlon, xlat = result
    elif igeom == 3:
        month, jday, tu, nc, nl = read(int, int, float, int, int)
        result = posgw(month, jday, tu, nc, nl)
        if result: asol, phi0, avis, phiv, xlon, xlat = result
    elif igeom == 4:
        month, jday, tu, nc, xlonan, hna = read(int, int, float, int, float, float)
        result = posnoa(month, jday, tu, nc, xlonan, hna, 1.0)
        if result: asol, phi0, avis, phiv, xlon, xlat = result
    elif igeom == 5:
        month, jday, tu, nc, xlonan, hna = read(int, int, float, int, float, float)
        result = posnoa(month, jday, tu, nc, xlonan, hna, -1.0)
        if result: asol, phi0, avis, phiv, xlon, xlat = result
    elif igeom == 6:
        month, jday, tu, xlon, xlat = read(int, int, float, float, float)
        asol, phi0, avis, phiv = posspo(month, jday, tu, xlon, xlat)
    elif igeom == 7:
        month, jday, tu, xlon, xlat = read(int, int, float, float, float)
        asol, phi0, avis, phiv = poslan(month, jday, tu, xlon, xlat)

    if ier_common.ier:
        raise RuntimeError("Error in geometry computation")

    dsol = varsol(jday, month)

    # Scattering angle cosine
    phi    = abs(phiv - phi0)
    phirad = (phi0 - phiv) * pi / 180.0
    if phirad < 0.0:    phirad += pi2
    if phirad > pi2:    phirad -= pi2

    xmus  = math.cos(asol * pi / 180.0)
    xmuv  = math.cos(avis * pi / 180.0)
    xmup  = math.cos(phirad)
    xmud  = -xmus * xmuv - math.sqrt(max(0.0, 1.0 - xmus**2)) * math.sqrt(max(0.0, 1.0 - xmuv**2)) * xmup
    xmud  = max(-1.0, min(1.0, xmud))

    # -----------------------------------------------------------------------
    # 2. Atmospheric model
    # -----------------------------------------------------------------------
    uw = 0.0;  uo3 = 0.0
    idatm = read(int)
    if idatm == 8:
        uw, uo3 = read(float, float)
    elif idatm == 7:
        for k in range(34):
            vals = read()
            atm.z[k], atm.p[k], atm.t[k], atm.wh[k], atm.wo[k] = (
                float(vals[0]), float(vals[1]), float(vals[2]),
                float(vals[3]), float(vals[4]))
    else:
        {1: tropic, 2: midsum, 3: midwin, 4: subsum, 5: subwin, 6: us62}.get(idatm, lambda: None)()

    if idatm in (0, 8):
        us62()

    # -----------------------------------------------------------------------
    # 3. Aerosol model
    # -----------------------------------------------------------------------
    c = [0.0, 0.0, 0.0, 0.0]
    iaer = read(int)
    if iaer == 4:
        c = [float(v) for v in read()]
    elif iaer == 1:   c = [0.70, 0.29, 0.00, 0.01]
    elif iaer == 2:   c = [0.00, 0.05, 0.95, 0.00]
    elif iaer == 3:   c = [0.17, 0.61, 0.00, 0.22]

    file2 = ''
    iaerp = 0

    # Mie size distribution parameters (iaer 8-11)
    if 8 <= iaer <= 11:
        from .commons import mie_in
        import math as _math

        if iaer == 8:   # Log-normal: rmin rmax icp; per mode: x1 x2 cij rn(10) ri(10)
            mie_in.rmin, mie_in.rmax, mie_in.icp = read(float, float, int)
            for _i in range(mie_in.icp):
                mie_in.x1[_i], mie_in.x2[_i], mie_in.cij[_i] = read(float, float, float)
                mie_in.rn[:, _i] = read(*([float]*10))
                mie_in.ri[:, _i] = read(*([float]*10))

        elif iaer == 9:  # Modified gamma: rmin rmax / x1 x2 x3 / rn(10) / ri(10)
            mie_in.rmin, mie_in.rmax = read(float, float)
            mie_in.x1[0], mie_in.x2[0], mie_in.x3[0] = read(float, float, float)
            mie_in.rn[:, 0] = read(*([float]*10))
            mie_in.ri[:, 0] = read(*([float]*10))
            mie_in.icp = 1

        elif iaer == 10:  # Junge power-law: rmin rmax / x1 / rn(10) / ri(10)
            mie_in.rmin, mie_in.rmax = read(float, float)
            mie_in.x1[0] = read(float)
            mie_in.rn[:, 0] = read(*([float]*10))
            mie_in.ri[:, 0] = read(*([float]*10))
            mie_in.icp = 1

        elif iaer == 11:  # Sun photometer: irsunph / per point: r dV/dlogr / rn(10) / ri(10)
            mie_in.irsunph = read(int)
            for _i in range(mie_in.irsunph):
                _r, _nv = read(float, float)
                mie_in.rsunph[_i]  = _r
                # dV/dlogr → dn/dr  (same conversion as Fortran main.f line 46)
                mie_in.nrsunph[_i] = _nv / (_r**4) / _math.log(10.0)
            mie_in.rmin = mie_in.rsunph[0]
            mie_in.rmax = mie_in.rsunph[mie_in.irsunph - 1] + 1e-7
            mie_in.rn[:, 0] = read(*([float]*10))
            mie_in.ri[:, 0] = read(*([float]*10))
            mie_in.icp = 1

        # Optional: iaerp flag (just metadata in 6S, not needed for computation)
        iaerp = read(int)
        if iaerp == 1:
            file2 = input_stream.readline().strip() + '.mie'

    elif iaer == 12:
        file2 = input_stream.readline().strip()

    aeroso(iaer, c, xmud, wldis, file2)

    # Aerosol concentration (visibility or AOT)
    taer55 = 0.0
    v      = read(float)
    if abs(v) < accu2:
        taer55 = read(float)
        v = math.exp(-math.log(taer55 / 2.7628) / 0.79902)
    elif v > 0:
        taer55 = oda550(iaer, v)
    # v < 0 means iaer == 0

    # -----------------------------------------------------------------------
    # 4. Surface altitude
    # -----------------------------------------------------------------------
    uwus = 1.424;  uo3us = 0.344
    xps  = read(float)
    if xps >= 0.0:
        xps = 0.0
    else:
        if idatm != 8:
            uw, uo3 = pressure(xps)
        else:
            uwus, uo3us = pressure(xps)

    # -----------------------------------------------------------------------
    # 5. Sensor altitude
    # -----------------------------------------------------------------------
    palt = 1000.0;  pps = 0.0;  ftray = 1.0
    taer55p = taer55;  puw = 0.0;  puo3 = 0.0
    puwus = uwus;  puo3us = uo3us

    xpp = read(float)
    xpp = -xpp

    if xpp <= 0.0:
        palt    = 0.0
        pps     = atm.p[0]
        idatmp  = 0
        taer55p = 0.0
        puw     = 0.0
        puo3    = 0.0
    elif xpp >= 100.0:
        palt    = 1000.0
        pps     = 0.0
        taer55p = taer55
        ftray   = 1.0
        idatmp  = 4
    else:
        puw, puo3 = read(float, float)
        if puw < 0.0:
            puw, puo3, xpp, ftray = presplane(xpp)
            idatmp = 2
            if idatm == 8:
                puwus = puw;  puo3us = puo3
                puw   = puw * uw / uwus
                puo3  = puo3 * uo3 / uo3us
                idatmp = 8
        else:
            puwus, puo3us, xpp, ftray = presplane(xpp)
            idatmp = 8
        if ier_common.ier:
            raise RuntimeError("Error in plane altitude calculation")

        from .commons import planesim
        palt = planesim.zpl[33] - atm.z[0]
        pps  = planesim.ppl[33]

        taer55p = read(float)
        if taer55p < 0.0 or (taer55 - taer55p) < accu2:
            taer55p = taer55 * (1.0 - math.exp(-palt / 2.0))
        else:
            sham = math.exp(-palt / 4.0)
            sha  = 1.0 - taer55p / taer55
            if sha >= sham:
                taer55p = taer55 * (1.0 - math.exp(-palt / 4.0))
            else:
                sha_km  = -palt / math.log(sha)
                taer55p = taer55 * (1.0 - math.exp(-palt / sha_km))

    # -----------------------------------------------------------------------
    # 6. Spectral conditions
    # -----------------------------------------------------------------------
    iwave = read(int)
    iinf  = 1;  isup = 1501;  wl_mono = 0.0

    if iwave == -2:
        wlinf, wlsup = read(float, float)
        ffu.wlinf = wlinf;  ffu.wlsup = wlsup
    elif iwave == -1:
        wl_mono = read(float)
        ffu.wlinf = wl_mono;  ffu.wlsup = wl_mono
    elif iwave == 0:
        wlinf, wlsup = read(float, float)
        ffu.wlinf = wlinf;  ffu.wlsup = wlsup
    elif iwave == 1:
        wlinf, wlsup = read(float, float)
        ffu.wlinf = wlinf;  ffu.wlsup = wlsup
        iinf_r = int((wlinf - 0.25) / 0.0025 + 1.5)
        isup_r = int((wlsup - 0.25) / 0.0025 + 1.5)
        s[iinf_r - 1:isup_r] = 0.0
        vals = [float(v) for v in input_stream.readline().split()]
        for idx, v in enumerate(vals):
            if iinf_r - 1 + idx < isup_r:
                s[iinf_r - 1 + idx] = v
    elif iwave == 2:  meteo()
    elif iwave in (3, 4):   goes(iwave - 2)
    elif 5 <= iwave <= 16:  avhrr(iwave - 4 - 1)
    elif 17 <= iwave <= 24: hrv(iwave - 17)
    elif 25 <= iwave <= 30: tm(iwave - 25)
    elif 31 <= iwave <= 34: mss(iwave - 31)
    elif 35 <= iwave <= 41: mas(iwave - 35)
    elif 42 <= iwave <= 48: modis(iwave - 42)
    elif iwave in (49, 50): avhrr(iwave - 49 + 12)
    elif iwave in (51, 52): avhrr(iwave - 51 + 14)
    elif 53 <= iwave <= 60: polder(iwave - 53)

    iinf = int((ffu.wlinf - 0.25) / 0.0025 + 1.5)
    isup = int((ffu.wlsup - 0.25) / 0.0025 + 1.5)

    # -----------------------------------------------------------------------
    # 7. Compute equivalent wavelength and atmospheric properties
    # -----------------------------------------------------------------------
    if iwave != -1:
        wlmoy = equivwl(iinf, isup, step)
    else:
        wlmoy = wl_mono

    discom(idatmp, iaer, xmus, xmuv, phi,
           taer55, taer55p, palt,
           phirad, nt, mu, np_, rm, gb, rp,
           ftray, xlm1, xlm2)

    tamoy = tamoyp = pizmoy = pizmoyp = 0.0
    if iaer != 0:
        tamoy, tamoyp, pizmoy, pizmoyp = specinterp(wlmoy, taer55, taer55p)

    trmoy  = odrayl(wlmoy)
    trmoyp = trmoy * ftray
    if idatmp == 4:
        trmoyp = trmoy;  tamoyp = tamoy
    if idatmp == 0:
        trmoyp = 0.0;    tamoyp = 0.0

    # -----------------------------------------------------------------------
    # 8. Ground reflectance
    # -----------------------------------------------------------------------
    rad    = 0.0
    fr     = 0.0
    idirec = 0
    albbrdf = 0.0

    inhomo = read(int)

    if inhomo == 0:
        idirec = read(int)

        if idirec == 1:
            # BRDF conditions
            rm[ji(-mu)] = -xmuv;  rm[ji(mu)] = xmuv;  rm[ji(0)] = -xmus
            os_sos(tamoy, trmoy, pizmoy, tamoyp, trmoyp, 1000.0,
                   phirad, nt, mu, np_, rm, gb, rp, xlmus)

            if idatmp != 0:
                rm[ji(-mu)] = -xmus;  rm[ji(mu)] = xmus;  rm[ji(0)] = -xmuv
                os_sos(tamoyp, trmoyp, pizmoy, tamoyp, trmoyp, 1000.0,
                       phirad, nt, mu, np_, rm, gb, rp, xlmuv)

            ibrdf = read(int)

            if ibrdf == 0:
                for k in range(13):
                    row = [float(v) for v in input_stream.readline().split()]
                    for j in range(10):
                        brdfdats[9 - j, k] = row[j]
                for k in range(13):
                    row = [float(v) for v in input_stream.readline().split()]
                    for j in range(10):
                        brdfdatv[9 - j, k] = row[j]
                albbrdf = read(float)
                rodir   = read(float)
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = brdfgrid(mu, np_, rm, rp, brdfdats, angmu, angphi)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = brdfgrid(mu, np_, rm, rp, brdfdatv, angmu, angphi)
                brdfints[ji(mu), 0] = rodir
            elif ibrdf == 1:
                par1, par2, par3, par4 = read(float, float, float, float)
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = hapkbrdf(par1, par2, par3, par4, mu, np_, rm, rp)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = hapkbrdf(par1, par2, par3, par4, mu, np_, rm, rp)
                albbrdf  = hapkalbe(par1, par2, par3, par4)
            elif ibrdf == 2:
                opt345 = [int(v) for v in input_stream.readline().split()]
                options = [1, 1] + opt345
                struct  = [float(v) for v in input_stream.readline().split()]
                optics  = [float(v) for v in input_stream.readline().split()]
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = versbrdf(options, optics, struct, mu, np_, rm, rp)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = versbrdf(options, optics, struct, mu, np_, rm, rp)
                albbrdf  = versalbe(options, optics, struct)
            elif ibrdf == 3:
                par1, par2, par3 = read(float, float, float)
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = roujbrdf(par1, par2, par3, mu, np_, rm, rp)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = roujbrdf(par1, par2, par3, mu, np_, rm, rp)
                albbrdf  = roujalbe(par1, par2, par3)
            elif ibrdf == 4:
                par1, par2, par3, par4 = read(float, float, float, float)
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = waltbrdf(par1, par2, par3, par4, mu, np_, rm, rp)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = waltbrdf(par1, par2, par3, par4, mu, np_, rm, rp)
                albbrdf  = waltalbe(par1, par2, par3, par4)
            elif ibrdf == 5:
                par1, par2 = read(float, float)
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = minnbrdf(par1, par2, mu, np_, rm)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = minnbrdf(par1, par2, mu, np_, rm)
                albbrdf  = minnalbe(par1, par2)
            elif ibrdf == 6:
                # Ocean BRDF: pws, phi_wind, xsal, pcl
                pws, phi_wind, xsal_v, pcl = read(float, float, float, float)
                if xsal_v < 0.001: xsal_v = 34.3
                paw = phi0 - phi_wind
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = oceabrdf(pws, paw, xsal_v, pcl, wlmoy, mu, np_, rm, rp)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = oceabrdf(pws, paw, xsal_v, pcl, wlmoy, mu, np_, rm, rp)
                albbrdf  = oceaalbe(pws, paw, xsal_v, pcl, wlmoy)

            elif ibrdf == 7:
                # Iaquinta-Pinty canopy BRDF
                pild_v, pihs_v = read(int, int)
                pxLt_v, pc_v   = read(float, float)
                pRl_v, pTl_v, pRs_v = read(float, float, float)
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = iapibrdf(pild_v, pxLt_v, pRl_v, pTl_v, pRs_v, pihs_v, pc_v, mu, np_, rm, rp)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = iapibrdf(pild_v, pxLt_v, pRl_v, pTl_v, pRs_v, pihs_v, pc_v, mu, np_, rm, rp)
                albbrdf  = iapialbe(pild_v, pxLt_v, pRl_v, pTl_v, pRs_v, pihs_v, pc_v)

            elif ibrdf == 8:
                par1, par2, par3 = read(float, float, float)
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = rahmbrdf(par1, par2, par3, mu, np_, rm, rp)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = rahmbrdf(par1, par2, par3, mu, np_, rm, rp)
                albbrdf  = rahmalbe(par1, par2, par3)

            elif ibrdf == 9:
                # Kuusk canopy BRDF:
                # line1: ee thm ul sl rsl1
                # line2: wlmoy rnc cab cw vai
                ee_v, thm_v, ul_v, sl_v, rsl1_v = read(float, float, float, float, float)
                rnc_v, cab_v, cw_v, vai_v        = read(float, float, float, float)
                rm[ji(-mu)] = phirad;  rm[ji(mu)] = xmuv;  rm[ji(0)] = xmus
                brdfints = akbrdf(ee_v, thm_v, ul_v, sl_v, rsl1_v, wlmoy,
                                  rnc_v, cab_v, cw_v, vai_v, mu, np_, rm, rp)
                rm[ji(-mu)] = pi2 - phirad;  rm[ji(mu)] = xmus;  rm[ji(0)] = xmuv
                brdfintv = akbrdf(ee_v, thm_v, ul_v, sl_v, rsl1_v, wlmoy,
                                  rnc_v, cab_v, cw_v, vai_v, mu, np_, rm, rp)
                albbrdf  = akalbe(ee_v, thm_v, ul_v, sl_v, rsl1_v, wlmoy,
                                  rnc_v, cab_v, cw_v, vai_v)

            # Compute BRDF-weighted reflectances
            robar1 = xnorm1 = 0.0
            for j_idx in range(np_):
                rob = xnor = 0.0
                for k in range(1, mu):
                    rdown = xlmus[ji(-k), j_idx]
                    rdir  = brdfintv[ji(k), j_idx]
                    rob  += rdown * rdir  * rm[ji(k)] * gb[ji(k)]
                    xnor += rdown         * rm[ji(k)] * gb[ji(k)]
                robar1 += rob  * gp[j_idx]
                xnorm1 += xnor * gp[j_idx]

            robar2 = xnorm2 = 0.0
            for j_idx in range(np_):
                rob = xnor = 0.0
                for k in range(1, mu):
                    rdown = xlmuv[ji(-k), j_idx]
                    rdir  = brdfints[ji(k), j_idx]
                    rob  += rdown * rdir  * rm[ji(k)] * gb[ji(k)]
                    xnor += rdown         * rm[ji(k)] * gb[ji(k)]
                robar2 += rob  * gp[j_idx]
                xnorm2 += xnor * gp[j_idx]

            for l in range(iinf - 1, isup):
                rocl[l]  = brdfints[ji(mu), 0]
                roel[l]  = brdfints[ji(mu), 0]
                robar[l] = robar1 / xnorm1 if xnorm1 else 0.0
                robarp[l] = (robar2 / xnorm2 if idatmp != 0 and xnorm2 else 0.0)
                robard[l] = albbrdf

        else:
            # Lambertian surface
            igroun = read(int)
            if igroun == -1:
                vals = [float(v) for v in input_stream.readline().split()]
                for idx, v in enumerate(vals):
                    if iinf - 1 + idx < isup:
                        rocl[iinf - 1 + idx] = v
            elif igroun == 0:
                ro = read(float)
                rocl[iinf - 1:isup] = ro
            else:
                tmp = {'1': vegeta, '2': clearw, '3': sand, '4': lakew}.get(str(igroun))
                if tmp:
                    rocl[:] = tmp()
            roel[:] = rocl

    else:
        # Non-uniform surface
        igrou1, igrou2, rad = read(int, int, float)

        def _load_reflectance(igroun, arr):
            if igroun == -1:
                vals = [float(v) for v in input_stream.readline().split()]
                for idx, v in enumerate(vals):
                    if iinf - 1 + idx < isup:
                        arr[iinf - 1 + idx] = v
            elif igroun == 0:
                roc_v = read(float)
                arr[iinf - 1:isup] = roc_v
            else:
                src = {'1': vegeta, '2': clearw, '3': sand, '4': lakew}.get(str(igroun))
                if src: arr[:] = src()

        _load_reflectance(igrou1, rocl)
        _load_reflectance(igrou2, roel)

    # -----------------------------------------------------------------------
    # 9. Atmospheric correction flag
    # -----------------------------------------------------------------------
    rapp = read(float)

    # -----------------------------------------------------------------------
    # 10. Estonian custom variables (from m_* variables in main.f)
    # -----------------------------------------------------------------------
    m_paev   = jday;   m_kuu    = month
    m_korgus = asol    # solar zenith angle
    m_h2o    = uw;     m_o      = uo3;   m_aot = taer55

    # -----------------------------------------------------------------------
    # 11. Spectral integration loop
    # -----------------------------------------------------------------------
    # Pre-compute solar-weighted average surface reflectance
    rocave = roeave = seb_pre = 0.0
    for l in range(iinf - 1, isup):
        sbor = s[l]
        if l == iinf - 1 or l == isup - 1:
            sbor *= 0.5
        wl_curr = 0.25 + l * step
        swl = solirr(wl_curr) * dsol
        coef = sbor * step * swl
        rocave += rocl[l] * coef
        roeave += roel[l] * coef
        seb_pre += coef
    if seb_pre > 0:
        rocave /= seb_pre;   roeave /= seb_pre

    # Accumulators
    sb = seb = 0.0
    refet = refet1 = refet2 = refet3 = alumet = tgasm = rog = dgasm = ugasm = 0.0
    sdwava = sdozon = sddica = sdoxyg = sdniox = sdmoca = sdmeth = 0.0
    suwava = suozon = sudica = suoxyg = suniox = sumoca = sumeth = 0.0
    stwava = stozon = stdica = stoxyg = stniox = stmoca = stmeth = 0.0
    sodray = sodaer = sodtot = sodrayp = sodaerp = sodtotp = 0.0
    fophsr = fophsa = sroray = sroaer = srotot = ssdaer = 0.0
    sdtotr = sdtota = sdtott = sutotr = sutota = sutott = 0.0
    sasr = sasa = sast = 0.0

    ani  = [[0.0] * 3 for _ in range(2)]
    anr  = [[0.0] * 3 for _ in range(2)]
    aini = [[0.0] * 3 for _ in range(2)]
    ainr = [[0.0] * 3 for _ in range(2)]

    # per-wavelength spectral irradiance output (Matti)
    spec_wl  = []   # wavelength (µm)
    spec_dir = []   # direct beam irradiance at each wavelength (W m⁻² µm⁻¹)
    spec_dif = []   # diffuse + env irradiance at each wavelength (W m⁻² µm⁻¹)
    # Note: these are spectral densities, not band integrals.
    # m_dir / m_dif are the corresponding filter-weighted band integrals (W m⁻²).

    # Spectral loop
    for l in range(iinf - 1, isup):
        sbor = s[l]
        if l == iinf - 1 or l == isup - 1:
            sbor *= 0.5
        if iwave == -1:
            sbor = 1.0 / step

        roc = rocl[l]
        roe = roel[l]
        wl_curr = 0.25 + l * step

        # Gas transmittances (half-path first for continuum)
        gt_half = abstra(idatm, wl_curr, xmus, xmuv, uw / 2.0, uo3, uwus, uo3us,
                         idatmp, puw / 2.0, puo3, puwus, puo3us)
        gt_full = abstra(idatm, wl_curr, xmus, xmuv, uw, uo3, uwus, uo3us,
                         idatmp, puw, puo3, puwus, puo3us)

        # Clamp
        for key in gt_full:
            if gt_full[key] < accu3:
                gt_full[key] = 0.0

        attwava = gt_half['ttwava']
        dtwava  = gt_full['dtwava'];   utwava = gt_full['utwava'];  ttwava = gt_full['ttwava']
        dtozon  = gt_full['dtozon'];   utozon = gt_full['utozon'];  ttozon = gt_full['ttozon']
        dtdica  = gt_full['dtdica'];   utdica = gt_full['utdica'];  ttdica = gt_full['ttdica']
        dtoxyg  = gt_full['dtoxyg'];   utoxyg = gt_full['utoxyg'];  ttoxyg = gt_full['ttoxyg']
        dtniox  = gt_full['dtniox'];   utniox = gt_full['utniox'];  ttniox = gt_full['ttniox']
        dtmeth  = gt_full['dtmeth'];   utmeth = gt_full['utmeth'];  ttmeth = gt_full['ttmeth']
        dtmoca  = gt_full['dtmoca'];   utmoca = gt_full['utmoca'];  ttmoca = gt_full['ttmoca']

        swl  = solirr(wl_curr) * dsol
        coef = sbor * step * swl

        # Atmospheric scattering interpolation
        res = interp_wl(iaer, idatmp, wl_curr, taer55, taer55p, xmud)
        romix = res['romix'];    rorayl = res['rorayl'];  roaero = res['roaero']
        phaa  = res['phaa'];     phar   = res['phar'];    tsca   = res['tsca']
        tray  = res['tray'];     trayp  = res['trayp'];   taer   = res['taer']
        taerp = res['taerp'];    dtott  = res['dtott'];   utott  = res['utott']
        astot = res['astot'];    asray  = res['asray'];   asaer  = res['asaer']
        utotr = res['utotr'];    utota  = res['utota'];   dtotr  = res['dtotr']
        dtota = res['dtota']

        dgtot = dtwava * dtozon * dtdica * dtoxyg * dtniox * dtmeth * dtmoca
        tgtot = ttwava * ttozon * ttdica * ttoxyg * ttniox * ttmeth * ttmoca
        ugtot = utwava * utozon * utdica * utoxyg * utniox * utmeth * utmoca
        tgp1  = ttozon * ttdica * ttoxyg * ttniox * ttmeth * ttmoca
        tgp2  = attwava * ttozon * ttdica * ttoxyg * ttniox * ttmeth * ttmoca

        edifr = utotr - math.exp(-trayp / xmuv) if xmuv > 0 else 0.0
        edifa = utota - math.exp(-taerp / xmuv) if xmuv > 0 else 0.0

        if idirec == 1:
            tdird  = math.exp(-(trayp + taerp) / xmus) if xmus > 0 else 0.0
            tdiru  = math.exp(-(trayp + taerp) / xmuv) if xmuv > 0 else 0.0
            tdifd  = dtott - tdird
            tdifu  = utott - tdiru
            avr    = robard[l]
            denom  = 1.0 - astot * avr
            if abs(denom) > 1e-10:
                rsurf = (roc * tdird * tdiru +
                         robar[l] * tdifd * tdiru +
                         robarp[l] * tdifu * tdird +
                         robard[l] * tdifd * tdifu +
                         (tdifd + tdird) * (tdifu + tdiru) * astot * avr**2 / denom)
            else:
                rsurf = 0.0
        else:
            fra, fae, fr = enviro(edifr, edifa, rad, palt, xmuv)
            avr   = roc * fr + (1.0 - fr) * roe
            denom = 1.0 - avr * astot
            if abs(denom) > 1e-10:
                rsurf = (roc * dtott * math.exp(-(trayp + taerp) / xmuv) / denom
                         + avr * dtott * (utott - math.exp(-(trayp + taerp) / xmuv)) / denom)
            else:
                rsurf = 0.0

        ratm1 = (romix - rorayl) * tgtot + rorayl * tgp1
        ratm3 = romix * tgp1
        ratm2 = (romix - rorayl) * tgp2 + rorayl * tgp1

        romeas1 = ratm1 + rsurf * tgtot
        romeas2 = ratm2 + rsurf * tgtot
        romeas3 = ratm3 + rsurf * tgtot
        alumeas = xmus * swl * romeas2 / pi

        # Accumulate
        fophsa += phaa * coef;    fophsr += phar * coef
        sasr   += asray * coef;   sasa   += asaer * coef;  sast += astot * coef
        sroray += rorayl * coef;  sroaer += roaero * coef
        sodray += tray * coef;    sodaer += taer * coef
        sodrayp+= trayp * coef;   sodaerp+= taerp * coef
        ssdaer += tsca * coef;    sodtot += (taer + tray) * coef
        sodtotp+= (taerp + trayp) * coef
        srotot += romix * coef
        rog    += roc * coef;     refet  += romeas2 * coef
        refet1 += romeas1 * coef; refet2 += romeas2 * coef; refet3 += romeas3 * coef
        alumet += alumeas * sbor * step
        tgasm  += tgtot * coef;   dgasm  += dgtot * coef;   ugasm += ugtot * coef
        sdwava += dtwava * coef;  sdozon += dtozon * coef;  sddica += dtdica * coef
        sdoxyg += dtoxyg * coef;  sdniox += dtniox * coef
        sdmeth += dtmeth * coef;  sdmoca += dtmoca * coef
        suwava += utwava * coef;  suozon += utozon * coef;  sudica += utdica * coef
        suoxyg += utoxyg * coef;  suniox += utniox * coef
        sumeth += utmeth * coef;  sumoca += utmoca * coef
        stwava += ttwava * coef;  stozon += ttozon * coef;  stdica += ttdica * coef
        stoxyg += ttoxyg * coef;  stniox += ttniox * coef
        stmeth += ttmeth * coef;  stmoca += ttmoca * coef
        sdtotr += dtotr * coef;   sdtota += dtota * coef;  sdtott += dtott * coef
        sutotr += utotr * coef;   sutota += utota * coef;  sutott += utott * coef
        sb     += sbor * step;    seb    += coef

        # Ground-level irradiances
        tdir  = math.exp(-(tray + taer) / xmus) if xmus > 0 else 0.0
        tdif  = dtott - tdir
        etn   = dtott * dgtot / (1.0 - avr * astot) if abs(1.0 - avr * astot) > 1e-10 else 0.0
        esn   = tdir * dgtot
        es    = tdir * dgtot * xmus * swl
        ea0n  = tdif * dgtot
        ea0   = tdif * dgtot * xmus * swl
        ee0n  = dgtot * avr * astot * dtott / (1.0 - avr * astot) if abs(1.0 - avr * astot) > 1e-10 else 0.0
        ee0   = xmus * swl * ee0n

        if etn > accu3:
            ani[0] = [esn / etn, ea0n / etn, ee0n / etn]
        else:
            ani[0] = [0.0, 0.0, 0.0]
        ani[1] = [es, ea0, ee0]

        for j in range(3):
            aini[0][j] += ani[0][j] * coef
            aini[1][j] += ani[1][j] * sbor * step

        # store spectral irradiance at this wavelength (W m⁻² µm⁻¹)
        # es     = tdir * dgtot * xmus * swl    (direct beam on horizontal surface)
        # ea0    = tdif * dgtot * xmus * swl    (diffuse sky)
        # ee0    = ee0n * xmus * swl            (env: surface-reflected back down)
        # Note: m_dir = Σ(es * sbor * Δλ)  is the band-integrated version [W m⁻²]
        spec_wl.append(wl_curr)
        spec_dir.append(es)           # direct beam irradiance  (W m⁻² µm⁻¹)
        spec_dif.append(ea0 + ee0)    # diffuse + env irradiance (W m⁻² µm⁻¹)

        # Satellite-level radiances
        tmdir = math.exp(-(tray + taer) / xmuv) if xmuv > 0 else 0.0
        tmdif = utott - tmdir
        xla0n = ratm2
        xla0  = xla0n * xmus * swl / pi
        xltn  = roc * dtott * tmdir * tgtot / (1.0 - avr * astot) if abs(1.0 - avr * astot) > 1e-10 else 0.0
        xlt   = xltn * xmus * swl / pi
        xlen  = avr * dtott * tmdif * tgtot / (1.0 - avr * astot) if abs(1.0 - avr * astot) > 1e-10 else 0.0
        xle   = xlen * xmus * swl / pi

        anr[0] = [xla0n, xlen, xltn]
        anr[1] = [xla0,  xle,  xlt]
        for j in range(3):
            ainr[0][j] += anr[0][j] * coef
            ainr[1][j] += anr[1][j] * sbor * step

    # -----------------------------------------------------------------------
    # 12. Normalise integrated quantities
    # -----------------------------------------------------------------------
    def _div(x): return x / seb if seb > 0 else 0.0
    def _divb(x): return x / sb if sb > 0 else 0.0

    refet   = _div(refet);   refet1 = _div(refet1);  refet2 = _div(refet2); refet3 = _div(refet3)
    tgasm   = _div(tgasm);   dgasm  = _div(dgasm);   ugasm  = _div(ugasm)
    sasr    = _div(sasr);    sasa   = _div(sasa);    sast   = _div(sast)
    sdwava  = _div(sdwava);  sdozon = _div(sdozon);  sddica = _div(sddica)
    sdoxyg  = _div(sdoxyg);  sdniox = _div(sdniox);  sdmeth = _div(sdmeth); sdmoca = _div(sdmoca)
    suwava  = _div(suwava);  suozon = _div(suozon);  sudica = _div(sudica)
    suoxyg  = _div(suoxyg);  suniox = _div(suniox);  sumeth = _div(sumeth); sumoca = _div(sumoca)
    stwava  = _div(stwava);  stozon = _div(stozon);  stdica = _div(stdica)
    stoxyg  = _div(stoxyg);  stniox = _div(stniox);  stmeth = _div(stmeth); stmoca = _div(stmoca)
    sdtotr  = _div(sdtotr);  sdtota = _div(sdtota);  sdtott = _div(sdtott)
    sutotr  = _div(sutotr);  sutota = _div(sutota);  sutott = _div(sutott)
    rog     = _div(rog);     sroray = _div(sroray);  sroaer = _div(sroaer); srotot = _div(srotot)
    alumet  = _divb(alumet)
    pizera  = ssdaer / sodaer if sodaer > 0 and iaer != 0 else 0.0
    sodray  = _div(sodray);  sodaer = _div(sodaer);  sodtot = _div(sodtot)
    sodrayp = _div(sodrayp); sodaerp= _div(sodaerp); sodtotp= _div(sodtotp)
    fophsa  = _div(fophsa);  fophsr = _div(fophsr)

    m_dir   = aini[1][0]
    m_dif   = aini[1][1] + aini[1][2]
    m_sb    = sb;   m_seb = seb

    for j in range(3):
        aini[0][j] = _div(aini[0][j])
        ainr[0][j] = _div(ainr[0][j])
        aini[1][j] = _divb(aini[1][j])
        ainr[1][j] = _divb(ainr[1][j])

    # -----------------------------------------------------------------------
    # 13. Output (Estonian Matti output format preserved)
    # -----------------------------------------------------------------------
    print(f"{'mm':>4} {'dd':>4} {'h_s':>8} {'H2O':>8} {'O3':>8} {'aot':>8} "
          f"{'dir':>12} {'dif':>12} {'seb':>12}", file=output_stream)
    print(f"{m_kuu:4d} {m_paev:4d} {m_korgus:8.3f} {m_h2o:8.4f} {m_o:8.4f} {m_aot:8.4f} "
          f"{m_dir:12.4e} {m_dif:12.4e} {seb:12.4e}", file=output_stream)

    if _opened:
        input_stream.close()

    # -----------------------------------------------------------------------
    return {
        # Input metadata (m_* variable names are Estonian legacy from main.f)
        'day':      m_paev,   'month':   m_kuu,
        'sza':      m_korgus, 'h2o':     m_h2o,
        'o3':       m_o,      'aot550':  m_aot,
        # Per-wavelength spectral irradiance arrays
        'spec_wl':  spec_wl,  'spec_dir': spec_dir, 'spec_dif': spec_dif,
        # Band-integrated direct and diffuse irradiance (W m⁻² µm⁻¹)
        # = filter-weighted integrals of spec_dir and spec_dif
        'direct_irr':  m_dir, 'diffuse_irr': m_dif,
        # Geometry
        'asol': asol, 'phi0': phi0, 'avis': avis, 'phiv': phiv,
        'xmus': xmus, 'xmuv': xmuv, 'xmud': xmud,
        # Apparent reflectance
        'apparent_reflectance': refet2,
        'apparent_reflectance_atm_only': refet1,
        'apparent_reflectance_Rayleigh': refet3,
        'apparent_radiance':   alumet,
        # Gas transmittances
        'tgasm': tgasm, 'dgasm': dgasm, 'ugasm': ugasm,
        'sdwava': sdwava, 'suwava': suwava, 'stwava': stwava,
        'sdozon': sdozon, 'suozon': suozon, 'stozon': stozon,
        # Aerosol / Rayleigh
        'sodray': sodray, 'sodaer': sodaer, 'sodtot': sodtot,
        'sroray': sroray, 'sroaer': sroaer, 'srotot': srotot,
        'spherical_albedo_ray': sasr, 'spherical_albedo_aer': sasa,
        'spherical_albedo_tot': sast,
        # Transmittances
        'sdtotr': sdtotr, 'sdtota': sdtota, 'sdtott': sdtott,
        'sutotr': sutotr, 'sutota': sutota, 'sutott': sutott,
        # Ground irradiances
        'ground_direct_fraction':  aini[0][0],
        'ground_diffuse_fraction': aini[0][1],
        'ground_env_fraction':     aini[0][2],
        'ground_direct_irr':  aini[1][0],
        'ground_diffuse_irr': aini[1][1],
        'ground_env_irr':     aini[1][2],
        # Satellite radiances
        'atm_radiance':     ainr[0][0],
        'env_radiance':     ainr[0][1],
        'target_radiance':  ainr[0][2],
        'atm_radiance_wm2': ainr[1][0],
        'env_radiance_wm2': ainr[1][1],
        'target_radiance_wm2': ainr[1][2],
        # Surface reflectance
        'rog': rog,
        # pizera = aerosol single-scatter albedo (omega_0), NOT spherical albedo.
        # Use spherical_albedo_tot ('sast') as 's' in the retrieval formula.
        'pizera': pizera,
    }


def main():
    """Entry point for command-line use."""
    results = run6S(sys.stdin, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())