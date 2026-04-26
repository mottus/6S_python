"""
aeroso.py
---------
Translated from AEROSO.f

Computes optical properties of aerosol mixtures from basic SRA components
(dust, water-soluble, oceanic, soot) or special models.
Populates the /sixs_aer/ and /sixs_sos/ common blocks.
"""

import math
import numpy as np

from .commons           import aer, disc
from .aerosol_components import aerbas, dust, wate, ocea, soot
from .gauss             import gauss


# ---------------------------------------------------------------------------
# SRA basic component optical data (extinction, scattering at 10 wavelengths)
# i: 1=dust-like 2=water-soluble 3=oceanic 4=soot
# ---------------------------------------------------------------------------

# Extinction and scattering coefficients  ex(4,10), sc(4,10)  [km-1]
_EX = np.array([
    # dust-like (i=1)
    [0.1796674e-01, 0.1815135e-01, 0.1820247e-01, 0.1827016e-01,
     0.1842182e-01, 0.1853081e-01, 0.1881427e-01, 0.1974608e-01,
     0.1910712e-01, 0.1876025e-01],
    # water-soluble (i=2)
    [0.7653460e-06, 0.6158538e-06, 0.5793444e-06, 0.5351736e-06,
     0.4480091e-06, 0.3971033e-06, 0.2900993e-06, 0.1161433e-06,
     0.3975192e-07, 0.1338443e-07],
    # oceanic (i=3)
    [0.3499458e-02, 0.3574996e-02, 0.3596592e-02, 0.3622467e-02,
     0.3676341e-02, 0.3708866e-02, 0.3770822e-02, 0.3692255e-02,
     0.3267943e-02, 0.2801670e-02],
    # soot (i=4)
    [0.8609083e-06, 0.6590103e-06, 0.6145787e-06, 0.5537643e-06,
     0.4503008e-06, 0.3966041e-06, 0.2965532e-06, 0.1493927e-06,
     0.1017134e-06, 0.6065031e-07],
], dtype=np.float64)

_SC = np.array([
    # dust-like
    [0.1126647e-01, 0.1168918e-01, 0.1180978e-01, 0.1196792e-01,
     0.1232056e-01, 0.1256952e-01, 0.1319347e-01, 0.1520712e-01,
     0.1531952e-01, 0.1546761e-01],
    # water-soluble
    [0.7377123e-06, 0.5939413e-06, 0.5587120e-06, 0.5125148e-06,
     0.4289210e-06, 0.3772760e-06, 0.2648252e-06, 0.9331806e-07,
     0.3345499e-07, 0.1201109e-07],
    # oceanic
    [0.3499455e-02, 0.3574993e-02, 0.3596591e-02, 0.3622465e-02,
     0.3676338e-02, 0.3708858e-02, 0.3770696e-02, 0.3677038e-02,
     0.3233194e-02, 0.2728013e-02],
    # soot
    [0.2299196e-06, 0.1519321e-06, 0.1350890e-06, 0.1155423e-06,
     0.8200095e-07, 0.6469735e-07, 0.3610638e-07, 0.6227224e-08,
     0.1779378e-08, 0.3050002e-09],
], dtype=np.float64)

# Asymmetry parameters  asy(4,10)
_ASY = np.array([
    [0.896, 0.885, 0.880, 0.877, 0.867, 0.860, 0.845, 0.836, 0.905, 0.871],
    [0.642, 0.633, 0.631, 0.628, 0.621, 0.616, 0.610, 0.572, 0.562, 0.495],
    [0.795, 0.790, 0.788, 0.781, 0.783, 0.782, 0.778, 0.783, 0.797, 0.750],
    [0.397, 0.359, 0.348, 0.337, 0.311, 0.294, 0.253, 0.154, 0.103, 0.055],
], dtype=np.float32)

# Number density and volume for mixing calculation
_VI = np.array([113.983516, 113.983516e-6, 5.1444150196, 59.77353425e-6])
_NI = np.array([54.734, 1.86855e6, 276.05, 1.80582e6])

# Gauss angles and weights for phase function interpolation (83 angles)
def _init_sos_angles():
    """Initialise the Gauss angles used in the phase function (83 points)."""
    anglem, weightm = gauss(-1.0, 1.0, 83)
    return anglem, weightm

_CGAUS, _PDGS = _init_sos_angles()


def aeroso(iaer, co, xmud, wldis, file2=''):
    """
    Compute aerosol optical properties and populate /sixs_aer/ and /sixs_sos/.

    Parameters
    ----------
    iaer  : int       – aerosol model index
              0  = no aerosol
              1  = continental (dust+water-soluble+soot)
              2  = maritime (water-soluble+oceanic+soot)
              3  = urban (water-soluble+dust+soot)
              4  = smoke (custom)
              5  = desert / background desert
              6  = biomass burning
              7  = stratospheric
              8-11 = size-distribution Mie models
              12 = user-defined from file
    co    : array of 4 floats – volume fractions [dust, water-sol, oceanic, soot]
    xmud  : float – cos(scattering angle) at the solar angle
    wldis : array of 10 floats – discrete wavelengths (µm)
    file2 : str – path for user-defined aerosol file (iaer=12)

    Side-effects
    ------------
    Sets aer.ext, aer.ome, aer.gasym, aer.phase (10 elements each).
    Sets the /sixs_sos/ phase function arrays.
    """
    # Use module-level arrays for SOS common
    # /sixs_sos/ phasel(10,83), cgaus(83), pdgs(83)
    # We store them directly on the disc object for now
    if not hasattr(disc, 'phasel'):
        disc.phasel = np.zeros((10, 83), dtype=np.float32)
        disc.cgaus  = np.array(_CGAUS, dtype=np.float32)
        disc.pdgs   = np.array(_PDGS,  dtype=np.float32)
        disc.wldis  = np.array(wldis,  dtype=np.float32)

    ext    = aer.ext
    ome    = aer.ome
    gasym  = aer.gasym
    phase  = aer.phase

    ext[:] = 0.0; ome[:] = 0.0; gasym[:] = 0.0; phase[:] = 0.0
    sca    = np.zeros(10, dtype=np.float32)
    disc.phasel[:] = 0.0
    disc.wldis[:] = wldis

    # Volume fractions
    ci = np.array(co[:4], dtype=np.float64)

    if iaer == 0:
        ext[3] = 1.0   # avoid division by zero in discom
        return

    # Find scattering angle bracket in CGAUS
    j1 = 0
    for k in range(82):
        if _CGAUS[k] <= xmud < _CGAUS[k + 1]:
            j1 = k
            break
    j2   = j1 + 1
    coef = -(xmud - _CGAUS[j1]) / (_CGAUS[j2] - _CGAUS[j1])

    # --- Phase function data arrays ---
    # dd(4,10)  – phase function at xmud per component per wavelength
    dd      = np.zeros((4, 10), dtype=np.float32)
    pha_arr = np.zeros((4, 10, 83), dtype=np.float32)

    ex = _EX.copy()
    sc = _SC.copy()
    asy = _ASY.copy()

    # Load component phase functions
    if iaer <= 4 or (iaer >= 5 and iaer <= 11):
        # Standard SRA components
        dust()
        dd[0, :] = aerbas.ph[:, j1] + coef * (aerbas.ph[:, j1] - aerbas.ph[:, j2])
        pha_arr[0, :, :] = aerbas.ph

        wate()
        dd[1, :] = aerbas.ph[:, j1] + coef * (aerbas.ph[:, j1] - aerbas.ph[:, j2])
        pha_arr[1, :, :] = aerbas.ph

        ocea()
        dd[2, :] = aerbas.ph[:, j1] + coef * (aerbas.ph[:, j1] - aerbas.ph[:, j2])
        pha_arr[2, :, :] = aerbas.ph

        soot()
        dd[3, :] = aerbas.ph[:, j1] + coef * (aerbas.ph[:, j1] - aerbas.ph[:, j2])
        pha_arr[3, :, :] = aerbas.ph

    icp = 4   # number of components

    if iaer >= 5 and iaer <= 11:
        # Special single-component aerosol models
        icp = 1
        cij = np.array([1.0, 0.0, 0.0, 0.0])

        if iaer == 5:   # Background Desert Model
            from .aerosol_components import (bdm,
                _EX_BDM, _SC_BDM, _ASY_BDM)
            bdm()
            ex[0, :] = _EX_BDM;  sc[0, :] = _SC_BDM;  asy[0, :] = _ASY_BDM

        elif iaer == 6:  # Biomass Burning Model
            from .aerosol_components import (bbm,
                _EX_BBM, _SC_BBM, _ASY_BBM)
            bbm()
            ex[0, :] = _EX_BBM;  sc[0, :] = _SC_BBM;  asy[0, :] = _ASY_BBM

        elif iaer == 7:  # Stratospheric Model
            from .aerosol_components import (stm,
                _EX_STM, _SC_STM, _ASY_STM)
            stm()
            ex[0, :] = _EX_STM;  sc[0, :] = _SC_STM;  asy[0, :] = _ASY_STM

        else:  # iaer 8-11: Mie size distributions
            from .mie import mie as _mie
            from .commons import mie_in
            cgaus_arr = disc.cgaus if hasattr(disc, 'cgaus') else np.array(_CGAUS)
            pdgs_arr  = disc.pdgs  if hasattr(disc, 'pdgs')  else np.array(_PDGS)
            ph_mie, ex_mie, sc_mie, asy_mie = _mie(
                iaer, wldis, ex, sc, asy[0, :],
                mie_in.icp,    mie_in.rmin,    mie_in.rmax,
                mie_in.rn,     mie_in.ri,
                mie_in.x1,     mie_in.x2,      mie_in.x3,
                mie_in.cij,
                mie_in.irsunph, mie_in.rsunph, mie_in.nrsunph,
                cgaus_arr, pdgs_arr,
            )
            ex[0, :]  = ex_mie[0, :]
            sc[0, :]  = sc_mie[0, :]   # needed by mixing loop: sca[l] += sc[j,l]*cij[j]
            asy[0, :] = asy_mie[0, :]
            aerbas.ph[:, :] = ph_mie
            dd[0, :]        = ph_mie[:, j1] + coef * (ph_mie[:, j1] - ph_mie[:, j2])
            pha_arr[0, :, :] = ph_mie

        if iaer <= 7:  # BDM/BBM/STM: fetch phase function the same way as SRA components
            dd[0, :]        = aerbas.ph[:, j1] + coef * (aerbas.ph[:, j1] - aerbas.ph[:, j2])
            pha_arr[0, :, :] = aerbas.ph

        nis = 1.0 / ex[0, 3] if ex[0, 3] > 0 else 1.0
    elif iaer == 12:
        # User-defined from file
        try:
            with open(file2) as fp:
                fp.readline()  # skip header
                for l in range(10):
                    parts = fp.readline().split()
                    ext[l]   = float(parts[0])
                    sca[l]   = float(parts[1])
                    ome[l]   = float(parts[2])
                    gasym[l] = float(parts[3])
                fp.readline(); fp.readline(); fp.readline()
                for k in range(83):
                    parts = fp.readline().split()
                    for l in range(10):
                        disc.phasel[l, k] = float(parts[l + 1])
            for l in range(10):
                phase[l] = (disc.phasel[l, j1]
                            + coef * (disc.phasel[l, j1] - disc.phasel[l, j2]))
        except Exception as e:
            print(f"aeroso: error reading user aerosol file: {e}")
        return
    else:
        # Standard mixture: compute cij
        sigm = np.sum(ci / _VI)
        cij  = (ci / _VI) / sigm
        sumni = np.sum(cij / _NI)
        nis  = 1.0 / sumni

    # Mix optical properties
    for l in range(10):
        for j in range(icp):
            ext[l]    += ex[j, l]   * cij[j]
            sca[l]    += sc[j, l]   * cij[j]
            gasym[l]  += sc[j, l]   * cij[j] * asy[j, l]
            phase[l]  += sc[j, l]   * cij[j] * dd[j, l]
            for k in range(83):
                disc.phasel[l, k] += sc[j, l] * cij[j] * pha_arr[j, l, k]

        ome[l]   = sca[l] / ext[l]
        gasym[l] = gasym[l] / sca[l]
        phase[l] = phase[l] / sca[l]
        disc.phasel[l, :] /= sca[l]

        ext[l] *= nis
        sca[l] *= nis
