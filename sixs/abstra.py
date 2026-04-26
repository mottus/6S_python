"""
abstra.py
---------
Translated from ABSTRA.f

Gaseous transmittance calculation for all atmospheric absorbers:
water vapour, ozone, CO2, O2, N2O, CH4, CO.
"""

import math
from .commons     import atm, planesim
from .gas_tables  import get_gas_coeff

# Wavenumber band boundaries (cm-1)
_IVLI = [2500, 5060, 7620, 10180, 12740, 15300]

# Ozone cross-section table (102 values, cm^2/molecule)
_CO3 = [
    4.50e-03, 8.00e-03, 1.07e-02, 1.10e-02, 1.27e-02, 1.71e-02,
    2.00e-02, 2.45e-02, 3.07e-02, 3.84e-02, 4.78e-02, 5.67e-02,
    6.54e-02, 7.62e-02, 9.15e-02, 1.00e-01, 1.09e-01, 1.20e-01,
    1.28e-01, 1.12e-01, 1.11e-01, 1.16e-01, 1.19e-01, 1.13e-01,
    1.03e-01, 9.24e-02, 8.28e-02, 7.57e-02, 7.07e-02, 6.58e-02,
    5.56e-02, 4.77e-02, 4.06e-02, 3.87e-02, 3.82e-02, 2.94e-02,
    2.09e-02, 1.80e-02, 1.91e-02, 1.66e-02, 1.17e-02, 7.70e-03,
    6.10e-03, 8.50e-03, 6.10e-03, 3.70e-03, 3.20e-03, 3.10e-03,
    2.55e-03, 1.98e-03, 1.40e-03, 8.25e-04, 2.50e-04, 0.,
    0.,        0.,        5.65e-04, 2.04e-03, 7.35e-03, 2.03e-02,
    4.98e-02, 1.18e-01, 2.46e-01, 5.18e-01, 1.02e+00, 1.95e+00,
    3.79e+00, 6.65e+00, 1.24e+01, 2.20e+01, 3.67e+01, 5.95e+01,
    8.50e+01, 1.26e+02, 1.68e+02, 2.06e+02, 2.42e+02, 2.71e+02,
    2.91e+02, 3.02e+02, 3.03e+02, 2.94e+02, 2.77e+02, 2.54e+02,
    2.26e+02, 1.96e+02, 1.68e+02, 1.44e+02, 1.17e+02, 9.75e+01,
    7.65e+01, 6.04e+01, 4.62e+01, 3.46e+01, 2.52e+01, 2.00e+01,
    1.57e+01, 1.20e+01, 1.00e+01, 8.80e+00, 8.30e+00, 8.60e+00,
]

# H2O continuum correction coefficients
_CCH2O = [0.00, 0.19, 0.15, 0.12, 0.10, 0.09, 0.10, 0.12,
           0.15, 0.17, 0.20, 0.24, 0.28, 0.33, 0.00]

# Gas table name lookup: (id 1..6, idgaz 1..7) → table name or None
def _gas_table_name(id_, idgaz):
    """Return gas table name or None if not applicable."""
    table = {
        (1, 1): 'wava1', (1, 2): 'dica1', (1, 4): 'ozon1',
        (1, 5): 'niox1', (1, 6): 'meth1', (1, 7): 'moca1',
        (2, 1): 'wava2', (2, 2): 'dica2',
        (2, 5): 'niox2', (2, 6): 'meth2', (2, 7): 'moca2',
        (3, 1): 'wava3', (3, 2): 'dica3', (3, 3): 'oxyg3',
        (3, 5): 'niox3', (3, 6): 'meth3', (3, 7): 'moca3',
        (4, 1): 'wava4', (4, 3): 'oxyg4',
        (4, 5): 'niox4', (4, 6): 'meth4', (4, 7): 'moca4',
        (5, 1): 'wava5', (5, 3): 'oxyg5',
        (5, 5): 'niox5', (5, 6): 'meth5', (5, 7): 'moca5',
        (6, 1): 'wava6', (6, 3): 'oxyg6',
        (6, 5): 'niox6', (6, 6): 'meth6', (6, 7): 'moca6',
    }
    return table.get((id_, idgaz), None)


def abstra(idatm, wl, xmus, xmuv, uw, uo3, uwus, uo3us,
           idatmp, uwpl, uo3pl, uwusp, uo3usp):
    """
    Compute gaseous transmittances for all species along sun and view paths.

    Parameters
    ----------
    idatm   : int   – atmospheric model (0 = no atmosphere)
    wl      : float – wavelength (µm)
    xmus    : float – cos(solar zenith)
    xmuv    : float – cos(view zenith)
    uw      : float – water vapour column (g/cm²)
    uo3     : float – ozone column (Dobson units)
    uwus    : float – US-standard water vapour column
    uo3us   : float – US-standard ozone column
    idatmp  : int   – plane simulation flag (0 = ground, 4 = satellite)
    uwpl    : float – water vapour above plane
    uo3pl   : float – ozone above plane
    uwusp   : float – US-standard water vapour above plane
    uo3usp  : float – US-standard ozone above plane

    Returns
    -------
    dict with keys:
        dtwava, dtozon, dtdica, dtoxyg, dtniox, dtmeth, dtmoca  (downward)
        utwava, utozon, utdica, utoxyg, utniox, utmeth, utmoca  (upward)
        ttwava, ttozon, ttdica, ttoxyg, ttniox, ttmeth, ttmoca  (total)
    """
    from .utils import print_error

    # Initialise all transmittances to 1
    out = {}
    for pfx in ('dt', 'ut', 'tt'):
        for gas in ('wava', 'ozon', 'dica', 'oxyg', 'niox', 'meth', 'moca'):
            out[pfx + gas] = 1.0

    if idatm == 0:
        return out
    if xmus == 0.0 or xmuv == 0.0:
        print_error('Error on zenithal angle ( near 90 deg )')
        return out

    # Physical constants
    accu  = 1.0e-10
    p0    = 1013.25
    g     = 98.1
    t0    = 250.0
    air   = 0.028964 / 0.0224
    roco2 = 0.044    / 0.0224
    rmo2  = 0.032    / 0.0224
    rmo3  = 0.048    / 0.0224
    rmn2o = 0.044    / 0.0224
    rmch4 = 0.016    / 0.0224
    rmco  = 0.028    / 0.0224

    uwus_std  = 1.424
    uo3us_std = 0.344

    # Scaling ratios for idatm==8 (user-defined atmosphere)
    rat = [1.0] * 11   # 1-indexed, rat[1..10]
    if idatm == 8:
        rat[1]  = uw  / uwus_std
        rat[4]  = uo3 / uo3us_std
        rat[8]  = uw  / uwus_std
        rat[9]  = uw  / uwus_std
        rat[10] = uw  / uwus_std

    # Wavenumber
    v  = 1.0e4 / wl
    iv = int(v / 5) * 5
    id_ = (int((iv - 2500) / 10) // 256) + 1
    id_ = max(1, min(6, id_))
    inu = (iv - _IVLI[id_ - 1]) // 10 + 1
    inu = max(1, min(256, inu))

    tnu = [[1.0] * 4 for _ in range(11)]   # tnu[idgaz][path]  1-indexed

    for idgaz in range(1, 8):
        # Get spectral coefficients
        tname = _gas_table_name(id_, idgaz)
        if tname is None:
            a = [0.0] * 8
        else:
            a = get_gas_coeff(tname, inu)

        # Mixing ratios along full atmosphere
        z  = atm.z;  p = atm.p;  t = atm.t
        wh = atm.wh; wo = atm.wo
        rm = [0.0] * 34
        r2 = [0.0] * 34
        r3 = [0.0] * 34
        tp = [0.0] * 34

        for k in range(33):
            roair   = air * 273.16 * p[k] / (1013.25 * t[k])
            tp[k]   = (t[k] + t[k+1]) / 2.0
            te      = tp[k] - t0
            te2     = te * te
            phi     = math.exp(a[2] * te + a[3] * te2)
            psi     = math.exp(a[4] * te + a[5] * te2)
            if idgaz == 1:   rm[k] = wh[k] / (roair * 1000.0)
            elif idgaz == 2: rm[k] = 3.3e-4 * roco2 / air
            elif idgaz == 3: rm[k] = 0.20947 * rmo2 / air
            elif idgaz == 4: rm[k] = wo[k]  / (roair * 1000.0)
            elif idgaz == 5: rm[k] = 310.0e-9 * rmn2o / air
            elif idgaz == 6: rm[k] = 1.72e-6  * rmch4 / air
            elif idgaz == 7: rm[k] = 1.0e-9   * rmco  / air
            r2[k] = rm[k] * phi
            r3[k] = rm[k] * psi

        uu = u = up = 0.0
        for k in range(1, 33):
            ds  = (p[k-1] - p[k])       / p[0]
            ds2 = (p[k-1]**2 - p[k]**2) / (2.0 * p[0] * p0)
            uu += ((rm[k] + rm[k-1]) / 2.0) * ds  * rat[idgaz]
            u  += ((r2[k] + r2[k-1]) / 2.0) * ds  * rat[idgaz]
            up += ((r3[k] + r3[k-1]) / 2.0) * ds2 * rat[idgaz]
        uu *= p[0] * 100.0 / g
        u  *= p[0] * 100.0 / g
        up *= p[0] * 100.0 / g

        if idgaz == 4: uu = 1000.0 * uu / rmo3
        if idgaz == 2: uu = 1000.0 * uu / roco2
        if idgaz == 5: uu = 1000.0 * uu / rmn2o
        if idgaz == 6: uu = 1000.0 * uu / rmch4
        if idgaz == 7: uu = 1000.0 * uu / rmco

        # Plane-level mixing ratios
        if idatmp in (0, 4):
            uupl = uu; upl = u; uppl = up
        else:
            zpl  = planesim.zpl;  ppl  = planesim.ppl;  tpl  = planesim.tpl
            whpl = planesim.whpl; wopl = planesim.wopl
            rmpl = [0.0] * 34; r2pl = [0.0] * 34; r3pl = [0.0] * 34
            ratpl = [1.0] * 11
            if idatmp == 8:
                ratpl[1] = uwpl  / uwusp if uwusp else 1.0
                ratpl[4] = uo3pl / uo3usp if uo3usp else 1.0
                for idx in (8, 9, 10):
                    ratpl[idx] = ratpl[1]
            for k in range(33):
                roair = air * 273.16 * ppl[k] / (1013.25 * tpl[k])
                tp[k] = (tpl[k] + tpl[k+1]) / 2.0
                te = tp[k] - t0; te2 = te * te
                phi = math.exp(a[2] * te + a[3] * te2)
                psi = math.exp(a[4] * te + a[5] * te2)
                if idgaz == 1:   rmpl[k] = whpl[k] / (roair * 1000.0)
                elif idgaz == 2: rmpl[k] = 3.3e-4 * roco2 / air
                elif idgaz == 3: rmpl[k] = 0.20947 * rmo2 / air
                elif idgaz == 4: rmpl[k] = wopl[k] / (roair * 1000.0)
                elif idgaz == 5: rmpl[k] = 310.0e-9 * rmn2o / air
                elif idgaz == 6: rmpl[k] = 1.72e-6  * rmch4 / air
                elif idgaz == 7: rmpl[k] = 1.0e-9   * rmco  / air
                r2pl[k] = rmpl[k] * phi
                r3pl[k] = rmpl[k] * psi
            uupl = upl = uppl = 0.0
            for k in range(1, 33):
                ds  = (ppl[k-1] - ppl[k]) / ppl[0]
                ds2 = (ppl[k-1]**2 - ppl[k]**2) / (2.0 * ppl[0] * p0)
                uupl += ((rmpl[k] + rmpl[k-1]) / 2.0) * ds  * ratpl[idgaz]
                upl  += ((r2pl[k] + r2pl[k-1]) / 2.0) * ds  * ratpl[idgaz]
                uppl += ((r3pl[k] + r3pl[k-1]) / 2.0) * ds2 * ratpl[idgaz]
            uupl *= ppl[0] * 100.0 / g
            upl  *= ppl[0] * 100.0 / g
            uppl *= ppl[0] * 100.0 / g
            if idgaz == 4: uupl = 1000.0 * uupl / rmo3
            if idgaz == 2: uupl = 1000.0 * uupl / roco2
            if idgaz == 5: uupl = 1000.0 * uupl / rmn2o
            if idgaz == 6: uupl = 1000.0 * uupl / rmch4
            if idgaz == 7: uupl = 1000.0 * uupl / rmco

        uud = uu   / xmus
        uuu = uupl / xmuv
        uut = uu   / xmus + uupl / xmuv

        # Special treatment for H2O (1) and O3 (4)
        if idgaz in (1, 4):
            # H2O continuum (2350-3000 cm-1)
            dtcont = utcont = ttcont = 1.0
            if idgaz == 1 and 2350 <= iv <= 3000:
                xi = (v - 2350.0) / 50.0 + 1.0
                nh = int(xi + 1.001); xh = xi - nh
                ah2o = _CCH2O[nh - 1] + xh * (_CCH2O[nh - 1] - _CCH2O[nh - 2])
                dtcont = math.exp(-ah2o * uud)
                utcont = math.exp(-ah2o * uuu)
                ttcont = math.exp(-ah2o * uut)

            # O3 UV/Vis absorption (13000-23400 and 27500+ cm-1)
            if idgaz == 4:
                if iv < 13000 or iv > 3020:
                    pass  # handled below
                if 13000 <= iv <= 23400:
                    xi   = (v - 13000.0) / 200.0 + 1.0
                    n    = int(xi + 1.001); xd = xi - n
                    ako3 = _CO3[n - 1] + xd * (_CO3[n - 1] - _CO3[n - 2])
                    for path, ud_val in enumerate([uud, uuu, uut], 1):
                        test = min(ako3 * ud_val, 86.0)
                        tnu[4][path] = math.exp(-test)
                    continue
                elif iv >= 27500:
                    xi   = (v - 27500.0) / 500.0 + 57.0
                    n    = int(xi + 1.001); xd = xi - n
                    ako3 = _CO3[n - 1] + xd * (_CO3[n - 1] - _CO3[n - 2])
                    for path, ud_val in enumerate([uud, uuu, uut], 1):
                        test = min(ako3 * ud_val, 86.0)
                        tnu[4][path] = math.exp(-test)
                    continue
                else:
                    tnu[4][1] = tnu[4][2] = tnu[4][3] = 1.0
                    continue

            # Skip if outside range
            if idgaz == 2 and iv > 9620:
                tnu[idgaz][1] = tnu[idgaz][2] = tnu[idgaz][3] = 1.0
                continue
            if idgaz == 3 and iv > 15920:
                tnu[idgaz][1] = tnu[idgaz][2] = tnu[idgaz][3] = 1.0
                continue

        # Band-model transmittance calculation
        def band_trans(ud, upd):
            udt  = ud  if (ud != 0.0 or upd != 0.0) else 1.0
            atest = a[1] if (a[1] != 0.0 or a[0] != 0.0) else 1.0
            updt  = upd if (ud != 0.0 or upd != 0.0) else 1.0
            tn    = a[1] * updt / (2.0 * udt)
            tt    = 1.0 + 4.0 * (a[0] / atest) * (ud**2 / updt)
            if idgaz == 1:
                y = -a[0] * ud / math.sqrt(max(1e-30, 1.0 + (a[0] / atest) * (ud**2 / updt)))
            else:
                y = -tn * (math.sqrt(max(0.0, tt)) - 1.0)
            return math.exp(max(-87.0, min(0.0, y)))

        # Downward
        ud_d  = u  / xmus
        upd_d = up / xmus
        tnu[idgaz][1] = band_trans(ud_d, upd_d)

        # Upward (uses plane profile)
        ud_u  = upl  / xmuv
        upd_u = uppl / xmuv
        tnu[idgaz][2] = band_trans(ud_u, upd_u)

        # Total path
        ut_t  = u   / xmus + upl  / xmuv
        upt_t = up  / xmus + uppl / xmuv
        tnu[idgaz][3] = band_trans(ut_t, upt_t)

    # Assemble output
    def safe(val):
        return val if val > accu else 0.0

    dtcont = utcont = ttcont = 1.0  # re-use from H2O block if set

    names = ['', 'wava', 'dica', 'oxyg', 'ozon', 'niox', 'meth', 'moca']
    for idgaz in range(1, 8):
        nm = names[idgaz]
        out[f'dt{nm}'] = safe(tnu[idgaz][1])
        out[f'ut{nm}'] = safe(tnu[idgaz][2])
        out[f'tt{nm}'] = safe(tnu[idgaz][3])

    # Apply H2O continuum
    out['dtwava'] = safe(tnu[1][1] * dtcont)
    out['utwava'] = safe(tnu[1][2] * utcont)
    out['ttwava'] = safe(tnu[1][3] * ttcont)

    # For ground case, upward = 1 and total = downward
    if idatmp == 0:
        for gas in ('wava', 'dica', 'oxyg', 'ozon', 'niox', 'meth', 'moca'):
            out[f'tt{gas}'] = out[f'dt{gas}']
            out[f'ut{gas}'] = 1.0

    return out
