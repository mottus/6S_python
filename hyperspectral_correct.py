"""
hyperion_atm_correction.py
--------------------------
Atmospheric correction of EO-1 Hyperion L1T hyperspectral imagery using 6S
(Second Simulation of the Satellite Signal in the Solar Spectrum).

Usage
-----
  GUI    : python hyperion_atm_correction.py
  Script : python hyperion_atm_correction.py file.hdr [file.L1T]
  Import : from hyperion_atm_correction import read_mtl, read_hdr, correct_hyperion

Processing chain
----------------
1.  Read metadata
    - MTL  -> geometry (SZA, SAA, VZA, VAA), date, radiance scaling factors
    - HDR  -> image dimensions, wavelengths (nm), FWHM, bad-band list (bbl)

2.  Build valid-pixel mask
    - External mask file if given (any ENVI file, non-zero = valid)
    - Auto: pixels where ALL bands are zero (sensor fill / black edges)

3.  Create output file before computation
    - Allocates disk space immediately (int16, interleave managed by spectral)
    - Pre-filled with nodata = -9999

4.  Open input as read-only memory map  (one band read at a time)

5.  For each good band (bbl=1, wavelength 0.40-2.50 um):
    a. Run 6S (monochromatic, iwave=-1) -> correction coefficients:
         xa = atmospheric path reflectance   (= srotot)
         xb = 1 / (T_down * T_up)
         xc = spherical_albedo / (T_down * T_up)
       If inhomo=1 is chosen, 6S also accounts for the adjacency effect:
       light reflected by the surrounding environment is scattered toward
       the sensor, making dark pixels near bright areas appear brighter.
       The environment type (vegetation/water/sand/lake) and patch radius
       are user-selectable.
    b. Get TOA solar irradiance E0 from built-in 6S solar spectrum (solirr),
       corrected for Earth-Sun distance on the acquisition date (Spencer 1971
       approximation based on day-of-year).
    c. DN -> radiance:  L [W/m2/sr/um] = DN / scale_factor
         VNIR (bands 1-70):  scale = 40
         SWIR (bands 71-242): scale = 80
    d. TOA reflectance:  rho_toa = pi * L / (E0 * cos(SZA))
       Units: L and E0 both in W/m2, ratio is dimensionless.
    e. Surface reflectance (Vermote et al. 1997):
         rho_s = (rho_toa - xa) / (xb * rho_toa + xc)
    f. Scale to int16:  stored = round(rho_s * 10000)
       To recover reflectance: rho_s = stored_value / 10000.0
       Nodata (masked pixels) = -9999

6.  Write ENVI header:
    - sensor type = EO-1 Hyperion
    - band names = B1 (356nm), B2 (366nm), ...
    - wavelengths rounded to 2 decimal places (nm)
    - description with all processing parameters

Key constants (change here to modify behaviour)
-----------------------------------------------
  VNIR_SWIR_BOUNDARY = 70    bands 1-70 VNIR (scale 40), 71-242 SWIR (scale 80)

  Hyperion L1GST unit chain (confirmed from FLAASH docs and sample data):
    L [uW/cm2/sr/nm] = DN / 400 (VNIR) or DN / 800 (SWIR)  <- FLAASH units
    L [W/m2/sr/um]   = DN / 40  (VNIR) or DN / 80  (SWIR)  <- SI, used here
    1 uW/cm2/sr/nm = 10 W/m2/sr/um (exact conversion)
    rho_toa = pi * L[W/m2/sr/um] / (E0[W/m2/um] * cos(SZA))

  6S xa = srotot: aerosol path refl from SOS solver, NOT total path refl.
  Total path refl seen by sensor = chand(tau_R)+sroaer ~ 0.147 at 427nm.
  But xa=srotot IS correct in rho_s=(rho_toa-xa)/(xb*rho_toa+xc) because
  xb=1/(T_down*T_up) and xc=S*xb encode the full Rayleigh+aerosol transmittance.
  Replacing srotot with chand()+sroaer would double-count Rayleigh and give
  wrong surface reflectances. See sixs/utils.py for full derivation.
  Reflectance scale  = 10000 stored integer = reflectance * 10000
  Nodata             = -9999 int16 fill for masked / bad-band pixels
  Valid wl range     = 0.40-2.50 um  (6S operating limits)

Requirements: pip install spectral numpy scipy
              pip install -e /path/to/sixs_python
"""

import os, io, re, math, datetime, warnings
import numpy as np
warnings.filterwarnings("ignore")


# -- Atmospheric / aerosol model menus ----------------------------------------
ATM_MODELS = {
    "No absorption (idatm=0)":       0,
    "Tropical (idatm=1)":            1,
    "Mid-lat summer (idatm=2)":      2,
    "Mid-lat winter (idatm=3)":      3,
    "Subarctic summer (idatm=4)":    4,
    "Subarctic winter (idatm=5)":    5,
    "US Standard 1962 (idatm=6)":    6,
    "User H2O + O3 (idatm=8)":       8,
}
DEFAULT_ATM = "Mid-lat summer (idatm=2)"

AEROSOL_MODELS = {
    "None (iaer=0)":         0,
    "Continental (iaer=1)":  1,
    "Maritime (iaer=2)":     2,
    "Urban (iaer=3)":        3,
}
DEFAULT_AER = "Continental (iaer=1)"

# inhomo=1 environment surface types (igrou2 codes for 6S)
ENV_MODELS = {
    "Uniform (inhomo=0, no adjacency)": None,   # None = inhomo=0
    "Vegetation":    1,
    "Clear water":   2,
    "Sand":          3,
    "Lake water":    4,
}
DEFAULT_ENV = "Vegetation"
DEFAULT_ENV_RADIUS_KM = 2.0   # typical atmospheric PSF radius at low AOT

# ── Sensor profiles ───────────────────────────────────────────────────────────
# Each profile defines the sensor-specific parameters that do not come from
# the image header or the atmospheric model.
#
# Keys:
#   display_name  : shown in the GUI dropdown
#   sensor_type   : written to the ENVI output header
#   input_type    : "radiance" — image stores DN or physical radiance,
#                               ρ_toa = π·L/(E0·cos(SZA)), E0 from solirr()
#                   "toa_refl" — image already stores TOA reflectance,
#                               skip L→ρ_toa step
#   rad_scale     : how to build the per-band radiance scale array.
#                   "uniform:<val>"  — divide every band by <val>
#                   "split:<vnir>:<swir>:<boundary>" — bands 1..<boundary>
#                       divide by <vnir>, remainder by <swir> (1-based boundary)
#                   "none"           — data already in W/m2/sr/um (scale = 1)
#   pixel_size_m  : nominal ground sampling distance (used for adj2 default)
#   vaa_source    : "orbit_eo1" — compute VAA from EO-1 orbital geometry
#                   "manual"   — user must enter VAA directly
#   notes         : free text shown as tooltip / help string

SENSOR_PROFILES = {
    "EO-1 Hyperion": {
        "display_name": "EO-1 Hyperion",
        "sensor_type":  "EO-1 Hyperion",
        "input_type":   "radiance",
        # VNIR (bands 1-70) scale 40, SWIR (bands 71-242) scale 80
        # Units after scaling: W/m2/sr/um  (confirmed from NASA calibration docs)
        "rad_scale":    "split:40:80:70",
        "pixel_size_m": 30.0,
        "vaa_source":   "orbit_eo1",
        "notes": (
            "EO-1 Hyperion pushbroom, 242 bands 356-2577 nm. "
            "L1GST: FLAASH units DN/400=L[uW/cm2/sr/nm] VNIR, DN/800 SWIR. "
            "SI units DN/40=L[W/m2/sr/um] VNIR, DN/80 SWIR "
            "(1 uW/cm2/sr/nm = 10 W/m2/sr/um). "
            "VAA computed from orbital geometry (inclination 98.7 deg)."
        ),
    },
    # ── Add further sensors below ─────────────────────────────────────────────
    # "Sentinel-2 MSI": {
    #     "display_name": "Sentinel-2 MSI",
    #     "sensor_type":  "Sentinel-2 MSI",
    #     "input_type":   "toa_refl",   # L1C product is BOA or TOA reflectance
    #     "rad_scale":    "none",
    #     "pixel_size_m": 10.0,
    #     "vaa_source":   "manual",
    #     "notes": "Sentinel-2 L1C TOA reflectance. No radiance scaling needed.",
    # },
    # "Generic radiance sensor": {
    #     "display_name": "Generic radiance sensor",
    #     "sensor_type":  "Unknown",
    #     "input_type":   "radiance",
    #     "rad_scale":    "uniform:1",   # already in W/m2/sr/um
    #     "pixel_size_m": 30.0,
    #     "vaa_source":   "manual",
    #     "notes": "Generic sensor: image in W/m2/sr/um, enter all geometry manually.",
    # },
}
DEFAULT_PROFILE = "EO-1 Hyperion"


def build_rad_scale(profile_rad_scale, n_bands):
    """
    Build a numpy array of per-band radiance scale factors from a
    profile rad_scale string.  Returns array of shape (n_bands,).
    """
    spec = profile_rad_scale
    if spec == "none" or spec.startswith("uniform:1"):
        return np.ones(n_bands, dtype=np.float32)
    if spec.startswith("uniform:"):
        val = float(spec.split(":")[1])
        return np.full(n_bands, val, dtype=np.float32)
    if spec.startswith("split:"):
        _, vnir_s, swir_s, bnd_s = spec.split(":")
        vnir_scale = float(vnir_s)
        swir_scale = float(swir_s)
        boundary   = int(bnd_s)          # 1-based last VNIR band
        arr = np.full(n_bands, swir_scale, dtype=np.float32)
        arr[:boundary] = vnir_scale      # bands 0..boundary-1 (0-based)
        return arr
    raise ValueError(f"Unknown rad_scale spec: '{spec}'")


# =============================================================================
# SECTION 1 -- Metadata readers
# Return plain dicts so the computation has no file-format coupling.
# Values can also be supplied directly to correct_hyperion() without file I/O.
# =============================================================================

def read_mtl(mtl_path):
    """
    Parse an EO-1 Hyperion L1T MTL file.

    Returns dict with keys:
        acq_date, start_time, month, day,
        sza, saa, vza, vaa,
        scale_vnir, scale_swir,
        n_lines, n_samples
    """
    raw = {}
    with open(mtl_path) as f:
        for line in f:
            m = re.match(r'\s+(\w+)\s*=\s*"?([^"\n]+)"?\s*$', line)
            if m:
                raw[m.group(1)] = m.group(2).strip()

    acq_date      = raw["ACQUISITION_DATE"]
    start_time    = raw["START_TIME"]
    sun_elevation = float(raw["SUN_ELEVATION"])
    sun_azimuth   = float(raw["SUN_AZIMUTH"])
    look_angle    = float(raw["SENSOR_LOOK_ANGLE"])  # signed: + east, - west
    scale_vnir    = float(raw.get("SCALING_FACTOR_VNIR", 40))
    scale_swir    = float(raw.get("SCALING_FACTOR_SWIR", 80))
    dt  = datetime.date.fromisoformat(acq_date)
    sza = 90.0 - sun_elevation
    saa = sun_azimuth

    # VZA is always the absolute off-nadir angle (positive in 6S).
    vza = abs(look_angle)

    # ── Compute VAA from EO-1 orbit geometry ─────────────────────────────────
    #
    # Background
    # ----------
    # The MTL file provides SENSOR_LOOK_ANGLE with a SIGN but no azimuth:
    #   positive = looking east of the ground track
    #   negative = looking west of the ground track
    # 6S requires the VIEW AZIMUTH (VAA) in degrees from North, clockwise —
    # the same convention as the solar azimuth (SAA).
    # The correct VAA is the azimuth of the line from the pixel to the sensor,
    # which is perpendicular to the satellite ground track.
    #
    # EO-1 orbit
    # ----------
    # EO-1 is sun-synchronous, inclination i = 98.7 deg (slightly retrograde).
    # Standard data are acquired on the DESCENDING node (daytime, flying south).
    # The ground track azimuth for a descending pass at latitude φ is:
    #
    #   For the ascending node (flying north):
    #       sin(az_asc) = cos(i) / cos(φ)
    #   For the descending node (flying south):
    #       az_desc = 180° − arcsin(|cos(i)| / cos(φ))
    #
    # Because i > 90°, cos(i) < 0, so the ascending node flies slightly east
    # of north; the descending node flies slightly east of south.
    # Example at φ = 62° N, i = 98.7°:
    #   |cos(98.7°)| / cos(62°) ≈ 0.151 / 0.469 ≈ 0.322
    #   az_desc ≈ 180° − arcsin(0.322) ≈ 180° − 18.8° ≈ 161° (SSE)
    #
    # Cross-track azimuth
    # -------------------
    # The sensor looks perpendicular to the track.  The cross-track azimuth
    # to the east side of the track is az_track + 90° and to the west side
    # az_track − 90°.  The sign of SENSOR_LOOK_ANGLE selects which side.
    #
    # 6S convention
    # -------------
    # 6S uses phiv = view azimuth from North, clockwise. Internally it computes
    # the relative azimuth phi = SAA − VAA, so it is the DIFFERENCE that matters
    # for path radiance.  Using the physical azimuth of the look direction here
    # is therefore correct.

    EO1_INCLINATION_DEG = 98.7   # EO-1 orbital inclination (retrograde S-S)

    # Scene centre latitude from image corner coordinates
    try:
        lat_centre = (float(raw["IMAGE_UL_CORNER_LAT"]) +
                      float(raw["IMAGE_LL_CORNER_LAT"]) +
                      float(raw["IMAGE_UR_CORNER_LAT"]) +
                      float(raw["IMAGE_LR_CORNER_LAT"])) / 4.0
    except (KeyError, ValueError):
        lat_centre = 50.0   # fallback if corners absent from MTL

    inc_rad = math.radians(EO1_INCLINATION_DEG)
    lat_rad = math.radians(lat_centre)

    # Descending pass ground track azimuth
    ratio = abs(math.cos(inc_rad)) / math.cos(lat_rad)
    if ratio <= 1.0:
        az_track = 180.0 - math.degrees(math.asin(ratio))
    else:
        az_track = 180.0   # fallback: due south (equator or rounding)

    # Cross-track view azimuth: east = +90 from track, west = -90
    if look_angle >= 0.0:
        vaa = (az_track + 90.0) % 360.0   # east-looking
    else:
        vaa = (az_track - 90.0) % 360.0   # west-looking

    return dict(
        acq_date=acq_date, start_time=start_time.strip(),
        month=dt.month, day=dt.day,
        sza=sza, saa=saa, vza=vza, vaa=round(vaa, 1),
        # Legacy fields kept for backward compat; new code uses rad_scale_spec
        scale_vnir=scale_vnir, scale_swir=scale_swir,
        rad_scale_spec=f"split:{scale_vnir}:{scale_swir}:70",
    )


def read_hdr(hdr_path):
    """
    Parse an ENVI header file (via spectral).

    Returns dict with keys:
        n_bands, n_lines, n_samples,
        wl_nm, wl_um, fwhm_nm, bbl, envi_meta
    """
    import spectral.io.envi as envi
    hdr  = envi.open(hdr_path)
    meta = hdr.metadata

    wl_nm   = np.array([float(w) for w in meta["wavelength"]])
    fwhm_nm = np.array([float(f) for f in meta["fwhm"]])
    bbl     = np.array([int(b)   for b in meta["bbl"]])

    return dict(
        n_bands   = int(meta["bands"]),
        n_lines   = int(meta["lines"]),
        n_samples = int(meta["samples"]),
        wl_nm     = wl_nm,
        wl_um     = wl_nm / 1000.0,
        fwhm_nm   = fwhm_nm,
        bbl       = bbl,
        envi_meta = meta,
    )


# =============================================================================
# SECTION 2 -- 6S helpers
# =============================================================================

def _make_6s_input(wl_um, sza, saa, vza, vaa, month, day,
                   idatm, uh2o, uo3, iaer, aot550, target_alt_km,
                   env_model=None, env_radius_km=2.0):
    """
    Build a 6S input string for one monochromatic wavelength.

    env_model    : int or None — igrou2 environment surface code (1-4).
                   None means inhomo=0 (uniform surface, no adjacency).
    env_radius_km: float — radius of the target patch in km (inhomo=1 only).
                   6S uses this to weight the adjacency contribution.
    """
    atm_extra = f"\n{uh2o:.4f}   {uo3:.4f}" if idatm == 8 else ""

    if env_model is not None:
        # inhomo=1: target is constant rho=0 (placeholder; we derive rho
        # from the image), environment is env_model, radius = env_radius_km.
        # idirec=0 (Lambertian), igroun=0 (constant), rho=0.
        surface = (
            f"1\n"                               # inhomo=1
            f"0 {env_model} {env_radius_km:.2f}\n"  # target env radius
            f"0\n"                               # idirec=0
            f"0\n"                               # igroun=0 (constant)
            f"0.0\n"                             # rho=0 placeholder
        )
    else:
        # inhomo=0: uniform surface
        surface = "0\n0\n0\n0.0\n"

    return (
        f"0\n"
        f"{sza:.4f} {saa:.4f} {vza:.4f} {vaa:.4f} {month} {day}\n"
        f"{idatm}{atm_extra}\n"
        f"{iaer}\n"
        f"0\n"
        f"{aot550:.4f}\n"
        f"{-abs(target_alt_km):.4f}\n"
        f"-1000\n"
        f"-1\n"
        f"{wl_um}\n"
        f"{surface}"
        f"-2.0\n"
    )


def _coefficients(r6s):
    """
    Derive correction coefficients from a 6S result dict.

    Stage-1 correction (uniform surface assumption):
        rho_s = (rho_toa - xa) / (xb * rho_toa + xc)

    Stage-2 adjacency correction (spatially variable environment):
        rho_s2 = (rho_toa - xa) / (xb_down) - rho_env * S / xb_down
    where xb_down = T_down  and rho_env is a spatially smoothed rho_s.

    Returns
    -------
    xa, xb, xc        : stage-1 coefficients
    T_down, T_up, S   : needed for the stage-2 formula

    Note on xa = srotot and the 6S formula:
        srotot is the path reflectance output from the 6S SOS solver.
        It equals what the sensor sees over a black surface (rho_s=0).
        At 427nm, SZA=43.7, AOT=0.06: srotot ~ 0.012 (1.2% path refl).

        This seems low given Rayleigh tau=0.275, but is physically correct:
        chand(tau_R)=0.135 is the reflectance of a SEMI-INFINITE conservative
        Rayleigh slab — not the thin real atmosphere over a surface. In the
        real geometry, most photons transmit directly (T_dir=0.68) and only
        ~1.2% are backscattered to the sensor. The two quantities are entirely
        different radiative transfer problems.

        The 6S forward model (verified by fitting) is:
            rho_toa = xa + T_down*T_up * rho_s / (1 - s_tot * rho_s)
        where s_tot = spherical_albedo_tot (TOTAL atmospheric spherical albedo).
        The exact retrieval is then:
            rho_s = (rho_toa - xa) / (T_d*T_u + s_tot*(rho_toa - xa))

        CRITICAL: s_tot = spherical_albedo_tot (~0.029 at 427nm, AOT=0.06)
        NOT pizera (~0.90, which is the aerosol-only spherical albedo).
        Using pizera gives a denominator ~30x too large, causing rho_s to
        be underestimated by 1-3 pp. Verified: only s_tot gives zero error.

        sroray (Rayleigh component in srotot) is negative in the blue because
        it is the COUPLING CORRECTION: Rayleigh_in_coupled_atm - Rayleigh_alone.
        Aerosol forward-scatters photons that would otherwise contribute to
        Rayleigh backscatter, reducing it. So sroray < 0 in the blue.
        The total srotot = sroray + sroaer remains small and positive.
    """
    rho_atm = r6s["srotot"]
    T_down  = r6s["sdtott"]
    T_up    = r6s["sutott"]
    # CRITICAL: use spherical_albedo_tot (total atmospheric spherical albedo),
    # NOT pizera. pizera is the aerosol-only spherical albedo (~0.90 in blue).
    # The correct s in the retrieval formula rho_s=(rho_toa-xa)/(T_d*T_u+s*(rho_toa-xa))
    # is the TOTAL spherical albedo = spherical_albedo_tot (~0.029 at 427nm, AOT=0.06).
    # Using pizera causes the denominator to be ~30x too large, severely
    # underestimating rho_s (by 1-3pp at typical surface reflectances).
    # Verified by fitting the 6S forward model: rho_toa = xa + T_d*T_u*rho_s/(1-s*rho_s)
    # gives exact round-trip recovery only with s = spherical_albedo_tot.
    S       = r6s["spherical_albedo_tot"]
    denom   = T_down * T_up
    if denom < 1e-6:
        return 0.0, 1.0, 0.0, 1.0, 0.0
    return rho_atm, 1.0 / denom, S / denom, T_down, T_up, S


# =============================================================================
# SECTION 3 -- Main computation
# =============================================================================

def correct_hyperion(params, log=print):
    """
    Full atmospheric correction pipeline.

    params dict keys
    ----------------
    File / image info:
        hdr_file, out_base
        n_bands, n_lines, n_samples
        wl_nm, wl_um, fwhm_nm, bbl, envi_meta

    Acquisition geometry (from MTL or supplied directly):
        acq_date, start_time, month, day   (acq_date/start_time may be "")
        sza, saa, vza, vaa  [degrees]

    Sensor / calibration:
        sensor_name   : str — written to output header "sensor type" field
        input_type    : "radiance"  — apply DN→L→rho_toa chain
                        "toa_refl" — data already is TOA reflectance; skip E0
        rad_scale_spec: str — profile rad_scale string, passed to build_rad_scale()
                        e.g. "split:40:80:70"  or  "uniform:1"  or  "none"

    Atmospheric correction parameters:
        idatm         [0-8]
        uh2o          [g/cm2, used only when idatm=8]
        uo3           [cm-atm, used only when idatm=8]
        iaer          [0-3]
        aot550        [AOT at 550 nm]
        target_alt_km [km above sea level]

    mask_file : str or None (optional)
        Path to a single-band binary mask image (same spatial extent as input,
        any dtype; non-zero = valid pixel).  If None, a mask is derived
        automatically: pixels where ALL bands are zero are treated as nodata.

    log : callable for progress output (default: print)
    """
    from sixs.sixs_main import run6S
    import spectral.io.envi as envi

    # Unpack
    hdr_file      = params["hdr_file"]
    out_base      = params["out_base"]
    acq_date      = params["acq_date"]
    start_time    = params["start_time"]
    month, day    = params["month"], params["day"]
    sza, saa      = params["sza"], params["saa"]
    vza, vaa      = params["vza"], params["vaa"]
    sensor_name    = params.get("sensor_name",    "Unknown sensor")
    input_type     = params.get("input_type",     "radiance")
    rad_scale_spec = params.get("rad_scale_spec", "uniform:1")
    # Legacy support: if old scale_vnir/scale_swir keys present, convert
    if "scale_vnir" in params and "rad_scale_spec" not in params:
        sv = params["scale_vnir"]; ss = params["scale_swir"]
        rad_scale_spec = f"split:{sv}:{ss}:70"
    n_lines       = params["n_lines"]
    n_samples     = params["n_samples"]
    n_bands       = params["n_bands"]
    wl_um         = params["wl_um"]
    fwhm_nm       = params["fwhm_nm"]
    bbl           = params["bbl"].copy()
    envi_meta     = params["envi_meta"]
    idatm         = params["idatm"]
    uh2o          = params["uh2o"]
    uo3           = params["uo3"]
    iaer          = params["iaer"]
    aot550        = params["aot550"]
    target_alt_km = params["target_alt_km"]
    env_model      = params.get("env_model",      None)
    do_adj2        = params.get("do_adj2",        False)
    adj2_out_base  = params.get("adj2_out_base",  None)
    adj2_radius_km = params.get("adj2_radius_km", 1.0)
    pixel_size_m   = params.get("pixel_size_m",   30.0)  # Hyperion GSD = 30 m
    env_radius_km  = params.get("env_radius_km",  2.0)
    config_file    = params.get("config_file",    None)
    spec_csv_file  = params.get("spec_csv_file",  None)
    interleave     = params.get("interleave",     "bip").lower().strip()
    drop_bad_bands = params.get("drop_bad_bands", True)
    mask_file      = params.get("mask_file",      None)

    cos_sza = float(np.cos(np.radians(sza)))

    # -- Print all parameters -------------------------------------------------
    log("=" * 62)
    log("  HYPERION 6S ATMOSPHERIC CORRECTION -- PARAMETERS")
    log("=" * 62)
    log(f"  Input HDR      : {hdr_file}")
    log(f"  Output base    : {out_base}")
    log(f"  Acquisition    : {acq_date}  {start_time}")
    log(f"  Month / day    : {month} / {day}")
    log(f"  SZA            : {sza:.3f} deg")
    log(f"  SAA            : {saa:.3f} deg")
    log(f"  VZA            : {vza:.3f} deg")
    log(f"  VAA            : {vaa:.3f} deg")
    log(f"  cos(SZA)       : {cos_sza:.4f}")
    log(f"  Sensor         : {sensor_name}")
    log(f"  Input type     : {input_type}")
    log(f"  Radiance scale : {rad_scale_spec}")
    log(f"  Atm model      : idatm = {idatm}")
    if idatm == 8:
        log(f"    H2O          : {uh2o:.4f} g/cm2")
        log(f"    O3           : {uo3:.4f} cm-atm")
    log(f"  Aerosol model  : iaer = {iaer}")
    log(f"  AOT @ 550 nm   : {aot550:.4f}")
    log(f"  Target alt     : {target_alt_km:.3f} km a.s.l.")
    log(f"  Image size     : {n_lines} x {n_samples} x {n_bands}")
    env_label = next((k for k, v in ENV_MODELS.items() if v == env_model),
                     "Uniform (inhomo=0)")
    log(f"  Environment    : {env_label}" +
        (f"  radius={env_radius_km:.1f} km" if env_model is not None else ""))
    log(f"  Mask file      : {mask_file if mask_file else '(auto: all-zero pixels)'}")
    log(f"  6S config out  : {config_file if config_file else '(not saved)'}")
    log(f"  Spectral CSV   : {spec_csv_file if spec_csv_file else '(not saved)'}")
    log(f"  Output interleave: {interleave.upper()}")
    log(f"  Drop bad bands : {'yes' if drop_bad_bands else 'no'}")
    if do_adj2:
        log(f"  Stage-2 adj    : ON  radius={adj2_radius_km:.1f} km  "
            f"pixel={pixel_size_m:.0f} m  output={adj2_out_base}")
    else:
        log(f"  Stage-2 adj    : OFF")
    log(f"  Good bands     : {bbl.sum()} / {n_bands}")
    log(f"  Valid wl range : 0.40 - 2.50 um (6S limit)")
    log("=" * 62)

    # -- 6S per band: correction coefficients --------------------------------
    xa        = np.zeros(n_bands)
    xb        = np.ones(n_bands)
    xc        = np.zeros(n_bands)
    T_down_arr = np.ones(n_bands)   # downward transmittance, needed for stage 2
    T_up_arr   = np.ones(n_bands)   # upward transmittance, needed for stage 2
    S_arr      = np.zeros(n_bands)  # spherical albedo, needed for stage 2

    # TOA solar irradiance from the 6S built-in solar spectrum (no extra run).
    # solirr(wl_um) returns W/m2/um at 1 AU; correct for Earth-Sun distance.
    # Earth is near aphelion in July: (R0/R)^2 slightly below 1.
    from sixs.utils import solirr as _solirr
    import datetime as _dt
    _doy  = _dt.date(2000, month, day).timetuple().tm_yday
    _B    = 2 * np.pi * (_doy - 1) / 365.0
    _dsq  = (1.000110 + 0.034221*np.cos(_B) + 0.001280*np.sin(_B)
             + 0.000719*np.cos(2*_B) + 0.000077*np.sin(2*_B))
    E0_um = np.array([_solirr(w) * _dsq if 0.25 <= w <= 4.0 else 0.0
                      for w in wl_um])
    log(f"  E0 from built-in solar spectrum, Earth-Sun factor = {_dsq:.5f}")

    # Write a single 6S template config file with a wavelength placeholder.
    # The user can replace FILL_HERE_WAVELENGTH_IN_um with any wavelength
    # in the valid range (0.40-2.50 um) to re-run 6S for that band manually.
    # All other parameters are identical for every band.
    if config_file:
        template_inp = _make_6s_input(
            "FILL_HERE_WAVELENGTH_IN_um",
            sza, saa, vza, vaa, month, day,
            idatm, uh2o, uo3, iaer, aot550, target_alt_km,
            env_model=env_model, env_radius_km=env_radius_km,
        )
        with open(config_file, "w") as cfg_fh:
            cfg_fh.write(
                "# 6S configuration template\n"
                f"# Scene: {os.path.basename(hdr_file)}\n"
                f"# Date:  {acq_date}  SZA={sza:.3f} SAA={saa:.3f} "
                f"VZA={vza:.3f} VAA={vaa:.3f}\n"
                "# Replace FILL_HERE_WAVELENGTH_IN_um with the band centre\n"
                "# wavelength in micrometres (e.g. 0.427 for Hyperion band 8)\n"
                "#\n"
            )
            cfg_fh.write(template_inp)
        log(f"  6S template config written to {config_file}")

    # ── Collect spectral irradiance outputs for CSV ─────────────────────────
    # The CSV is written after the 6S loop completes (first iteration only).
    # Columns: wavelength_nm, atm_radiance, env_radiance,
    #          ground_direct_irr, ground_diffuse_irr, ground_env_irr.
    # All irradiance values are in W m⁻² µm⁻¹ (band-integrated).
    # atm_radiance and env_radiance are at the sensor (TOA), W m⁻² sr⁻¹ µm⁻¹.
    _spec_rows = []   # filled during the 6S loop below

    log(f"\nRunning 6S for {bbl.sum()} good bands...")
    n_ok = 0
    for b in range(n_bands):
        if bbl[b] == 0:
            continue
        wl = wl_um[b]
        if wl < 0.40 or wl > 2.50:
            bbl[b] = 0
            continue

        # Atmospheric correction coefficients
        inp = _make_6s_input(wl, sza, saa, vza, vaa, month, day,
                             idatm, uh2o, uo3, iaer, aot550, target_alt_km,
                             env_model=env_model, env_radius_km=env_radius_km)
        try:
            r = run6S(io.StringIO(inp), io.StringIO())
            xa[b], xb[b], xc[b], T_down_arr[b], T_up_arr[b], S_arr[b] = _coefficients(r)
            _spec_rows.append((
                wl_nm[b],
                r.get("atm_radiance",      float("nan")),
                r.get("env_radiance",      float("nan")),
                r.get("ground_direct_irr", float("nan")),
                r.get("ground_diffuse_irr",float("nan")),
                r.get("ground_env_irr",    float("nan")),
            ))
        except Exception as e:
            log(f"  Band {b+1} ({wl*1000:.1f} nm): 6S failed -- {e}")
            bbl[b] = 0
            continue

        n_ok += 1
        if n_ok % 10 == 0:
            log(f"  Band {b+1:3d}/{n_bands}  {wl*1000:7.1f} nm  "
                f"xa={xa[b]:.4f}  xb={xb[b]:.4f}  xc={xc[b]:.4f}  "
                f"E0={E0_um[b]:.1f} W/m2/um")

    log(f"\n6S done: {n_ok} bands processed.")

    if spec_csv_file and _spec_rows:
        try:
            with open(spec_csv_file, "w") as _csv_fh:
                _csv_fh.write(
                    "wavelength_nm,"
                    "atm_radiance_W_m2_sr_um,"
                    "env_radiance_W_m2_sr_um,"
                    "ground_direct_irr_W_m2_um,"
                    "ground_diffuse_irr_W_m2_um,"
                    "ground_env_irr_W_m2_um\n"
                )
                for row in _spec_rows:
                    _csv_fh.write(",".join(f"{v:.6g}" for v in row) + "\n")
            log(f"  Spectral CSV written: {spec_csv_file} ({len(_spec_rows)} bands)")
        except Exception as _e:
            log(f"  WARNING: could not write spectral CSV: {_e}")

    # -- Build output header and create output file before computations --------
    # The output file is created now (allocating disk space) so we can
    # write each band immediately after correction without holding the
    # full cube in memory.
    now_str  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    atm_desc = (f"idatm={idatm}" +
                (f" H2O={uh2o}g/cm2 O3={uo3}cm-atm" if idatm == 8 else ""))
    description = (
        f"EO-1 Hyperion surface reflectance x10000, 6S correction. "
        f"Source: {os.path.basename(hdr_file)}. "
        f"Date: {acq_date} {start_time}. "
        f"SZA={sza:.2f}deg SAA={saa:.2f}deg VZA={vza:.2f}deg. "
        f"Atm: {atm_desc} iaer={iaer} AOT550={aot550}. "
        f"Env: {env_label}"
        + (f" r={env_radius_km:.1f}km" if env_model is not None else "") + ". "
        + f"Input: {input_type}  Scale: {rad_scale_spec}. "
        f"Nodata=-9999. Processed {now_str}."
    )

    # Build a clean output header — do NOT copy everything from input.
    # Keep spatial/projection fields; replace content-specific ones.
    KEEP_KEYS = {
        "samples", "lines", "bands",
        "header offset", "file type", "byte order",
        "x start", "y start",
        "map info", "coordinate system string",
        "wavelength units", "wavelength", "fwhm", "bbl",
    }
    out_meta = {k: v for k, v in envi_meta.items() if k.lower() in KEEP_KEYS}

    # Determine which bands to write (drop_bad_bands=True by default).
    if drop_bad_bands:
        good_idx    = np.where(bbl == 1)[0]
        out_n_bands = len(good_idx)
        log(f"  Dropping bad bands: output will have {out_n_bands} / {n_bands} bands")
    else:
        good_idx    = np.arange(n_bands)
        out_n_bands = n_bands

    # Build per-band metadata filtered to the output band subset (good_idx).
    # Band names reference original band numbers so they remain traceable.
    wl_nm_all   = [float(w) for w in envi_meta.get("wavelength", [])]
    fwhm_all    = [float(f) for f in envi_meta.get("fwhm", [])]
    bbl_all     = [int(b)   for b in envi_meta.get("bbl",  [])]

    band_names_out = [f"B{good_idx[i]+1} ({wl_nm_all[good_idx[i]]:.0f}nm)"
                      for i in range(out_n_bands)]
    wl_out   = [wl_nm_all[i] for i in good_idx]
    fwhm_out = [fwhm_all[i]  for i in good_idx]
    bbl_out  = [bbl_all[i]   for i in good_idx]

    # Set correct dimensions before create_image allocates the file
    out_meta["lines"]             = str(n_lines)
    out_meta["samples"]           = str(n_samples)
    out_meta["bands"]             = str(out_n_bands)
    out_meta["description"]       = "{ " + description + " }"
    out_meta["sensor type"]       = sensor_name
    out_meta["data type"]         = "2"        # int16
    out_meta["data ignore value"] = "-9999"
    out_meta["band names"]        = band_names_out
    out_meta["wavelength"]        = wl_out
    out_meta["fwhm"]              = fwhm_out
    out_meta["bbl"]               = bbl_out
    out_meta["interleave"]        = interleave

    out_hdr = out_base + ".hdr"

    # Safety check: refuse to overwrite the input file.
    # Compare resolved absolute paths so symlinks and relative paths
    # cannot trick us into overwriting the source data.
    in_base  = os.path.splitext(os.path.abspath(hdr_file))[0]
    out_base_abs = os.path.abspath(out_base)
    if out_base_abs == in_base:
        raise ValueError(
            f"Output path '{out_base}' is the same as the input file — "
            "this would overwrite the source data. Choose a different output name."
        )

    # envi.create_image writes the header and allocates the data file.
    # open_memmap always returns (lines, samples, bands) — band is last axis.
    out_img_obj = envi.create_image(out_hdr, out_meta, dtype=np.int16, interleave=interleave, force=True)
    out_mm = out_img_obj.open_memmap(writable=True)
    out_mm[:] = -9999
    out_mm.flush()
    log(f"Output file created: {out_hdr}  ({out_mm.nbytes/1e6:.1f} MB)")

    # -- Open input via spectral — returns (lines, samples, bands) -----------
    log("\nOpening input image as read-only memory-map...")
    in_obj = envi.open(hdr_file)
    in_mm  = in_obj.open_memmap(writable=False)
    log(f"  {hdr_file}")
    log(f"  shape={in_mm.shape}  dtype={in_mm.dtype}")

    # Build per-band radiance scale from the sensor profile spec
    rad_scale = build_rad_scale(rad_scale_spec, n_bands)

    # -- Build valid-pixel mask (lines x samples, bool) -----------------------
    if mask_file and os.path.isfile(mask_file):
        # Load external mask: any non-zero value = valid pixel.
        # Accept any single-band ENVI file or a raw binary file whose
        # shape matches the image.
        try:
            mask_hdr = mask_file if mask_file.lower().endswith(".hdr")                        else mask_file + ".hdr"
            if os.path.isfile(mask_hdr):
                mask_mm = envi.open(mask_hdr).open_memmap(writable=False)
                # squeeze removes any singleton band dimension for single-band masks
                valid = np.squeeze(mask_mm).astype(bool)
                del mask_mm
            else:
                log(f"  No .hdr found for mask, skipping.")
                mask_file = None
            log(f"  Mask loaded: {valid.sum()} / {valid.size} valid pixels")
        except Exception as e:
            log(f"  Mask load failed ({e}), falling back to auto-mask.")
            mask_file = None

    if not mask_file:
        # Auto-mask: a pixel is invalid if ALL input bands are zero.
        log("  Building auto-mask (pixels where all bands are zero)...")
        # spectral memmap is (lines, samples, bands) -> sum bands on axis 2
        band_sum = in_mm.astype(np.int32).sum(axis=2)  # -> (lines, samples)
        valid    = band_sum != 0
        log(f"  Auto-mask: {valid.sum()} / {valid.size} valid pixels")

    # -- Apply correction band by band, writing directly to output memmap ------
    log(f"Applying correction to {out_n_bands} bands...")
    _pct1 = -1
    for out_b, b in enumerate(good_idx):
        pct1 = int((out_b + 1) / out_n_bands * 100)
        if pct1 // 10 > _pct1 // 10:
            _pct1 = pct1
            log(f"  {pct1:3d}%  (band {b+1})")
        # spectral memmap: (lines, samples, bands) — band is last
        raw = in_mm[:, :, b].astype(np.float32)

        if input_type == "toa_refl":
            # Data already stores TOA reflectance (e.g. Sentinel-2 L1C)
            rho_toa = raw / rad_scale[b]   # rad_scale = 1 or a scale factor
        else:
            # Radiance path: DN -> L [W/m2/sr/um] -> rho_toa.
            # For Hyperion L1GST: rad_scale=40 (VNIR) / 80 (SWIR).
            # This is DN/400*10 or DN/800*10 (FLAASH->SI: 1 uW/cm2/sr/nm = 10 W/m2/sr/um).
            # E0 from solirr() is W/m2/um: consistent units, ratio is dimensionless.
            # 6S uses Wehrli E0 internally; TSIS-1 vs Wehrli <2% difference.
            L = raw / rad_scale[b]
            if E0_um[b] < 1e-6:
                continue
            rho_toa = (np.pi * L) / (E0_um[b] * cos_sza)
        rho_toa = np.clip(rho_toa, 0.0, 1.5)

        # Surface reflectance retrieval.
        #
        # The 6S forward model (Vermote et al. 1997, eq. 2) is:
        #   rho_toa = xa + T_up * T_down * rho_s / (1 - S * rho_s)
        #
        # Solving exactly for rho_s:
        #   (rho_toa - xa)(1 - S*rho_s) = T_up * T_down * rho_s
        #   rho_toa - xa = rho_s * [T_up*T_down + S*(rho_toa - xa)]
        #   rho_s = (rho_toa - xa) / (T_up*T_down + S*(rho_toa - xa))
        #
        # NOTE: The Vermote (1997) eq. 7 approximation:
        #   rho_s ≈ (rho_toa - xa) / (xb*rho_toa + xc)
        # where xb = 1/(T_down*T_up) and xc = S*xb, i.e. denominator = (rho_toa+S)/(T_d*T_u),
        # is NOT the exact inverse of eq. 2. The exact denominator is T_d*T_u + S*(rho_toa-xa),
        # not (rho_toa+S)/(T_d*T_u). The approximation underestimates rho_s by 1-6 pp
        # in the blue band at AOT=0.20, growing with rho_s and with S (spherical albedo).
        # The exact formula is used here instead.
        #
        # With xb = 1/(T_d*T_u): denominator = 1/xb + S*(rho_toa - xa)
        numer = rho_toa - xa[b]
        denom = (1.0 / xb[b]) + S_arr[b] * numer   # exact: T_d*T_u + S*(rho_toa-xa)
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_s = np.where(np.abs(denom) > 1e-6, numer / denom, 0.0)
        rho_s = np.clip(rho_s, -0.1, 1.5)

        # Scale to int16, apply mask (nodata=-9999 for invalid pixels)
        band_out = np.clip(rho_s * 10000, -32768, 32767).astype(np.int16)
        band_out[~valid] = -9999

        # Write directly into output memmap
        out_mm[:, :, out_b] = band_out   # (lines, samples, bands)

    # Flush and close memmaps
    out_mm.flush()
    del out_mm

    def _fmt_val(k, v):
        """Format a metadata value for the ENVI header.
        Wavelength / fwhm values are rounded to 2 decimal places (nm resolution
        is more than sufficient for 10-nm Hyperion bands).
        Lists/tuples are written as {a, b, c, ...}.
        """
        if isinstance(v, (list, tuple)):
            key_lower = k.lower()
            if key_lower in ("wavelength", "fwhm"):
                # Round to 2 decimal places — sub-nm precision is meaningless
                items = ", ".join(f"{float(x):.2f}" for x in v)
            elif key_lower == "band names":
                items = ", ".join(str(x) for x in v)
            else:
                items = ", ".join(str(x) for x in v)
            return f"{k} = {{{items}}}"
        return f"{k} = {v}"

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Adjacency correction
    # ══════════════════════════════════════════════════════════════════════════
    # The Stage-1 output assumes every pixel is surrounded by a uniform surface
    # of the same reflectance as the pixel itself (inhomo=0) or a fixed
    # environment type (inhomo=1, igrou2).  Neither accounts for spatial
    # variation in the surrounding reflectance.
    #
    # Stage 2 estimates the "environmental reflectance" rho_env(x,y) for each
    # pixel by Gaussian-smoothing the Stage-1 surface reflectance image with a
    # kernel whose sigma is derived from the adjacency radius.  This captures the
    # influence of the neighbourhood reflectance on the measured signal.
    #
    # The corrected reflectance is:
    #   rho_s2 = (rho_toa - xa) / T_down  -  rho_env * S / T_down
    #          = (rho_s1 * T_up * T_down + xc * T_down * rho_s1
    #             - S * rho_env) / T_down
    # Simplified from first principles (Vermote & Tanré 1992):
    #   rho_s2 = (rho_toa - xa) / T_down  -  S * rho_env / T_down
    # which equals rho_s1 when rho_env == rho_s1 (uniform surface).
    #
    # Gaussian sigma:  sigma_pixels = (radius_km * 1000) / pixel_size_m
    # The kernel integrates the atmospheric PSF; for low AOT a radius of 0.5-1 km
    # is typical, for heavy aerosol up to 3-5 km.
    if do_adj2 and adj2_out_base:
        from scipy.ndimage import gaussian_filter
        log(f"\nStage-2 adjacency correction  radius={adj2_radius_km:.1f} km ...")

        sigma = (adj2_radius_km * 1000.0) / pixel_size_m
        log(f"  Gaussian sigma = {sigma:.1f} pixels  "
            f"({adj2_radius_km:.1f} km / {pixel_size_m:.0f} m pixel)")

        # Create Stage-2 output file
        # Safety: refuse to overwrite input or Stage-1 output
        if os.path.abspath(adj2_out_base) in (in_base, out_base_abs):
            raise ValueError(
                f"Stage-2 output path '{adj2_out_base}' conflicts with "
                "input or Stage-1 output — choose a different name."
            )
        adj2_hdr = adj2_out_base + ".hdr"
        adj2_meta = dict(out_meta)
        adj2_meta["description"] = (
            "{ Stage-2 adjacency-corrected surface reflectance x10000. "
            f"Gaussian radius={adj2_radius_km:.1f} km ({sigma:.1f} px). "
            + out_meta["description"][2:]  # reuse stage-1 description
        )
        adj2_img_obj = envi.create_image(adj2_hdr, adj2_meta, dtype=np.int16, interleave=interleave, force=True)
        adj2_mm = adj2_img_obj.open_memmap(writable=True)
        adj2_mm[:] = -9999

        # Re-open Stage-1 output as read source via spectral
        s1_obj = envi.open(out_hdr)
        s1_mm  = s1_obj.open_memmap(writable=False)

        log(f"  Smoothing and correcting {out_n_bands} bands...")
        _pct2 = -1
        for out_b, b in enumerate(good_idx):
            pct2 = int((out_b + 1) / out_n_bands * 100)
            if pct2 // 10 > _pct2 // 10:
                _pct2 = pct2
                log(f"    {pct2:3d}%  (band {b+1})")
            if T_down_arr[b] < 1e-4:
                continue

            # Stage-1 reflectance for this band (float, nodata=NaN)
            rho_s1 = s1_mm[:, :, out_b].astype(np.float32) / 10000.0
            rho_s1[rho_s1 < -0.09] = np.nan     # mask nodata

            # Environmental reflectance = spatially smoothed Stage-1
            # NaN pixels are replaced by local mean before smoothing
            fill = np.where(np.isnan(rho_s1),
                            np.nanmean(rho_s1), rho_s1)
            rho_env = gaussian_filter(fill.astype(np.float64), sigma=sigma
                                       ).astype(np.float32)

            # Stage-2 adjacency correction
            #
            # Stage-1 used the uniform-surface assumption: every pixel is
            # surrounded by a surface of identical reflectance.  The forward
            # model (Vermote et al. 1997, eq. 2) for a NON-UNIFORM surface is:
            #
            #   rho_toa = xa + T_up * T_down * rho_pixel / (1 - S * rho_env)
            #
            # where rho_env is the effective background reflectance seen by the
            # atmosphere (here approximated as the Gaussian-smoothed Stage-1 image).
            #
            # Solving for rho_pixel:
            #   rho_pixel = (rho_toa - xa) * (1 - S * rho_env) / (T_up * T_down)
            #             = (rho_toa - xa) * (1 - S * rho_env) * xb
            #   (since xb = 1 / (T_down * T_up))
            #
            # When rho_env = rho_pixel (uniform case) this reduces exactly to
            # the Stage-1 formula.
            #
            # Step 1: recover rho_toa per pixel by exactly inverting Stage-1.
            #   The EXACT Stage-1 forward model (from eq. 2) is:
            #     rho_toa = xa + (T_d*T_u*rho_s1) / (1 - S*rho_s1)
            #   Exact inverse:
            #     rho_toa = xa + rho_s1*(1/xb) / (1 - S*rho_s1)
            #             = [xa*(1-S*rho_s1) + rho_s1/xb] / (1 - S*rho_s1)
            #
            # Note: the Vermote approximation rho_toa=(xa+xc*rho_s1)/(1-xb*rho_s1)
            # is NOT the exact inverse. Using the exact form avoids round-trip error.

            _xb = xb[b];  _xc = xc[b];  _xa = xa[b]
            _S  = S_arr[b]
            _TdTu = 1.0 / _xb   # T_down * T_up

            # Exact rho_toa recovery: rho_toa = xa + T_d*T_u*rho_s1/(1-S*rho_s1)
            denom_inv = 1.0 - _S * rho_s1
            with np.errstate(divide="ignore", invalid="ignore"):
                rho_toa_rec = np.where(
                    np.abs(denom_inv) > 1e-6,
                    _xa + _TdTu * rho_s1 / denom_inv,
                    np.nan)

            # Step 2: apply the non-uniform forward model inverted for rho_pixel
            rho_s2 = (rho_toa_rec - _xa) * (1.0 - _S * rho_env) * _xb
            rho_s2 = np.clip(rho_s2, -0.1, 1.5)

            # Restore nodata mask and write
            band_out = np.clip(rho_s2 * 10000, -32768, 32767).astype(np.int16)
            band_out[np.isnan(rho_s1)] = -9999
            band_out[~valid] = -9999
            adj2_mm[:, :, out_b] = band_out

        log("    100%  done.")
        adj2_mm.flush()
        del adj2_mm, s1_mm

        # Write Stage-2 header with correct dimensions
        adj2_meta["lines"]   = str(n_lines)
        adj2_meta["samples"] = str(n_samples)
        adj2_meta["bands"]   = str(out_n_bands)
        with open(adj2_hdr, "w") as _f:
            _f.write("ENVI\n")
            for k, v in adj2_meta.items():
                _f.write(_fmt_val(k, v) + "\n")

        log(f"  Stage-2 output: {adj2_hdr}")

    del in_mm

    log("  100%  done.")
    log(f"\nOutput written: {out_hdr}")
    log(f"\nDescription:\n  {description}")
    return out_hdr


# =============================================================================
# SECTION 4 -- Tk GUI
# =============================================================================

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog

    root = tk.Tk()
    root.title("Hyperion 6S Atmospheric Correction")
    root.geometry("800x680")
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))

    # ------------------------------------------------------------------
    # All user-editable fields are plain Entry widgets stored in a dict.
    # We never rely on tk.StringVar / tk.DoubleVar so Spyder's reuse of
    # the Tk root cannot silently break the bindings.
    # Read  : e[key].get()
    # Write : _set(key, value)
    # ------------------------------------------------------------------
    e = {}   # key -> Entry or Combobox widget

    def _set(key, value):
        """Write value into widget, works for Entry and Combobox."""
        w = e[key]
        if isinstance(w, ttk.Combobox):
            w.set(str(value))
        else:
            w.delete(0, tk.END)
            w.insert(0, str(value))

    def _get(key, typ=str):
        try:
            return typ(e[key].get())
        except (ValueError, KeyError):
            return typ()

    def _entry(parent, key, default, width=14, **grid_kw):
        """Create an Entry, pre-fill it, store in e[key], grid it."""
        w = ttk.Entry(parent, width=width)
        w.insert(0, str(default))
        w.grid(**grid_kw)
        e[key] = w
        return w

    def _combo(parent, key, values, default, width=30, **grid_kw):
        """Create a readonly Combobox, store in e[key], grid it."""
        w = ttk.Combobox(parent, values=values, state="readonly", width=width)
        w.set(default)
        w.grid(**grid_kw)
        e[key] = w
        return w

    # ── status bar ────────────────────────────────────────────────────
    status_lbl = ttk.Label(root, text="Ready.", anchor=tk.W, relief=tk.SUNKEN)
    status_lbl.pack(side=tk.BOTTOM, fill=tk.X)

    def set_status(msg):
        status_lbl.config(text=msg)

    # ── button bar ────────────────────────────────────────────────────
    btn_bar = ttk.Frame(root)
    btn_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

    # ── notebook ──────────────────────────────────────────────────────
    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    # ═══════════════════════════════════════════════
    # TAB 1 — Files & Geometry
    # ═══════════════════════════════════════════════
    tab1 = ttk.Frame(nb, padding=10)
    nb.add(tab1, text="Files & Geometry")

    # Files frame
    fg = ttk.LabelFrame(tab1, text="Files", padding=6)
    fg.pack(fill=tk.X, pady=(0, 10))
    fg.columnconfigure(1, weight=1)

    def _file_row(parent, row, label, key, cmd, hint=None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        w = ttk.Entry(parent)
        w.grid(row=row, column=1, sticky=tk.EW, padx=4)
        e[key] = w
        if cmd:
            ttk.Button(parent, text="Browse...", command=cmd).grid(row=row, column=2)
        elif hint:
            ttk.Label(parent, text=hint, foreground="gray").grid(
                row=row, column=2, sticky=tk.W)

    def browse_hdr():
        p = filedialog.askopenfilename(
            title="Select ENVI header (.hdr)",
            filetypes=[("HDR files", "*.hdr"), ("All files", "*.*")])
        if p:
            _set("hdr_file", p)
            if not _get("out_base"):
                _set("out_base", os.path.splitext(p)[0] + "_6S")
            # Pre-fill Stage-2 output name
            _set("adj2_out_base", os.path.splitext(p)[0] + "_6Sadj")

    def browse_mtl():
        p = filedialog.askopenfilename(
            title="Select Hyperion MTL file",
            filetypes=[("L1T files", "*.L1T"), ("TXT files", "*.txt"),
                       ("All files", "*.*")])
        if not p:
            return
        _set("mtl_file", p)
        try:
            m = read_mtl(p)
            _set("sza",        round(m["sza"], 3))
            _set("saa",        round(m["saa"], 3))
            _set("vza",        round(m["vza"], 3))
            _set("vaa",        round(m["vaa"], 3))
            _set("month",      m["month"])
            _set("day",        m["day"])
            # Update rad_scale_spec if profile still matches Hyperion defaults
            _set("rad_scale_spec", m["rad_scale_spec"])
            set_status(
                f"MTL loaded: {m['acq_date']}  "
                f"SZA={m['sza']:.2f}  VZA={m['vza']:.2f}")
        except Exception as ex:
            set_status(f"MTL error: {ex}")

    def browse_mask():
        p = filedialog.askopenfilename(
            title="Select mask file (optional)",
            filetypes=[("ENVI HDR", "*.hdr"), ("All files", "*.*")])
        if p:
            _set("mask_file", p)

    _file_row(fg, 0, "HDR file:",           "hdr_file",  browse_hdr)
    _file_row(fg, 1, "MTL file:",           "mtl_file",  browse_mtl)
    _file_row(fg, 2, "Output 1st stage:",   "out_base",  None)

    # Stage-2 output row — enable/disable controlled by adj2_cb below
    ttk.Label(fg, text="Output 2nd stage:").grid(row=3, column=0, sticky=tk.W, pady=3)
    adj2_out_e = ttk.Entry(fg)
    adj2_out_e.grid(row=3, column=1, sticky=tk.EW, padx=4)
    e["adj2_out_base"] = adj2_out_e

    def browse_adj2_out():
        hdr_base = _get("hdr_file")
        init = os.path.splitext(hdr_base)[0] + "_6Sadj" if hdr_base else ""
        p = filedialog.asksaveasfilename(
            title="Stage-2 output base name (no extension)",
            initialfile=os.path.basename(init),
            initialdir=os.path.dirname(init) if init else ".")
        if p:
            _set("adj2_out_base", p)

    adj2_browse_btn = ttk.Button(fg, text="Browse...", command=browse_adj2_out)
    adj2_browse_btn.grid(row=3, column=2)

    _file_row(fg, 4, "Mask (optional):",   "mask_file", browse_mask,
              "(empty = auto from zero pixels)")

    def browse_config():
        p = filedialog.asksaveasfilename(
            title="Save 6S config file as",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p:
            config_entry.config(state=tk.NORMAL)
            _set("config_file", p)

    ttk.Label(fg, text="Save 6S config:").grid(row=5, column=0, sticky=tk.W, pady=3)
    config_entry = ttk.Entry(fg)
    config_entry.insert(0, "<not saved>")
    config_entry.config(state=tk.DISABLED)
    config_entry.grid(row=5, column=1, sticky=tk.EW, padx=4)
    e["config_file"] = config_entry
    ttk.Button(fg, text="Browse...", command=browse_config).grid(row=5, column=2)

    def browse_spec_csv():
        p = filedialog.asksaveasfilename(
            title="Save spectral irradiance CSV as",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            spec_csv_entry.config(state=tk.NORMAL)
            _set("spec_csv_file", p)

    ttk.Label(fg, text="Save spectral CSV:").grid(row=6, column=0, sticky=tk.W, pady=3)
    spec_csv_entry = ttk.Entry(fg)
    spec_csv_entry.insert(0, "<not saved>")
    spec_csv_entry.config(state=tk.DISABLED)
    spec_csv_entry.grid(row=6, column=1, sticky=tk.EW, padx=4)
    e["spec_csv_file"] = spec_csv_entry
    ttk.Button(fg, text="Browse...", command=browse_spec_csv).grid(row=6, column=2)
    ttk.Label(fg,
              text="atm/env radiance + ground direct/diffuse/env irradiance, W m⁻² µm⁻¹",
              foreground="gray").grid(row=7, column=0, columnspan=3, sticky=tk.W)

    # ── Checkboxes ──────────────────────────────────────────────────────────
    # All tk variable types (BooleanVar, IntVar) can show an indeterminate
    # visual state in Spyder because Spyder reuses the Tk root between runs,
    # leaving stale traces on variable names.
    # Solution: do NOT bind a variable at all. Instead manage the state
    # entirely through the widget's own .state() / .instate() API and
    # store the boolean in a plain Python list so there is nothing for
    # Tk to get confused about.

    # Exclude bad bands — on by default
    _drop = [True]   # mutable container; _drop[0] is the current value

    def _drop_toggle():
        _drop[0] = not _drop[0]
        drop_cb.state(["!alternate", "selected"] if _drop[0] else ["!alternate", "!selected"])

    drop_cb = ttk.Checkbutton(fg,
                               text="Exclude bad bands from output  (saves disk space)",
                               command=_drop_toggle)
    drop_cb.grid(row=6, column=0, columnspan=3, sticky=tk.W, pady=(6, 2))
    drop_cb.state(["!alternate", "selected"])   # clear alternate, set ticked

    # Enable Stage-2 adjacency correction — on by default
    _adj2 = [True]

    def on_adj2_toggle():
        _adj2[0] = not _adj2[0]
        adj2_cb.state(["!alternate", "selected"] if _adj2[0] else ["!alternate", "!selected"])
        state = tk.NORMAL if _adj2[0] else tk.DISABLED
        adj2_out_e.config(state=state)
        adj2_browse_btn.config(state=state)

    adj2_cb = ttk.Checkbutton(fg,
                               text="Enable Stage-2 adjacency correction",
                               command=on_adj2_toggle)
    adj2_cb.grid(row=7, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
    adj2_cb.state(["!alternate", "selected"])   # clear alternate, set ticked

    # Apply initial enable state (Stage-2 on -> fields active)
    adj2_out_e.config(state=tk.NORMAL)
    adj2_browse_btn.config(state=tk.NORMAL)

    # Output file interleave
    il_frame = ttk.LabelFrame(tab1, text="Output file interleave", padding=6)
    il_frame.pack(fill=tk.X, pady=(6, 0))

    ttk.Label(il_frame, text="Interleave:").grid(row=0, column=0, sticky=tk.W)
    il_cb = _combo(il_frame, "interleave",
                   ["bip", "bil", "bsq"], "bip",
                   width=8, row=0, column=1, sticky=tk.W, padx=6)
    ttk.Label(il_frame,
              text="BIP = pixel (best for spectral access)  "
                   "BIL = line  BSQ = band",
              foreground="gray").grid(row=0, column=2, sticky=tk.W)

    # Geometry frame — all values must come from the MTL file.
    # Only VAA defaults to 0 (Hyperion pushbroom flies along-track).
    gg = ttk.LabelFrame(tab1, text="Geometry  (auto-filled from MTL)", padding=8)
    gg.pack(fill=tk.X, pady=(0, 6))
    for r, (lbl, key, default) in enumerate([
        ("Solar zenith SZA (deg):", "sza",   ""),   # no default: load from MTL
        ("Solar azimuth SAA (deg):", "saa",  ""),   # no default: load from MTL
        ("View zenith VZA (deg):",  "vza",   ""),   # no default: load from MTL
        ("View azimuth VAA (deg):", "vaa",    0.0), # 0 is correct for pushbroom
        ("Month:",                  "month",  ""),  # no default: load from MTL
        ("Day:",                    "day",    ""),  # no default: load from MTL
    ]):
        ttk.Label(gg, text=lbl).grid(row=r, column=0, sticky=tk.W, pady=1)
        _entry(gg, key, default, width=12, row=r, column=1, sticky=tk.W, padx=6)

    # Sensor profile selector + per-sensor fields
    sg = ttk.LabelFrame(tab1, text="Sensor", padding=6)
    sg.pack(fill=tk.X, pady=(0, 6))
    sg.columnconfigure(1, weight=1)

    ttk.Label(sg, text="Sensor profile:").grid(row=0, column=0, sticky=tk.W, pady=2)
    profile_cb = _combo(sg, "sensor_profile",
                        list(SENSOR_PROFILES.keys()), DEFAULT_PROFILE,
                        width=28, row=0, column=1, sticky=tk.W, padx=6)
    ttk.Label(sg, text="(adding sensors requires modifying SENSOR_PROFILES in the code)",
              foreground="gray").grid(row=0, column=2, sticky=tk.W)

    ttk.Label(sg, text="Sensor name (in header):").grid(row=1, column=0, sticky=tk.W, pady=2)
    _entry(sg, "sensor_name", SENSOR_PROFILES[DEFAULT_PROFILE]["sensor_type"],
           width=28, row=1, column=1, sticky=tk.W, padx=6)

    ttk.Label(sg, text="Input type:").grid(row=2, column=0, sticky=tk.W, pady=2)
    _combo(sg, "input_type",
           ["radiance", "toa_refl"], "radiance",
           width=14, row=2, column=1, sticky=tk.W, padx=6)
    ttk.Label(sg, text="radiance: DN→L→ρ_toa→ρ_s   toa_refl: skip L and E0 steps",
              foreground="gray").grid(row=2, column=2, sticky=tk.W)

    ttk.Label(sg, text="Radiance scale:").grid(row=3, column=0, sticky=tk.W, pady=2)
    _entry(sg, "rad_scale_spec",
           SENSOR_PROFILES[DEFAULT_PROFILE]["rad_scale"],
           width=20, row=3, column=1, sticky=tk.W, padx=6)
    ttk.Label(sg, text='e.g. "split:40:80:70"  "uniform:1"  "none"',
              foreground="gray").grid(row=3, column=2, sticky=tk.W)

    def on_profile_change(*_):
        """Fill sensor fields from the selected profile."""
        name = _get("sensor_profile")
        if name not in SENSOR_PROFILES:
            return
        p = SENSOR_PROFILES[name]
        _set("sensor_name",    p["sensor_type"])
        _set("input_type",     p["input_type"])
        _set("rad_scale_spec", p["rad_scale"])
        _set("pixel_size_m",   p["pixel_size_m"])
        # VAA: if orbit_eo1, restore computed value; if manual, leave as-is
        if p["vaa_source"] == "manual":
            _set("vaa", 0.0)

    profile_cb.bind("<<ComboboxSelected>>", on_profile_change)

    # ═══════════════════════════════════════════════
    # TAB 2 — Atmosphere
    # ═══════════════════════════════════════════════
    tab2 = ttk.Frame(nb, padding=10)
    nb.add(tab2, text="Atmosphere")

    # Atmospheric profile
    ag = ttk.LabelFrame(tab2, text="Atmospheric profile", padding=8)
    ag.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(ag, text="Profile:").grid(row=0, column=0, sticky=tk.W, pady=3)
    atm_cb = _combo(ag, "idatm_name", list(ATM_MODELS.keys()), DEFAULT_ATM,
                    width=32, row=0, column=1, sticky=tk.W, padx=6)

    ttk.Label(ag, text="H2O (g/cm2):").grid(row=1, column=0, sticky=tk.W, pady=3)
    h2o_entry = _entry(ag, "uh2o", 1.75, width=10, row=1, column=1,
                        sticky=tk.W, padx=6)
    ttk.Label(ag, text="only for idatm=8",
              foreground="gray").grid(row=1, column=2, sticky=tk.W)

    ttk.Label(ag, text="O3 (cm-atm):").grid(row=2, column=0, sticky=tk.W, pady=3)
    o3_entry = _entry(ag, "uo3", 0.35, width=10, row=2, column=1,
                       sticky=tk.W, padx=6)
    ttk.Label(ag, text="only for idatm=8",
              foreground="gray").grid(row=2, column=2, sticky=tk.W)

    def on_atm_change(*_):
        is_user = ATM_MODELS.get(e["idatm_name"].get(), 0) == 8
        state = tk.NORMAL if is_user else tk.DISABLED
        h2o_entry.config(state=state)
        o3_entry.config(state=state)

    atm_cb.bind("<<ComboboxSelected>>", on_atm_change)
    on_atm_change()   # set initial state

    # Aerosol
    aerg = ttk.LabelFrame(tab2, text="Aerosol", padding=8)
    aerg.pack(fill=tk.X, pady=(0, 10))

    ttk.Label(aerg, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=3)
    _combo(aerg, "iaer_name", list(AEROSOL_MODELS.keys()), DEFAULT_AER,
           width=25, row=0, column=1, sticky=tk.W, padx=6)

    ttk.Label(aerg, text="AOT @ 550 nm:").grid(row=1, column=0, sticky=tk.W, pady=3)
    _entry(aerg, "aot550", 0.06, width=10, row=1, column=1, sticky=tk.W, padx=6)

    # Target
    tg = ttk.LabelFrame(tab2, text="Target", padding=8)
    tg.pack(fill=tk.X, pady=(0, 10))
    ttk.Label(tg, text="Altitude (km a.s.l.):").grid(row=0, column=0, sticky=tk.W)
    _entry(tg, "target_alt_km", 0.20, width=10, row=0, column=1,
            sticky=tk.W, padx=6)

    # Adjacency / environment
    envg = ttk.LabelFrame(tab2, text="Adjacency correction (inhomo)", padding=8)
    envg.pack(fill=tk.X)

    ttk.Label(envg, text="Environment type:").grid(row=0, column=0, sticky=tk.W, pady=3)
    _combo(envg, "env_model_name", list(ENV_MODELS.keys()), DEFAULT_ENV,
           width=32, row=0, column=1, sticky=tk.W, padx=6, columnspan=2)

    ttk.Label(envg, text="Patch radius (km):").grid(row=1, column=0, sticky=tk.W, pady=3)
    _entry(envg, "env_radius_km", DEFAULT_ENV_RADIUS_KM, width=10,
           row=1, column=1, sticky=tk.W, padx=6)
    ttk.Label(envg, text="radius of uniform target patch",
              foreground="gray").grid(row=1, column=2, sticky=tk.W)

    # ── Stage-2 adjacency correction ─────────────────────────────────────────
    # Parameters are here (Atmosphere tab) because they describe the atmospheric
    # smoothing kernel; the output file name is in Files & Geometry tab.
    adj2g = ttk.LabelFrame(tab2, text="Stage-2 adjacency correction", padding=8)
    adj2g.pack(fill=tk.X, pady=(8, 0))
    adj2g.columnconfigure(1, weight=1)

    # Smoothing radius
    ttk.Label(adj2g, text="Smoothing radius (km):").grid(row=0, column=0, sticky=tk.W, pady=2)
    adj2_radius_e = ttk.Entry(adj2g, width=10)
    adj2_radius_e.insert(0, "1.0")
    adj2_radius_e.grid(row=0, column=1, sticky=tk.W, padx=4)
    e["adj2_radius_km"] = adj2_radius_e
    ttk.Label(adj2g, text="Gaussian kernel radius for rho_env estimate",
              foreground="gray").grid(row=0, column=2, columnspan=2, sticky=tk.W)

    # Pixel size
    ttk.Label(adj2g, text="Pixel size (m):").grid(row=1, column=0, sticky=tk.W, pady=2)
    adj2_pixel_e = ttk.Entry(adj2g, width=10)
    adj2_pixel_e.insert(0, "30")
    adj2_pixel_e.grid(row=1, column=1, sticky=tk.W, padx=4)
    e["pixel_size_m"] = adj2_pixel_e
    ttk.Label(adj2g, text="Hyperion GSD = 30 m",
              foreground="gray").grid(row=1, column=2, sticky=tk.W)

    # ═══════════════════════════════════════════════
    # TAB 3 — Log
    # ═══════════════════════════════════════════════
    tab3 = ttk.Frame(nb, padding=4)
    nb.add(tab3, text="Log")
    log_text = tk.Text(tab3, font=("Courier", 9), wrap=tk.NONE)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ttk.Scrollbar(tab3, orient=tk.VERTICAL,
                  command=log_text.yview).pack(side=tk.RIGHT, fill=tk.Y)

    # ── Run button ────────────────────────────────────────────────────
    def run_correction():
        run_btn.config(state=tk.DISABLED)
        log_text.delete("1.0", tk.END)
        nb.select(tab3)
        set_status("Running...")
        root.update()

        hdr_path = _get("hdr_file")
        if not hdr_path or not os.path.isfile(hdr_path):
            set_status("Error: HDR file not found.")
            run_btn.config(state=tk.NORMAL)
            return

        try:
            hdr_data = read_hdr(hdr_path)
        except Exception as ex:
            set_status(f"HDR error: {ex}")
            run_btn.config(state=tk.NORMAL)
            return

        mask_path = _get("mask_file") or None
        if mask_path and not os.path.isfile(mask_path):
            mask_path = None

        params = dict(
            hdr_file      = hdr_path,
            out_base      = _get("out_base") or os.path.splitext(hdr_path)[0] + "_6S",
            acq_date      = f"{_get('month')}/{_get('day')}",
            start_time    = "",
            month         = _get("month", int),
            day           = _get("day",   int),
            sza           = _get("sza",   float),
            saa           = _get("saa",   float),
            vza           = _get("vza",   float),
            vaa           = _get("vaa",   float),
            sensor_name    = _get("sensor_name"),
            input_type     = _get("input_type"),
            rad_scale_spec = _get("rad_scale_spec"),
            idatm         = ATM_MODELS.get(_get("idatm_name"), 2),
            uh2o          = _get("uh2o",          float),
            uo3           = _get("uo3",            float),
            iaer          = AEROSOL_MODELS.get(_get("iaer_name"), 1),
            aot550        = _get("aot550",         float),
            target_alt_km = _get("target_alt_km",  float),
            mask_file     = mask_path,
            config_file   = (_get("config_file") or None)
                            if _get("config_file") not in ("", "<not saved>") else None,
            spec_csv_file = (_get("spec_csv_file") or None)
                            if _get("spec_csv_file") not in ("", "<not saved>") else None,
            interleave      = _get("interleave") or "bip",
            drop_bad_bands  = _drop[0],
            do_adj2         = _adj2[0],
            adj2_out_base   = (_get("adj2_out_base") or None) if _adj2[0] else None,
            adj2_radius_km  = _get("adj2_radius_km", float),
            pixel_size_m    = _get("pixel_size_m",   float),
            env_model       = ENV_MODELS.get(_get("env_model_name"), None),
            env_radius_km = _get("env_radius_km", float),
            **{k: hdr_data[k] for k in
               ["n_bands","n_lines","n_samples","wl_nm","wl_um","fwhm_nm",
                "bbl","envi_meta"]},
        )

        def log(msg, end="\n"):
            # end= is ignored in the GUI — always appends a new line.
            log_text.insert(tk.END, msg + "\n")
            log_text.see(tk.END)
            root.update()

        try:
            out_hdr = correct_hyperion(params, log=log)
            set_status(f"Done -> {os.path.basename(out_hdr)}")
        except Exception as ex:
            import traceback
            log(traceback.format_exc())
            set_status(f"Error: {ex}")
        finally:
            run_btn.config(state=tk.NORMAL)

    run_btn = ttk.Button(btn_bar, text="Run correction", command=run_correction)
    run_btn.pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_bar, text="Close",
               command=lambda: (root.quit(), root.destroy())).pack(side=tk.LEFT)

    root.mainloop()

# =============================================================================
# SECTION 5 -- Entry point
# =============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Script mode: hyperion_atm_correction.py file.hdr [file.L1T]
        hdr_path = sys.argv[1]
        mtl_path = sys.argv[2] if len(sys.argv) > 2 else None
        base     = os.path.splitext(hdr_path)[0]

        mtl_data = read_mtl(mtl_path) if mtl_path else dict(
            acq_date="unknown", start_time="",
            month=7, day=3,
            sza=43.7, saa=139.1, vza=18.6, vaa=0.0,
            sensor_name="EO-1 Hyperion",
        input_type="radiance",
        rad_scale_spec="split:40:80:70",
            n_lines=0, n_samples=0,
        )
        hdr_data = read_hdr(hdr_path)

        params = dict(
            hdr_file=hdr_path,
            out_base=base + "_6S",
            idatm=2, uh2o=1.75, uo3=0.35,
            iaer=1,  aot550=0.06, target_alt_km=0.20,
        mask_file=None, config_file=None, interleave="bip", drop_bad_bands=True,
        do_adj2=False, adj2_out_base=None, adj2_radius_km=1.0,
        pixel_size_m=30.0,
        env_model=None,  env_radius_km=2.0,
            **mtl_data,
            **{k: hdr_data[k] for k in
               ["n_bands","wl_nm","wl_um","fwhm_nm","bbl","envi_meta"]},
        )
        correct_hyperion(params)
    else:
        run_gui()