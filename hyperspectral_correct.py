"""
hyperspectral_correct.py
--------------------------
Atmospheric correction of EO-1 Hyperion L1T hyperspectral imagery using 6S
(Second Simulation of the Satellite Signal in the Solar Spectrum).

Usage
-----
  GUI    : python hyperspectral_correct.py
  Script : python hyperspectral_correct.py file.hdr [file.L1T]
  Import : from hyperspectral_correct import read_mtl, read_hdr, correct_hyperion

Processing chain
----------------
1.  Read metadata
    - MTL  -> geometry (SZA, SAA, VZA, VAA), date, radiance scaling factors
    - HDR  -> image dimensions, wavelengths (nm), FWHM, bad-band list (bbl)

2.  Build valid-pixel mask
    - External mask file if given (any ENVI file, non-zero = valid)
    - Auto: pixels equal to the input HDR "data ignore value" (e.g. -9999
      from hyperion_convert.py) or where all bands are zero (black border)

3.  Create output file before computation
    - Allocates disk space immediately (int16, interleave managed by spectral)
    - Pre-filled with nodata = -9999

4.  Open input as read-only memory map  (one band read at a time)

5.  For each good band (bbl=1, wavelength 0.40-2.50 um):
    a. Run 6S in band-integrated mode (iwave=-2, flat top-hat ±1 FWHM):
         6S steps through the band at 2.5 nm intervals and solar-irradiance-
         weighted averages all outputs. This gives correct band-averaged gas
         transmittance, fixing the monochromatic artefact (iwave=-1) that
         produced wildly wrong reflectances at O2 (762nm) and H2O absorption
         bands (720, 820, 940, 1130, 1380nm).
       Key output coefficients (see _coefficients() for full details):
         xa    = srotot (path reflectance, already includes gas effects)
         T_down = sdtott * dgasm  (scattering * downward gas transmittance)
         T_up   = sutott * ugasm  (scattering * upward gas transmittance)
         TdTu   = T_down * T_up
         S     = spherical_albedo_tot (NOT pizera)
       Verified against original 6S Fortran (main.f):
         xb = srotot / (sutott * sdtott * tgasm)  -- tgasm ~ dgasm * ugasm
       Without gas correction, TdTu is up to 6x too large at 940nm (H2O),
       giving rho_s 439% too low (correction essentially absent).
    b. Get TOA solar irradiance E0 from built-in 6S solar spectrum (solirr),
       corrected for Earth-Sun distance on acquisition date (Spencer 1971).
    c. DN -> radiance:  L [W/m2/sr/um] = DN / scale_factor
         VNIR (bands 1-70):  scale = 40
         SWIR (bands 71-242): scale = 80
    d. TOA reflectance:  rho_toa = pi * L / (E0 * cos(SZA))
    e. Surface reflectance (linear, inhomo black-pixel model):
         rho_s = (rho_toa - xa_eff) / TdTu
       xa_eff = xa + rho_env_model * coeff  (Stage-1: env_model adjacency removed)
       xa_eff = xa + rho_env_smoothed * coeff  (Stage-2: residual adjacency)
    f. Scale to int16:  stored = round(rho_s * 10000)
       To recover: rho_s = stored / 10000.0.  Nodata = -9999
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
# Recommended Stage-2 Gaussian smoothing radius by AOT range.
# The atmospheric PSF has a sharp central peak; 0.1-0.5 km captures it
# for Hyperion 30 m pixels. Larger values wash out spatial variation.
DEFAULT_ADJ2_RADIUS_BY_AOT = {
    # AOT upper bound → recommended radius (km)
    0.10: 0.15,   # very clean: tight PSF, short radius
    0.25: 0.20,   # clean (typical Finland/boreal)
    0.50: 0.35,   # moderate aerosol
    1.00: 0.60,   # hazy
    9.99: 1.00,   # very hazy / smoke
}

def _sixs_band_worker(args):
    """
    Module-level worker function for parallel 6S band processing.
    Must be at module level (not a closure) so multiprocessing can pickle it
    on Windows (spawn-based) as well as Linux (fork-based).
    args: (band_index, sixs_config_string, n_harmonics)
    """
    b_idx, cfg, nh = args
    from sixs.sixs_main import run6S as _r6s
    import io as _io
    r = _r6s(_io.StringIO(cfg), _io.StringIO(), n_harmonics=nh)
    return b_idx, r


def _recommended_adj2_radius(aot550: float) -> float:
    """Return recommended Stage-2 Gaussian smoothing radius (km) for a given AOT."""
    for aot_limit, radius in sorted(DEFAULT_ADJ2_RADIUS_BY_AOT.items()):
        if aot550 <= aot_limit:
            return radius
    return 1.0

DEFAULT_ENV_RADIUS_KM = None   # set from AOT at GUI creation; see _recommended_adj2_radius

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

def find_mtl(hdr_path):
    """
    Search for an MTL/metadata file alongside an ENVI HDR, without assuming
    any particular extension.

    Strategy (in order):
      1. Look in the same directory as hdr_path for any file whose name
         contains "MTL" or "mtl" (case-insensitive).
      2. Among candidates, prefer files that actually contain the key
         ACQUISITION_DATE (confirming it is a valid Hyperion MTL).
      3. Return the best match, or None if nothing is found.
    """
    import os
    hdr_dir  = os.path.dirname(os.path.abspath(hdr_path))
    stem     = os.path.splitext(os.path.basename(hdr_path))[0]

    # Collect all files in the same directory
    try:
        siblings = os.listdir(hdr_dir)
    except OSError:
        return None

    # Priority 1: files whose name contains "MTL" (case-insensitive)
    mtl_candidates = [
        os.path.join(hdr_dir, f) for f in siblings
        if "mtl" in f.lower() and os.path.isfile(os.path.join(hdr_dir, f))
    ]

    # Priority 2: if no MTL-named files, try every text-like file and check content
    if not mtl_candidates:
        text_exts = {".txt", ".l1t", ".l1g", ".met", ".xml", ""}
        mtl_candidates = [
            os.path.join(hdr_dir, f) for f in siblings
            if os.path.splitext(f)[1].lower() in text_exts
            and os.path.isfile(os.path.join(hdr_dir, f))
        ]

    # Among candidates, prefer those containing ACQUISITION_DATE
    for path in mtl_candidates:
        try:
            with open(path, errors="ignore") as fh:
                head = fh.read(4096)
            if "ACQUISITION_DATE" in head:
                return path
        except OSError:
            continue

    # Fall back to first MTL-named candidate even without content check
    return mtl_candidates[0] if mtl_candidates else None


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

    # VZA: angle from the zenith (vertical) to the direction from target to sensor.
    # Equivalently: the sensor off-nadir angle (both are the same number because
    # the target→sensor direction and the sensor nadir direction share the vertical).
    # 6S parameter "avis": angle in [0°, 90°], always positive.
    # SENSOR_LOOK_ANGLE in the MTL is signed:
    #   positive → sensor looks EAST  → sensor is WEST of target
    #   negative → sensor looks WEST  → sensor is EAST of target
    # The absolute value gives the zenith/off-nadir magnitude (VZA).
    # The sign is used below to compute VAA (target→sensor azimuth).
    vza = abs(look_angle)

    # ── Compute VAA from EO-1 orbit geometry ─────────────────────────────────
    #
    # Background
    # ----------
    # The MTL file provides SENSOR_LOOK_ANGLE with a SIGN but no azimuth:
    #   positive = looking east of the ground track (sensor WEST of target)
    #   negative = looking west of the ground track (sensor EAST of target)
    # 6S requires the VIEW AZIMUTH (VAA, called phiv internally):
    #   The azimuth of the direction from the TARGET to the SENSOR,
    #   measured clockwise from North — the same convention as SAA.
    #   This is NOT the direction the sensor is looking; it is the reverse:
    #   the direction you would face standing at the target to look up at the sensor.
    #   EO-1 descending over Finland: track SSW, SENSOR_LOOK_ANGLE=+18.641°
    #   → sensor looks east → sensor west of target → VAA ≈ 289° (WNW).
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
    # 6S parameter phiv = VAA = azimuth of the target→sensor direction,
    # measured clockwise from North. Internally 6S computes the relative
    # azimuth phi = |phi0 − phiv| (SAA − VAA), so it is the DIFFERENCE
    # between solar and view azimuths that drives the scattering geometry.
    # Using the physical target→sensor azimuth here is correct.

    EO1_INCLINATION_DEG = 98.7   # EO-1 orbital inclination (retrograde S-S)

    # Scene centre lat/lon from image corner coordinates
    try:
        lat_centre = (float(raw["IMAGE_UL_CORNER_LAT"]) +
                      float(raw["IMAGE_LL_CORNER_LAT"]) +
                      float(raw["IMAGE_UR_CORNER_LAT"]) +
                      float(raw["IMAGE_LR_CORNER_LAT"])) / 4.0
        lon_centre = (float(raw["IMAGE_UL_CORNER_LON"]) +
                      float(raw["IMAGE_LL_CORNER_LON"]) +
                      float(raw["IMAGE_UR_CORNER_LON"]) +
                      float(raw["IMAGE_LR_CORNER_LON"])) / 4.0
    except (KeyError, ValueError):
        lat_centre = 50.0   # fallback if corners absent from MTL
        lon_centre = 25.0

    inc_rad = math.radians(EO1_INCLINATION_DEG)
    lat_rad = math.radians(lat_centre)

    # Descending pass ground track azimuth.
    # For inclination i > 90° (retrograde sun-synchronous), the satellite
    # flies southward with a slight westward component:
    #   sin(delta) = |cos(i)| / cos(lat)
    #   az_track = 180° + delta   (SSW, confirmed by observed NNE→SSW strip)
    # Note: 180° − delta would be the ascending pass (northward, NNE direction).
    ratio = abs(math.cos(inc_rad)) / math.cos(lat_rad)
    if ratio <= 1.0:
        az_track = 180.0 + math.degrees(math.asin(ratio))
    else:
        az_track = 180.0   # fallback: due south

    # SENSOR_LOOK_ANGLE sign convention:
    #   positive → sensor looks EAST  → sensor is WEST of target
    #   negative → sensor looks WEST  → sensor is EAST of target
    #
    # VAA = azimuth of target→sensor direction, clockwise from North [0–360°].
    # 6S parameter phiv uses this directly.
    # Track runs toward az_track (SSW for EO-1 descending pass).
    #
    # Sensor west of target (look_angle > 0):
    #   target→sensor points westward = az_track + 90°
    #   EO-1 Finland example: az_track≈199° → VAA≈289° (WNW) → FLAASH = −71° ✓
    # Sensor east of target (look_angle < 0):
    #   target→sensor points eastward = az_track − 90°
    #
    # FLAASH "azimuth of sensor as viewed from the ground" = same direction as VAA,
    #   but expressed in range −180..+180 (west = negative).
    #   Conversion: FLAASH = VAA if VAA <= 180, else VAA − 360.
    if look_angle >= 0.0:
        # sensor looks EAST → sensor WEST of target → target→sensor is westward
        vaa = (az_track + 90.0) % 360.0   # WNW for EO-1/Finland ≈ 289°
    else:
        # sensor looks WEST → sensor EAST of target → target→sensor is eastward
        vaa = (az_track - 90.0) % 360.0

    return dict(
        acq_date=acq_date, start_time=start_time.strip(),
        month=dt.month, day=dt.day,
        sza=sza, saa=saa, vza=vza, vaa=round(vaa, 1),
        scene_centre_lat=lat_centre,
        scene_centre_lon=lon_centre,
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

def _make_6s_input(wl_um, fwhm_um, sza, saa, vza, vaa, month, day,
                   idatm, uh2o, uo3, iaer, aot550, target_alt_km,
                   env_model=None, env_radius_km=2.0):
    """
    Build a 6S input string for one sensor band.

    Uses iwave=-2 (band-integrated flat top-hat filter) spanning
    [wl_um - fwhm_um, wl_um + fwhm_um]. 6S steps through the band at
    2.5 nm intervals and solar-irradiance-weighted averages all outputs.
    This gives correct band-averaged transmittances for all gases.

    Previously used iwave=-1 (monochromatic at band centre). That computed
    gas transmittance at a single wavelength, which could land in a deep
    absorption line or a window — giving wildly wrong corrections at gas
    absorption bands. With iwave=-2 the averaging is done inside 6S, giving
    the same result as a proper line-by-line instrument response convolution
    for a flat (top-hat) spectral response function.

    Note: the gas transmittances dgasm/ugasm returned by run6S() must be
    multiplied into T_down/T_up in _coefficients() — see that function.

    vza : float — view zenith angle [deg, 0-90]: angle from zenith to the
                  target->sensor direction = sensor off-nadir angle. 6S: "avis".
                  EO-1 SENSOR_LOOK_ANGLE=+18.641 -> VZA=18.641.
    vaa : float — view azimuth [deg, clockwise from North, 0-360]: azimuth of
                  target->sensor direction. 6S: "phiv". NOT the sensor look
                  direction (sensor->target = VAA+180 mod 360).
                  FLAASH "azimuth from ground": same direction, range -180..+180.
                  EO-1 Finland: sensor looks east -> sensor west of target
                  -> VAA ~ 289 (WNW) -> FLAASH = -71.
    env_model    : int or None -- igrou2 environment surface code (1-4).
                   None means inhomo=0 (uniform surface, no adjacency).
    env_radius_km: float -- radius of the target patch in km (inhomo=1 only).
    """
    atm_extra = f"\n{uh2o:.4f}   {uo3:.4f}" if idatm == 8 else ""

    if env_model is not None:
        surface = (
            f"1\n"
            f"0 {env_model} {env_radius_km:.2f}\n"
            f"0\n"
            f"0\n"
            f"0.0\n"
        )
    else:
        surface = "0\n0\n0\n0.0\n"

    # Band integration limits: +/-1 FWHM around centre wavelength.
    # Clamp to 6S valid range [0.25, 4.0 um].
    wlinf = max(0.25, wl_um - fwhm_um)
    wlsup = min(4.00, wl_um + fwhm_um)

    return (
        f"0\n"
        f"{sza:.4f} {saa:.4f} {vza:.4f} {vaa:.4f} {month} {day}\n"
        f"{idatm}{atm_extra}\n"
        f"{iaer}\n"
        f"0\n"
        f"{aot550:.4f}\n"
        f"{-abs(target_alt_km):.4f}\n"
        f"-1000\n"
        f"-2\n"
        f"{wlinf:.5f} {wlsup:.5f}\n"
        f"{surface}"
        f"-2.0\n"
    )
def _coefficients(r6s):
    """
    Derive correction coefficients from a 6S result dict.

    Retrieval formula
    -----------------
    The correction is linear in rho_pixel (pixel is black in the 6S run,
    so there is no multiple-scattering coupling between pixel and atmosphere):

        rho_s = (rho_toa - xa_eff) / TdTu

    where:
        xa_eff = xa + rho_env_model * coeff      (Stage-1: assumed environment)
        xa_eff = xa + rho_env_smoothed * coeff   (Stage-2: actual local environment)
        coeff  = T_up * S * T_down               (adjacency sensitivity)
        TdTu   = T_down * T_up                   (total two-way transmittance)

    Total transmittance — gas absorption included
    ---------------------------------------------
    6S outputs two separate components:
        sdtott  = scattering-only downward transmittance (Rayleigh + aerosol)
        sutott  = scattering-only upward transmittance
        dgasm   = downward total gas transmittance (H2O * O3 * O2 * CO2 * CH4 * N2O * CO)
        ugasm   = upward total gas transmittance

    The correct total transmittances are:
        T_down = sdtott * dgasm
        T_up   = sutott * ugasm

    This matches the original 6S atmospheric correction formula (main.f, commented
    section, lines ~2371-2372):
        xb = srotot / (sutott * sdtott * tgasm)
    where tgasm ~ dgasm * ugasm for the two-path total.

    Numerical verification (Fortran vs Python, iwave=-2, Hyperion geometry):
        Band        dgasm   ugasm   TdTu_old  TdTu_new  rho_s_error_old
        427nm win   1.000   1.000   0.692     0.692       0%   (unaffected)
        865nm win   1.000   1.000   0.958     0.958       0%   (unaffected)
        762nm O2    0.663   0.695   0.942     0.434    +117%   (O2 A-band)
        720nm H2O   0.859   0.882   0.935     0.707     +32%   (H2O weak)
        820nm H2O   0.814   0.841   0.952     0.651     +46%   (H2O weak)
        940nm H2O   0.407   0.456   0.962     0.178    +439%   (H2O strong)
    Without gas, the correction was effectively absent at 940nm and severely
    undercorrected at all H2O and O2 absorption bands.

    Notes on xa and S
    -----------------
    xa = srotot: path reflectance over black surface. Already correct —
    the spectral loop in main.f accumulates srotot as romix*coef where romix
    already includes gas transmittance via tgtot (line 2115: srotot += romix*coef,
    line 2089: ratm = romix*tgtot + ...). No adjustment needed.

    S = spherical_albedo_tot (total atmospheric spherical albedo, ~0.03-0.20).
    Must NOT use pizera (aerosol single-scattering albedo ~0.90): that gives
    a denominator 30x too large, underestimating rho_s by 1-3 pp in the blue.
    """
    rho_atm = r6s["srotot"]
    # dgasm/ugasm: downward/upward total gas transmittance from the 6S spectral
    # loop. Default to 1.0 only if absent (e.g. old output without gas fields).
    dgasm   = r6s.get("dgasm", 1.0)
    ugasm   = r6s.get("ugasm", 1.0)
    # Total transmittance = scattering * gas (both directions)
    T_down  = r6s["sdtott"] * dgasm
    T_up    = r6s["sutott"] * ugasm
    S       = r6s["spherical_albedo_tot"]
    denom   = T_down * T_up
    if denom < 1e-6:
        return 0.0, 1.0, 0.0, 1.0, 0.0, S
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
        sza, saa [degrees] — solar zenith and azimuth
        vza [degrees] — view zenith: angle from zenith to target→sensor direction
                        = sensor off-nadir angle.  0°=nadir.
        vaa [degrees] — view azimuth: target→sensor direction, clockwise from North.
                        Positive = sensor east of target (sensor looks west).
                        SENSOR_LOOK_ANGLE positive → sensor looks east → VAA ~289°.
                        FLAASH equivalent: (VAA−360) if VAA>180, else VAA.

    Sensor / calibration:
        sensor_name   : str — written to output header "sensor type" field
        input_type    : "radiance"  — apply DN→L→rho_toa chain
                        "toa_refl" — data already is TOA reflectance; skip E0
        rad_scale_spec: str — profile rad_scale string, passed to build_rad_scale()
                        e.g. "split:40:80:70"  or  "uniform:1"  or  "none"
        interleave    : "bip", "bil", or "bsq" (default "bip")
        drop_bad_bands: bool — exclude bands where bbl=0 (default True)

    Atmospheric correction parameters:
        idatm         [0-8]
        uh2o          [g/cm2, used only when idatm=8]
        uo3           [cm-atm, used only when idatm=8]
        iaer          [0-3]
        aot550        [AOT at 550 nm]
        target_alt_km [km above sea level]
        n_harmonics   [int, default 3 — Fourier azimuth harmonics for SOS;
                       3 = exact for Rayleigh, <0.02% error for AOT<0.5;
                       increase to 6-10 for heavy aerosol or asymmetric phase fn]
        n_workers     [int, default 1 — parallel 6S worker processes]

    Environment / adjacency (Stage-1):
        env_model     : int or None — igrou2 surface type for inhomo 6S run.
                        None = inhomo=0 (no adjacency in Stage-1).
                        1=Vegetation, 2=ClearWater, 3=Sand, 4=LakeWater.
                        Stage-1 removes this environment's adjacency contribution
                        by including rho_env_model in xa_eff.
        env_radius_km : float — Stage-1 patch radius (km), default 0.2.
                        Controls PSF weighting of environment vs target in 6S.

    Stage-2 adjacency correction (optional):
        do_adj2       : bool — enable Stage-2 Gaussian adjacency correction.
        adj2_out_base : str or None — output path for Stage-2 result.
        adj2_radius_km: float — Gaussian smoothing radius (km) for rho_env.
                        Should equal env_radius_km. Both updated together from AOT.
        pixel_size_m  : float — pixel size in metres (default 30.0 for Hyperion).
                        Used to convert adj2_radius_km to sigma in pixels.

    Optional outputs:
        mask_file     : str or None — external single-band mask (non-zero = valid).
                        Auto-mask: pixels equal to HDR "data ignore value" (−9999)
                        or where all bands are zero (black border).
        config_file   : str or None — path to save the 6S configuration as text.
        spec_csv_file : str or None — path to save per-band atmospheric quantities:
                        wavelength, atm_radiance, env_radiance,
                        ground_direct_irr, ground_diffuse_irr, ground_env_irr
                        [W m⁻² µm⁻¹].

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
    wl_nm         = params["wl_nm"]
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
    adj2_radius_km = params.get("adj2_radius_km", 0.2)
    pixel_size_m   = params.get("pixel_size_m",   30.0)  # Hyperion GSD = 30 m
    env_radius_km  = params.get("env_radius_km",  2.0)
    config_file    = params.get("config_file",    None)
    spec_csv_file  = params.get("spec_csv_file",  None)
    n_harmonics    = int(params.get("n_harmonics", 3))
    n_workers      = int(params.get("n_workers",   1))
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
    log(f"  VZA            : {vza:.3f} deg  (target→sensor from zenith = off-nadir angle)")
    log(f"  VAA            : {vaa:.3f} deg  (target→sensor azimuth N cw; "
         f"FLAASH={vaa if vaa<=180 else vaa-360:.1f} deg)")
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
    log(f"  Mask file      : {mask_file if mask_file else '(auto: data ignore value or all-zero)'}")
    log(f"  6S config out  : {config_file if config_file else '(not saved)'}")
    log(f"  Spectral CSV   : {spec_csv_file if spec_csv_file else '(not saved)'}")
    log(f"  SOS harmonics  : {n_harmonics}")
    log(f"  Output interleave: {interleave.upper()}")
    log(f"  Drop bad bands : {'yes' if drop_bad_bands else 'no'}")
    if do_adj2:
        log(f"  Stage-2 adj    : ON  radius={adj2_radius_km:.2f} km  "
            f"sigma={adj2_radius_km*1000/pixel_size_m:.1f} px  output={adj2_out_base}")
    else:
        log(f"  Stage-2 adj    : OFF")
    log(f"  Good bands     : {bbl.sum()} / {n_bands}")
    log(f"  Valid wl range : 0.40 - 2.50 um (6S limit)")
    log("=" * 62)

    # -- 6S per band: correction coefficients --------------------------------
    xa        = np.zeros(n_bands)
    xb        = np.ones(n_bands)
    xc        = np.zeros(n_bands)
    T_down_arr = np.ones(n_bands)   # sdtott: total downward transmittance
    T_up_arr   = np.ones(n_bands)   # sutott: total upward transmittance
    S_arr      = np.zeros(n_bands)  # spherical_albedo_tot
    T_dir_arr  = np.ones(n_bands)   # direct-beam downward transmittance

    # Per-band effective reflectance of the environment model assumed in Stage-1.
    # Recovered from the 6S inhomo run:
    #   rho_env_model[b] = ground_env_irr[b] / (S[b] * (dir_irr[b] + dif_irr[b]))
    # This is the rho_env that the env_model type (e.g. Vegetation) represents
    # at each wavelength. It varies spectrally (e.g. vegetation is bright in NIR).
    # If env_model=None (inhomo=0), ground_env_irr=0 so rho_env_model stays 0.
    rho_env_model_arr = np.zeros(n_bands)

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
            "FILL_HERE_FWHM_IN_um",
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
    band_inputs = []   # list of (band_index, 6S_config_string)
    n_ok = 0
    for b in range(n_bands):
        if bbl[b] == 0:
            continue
        wl = wl_um[b]
        if wl < 0.40 or wl > 2.50:
            bbl[b] = 0
            continue

        # Atmospheric correction coefficients
        inp = _make_6s_input(wl, fwhm_nm[b] / 1000.0,
                             sza, saa, vza, vaa, month, day,
                             idatm, uh2o, uo3, iaer, aot550, target_alt_km,
                             env_model=env_model, env_radius_km=env_radius_km)
        band_inputs.append((b, inp))

    # ── Dispatch: parallel or sequential ─────────────────────────────────────
    # _sixs_band_worker is defined at module level so multiprocessing can
    # pickle it on Windows (spawn-based) as well as Linux (fork-based).
    import concurrent.futures as _cf

    jobs = [(b, inp, n_harmonics) for b, inp in band_inputs]
    n_jobs = len(jobs)
    completed = []
    if n_workers > 1:
        log(f"  Launching {n_workers} parallel worker processes...")
        with _cf.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_sixs_band_worker, j): j for j in jobs}
            for i, fut in enumerate(_cf.as_completed(futures), 1):
                result = fut.result()
                completed.append(result)
                b_done, r_done = result
                wl_done = wl_um[b_done]
                # Log every 10th completed band (arrival order, not band order)
                if i % 10 == 0:
                    log(f"  {i:3d}/{n_jobs} bands done  "
                        f"(last: {wl_done*1000:.1f} nm  "
                        f"xa={r_done.get('srotot',0):.4f})")
                log(f"__PROGRESS__ {i} {n_jobs}")
    else:
        for i, j in enumerate(jobs, 1):
            result = _sixs_band_worker(j)
            completed.append(result)
            b_done, r_done = result
            wl_done = wl_um[b_done]
            if i % 10 == 0:
                log(f"  Band {b_done+1:3d}/{n_bands}  {wl_done*1000:7.1f} nm  "
                    f"xa={r_done.get('srotot',0):.4f}  "
                    f"xb={1/(r_done.get('sdtott',1)*r_done.get('sutott',1)):.4f}")
            log(f"__PROGRESS__ {i} {n_jobs}")

    band_result_map = {b: r for b, r in completed}
    n_ok = 0
    for b, inp in sorted(band_inputs, key=lambda x: x[0]):   # process in band order
        wl = wl_um[b]
        try:
            r = band_result_map[b]
            xa[b], xb[b], xc[b], T_down_arr[b], T_up_arr[b], S_arr[b] = _coefficients(r)
            T_dir_arr[b] = (r.get("ground_direct_irr", 0.0) / (E0_um[b] * cos_sza)
                            if E0_um[b] * cos_sza > 1e-6 else T_down_arr[b])

            # Effective reflectance of the assumed environment model at this band.
            # ground_env_irr = S * rho_env_model * (direct_irr + diffuse_irr)
            # Solving for rho_env_model gives the spectral reflectance that the
            # chosen environment type (Vegetation, Sand, etc.) has at this wavelength.
            # If env_model=None (inhomo=0), ground_env_irr=0 so rho_env_model=0.
            _dir_dif = r.get("ground_direct_irr", 0.0) + r.get("ground_diffuse_irr", 0.0)
            if S_arr[b] * _dir_dif > 1e-6:
                rho_env_model_arr[b] = r.get("ground_env_irr", 0.0) / (S_arr[b] * _dir_dif)
            else:
                rho_env_model_arr[b] = 0.0
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
        # Auto-mask: invalid pixels are those whose DN equals the input file's
        # "data ignore value" (commonly -9999 from hyperion_convert.py), OR
        # where all bands are zero (sensor fill / black border).
        # Read the ignore value from the input HDR; fall back to zero only.
        ignore_val = None
        raw_ignore = envi_meta.get("data ignore value", "").strip()
        if raw_ignore:
            try:
                ignore_val = int(float(raw_ignore))
            except ValueError:
                pass

        log("  Building auto-mask...")
        # Cast to int32 to avoid overflow when summing; Hyperion DNs are int16.
        band_sum = in_mm.astype(np.int32).sum(axis=2)   # (lines, samples)

        if ignore_val is not None:
            # Invalid if any band equals the ignore value — use the first band
            # as a proxy (ignore pixels are fill across all bands)
            first_band = in_mm[:, :, 0].astype(np.int32)
            valid = (band_sum != 0) & (first_band != ignore_val)
            log(f"  Auto-mask: ignore value={ignore_val}, "
                f"{valid.sum()} / {valid.size} valid pixels")
        else:
            valid = band_sum != 0
            log(f"  Auto-mask (all-zero): {valid.sum()} / {valid.size} valid pixels")

    # -- Apply correction band by band, writing directly to output memmap ------
    log(f"Applying correction to {out_n_bands} bands...")
    log("__RESET__")   # reset progress bar for the correction-application phase
    for out_b, b in enumerate(good_idx):
        log(f"__PROGRESS__ {out_b + 1} {out_n_bands}")
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
        # ── Stage-1 inversion ─────────────────────────────────────────────────
        # The measured rho_toa contains two contributions:
        #   (1) Atmospheric path radiance:  xa = srotot  (black-surface run)
        #   (2) Environment adjacency:      rho_env_model * coeff
        #       where coeff = T_up * S * T_d  (verified: env_model does not
        #       change xa, S, T_d, T_u — it only appears in ground_env_irr)
        #
        # By incorporating rho_env_model into the effective xa before inversion,
        # Stage-1 now correctly accounts for the chosen environment type.
        # Stage-2 then only needs to correct the residual deviation between
        # the actual local rho_env_smoothed and this assumed rho_env_model.
        #
        # The inhomo forward model is LINEAR in rho_pixel — the pixel is black
        # in the 6S run so there is no multiple-scattering coupling between
        # the pixel and the atmosphere (only the environment couples via S):
        #   rho_toa = xa_eff + TdTu * rho_pixel
        # Inversion: rho_pixel = (rho_toa - xa_eff) / TdTu
        # The (1-S*rho_s) denominator of the uniform model does NOT apply.
        _coeff   = T_up_arr[b] * S_arr[b] * T_down_arr[b]   # T_up*S*T_d
        xa_eff_b = xa[b] + rho_env_model_arr[b] * _coeff
        TdTu_b   = T_down_arr[b] * T_up_arr[b]
        numer = rho_toa - xa_eff_b
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_s = np.where(np.abs(TdTu_b) > 1e-6, numer / TdTu_b, 0.0)
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
    # STAGE 2 — Adjacency correction (residual deviation from env_model)
    # ══════════════════════════════════════════════════════════════════════════
    # Stage-1 now accounts for the environment model's adjacency contribution
    # by using xa_eff = xa + rho_env_model * coeff in the inversion.
    # This exactly removes the adjacency assumed by the chosen env_model type
    # (e.g. Vegetation, Sand) at every wavelength.
    #
    # Stage-2 corrects the RESIDUAL: pixels whose actual local environment
    # differs from the assumed env_model. For example, a water body in a
    # vegetated scene has rho_env_actual < rho_env_model (vegetation), so
    # Stage-1 over-subtracted adjacency → Stage-2 adds it back.
    # Conversely, a clearcut in forest has rho_env_actual > rho_env_model
    # → Stage-1 under-subtracted → Stage-2 reduces further.
    #
    # The linear re-inversion (coeff = T_up * S * T_d per band):
    #
    #   rho_toa  recovered from Stage-1: xa_eff1 + TdTu * rho_s1  (linear!)
    #   xa_eff2  = xa + rho_env_smoothed * coeff   [actual local environment]
    #   rho_s2   = (rho_toa - xa_eff2) / TdTu      (linear inversion)
    #
    # Note: xa = xa[b] (black-surface), xa_eff = xa + rho_env_model * coeff
    # (used in Stage-1), xa_eff2 = xa + rho_env_smoothed * coeff (used here).
    # When rho_env_smoothed == rho_env_model: xa_eff2 == xa_eff → rho_s2 == rho_s1.
    # When rho_env_smoothed == 0 and env_model=None: both stages use xa → xa_eff=xa.
    #
    # Magnitude of Stage-2 residual correction (427nm, AOT=0.06, coeff≈0.137):
    #   |rho_env_smoothed - rho_env_model| = 0.10 → |Δrho| ≈ 1.4 pp
    #   |rho_env_smoothed - rho_env_model| = 0.40 → |Δrho| ≈ 5.5 pp
    # Correction is zero for pixels matching the assumed environment.
    if do_adj2 and adj2_out_base:
        from scipy.ndimage import gaussian_filter
        log(f"\nStage-2 adjacency correction  radius={adj2_radius_km:.1f} km ...")
        log("__RESET__")   # tell the GUI to reset the progress bar to 0%

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

        # Stage-2 is pure NumPy/SciPy — parallelise with threads.
        # scipy.ndimage.gaussian_filter releases the GIL, so ThreadPoolExecutor
        # gives real concurrency without process-spawn overhead.
        import concurrent.futures as _cf2

        def _adj2_band(args):
            out_b, b = args
            if T_down_arr[b] < 1e-4:
                return out_b, None

            rho_s1 = s1_mm[:, :, out_b].astype(np.float32) / 10000.0
            rho_s1[rho_s1 < -0.09] = np.nan

            fill    = np.where(np.isnan(rho_s1), np.nanmean(rho_s1), rho_s1)
            rho_env = gaussian_filter(fill.astype(np.float64),
                                      sigma=sigma).astype(np.float32)

            # Exact Stage-2 re-inversion.
            # Stage-1 used xa_eff = xa + rho_env_model * coeff.
            # Stage-2 re-inverts using xa_eff2 = xa + rho_env_smoothed * coeff
            # (the actual local environment).
            # When rho_env_smoothed == rho_env_model, rho_s2 == rho_s1 (no change).
            _xa    = xa[b]
            _S     = S_arr[b]
            _TdTu  = T_down_arr[b] * T_up_arr[b]
            _coeff = T_up_arr[b] * _S * T_down_arr[b]   # T_up * S * T_d

            # xa used in Stage-1 (includes assumed environment model)
            xa_eff1 = _xa + rho_env_model_arr[b] * _coeff

            with np.errstate(divide="ignore", invalid="ignore"):
                # Recover rho_toa from Stage-1 (linear forward model):
                #   rho_toa = xa_eff1 + TdTu * rho_s1
                # No (1-S*rho_s) denominator: pixel is black in the 6S run.
                rho_toa = xa_eff1 + _TdTu * rho_s1
                # Re-invert with actual rho_env (also linear):
                xa_eff2 = _xa + rho_env * _coeff
                numer   = rho_toa - xa_eff2
                rho_s2  = np.where(np.abs(_TdTu) > 1e-6, numer / _TdTu, rho_s1)

            rho_s2 = np.clip(rho_s2, -0.1, 1.5)

            band_out = np.clip(rho_s2 * 10000, -32768, 32767).astype(np.int16)
            band_out[np.isnan(rho_s1)] = -9999
            band_out[~valid] = -9999
            return out_b, band_out

        jobs2 = list(enumerate(good_idx))
        n_threads = max(1, n_workers)   # reuse the same n_workers setting
        with _cf2.ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = {pool.submit(_adj2_band, j): j for j in jobs2}
            done = 0
            _pct2 = -1
            for fut in _cf2.as_completed(futures):
                out_b, band_out = fut.result()
                done += 1
                pct2 = int(done / out_n_bands * 100)
                if pct2 // 10 > _pct2 // 10:
                    _pct2 = pct2
                    log(f"    {pct2:3d}%  ({done}/{out_n_bands} bands)")
                log(f"__PROGRESS__ {done} {out_n_bands}")
                if band_out is not None:
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

    # Hidden entry widgets for values loaded from MTL but not shown in the GUI.
    # Using real (hidden) Entry widgets so _set/_get work uniformly.
    _hidden_frame = tk.Frame(root)   # never packed — just a parent for hidden widgets
    for _hkey in ("acq_date", "start_time", "scene_lat", "scene_lon"):
        _hw = tk.Entry(_hidden_frame)
        e[_hkey] = _hw

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
    # TAB 1 — Files & Geometry (scrollable — content may exceed window height)
    # ═══════════════════════════════════════════════
    _tab1_outer = ttk.Frame(nb)
    nb.add(_tab1_outer, text="Files & Geometry")

    _t1_canvas = tk.Canvas(_tab1_outer, borderwidth=0, highlightthickness=0)
    _t1_scroll = ttk.Scrollbar(_tab1_outer, orient=tk.VERTICAL,
                                command=_t1_canvas.yview)
    _t1_canvas.configure(yscrollcommand=_t1_scroll.set)
    _t1_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    _t1_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    tab1 = ttk.Frame(_t1_canvas, padding=10)
    _t1_win = _t1_canvas.create_window((0, 0), window=tab1, anchor="nw")

    def _t1_resize(event):
        _t1_canvas.configure(scrollregion=_t1_canvas.bbox("all"))
    tab1.bind("<Configure>", _t1_resize)
    _t1_canvas.bind("<Configure>",
        lambda e: _t1_canvas.itemconfig(_t1_win, width=e.width))
    _t1_canvas.bind_all("<MouseWheel>",
        lambda e: _t1_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

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

    def _apply_mtl(m):
        """Apply all fields from a read_mtl() result dict to the GUI."""
        _set("sza",        round(m["sza"], 3))
        _set("saa",        round(m["saa"], 3))
        _set("vza",        round(m["vza"], 3))
        _set("vaa",        round(m["vaa"], 3))
        _set("month",      m["month"])
        _set("day",        m["day"])
        _set("acq_date",   m["acq_date"])
        _set("start_time", m["start_time"])
        _set("scene_lat",  round(m["scene_centre_lat"], 4))
        _set("scene_lon",  round(m["scene_centre_lon"], 4))
        _set("rad_scale_spec", m["rad_scale_spec"])

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
            # Auto-fill log file path
            log_default = os.path.splitext(p)[0] + "_6Slog.txt"
            if e["log_file"].get() in ("", "<not saved>"):
                log_file_entry.config(state=tk.NORMAL)
                log_file_entry.delete(0, tk.END)
                log_file_entry.insert(0, log_default)
            # Auto-fill spectral CSV path
            csv_default = os.path.splitext(p)[0] + "_spectral.csv"
            if not e["spec_csv_file"].get():
                _set("spec_csv_file", csv_default)
            # Auto-detect MTL if not already set
            if not _get("mtl_file"):
                found = find_mtl(p)
                if found:
                    _set("mtl_file", found)
                    try:
                        m = read_mtl(found)
                        _apply_mtl(m)
                        set_status(
                            f"MTL auto-detected: {os.path.basename(found)}  "
                            f"SZA={m['sza']:.2f}  VZA={m['vza']:.2f}")
                    except Exception:
                        pass  # silently ignore auto-detect parse errors

    def browse_mtl():
        p = filedialog.askopenfilename(
            title="Select Hyperion MTL file",
            filetypes=[("All files", "*.*")])
        if not p:
            return
        _set("mtl_file", p)
        try:
            m = read_mtl(p)
            _apply_mtl(m)
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
            _set("spec_csv_file", p)

    ttk.Label(fg, text="Spectral CSV:").grid(row=6, column=0, sticky=tk.W, pady=3)
    spec_csv_entry = ttk.Entry(fg)
    spec_csv_entry.insert(0, "")
    spec_csv_entry.grid(row=6, column=1, sticky=tk.EW, padx=4)
    e["spec_csv_file"] = spec_csv_entry
    ttk.Button(fg, text="Browse...", command=browse_spec_csv).grid(row=6, column=2)
    ttk.Label(fg,
              text="atm/env radiance + ground direct/diffuse/env irradiance, W m⁻² µm⁻¹",
              foreground="gray").grid(row=7, column=0, columnspan=3, sticky=tk.W)

    # Log file row
    def browse_log_file():
        p = filedialog.asksaveasfilename(
            title="Save log file as",
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if p:
            _set("log_file", p)

    ttk.Label(fg, text="Save log to file:").grid(row=8, column=0, sticky=tk.W, pady=3)
    log_file_entry = ttk.Entry(fg)
    log_file_entry.insert(0, "<not saved>")
    log_file_entry.grid(row=8, column=1, sticky=tk.EW, padx=4)
    e["log_file"] = log_file_entry
    ttk.Button(fg, text="Browse...", command=browse_log_file).grid(row=8, column=2)
    ttk.Label(fg, text="default: <out_base>_6Slog.txt",
              foreground="gray").grid(row=9, column=0, columnspan=3, sticky=tk.W)

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
    drop_cb.grid(row=10, column=0, columnspan=3, sticky=tk.W, pady=(6, 2))
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
    adj2_cb.grid(row=11, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
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
        # SZA: angle from zenith (vertical) to the sun. 0°=overhead, 90°=horizon.
        # SAA: azimuth of the sun, measured clockwise from North.
        # VZA: angle from zenith to the target→sensor direction (= sensor off-nadir angle).
        #      0°=nadir view, positive toward horizon. Same number as nadir angle
        #      because target→sensor and sensor→nadir share the same vertical.
        # VAA: azimuth of target→sensor direction, clockwise from North [0–360°].
        #      Sensor looks east (+SENSOR_LOOK_ANGLE) → sensor west of target
        #      → VAA is westward (~289° for EO-1 Finland). FLAASH: same value
        #      but −180..+180 (west = negative, e.g. −71° for this scene).
        ("Solar zenith SZA (deg):",              "sza",   ""),   # from MTL
        ("Solar azimuth SAA (deg, N clockwise):", "saa",  ""),   # from MTL
        # VZA: angle from zenith to target→sensor direction (= off-nadir angle).
        # VAA: target→sensor azimuth, N clockwise, 0–360°. FLAASH: (VAA−360) if VAA>180.
        ("View zenith VZA (deg, 0=nadir):",              "vza",   ""),  # from MTL
        ("View azimuth VAA (deg, target→sensor, N cw):", "vaa",    0.0), # from MTL
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

    # ── AERONET retrieval button ───────────────────────────────────────────────
    aeronet_status = tk.StringVar(value="")

    def _fetch_aeronet():
        """Retrieve AOT (and H2O/O3 if idatm=8) from the nearest AERONET site."""
        import threading
        import datetime as _dt

        # If already running, clicking the button aborts the current fetch
        if _aeronet_abort[0] is not None:
            _aeronet_abort[0][0] = True
            aeronet_btn.config(text="Fetch from AERONET…")
            _aeronet_abort[0] = None
            aeronet_status.set("AERONET fetch aborted.")
            return

        abort_flag = [False]
        _aeronet_abort[0] = abort_flag
        aeronet_btn.config(text="Abort fetching")

        def _worker():
            try:
                # ── Read scene coordinates and date from the e dict ────────────
                def _get_val(key):
                    w = e.get(key)
                    return w.get().strip() if w and hasattr(w, "get") else ""

                lat_s = _get_val("scene_lat")
                lon_s = _get_val("scene_lon")
                date_s = _get_val("acq_date")[:10]   # YYYY-MM-DD
                # start_time from MTL is "YYYY DOY HH:MM:SS" — extract HH:MM
                time_s = ""
                raw_time = _get_val("start_time")
                import re as _re
                m_t = _re.search(r'(\d{2}:\d{2}):\d{2}', raw_time)
                if m_t:
                    time_s = m_t.group(1)

                if not lat_s or not lon_s:
                    msg = "AERONET: No scene coordinates — load MTL first."
                    root.after(0, lambda: aeronet_status.set(msg))
                    root.after(0, lambda: set_status(msg))
                    return
                if not date_s:
                    msg = "AERONET: No acquisition date — load MTL first."
                    root.after(0, lambda: aeronet_status.set(msg))
                    root.after(0, lambda: set_status(msg))
                    return

                lat = float(lat_s)
                lon = float(lon_s)
                date = _dt.date.fromisoformat(date_s)
                time_utc = None
                try:
                    if time_s:
                        time_utc = _dt.time.fromisoformat(time_s + ":00")
                except Exception:
                    pass

                searching = f"AERONET: Searching near ({lat:.2f}N, {lon:.2f}E) on {date_s}…"
                root.after(0, lambda: aeronet_status.set(searching))
                root.after(0, lambda: set_status(searching))

                def _append_log(msg):
                    root.after(0, lambda m=msg: (
                        log_text.insert(tk.END, m + "\n"),
                        log_text.see(tk.END)
                    ))

                # ── Import aeronet_fetch from same directory ────────────────────
                import importlib.util, os as _os
                af_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                        "aeronet_fetch.py")
                spec = importlib.util.spec_from_file_location("aeronet_fetch", af_path)
                af   = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(af)

                def _log(msg):
                    root.after(0, lambda m=msg: aeronet_status.set(m))
                    _append_log(f"  AERONET: {msg}")

                res = af.retrieve(lat, lon, date, time_utc,
                                  target_wl_nm=550.0, level="15",
                                  log=_log, abort_flag=abort_flag)

                if res["n_aod_obs"] == 0:
                    msg = "AERONET: No data found for this date/location."
                    root.after(0, lambda: aeronet_status.set(msg))
                    root.after(0, lambda: set_status(msg))
                    _append_log(msg)
                    return

                aod = res.get("aod_target")
                pwv = res.get("pwv_cm")
                o3  = res.get("ozone_du")
                site = res.get("site", "?")
                dist = res.get("site_dist_km", 0.0)

                def _fill():
                    if aod is not None:
                        e["aot550"].delete(0, tk.END)
                        e["aot550"].insert(0, f"{aod:.4f}")
                        _update_radius_from_aot()

                    # Always fill H2O and O3 — even when greyed out (idatm≠8).
                    # Temporarily enable the widget to write, then restore its
                    # state. Values are ready when the user switches to idatm=8.
                    if pwv is not None:
                        e["uh2o"].config(state=tk.NORMAL)
                        e["uh2o"].delete(0, tk.END)
                        e["uh2o"].insert(0, f"{pwv:.3f}")
                    if o3 is not None:
                        # Convert Dobson units to cm-atm (1 DU = 0.001 cm-atm)
                        e["uo3"].config(state=tk.NORMAL)
                        e["uo3"].delete(0, tk.END)
                        e["uo3"].insert(0, f"{o3 * 0.001:.4f}")
                    # Restore disabled state for non-idatm=8 profiles
                    on_atm_change()

                    is_user = ATM_MODELS.get(e["idatm_name"].get(), 0) == 8
                    pwv_str  = f"  PWV={pwv:.3f}cm"   if pwv else ""
                    o3_str   = f"  O3={o3:.0f}DU"     if o3  else ""
                    h2o_note = "" if is_user else "  (H2O/O3 pre-filled for idatm=8)"
                    summary = (f"AERONET: {site} ({dist:.0f}km)  AOT={aod:.4f}"
                               f"{pwv_str}{o3_str}{h2o_note}")
                    aeronet_status.set(summary)
                    set_status(summary)
                    _append_log(f"\n{summary}")
                    _append_log(f"  → H2O = {pwv:.3f} g/cm2  "
                                f"O3 = {o3*0.001:.4f} cm-atm ({o3:.1f} DU)"
                                if (pwv and o3) else
                                f"  → H2O = {pwv:.3f} g/cm2" if pwv else "")

                root.after(0, _fill)

            except Exception as ex:
                msg = f"AERONET error: {ex}"
                root.after(0, lambda: aeronet_status.set(msg))
                root.after(0, lambda: set_status(msg))
                _append_log(msg)
            finally:
                _aeronet_abort[0] = None
                root.after(0, lambda: aeronet_btn.config(text="Fetch from AERONET…"))

        threading.Thread(target=_worker, daemon=True).start()

    _aeronet_abort = [None]   # holds the current abort_flag list, or None if idle

    aeronet_btn = ttk.Button(aerg, text="Fetch from AERONET…",
                             command=_fetch_aeronet)
    aeronet_btn.grid(row=1, column=2, sticky=tk.W, padx=(8, 0))
    ttk.Label(aerg, textvariable=aeronet_status,
              foreground="gray").grid(row=2, column=0, columnspan=4,
                                       sticky=tk.W, pady=(2, 0))

    def _update_radius_from_aot(event=None):
        """Sync both Stage-1 patch radius and Stage-2 Gaussian radius to the AOT-recommended value."""
        try:
            aot = float(e["aot550"].get())
            suggested = _recommended_adj2_radius(aot)
            for key in ("adj2_radius_km", "env_radius_km"):
                cur = e.get(key)
                if cur is not None:
                    cur.delete(0, tk.END)
                    cur.insert(0, f"{suggested:.2f}")
        except (ValueError, KeyError):
            pass

    e["aot550"].bind("<FocusOut>", _update_radius_from_aot)
    e["aot550"].bind("<Return>",   _update_radius_from_aot)

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
    _entry(envg, "env_radius_km", _recommended_adj2_radius(0.06), width=10,
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
    adj2_radius_e.insert(0, f"{_recommended_adj2_radius(0.06):.2f}")
    adj2_radius_e.grid(row=0, column=1, sticky=tk.W, padx=4)
    e["adj2_radius_km"] = adj2_radius_e
    ttk.Label(adj2g, text="Gaussian smoothing radius for rho_env. 0.1-0.5 km recommended for Hyperion 30 m pixels.",
              foreground="gray").grid(row=0, column=2, columnspan=2, sticky=tk.W)

    # Pixel size
    ttk.Label(adj2g, text="Pixel size (m):").grid(row=1, column=0, sticky=tk.W, pady=2)
    adj2_pixel_e = ttk.Entry(adj2g, width=10)
    adj2_pixel_e.insert(0, "30")
    adj2_pixel_e.grid(row=1, column=1, sticky=tk.W, padx=4)
    e["pixel_size_m"] = adj2_pixel_e
    ttk.Label(adj2g, text="Hyperion GSD = 30 m",
              foreground="gray").grid(row=1, column=2, sticky=tk.W)

    # SOS solver harmonics
    sosg = ttk.LabelFrame(tab2, text="SOS solver", padding=8)
    sosg.pack(fill=tk.X, pady=(8, 0))

    ttk.Label(sosg, text="Fourier harmonics:").grid(row=0, column=0, sticky=tk.W)
    harmonics_cb = _combo(sosg, "n_harmonics",
                          values=["1", "2", "3", "4", "6", "8", "10", "15", "20", "40", "81"],
                          default="3", width=6,
                          row=0, column=1, sticky=tk.W, padx=6)
    ttk.Label(sosg,
              text="3 = exact for Rayleigh, AOT<0.5  |  ≥6 for heavy aerosol  |  81 = Fortran default",
              foreground="gray").grid(row=0, column=2, sticky=tk.W)

    ttk.Label(sosg, text="Worker processes:").grid(row=1, column=0, sticky=tk.W, pady=(4,0))
    import os as _os
    n_cpu     = _os.cpu_count() or 1
    n_default = max(1, (n_cpu + 1) // 2)   # half the cores, rounded up
    ttk.Label(sosg, text="Worker processes:").grid(row=1, column=0, sticky=tk.W, pady=3)
    workers_sb = tk.Spinbox(sosg, from_=1, to=n_cpu * 2,
                            increment=1, width=5, justify=tk.CENTER)
    workers_sb.delete(0, tk.END)
    workers_sb.insert(0, str(n_default))
    workers_sb.grid(row=1, column=1, sticky=tk.W, padx=6)
    e["n_workers"] = workers_sb
    ttk.Label(sosg,
              text=f"default={n_default} (½ cores)  |  max useful = {n_cpu}  — each band runs in its own process",
              foreground="gray").grid(row=1, column=2, sticky=tk.W, pady=(4, 0))

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
    # ── Threading infrastructure ──────────────────────────────────────────────
    import queue as _queue
    import threading as _threading

    _log_queue   = _queue.Queue()   # log messages: worker thread → GUI
    _worker_thread = [None]         # reference to running thread
    _log_fh      = [None]           # open log file handle (or None)

    # Progress bar (indeterminate pulse while running)
    prog_bar = ttk.Progressbar(btn_bar, mode="determinate", length=220, maximum=100)

    def _poll_log():
        """Drain the log queue every 100 ms in the GUI thread."""
        try:
            while True:
                msg = _log_queue.get_nowait()
                if msg is None:       # sentinel: worker finished
                    prog_bar["value"] = 100
                    root.after(400, prog_bar.pack_forget)
                    run_btn.config(state=tk.NORMAL)
                    stop_btn.config(state=tk.DISABLED)
                    if _log_fh[0]:
                        try:
                            _log_fh[0].write("=== END ===\n")
                            _log_fh[0].flush()
                            _log_fh[0].close()
                        except Exception:
                            pass
                        _log_fh[0] = None
                    return
                if msg.startswith("__PROGRESS__ "):
                    _, done_s, total_s = msg.split()
                    pct = int(int(done_s) / int(total_s) * 100)
                    prog_bar["value"] = pct
                    continue
                if msg == "__RESET__":
                    prog_bar["value"] = 0
                    continue
                log_text.insert(tk.END, msg + "\n")
                log_text.see(tk.END)
                # Write to log file immediately and flush so crash leaves a trail
                if _log_fh[0]:
                    try:
                        _log_fh[0].write(msg + "\n")
                        _log_fh[0].flush()
                    except Exception:
                        try:
                            _log_fh[0].close()
                        except Exception:
                            pass
                        _log_fh[0] = None
        except _queue.Empty:
            pass
        root.after(100, _poll_log)

    def _worker(params):
        """Run correct_hyperion in a background thread."""
        def threadsafe_log(msg, end="\n"):
            _log_queue.put(msg)
        try:
            out_hdr = correct_hyperion(params, log=threadsafe_log)
            _log_queue.put(f"\n✓ Done → {os.path.basename(out_hdr)}")
            root.after(0, lambda: set_status(f"Done → {os.path.basename(out_hdr)}"))
        except Exception as ex:
            import traceback
            _log_queue.put(traceback.format_exc())
            root.after(0, lambda: set_status(f"Error: {ex}"))
        finally:
            _log_queue.put(None)   # sentinel

    def run_correction():
        if _worker_thread[0] and _worker_thread[0].is_alive():
            set_status("Already running — please wait.")
            return

        run_btn.config(state=tk.DISABLED)
        stop_btn.config(state=tk.NORMAL)
        log_text.delete("1.0", tk.END)
        nb.select(tab3)
        set_status("Running...")

        hdr_path = _get("hdr_file")
        if not hdr_path or not os.path.isfile(hdr_path):
            set_status("Error: HDR file not found.")
            run_btn.config(state=tk.NORMAL)
            stop_btn.config(state=tk.DISABLED)
            return

        try:
            hdr_data = read_hdr(hdr_path)
        except Exception as ex:
            set_status(f"HDR error: {ex}")
            run_btn.config(state=tk.NORMAL)
            stop_btn.config(state=tk.DISABLED)
            return

        mask_path = _get("mask_file") or None
        if mask_path and not os.path.isfile(mask_path):
            mask_path = None

        params = dict(
            hdr_file       = hdr_path,
            out_base       = _get("out_base") or os.path.splitext(hdr_path)[0] + "_6S",
            acq_date       = f"{_get('month')}/{_get('day')}",
            start_time     = "",
            month          = _get("month", int),
            day            = _get("day",   int),
            sza            = _get("sza",   float),
            saa            = _get("saa",   float),
            vza            = _get("vza",   float),
            vaa            = _get("vaa",   float),
            sensor_name    = _get("sensor_name"),
            input_type     = _get("input_type"),
            rad_scale_spec = _get("rad_scale_spec"),
            idatm          = ATM_MODELS.get(_get("idatm_name"), 2),
            uh2o           = _get("uh2o",          float),
            uo3            = _get("uo3",            float),
            iaer           = AEROSOL_MODELS.get(_get("iaer_name"), 1),
            aot550         = _get("aot550",         float),
            target_alt_km  = _get("target_alt_km",  float),
            mask_file      = mask_path,
            config_file    = (_get("config_file") or None)
                             if _get("config_file") not in ("", "<not saved>") else None,
            spec_csv_file  = (_get("spec_csv_file") or None)
                             if _get("spec_csv_file") not in ("", "<not saved>") else None,
            interleave     = _get("interleave") or "bip",
            drop_bad_bands = _drop[0],
            do_adj2        = _adj2[0],
            adj2_out_base  = (_get("adj2_out_base") or None) if _adj2[0] else None,
            adj2_radius_km = _get("adj2_radius_km", float),
            pixel_size_m   = _get("pixel_size_m",   float),
            n_harmonics    = int(_get("n_harmonics") or 3),
            n_workers      = int(_get("n_workers")   or 1),
            env_model      = ENV_MODELS.get(_get("env_model_name"), None),
            env_radius_km  = _get("env_radius_km", float),
            **{k: hdr_data[k] for k in
               ["n_bands","n_lines","n_samples","wl_nm","wl_um","fwhm_nm",
                "bbl","envi_meta"]},
        )

        # Clear any stale queue messages
        while not _log_queue.empty():
            try: _log_queue.get_nowait()
            except _queue.Empty: break

        # ── Open log file ──────────────────────────────────────────────────
        log_path = e["log_file"].get().strip()
        if log_path and log_path not in ("<not saved>", ""):
            try:
                _log_fh[0] = open(log_path, "w", encoding="utf-8", buffering=1)
                import datetime as _dt_log
                _log_fh[0].write(f"=== 6S Atmospheric Correction Log ===\n")
                _log_fh[0].write(f"Started: {_dt_log.datetime.now().isoformat(timespec='seconds')}\n")
                _log_fh[0].write(f"\n--- Input parameters ---\n")
                # Write all filled fields and checked options
                for label, key in [
                    ("HDR file",         "hdr_file"),
                    ("Output base",      "out_base"),
                    ("MTL file",         "mtl_file"),
                    ("Acq. date",        "acq_date"),
                    ("Start time",       "start_time"),
                    ("Month / day",      None),
                    ("SZA (deg)",        "sza"),
                    ("SAA (deg)",        "saa"),
                    ("VZA (deg)",        "vza"),
                    ("VAA (deg)",        "vaa"),
                    ("Sensor",           "sensor_name"),
                    ("Input type",       "input_type"),
                    ("Radiance scale",   "rad_scale_spec"),
                    ("Atm profile",      "idatm_name"),
                    ("H2O (g/cm2)",      "uh2o"),
                    ("O3 (cm-atm)",      "uo3"),
                    ("Aerosol model",    "iaer_name"),
                    ("AOT @ 550nm",      "aot550"),
                    ("Target alt (km)",  "target_alt_km"),
                    ("Env model",        "env_model_name"),
                    ("Env radius (km)",  "env_radius_km"),
                    ("Adj2 radius (km)", "adj2_radius_km"),
                    ("Pixel size (m)",   "pixel_size_m"),
                    ("N harmonics",      "n_harmonics"),
                    ("N workers",        "n_workers"),
                    ("Interleave",       "interleave"),
                    ("Mask file",        "mask_file"),
                    ("6S config file",   "config_file"),
                    ("Spectral CSV",     "spec_csv_file"),
                    ("Log file",         "log_file"),
                    ("Adj2 output",      "adj2_out_base"),
                ]:
                    if key is None:
                        m_val = f"{_get('month')}/{_get('day')}"
                        _log_fh[0].write(f"  {'Month / day':<22}: {m_val}\n")
                        continue
                    val = _get(key)
                    if val:
                        _log_fh[0].write(f"  {label:<22}: {val}\n")
                # Checkboxes
                _log_fh[0].write(f"  {'Drop bad bands':<22}: {_drop[0]}\n")
                _log_fh[0].write(f"  {'Stage-2 adjacency':<22}: {_adj2[0]}\n")
                _log_fh[0].write(f"\n--- Processing log ---\n")
                _log_fh[0].flush()
            except Exception as ex:
                set_status(f"Log file error: {ex}")
                _log_fh[0] = None

        prog_bar.pack(side=tk.LEFT, padx=(10, 0))
        prog_bar["value"] = 0

        t = _threading.Thread(target=_worker, args=(params,), daemon=True)
        _worker_thread[0] = t
        t.start()
        root.after(100, _poll_log)

    run_btn = ttk.Button(btn_bar, text="Run correction", command=run_correction)
    run_btn.pack(side=tk.LEFT, padx=(0, 6))
    stop_btn = ttk.Button(btn_bar, text="Stop", state=tk.DISABLED,
                          command=lambda: set_status("Stop requested — waiting for current band..."))
    stop_btn.pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_bar, text="Close",
               command=lambda: (root.quit(), root.destroy())).pack(side=tk.LEFT)

    root.mainloop()

# =============================================================================
# SECTION 5 -- Entry point
# =============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Script mode: hyperspectral_correct.py file.hdr [file.L1T]
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