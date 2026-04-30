# Hyperion Atmospheric Correction — Processing Description

## Overview

This script corrects EO-1 Hyperion L1T hyperspectral imagery for atmospheric
effects using the 6S radiative transfer model (Second Simulation of the
Satellite Signal in the Solar Spectrum, Vermote et al. 1997). The correction
converts top-of-atmosphere (TOA) radiance to surface reflectance.

---

## Input data

| Item | Description |
|------|-------------|
| ENVI HDR file | Hyperion L1T image in ENVI format (BIL, BSQ or BIP interleave) |
| MTL file | Landsat-style metadata file containing acquisition geometry, date and radiance scaling factors |
| Mask file (optional) | Single-band ENVI file marking valid pixels (non-zero = valid). If absent, pixels where all bands are zero are treated as fill (black edges). |

---

## Radiance calibration

Hyperion L1T data are stored as scaled integers. Radiance is recovered as:

    L [W/m²/sr/µm] = DN / scaling_factor

where the scaling factor is read from the MTL file:

- VNIR detector (bands 1–70, 356–1058 nm): factor = 40
- SWIR detector (bands 71–242, 852–2577 nm): factor = 80

Note: the unit is **W/m²/sr/µm**, not µW/cm²/sr/nm. The 6S solar spectrum
(solirr) is also in W/m²/µm, so no unit conversion is needed.

---

## View azimuth angle

The MTL file provides `SENSOR_LOOK_ANGLE` with a sign but no azimuth:
positive = east of nadir, negative = west of nadir.

The view azimuth angle (VAA) required by 6S is computed from the EO-1 orbital
geometry. EO-1 is in a sun-synchronous orbit (inclination = 98.7°) and
acquires standard data on the descending node. The ground track azimuth at
the scene centre latitude φ is:

    az_track = 180° − arcsin(|cos(inclination)| / cos(φ))

The cross-track look azimuth is then:

    VAA = az_track + 90°   (east-looking, positive SENSOR_LOOK_ANGLE)
    VAA = az_track − 90°   (west-looking, negative SENSOR_LOOK_ANGLE)

6S uses azimuths measured clockwise from north, the same convention as the
solar azimuth (SAA).

---

## 6S configuration

6S is run once per usable band in monochromatic mode (iwave = -1). The
wavelength is the band centre in µm from the ENVI header. All other
parameters are fixed for all bands:

| 6S parameter | Value | Source |
|---|---|---|
| igeom = 0 | User geometry | — |
| SZA, SAA | Solar angles | MTL |
| VZA | abs(SENSOR_LOOK_ANGLE) | MTL |
| VAA | Computed from orbit | See above |
| idatm | Atmospheric profile (0–8) | User |
| H₂O, O₃ | Gas columns | User (if idatm = 8) |
| iaer | Aerosol model | User |
| AOT₅₅₀ | Aerosol optical depth | User |
| xps | −target_altitude_km | User |
| xpp = −1000 | Satellite sensor | Fixed |
| inhomo | Surface model (0 = uniform, 1 = mixed) | User |
| rapp = −2 | Forward simulation (no retrieval) | Fixed |

The TOA solar irradiance E₀(λ) for each band is taken directly from the 6S
built-in solar spectrum (solirr function), corrected for the Earth–Sun
distance on the acquisition date using the Spencer (1971) formula:

    E₀ = solirr(λ) × (R₀/R)²

where (R₀/R)² is the ratio squared of mean to actual Earth–Sun distance,
computed from the day of year.

---

## Atmospheric correction coefficients

From each 6S run the following quantities are extracted:

| Symbol | 6S output key | Description |
|--------|--------------|-------------|
| xₐ | srotot | Atmospheric path reflectance |
| T↓ | sdtott | Total downward transmittance |
| T↑ | sutott | Total upward transmittance |
| S | pizera | Atmospheric spherical albedo |

Derived correction coefficients:

    xb = 1 / (T↓ · T↑)
    xc = S · xb

---

## Stage-1 correction (uniform surface assumption)

TOA reflectance:

    ρ_toa = π · L / (E₀ · cos(SZA))

Surface reflectance (Vermote et al. 1997, eq. 7):

    ρ_s1 = (ρ_toa − xₐ) / (xb · ρ_toa + xc)

This formula assumes that every pixel is surrounded by a surface with the same
reflectance as the pixel itself (uniform environment). The output is stored as
int16 scaled by 10000. Nodata = −9999.

---

## Stage-2 adjacency correction (optional)

The Stage-1 correction treats the atmosphere–surface coupling as if every
pixel were surrounded by a uniform surface at its own reflectance. In reality,
light scattered from bright neighbouring pixels reaches dark pixels through
the atmosphere, making dark areas appear brighter than they are (adjacency
effect).

### Environmental reflectance

The environmental reflectance ρ_env(x, y) is estimated by Gaussian-smoothing
the Stage-1 surface reflectance image:

    ρ_env = gaussian_filter(ρ_s1,  σ = radius_km × 1000 / pixel_size_m)

The kernel radius (in km) represents the spatial scale of atmospheric
scattering. For a clean atmosphere (AOT ≈ 0.05) a radius of 0.5–1 km is
appropriate; for heavy aerosol loading (AOT > 0.3) up to 3–5 km may be needed.

### Stage-2 retrieval formula

Step 1 — Recover ρ_toa from Stage-1 result (exact inversion):

    ρ_toa = (xₐ + xc · ρ_s1) / (1 − xb · ρ_s1)

Step 2 — Apply the full Vermote & Tanré (1992) formula with spatial ρ_env:

    ρ_s2 = [(ρ_toa − xₐ) · (1 − S · ρ_env) / T↑  −  S · ρ_env] / T↓

This is the complete expression from Vermote & Tanré (1992, eq. 10). It
reduces to the Stage-1 formula when ρ_env = ρ_s1.

**Note on the approximation:** A common simplified version

    ρ_s2 ≈ ρ_s1 + (ρ_s1 − ρ_env) · S / T↓

amplifies errors in ρ_env by S/T↓ ≈ 0.9 in the visible, producing noise
much larger than the correction itself. The full formula above is used instead.

---

## Output files

### Stage-1: `<output_base>.hdr / .img`
Surface reflectance, scaled int16 × 10000. Bad bands (bbl = 0) are excluded
by default. Band names reference the original Hyperion band numbers.

### Stage-2: `<adj2_output_base>.hdr / .img` (if enabled)
Adjacency-corrected surface reflectance, same format as Stage-1.

Both output files include a `description` field in the header with all key
processing parameters (geometry, atmospheric model, AOT, date, etc.).

---

## References

- Vermote, E.F., Tanré, D., Deuzé, J.L., Herman, M., & Morcrette, J.-J. (1997).
  Second Simulation of the Satellite Signal in the Solar Spectrum, 6S: An Overview.
  *IEEE Transactions on Geoscience and Remote Sensing*, 35(3), 675–686.

- Vermote, E.F. & Tanré, D. (1992).
  Analytical expressions for radiative properties of planar Rayleigh scattering
  media, including polarization contributions.
  *Journal of Quantitative Spectroscopy and Radiative Transfer*, 47(4), 305–314.

- Spencer, J.W. (1971).
  Fourier series representation of the position of the sun.
  *Search*, 2(5), 172.