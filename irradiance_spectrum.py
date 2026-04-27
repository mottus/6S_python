"""
irradiance_spectrum.py
----------------------
Compute spectral ground-level irradiance from the Hyytiälä 2019
configuration, by running 6S monochromatically across 0.4 – 2.5 µm.

Place this file in the same folder as setup.py and data_Hyytiala2019,
then run:

    python irradiance_spectrum.py

Output columns (W/m²/µm):
    wl       – wavelength (µm)
    dir      – direct-beam irradiance at surface
    dif      – diffuse (sky) irradiance at surface
    total    – dir + dif
"""

import io
import warnings
import numpy as np
from sixs.sixs_main import run6S

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Spectral grid: 0.40 – 2.50 µm at 0.05 µm steps (43 points)
# Finer grid example: np.arange(0.40, 2.51, 0.025)  →  85 points
# ---------------------------------------------------------------------------
# WL_GRID = np.arange(0.40, 2.51, 0.05)
WL_GRID = np.arange(0.40, 2.51, 0.005)

# ---------------------------------------------------------------------------
# Configuration — translated from data_Hyytiala2019
#
# Hyytiälä forest station, July 13 2019
#   igeom=0  : user geometry
#   asol=52.7°, phi0=100°, avis=0° (nadir), phi_v=0°
#   idatm=8  : US62 profile, user H2O=1.5 g/cm², O3=0.40 cm-atm
#   iaer=1   : continental aerosol, AOT@550=0.08 (clean boreal conditions)
#   xps=-0.2 : target at 200 m altitude (Hyytiälä ~180 m a.s.l.)
#   xpp=-1000: satellite → full-column irradiance reaching the surface
#   iwave=-1 : monochromatic, wavelength set per call (see {wl} below)
#   inhomo=0 : uniform surface
#   idirec=0 : Lambertian
#   igroun=1 : vegetation spectrum
#   rapp=-2  : forward simulation, no retrieval
# ---------------------------------------------------------------------------
CONFIG_TEMPLATE = """
0
52.7 100.0 0.0 0.0 7 13
8
1.5 0.40
1
0
0.08
-0.2
-1000
-1
{wl}
0
0
1
-2.0
"""


def run_spectrum(wl_grid, template):
    """Run 6S once per wavelength and return arrays of irradiance."""
    wl_out  = []
    dir_out = []
    dif_out = []

    for wl in wl_grid:
        print("#", end="")
        inp = template.format(wl=f"{wl:.4f}")
        try:
            r = run6S(io.StringIO(inp), io.StringIO())   # suppress Matti print line
            wl_out.append(wl)
            dir_out.append(r["m_dir"])
            dif_out.append(r["m_dif"])
        except Exception as e:
            # Strong gas-absorption bands can be numerically fragile; record NaN
            print(f"  Warning: wl={wl:.3f} µm failed ({e}), storing NaN")
            wl_out.append(wl)
            dir_out.append(float("nan"))
            dif_out.append(float("nan"))

    return np.array(wl_out), np.array(dir_out), np.array(dif_out)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"Running 6S at {len(WL_GRID)} wavelengths "
      f"({WL_GRID[0]:.2f}–{WL_GRID[-1]:.2f} µm, step {WL_GRID[1]-WL_GRID[0]:.3f} µm) …")

wl, direct, diffuse = run_spectrum(WL_GRID, CONFIG_TEMPLATE)
total = direct + diffuse

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print()
print(f"{'wl (µm)':>9}  {'direct':>14}  {'diffuse':>14}  {'total':>14}  W/m²/µm")
print("-" * 60)
for i in range(len(wl)):
    print(f"{wl[i]:9.3f}  {direct[i]:14.2f}  {diffuse[i]:14.2f}  {total[i]:14.2f}")

# ---------------------------------------------------------------------------
# Band-integrated totals (trapezoidal rule)
# ---------------------------------------------------------------------------
def _trapz(y, x):
    """Trapezoidal integration, compatible with numpy 1.x and 2.x."""
    try:
        return np.trapezoid(y, x)   # numpy >= 2.0
    except AttributeError:
        return np.trapz(y, x)       # numpy < 2.0

mask = np.isfinite(total)
if mask.sum() > 1:
    integrated = _trapz(total[mask], wl[mask])
    print(f"\nBand-integrated irradiance (0.4–2.5 µm): {integrated:.1f} W/m²")
    print(f"  of which direct : {_trapz(direct[mask], wl[mask]):.1f} W/m²")
    print(f"  of which diffuse: {_trapz(diffuse[mask], wl[mask]):.1f} W/m²")

with open("SQratio.txt", "w") as f:
    f.write("# wl(nm)\tSQratio\n")
    for w, t in zip(wl*1000, direct/total):
        f.write(f"{w:.0f}\t{t:.4f}\n")