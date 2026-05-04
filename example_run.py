"""
example_run.py
--------------
Examples for running the 6S radiative transfer model via the Python API.

Install the package first:
    pip install -e .
or run directly (the sixs/ folder must be present):
    python example_run.py
"""

import io
from sixs.sixs_main import run6S


# ---------------------------------------------------------------------------
# Example 1 — Simple satellite observation, Lambertian sand surface
# ---------------------------------------------------------------------------

input_text_1 = """
0                          ! igeom: user-defined geometry
30.0 0.0 0.0 180.0 7 1    ! SZA SAA VZA VAA month day
2                          ! idatm: mid-latitude summer atmosphere
1                          ! iaer:  continental aerosol model
23.0                       ! visibility [km]
0.0                        ! xps:  target at sea level
-1000                      ! xpp:  satellite sensor
5                          ! iwave: AVHRR NOAA-6 band 1 (0.55-0.75 µm)
0                          ! inhomo: uniform surface
0                          ! idirec: Lambertian
3                          ! igroun: sand spectrum
-2.0                       ! rapp:  forward simulation (no retrieval)
"""

print("=" * 60)
print("Example 1: continental aerosol, sand surface, AVHRR band 1")
print("=" * 60)
r1 = run6S(io.StringIO(input_text_1))

print(f"  Apparent reflectance       : {r1['apparent_reflectance']:.4f}")
print(f"  Path reflectance (srotot)  : {r1['srotot']:.4f}")
print(f"    Rayleigh–aerosol coupling: {r1['sroray']:.4f}  (see note below)")
print(f"    Aerosol contribution     : {r1['sroaer']:.4f}")
print(f"  AOT at 550 nm              : {r1['aot550']:.4f}")
print(f"  Direct surface irradiance  : {r1['direct_irr']:.2e} W/m²/µm")
print(f"  Diffuse surface irradiance : {r1['diffuse_irr']:.2e} W/m²/µm")
print(f"  Down transmittance (total) : {r1['sdtott']:.4f}")
print(f"  Up transmittance (total)   : {r1['sutott']:.4f}")
print(f"  Spherical albedo (total)   : {r1['spherical_albedo_tot']:.4f}  <- use as 's' in retrieval")
print(f"  pizera (aerosol SSA ω₀)   : {r1['pizera']:.4f}  <- NOT spherical albedo")
print()
print("  Note on sroray: in the 6S SOS decomposition, sroray is the")
print("  Rayleigh–aerosol coupling correction, not the Rayleigh path")
print("  reflectance itself. It can be negative in the blue because aerosol")
print("  forward-scattering reduces the Rayleigh backscatter contribution.")
print("  The total path refl. seen by the sensor = chand(tau_R) + sroaer.")
print("  srotot (= sroray + sroaer) is correct as xa in the retrieval formula.")
print()


# ---------------------------------------------------------------------------
# Example 2 — Same scene, retrieve surface reflectance from apparent refl.
# ---------------------------------------------------------------------------

input_text_2 = """
0
30.0 0.0 0.0 180.0 7 1
2
1
23.0
0.0
-1000
5
0
0
3
-0.18      ! rapp < 0: retrieve surface refl from apparent reflectance 0.18
"""

print("=" * 60)
print("Example 2: atmospheric correction (retrieve surface refl.)")
print("=" * 60)
r2 = run6S(io.StringIO(input_text_2))

print(f"  Input apparent reflectance  : 0.18")
print(f"  Retrieved surface refl (rog): {r2['rog']:.4f}")
print()


# ---------------------------------------------------------------------------
# Example 3 — Kuusk canopy BRDF, Landsat TM band 4 (NIR)
# ---------------------------------------------------------------------------

input_text_3 = """
0
30.0 0.0 0.0 90.0 7 1
2
1
23.0
0.0
-1000
28                         ! iwave: TM Landsat-5 band 4 NIR (0.73-0.95 µm)
0
1                          ! idirec: BRDF surface
9                          ! ibrdf:  Kuusk MSRM canopy model
0.7 57.0 3.0 0.05 0.10    ! ee thm LAI sl rsl1
1.0 40.0 0.02 1.5          ! rnc cab(µg/cm²) cw(cm) N
-2.0
"""

print("=" * 60)
print("Example 3: Kuusk canopy BRDF, TM4 NIR")
print("=" * 60)
r3 = run6S(io.StringIO(input_text_3))

print(f"  Apparent reflectance        : {r3['apparent_reflectance']:.4f}")
print(f"  Direct surface irradiance   : {r3['ground_direct_irr']:.1f} W/m²/µm")
print(f"  Diffuse surface irradiance  : {r3['ground_diffuse_irr']:.1f} W/m²/µm")
print()


# ---------------------------------------------------------------------------
# Example 4 — Apparent reflectance vs solar zenith angle
# ---------------------------------------------------------------------------

print("=" * 60)
print("Example 4: apparent reflectance vs solar zenith angle")
print("=" * 60)
print(f"  {'SZA':>6}  {'rapp':>8}  {'direct_irr':>14}  {'diffuse_irr':>14}")

for sza in [10, 20, 30, 40, 50, 60, 70]:
    inp = f"""
0
{sza}.0 0.0 0.0 180.0 7 1
2
1
23.0
0.0
-1000
5
0
0
3
-2.0
"""
    r = run6S(io.StringIO(inp))
    print(f"  {sza:>5}°  {r['apparent_reflectance']:>8.4f}"
          f"  {r['direct_irr']:>14.2e}  {r['diffuse_irr']:>14.2e}")


# ---------------------------------------------------------------------------
# Example 5 — Run from a configuration file (data0)
#
# data0: aircraft measurement scenario
#   Geometry  : SZA=40°, SAA=100°, view zenith=45°, view azimuth=50°, July 23
#   Atmosphere : user H₂O=3.0 g/cm², O₃=0.35 cm-atm (US62 profile shape)
#   Aerosol   : equal-mix custom (25% each: dust, water-sol, oceanic, soot)
#               AOT @ 550 nm = 0.50
#   Target    : 0.2 km altitude, non-uniform surface (circular)
#               target=clear water, environment=vegetation, radius=0.5 km
#   Sensor    : aircraft at 3.3 km above ground, AVHRR NOAA-9 band 1
#   Output    : retrieve surface refl from apparent reflectance 0.10
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Example 5: run from configuration file  data0")
print("=" * 60)

# run6S accepts a file path directly
r5 = run6S("data0")

print(f"  Solar zenith angle    : {r5['sza']:.1f}°")
print(f"  Date                  : {r5['month']}/{r5['day']}")
print(f"  AOT at 550 nm         : {r5['aot550']:.4f}")
print(f"  Water vapour          : {r5['h2o']:.2f} g/cm²")
print(f"  Ozone                 : {r5['o3']:.4f} cm-atm")
print(f"  Apparent reflectance  : {r5['apparent_reflectance']:.4f}")
print(f"  Retrieved surface refl: {r5['rog']:.4f}")
print(f"  Direct surface irr.   : {r5['direct_irr']:.4e} W/m²/µm")
print(f"  Diffuse surface irr.  : {r5['diffuse_irr']:.4e} W/m²/µm")