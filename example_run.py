"""
example_run.py
--------------
Example script for running the 6S radiative transfer model.
Place this file in the same folder as setup.py, then install the package:

    pip install -e .

or run directly without installing (the sixs/ folder must be here):

    python example_run.py
"""

import io
from sixs.sixs_main import run

# simplest way to run on existing file
fn = "data0"
with open(fn) as f:
    results = run(f)
    # results = run(f, io.StringIO())   # redirect stdout to /dev/null

# results['m_kuu']        # month
# results['m_paev']       # day
# results['m_korgus']     # solar zenith angle (°)
# results['m_h2o']        # water vapour column (g/cm²)
# results['m_o']          # ozone column (cm-atm)
# results['m_aot']        # aerosol optical depth at 550 nm
# results['m_dir']        # direct ground irradiance, band-integrated (W/m²/µm × µm)
# results['m_dif']        # diffuse + env ground irradiance, same units
# results['m_sb']         # band weight (sum of filter × Δλ)
# results['m_seb']        # solar-weighted band integral
# results['apparent_reflectance']   # what the sensor sees (rapp)
# results['rog']          # retrieved or input surface reflectance
# results['pizera']       # surface spherical albedo
# results['srotot']       # total path reflectance
# results['sroray']       # Rayleigh component
# results['sroaer']       # aerosol component
# results['sodaer']       # aerosol optical depth (at band centre)
# results['sodray']       # Rayleigh optical depth
# results['sodtot']       # total optical depth
# results['sdtotr']       # downward Rayleigh transmittance
# results['sdtota']       # downward aerosol transmittance
# results['sdtott']       # total downward transmittance
# results['sutotr']       # upward Rayleigh transmittance
# results['sutota']       # upward aerosol transmittance
# results['sutott']       # total upward transmittance
# results['ground_direct_irr']    # direct beam (W/m²/µm)
# results['ground_diffuse_irr']   # sky diffuse (W/m²/µm)
# results['ground_env_irr']       # surface-reflected back down (W/m²/µm)
# results['ground_direct_fraction']
# results['ground_diffuse_fraction']
# results['ground_env_fraction']
# results['atm_radiance']         # path radiance (normalised)
# results['env_radiance']         # environmental radiance (normalised)
# results['target_radiance']      # target radiance (normalised)
# results['atm_radiance_wm2']     # path radiance (absolute, W/m²/sr/µm)
# results['env_radiance_wm2']     # environmental radiance (absolute, W/m²/sr/µm)
# results['target_radiance_wm2']  # target radiance (absolute, W/m²/sr/µm)
# results['spec_wl']   # list of wavelengths (µm) — one per spectral step
# results['spec_dir']  # direct beam irradiance at each wavelength (W/m²/µm)
# results['spec_dif']  # diffuse + environmental irradiance (W/m²/µm)

# ---------------------------------------------------------------------------
# Example 1 — Simple satellite observation, Lambertian sand surface
# ---------------------------------------------------------------------------

input_text_1 = """
0                          ! igeom: user-defined geometry
30.0 0.0 0.0 180.0 7 1    ! asol phi0 avis phiv month day
2                          ! idatm: mid-latitude summer atmosphere
1                          ! iaer:  continental aerosol
23.0                       ! visibility = 23 km
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
r1 = run(io.StringIO(input_text_1))

print(f"  Apparent reflectance : {r1['apparent_reflectance']:.4f}")
print(f"  Atmospheric path refl: {r1['srotot']:.4f}")
print(f"  Rayleigh reflectance : {r1['sroray']:.4f}")
print(f"  Aerosol reflectance  : {r1['sroaer']:.4f}")
print(f"  AOT at 550 nm        : {r1['m_aot']:.4f}")
print(f"  Direct irradiance    : {r1['m_dir']:.2e} W/m²/µm")
print(f"  Diffuse irradiance   : {r1['m_dif']:.2e} W/m²/µm")
print(f"  Down transmittance   : {r1['sdtott']:.4f}")
print(f"  Up transmittance     : {r1['sutott']:.4f}")
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
-0.18                      ! rapp < 0: retrieve surface refl from apparent refl 0.18
"""

print("=" * 60)
print("Example 2: atmospheric correction (retrieve surface refl.)")
print("=" * 60)
r2 = run(io.StringIO(input_text_2))

print(f"  Input apparent refl  : 0.18")
print(f"  Retrieved surface refl: {r2['rog']:.4f}")
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
r3 = run(io.StringIO(input_text_3))

print(f"  Apparent reflectance : {r3['apparent_reflectance']:.4f}")
print(f"  Ground direct irr.   : {r3['ground_direct_irr']:.1f} W/m²/µm")
print(f"  Ground diffuse irr.  : {r3['ground_diffuse_irr']:.1f} W/m²/µm")
print()


# ---------------------------------------------------------------------------
# Example 4 — Multiple geometries in a loop
# ---------------------------------------------------------------------------

print("=" * 60)
print("Example 4: apparent reflectance vs solar zenith angle")
print("=" * 60)
print(f"  {'SZA':>6}  {'rapp':>8}  {'direct_irr':>12}")

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
    r = run(io.StringIO(inp))
    print(f"  {sza:>6}°  {r['apparent_reflectance']:>8.4f}  {r['m_dir']:>12.2e}")


# ---------------------------------------------------------------------------
# Example 5 — Run from a configuration file (data0)
#
# data0 describes an aircraft measurement scenario:
#   Geometry  : SZA=40°, SAA=100°, view zenith=45°, view azimuth=50°, July 23
#   Atmosphere : user H2O=3.0 g/cm², O3=0.35 cm-atm (US62 profile shape)
#   Aerosol   : equal-mix custom (25% each of dust, water-sol, oceanic, soot)
#               AOT @ 550 nm = 0.50
#   Target    : 0.2 km altitude, non-uniform surface (circular target)
#               target=clear water, environment=vegetation, radius=0.5 km
#   Sensor    : aircraft at 3.3 km above ground, AVHRR NOAA-9 band 1
#               H2O=-1.5, O3=-0.35 (negative → interpolate from profile)
#               AOT under plane = 0.25
#   Output    : retrieve surface reflectance from apparent reflectance 0.10
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Example 5: run from configuration file  data0")
print("=" * 60)

with open("data0") as f:
    r5 = run(f)

print(f"  Solar zenith angle   : {r5['m_korgus']:.1f}°")
print(f"  Date                 : {r5['m_kuu']}/{r5['m_paev']}")
print(f"  AOT at 550 nm        : {r5['m_aot']:.4f}")
print(f"  Water vapour         : {r5['m_h2o']:.2f} g/cm²")
print(f"  Ozone                : {r5['m_o']:.4f} cm-atm")
print(f"  Apparent reflectance : {r5['apparent_reflectance']:.4f}")
print(f"  Retrieved surface refl (rog): {r5['rog']:.4f}")
print(f"  Direct irradiance    : {r5['m_dir']:.4e} W/m²/µm")
print(f"  Diffuse irradiance   : {r5['m_dif']:.4e} W/m²/µm")
