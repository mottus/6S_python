"""
example_run.py
--------------
Worked examples for the 6S radiative transfer model Python API.

Install:
    pip install -e .          (from the sixs_python directory)
Or run directly if sixs/ is on the path:
    python example_run.py

Output dictionary of run6S — complete reference
------------------------------------------------
Input metadata:
    day, month          acquisition date used in the run
    sza                 solar zenith angle (degrees)
    h2o                 precipitable water vapour (g cm⁻²)
    o3                  ozone column (cm-atm, Dobson units / 1000)
    aot550              aerosol optical depth at 550 nm

Geometry:
    asol                solar zenith angle (degrees, same as sza)
    phi0                solar azimuth angle (degrees)
    avis                view zenith angle (degrees)
    phiv                view azimuth angle (degrees)
    xmus                cos(solar zenith)
    xmuv                cos(view zenith)
    xmud                cos(scattering angle)

Apparent reflectance (band-integrated, dimensionless):
    apparent_reflectance         TOA reflectance = π L_toa / (E₀ cos SZA)
                                 Includes surface + atmospheric path signal.
    apparent_reflectance_atm_only  atmospheric path contribution only (black surface)
    apparent_reflectance_Rayleigh  Rayleigh-only path contribution
    apparent_radiance            TOA radiance (W m⁻² sr⁻¹ µm⁻¹)

Surface reflectance (only meaningful when rapp < 0, i.e. retrieval mode):
    rog                 retrieved surface reflectance

Path reflectance decomposition (SOS solver output):
    srotot              total path reflectance = sroray + sroaer
                        CAUTION: this is NOT the physical path refl. seen by the
                        sensor.  Rayleigh path refl. from chand() is ~0.13 at
                        427 nm; srotot is the SOS coupling residual.  srotot IS
                        the correct xa to use in the retrieval formula because
                        xb/xc already encode the full Rayleigh transmittance.
    sroray              Rayleigh–aerosol coupling correction (can be negative in
                        the blue because aerosol forward-scattering reduces the
                        Rayleigh backscatter contribution)
    sroaer              aerosol contribution to path reflectance

Optical depths (band-integrated):
    sodray              Rayleigh optical depth
    sodaer              aerosol optical depth
    sodtot              total (Rayleigh + aerosol) optical depth

Spherical albedo (band-integrated):
    spherical_albedo_tot (sast)  TOTAL atmospheric spherical albedo.  USE THIS
                                 as 's' in the retrieval formula denominator:
                                 rho_s = (rho_toa - xa) / (T_d*T_u + s*(rho_toa-xa))
    spherical_albedo_ray (sasr)  Rayleigh component
    spherical_albedo_aer (sasa)  aerosol component
    pizera                       aerosol single-scatter albedo ω₀ (NOT spherical
                                 albedo; do NOT use as 's' in retrieval formula)

Transmittances (band-integrated):
    sdtott / sutott     total downward / upward transmittance (Rayleigh + aerosol + gas)
    sdtotr / sutotr     Rayleigh component of transmittance
    sdtota / sutota     aerosol component of transmittance
    sdwava / suwava / stwava   water vapour down / up / total transmittance
    sdozon / suozon / stozon   ozone down / up / total transmittance
    tgasm               total gas transmittance (all absorbers combined)
    dgasm               downward gas transmittance
    ugasm               upward gas transmittance

Ground irradiances (W m⁻² µm⁻¹ at the target surface):
    ground_direct_irr   direct beam irradiance  (= direct_irr)
    ground_diffuse_irr  diffuse sky irradiance
    ground_env_irr      environment (adjacency) irradiance
    ground_direct_fraction   fraction of E₀ cos(SZA) reaching surface as direct beam
    ground_diffuse_fraction  fraction reaching surface as diffuse
    ground_env_fraction      fraction reaching surface as environment

Satellite radiances at TOA (W m⁻² sr⁻¹ µm⁻¹ and W m⁻²):
    atm_radiance / atm_radiance_wm2        atmospheric path radiance
    env_radiance / env_radiance_wm2        environment contribution
    target_radiance / target_radiance_wm2  target surface contribution

Band-integrated irradiance (filter-weighted):
    direct_irr     direct beam at surface (W m⁻² µm⁻¹)  = ground_direct_irr
    diffuse_irr    diffuse sky at surface (W m⁻² µm⁻¹)  = ground_diffuse_irr

Per-wavelength arrays (one value per wavelength in the spectral loop):
    spec_wl    wavelength (µm)
    spec_dir   direct irradiance at that wavelength (W m⁻² µm⁻¹)
    spec_dif   diffuse + environment irradiance (W m⁻² µm⁻¹)
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
-2.0                       ! rapp: forward simulation (no retrieval)
"""

print("=" * 60)
print("Example 1: continental aerosol, sand surface, AVHRR band 1")
print("=" * 60)
r1 = run6S(io.StringIO(input_text_1))

print(f"  apparent_reflectance       : {r1['apparent_reflectance']:.4f}")
print(f"  path refl. (srotot)        : {r1['srotot']:.4f}")
print(f"    Rayleigh coupling (sroray): {r1['sroray']:.4f}  (see note below)")
print(f"    Aerosol (sroaer)          : {r1['sroaer']:.4f}")
print(f"  AOT at 550 nm              : {r1['aot550']:.4f}")
print(f"  Rayleigh OD (sodray)       : {r1['sodray']:.4f}")
print(f"  Aerosol OD  (sodaer)       : {r1['sodaer']:.4f}")
print(f"  Direct irr. at surface     : {r1['ground_direct_irr']:.1f} W/m²/µm")
print(f"  Diffuse irr. at surface    : {r1['ground_diffuse_irr']:.1f} W/m²/µm")
print(f"  Total downward T (sdtott)  : {r1['sdtott']:.4f}")
print(f"  Total upward T   (sutott)  : {r1['sutott']:.4f}")
print(f"  Spherical albedo (sast)    : {r1['spherical_albedo_tot']:.4f}  <- use as 's'")
print(f"  pizera (aerosol SSA ω₀)   : {r1['pizera']:.4f}  <- NOT spherical albedo")
print(f"  Gas transmittance (tgasm)  : {r1['tgasm']:.4f}")
print()
print("  Note on sroray: in the 6S SOS decomposition, sroray is the")
print("  Rayleigh-aerosol coupling correction, not the Rayleigh path")
print("  reflectance. It is negative in the blue because aerosol forward-")
print("  scattering reduces the Rayleigh backscatter contribution.")
print("  srotot (= sroray + sroaer) is correct as xa in the retrieval formula.")
print()


# ---------------------------------------------------------------------------
# Example 2 — Retrieve surface reflectance from apparent reflectance
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

print(f"  apparent_reflectance        : {r3['apparent_reflectance']:.4f}")
print(f"  target_radiance             : {r3['target_radiance']:.4f} W/m²/sr/µm")
print(f"  atm_radiance                : {r3['atm_radiance']:.4f} W/m²/sr/µm")
print(f"  ground_direct_irr           : {r3['ground_direct_irr']:.1f} W/m²/µm")
print(f"  ground_diffuse_irr          : {r3['ground_diffuse_irr']:.1f} W/m²/µm")
print(f"  ground_env_irr              : {r3['ground_env_irr']:.1f} W/m²/µm")
print()


# ---------------------------------------------------------------------------
# Example 4 — Apparent reflectance vs solar zenith angle
# ---------------------------------------------------------------------------

print("=" * 60)
print("Example 4: apparent reflectance vs solar zenith angle")
print("=" * 60)
print(f"  {'SZA':>6}  {'rapp':>8}  {'direct_irr':>12}  {'sast':>8}  {'sodray':>8}")

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
    print(f"  {sza:>5}°  {r['apparent_reflectance']:>8.4f}  "
          f"{r['direct_irr']:>12.2e}  "
          f"{r['spherical_albedo_tot']:>8.5f}  "
          f"{r['sodray']:>8.4f}")


# ---------------------------------------------------------------------------
# Example 5 — Run from a configuration file (data0)
#
# data0 describes an aircraft measurement:
#   Geometry  : SZA=40°, SAA=100°, view zenith=45°, view azimuth=50°, July 23
#   Atmosphere : user H₂O=3.0 g/cm², O₃=0.35 cm-atm (US62 profile shape)
#   Aerosol   : equal-mix custom (25% each: dust, water-sol, oceanic, soot)
#               AOT @ 550 nm = 0.50
#   Target    : 0.2 km altitude, non-uniform surface (circular, 0.5 km radius)
#               target = clear water, environment = vegetation
#   Sensor    : aircraft at 3.3 km above ground, AVHRR NOAA-9 band 1
#   Output    : retrieve surface refl from apparent reflectance 0.10
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Example 5: run from configuration file 'data0'")
print("=" * 60)

# run6S accepts a file path string directly
r5 = run6S("data0")

print(f"  Solar zenith angle     : {r5['sza']:.1f}°")
print(f"  Date                   : month={r5['month']}, day={r5['day']}")
print(f"  H₂O / O₃              : {r5['h2o']:.2f} g/cm²  /  {r5['o3']:.4f} cm-atm")
print(f"  AOT at 550 nm          : {r5['aot550']:.4f}")
print(f"  apparent_reflectance   : {r5['apparent_reflectance']:.4f}")
print(f"  Retrieved surface refl : {r5['rog']:.4f}")
print(f"  direct_irr             : {r5['direct_irr']:.4e} W/m²/µm")
print(f"  diffuse_irr            : {r5['diffuse_irr']:.4e} W/m²/µm")
print(f"  ground_env_irr         : {r5['ground_env_irr']:.4e} W/m²/µm")
print(f"  sodray / sodaer / sodtot: "
      f"{r5['sodray']:.4f} / {r5['sodaer']:.4f} / {r5['sodtot']:.4f}")
print(f"  sdtott / sutott        : {r5['sdtott']:.4f} / {r5['sutott']:.4f}")
print(f"  tgasm / dgasm / ugasm  : "
      f"{r5['tgasm']:.4f} / {r5['dgasm']:.4f} / {r5['ugasm']:.4f}")
print(f"  sast (spherical albedo): {r5['spherical_albedo_tot']:.5f}")
print(f"  pizera (aerosol SSA ω₀): {r5['pizera']:.5f}")
print(f"  srotot / sroray / sroaer: "
      f"{r5['srotot']:.5f} / {r5['sroray']:.5f} / {r5['sroaer']:.5f}")
print(f"  Per-wavelength arrays  : {len(r5['spec_wl'])} points, "
      f"wl range {min(r5['spec_wl']):.4f}–{max(r5['spec_wl']):.4f} µm")