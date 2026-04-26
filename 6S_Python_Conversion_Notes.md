# 6S Radiative Transfer Model — Fortran 77 → Python Conversion Notes

## Overview

The 6S (Second Simulation of the Satellite Signal in the Solar Spectrum) Version 4.1
was translated from Fortran 77 to Python. The original source comprised **116 Fortran
files totalling approximately 30 500 lines**. The resulting Python package has **31
modules and ~17 000 lines** and preserves the same input/output interface as the
original binary.

The code originated from the Estonian Tartu Observatory and contains Estonian variable
names (`m_korgus`, `m_h2o`, `m_aot`, `m_dir`, `m_dif`, `m_paev`, `m_kuu`) in the
main output routine.

---

## Conversion Approach

### 1. Structural mapping

Each Fortran `COMMON` block was converted to a Python dataclass-like namespace object
(`commons.py`). This preserved the shared mutable state that Fortran subroutines
communicate through, without requiring a full redesign of the call graph.

| Fortran construct | Python equivalent |
|-------------------|-------------------|
| `COMMON /sixs_disc/` | `disc` namespace in `commons.py` |
| `COMMON /sixs_aer/` | `aer` namespace |
| `COMMON /sixs_sos/` | `trunc` namespace |
| `COMMON /sixs_ffu/` | `ffu` namespace |
| `COMMON /mie_in/` | `mie_in` namespace |
| `SUBROUTINE foo(…)` | `def foo(…):` in the appropriate module |
| `BLOCK DATA valeur` | Module-level Python lists/arrays |

### 2. Array indexing

Fortran arrays are 1-based; Python arrays are 0-based. Every occurrence was
adjusted. The most error-prone cases were arrays with *signed* index ranges used in
the successive-orders-of-scattering (SOS) solver:

```fortran
real rm(-mu:mu)      ! Fortran: index -mu … +mu
```

These were mapped to 0-based arrays of length `2*mu+1` using an offset function:

```python
def ji(k): return k + mu   # signed → 0-based
```

Forgetting this offset — and accidentally using Python's *negative index wraparound*
semantics — was the source of several hard-to-find bugs (see §4 below).

### 3. Data tables

All `DATA` statements for spectral lookup tables (gas absorption coefficients,
aerosol phase functions, leaf optical properties, soil basis functions, water
refractive indices, etc.) were parsed and embedded as Python lists. The largest
single table is `gas_tables.py` at ~8 400 lines.

### 4. Computed `GOTO` statements

The `gmd92` subroutine in `AKTOOL.f` uses Fortran assigned `GOTO` (a feature absent
in Python). These were rewritten as explicit `if/elif` chains that reproduce the
same branching logic.

---

## Bugs Found and Fixed

Ten bugs were identified and fixed during translation and validation. They fell into
four categories: **array indexing errors**, **numerical stability issues**, one
**fundamental algorithmic error** in the radiative transfer output, and two
**missing wiring** issues where translated components were not connected to the
input-parsing and dispatch infrastructure.

### Bug 1 — `trunc.pha` not populated before `trunca()`

**File:** `scattering.py`

`trunca()` computes the delta-M Legendre expansion of the aerosol phase function
from `trunc.pha`. The caller (the `discom` loop) never copied the relevant row of
`disc.phasel` into `trunc.pha`, so `trunca()` always operated on zeros.

**Fix:** Added `trunc.pha[:] = disc.phasel[l, :]` before each call to `trunca()`.

---

### Bug 2 — Divide-by-zero in the SOS integration loops

**Files:** `iso.py`, `os_module.py`

The layer-by-layer integration uses `exp(-f/yy)` where `yy = rm[ji(k)]` is a
Gauss quadrature cosine value. The boundary slots `ji(-mu)` and `ji(mu)` are
assigned special values (sun/view directions), but the interior slot `ji(0)` is
set to `-xmus`. If the solar zenith angle is exactly 90° then `xmus = 0` and the
division is undefined.

**Fix:** Added guards `if abs(yy) < 1e-10: continue` and replaced `exp(arg)` calls
with `exp(max(-87.0, arg))` throughout both modules.

---

### Bug 3 — Negative Python index wraparound in `atmref.py`

**File:** `atmref.py`

The Fortran array `rm(-mu:mu)` is addressed with `rm(-mu)` meaning "the element at
offset -mu". In the translated code `rm` is a plain 0-based array and `rm[-mu]`
means "count from the end", silently reading and writing the *wrong* element.

**Fix:** All accesses were rewritten using the offset function: `rm[ji(-mu)]`.

---

### Bug 4 — Same wraparound bug plus wrong sign for the sun direction

**File:** `scattering.py`

The same `-mu` wraparound appeared in six separate places inside `scatra()`.
Additionally, the solar-direction boundary condition was set to `rm[ji(0)] = +xmus`
(upward) instead of the correct `rm[ji(0)] = -xmus` (downward).

**Fix:** All six index expressions were corrected and the sign was flipped.

---

### Bug 5 — Missing `ji()` offset function inside `scatra()`

**File:** `scattering.py`

The `scatra()` function used raw integer indices `rm[k]` in several inner loops
instead of the offset form `rm[ji(k)]`. Because the Python array has length
`2*mu+1`, using raw signed indices either accessed the wrong element or raised an
`IndexError`.

**Fix:** The offset function `ji` was added to `scatra()` and every loop corrected.

---

### Bug 6 — Transmitted field returned instead of reflected field (satellite case)

**File:** `os_module.py`

This was the most consequential bug. The successive-orders-of-scattering solver
accumulates two fields simultaneously:

- `i3[ji(k)]` for `k > 0`: upward radiation at *ground level* (transmitted direct
  beam plus diffuse upwelling) — the *transmittance*.
- `i3[ji(k)]` for `k < 0`: downward radiation at *TOA* — the atmospheric
  *reflectance* seen by a satellite sensor.

For the satellite observation geometry (`palt > 900 km`), the original translation
returned `i3[ji(mu)]` (the ground-level upwelling, i.e. transmittance) normalised
by `xmus`. This produced values of 0.9–1.1 for the atmospheric reflectance, whereas
the correct value is 0.02–0.15.

The correct quantity is the *accumulated downwelling field at TOA*: `i3[ji(-mu)]`,
which is zero after the primary order and grows with each scattering order as light
is scattered back toward the sensor.

**Fix:**
```python
# Before (wrong — returns transmittance, not reflectance)
xl[ji(-mu), 0] = roavion          # roavion = i1[0, ji(mu)]

# After (correct — returns TOA atmospheric reflectance)
if palt > 900.0:
    xl[ji(-mu), 0] = i3[ji(-mu)]  # accumulated TOA downward field
else:
    xl[ji(-mu), 0] = roavion       # plane/ground: upwelling at sensor level
```

---

### Bug 7 — Numerical overflow at grazing illumination angles (Kuusk model)

**File:** `aktool.py`

In the `biz()` function (single-scattering BRDF term), the hot-spot scale factor
`gma = alp2 / (sl * sqrt(ctt1))` diverges when `ctt1 = cos(θ_sun) · cos(θ_view)`
approaches zero (grazing angles). With default parameters this made `exp(-ulg - gma)`
underflow to zero correctly, but the subsequent `xx2` term computed
`(1 - 0) / (very_small_denominator)` and overflowed to `~10²⁸`.

The hemispherical albedo integration (`akalbe`) evaluates `biz` at solar zenith
angles up to 81°, reliably hitting this condition.

**Fix:** Added `max(..., 1e-30)` guards on all denominator terms and clamped the
final `bi` result to `[0, 1]`:
```python
if gg1h < 1e-30 or gg1 + bam < 1e-30:
    xx2 = 0.0
else:
    xx2 = (1 - easte2) / gg1h - (1 - easte4) / (gg1 + bam)
_s.bi = max(0.0, min(1.0, bc1d + bcsp + bc1hs + bs1))
```

---

### Bug 8 — `exp(+m)` overflow in the SAIL diffuse-flux solver

**File:** `aktool.py` (`difr92` subroutine)

The SAIL two-stream equations compute a matrix element
`m22 = (1 - rrsoil·h2) · exp(+m)` where `m = sqrt(att² - sig²)`. For
optically thick canopies `m` can exceed 87, causing a Python `OverflowError`
since `math.exp` does not silently clamp like Fortran's runtime.

**Fix:** Changed `math.exp(-min(-m, 87.0))` (which computed `exp(+m)` unclamped
when `m > 87`) to `math.exp(min(m, 87.0))`, and added a matrix-singularity guard:
```python
if abs(det) < 1e-20:
    _s.bd = _s.rrsoil * math.exp(-min(ko, 87.0))
    return
```

---

### Bug 9 — Infinite bisection loop in `discre()` for large aerosol ODs

**File:** `scattering.py`

The `discre()` function finds the altitude at which the cumulative optical depth
equals a target value, using binary bisection. The inner loop exits only when
`abs(ti - x2) < 0.00001`. For Mie size-distribution aerosols (`iaer` 8–11) the
integrated extinction can be very large, making `x2` decrease far more slowly than
for the tabulated models. The bisection converges in arithmetic, but floating-point
precision means `xd` can stabilise above the threshold — the loop runs forever.

Diagnosis was performed by registering a `SIGALRM` handler that printed a Python
stack trace when a 15-second deadline elapsed, which immediately identified the
hanging line.

**Fix:** Added a 100-iteration ceiling on the inner bisection loop:
```python
_niter = 0
while True:
    _niter += 1
    y2 = (y1 + y3) * 0.5
    ...
    if xd < 0.00001 or _niter >= 100:
        break
```

---

### Bug 10 — `mie_in` common block not wired into `sixs_main.py`

**Files:** `commons.py`, `sixs_main.py`, `aeroso.py`

The Mie solver (`mie.py`) was translated correctly but the `/mie_in/` Fortran common
block — which carries the size distribution parameters read from user input — existed
nowhere in the Python package. Calls to `aeroso()` with `iaer` 8–11 silently fell
through to a stub that used pre-loaded dust data instead of the user-specified size
distribution.

Three coordinated changes were required:

**`commons.py`** — Added the `mie_in` namespace mirroring the Fortran common block:
```python
class _MieIn:
    rmax, rmin      # radius integration limits (µm)
    icp             # number of components (1–4)
    rn[10,4]        # real refractive index at 10 wavelengths, 4 components
    ri[10,4]        # imaginary refractive index
    x1[4],x2[4],x3[4]  # distribution shape parameters
    cij[4]          # component volume fractions
    irsunph         # number of sun-photometer data points
    rsunph[50]      # radii (µm)
    nrsunph[50]     # number densities (converted from dV/d(log r))
mie_in = _MieIn()
```

**`sixs_main.py`** — Added format-specific input-parsing blocks triggered when
`8 ≤ iaer ≤ 11`. Each block reads the parameters defined in the Fortran `main.f`
for that distribution type:

| `iaer` | Distribution | Input format |
|--------|-------------|--------------|
| 8 | Log-normal (up to 4 modes) | `rmin rmax icp` / per mode: `x1 x2 cij`, `rn(10)`, `ri(10)` |
| 9 | Modified gamma | `rmin rmax` / `x1 x2 x3` / `rn(10)` / `ri(10)` |
| 10 | Junge power-law | `rmin rmax` / `x1` / `rn(10)` / `ri(10)` |
| 11 | Sun photometer | `irsunph` / per point: `r dV/d(log r)` / `rn(10)` / `ri(10)` |

For `iaer=11` the input dV/d(log r) values are converted to dn/dr in-line,
reproducing the transform from the original Fortran:
`nrsunph = nv / r⁴ / log(10)`.

**`aeroso.py`** — Replaced the old stub with a proper dispatch that calls
`mie.mie()` with all parameters drawn from `mie_in`, then stores the resulting
phase function, extinction, scattering and asymmetry arrays into the shared
commons so that `discom()` picks them up correctly. The scattering coefficient
array `sc[0,:]` is also assigned (not just the asymmetry and extinction) so that
the downstream mixing loop, which weights the phase function by scattering
coefficient, receives the correct values.

---

## Validation

After all fixes, a suite of 17 test configurations was run covering every aerosol
model (`iaer` 0–11), all four Lambertian surface types, and all BRDF models
(`ibrdf` 0–9). All 17 produced physically plausible results:

| Quantity | Expected range | Observed |
|----------|----------------|----------|
| Rayleigh reflectance at 0.55 µm, SZA 30° | 0.02–0.05 | 0.023 ✓ |
| Continental aerosol reflectance (AOT 0.23) | 0.02–0.05 | 0.022 ✓ |
| Total downward transmittance | 0.85–0.95 | 0.913 ✓ |
| Ground direct irradiance (SZA 30°, AVHRR 1) | 90–130 W m⁻² µm⁻¹ | 105 ✓ |
| Log-normal Mie (iaer=8), sand, AOT 0.23 | 0.10–0.30 | 0.185 ✓ |
| Modified-gamma Mie (iaer=9) | 0.05–0.30 | 0.153 ✓ |
| Junge Mie (iaer=10, ν=3.5) | 0.10–0.30 | 0.179 ✓ |
| Sun-photometer Mie (iaer=11) | 0.10–0.30 | 0.182 ✓ |
| Ocean BRDF (ibrdf=6, Cox-Munk + Morel) | 0.01–0.20 | 0.052 ✓ |
| Kuusk canopy albedo red / NIR | 0.03–0.08 / 0.30–0.55 | 0.060 / 0.356 ✓ |

---

## Remaining Known Limitations

- **Performance:** The Python implementation runs 5–20× slower than the compiled
  Fortran for a single call (7–22 seconds vs. < 1 second). The bottleneck is the
  SOS iteration in pure Python loops. This could be addressed with NumPy
  vectorisation or Numba JIT compilation.

- **Fourier azimuth harmonics:** The SOS solver computes only the `is = 0`
  (azimuthally isotropic) Fourier component. For nadir viewing this is exact
  because higher harmonics vanish (`bp = 0` for `is > 0` when `xmuv = 1`). For
  off-nadir views the missing harmonics introduce a small error in the azimuthal
  distribution of path radiance, though the hemispherically-integrated quantities
  (transmittances, spherical albedo) are unaffected.
