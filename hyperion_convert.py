"""
hyperion_convert.py
-------------------
Convert a USGS EO-1 Hyperion L1G/T archive (ZIP or folder of per-band
GeoTIFF files) to a single ENVI hyperspectral cube.

Equivalent to the basic import functionality of Hyperion Tools 2.0
(White 2016), without georeferencing interpolation.

Usage
-----
  GUI    : python hyperion_convert.py
  Script : python hyperion_convert.py  input.zip  output_base  [bsq|bil|bip]

Processing
----------
1. Unzip the archive (or read a folder) to a temporary directory.
2. Find the MTL metadata file (filename contains "MTL").
3. Read all per-band GeoTIFF files (B001..B242). Warn if not exactly 242.
4. Concatenate into a single ENVI file in the chosen interleave.
5. Build a valid-pixel mask: pixels where ALL bands are zero are fill/nodata.
6. Write:
     <output_base>.hdr / .img          — radiance cube (int16, DN)
     <output_base>_mask.hdr / .img     — mask (1=valid, 0=fill, int16, BSQ)
     <output_dir>/<scene_id>_MTL.L1T   — copy of the MTL file

Radiance scaling (applied OUTSIDE this tool, e.g. in atmospheric correction):
  VNIR (bands 1-70):   L [W/m2/sr/um] = DN / 40
  SWIR (bands 71-242): L [W/m2/sr/um] = DN / 80

Requirements: pip install spectral rasterio numpy
"""

import os, re, shutil, zipfile, tempfile, datetime, glob
import numpy as np

# ── Hyperion spectral calibration ─────────────────────────────────────────────
# Hard-coded from the EO-1 Hyperion instrument specification.
# Values sourced from a real L1T header (identical for all scenes).
# 242 bands total: bands 1-70 = VNIR detector, 71-242 = SWIR detector.
# Note: bands 71-76 of the SWIR detector overlap with VNIR in wavelength
# (this is normal — the two detectors have independent focal planes).

HYPERION_WL_NM = [
    355.59, 365.76, 375.94, 386.11, 396.29, 406.46, 416.64, 426.82, 436.99,
    447.17, 457.34, 467.52, 477.69, 487.87, 498.04, 508.22, 518.39, 528.57,
    538.74, 548.92, 559.09, 569.27, 579.45, 589.62, 599.80, 609.97, 620.15,
    630.32, 640.50, 650.67, 660.85, 671.02, 681.20, 691.37, 701.55, 711.72,
    721.90, 732.07, 742.25, 752.43, 762.60, 772.78, 782.95, 793.13, 803.30,
    813.48, 823.65, 833.83, 844.00, 854.18, 864.35, 874.53, 884.70, 894.88,
    905.05, 915.23, 925.41, 935.58, 945.76, 955.93, 966.11, 976.28, 986.46,
    996.63, 1006.81, 1016.98, 1027.16, 1037.33, 1047.51, 1057.68, 851.92, 862.01,
    872.10, 882.19, 892.28, 902.36, 912.45, 922.54, 932.64, 942.73, 952.82,
    962.91, 972.99, 983.08, 993.17, 1003.30, 1013.30, 1023.40, 1033.49, 1043.59,
    1053.69, 1063.79, 1073.89, 1083.99, 1094.09, 1104.19, 1114.19, 1124.28, 1134.38,
    1144.48, 1154.58, 1164.68, 1174.77, 1184.87, 1194.97, 1205.07, 1215.17, 1225.17,
    1235.27, 1245.36, 1255.46, 1265.56, 1275.66, 1285.76, 1295.86, 1305.96, 1316.05,
    1326.05, 1336.15, 1346.25, 1356.35, 1366.45, 1376.55, 1386.65, 1396.74, 1406.84,
    1416.94, 1426.94, 1437.04, 1447.14, 1457.23, 1467.33, 1477.43, 1487.53, 1497.63,
    1507.73, 1517.83, 1527.92, 1537.92, 1548.02, 1558.12, 1568.22, 1578.32, 1588.42,
    1598.51, 1608.61, 1618.71, 1628.81, 1638.81, 1648.90, 1659.00, 1669.10, 1679.20,
    1689.30, 1699.40, 1709.50, 1719.60, 1729.70, 1739.70, 1749.79, 1759.89, 1769.99,
    1780.09, 1790.19, 1800.29, 1810.38, 1820.48, 1830.58, 1840.58, 1850.68, 1860.78,
    1870.87, 1880.98, 1891.07, 1901.17, 1911.27, 1921.37, 1931.47, 1941.57, 1951.57,
    1961.66, 1971.76, 1981.86, 1991.96, 2002.06, 2012.15, 2022.25, 2032.35, 2042.45,
    2052.45, 2062.55, 2072.65, 2082.75, 2092.84, 2102.94, 2113.04, 2123.14, 2133.24,
    2143.34, 2153.34, 2163.43, 2173.53, 2183.63, 2193.73, 2203.83, 2213.93, 2224.03,
    2234.12, 2244.22, 2254.22, 2264.32, 2274.42, 2284.52, 2294.61, 2304.71, 2314.81,
    2324.91, 2335.01, 2345.11, 2355.21, 2365.20, 2375.30, 2385.40, 2395.50, 2405.60,
    2415.70, 2425.80, 2435.89, 2445.99, 2456.09, 2466.09, 2476.19, 2486.29, 2496.39,
    2506.48, 2516.59, 2526.68, 2536.78, 2546.88, 2556.98, 2566.98, 2577.08,
]

HYPERION_FWHM_NM = [
    11.3871, 11.3871, 11.3871, 11.3871, 11.3871, 11.3871, 11.3871, 11.3871, 11.3871,
    11.3871, 11.3871, 11.3871, 11.3871, 11.3784, 11.3538, 11.3133, 11.2580, 11.1907,
    11.1119, 11.0245, 10.9321, 10.8368, 10.7407, 10.6482, 10.5607, 10.4823, 10.4147,
    10.3595, 10.3188, 10.2942, 10.2856, 10.2980, 10.3349, 10.3909, 10.4592, 10.5322,
    10.6004, 10.6562, 10.6933, 10.7058, 10.7276, 10.7907, 10.8833, 10.9938, 11.1044,
    11.1980, 11.2600, 11.2824, 11.2822, 11.2816, 11.2809, 11.2797, 11.2782, 11.2771,
    11.2765, 11.2756, 11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.2754,
    11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.2754, 11.0457, 11.0457,
    11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457,
    11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0457, 11.0451, 11.0423, 11.0372,
    11.0302, 11.0218, 11.0122, 11.0013, 10.9871, 10.9732, 10.9572, 10.9418, 10.9248,
    10.9065, 10.8884, 10.8696, 10.8513, 10.8335, 10.8154, 10.7979, 10.7822, 10.7663,
    10.7520, 10.7385, 10.7270, 10.7174, 10.7091, 10.7022, 10.6970, 10.6946, 10.6937,
    10.6949, 10.6996, 10.7058, 10.7163, 10.7283, 10.7437, 10.7612, 10.7807, 10.8034,
    10.8267, 10.8534, 10.8818, 10.9110, 10.9422, 10.9743, 11.0074, 11.0414, 11.0759,
    11.1108, 11.1461, 11.1811, 11.2156, 11.2496, 11.2826, 11.3146, 11.3460, 11.3753,
    11.4037, 11.4302, 11.4538, 11.4760, 11.4958, 11.5133, 11.5286, 11.5404, 11.5505,
    11.5580, 11.5621, 11.5634, 11.5617, 11.5563, 11.5477, 11.5346, 11.5193, 11.5002,
    11.4789, 11.4548, 11.4279, 11.3994, 11.3688, 11.3366, 11.3036, 11.2696, 11.2363,
    11.2007, 11.1666, 11.1333, 11.1018, 11.0714, 11.0424, 11.0155, 10.9912, 10.9698,
    10.9508, 10.9355, 10.9230, 10.9139, 10.9083, 10.9069, 10.9057, 10.9013, 10.8951,
    10.8854, 10.8740, 10.8591, 10.8429, 10.8242, 10.8039, 10.7820, 10.7592, 10.7342,
    10.7092, 10.6834, 10.6572, 10.6312, 10.6052, 10.5803, 10.5560, 10.5328, 10.5101,
    10.4904, 10.4722, 10.4552, 10.4408, 10.4285, 10.4197, 10.4129, 10.4088, 10.4077,
    10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077,
    10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077,
    10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077, 10.4077,
]

# Bad-band list: 1 = usable, 0 = unusable (uncalibrated or artefact-affected).
# Bad ranges: bands 1-7 (VNIR pre-range), 58-76 (VNIR/SWIR overlap / gap),
#             225-242 (SWIR end-of-range).
HYPERION_BBL = [
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
]

EXPECTED_BANDS = 242

# Short band names for the ENVI header, referencing original band numbers.
BAND_NAMES = [f"B{i+1} ({HYPERION_WL_NM[i]:.2f}nm)" for i in range(EXPECTED_BANDS)]

assert len(HYPERION_WL_NM)   == EXPECTED_BANDS
assert len(HYPERION_FWHM_NM) == EXPECTED_BANDS
assert len(HYPERION_BBL)     == EXPECTED_BANDS


# =============================================================================
# SECTION 1 — MTL reader
# =============================================================================

def parse_mtl(mtl_path):
    """Read a Hyperion L1T/MTL text file and return a flat key-value dict."""
    meta = {}
    with open(mtl_path, errors="replace") as f:
        for line in f:
            m = re.match(r'\s+(\w+)\s*=\s*"?([^"\n]+)"?\s*$', line)
            if m:
                meta[m.group(1)] = m.group(2).strip()
    return meta


# =============================================================================
# SECTION 2 — Conversion
# =============================================================================

def convert_hyperion(params, log=print):
    """
    Convert a Hyperion L1G/T archive to ENVI format.

    params keys
    -----------
    input_path  : str — .zip archive or folder containing band GeoTIFFs
    out_base    : str — output base path (no extension)
    interleave  : str — 'bsq', 'bil', or 'bip'
    """
    import rasterio
    import spectral.io.envi as envi

    input_path = params["input_path"]
    out_base   = params["out_base"]
    interleave = params["interleave"].lower().strip()

    # ── Unzip or use folder ───────────────────────────────────────────────────
    tmpdir = None
    if zipfile.is_zipfile(input_path):
        log(f"Extracting {os.path.basename(input_path)} ...")
        tmpdir   = tempfile.mkdtemp(prefix="hyperion_")
        work_dir = tmpdir
        with zipfile.ZipFile(input_path) as zf:
            zf.extractall(tmpdir)
    elif os.path.isdir(input_path):
        work_dir = input_path
        log(f"Input folder: {work_dir}")
    else:
        raise ValueError(f"Input must be a .zip file or folder: {input_path}")

    try:
        # ── Find MTL file ─────────────────────────────────────────────────────
        mtl_candidates = glob.glob(
            os.path.join(work_dir, "**", "*MTL*"), recursive=True)
        mtl_candidates = [p for p in mtl_candidates
                          if not p.endswith((".hdr", ".img"))]
        if not mtl_candidates:
            raise FileNotFoundError("No MTL file found in archive.")
        mtl_path = sorted(mtl_candidates)[0]
        log(f"MTL: {os.path.basename(mtl_path)}")

        mtl      = parse_mtl(mtl_path)
        acq_date = mtl.get("ACQUISITION_DATE", "unknown")
        start_t  = mtl.get("START_TIME", "").strip()

        # ── Find per-band GeoTIFFs ────────────────────────────────────────────
        tifs = sorted(
            glob.glob(os.path.join(work_dir, "**", "*.TIF"), recursive=True) +
            glob.glob(os.path.join(work_dir, "**", "*.tif"), recursive=True))

        band_pat = re.compile(r'_B(\d{3})', re.IGNORECASE)
        band_files = {}
        for tf in tifs:
            m = band_pat.search(os.path.basename(tf))
            if m:
                band_files[int(m.group(1))] = tf

        n_found = len(band_files)
        if n_found != EXPECTED_BANDS:
            log(f"WARNING: found {n_found} band files, expected {EXPECTED_BANDS}!")
        else:
            log(f"Found {n_found} band files.")

        bands_sorted = sorted(band_files.keys())

        # ── Image dimensions from first band ──────────────────────────────────
        with rasterio.open(band_files[bands_sorted[0]]) as src:
            n_lines, n_samples = src.height, src.width
            transform = src.transform
            crs       = src.crs

        log(f"Size: {n_lines} lines × {n_samples} samples × {n_found} bands")

        # ── ENVI metadata ─────────────────────────────────────────────────────
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        description = (
            f"EO-1 Hyperion L1T radiance cube. "
            f"Acquisition: {acq_date}  {start_t}. "
            f"Converted to ENVI: {now_str}. "
            f"Source: {os.path.basename(input_path)}. "
            f"Bands: {n_found}. "
            f"Radiance: DN/40=W/m2/sr/um (VNIR), DN/80=W/m2/sr/um (SWIR). "
            f"Nodata=-9999 (all-zero pixels)."
        )

        # UTM map info from rasterio geotransform and CRS.
        # ENVI map info format:
        #   {UTM, tie_x, tie_y, easting, northing, pixel_x, pixel_y,
        #    zone, hemisphere, datum, units=Meters}
        # tie_x/tie_y = pixel coordinate of the tie point (1,1 = upper-left corner).
        # easting/northing = map coordinates of that tie point (transform.c, transform.f).
        # pixel_x/pixel_y = pixel size in map units (always positive).
        map_info = None
        if transform and crs and crs.is_projected:
            # Extract UTM zone from EPSG code (326xx = UTM North, 327xx = UTM South)
            # or fall back to parsing the CRS WKT/Proj4 string.
            zone, hem = None, None
            if crs.to_epsg():
                epsg = crs.to_epsg()
                if 32601 <= epsg <= 32660:
                    zone, hem = epsg - 32600, "North"
                elif 32701 <= epsg <= 32760:
                    zone, hem = epsg - 32700, "South"
            if zone is None:
                # Fallback: search CRS string for zone number
                zone_m = re.search(r'(?:zone|utm)[_\s]*(\d+)',
                                   str(crs), re.IGNORECASE)
                if zone_m:
                    zone = int(zone_m.group(1))
                    # Hemisphere from northing: southern if False Northing = 10000000
                    hem = "South" if "10000000" in str(crs) else "North"

            if zone:
                map_info = (
                    f"{{UTM, 1.000, 1.000, "
                    f"{transform.c:.3f}, {transform.f:.3f}, "
                    f"{abs(transform.a):.10e}, {abs(transform.e):.10e}, "
                    f"{zone}, {hem}, WGS-84, units=Meters}}"
                )
            else:
                # Non-UTM projected CRS: write what we can
                map_info = (
                    f"{{Arbitrary, 1.000, 1.000, "
                    f"{transform.c:.3f}, {transform.f:.3f}, "
                    f"{abs(transform.a):.10e}, {abs(transform.e):.10e}}}"
                )

        envi_meta = {
            "description":       "{ " + description + " }",
            "samples":           str(n_samples),
            "lines":             str(n_lines),
            "bands":             str(n_found),
            "header offset":     "0",
            "file type":         "ENVI Standard",
            "data type":         "2",          # int16
            "interleave":        interleave,
            "sensor type":       "EO-1 Hyperion",
            "byte order":        "0",
            "data ignore value": "-9999",
            "wavelength units":  "Nanometers",
            "wavelength":        [f"{v:.2f}" for v in HYPERION_WL_NM[:n_found]],
            "fwhm":              [f"{v:.4f}" for v in HYPERION_FWHM_NM[:n_found]],
            "bbl":               [str(v)    for v in HYPERION_BBL[:n_found]],
            "band names":        BAND_NAMES[:n_found],
        }
        if map_info:
            envi_meta["map info"] = map_info

        # ── Safety: do not overwrite input ────────────────────────────────────
        if os.path.abspath(out_base) == os.path.abspath(
                os.path.splitext(input_path)[0]):
            raise ValueError(
                "Output path is the same as the input — choose a different name.")

        # ── Create output ENVI file ───────────────────────────────────────────
        out_hdr = out_base + ".hdr"
        log(f"\nCreating output: {out_hdr}")
        out_obj = envi.create_image(out_hdr, envi_meta,
                                     dtype=np.int16, interleave=interleave,
                                     force=True)
        out_mm = out_obj.open_memmap(writable=True)
        out_mm[:] = -9999
        out_mm.flush()
        log(f"  {out_mm.nbytes/1e6:.1f} MB allocated  ({interleave.upper()})")

        # ── Read bands and write ──────────────────────────────────────────────
        log(f"Reading and writing {n_found} bands...")
        _pct_logged = -1
        for out_b, band_num in enumerate(bands_sorted):
            with rasterio.open(band_files[band_num]) as src:
                data = src.read(1)            # (lines, samples)
            out_mm[:, :, out_b] = data.astype(np.int16)
            pct = int((out_b + 1) / n_found * 100)
            # Log a progress line at each 10% milestone
            if pct // 10 > _pct_logged // 10:
                _pct_logged = pct
                log(f"  {pct:3d}%  (band {band_num}/{bands_sorted[-1]})")
        log("  Done.")
        out_mm.flush()

        # ── Valid-pixel mask ──────────────────────────────────────────────────
        log("\nBuilding valid-pixel mask (all-zero pixels = fill)...")
        band_sum = np.zeros((n_lines, n_samples), dtype=np.int32)
        for out_b in range(n_found):
            band_sum += np.abs(out_mm[:, :, out_b].astype(np.int32))
        valid = (band_sum > 0)
        log(f"  Valid: {valid.sum()}  Fill: {(~valid).sum()}")

        # Apply nodata to fill pixels
        for out_b in range(n_found):
            tmp = out_mm[:, :, out_b].copy()
            tmp[~valid] = -9999
            out_mm[:, :, out_b] = tmp
        out_mm.flush()
        del out_mm

        # ── Mask file ─────────────────────────────────────────────────────────
        mask_hdr  = out_base + "_mask.hdr"
        mask_meta = {
            "description":   (f"{{ EO-1 Hyperion valid-pixel mask. "
                               f"1=valid 0=fill (all-zero DN). "
                               f"Source: {os.path.basename(input_path)}. "
                               f"Converted: {now_str}. }}"),
            "samples":       str(n_samples),
            "lines":         str(n_lines),
            "bands":         "1",
            "header offset": "0",
            "file type":     "ENVI Standard",
            "data type":     "2",
            "interleave":    "bsq",
            "sensor type":   "EO-1 Hyperion",
            "byte order":    "0",
        }
        if map_info:
            mask_meta["map info"] = map_info

        log(f"Writing mask: {mask_hdr}")
        mask_obj = envi.create_image(mask_hdr, mask_meta,
                                      dtype=np.int16, interleave="bsq",
                                      force=True)
        mask_mm = mask_obj.open_memmap(writable=True)
        mask_mm[:, :, 0] = valid.astype(np.int16)
        mask_mm.flush()
        del mask_mm

        # ── Copy MTL ──────────────────────────────────────────────────────────
        out_dir  = os.path.dirname(os.path.abspath(out_base))
        mtl_ext  = os.path.splitext(mtl_path)[1]
        scene_id = os.path.splitext(os.path.basename(mtl_path))[0]
        mtl_dest = os.path.join(out_dir, scene_id + "_MTL" + mtl_ext)
        shutil.copy2(mtl_path, mtl_dest)
        log(f"MTL copied: {os.path.basename(mtl_dest)}")

        log(f"\nDone.")
        log(f"  Image : {out_hdr}")
        log(f"  Mask  : {mask_hdr}")
        log(f"  MTL   : {mtl_dest}")
        return out_hdr

    finally:
        if tmpdir and os.path.isdir(tmpdir):
            shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# SECTION 3 — GUI
# =============================================================================

def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog

    root = tk.Tk()
    root.title("Hyperion L1G/T → ENVI Converter")
    root.geometry("760x560")
    root.protocol("WM_DELETE_WINDOW", lambda: (root.quit(), root.destroy()))

    e = {}

    def _set(key, value):
        w = e[key]
        if isinstance(w, ttk.Combobox):
            w.set(str(value))
        else:
            w.delete(0, tk.END)
            w.insert(0, str(value))

    def _get(key):
        return e[key].get()

    status_lbl = ttk.Label(root, text="Ready.", anchor=tk.W, relief=tk.SUNKEN)
    status_lbl.pack(side=tk.BOTTOM, fill=tk.X)

    btn_bar = ttk.Frame(root)
    btn_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    # ── Files tab ─────────────────────────────────────────────────────────────
    tab1 = ttk.Frame(nb, padding=10)
    nb.add(tab1, text="Files")

    fg = ttk.LabelFrame(tab1, text="Input / Output", padding=8)
    fg.pack(fill=tk.X, pady=(0, 10))
    fg.columnconfigure(1, weight=1)

    def browse_input():
        p = filedialog.askopenfilename(
            title="Select Hyperion ZIP archive",
            filetypes=[("ZIP", "*.zip"), ("All", "*.*")])
        if not p:
            p = filedialog.askdirectory(title="Or select folder with band GeoTIFFs")
        if p:
            _set("input_path", p)
            base = os.path.splitext(p)[0]
            _set("out_base", base + "_ENVI")
            status_lbl.config(text=f"Input: {os.path.basename(p)}")

    def browse_out():
        p = filedialog.asksaveasfilename(
            title="Output base name (no extension)")
        if p:
            _set("out_base", p)

    for r, (lbl, key, cmd) in enumerate([
        ("Input ZIP / folder:", "input_path", browse_input),
        ("Output base:",        "out_base",   browse_out),
    ]):
        ttk.Label(fg, text=lbl).grid(row=r, column=0, sticky=tk.W, pady=3)
        entry = ttk.Entry(fg)
        entry.grid(row=r, column=1, sticky=tk.EW, padx=4)
        e[key] = entry
        ttk.Button(fg, text="Browse...", command=cmd).grid(row=r, column=2)

    # Interleave selector
    ig = ttk.LabelFrame(tab1, text="Output interleave", padding=8)
    ig.pack(fill=tk.X)
    ttk.Label(ig, text="Interleave:").grid(row=0, column=0, sticky=tk.W)
    il_cb = ttk.Combobox(ig, values=["bsq","bil","bip"],
                          state="readonly", width=8)
    il_cb.set("bsq")
    il_cb.grid(row=0, column=1, sticky=tk.W, padx=6)
    e["interleave"] = il_cb
    ttk.Label(ig,
              text="BSQ = band sequential (default)  "
                   "BIL = by line  BIP = by pixel",
              foreground="gray").grid(row=0, column=2, sticky=tk.W)

    # ── Log tab ───────────────────────────────────────────────────────────────
    tab2 = ttk.Frame(nb, padding=4)
    nb.add(tab2, text="Log")
    log_text = tk.Text(tab2, font=("Courier", 9), wrap=tk.NONE)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    ttk.Scrollbar(tab2, orient=tk.VERTICAL,
                  command=log_text.yview).pack(side=tk.RIGHT, fill=tk.Y)

    # ── Run / Close ───────────────────────────────────────────────────────────
    def run_conversion():
        run_btn.config(state=tk.DISABLED)
        log_text.delete("1.0", tk.END)
        nb.select(tab2)
        status_lbl.config(text="Converting...")
        root.update()

        inp = _get("input_path")
        if not inp or not os.path.exists(inp):
            status_lbl.config(text="Error: input not found.")
            run_btn.config(state=tk.NORMAL)
            return

        params = {
            "input_path": inp,
            "out_base":   _get("out_base") or os.path.splitext(inp)[0]+"_ENVI",
            "interleave": _get("interleave") or "bsq",
        }

        def log(msg, end="\n"):
            # Always append a new line — no in-place overwriting.
            # The end= argument is accepted for compatibility with script mode
            # but treated as a newline in the GUI to avoid widget corruption.
            log_text.insert(tk.END, msg + "\n")
            log_text.see(tk.END)
            root.update()

        try:
            out_hdr = convert_hyperion(params, log=log)
            status_lbl.config(text=f"Done → {os.path.basename(out_hdr)}")
        except Exception as ex:
            import traceback
            log(traceback.format_exc())
            status_lbl.config(text=f"Error: {ex}")
        finally:
            run_btn.config(state=tk.NORMAL)

    run_btn = ttk.Button(btn_bar, text="▶  Convert", command=run_conversion)
    run_btn.pack(side=tk.LEFT, padx=(0,6))
    ttk.Button(btn_bar, text="Close",
               command=lambda: (root.quit(), root.destroy())).pack(side=tk.LEFT)

    root.mainloop()


# =============================================================================
# SECTION 4 — Entry point
# =============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        convert_hyperion({
            "input_path": sys.argv[1],
            "out_base":   sys.argv[2] if len(sys.argv)>2
                          else os.path.splitext(sys.argv[1])[0]+"_ENVI",
            "interleave": sys.argv[3] if len(sys.argv)>3 else "bsq",
        })
    else:
        run_gui()