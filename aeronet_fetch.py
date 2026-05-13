"""
aeronet_fetch.py
----------------
Retrieve AERONET aerosol, ozone, and precipitable water measurements
for a specific location and date, with temporal and spectral interpolation.

The tool automatically finds the nearest AERONET site with data.
If a site has no data for the requested date it moves to the next-closest
site, reporting each attempt, until data are found.

Usage
-----
  GUI:    python aeronet_fetch.py
  CLI:    python aeronet_fetch.py --lat 62.0 --lon 27.0 --date 2016-06-03
          python aeronet_fetch.py --lat 62.0 --lon 27.0 --date 2016-06-03 \
                                  --time 10:30 --wl 550 --force-site Kuopio

AERONET Web Service v3
  Site list:         https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt
  AOD (all points):  https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3
  Inversions (PWV):  https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3

Data levels
  Level 1.0 — unscreened
  Level 1.5 — cloud-screened (default; usually available within days)
  Level 2.0 — quality-assured (may have a delay of months)

AOD wavelengths available (nm): 340, 380, 440, 500, 675, 870, 1020, 1640
Spectral interpolation: Angstrom exponent fit over the log-log AOD spectrum.
Temporal interpolation: linear between bracketing measurements.
Precipitable water (PWV) and ozone (O3) come from the inversion product.
"""

import re
import csv
import math
import datetime
import urllib.request
import urllib.parse

# ── AERONET API endpoints ────────────────────────────────────────────────────
AOD_URL   = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3"
INV_URL   = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3"
SITES_URL   = "https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt"

# Bundled site list — loaded by default; call update_site_list() to refresh.
import os as _os
SITES_LOCAL = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            "aeronet_sites.txt")

# AERONET AOD wavelength columns (nm)
AOD_WL_NM = [1640, 1020, 870, 675, 500, 440, 380, 340]

# ── Fetch helpers ─────────────────────────────────────────────────────────────

def _fetch(url: str, params: dict = None, timeout: int = 20) -> str:
    """HTTP GET. Returns response text."""
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "aeronet_fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_csv(text: str) -> list:
    """Parse AERONET CSV: skip preamble lines, find column header, return list of dicts."""
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        # Column header starts with "Date(" in various formats, possibly preceded
        # by "AERONET_Site," — match any line containing "Date(" or "Date-" early on.
        if re.search(r"Date[(\-]", line[:40], re.I):
            header_idx = i
            break
    if header_idx is None:
        return []
    reader = csv.DictReader(lines[header_idx:])
    rows = []
    for row in reader:
        clean = {k.strip(): v.strip() for k, v in row.items() if k}
        # Accept rows that have a non-empty value in any date-keyed column
        if not any(v for k, v in clean.items() if "date" in k.lower()):
            continue
        rows.append(clean)
    return rows
def _to_decimal_hours(row: dict) -> float | None:
    time_str = row.get("Time(hh:mm:ss)") or row.get("Time(HH:MM:SS)")
    if not time_str:
        return None
    try:
        h, m, s = time_str.split(":")
        return int(h) + int(m) / 60 + int(s) / 3600
    except Exception:
        return None


def _float(row: dict, key: str) -> float | None:
    """Lookup key case-insensitively, return float or None for missing/-999."""
    # Try exact match first, then case-insensitive
    val = row.get(key)
    if val is None:
        kl = key.lower()
        val = next((v for k, v in row.items() if k.lower() == kl), None)
    if val is None:
        return None
    try:
        f = float(val)
        return None if f < -990 else f
    except (ValueError, TypeError):
        return None


def _find_key(row: dict, *candidates) -> str | None:
    """Return the first key in row that matches any candidate (case-insensitive)."""
    for c in candidates:
        cl = c.lower()
        for k in row:
            if k.lower() == cl:
                return k
    return None


def _get_pwv(row: dict) -> float | None:
    """Extract precipitable water from a row using known AERONET column name variants."""
    for candidate in ("Precipitable_Water(cm)", "Precipitable_Water",
                      "Water(cm)", "PWV(cm)", "PW(cm)"):
        v = _float(row, candidate)
        if v is not None:
            return v
    return None


def _get_ozone(row: dict) -> float | None:
    """Extract ozone from a row using known AERONET column name variants."""
    for candidate in ("Ozone(Dobson)", "Ozone(du)", "Ozone",
                      "Total_Ozone(Dobson)", "O3(Dobson)"):
        v = _float(row, candidate)
        if v is not None:
            return v
    return None


# ── Site list ─────────────────────────────────────────────────────────────────

_SITES_CACHE: list | None = None   # cached after first download

def _parse_site_text(text: str) -> list:
    """Parse the AERONET site list text into a list of {name, lat, lon, elev} dicts."""
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "SiteName" in line or "Site_Name" in line or "Site Name" in line:
            header_idx = i
            break
    if header_idx is None:
        return []
    # Strip units from column names, e.g. "Longitude(decimaldegrees)" -> "Longitude"
    import re as _re
    lines[header_idx] = _re.sub(r"\(.*?\)", "", lines[header_idx])
    reader = csv.DictReader(lines[header_idx:])
    sites = []
    for row in reader:
        try:
            name = (row.get("SiteName") or row.get("Site_Name") or
                    row.get("Site Name") or "").strip()
            lon  = float(row.get("Longitude") or 0)
            lat  = float(row.get("Latitude")  or 0)
            elev = float(row.get("Elevation") or 0)
            if name:
                sites.append({"name": name, "lat": lat, "lon": lon, "elev": elev})
        except (ValueError, TypeError):
            continue
    return sites


def fetch_site_list(local: bool = True, log=None) -> list:
    """
    Load the AERONET site list. Cached in memory after the first call.

    local=True  (default): read from aeronet_sites.txt bundled with this module.
                           Fast, no network required.
    local=False:           download the full list (~1400 sites) from AERONET.
                           Use update_site_list() to also save it to disk.

    Returns list of dicts: {name, lat, lon, elev}.
    """
    global _SITES_CACHE
    if _SITES_CACHE is not None:
        return _SITES_CACHE

    if local and _os.path.isfile(SITES_LOCAL):
        if log:
            log(f"Loading site list from {_os.path.basename(SITES_LOCAL)}…")
        with open(SITES_LOCAL, encoding="utf-8") as f:
            text = f.read()
    else:
        if log:
            log("Downloading AERONET site list…")
        text = _fetch(SITES_URL)

    sites = _parse_site_text(text)
    _SITES_CACHE = sites
    if log:
        log(f"  {len(sites)} AERONET sites loaded.")
    return sites


def update_site_list(path: str = None, log=print) -> int:
    """
    Download the full AERONET site list and save it to disk so it is used
    automatically on future calls to fetch_site_list().

    path  : destination path (default: aeronet_sites.txt next to this module)
    log   : progress callback, or None to suppress

    Returns the number of sites saved.

    Example
    -------
    import aeronet_fetch
    aeronet_fetch.update_site_list()   # download and save the full list
    """
    global _SITES_CACHE
    dest = path or SITES_LOCAL
    if log:
        log("Downloading full AERONET site list from server…")
    text  = _fetch(SITES_URL)
    sites = _parse_site_text(text)
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    _SITES_CACHE = sites
    if log:
        log(f"  {len(sites)} sites saved to {_os.path.basename(dest)}.")
    return len(sites)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def sites_by_distance(lat: float, lon: float,
                       force_site: str | None = None,
                       log=None) -> list:
    """
    Return AERONET sites sorted by distance from (lat, lon).
    If force_site is given, that site is placed first regardless of distance.
    Returns list of dicts with added 'dist_km' key.
    """
    sites = fetch_site_list(local=True, log=log)
    for s in sites:
        s["dist_km"] = haversine_km(lat, lon, s["lat"], s["lon"])
    ordered = sorted(sites, key=lambda s: s["dist_km"])

    if force_site:
        # Move the forced site to front (case-insensitive match)
        fs = force_site.strip().lower()
        forced = [s for s in ordered if s["name"].lower() == fs]
        rest   = [s for s in ordered if s["name"].lower() != fs]
        if not forced:
            if log:
                log(f"  Warning: forced site '{force_site}' not found in site list.")
        ordered = forced + rest

    return ordered


# ── Spectral interpolation ───────────────────────────────────────────────────

def angstrom_exponent(aod1: float, aod2: float,
                       wl1_nm: float, wl2_nm: float) -> float:
    if aod1 <= 0 or aod2 <= 0:
        return 1.3
    return -math.log(aod1 / aod2) / math.log(wl1_nm / wl2_nm)


def interpolate_aod(row: dict, target_wl_nm: float) -> tuple:
    """
    Interpolate/extrapolate AOD to target_wl_nm using Angstrom fit.
    Returns (aod_at_target, alpha).
    """
    pairs = []
    for wl in AOD_WL_NM:
        col = f"AOD_{wl}nm"
        aod = _float(row, col)
        if aod is not None:
            pairs.append((wl, aod))
    if not pairs:
        return None, float("nan")
    pairs.sort()

    lo = hi = None
    for wl, aod in pairs:
        if wl <= target_wl_nm:
            lo = (wl, aod)
        if wl >= target_wl_nm and hi is None:
            hi = (wl, aod)
    if lo is None: lo = pairs[0]
    if hi is None: hi = pairs[-1]
    if lo == hi:   return lo[1], 0.0

    alpha = angstrom_exponent(lo[1], hi[1], lo[0], hi[0])
    return lo[1] * (target_wl_nm / lo[0]) ** (-alpha), alpha


# ── Temporal interpolation ───────────────────────────────────────────────────

def interpolate_time(rows: list, target_hour: float, key_fn) -> float | None:
    timed = [(t, v) for r in rows
             for t, v in [(_to_decimal_hours(r), key_fn(r))]
             if t is not None and v is not None]
    if not timed:
        return None
    timed.sort()
    lo = hi = None
    for t, v in timed:
        if t <= target_hour: lo = (t, v)
        if t >= target_hour and hi is None: hi = (t, v)
    if lo is None: return hi[1] if hi else None
    if hi is None: return lo[1]
    if lo[0] == hi[0]: return lo[1]
    frac = (target_hour - lo[0]) / (hi[0] - lo[0])
    return lo[1] + frac * (hi[1] - lo[1])


# ── Per-site retrieval ────────────────────────────────────────────────────────

def dump_raw(site: str, date, level: str = "15", product: str = "aod") -> str:
    """
    Fetch and return the raw unparsed text from the AERONET server for
    one site and date. Useful for debugging column names and data format.

    product: 'aod' (default) or 'pwv'

    Example
    -------
    print(aeronet_fetch.dump_raw("Kuopio", "2016-06-03"))
    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    level_aod = {"10": "AOD10", "15": "AOD15", "20": "AOD20"}.get(level, "AOD15")
    level_inv = {"10": "ALM10", "15": "ALM15", "20": "ALM20"}.get(level, "ALM15")

    if product == "pwv":
        params = {
            "site": site,
            "year": date.year, "month": date.month, "day": date.day,
            "year2": date.year, "month2": date.month, "day2": date.day,
            "product": "PWV", level_inv: 1, "AVG": 10, "if_no_html": 1,
        }
        return _fetch(INV_URL, params)
    else:
        params = {
            "site": site,
            "year": date.year, "month": date.month, "day": date.day,
            "year2": date.year, "month2": date.month, "day2": date.day,
            level_aod: 1, "AVG": 10, "if_no_html": 1,
        }
        return _fetch(AOD_URL, params)


    level_key = {"10": "AOD10", "15": "AOD15", "20": "AOD20"}.get(level, "AOD15")
    params = {
        "site": site,
        "year": date.year, "month": date.month, "day": date.day,
        "year2": date.year, "month2": date.month, "day2": date.day,
        level_key: 1, "AVG": 10, "if_no_html": 1,
    }
    return _parse_csv(_fetch(AOD_URL, params))


def fetch_aod(site: str, date: datetime.date, level: str = "15",
              timeout: int = 15) -> list:
    """Retrieve all AOD measurement rows for one site on one day."""
    level_key = {"10": "AOD10", "15": "AOD15", "20": "AOD20"}.get(level, "AOD15")
    params = {
        "site": site,
        "year": date.year, "month": date.month, "day": date.day,
        "year2": date.year, "month2": date.month, "day2": date.day,
        level_key: 1, "AVG": 10, "if_no_html": 1,
    }
    return _parse_csv(_fetch(AOD_URL, params, timeout=timeout))


def fetch_pwv_ozone(site: str, date: datetime.date, level: str = "15",
                    timeout: int = 15) -> list:
    level_key = {"10": "ALM10", "15": "ALM15", "20": "ALM20"}.get(level, "ALM15")
    params = {
        "site": site,
        "year": date.year, "month": date.month, "day": date.day,
        "year2": date.year, "month2": date.month, "day2": date.day,
        "product": "PWV", level_key: 1, "AVG": 10, "if_no_html": 1,
    }
    return _parse_csv(_fetch(INV_URL, params, timeout=timeout))


def retrieve_from_site(site_info: dict,
                        date: datetime.date,
                        time_utc: datetime.time | None,
                        target_wl_nm: float,
                        level: str,
                        timeout: int = 15) -> dict:
    """
    Attempt retrieval from one specific site.
    Returns a result dict; n_aod_obs == 0 means no data.
    """
    name = site_info["name"]
    aod_rows = fetch_aod(name, date, level, timeout=timeout)
    pwv_rows = fetch_pwv_ozone(name, date, level, timeout=timeout)
    target_hour = (time_utc.hour + time_utc.minute/60 + time_utc.second/3600
                   if time_utc else None)

    res = dict(
        site=name, site_lat=site_info.get("lat"), site_lon=site_info.get("lon"),
        site_dist_km=site_info.get("dist_km", 0.0),
        date=date, time_utc=time_utc, level=level,
        target_wl_nm=target_wl_nm,
        aod_target=None, alpha=float("nan"),
        aod_440=None, aod_500=None, aod_675=None, aod_870=None,
        pwv_cm=None, ozone_du=None,
        n_aod_obs=len(aod_rows),
        aod_rows=aod_rows, pwv_rows=pwv_rows,
        search_log=[],
    )
    if not aod_rows:
        return res

    def mean_of(rows, key_fn):
        vals = [v for r in rows for v in [key_fn(r)] if v is not None]
        return sum(vals)/len(vals) if vals else None

    if target_hour is not None:
        fake_row = {}
        for wl in AOD_WL_NM:
            col = f"AOD_{wl}nm"
            v = interpolate_time(aod_rows, target_hour, lambda r, c=col: _float(r, c))
            fake_row[col] = str(v) if v is not None else "-999"
        aod_t, alpha = interpolate_aod(fake_row, target_wl_nm)
        res.update(aod_target=aod_t, alpha=alpha,
                   aod_440=_float(fake_row, "AOD_440nm"),
                   aod_500=_float(fake_row, "AOD_500nm"),
                   aod_675=_float(fake_row, "AOD_675nm"),
                   aod_870=_float(fake_row, "AOD_870nm"))
        # PWV and ozone: prefer AOD rows (AERONET includes them in the AOD file),
        # fall back to inversion (pwv_rows) if not present in AOD rows.
        pwv_from_aod = interpolate_time(aod_rows, target_hour, _get_pwv)
        o3_from_aod  = interpolate_time(aod_rows, target_hour, _get_ozone)
        if pwv_from_aod is not None:
            res["pwv_cm"]   = pwv_from_aod
            res["ozone_du"] = o3_from_aod
        elif pwv_rows:
            res["pwv_cm"]   = interpolate_time(pwv_rows, target_hour, _get_pwv)
            res["ozone_du"] = interpolate_time(pwv_rows, target_hour, _get_ozone)
    else:
        fake_row = {}
        for wl in AOD_WL_NM:
            col = f"AOD_{wl}nm"
            m = mean_of(aod_rows, lambda r, c=col: _float(r, c))
            fake_row[col] = str(m) if m is not None else "-999"
        aod_t, alpha = interpolate_aod(fake_row, target_wl_nm)
        res.update(aod_target=aod_t, alpha=alpha,
                   aod_440=_float(fake_row, "AOD_440nm"),
                   aod_500=_float(fake_row, "AOD_500nm"),
                   aod_675=_float(fake_row, "AOD_675nm"),
                   aod_870=_float(fake_row, "AOD_870nm"))
        pwv_from_aod = mean_of(aod_rows, _get_pwv)
        o3_from_aod  = mean_of(aod_rows, _get_ozone)
        if pwv_from_aod is not None:
            res["pwv_cm"]   = pwv_from_aod
            res["ozone_du"] = o3_from_aod
        elif pwv_rows:
            res["pwv_cm"]   = mean_of(pwv_rows, _get_pwv)
            res["ozone_du"] = mean_of(pwv_rows, _get_ozone)
    return res


# ── Main search: nearest site with data ──────────────────────────────────────

def retrieve(lat: float, lon: float,
             date: datetime.date,
             time_utc: datetime.time | None = None,
             target_wl_nm: float = 550.0,
             level: str = "15",
             force_site: str | None = None,
             log=None,
             abort_flag: list | None = None,
             max_retries: int = 3,
             timeout: int = 15) -> dict:
    """
    Find the nearest AERONET site with data for the given date.

    abort_flag : a one-element list [False]; set abort_flag[0]=True from
                 another thread to stop the search cleanly.
    max_retries: number of times to retry the same site on timeout before
                 moving to the next site (default 3).
    timeout    : per-request timeout in seconds (default 15).

    Logic:
      - Timeout / network error → retry the same site (up to max_retries).
      - Empty response (no data) → move to next site.
      - abort_flag[0] set       → stop immediately and return empty result.
    """
    import urllib.error

    if log is None:
        log = lambda _: None
    if abort_flag is None:
        abort_flag = [False]

    ordered = sites_by_distance(lat, lon, force_site=force_site, log=log)
    search_log = []

    for site_info in ordered:
        if abort_flag[0]:
            msg = "Search aborted by user."
            log(msg); search_log.append(msg)
            break

        name = site_info["name"]
        dist = site_info.get("dist_km", 0.0)

        res = None
        for attempt in range(1, max_retries + 1):
            if abort_flag[0]:
                break

            if attempt == 1:
                msg = f"Trying {name} ({dist:.0f} km)…"
            else:
                msg = f"  {name}: retry {attempt}/{max_retries}…"
            log(msg); search_log.append(msg)

            try:
                res = retrieve_from_site(site_info, date, time_utc,
                                         target_wl_nm, level,
                                         timeout=timeout)
                break   # got a response (may be empty) — don't retry

            except (TimeoutError, urllib.error.URLError) as ex:
                msg = f"  {name}: timeout ({ex}) — {'retrying' if attempt < max_retries else 'giving up'}"
                log(msg); search_log.append(msg)
                if attempt == max_retries:
                    res = None   # exhausted retries → treat as no data, try next site
                # loop continues for retry

            except Exception as ex:
                msg = f"  {name}: error — {ex}"
                log(msg); search_log.append(msg)
                res = None
                break   # non-timeout error → skip to next site immediately

        if abort_flag[0]:
            break

        if res is None or res["n_aod_obs"] == 0:
            if res is not None:
                msg = f"  {name}: no data for {date}."
                log(msg); search_log.append(msg)
            continue

        msg = f"  {name}: {res['n_aod_obs']} observations found."
        log(msg); search_log.append(msg)
        res["search_log"] = search_log
        return res

    # No site found or aborted
    return dict(
        site=None, date=date, time_utc=time_utc, level=level,
        target_wl_nm=target_wl_nm,
        aod_target=None, alpha=float("nan"),
        aod_440=None, aod_500=None, aod_675=None, aod_870=None,
        pwv_cm=None, ozone_du=None, n_aod_obs=0,
        aod_rows=[], pwv_rows=[], search_log=search_log,
    )


def fetch_by_site(site: str,
                  date,
                  time_utc=None,
                  target_wl_nm: float = 550.0,
                  level: str = "15",
                  log=print) -> dict:
    """
    Convenience wrapper: fetch by site name rather than lat/lon.

    Tries the named site first. If it has no data, looks up its coordinates
    from the site list and falls back to the nearest-site search loop,
    reporting each attempt — same behaviour as retrieve() with lat/lon.

    Examples
    --------
    from aeronet_fetch import fetch_by_site, format_result
    res = fetch_by_site("Kuopio", "2016-06-03")
    print(format_result(res))
    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    if isinstance(time_utc, str):
        time_utc = datetime.time.fromisoformat(
            time_utc if ":" in time_utc else time_utc + ":00")

    # Use retrieve() with force_site — it already implements the full fallback loop
    # We need coordinates to sort the fallback list. Look up the named site.
    sites = fetch_site_list(local=True, log=None)
    match = next((s for s in sites if s["name"].lower() == site.lower()), None)
    if match:
        lat, lon = match["lat"], match["lon"]
    else:
        # Not in local list — try downloading the full list
        if log:
            log(f"  '{site}' not in local site list; downloading full list…")
        sites = fetch_site_list(local=False, log=log)
        match = next((s for s in sites if s["name"].lower() == site.lower()), None)
        if match:
            lat, lon = match["lat"], match["lon"]
        else:
            # Still not found — use 0,0 as fallback; force_site will still try it
            if log:
                log(f"  Warning: '{site}' not found in AERONET site list.")
            lat, lon = 0.0, 0.0

    return retrieve(lat, lon, date, time_utc, target_wl_nm, level,
                    force_site=site, log=log)



def format_result(res: dict) -> str:
    def fmt(v, spec): return format(v, spec) if v is not None else "N/A"
    nan = float("nan")
    alpha = res.get("alpha", nan)

    lines = ["AERONET retrieval result",
             f"  Site        : {res['site'] or '(none found)'}"]
    if res.get("site_dist_km") is not None:
        lines.append(f"  Distance    : {res['site_dist_km']:.1f} km from query point")
    if res.get("site_lat") is not None:
        lines.append(f"  Site coords : {res['site_lat']:.4f}N  {res['site_lon']:.4f}E")
    lines += [
        f"  Date        : {res['date']}",
        f"  Time (UTC)  : {res['time_utc'] or 'daily mean'}",
        f"  Data level  : {res['level']}",
        f"  Observations: {res['n_aod_obs']} AOD measurements",
        "",
        f"  Target wl   : {res['target_wl_nm']:.1f} nm",
        f"  AOD({res['target_wl_nm']:.0f}nm)  : {fmt(res['aod_target'], '.4f')}",
        f"  Angstrom a  : {fmt(alpha if not math.isnan(alpha) else None, '.3f')}",
        "",
        f"  AOD @ 440nm : {fmt(res['aod_440'], '.4f')}",
        f"  AOD @ 500nm : {fmt(res['aod_500'], '.4f')}",
        f"  AOD @ 675nm : {fmt(res['aod_675'], '.4f')}",
        f"  AOD @ 870nm : {fmt(res['aod_870'], '.4f')}",
        "",
        f"  PWV (cm)    : {fmt(res['pwv_cm'], '.3f')}",
        f"  Ozone (DU)  : {fmt(res['ozone_du'], '.1f')}",
    ]
    if res.get("search_log"):
        lines += ["", "Search log:"]
        lines += [f"  {l}" for l in res["search_log"]]
    return "\n".join(lines)


# ── GUI ───────────────────────────────────────────────────────────────────────

def run_gui():
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("AERONET Retrieval Tool")
    root.resizable(True, True)
    def _quit():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _quit)  # clean exit on close

    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    # ── Tab 1: Query ──────────────────────────────────────────────────────────
    tab1 = ttk.Frame(nb, padding=10)
    nb.add(tab1, text="Query")

    fields = {}

    def _label_entry(parent, row, label, key, default, width=22, hint=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
        e = ttk.Entry(parent, width=width)
        e.insert(0, default)
        e.grid(row=row, column=1, sticky=tk.EW, padx=(6, 0))
        fields[key] = e
        if hint:
            ttk.Label(parent, text=hint, foreground="gray").grid(
                row=row, column=2, sticky=tk.W, padx=6)
        return e

    # Site name — if given, used directly (overrides lat/lon for the first attempt)
    _label_entry(tab1, 0, "Site name (optional):", "force_site", "",
                 hint="exact AERONET site name; overrides coordinates; falls back to nearest if no data")

    # Location — used for nearest-site search when site name is blank or has no data
    _label_entry(tab1, 1, "Latitude (deg N):", "lat", "62.0",
                 hint="used for nearest-site search when site name is blank or has no data")
    _label_entry(tab1, 2, "Longitude (deg E):", "lon", "27.0")

    # Date
    _label_entry(tab1, 3, "Date (YYYY-MM-DD):", "date", "2016-06-03")

    # Time UTC
    _label_entry(tab1, 4, "Time UTC (HH:MM, optional):", "time", "",
                 hint="leave blank for daily mean")

    # Target wavelength
    _label_entry(tab1, 5, "Target wavelength (nm):", "wl", "550",
                 hint="AOD interpolated here via Angstrom exponent")

    # Level
    ttk.Label(tab1, text="Data quality level:").grid(row=6, column=0, sticky=tk.W, pady=3)
    level_var = tk.StringVar(value="15")
    lf = ttk.Frame(tab1)
    lf.grid(row=6, column=1, sticky=tk.W, padx=(6, 0))
    for lv, txt in [("10", "1.0 unscreened"), ("15", "1.5 cloud-screened"),
                    ("20", "2.0 quality-assured")]:
        ttk.Radiobutton(lf, text=txt, variable=level_var, value=lv).pack(
            side=tk.LEFT, padx=(0, 8))
    fields["level"] = level_var

    tab1.columnconfigure(1, weight=1)

    # Status + button
    btn_bar = ttk.Frame(tab1)
    btn_bar.grid(row=7, column=0, columnspan=3, sticky=tk.EW, pady=(12, 0))
    status_var = tk.StringVar(value="Ready.")
    ttk.Label(btn_bar, textvariable=status_var, foreground="gray").pack(side=tk.LEFT)

    def _run():
        import threading

        def _worker():
            try:
                force_site = fields["force_site"].get().strip() or None
                lat_str    = fields["lat"].get().strip()
                lon_str    = fields["lon"].get().strip()
                date_str   = fields["date"].get().strip()
                time_str   = fields["time"].get().strip()
                wl         = float(fields["wl"].get().strip())
                level      = level_var.get()

                date     = datetime.date.fromisoformat(date_str)
                time_utc = (datetime.time.fromisoformat(time_str + ":00")
                            if time_str else None)

                def _log(msg):
                    root.after(0, lambda m=msg: status_var.set(m))

                # Site name given → use fetch_by_site (no lat/lon required)
                if force_site and not lat_str and not lon_str:
                    res = fetch_by_site(force_site, date, time_utc, wl, level, log=_log)
                else:
                    lat = float(lat_str) if lat_str else 0.0
                    lon = float(lon_str) if lon_str else 0.0
                    res = retrieve(lat, lon, date, time_utc, wl, level,
                                   force_site=force_site, log=_log)
                root.after(0, lambda: _show(res))

            except Exception as ex:
                import traceback
                root.after(0, lambda: (
                    status_var.set(f"Error: {ex}"),
                    messagebox.showerror("Error", traceback.format_exc())
                ))

        threading.Thread(target=_worker, daemon=True).start()

    ttk.Button(btn_bar, text="Fetch", command=_run).pack(side=tk.RIGHT)
    ttk.Button(btn_bar, text="Close", command=_quit).pack(side=tk.RIGHT, padx=(0, 4))

    # ── Tab 2: Results ────────────────────────────────────────────────────────
    tab2 = ttk.Frame(nb, padding=4)
    nb.add(tab2, text="Results")
    res_text = tk.Text(tab2, font=("Courier", 10), wrap=tk.NONE, state=tk.DISABLED)
    res_text.pack(fill=tk.BOTH, expand=True)

    # ── Tab 3: All measurements ───────────────────────────────────────────────
    tab3 = ttk.Frame(nb, padding=4)
    nb.add(tab3, text="All measurements")
    cols = ("Time (UTC)", "AOD 440nm", "AOD 500nm", "AOD 675nm",
            "AOD 870nm", "AOD 1020nm", "Angstrom a (440-870)",
            "PWV (cm)", "Ozone (Du)")
    tree = ttk.Treeview(tab3, columns=cols, show="headings", height=20)
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=115, anchor=tk.CENTER)
    vsb = ttk.Scrollbar(tab3, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    vsb.pack(side=tk.RIGHT, fill=tk.Y)

    # Save-to-CSV button bar below the treeview
    tab3_btn = ttk.Frame(tab3)
    tab3_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
    save_status = tk.StringVar(value="")
    ttk.Label(tab3_btn, textvariable=save_status,
              foreground="gray").pack(side=tk.LEFT)

    def _save_csv():
        from tkinter import filedialog
        if not tree.get_children():
            save_status.set("No data to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save measurements as CSV",
        )
        if not path:
            return
        import csv as _csv
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = _csv.writer(f)
                w.writerow(cols)
                for iid in tree.get_children():
                    w.writerow(tree.item(iid, "values"))
            save_status.set(f"Saved: {path}")
        except Exception as ex:
            save_status.set(f"Error: {ex}")

    ttk.Button(tab3_btn, text="Save as CSV…",
               command=_save_csv).pack(side=tk.RIGHT)

    def _show(res: dict):
        site_str = res["site"] or "(none found)"
        status_var.set(
            f"Done. {res['n_aod_obs']} AOD obs — {site_str} — {res['date']}")

        res_text.config(state=tk.NORMAL)
        res_text.delete("1.0", tk.END)
        res_text.insert(tk.END, format_result(res))
        res_text.config(state=tk.DISABLED)
        nb.select(tab2)

        for row in tree.get_children():
            tree.delete(row)
        for r in res["aod_rows"]:
            t     = r.get("Time(hh:mm:ss)", r.get("Time(HH:MM:SS)", ""))
            a440  = _float(r, "AOD_440nm")
            a500  = _float(r, "AOD_500nm")
            a675  = _float(r, "AOD_675nm")
            a870  = _float(r, "AOD_870nm")
            a1020 = _float(r, "AOD_1020nm")
            # Prefer the Angstrom exponent pre-computed by AERONET (440-870nm)
            ae_file = _float(r, "440-870_Angstrom_Exponent")
            if ae_file is not None:
                alpha_str = f"{ae_file:.3f}"
            elif a440 and a870:
                alpha_str = f"{angstrom_exponent(a440, a870, 440, 870):.3f}"
            else:
                alpha_str = "N/A"
            pwv  = _get_pwv(r)
            o3   = _get_ozone(r)
            tree.insert("", tk.END, values=(
                t,
                f"{a440:.4f}"  if a440  else "N/A",
                f"{a500:.4f}"  if a500  else "N/A",
                f"{a675:.4f}"  if a675  else "N/A",
                f"{a870:.4f}"  if a870  else "N/A",
                f"{a1020:.4f}" if a1020 else "N/A",
                alpha_str,
                f"{pwv:.3f}"   if pwv   else "N/A",
                f"{o3:.1f}"    if o3    else "N/A",
            ))

    root.mainloop()


# ── CLI ───────────────────────────────────────────────────────────────────────

def run_cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Fetch AERONET data for a location, finding the nearest site with data.")
    p.add_argument("--site",        default=None,
                   help="Site name shorthand (sets --force-site and uses site coords)")
    p.add_argument("--lat",         type=float, default=None,  help="Latitude (deg N)")
    p.add_argument("--lon",         type=float, default=None,  help="Longitude (deg E)")
    p.add_argument("--date",        required=True,              help="Date YYYY-MM-DD")
    p.add_argument("--time",        default=None,               help="UTC time HH:MM (optional)")
    p.add_argument("--wl",          type=float, default=550.0,  help="Target wavelength nm")
    p.add_argument("--level",       default="15",
                   choices=["10","15","20"],                    help="Data quality level")
    p.add_argument("--force-site",  default=None,
                   help="Try this site first; fall back to nearest if no data")
    args = p.parse_args()

    date     = datetime.date.fromisoformat(args.date)
    time_utc = datetime.time.fromisoformat(args.time + ":00") if args.time else None

    if args.site:
        res = fetch_by_site(args.site, date, time_utc, args.wl, args.level,
                            log=print)
    else:
        if args.lat is None or args.lon is None:
            p.error("either --site or both --lat and --lon are required")
        res = retrieve(args.lat, args.lon, date, time_utc, args.wl, args.level,
                       force_site=args.force_site, log=print)
    print()
    print(format_result(res))

    if res["aod_rows"]:
        print(f"\nAll {len(res['aod_rows'])} AOD measurements from {res['site']}:")
        print(f"  {'Time':10}  {'AOD_440':>8}  {'AOD_500':>8}  {'AOD_675':>8}  {'AOD_870':>8}  {'alpha':>6}")
        for r in res["aod_rows"]:
            t    = r.get("Time(hh:mm:ss)", r.get("Time(HH:MM:SS)", ""))
            a440 = _float(r, "AOD_440nm")
            a500 = _float(r, "AOD_500nm")
            a675 = _float(r, "AOD_675nm")
            a870 = _float(r, "AOD_870nm")
            a_s  = (f"{angstrom_exponent(a440, a870, 440, 870):6.3f}"
                    if a440 and a870 else "   N/A")
            print(f"  {t:10}  "
                  f"{f'{a440:.4f}' if a440 else '     N/A':>8}  "
                  f"{f'{a500:.4f}' if a500 else '     N/A':>8}  "
                  f"{f'{a675:.4f}' if a675 else '     N/A':>8}  "
                  f"{f'{a870:.4f}' if a870 else '     N/A':>8}  "
                  f"{a_s}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_cli()
    else:
        run_gui()