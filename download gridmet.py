"""
Download gridMET for San Diego County only, over OPeNDAP (THREDDS).

Why: the full CONUS yearly files are large. OPeNDAP lets us pull just the
bounding box we need, then save one small NetCDF per variable-year using the
names the Phase 2 script expects (tmmx_2015.nc, rmin_2015.nc, vs_2015.nc).

Requires: xarray and netCDF4 (pip install xarray netCDF4).
If opening the URL fails with a DAP/protocol error, try:  pip install pydap
and change ENGINE below to "pydap".

Untested sketch. The THREDDS URL pattern below is inferred from an example on
the gridMET page (MET/<var>/<var>_<year>.nc); if it fails, open the THREDDS
Catalog (OPeNDAP) link on https://www.climatologylab.org/gridmet.html and copy
the exact OPeNDAP address for one file, then edit BASE.
"""
import time

import numpy as np
import xarray as xr

from config import GRIDMET, PATHS, YEARS

# San Diego County bounding box plus ~1 gridMET cell (0.042 deg) of buffer
LON_MIN, LON_MAX = -117.8, -115.9
LAT_MIN, LAT_MAX = 32.4, 33.7

BASE = "http://thredds.northwestknowledge.net/thredds/dodsC/MET/{stem}/{stem}_{year}.nc"
SUFFIX = ""          # try "#fillmismatch" if you hit fill-value errors
ENGINE = None        # None = default (netCDF4); "pydap" as a fallback
RETRIES = 3


def fetch_subset(stem, year):
    url = BASE.format(stem=stem, year=year) + SUFFIX
    with xr.open_dataset(url, engine=ENGINE) as ds:
        # order-agnostic selection (gridMET latitude order differs between versions)
        lat = ds["lat"].values
        lon = ds["lon"].values
        lat_idx = np.flatnonzero((lat >= LAT_MIN) & (lat <= LAT_MAX))
        lon_idx = np.flatnonzero((lon >= LON_MIN) & (lon <= LON_MAX))
        if len(lat_idx) == 0 or len(lon_idx) == 0:
            raise ValueError("bounding box selected no grid cells")
        return ds.isel(lat=lat_idx, lon=lon_idx).load()   # only the subset is transferred


def main():
    out_dir = PATHS["gridmet_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, (stem, var) in GRIDMET.items():
        for year in YEARS:
            out = out_dir / f"{stem}_{year}.nc"
            if out.exists():
                print(f"skip {out.name} (already downloaded)")
                continue

            for attempt in range(1, RETRIES + 1):
                try:
                    sub = fetch_subset(stem, year)
                    if var not in sub.data_vars:
                        print(f"  note: '{var}' not found; variables are {list(sub.data_vars)}"
                              " -> update GRIDMET in config.py")
                    sub.to_netcdf(out)
                    print(f"saved {out.name}: {dict(sub.sizes)}")
                    break
                except Exception as e:                      # network hiccups are common
                    print(f"  {stem} {year} attempt {attempt}/{RETRIES} failed: {e}")
                    if attempt == RETRIES:
                        print(f"  giving up on {out.name}")
                    else:
                        time.sleep(5 * attempt)


if __name__ == "__main__":
    main()

