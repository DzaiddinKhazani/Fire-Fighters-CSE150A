"""
Clip a LANDFIRE raster (full CONUS or any larger download) to San Diego County
plus a 5 km buffer, and save it under the name config.py expects.

The first clip of LF2025_EVT/EVC/EVH was cut too far south: the north third of
the county (Fallbrook, Palomar, Anza-Borrego north) was nodata. check_inputs.py
now catches that.

Usage (one file at a time):
    python clip_landfire.py ~/Downloads/LF2025_EVT_CONUS/Tif/LC25_EVT_250.tif veg_type
    python clip_landfire.py <path to EVC tif> veg_cover
    python clip_landfire.py <path to EVH tif> veg_height
Then:  python check_inputs.py   and   python build_cell_day_table.py
"""
import sys

import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds

from config import COUNTY_NAME, COUNTY_STATE_FIPS, PATHS

BUFFER_M = 5000

if len(sys.argv) != 3 or sys.argv[2] not in ("veg_type", "veg_cover", "veg_height"):
    sys.exit(__doc__ + "\nNeeds two arguments: the downloaded LANDFIRE .tif, then "
             "veg_type (EVT), veg_cover (EVC) or veg_height (EVH).")
src_path, key = sys.argv[1], sys.argv[2]
out_path = PATHS[key]

counties = gpd.read_file(PATHS["county"])
county = counties[(counties.STATEFP == COUNTY_STATE_FIPS) & (counties.NAME == COUNTY_NAME)]

with rasterio.open(src_path) as src:
    l, b, r, t = county.to_crs(src.crs).buffer(BUFFER_M).total_bounds
    win = from_bounds(l, b, r, t, src.transform).round_offsets().round_lengths()
    data = src.read(1, window=win, boundless=True, fill_value=src.nodata)
    meta = src.meta.copy()
    meta.update(height=data.shape[0], width=data.shape[1],
                transform=src.window_transform(win), compress="lzw", tiled=True)

if out_path.resolve() == __import__("pathlib").Path(src_path).resolve():
    sys.exit("refusing to overwrite the input file")
with rasterio.open(out_path, "w", **meta) as dst:
    dst.write(data, 1)
print(f"wrote {out_path} ({data.shape[1]} x {data.shape[0]} px)")
