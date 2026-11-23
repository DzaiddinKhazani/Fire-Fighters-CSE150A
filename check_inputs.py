"""
Pre-flight check for Phase 2. Run this BEFORE build_cell_day_table.py:

    python check_inputs.py

Prints OK / PROBLEM for every input so you can fix paths and names in config.py.
Untested sketch.
"""
import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from pyogrio import list_layers, read_dataframe, read_info
from rasterio.warp import transform_bounds

from config import (COUNTY_NAME, COUNTY_STATE_FIPS, FIRE_FIELDS, GRIDMET, PATHS,
                    YEARS)

problems = []


def report(ok, msg):
    print(("  OK       " if ok else "  PROBLEM  ") + msg)
    if not ok:
        problems.append(msg)


def covers(bounds, county_bounds):
    l, b, r, t = bounds
    cl, cb, cr, ct = county_bounds
    return l <= cl and b <= cb and r >= cr and t >= ct


def gaps(bounds, county_bounds):
    l, b, r, t = bounds
    cl, cb, cr, ct = county_bounds
    out = []
    if l > cl:
        out.append(f"west edge short by {l - cl:.3f} deg")
    if b > cb:
        out.append(f"south edge short by {b - cb:.3f} deg")
    if r < cr:
        out.append(f"east edge short by {cr - r:.3f} deg")
    if t < ct:
        out.append(f"north edge short by {ct - t:.3f} deg")
    return ", ".join(out) if out else "fully covered"


# ---------------------------------------------------------------- county
print("\n[county boundary]")
county_bounds = None
try:
    counties = gpd.read_file(PATHS["county"])
    c = counties[(counties.STATEFP == COUNTY_STATE_FIPS) & (counties.NAME == COUNTY_NAME)]
    report(len(c) == 1, f"found {COUNTY_NAME} County ({len(c)} match)")
    county_bounds = tuple(c.to_crs("EPSG:4326").total_bounds)
    print("           lon/lat bounds:", tuple(round(x, 3) for x in county_bounds))
except Exception as e:
    report(False, f"cannot read {PATHS['county']}: {e}  (try extracting the zip)")

# ---------------------------------------------------------------- ignitions
print("\n[ignition data]")
try:
    layers = [row[0] for row in list_layers(PATHS["fires"])]
    report(PATHS["fires_layer"] in layers, f"layer '{PATHS['fires_layer']}' (layers found: {layers})")
    layer = PATHS["fires_layer"] if PATHS["fires_layer"] in layers else layers[0]
    fields = list(read_info(PATHS["fires"], layer=layer)["fields"])
    for key, col in FIRE_FIELDS.items():
        report(col in fields, f"field {col} ({key})")
    if county_bounds:
        l, b_, r, t = county_bounds
        where = (f"{FIRE_FIELDS['lat']} BETWEEN {b_} AND {t} AND "
                 f"{FIRE_FIELDS['lon']} BETWEEN {l} AND {r}")
        sample = read_dataframe(PATHS["fires"], layer=layer, where=where,
                                read_geometry=False, max_features=5)
        report(len(sample) > 0, f"bounding-box query returned rows ({len(sample)} sampled)")
        if len(sample):
            print("           date column dtype:", sample[FIRE_FIELDS["discovery_date"]].dtype,
                  "| example:", sample[FIRE_FIELDS["discovery_date"]].iloc[0])
    else:
        print("           (skipped row test: county boundary not loaded)")
except Exception as e:
    report(False, f"cannot read {PATHS['fires']}: {e}")

# ---------------------------------------------------------------- rasters
print("\n[rasters]")
for key in ["dem", "veg_type", "veg_cover", "veg_height"]:
    try:
        with rasterio.open(PATHS[key]) as src:
            px = src.width * src.height
            print(f"           {key}: {src.width} x {src.height} px, crs={src.crs}, "
                  f"nodata={src.nodata}, res={src.res[0]:.5g}")
            report(src.crs is not None, f"{key}: has a coordinate system")
            if key == "dem":
                if px > 5e8:
                    print(f"           note: DEM is {px/1e6:.0f}M pixels; cropping it to the county speeds things up")
            else:
                report(px < 4e8, f"{key}: size looks clipped (not full CONUS)")
            if county_bounds and key != "dem":
                # The bounding box can look fine while the inside is nodata
                # (that is how the first LANDFIRE clip was broken), so sample
                # actual pixel values on a grid over the county's land area.
                from shapely.geometry import Point
                land = c.to_crs(src.crs).geometry.iloc[0]
                xs = np.linspace(land.bounds[0], land.bounds[2], 60)
                ys = np.linspace(land.bounds[1], land.bounds[3], 60)
                pts = [(x, y) for x in xs for y in ys if land.contains(Point(x, y))]
                vals = np.array([v[0] for v in src.sample(pts)])
                bad = int(np.isin(vals, [src.nodata, -9999]).sum())   # LANDFIRE uses both
                frac = bad / max(len(pts), 1)
                report(frac < 0.15, f"{key}: {frac:.0%} of sample points inside the county are nodata "
                                    "(expect only ocean, roughly <15%)")
            if county_bounds:
                b = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                print(f"           {key} extent (lon/lat): {tuple(round(x, 3) for x in b)}")
                report(covers(b, county_bounds),
                       f"{key}: covers the county bounding box ({gaps(b, county_bounds)})")
    except Exception as e:
        report(False, f"cannot open {PATHS[key]}: {e}")

# ---------------------------------------------------------------- gridMET
print("\n[gridMET]")
for name, (stem, var) in GRIDMET.items():
    missing = [y for y in YEARS if not (PATHS["gridmet_dir"] / f"{stem}_{y}.nc").exists()]
    report(not missing, f"{stem}: all {len(list(YEARS))} yearly files present"
                        + (f" (missing {missing})" if missing else ""))
    first = PATHS["gridmet_dir"] / f"{stem}_{YEARS[0]}.nc"
    if first.exists():
        try:
            with xr.open_dataset(first) as ds:
                report(var in ds.data_vars, f"{stem}: variable '{var}' (found {list(ds.data_vars)})")
                if var in ds.data_vars:
                    print(f"           {stem}_{YEARS[0]}: {dict(ds.sizes)}")
                if county_bounds:
                    b = (float(ds.lon.min()), float(ds.lat.min()), float(ds.lon.max()), float(ds.lat.max()))
                    report(covers(b, county_bounds),
                           f"{stem}: grid covers the county bounding box ({gaps(b, county_bounds)})")
        except Exception as e:
            report(False, f"cannot open {first.name}: {e}  (is netCDF4 installed?)")

# ---------------------------------------------------------------- summary
print("\n" + ("All inputs look good. Try: python build_cell_day_table.py --years 2020"
              if not problems else f"{len(problems)} problem(s) to fix before running Phase 2."))

