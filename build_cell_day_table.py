"""
Phase 2: build the cell-day table for San Diego County, 2015-2024.

All settings (CRS, cell size, paths, field names, years) live in config.py.

Untested outline. Things to verify against your own downloads:
  - FPA-FOD field names and layer name (config.FIRE_FIELDS, PATHS["fires_layer"])
  - gridMET variable names (check `xr.open_dataset(f).data_vars`; config.GRIDMET)
  - raster CRS / nodata values

Usage:
    python build_cell_day_table.py                 # all years in config.YEARS
    python build_cell_day_table.py --years 2020    # test on one year first
    python build_cell_day_table.py --years 2019 2020

Design choice: instead of building ~4,500 polygons and doing spatial joins,
the grid is just an affine transform. A cell is an (row, col) index, so
assigning points to cells is integer arithmetic and rasters are resampled
straight onto the grid with rasterio.reproject.
"""
import argparse
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from pyogrio import read_dataframe
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

from config import (
    BURN_HISTORY_START_YEAR,
    BURN_VARS,
    CELL,
    COUNTY_NAME,
    COUNTY_STATE_FIPS,
    CRS,
    FIRE_FIELDS,
    GRIDMET,
    PATHS,
    SUB,
    TARGET,
    YEARS,
    cell_day_path,
)

BURN_COL = BURN_VARS[0]   # "yrs_since_fire"


# ----------------------------------------------------------------- 1. grid
@dataclass
class Grid:
    transform: Affine
    nrows: int
    ncols: int
    minx: float
    top: float
    ids: np.ndarray        # flat indices of cells that touch the county
    x: np.ndarray          # centroid x (CRS metres) for those cells
    y: np.ndarray
    county_geom: object    # county outline (CRS metres), used to keep only fires inside it


def make_grid():
    counties = gpd.read_file(PATHS["county"])
    county = counties[
        (counties.STATEFP == COUNTY_STATE_FIPS) & (counties.NAME == COUNTY_NAME)
    ].to_crs(CRS)
    if county.empty:
        raise ValueError(f"County {COUNTY_NAME} (state {COUNTY_STATE_FIPS}) not found in boundary file")

    minx, miny, maxx, maxy = county.total_bounds
    ncols = int(np.ceil((maxx - minx) / CELL))
    nrows = int(np.ceil((maxy - miny) / CELL))
    top = miny + nrows * CELL
    transform = Affine(CELL, 0, minx, 0, -CELL, top)

    mask = rasterize(
        county.geometry, out_shape=(nrows, ncols), transform=transform,
        all_touched=True, fill=0, default_value=1, dtype="uint8",
    ).astype(bool)

    ids = np.flatnonzero(mask.ravel())
    rows, cols = np.divmod(ids, ncols)
    x = minx + (cols + 0.5) * CELL
    y = top - (rows + 0.5) * CELL
    return Grid(transform, nrows, ncols, minx, top, ids, x, y, county.geometry.iloc[0])


# ----------------------------------------------------------------- 2. ignitions
def parse_dates(s):
    """FPA-FOD discovery dates are ISO strings in some formats and Julian day
    numbers (e.g. 2457023.5) in others. Handle both."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_datetime(s, unit="D", origin="julian").dt.normalize()
    return pd.to_datetime(s).dt.normalize()


def load_ignitions(grid):
    f = FIRE_FIELDS
    # Query by lon/lat bounding box on the numeric columns and ignore the stored
    # geometry (some SQLite exports encode it in a way GDAL cannot read).
    xmin, ymin, xmax, ymax = Transformer.from_crs(
        CRS, "EPSG:4326", always_xy=True
    ).transform_bounds(
        grid.minx, grid.top - grid.nrows * CELL, grid.minx + grid.ncols * CELL, grid.top
    )
    where = (f"{f['lat']} BETWEEN {ymin} AND {ymax} AND "
             f"{f['lon']} BETWEEN {xmin} AND {xmax}")
    df = read_dataframe(
        PATHS["fires"], layer=PATHS["fires_layer"], where=where, read_geometry=False,
        columns=[f["lat"], f["lon"], f["discovery_date"]],
    )
    g = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[f["lon"]], df[f["lat"]]), crs="EPSG:4269",
    ).to_crs(CRS)
    g = g[g.within(grid.county_geom)].reset_index(drop=True)   # inside the county outline
    print(f"fire records inside {COUNTY_NAME} County (all years): {len(g)}")

    date = parse_dates(g[f["discovery_date"]])
    col = np.floor((g.geometry.x - grid.minx) / CELL).astype(int)
    row = np.floor((grid.top - g.geometry.y) / CELL).astype(int)
    inside = (row >= 0) & (row < grid.nrows) & (col >= 0) & (col < grid.ncols)
    cell_id = row * grid.ncols + col

    ign = pd.DataFrame({
        "cell_id": cell_id[inside].astype("int32"),
        "date": date[inside].astype("datetime64[ns]"),
    })
    ign = ign[ign.cell_id.isin(grid.ids)]                       # valid cells only
    ign = ign[ign.date.dt.year >= BURN_HISTORY_START_YEAR]

    n_raw = len(ign)
    ign = ign.drop_duplicates()            # 2 fires, same cell, same day -> one label
    print(f"ignitions in grid: {n_raw}  (cell-day labels after dedup: {len(ign)})")
    return ign


# ----------------------------------------------------------------- 3. static features
def raster_to_grid(path, grid, resampling):
    dst = np.full((grid.nrows, grid.ncols), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_nodata=src.nodata, dst_nodata=np.nan,
            dst_transform=grid.transform, dst_crs=CRS, resampling=resampling,
        )
    return dst


def slope_to_grid(path_dem, grid):
    """Compute slope at ~30 m, THEN average to the 1-mile cell
    (averaging the DEM first would flatten terrain)."""
    res = CELL / SUB
    fine_t = Affine(res, 0, grid.minx, 0, -res, grid.top)
    fine = np.full((grid.nrows * SUB, grid.ncols * SUB), np.nan, dtype="float32")
    with rasterio.open(path_dem) as src:
        reproject(
            source=rasterio.band(src, 1), destination=fine,
            src_nodata=src.nodata, dst_nodata=np.nan,
            dst_transform=fine_t, dst_crs=CRS, resampling=Resampling.bilinear,
        )
    dzdy, dzdx = np.gradient(fine, res)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    blocks = slope.reshape(grid.nrows, SUB, grid.ncols, SUB)
    return np.nanmean(blocks, axis=(1, 3)).astype("float32")


def load_static(grid):
    static = {
        "elev": raster_to_grid(PATHS["dem"], grid, Resampling.average),
        "slope": slope_to_grid(PATHS["dem"], grid),
        # categorical / ordinal rasters: most common value in the cell
        "veg_type": raster_to_grid(PATHS["veg_type"], grid, Resampling.mode),
        "veg_cover": raster_to_grid(PATHS["veg_cover"], grid, Resampling.mode),
        "veg_height": raster_to_grid(PATHS["veg_height"], grid, Resampling.mode),
    }
    # flatten to one value per valid cell, same order as grid.ids
    return {k: v.ravel()[grid.ids] for k, v in static.items()}


# ----------------------------------------------------------------- 4. weather
def load_weather(year, grid):
    """Return (dates, {var: array[n_days, n_cells]}).
    gridMET is ~4 km, so many cells share a pixel."""
    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(grid.x, grid.y)
    lat_da = xr.DataArray(lat, dims="cell")
    lon_da = xr.DataArray(lon, dims="cell")

    out, dates = {}, None
    for name, (stem, var) in GRIDMET.items():
        with xr.open_dataset(PATHS["gridmet_dir"] / f"{stem}_{year}.nc") as ds:
            pts = ds[var].sel(lat=lat_da, lon=lon_da, method="nearest")
            arr = pts.transpose("day", "cell").values.astype("float32")
            if name == "tmax":
                arr = arr - 273.15                      # K -> degC
            out[name] = arr
            dates = pd.to_datetime(ds["day"].values).astype("datetime64[ns]")
    return dates, out


# ----------------------------------------------------------------- 5. burn history
def attach_burn_history(df, all_ign):
    """
    Years since the last recorded fire in the cell, using only fires STRICTLY
    BEFORE each date (no leakage from the same day or the future).
    Cells with no earlier record are NaN -> bin them as 'no fire since 1992'.
    """
    hist = all_ign.rename(columns={"date": "last_fire"}).sort_values("last_fire")
    df = df.sort_values("date")
    # merge_asof needs identical key dtypes on both sides
    hist = hist.astype({"cell_id": "int32", "last_fire": "datetime64[ns]"})
    df = df.astype({"cell_id": "int32", "date": "datetime64[ns]"})
    df = pd.merge_asof(
        df, hist, left_on="date", right_on="last_fire",
        by="cell_id", direction="backward", allow_exact_matches=False,
    )
    df[BURN_COL] = ((df["date"] - df["last_fire"]).dt.days / 365.25).astype("float32")
    return df.drop(columns="last_fire").sort_values(["date", "cell_id"]).reset_index(drop=True)


# ----------------------------------------------------------------- 6. assemble
def build_year(year, grid, static, all_ign):
    dates, weather = load_weather(year, grid)
    n_days, n_cells = len(dates), len(grid.ids)

    # row-major ravel of [day, cell] pairs matches tile (cells) / repeat (dates)
    df = pd.DataFrame({
        "cell_id": np.tile(grid.ids, n_days).astype("int32"),
        "date": np.repeat(dates, n_cells),
    })
    for name, arr in weather.items():
        df[name] = arr.ravel()
    for name, vals in static.items():
        df[name] = np.tile(vals, n_days)

    # label: 1 only on the discovery day
    lab = all_ign[all_ign.date.dt.year == year].assign(**{TARGET: np.int8(1)})
    df = df.merge(lab, on=["cell_id", "date"], how="left")
    df[TARGET] = df[TARGET].fillna(0).astype("int8")

    df = attach_burn_history(df, all_ign)
    df["year"] = np.int16(year)

    # sanity check: label count must equal ignitions in that year
    expected = len(lab)
    assert df[TARGET].sum() == expected, (
        f"{year}: {df[TARGET].sum()} labels vs {expected} ignitions"
    )
    return df


def main(years):
    grid = make_grid()
    print(f"grid: {grid.nrows} x {grid.ncols}, {len(grid.ids)} cells touch the county")

    static = load_static(grid)

    # The TIGER county outline includes coastal water, so some grid cells are all ocean
    # (no elevation). Drop those only. Cells with elevation but no LANDFIRE vegetation
    # value are kept (they can still ignite) and get their own "no data" bin in Phase 3.
    no_elev = np.isnan(static["elev"])
    no_veg = np.isnan(static["veg_type"])
    print(f"cells without elevation (offshore): {int(no_elev.sum())}")
    print(f"cells without vegetation type: {int(no_veg.sum())} "
          f"({int((no_veg & ~no_elev).sum())} of them on land)")
    land = ~no_elev
    print(f"keeping {int(land.sum())} land cells")
    grid.ids, grid.x, grid.y = grid.ids[land], grid.x[land], grid.y[land]
    static = {k: v[land] for k, v in static.items()}

    all_ign = load_ignitions(grid)          # 1992-2024, also feeds burn history

    for year in years:
        df = build_year(year, grid, static, all_ign)
        df.to_parquet(cell_day_path(year), index=False)
        print(f"{year}: {len(df):,} rows, {int(df[TARGET].sum())} ignitions, "
              f"{df.isna().mean().max():.1%} max null rate -> {cell_day_path(year).name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the cell-day table (Phase 2).")
    ap.add_argument("--years", type=int, nargs="+", default=list(YEARS),
                    help="years to build (default: all years in config.YEARS)")
    args = ap.parse_args()
    main(args.years)