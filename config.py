"""
config.py -- single source of truth for the wildfire Bayesian network project.

Active settings are what Phase 2 (build the cell-day table) needs.
Commented-out blocks are filled in AFTER Phase 2, once we have looked at the data.
"""
from pathlib import Path

# =====================================================================
# PROJECT ROOT / DIRECTORIES
# =====================================================================
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED = 42

# =====================================================================
# PHASE 2: SPATIAL GRID AND STUDY PERIOD
# =====================================================================
CRS = "EPSG:32611"          # UTM zone 11N (metric), suitable for San Diego County
CELL = 1609.344             # 1 mile in metres -> ~1 sq mile cells
SUB = 54                    # fine pixels per cell side for slope (CELL / 54 ~ 29.8 m)

COUNTY_STATE_FIPS = "06"    # California (Census FIPS, used for the county boundary)
STATE_ABBR = "CA"           # used for the FPA-FOD attribute filter
COUNTY_NAME = "San Diego"

START_YEAR = 2015
END_YEAR = 2024
YEARS = range(START_YEAR, END_YEAR + 1)
BURN_HISTORY_START_YEAR = 1992   # FPA-FOD starts in 1992; used for "years since last fire"

# =====================================================================
# PHASE 2: RAW DATA PATHS  (edit to match your downloads)
# =====================================================================
PATHS = dict(
    county=RAW_DIR / "tl_2024_us_county" / "tl_2024_us_county.shp",   # Census TIGER counties (geopandas reads .zip directly; extract if it fails)
    fires=RAW_DIR / "FPA_FOD_20260615.sqlite",  # USDA FPA FOD 7th ed. SQLite; adjust to the real file name after unzipping
    fires_layer="Fires",
    dem=RAW_DIR / "dem.tif",                     # USGS elevation
    veg_type=RAW_DIR / "LF2025_EVT_CONUS.tif",       # LANDFIRE existing vegetation type
    veg_cover=RAW_DIR / "LF2025_EVC_CONUS.tif",      # LANDFIRE existing vegetation cover
    veg_height=RAW_DIR / "LF2025_EVH_CONUS.tif",     # LANDFIRE existing vegetation height
    gridmet_dir=RAW_DIR / "gridmet",             # files like tmmx_2015.nc
    # fhsz=RAW_DIR / "fhsz_sd.shp",              # PHASE 6: CAL FIRE FHSZ polygons
)

# FPA-FOD fields used to filter to San Diego County (verify against your edition)
FIRE_FIELDS = dict(
    state="STATE",
    county="FIPS_NAME",
    discovery_date="DISCOVERY_DATE",
    lat="LATITUDE",
    lon="LONGITUDE",
)

# our column name -> (gridMET file stem, variable name inside the file)
GRIDMET = {
    "tmax": ("tmmx", "air_temperature"),      # Kelvin, converted to degC in the pipeline
    "rhmin": ("rmin", "relative_humidity"),   # percent
    "wind": ("vs", "wind_speed"),             # m/s
}

# =====================================================================
# PHASE 2: OUTPUT TABLE
# =====================================================================
def cell_day_path(year):
    return PROCESSED_DIR / f"cell_days_{year}.parquet"

# =====================================================================
# VARIABLE GROUPS  (column names produced by Phase 2)
# =====================================================================
KEYS = ["cell_id", "date", "year"]
TARGET = "fire"

WEATHER_VARS = ["tmax", "rhmin", "wind"]
FUEL_VARS = ["veg_type", "veg_cover", "veg_height"]
TERRAIN_VARS = ["elev", "slope"]
BURN_VARS = ["yrs_since_fire"]           # raw; binned to `burn_hist` in Phase 3

# =====================================================================
# PHASE 3: DISCRETIZATION  (uncomment after exploring distributions)
# =====================================================================
# Bin edges are frozen here so every model uses identical bins.
# Values below are placeholders to be replaced with domain thresholds or quantiles.
#
# BIN_EDGES = {
#     "tmax":   [-np.inf, 20, 28, 35, np.inf],      # degC
#     "rhmin":  [-np.inf, 10, 20, 35, np.inf],      # percent (<10 = critically dry)
#     "wind":   [-np.inf, 3, 6, 9, np.inf],         # m/s
#     "elev":   [-np.inf, 200, 600, 1200, np.inf],  # m
#     "slope":  [-np.inf, 5, 15, 25, np.inf],       # degrees
#     "yrs_since_fire": [-np.inf, 5, 15, 30, np.inf],   # NaN -> its own "no record" bin
# }
# CATEGORICAL_COLLAPSE = {
#     # collapse LANDFIRE codes into a few fuel classes
#     "veg_type": {},   # e.g. {codes: "chaparral", codes: "grass", codes: "forest", ...}
# }
# DISCRETE_FEATURES = ["tmax", "rhmin", "wind", "elev", "slope",
#                      "veg_type", "veg_cover", "veg_height", "burn_hist"]

# =====================================================================
# PHASE 3: TRAIN / TEST SPLIT AND SUBSAMPLING
# =====================================================================
# TRAIN_YEARS = range(2015, 2023)     # 2015-2022
# TEST_YEARS = range(2023, 2025)      # 2023-2024
# VAL_YEARS = (2021, 2022)            # temporal validation inside the training span
#
# NEG_KEEP = 0.02                     # fraction of Fire=0 rows kept in TRAIN only
#                                     # (test set stays full; prior is corrected afterward)

# =====================================================================
# PHASE 4: MODEL STRUCTURES
# =====================================================================
# NAIVE_BAYES_EDGES = [("fire", f) for f in DISCRETE_FEATURES]
#
# EXPERT_EDGES = [
#     # terrain -> weather / vegetation
#     ("elev", "tmax"),
#     ("elev", "veg_type"),
#     ("slope", "veg_type"),
#     # vegetation structure
#     ("veg_type", "veg_cover"),
#     ("veg_cover", "veg_height"),
#     ("burn_hist", "veg_cover"),
#     # weather interactions
#     ("tmax", "rhmin"),
#     # drivers of ignition
#     ("tmax", "fire"),
#     ("rhmin", "fire"),
#     ("wind", "fire"),
#     ("veg_type", "fire"),
#     ("veg_cover", "fire"),
#     ("slope", "fire"),
# ]
#
# ESTIMATORS = {
#     "mle": dict(kind="MaximumLikelihoodEstimator"),
#     "bdeu": dict(kind="BayesianEstimator", prior_type="BDeu", equivalent_sample_size=10),
# }

# =====================================================================
# PHASE 5: EVALUATION
# =====================================================================
# PROB_CLIP = (1e-6, 1 - 1e-6)        # clip predictions before log-loss
# CALIBRATION_BINS = 10
# BOOTSTRAP_REPS = 1000               # CI for metrics when test ignitions are few
# INDEPENDENCE_ALPHA = 0.05           # chi-square / G-test threshold for edge checks

# =====================================================================
# PHASE 6: FHSZ COMPARISON
# =====================================================================
# FHSZ_ORDER = {"Moderate": 1, "High": 2, "Very High": 3}
# STATIC_QUERY_VARS = TERRAIN_VARS + FUEL_VARS + ["burn_hist"]   # weather marginalized out