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
    veg_type=RAW_DIR / "LF2025_EVC_CONUS.tif",       # life form is derived from EVC codes (EVT file not needed)
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
# PHASE 3: DISCRETIZATION  (frozen -- every model uses these exact bins)
# =====================================================================
# Edges come from fire-weather thresholds where they exist and from the
# 2015-2022 distributions otherwise (see results/figures/distributions.png).
# Bins are right-open: [edge_i, edge_i+1).  NaN gets its own state (below).
import numpy as np

BIN_EDGES = {
    "tmax":   [-np.inf, 20, 28, 35, np.inf],       # degC: cool / mild / hot / extreme
    "rhmin":  [-np.inf, 10, 20, 35, np.inf],       # %: <10 critically dry, <20 red-flag range
    "wind":   [-np.inf, 2, 3, 4.5, np.inf],        # m/s daily mean: calm / light / moderate / strong
    "elev":   [-np.inf, 200, 600, 1200, np.inf],   # m: coast-valley / foothill / mountain / high
    "slope":  [-np.inf, 5, 15, 25, np.inf],        # degrees: flat / rolling / steep / very steep
    "veg_cover":  [-np.inf, 5, 20, 35, np.inf],    # % canopy: bare / sparse / moderate / dense
    "veg_height": [-np.inf, 0.5, 1.5, 3, np.inf],  # m: grass-low / low shrub / tall shrub / tree
    "yrs_since_fire": [-np.inf, 2, 10, np.inf],    # recent / mid / old  (+ "none" if no record)
}
BIN_LABELS = {
    "tmax": ["cool", "mild", "hot", "extreme"],
    "rhmin": ["crit_dry", "dry", "moderate", "humid"],
    "wind": ["calm", "light", "moderate", "strong"],
    "elev": ["low", "foothill", "mountain", "high"],
    "slope": ["flat", "rolling", "steep", "very_steep"],
    "veg_cover": ["bare", "sparse", "moderate", "dense"],
    "veg_height": ["low", "low_shrub", "tall_shrub", "tree"],
    "yrs_since_fire": ["recent", "mid", "old"],
}
# burn history is NaN when the cell had no fire since 1992.
NAN_LABEL = {"veg_type": "missing", "veg_cover": "missing", "veg_height": "missing",
             "yrs_since_fire": "none"}

# veg_type is the dominant life form (0 none, 1 tree, 2 shrub, 3 herb). Tree and herb
# cells are rare here (<4% of cells each), so they are merged.
CATEGORICAL_COLLAPSE = {
    "veg_type": {0: "nonburnable", 1: "other_veg", 2: "shrub", 3: "other_veg"},
}
RENAME_DISCRETE = {"yrs_since_fire": "burn_hist"}

DISCRETE_FEATURES = ["tmax", "rhmin", "wind", "elev", "slope",
                     "veg_type", "veg_cover", "veg_height", "burn_hist"]

# =====================================================================
# PHASE 3: TRAIN / TEST SPLIT AND SUBSAMPLING
# =====================================================================
TRAIN_YEARS = range(2015, 2023)     # 2015-2022
TEST_YEARS = range(2023, 2025)      # 2023-2024
VAL_YEARS = (2021, 2022)            # temporal validation inside the training span

NEG_KEEP = 0.02                     # fraction of fire=0 rows kept in TRAIN only
                                    # (test set stays full; prior is corrected afterward)
TRAIN_PATH = PROCESSED_DIR / "train_discrete.parquet"
TEST_PATH = PROCESSED_DIR / "test_discrete.parquet"
SPLIT_INFO_PATH = PROCESSED_DIR / "split_info.json"

# =====================================================================
# PHASE 4: MODEL STRUCTURES
# =====================================================================
NAIVE_BAYES_EDGES = [("fire", f) for f in DISCRETE_FEATURES]

TAN_ROOT = "elev" # tree root (terrain is upstream of everything)

# C. Expert DAG
#    fire gets 5 parents (4*4*4*4*4 = 1024 parent configurations)
EXPERT_EDGES = [
    # terrain
    ("elev", "slope"),          # higher ground in San Diego is mountain terrain, more rugged than the coastal plain
    ("elev", "tmax"),           # air cools with altitude (lapse rate); the coast is moderated by the ocean
    # vegetation
    ("elev", "veg_type"),       # elevation zones set the plant community: developed/grass low, chaparral mid, conifers high
    ("slope", "veg_type"),      # steep slopes are hard to build on or farm, so they stay in wildland shrub
    ("burn_hist", "veg_type"),  # frequent fire converts chaparral to annual grass (type conversion)
    ("veg_type", "veg_cover"),  # life form limits canopy cover (shrubland closes; grass/developed land stays open)
    ("veg_type", "veg_height"), # life form sets the height range: trees > shrubs > herbs
    ("burn_hist", "veg_cover"), # a recently burned cell has not regrown its canopy yet
    # weather
    ("tmax", "rhmin"),          # hotter air at the same moisture has lower relative humidity
    ("wind", "rhmin"),          # strong offshore (Santa Ana) winds bring very dry desert air
    # drivers of ignition
    ("tmax", "fire"),           # heat dries fine fuels and coincides with peak outdoor activity
    ("rhmin", "fire"),          # dry air lowers fuel moisture, making fuels easier to ignite
    ("veg_type", "fire"),       # fuel type sets ignitability; developed/roadside land is where people start fires
    ("burn_hist", "fire"),      # ignitions recur in the same places (roads, WUI edges): past fire marks hotspots
    ("slope", "fire"),          # most ignitions are human; steep terrain has little access and few people
]
# Omissions:
#  - wind -> fire: FPA-FOD records ignitions, and daily mean wind mostly drives spread,
#    not ignition; wind still reaches fire through rhmin.
#  - elev -> fire: its effect runs through tmax, veg_type and slope.
#  - veg_cover / veg_height -> fire: the model claims they add nothing once veg_type
#    and burn_hist are known. Testable with a conditional independence test (Phase 5).

# D. Expert DAG + latent "danger" node: a hidden daily fire-weather regime
#    (e.g. Santa Ana vs onshore marine layer) that drives all three weather
#    variables and fire. Weather->fire edges are replaced by danger->fire.
LATENT = "danger"
LATENT_CARDS = (2, 3, 4)            # fit one model per state count with EM
LATENT_EDGES = [
    (LATENT, "tmax"),               # a hot regime (offshore flow) raises temperature across the county
    (LATENT, "rhmin"),              # the same regime brings dry air
    (LATENT, "wind"),               # Santa Ana regimes are windy; marine-layer days are calm
    (LATENT, "fire"),               # ignition odds depend on the regime as a whole, not one weather reading
]
LATENT_DAG_EDGES = (
    [(a, b) for a, b in EXPERT_EDGES if not (a in WEATHER_VARS and b == "fire")]
    + LATENT_EDGES
)

# ---------------------------------------------------------------------
# Experimental structures (E-G). Each changes ONE idea relative to the expert DAG (C),
# so a score difference against C can be attributed to that idea.
# ---------------------------------------------------------------------
_EXPERT_INTO_FIRE = [(a, b) for a, b in EXPERT_EDGES if b == "fire"]
_EXPERT_NON_FIRE = [(a, b) for a, b in EXPERT_EDGES if b != "fire"]

# E. Fuel-structure DAG: does the full fuel description (type + cover + height) predict
#    ignition better than type alone? Fire's parents swap tmax and slope for veg_cover and
#    veg_height, keeping rhmin as the one weather input (4*5*5*4*4 = 1600 configurations).
FUEL_EDGES = _EXPERT_NON_FIRE + [
    ("veg_cover", "veg_height"),  # denser canopy goes with taller plants (strongest Chow-Liu link, 0.66 nats)
    ("veg_type", "fire"),
    ("veg_cover", "fire"),        # more continuous fuel bed -> an ember or spark finds fuel
    ("veg_height", "fire"),       # grass/low shrub cures fast and ignites easily; tall shrub/tree less so
    ("burn_hist", "fire"),
    ("rhmin", "fire"),
]

# F. No-weather DAG: the expert DAG with every weather -> fire edge removed.
#    If F scores the same as C, the daily weather adds nothing to ignition prediction.
NO_WEATHER_EDGES = [(a, b) for a, b in EXPERT_EDGES if not (a in WEATHER_VARS and b == "fire")]

# G. Fuel + history only: fire depends only on what is on the ground and where fires
#    started before (4*5*5*4 = 400 configurations). The leanest "human ignition" story.
FUEL_ONLY_EDGES = _EXPERT_NON_FIRE + [
    ("veg_cover", "veg_height"),
    ("veg_type", "fire"),
    ("veg_cover", "fire"),
    ("veg_height", "fire"),
    ("burn_hist", "fire"),
]

# H. Hill climbing (structure_search.py): no hand-picked edges at all. Greedy search over
#    add / remove / reverse edge moves, scored by BIC on the training data. Max parents per
#    node keeps every table estimable.
HILLCLIMB_MAX_PARENTS = 4

EXPERIMENTAL_STRUCTURES = {
    "fuel": ("E. Fuel-structure DAG", FUEL_EDGES),
    "no_weather": ("F. Expert DAG without weather -> fire", NO_WEATHER_EDGES),
    "fuel_only": ("G. Fuel + burn history only", FUEL_ONLY_EDGES),
}

ESTIMATORS = {
    "mle": dict(kind="MaximumLikelihoodEstimator"),
    "bdeu": dict(kind="BayesianEstimator", prior_type="BDeu", equivalent_sample_size=10),
    # Dirichlet prior centred on each variable's marginal instead of uniform: an unseen parent
    # configuration gets the overall rate (P(fire) ~ 0.0002) instead of BDeu's ~0.5. See bn.py.
    "dirichlet": dict(kind="Dirichlet, marginal-centred", equivalent_sample_size=10),
}

# =====================================================================
# PHASE 5: EVALUATION
# =====================================================================
PROB_CLIP = (1e-6, 1 - 1e-6)        # clip predictions before log-loss
CALIBRATION_BINS = 10               # quantile bins of predicted probability
BOOTSTRAP_REPS = 1000               # CI for metrics when test ignitions are few
                                    # (resamples whole test DAYS, since cells on the same day share weather)
# INDEPENDENCE_ALPHA = 0.05           # chi-square / G-test threshold for edge checks
MODELS_DIR = RESULTS_DIR / "models"   # fitted CPTs, one pickle per structure x estimator
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.parquet"
FITTED_STRUCTURES = ("naive_bayes", "tan", "expert", "fuel", "no_weather", "fuel_only", "hillclimb")   # A-C, E-H; D (latent, EM) is fitted by fit_latent.py

# =====================================================================
# PHASE 6: FHSZ COMPARISON
# =====================================================================
# FHSZ_ORDER = {"Moderate": 1, "High": 2, "Very High": 3}
# STATIC_QUERY_VARS = TERRAIN_VARS + FUEL_VARS + ["burn_hist"]   # weather marginalized out