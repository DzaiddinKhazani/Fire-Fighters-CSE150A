import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (BIN_EDGES, BIN_LABELS, CATEGORICAL_COLLAPSE, DISCRETE_FEATURES,
                    FIGURES_DIR, KEYS, NAN_LABEL, NEG_KEEP, RENAME_DISCRETE, SEED,
                    SPLIT_INFO_PATH, TARGET, TEST_PATH, TEST_YEARS, TRAIN_PATH,
                    TRAIN_YEARS, VAL_YEARS, cell_day_path)

CONTINUOUS = ["tmax", "rhmin", "wind", "elev", "slope", "veg_cover", "veg_height",
              "yrs_since_fire"]
UNITS = {"tmax": "degC", "rhmin": "%", "wind": "m/s", "elev": "m", "slope": "deg",
         "veg_cover": "% canopy", "veg_height": "m", "yrs_since_fire": "years"}

def load(years):
    return pd.concat([pd.read_parquet(cell_day_path(y)) for y in years], ignore_index=True)

# plot each variable's distribution
def plot_distributions(raw):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    fire = raw[raw[TARGET] == 1]
    nofire = raw[raw[TARGET] == 0].sample(min(500_000, (raw[TARGET] == 0).sum()), random_state=SEED)
    for ax, v in zip(axes.flat, CONTINUOUS):
        lo, hi = raw[v].quantile([0.001, 0.999])
        bins = np.linspace(lo, hi, 50)
        ax.hist(nofire[v].dropna(), bins=bins, density=True, alpha=0.55, color="#4C78A8",
                label="no fire")
        ax.hist(fire[v].dropna(), bins=bins, density=True, alpha=0.55, color="#E45756",
                label="fire")
        for e in BIN_EDGES[v][1:-1]:
            ax.axvline(e, color="black", ls="--", lw=1)
        miss = raw[v].isna().mean()
        ax.set_title(f"{v}  ({miss:.0%} missing)" if miss > 0 else v)
        ax.set_xlabel(UNITS[v])
        ax.set_yticks([])
    ax = axes.flat[-1]
    vt = raw.assign(v=raw.veg_type.map(CATEGORICAL_COLLAPSE["veg_type"]).fillna("missing"))
    share = pd.crosstab(vt.v, vt[TARGET], normalize="columns")
    share.plot.bar(ax=ax, color=["#4C78A8", "#E45756"], alpha=0.8, legend=False, rot=20)
    ax.set_title("veg_type (dominant life form)")
    ax.set_xlabel("")
    axes.flat[0].legend()
    fig.suptitle("Training years 2015-2022: distributions for fire vs no-fire cell-days "
                 "(dashed = frozen bin edges)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=130)
    plt.close(fig)

def plot_bin_rates(disc):
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    base = disc[TARGET].mean()
    for ax, v in zip(axes.flat, DISCRETE_FEATURES):
        r = disc.groupby(v, observed=True)[TARGET].agg(["mean", "size"])
        ax.bar(r.index.astype(str), r["mean"] * 1e4, color="#E45756", alpha=0.8)
        ax.axhline(base * 1e4, color="black", ls="--", lw=1)
        ax.set_title(v)
        ax.set_ylabel("fires per 10,000 cell-days")
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("Training years 2015-2022: ignition rate per bin (dashed = overall rate)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bin_fire_rates.png", dpi=130)
    plt.close(fig)

# discretize cell-day table values
def discretize(raw):
    out = raw[KEYS + [TARGET]].copy()
    # sort each value into a bin
    for v, edges in BIN_EDGES.items():
        lab = BIN_LABELS[v]
        col = pd.cut(raw[v], bins=edges, labels=lab, right=False).astype(object)
        # handle missing values
        if v in NAN_LABEL:
            col = col.where(raw[v].notna(), NAN_LABEL[v])
            lab = lab + [NAN_LABEL[v]]
        out[RENAME_DISCRETE.get(v, v)] = pd.Categorical(col, categories=lab)
    # handle vegetation type separately (since veg_type is already a category)
    vt = raw["veg_type"].map(CATEGORICAL_COLLAPSE["veg_type"]).fillna(NAN_LABEL["veg_type"])
    out["veg_type"] = pd.Categorical(vt, categories=["nonburnable", "shrub", "other_veg",
                                                     NAN_LABEL["veg_type"]])
    assert not out[DISCRETE_FEATURES].isna().any().any(), "a value fell outside every bin"
    return out[KEYS + DISCRETE_FEATURES + [TARGET]]

def main():
    rng = np.random.default_rng(SEED)

    train_raw = load(TRAIN_YEARS)
    plot_distributions(train_raw)
    train = discretize(train_raw)
    plot_bin_rates(train)
    del train_raw
    test = discretize(load(TEST_YEARS))

    # subsample non-fire rows in TRAIN only
    keep = (train[TARGET] == 1).to_numpy() | (rng.random(len(train)) < NEG_KEEP)
    full_prior = float(train[TARGET].mean())
    n_full, n_fire = len(train), int(train[TARGET].sum())
    train = train[keep].reset_index(drop=True)
    train["weight"] = np.where(train[TARGET] == 1, 1.0, 1.0 / NEG_KEEP).astype("float32")

    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)

    info = {
        "neg_keep": NEG_KEEP,
        "seed": SEED,
        "train_years": [min(TRAIN_YEARS), max(TRAIN_YEARS)],
        "test_years": [min(TEST_YEARS), max(TEST_YEARS)],
        "val_years_inside_train": list(VAL_YEARS),
        "train_rows_before_subsample": n_full,
        "train_rows_after_subsample": len(train),
        "train_fires": n_fire,
        "train_prior_full": full_prior,
        "train_prior_subsampled": float(train[TARGET].mean()),
        "test_rows": len(test),
        "test_fires": int(test[TARGET].sum()),
        "test_prior": float(test[TARGET].mean()),
        "prior_correction": "odds_true = odds_model * neg_keep  ->  "
                            "p = p_s*k / (p_s*k + 1 - p_s), with k = neg_keep",
        "note": "column 'weight' = 1/neg_keep for kept non-fire rows, 1 for fires, "
                "if you prefer weighted counts instead of the prior correction",
        "states": {v: list(train[v].cat.categories) for v in DISCRETE_FEATURES},
    }
    SPLIT_INFO_PATH.write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "states"}, indent=2))


if __name__ == "__main__":
    main()
