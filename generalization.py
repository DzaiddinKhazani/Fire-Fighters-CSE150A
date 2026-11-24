"""
generalization.py -- training-vs-test fit for every structure (dirichlet estimator), and the
fire CPT of the hill-climbing DAG (H).

Gain = % reduction in log-loss relative to a constant predictor that always outputs the
period's own fire rate:
    gain = 100 * (1 - logloss(model) / logloss(constant))
computed on the training years (2015-2022, weighted rows = the full table) and on the test
years (2023-2024). A large drop from train to test means the structure fits noise in its
CPTs (overfitting). `fires per fire-CPT row` = training fires / number of parent
configurations of fire: how much data each P(fire | parents) entry is estimated from.

Output: results/generalization.md
"""
import json

import numpy as np
import pandas as pd

from bn import BayesNet
from config import (FITTED_STRUCTURES, MODELS_DIR, PROB_CLIP, RESULTS_DIR, SPLIT_INFO_PATH,
                    TARGET, TEST_PATH, TRAIN_PATH)

ESTIMATOR = "dirichlet"


def log_loss(p, y, w=None):
    p = np.clip(p, *PROB_CLIP)
    return np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w)


def main():
    train, test = pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)
    states = json.loads(SPLIT_INFO_PATH.read_text())["states"]
    structures = json.loads((RESULTS_DIR / "structures.json").read_text())
    w, y, yt = train["weight"].to_numpy(), train[TARGET].to_numpy(), test[TARGET].to_numpy()
    n_fires = int(y.sum())
    base_tr = log_loss(np.full(len(y), np.average(y, weights=w)), y, w)
    base_te = log_loss(np.full(len(yt), yt.mean()), yt)

    rows = []
    for s in FITTED_STRUCTURES:
        bn = BayesNet.load(MODELS_DIR / f"{s}_{ESTIMATOR}.pkl")
        pa = structures[s]["fire_parents"]
        n_rows = int(np.prod([len(states[p]) for p in pa])) if pa else None
        g_tr = 100 * (1 - log_loss(bn.predict_fire(train), y, w) / base_tr)
        g_te = 100 * (1 - log_loss(bn.predict_fire(test), yt) / base_te)
        rows.append({"structure": s, "fire parents": ", ".join(pa) or "(none: fire is a root)",
                     "fire-CPT rows": n_rows if n_rows else "-",
                     "fires per fire-CPT row": round(n_fires / n_rows, 1) if n_rows else "-",
                     "train gain %": round(g_tr, 2), "test gain %": round(g_te, 2),
                     "drop": round(g_te - g_tr, 2)})
    df = pd.DataFrame(rows)

    h = BayesNet.load(MODELS_DIR / f"hillclimb_{ESTIMATOR}.pkl")
    pa = h.parents[TARGET]
    cpt = pd.DataFrame(h.cpts[TARGET][..., 1] * 1e4, index=states[pa[0]], columns=states[pa[1]])
    cpt.index.name = f"{pa[0]} \\ {pa[1]}"

    text = "\n".join([
        "# Generalization: training vs test fit (dirichlet estimator)\n",
        f"Gain = % reduction in log-loss vs a constant predictor (the period's own fire rate). "
        f"{n_fires:,} training fires. A big train-to-test drop = overfitting.\n",
        df.to_markdown(index=False) + "\n",
        "## Fire CPT of the hill-climbing DAG (H): P(fire = 1 | parents) per 10,000 cell-days\n",
        cpt.round(2).to_markdown() + "\n",
    ])
    (RESULTS_DIR / "generalization.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
