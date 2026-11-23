"""
fit_models.py -- todo Phase 4 for structures A-C.

Fits Naive Bayes, TAN and the expert DAG (edges from results/structures.json) with
  * mle : maximum likelihood
  * bdeu: Dirichlet (BDeu) smoothing, equivalent sample size from config.ESTIMATORS
using weighted counts (see bn.py for why that also takes care of the subsampling correction).

Then runs sanity / diagnostic queries with variable elimination and writes
  results/models/<structure>_<estimator>.pkl
  results/sanity_queries.md
"""
import json

import numpy as np
import pandas as pd

from bn import BayesNet, fit_cpts, parents_from_edges
from config import (DISCRETE_FEATURES, ESTIMATORS, FITTED_STRUCTURES, MODELS_DIR,
                    RESULTS_DIR, SEED, SPLIT_INFO_PATH, TARGET, TRAIN_PATH)

HOT_DRY_WINDY = {"tmax": "extreme", "rhmin": "crit_dry", "wind": "strong"}
COOL_HUMID_CALM = {"tmax": "cool", "rhmin": "humid", "wind": "calm"}
DIAGNOSTIC_VARS = ["rhmin", "tmax", "wind", "veg_type", "burn_hist"]


def load_states():
    states = json.loads(SPLIT_INFO_PATH.read_text())["states"]
    states[TARGET] = [0, 1]
    return states


def fit_all(train, states):
    structures = json.loads((RESULTS_DIR / "structures.json").read_text())
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = {}
    for s in FITTED_STRUCTURES:
        parents = parents_from_edges(structures[s]["edges"], DISCRETE_FEATURES + [TARGET])
        for est, spec in ESTIMATORS.items():
            cpts = fit_cpts(train, parents, states, estimator=est,
                            ess=spec.get("equivalent_sample_size", 0))
            name = f"{s}_{est}"
            models[name] = BayesNet(parents, states, cpts, name)
            models[name].save(MODELS_DIR / f"{name}.pkl")
    return models


def check_inference(model, train):
    """Batched prediction must equal a full variable-elimination query on the same rows."""
    rows = train.sample(200, random_state=SEED)
    fast = model.predict_fire(rows)
    ve = [model.query([TARGET], {v: r[v] for v in DISCRETE_FEATURES})[1]
          for _, r in rows.iterrows()]
    assert np.allclose(fast, ve, atol=1e-12), f"{model.name}: VE and batched predictions differ"


def main():
    train = pd.read_parquet(TRAIN_PATH)
    states = load_states()
    models = fit_all(train, states)

    lines = ["# Sanity and diagnostic queries (variable elimination, unobserved variables summed out)\n"]
    w = train["weight"]
    lines.append(f"Training fire rate (weighted, = full 2015-2022 table): "
                 f"{(w * train[TARGET]).sum() / w.sum():.6f}\n")

    lines.append("## Sanity: P(fire) on a hot/dry/windy vs a cool/humid/calm day\n")
    lines.append("Other variables are summed out. `P(fire)` = the model's overall fire rate, "
                 "which should match the training rate above.\n")
    lines.append("| model | P(fire) | hot/dry/windy | cool/humid/calm | ratio | higher when hot? |")
    lines.append("|---|---|---|---|---|---|")
    for name, m in models.items():
        check_inference(m, train)
        overall = m.query([TARGET])[1]
        hot = m.query([TARGET], HOT_DRY_WINDY)[1]
        cool = m.query([TARGET], COOL_HUMID_CALM)[1]
        lines.append(f"| {name} | {overall:.6f} | {hot:.6f} | {cool:.6f} | {hot / cool:.1f}x | "
                     f"{'PASS' if hot > cool else 'FAIL'} |")

    lines.append("\n## Diagnostic: P(variable | fire = 1) vs P(variable)  (dirichlet models)\n")
    lines.append("`data` columns are the weighted training frequencies, for comparison.\n")
    fires = train[train[TARGET] == 1]
    for v in DIAGNOSTIC_VARS:
        cols = {"data P(.|fire)": fires[v].value_counts(normalize=True)}
        for s in FITTED_STRUCTURES:
            cols[f"{s} P(.|fire)"] = models[f"{s}_dirichlet"].query([v], {TARGET: 1})
        cols["data P(.)"] = w.groupby(train[v], observed=False).sum() / w.sum()
        cols["expert P(.)"] = models["expert_dirichlet"].query([v])
        tab = pd.DataFrame(cols).reindex(states[v])
        tab.index.name = v
        lines.append(tab.to_markdown(floatfmt=".3f") + "\n")

    text = "\n".join(lines)
    (RESULTS_DIR / "sanity_queries.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
