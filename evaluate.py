"""
evaluate.py -- todo Phase 5 (prediction + metrics).

Scores every fitted Bayesian network (results/models/*.pkl, from fit_models.py) and two
baselines on the full 2023-2024 test set:
  * constant: the 2015-2022 fire rate for every cell-day
  * logistic: logistic regression on one-hot features, trained with the same weights

Metrics: log-loss, Brier score, PR-AUC (average precision), ROC-AUC, each with a 95%
bootstrap CI. The bootstrap resamples whole test DAYS: every cell on a day shares the same
weather, so rows are not independent and a row-level bootstrap would give CIs that are too
narrow. It also reports each network minus Naive Bayes (same estimator), with a CI, which is
the comparison the proposal's hypothesis is about.

Writes
  results/test_predictions.parquet   one row per test cell-day, one column per model (Phase 6 uses this)
  results/metrics.csv, results/metrics.md
  results/figures/calibration.png, results/figures/pr_curves.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from bn import BayesNet
from config import (BOOTSTRAP_REPS, CALIBRATION_BINS, DISCRETE_FEATURES, ESTIMATORS,
                    FIGURES_DIR, FITTED_STRUCTURES, MODELS_DIR, PREDICTIONS_PATH, PROB_CLIP,
                    RESULTS_DIR, SEED, TARGET, TEST_PATH, TRAIN_PATH)

METRICS = ["log_loss", "brier", "pr_auc", "roc_auc"]
LOWER_IS_BETTER = {"log_loss": True, "brier": True, "pr_auc": False, "roc_auc": False}

# figure styling (reference palette, categorical slots 1-4 in fixed order)
SERIES = {"logistic": ("#2a78d6", "o"), "naive_bayes_dirichlet": ("#eb6834", "s"),
          "tan_dirichlet": ("#1baf7a", "^"), "expert_dirichlet": ("#eda100", "D")}
LABEL = {"logistic": "Logistic regression", "naive_bayes_dirichlet": "A. Naive Bayes",
         "tan_dirichlet": "B. TAN", "expert_dirichlet": "C. Expert DAG"}
INK, MUTED, GRID = "#52514e", "#898781", "#e1e0d9"


# ------------------------------------------------------------------ predictions
def predict_all(train, test):
    preds = {}
    w = train["weight"].to_numpy()
    preds["constant"] = np.full(len(test), np.sum(w * train[TARGET]) / w.sum())

    enc = OneHotEncoder(handle_unknown="ignore")
    X_train = enc.fit_transform(train[DISCRETE_FEATURES].astype(str))
    # weights rescaled to mean 1: with raw weights (50 per non-fire row) the default lbfgs
    # solver stops after ~11 iterations, far from the optimum (test ROC-AUC 0.71 vs 0.82).
    lr = LogisticRegression(solver="newton-cholesky", max_iter=1000)
    lr.fit(X_train, train[TARGET], sample_weight=w / w.mean())
    preds["logistic"] = lr.predict_proba(enc.transform(test[DISCRETE_FEATURES].astype(str)))[:, 1]

    for s in FITTED_STRUCTURES:
        for est in ESTIMATORS:
            name = f"{s}_{est}"
            preds[name] = BayesNet.load(MODELS_DIR / f"{name}.pkl").predict_fire(test)
    return preds


# ------------------------------------------------------------------ metrics
def config_tables(test):
    """
    Every model's prediction depends only on the 9 discrete features, so the test set
    collapses to (day x feature-combination) counts. The bootstrap then works on those
    counts instead of 3.4M rows.
    """
    cfg = test.groupby(DISCRETE_FEATURES, observed=True, sort=False).ngroup().to_numpy()
    day = pd.factorize(test["date"])[0]
    y = test[TARGET].to_numpy()
    n_day, n_cfg = day.max() + 1, cfg.max() + 1
    pos = sparse.csr_matrix((y.astype(float), (day, cfg)), shape=(n_day, n_cfg))
    neg = sparse.csr_matrix(((1 - y).astype(float), (day, cfg)), shape=(n_day, n_cfg))
    first_row = pd.Series(np.arange(len(cfg))).groupby(cfg).first().to_numpy()
    return pos, neg, first_row


def metrics_from_counts(p, P, N):
    """
    p: prediction per feature combination (C,). P, N: fire / no-fire counts (R, C), one row
    per bootstrap replicate (or a single row of the real counts). Returns {metric: (R,)}.
    """
    pc = np.clip(p, *PROB_CLIP)
    n = P.sum(1) + N.sum(1)
    out = {"log_loss": -(P @ np.log(pc) + N @ np.log(1 - pc)) / n,
           "brier": (P @ (1 - p) ** 2 + N @ p ** 2) / n}

    # rank-based metrics: sort combinations by score, merge ties into one threshold
    order = np.argsort(-p, kind="stable")
    _, grp = np.unique(-p[order], return_inverse=True)
    n_grp = grp.max() + 1
    S = sparse.csr_matrix((np.ones(len(order)), (order, grp)), shape=(len(p), n_grp))
    Pg, Ng = P @ S, N @ S                                     # (R, groups), highest score first
    Pg, Ng = np.asarray(Pg), np.asarray(Ng)
    tp, fp = np.cumsum(Pg, 1), np.cumsum(Ng, 1)
    TP, FP = tp[:, -1:], fp[:, -1:]
    precision = tp / np.maximum(tp + fp, 1e-12)
    out["pr_auc"] = (Pg / TP * precision).sum(1)              # average precision, as in sklearn
    out["roc_auc"] = ((Ng * (tp - Pg / 2)).sum(1) / (TP * FP).ravel())
    return out


def bootstrap(preds, test):
    pos, neg, first_row = config_tables(test)
    rng = np.random.default_rng(SEED)
    n_day = pos.shape[0]
    point, reps = {}, {m: {k: [] for k in METRICS} for m in preds}
    for m, p in preds.items():
        point[m] = {k: v[0] for k, v in metrics_from_counts(
            p[first_row], pos.sum(0).A, neg.sum(0).A).items()}
    for start in range(0, BOOTSTRAP_REPS, 100):
        r = min(100, BOOTSTRAP_REPS - start)
        W = rng.multinomial(n_day, np.full(n_day, 1 / n_day), size=r).astype(float)
        P, N = (pos.T @ W.T).T, (neg.T @ W.T).T     # counts in each replicate, (r, C)
        for m, p in preds.items():
            for k, v in metrics_from_counts(p[first_row], P, N).items():
                reps[m][k].append(v)
    reps = {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in reps.items()}
    return point, reps


def check_against_sklearn(preds, test, point):
    y = test[TARGET].to_numpy()
    for m in ("logistic", "expert_dirichlet"):
        p = preds[m]
        ref = {"log_loss": log_loss(y, np.clip(p, *PROB_CLIP)), "brier": np.mean((p - y) ** 2),
               "pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p)}
        for k, v in ref.items():
            assert np.isclose(point[m][k], v, rtol=1e-6), f"{m} {k}: {point[m][k]} vs sklearn {v}"


def results_table(point, reps):
    rows = []
    for m in point:
        s, est = (m.rsplit("_", 1) if m not in ("constant", "logistic") else (m, "-"))
        row = {"model": m, "structure": s, "estimator": est}
        for k in METRICS:
            lo, hi = np.percentile(reps[m][k], [2.5, 97.5])
            row.update({k: point[m][k], f"{k}_lo": lo, f"{k}_hi": hi})
            base = f"naive_bayes_{est}"
            if s in FITTED_STRUCTURES and s != "naive_bayes":
                d = reps[m][k] - reps[base][k]
                row.update({f"{k}_vs_nb": point[m][k] - point[base][k],
                            f"{k}_vs_nb_lo": np.percentile(d, 2.5),
                            f"{k}_vs_nb_hi": np.percentile(d, 97.5)})
        rows.append(row)
    return pd.DataFrame(rows)


def markdown(df, test):
    fmt = {"log_loss": "{:.3e}", "brier": "{:.3e}", "pr_auc": "{:.3e}", "roc_auc": "{:.3f}"}
    cell = lambda r, k, suf="": (fmt[k].format(r[k + suf]) + " [" + fmt[k].format(r[k + suf + "_lo"])
                                 + ", " + fmt[k].format(r[k + suf + "_hi"]) + "]")
    out = [f"# Test-set results (2023-2024: {len(test):,} cell-days, {int(test[TARGET].sum())} fires)\n",
           "95% CIs from a day-level bootstrap with " f"{BOOTSTRAP_REPS} replicates. "
           "Lower is better for log-loss and Brier, higher for PR-AUC and ROC-AUC. "
           f"PR-AUC of a random guess = fire rate = {test[TARGET].mean():.5f}.\n",
           "| model | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for _, r in df.iterrows():
        out.append(f"| {r.model} | " + " | ".join(cell(r, k) for k in METRICS) + " |")
    out += ["\n## Difference from Naive Bayes with the same estimator\n",
            "A CI that excludes 0 means the difference is unlikely to be bootstrap noise. "
            "Better = negative for log-loss/Brier, positive for PR-AUC/ROC-AUC.\n",
            "| model | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for _, r in df[df.log_loss_vs_nb.notna()].iterrows():
        out.append(f"| {r.model} | " + " | ".join(cell(r, k, "_vs_nb") for k in METRICS) + " |")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ figures
def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def plot_calibration(preds, y):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    lo, hi = 1e-6, 2e-2
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1, label="Perfect calibration")
    note = []
    for m, (c, mk) in SERIES.items():
        p = preds[m]
        edges = np.unique(np.quantile(p, np.linspace(0, 1, CALIBRATION_BINS + 1)))
        b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
        g = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
        zero = (g.y == 0).sum()
        if zero:
            note.append(f"{LABEL[m]}: {zero} bin(s) with no fires not shown")
        g = g[g.y > 0]
        ax.plot(g.p, g.y, color=c, marker=mk, ms=7, lw=2, mec="white", mew=1.2, label=LABEL[m])
    ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
    ax.set_xlabel("Mean predicted P(fire) in bin", color=INK)
    ax.set_ylabel("Observed fire rate in bin", color=INK)
    ax.set_title(f"Calibration on 2023-2024 test set ({CALIBRATION_BINS} quantile bins)",
                 color=INK, fontsize=11, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    if note:
        fig.text(0.01, 0.01, "; ".join(note), fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=150)
    plt.close(fig)


def plot_pr(preds, y):
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for m, (c, _) in SERIES.items():
        pr, rc, _ = precision_recall_curve(y, preds[m])
        ax.plot(rc, pr, color=c, lw=2, drawstyle="steps-post",
                label=f"{LABEL[m]} (AP {average_precision_score(y, preds[m]):.4f})")
    ax.axhline(y.mean(), color=MUTED, ls="--", lw=1, label=f"Constant rate ({y.mean():.5f})")
    ax.set(yscale="log", xlim=(0, 1), ylim=(y.mean() / 2, 0.2))
    ax.set_xlabel("Recall", color=INK)
    ax.set_ylabel("Precision (log scale)", color=INK)
    ax.set_title("Precision-recall on 2023-2024 test set", color=INK, fontsize=11, loc="left")
    style(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=150)
    plt.close(fig)


def main():
    train, test = pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)
    preds = predict_all(train, test)

    out = test[["cell_id", "date", TARGET]].copy()
    for m, p in preds.items():
        out[m] = p.astype("float32")
    out.to_parquet(PREDICTIONS_PATH, index=False)

    point, reps = bootstrap(preds, test)
    check_against_sklearn(preds, test, point)
    df = results_table(point, reps)
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    md = markdown(df, test)
    (RESULTS_DIR / "metrics.md").write_text(md)
    print(md)

    y = test[TARGET].to_numpy()
    plot_calibration(preds, y)
    plot_pr(preds, y)


if __name__ == "__main__":
    main()
