"""
fit_latent.py -- Model D: expert DAG + latent "danger" node, learned with EM.

The latent node is a hidden fire-weather regime (think Santa Ana / offshore vs onshore
marine layer). It is a parent of tmax, rhmin, wind and fire, and replaces the direct
weather -> fire edges of the expert DAG (config.LATENT_DAG_EDGES). One model is fitted
for each number of hidden states in config.LATENT_CARDS.

EM (weighted, because training rows carry the subsampling weight):
  E-step  q_i(h) = P(danger = h | all observed variables of row i)
          Only the CPTs that mention `danger` depend on h, so
          log q_i(h) = log P(h) + log P(tmax_i | elev_i, h) + log P(rhmin_i | tmax_i, wind_i, h)
                       + log P(wind_i | h) + log P(fire_i | veg_type_i, burn_hist_i, slope_i, h) + const
  M-step  refit those CPTs from soft counts: each row is copied once per state h with
          weight w_i * q_i(h), then bn.fit_cpts is applied (marginal-centred Dirichlet
          smoothing, as in model C, so this is MAP-EM).
  CPTs that do not involve `danger` are fitted once from the observed data.
  Several random starts per state count (random initial CPTs for the latent part); the start
  with the highest final log-likelihood is kept.

Usage
  python fit_latent.py                 # fit every state count, then evaluate (~5 min)
  python fit_latent.py --k 3           # fit one state count only
  python fit_latent.py --report-only   # evaluate models already saved

Outputs
  results/models/latent_k{k}.pkl          fitted CPTs
  results/latent_em.md                    convergence, state profiles, test metrics
  results/latent_em_log.csv               log-likelihood per iteration, every start
  results/figures/em_convergence.png
  results/figures/latent_states_by_month.png
"""
import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bn import BayesNet, codes_of, fit_cpts, parents_from_edges
from config import (DISCRETE_FEATURES, ESTIMATORS, FIGURES_DIR, LATENT, LATENT_CARDS,
                    LATENT_DAG_EDGES, MODELS_DIR, RESULTS_DIR, SEED, SPLIT_INFO_PATH,
                    TARGET, TEST_PATH, TRAIN_PATH)
import evaluate

N_STARTS = 3
MAX_ITER = 400
TOL = 1e-9          # stop when the relative change in log-likelihood falls below this
ESS = ESTIMATORS["dirichlet"]["equivalent_sample_size"]
VARS = DISCRETE_FEATURES + [TARGET]


def load_states():
    states = json.loads(SPLIT_INFO_PATH.read_text())["states"]
    states[TARGET] = [0, 1]
    return states


class LatentModel:
    def __init__(self, k, states):
        self.k = k
        self.states = dict(states, **{LATENT: list(range(k))})
        self.parents = parents_from_edges(LATENT_DAG_EDGES, VARS + [LATENT])
        self.latent_vars = [LATENT] + [v for v in VARS if LATENT in self.parents[v]]
        self.cpts = {}

    # helpers
    def codes(self, df):
        return {v: codes_of(df[v], self.states[v]) for v in VARS if v in df}

    def log_terms(self, c, skip=()):
        """(n, k) array: sum over latent-involved CPTs of log P(var | parents) for each h."""
        n = len(next(iter(c.values())))
        out = np.tile(np.log(self.cpts[LATENT]), (n, 1))
        for v in self.latent_vars[1:]:
            if v in skip:
                continue
            axes = self.parents[v] + [v]
            for h in range(self.k):
                idx = tuple(np.full(n, h) if a == LATENT else c[a] for a in axes)
                out[:, h] += np.log(self.cpts[v][idx])
        return out

    # EM
    def m_step(self, df, w, q):
        rep = df.loc[np.repeat(np.arange(len(df)), self.k), [v for v in VARS]].reset_index(drop=True)
        rep[LATENT] = np.tile(np.arange(self.k), len(df))
        rep["weight"] = (w[:, None] * q).ravel()
        sub = {v: self.parents[v] for v in self.latent_vars}
        self.cpts.update(fit_cpts(rep, sub, self.states, estimator="dirichlet", ess=ESS))

    def fit(self, train, seed):
        rng = np.random.default_rng(seed)
        w = train["weight"].to_numpy(np.float64)
        c = self.codes(train)
        observed = {v: self.parents[v] for v in VARS if v not in self.latent_vars}
        self.cpts = fit_cpts(train, observed, self.states, estimator="dirichlet", ess=ESS)
        const = sum(np.sum(w * np.log(self.cpts[v][tuple(c[a] for a in observed[v] + [v])]))
                    for v in observed)                       # log-lik of the non-latent CPTs
        for v in self.latent_vars:                           # random initial CPTs
            shape = tuple(len(self.states[a]) for a in self.parents[v] + [v])
            self.cpts[v] = rng.dirichlet(np.ones(shape[-1]), size=shape[:-1])
        history = []
        for it in range(MAX_ITER):
            lt = self.log_terms(c)                           # E-step
            m = lt.max(1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(lt - m).sum(1))
            q = np.exp(lt - lse[:, None])
            history.append(float(np.sum(w * lse) + const))
            if it > 1 and abs(history[-1] - history[-2]) < TOL * abs(history[-2]):
                break
            self.m_step(train, w, q)                         # M-step
        self.history = history
        self.q_train = q
        return history[-1]

    def posterior(self, df):
        """q(h) = P(danger = h | every observed variable), used for the state profiles."""
        lt = self.log_terms(self.codes(df))
        q = np.exp(lt - lt.max(1, keepdims=True))
        return q / q.sum(1, keepdims=True)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        m = cls.__new__(cls)
        m.k, m.parents, m.states, m.cpts, m.history = d["k"], d["parents"], d["states"], d["cpts"], d["history"]
        m.latent_vars = [LATENT] + [v for v in VARS if LATENT in m.parents[v]]
        return m

    # inference
    def predict_fire(self, df):
        """P(fire = 1 | all 9 features): sum the latent state out, fire unobserved."""
        c = self.codes(df)
        lt = self.log_terms(c, skip=(TARGET,))               # P(h) * P(weather | ..., h)
        post = np.exp(lt - lt.max(1, keepdims=True))
        post /= post.sum(1, keepdims=True)                   # P(h | features)
        axes = self.parents[TARGET]
        n = len(df)
        p_fire = np.stack([self.cpts[TARGET][tuple(np.full(n, h) if a == LATENT else c[a]
                                                   for a in axes)][:, 1]
                           for h in range(self.k)], axis=1)
        return (post * p_fire).sum(1)

    def n_params(self):
        return int(sum((len(self.states[v]) - 1) * np.prod([len(self.states[p]) for p in self.parents[v]])
                       for v in self.latent_vars))

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(k=self.k, parents=self.parents, states=self.states,
                             cpts=self.cpts, history=self.history,
                             all_starts=getattr(self, "all_starts", {})), f)


# reporting
def state_profiles(model, train):
    """Weighted share of each weather bin, the fire rate and the season in each hidden state."""
    w = train["weight"].to_numpy(np.float64)
    q = model.q_train * w[:, None]
    tot = q.sum(0)
    rows = {"share of cell-days": tot / tot.sum()}
    fire = train[TARGET].to_numpy()
    rows["fire rate (per 10k)"] = 1e4 * (q * fire[:, None]).sum(0) / tot
    for v, levels in (("tmax", ["hot", "extreme"]), ("rhmin", ["crit_dry", "dry"]),
                      ("wind", ["strong"]), ("tmax", ["cool"]), ("rhmin", ["humid"])):
        m = train[v].isin(levels).to_numpy()
        rows[f"P({v} in {'/'.join(levels)})"] = (q * m[:, None]).sum(0) / tot
    month = pd.to_datetime(train["date"]).dt.month.to_numpy()
    for name, months in (("Sep-Dec", [9, 10, 11, 12]), ("Jun-Aug", [6, 7, 8]), ("Jan-May", [1, 2, 3, 4, 5])):
        m = np.isin(month, months)
        rows[f"share in {name}"] = (q * m[:, None]).sum(0) / tot
    df = pd.DataFrame(rows).T
    df.columns = [f"state {h}" for h in range(model.k)]
    return df


def plot_convergence(logs):
    fig, axes = plt.subplots(1, len(LATENT_CARDS), figsize=(4 * len(LATENT_CARDS), 3.2), sharey=False)
    for ax, k in zip(np.atleast_1d(axes), LATENT_CARDS):
        for s, h in logs[k].items():
            ax.plot(np.arange(1, len(h) + 1), np.array(h) / 1e6, lw=1.5, label=f"start {s}")
        ax.set_title(f"{k} hidden states")
        ax.set_xlabel("EM iteration")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("weighted train log-likelihood (x1e6)")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "em_convergence.png", dpi=150)
    plt.close(fig)


def plot_months(model, train):
    w = train["weight"].to_numpy(np.float64)
    month = pd.to_datetime(train["date"]).dt.month.to_numpy()
    q = model.q_train * w[:, None]
    share = np.array([q[month == mth].sum(0) / q[month == mth].sum() for mth in range(1, 13)])
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bottom = np.zeros(12)
    for h in range(model.k):
        ax.bar(np.arange(1, 13), share[:, h], bottom=bottom, label=f"state {h}")
        bottom += share[:, h]
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("share of cell-days")
    ax.set_title(f"Hidden weather state by month ({model.k} states)")
    ax.legend(fontsize=8, ncol=model.k, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "latent_states_by_month.png", dpi=150)
    plt.close(fig)


def fit_k(k, train, states):
    """Fit one state count from N_STARTS random starts; keep and save the best."""
    t0, best, starts = time.time(), None, {}
    for s in range(N_STARTS):
        m = LatentModel(k, states)
        ll = m.fit(train, seed=SEED + 100 * k + s)
        starts[s] = m.history
        print(f"k={k} start={s}: {len(m.history)} iterations, log-lik {ll:,.1f} ({time.time() - t0:.0f}s)")
        if best is None or ll > best.history[-1]:
            best = m
    best.all_starts = starts
    best.save(MODELS_DIR / f"latent_k{k}.pkl")
    return best


def report(train, test):
    best = {k: LatentModel.load(MODELS_DIR / f"latent_k{k}.pkl") for k in LATENT_CARDS}
    logs = {}
    for k in LATENT_CARDS:
        with open(MODELS_DIR / f"latent_k{k}.pkl", "rb") as f:
            logs[k] = pickle.load(f)["all_starts"]
        best[k].q_train = best[k].posterior(train)
    n_eff = int(json.loads(SPLIT_INFO_PATH.read_text())["train_rows_before_subsample"])
    summary = []
    for k in LATENT_CARDS:
        b, finals = best[k], [h[-1] for h in logs[k].values()]
        summary.append(dict(states=k, starts=len(finals), iterations=len(b.history),
                            converged=len(b.history) < MAX_ITER, train_loglik=b.history[-1],
                            spread_across_starts=max(finals) - min(finals),
                            latent_params=b.n_params(),
                            BIC=-2 * b.history[-1] + b.n_params() * np.log(n_eff)))
    pd.DataFrame([dict(states=k, start=s, iteration=i + 1, loglik=v)
                  for k in logs for s in logs[k] for i, v in enumerate(logs[k][s])]
                 ).to_csv(RESULTS_DIR / "latent_em_log.csv", index=False)
    plot_convergence(logs)
    # ---- test-set evaluation, same day-level bootstrap as evaluate.py (paired: same resamples)
    preds = {"naive_bayes_dirichlet": BayesNet.load(MODELS_DIR / "naive_bayes_dirichlet.pkl").predict_fire(test),
             "expert_dirichlet": BayesNet.load(MODELS_DIR / "expert_dirichlet.pkl").predict_fire(test)}
    for k in LATENT_CARDS:
        preds[f"latent_k{k}"] = best[k].predict_fire(test)
    point, reps = evaluate.bootstrap(preds, test)

    def fmt(v, k):
        return f"{v:.3f}" if k == "roc_auc" else f"{v:.3e}"
    met_rows, diff_rows = [], []
    for m in preds:
        met_rows.append({"model": m, **{k: f"{fmt(point[m][k], k)} [{fmt(np.percentile(reps[m][k], 2.5), k)}, "
                                           f"{fmt(np.percentile(reps[m][k], 97.5), k)}]"
                                        for k in evaluate.METRICS}})
        if m.startswith("latent"):
            d = {k: reps[m][k] - reps["expert_dirichlet"][k] for k in evaluate.METRICS}
            diff_rows.append({"model": m, **{k: f"{fmt(point[m][k] - point['expert_dirichlet'][k], k)} "
                                                f"[{fmt(np.percentile(d[k], 2.5), k)}, {fmt(np.percentile(d[k], 97.5), k)}]"
                                             for k in evaluate.METRICS}})

    best_k = min(LATENT_CARDS, key=lambda k: point[f"latent_k{k}"]["log_loss"])
    plot_months(best[best_k], train)
    sample_rate = float(np.mean(preds[f"latent_k{best_k}"]))

    md = ["# Model D: expert DAG + latent weather regime, fitted with EM", "",
          f"Weighted EM on {len(train):,} training rows, several random starts per state count (`starts`), "
          f"random initial CPTs, stop at relative log-likelihood change < {TOL:g} or {MAX_ITER} iterations. "
          "Marginal-centred Dirichlet smoothing (ESS = %g) in every M-step (MAP-EM). " % ESS,
          "BIC uses N = full training cell-days and counts only the CPTs that involve the latent node "
          "(the rest are identical across models).", "",
          "## Convergence", "", pd.DataFrame(summary).to_markdown(index=False, floatfmt=",.1f"), "",
          "`spread_across_starts` = best minus worst final log-likelihood over the random starts "
          "(0 means every start reached the same optimum). `converged` = False means the "
          f"{MAX_ITER}-iteration cap was hit while the log-likelihood was still rising slowly.", "",
          "## Test set (2023-2024), 95% day-level bootstrap CIs", "",
          pd.DataFrame(met_rows).to_markdown(index=False), "",
          "## Difference from the expert DAG (C, dirichlet), paired bootstrap", "",
          "Better = negative for log_loss/brier, positive for pr_auc/roc_auc.", "",
          pd.DataFrame(diff_rows).to_markdown(index=False), "",
          f"Mean predicted P(fire) on the test set, latent_k{best_k}: {sample_rate:.6f} "
          f"(observed test rate {test[TARGET].mean():.6f}).", ""]
    for k in LATENT_CARDS:
        md += [f"## What the hidden states look like ({k} states)", "",
               "Responsibility-weighted shares over the training data (weights undo the subsampling).", "",
               state_profiles(best[k], train).to_markdown(floatfmt=".3f"), ""]
    (RESULTS_DIR / "latent_em.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))



def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="*", help="fit only these state counts (default: all)")
    ap.add_argument("--report-only", action="store_true", help="skip fitting; evaluate saved models")
    args = ap.parse_args()
    train = pd.read_parquet(TRAIN_PATH)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        for k in (args.k or LATENT_CARDS):
            fit_k(k, train, load_states())
    if args.report_only or not args.k:
        report(train, pd.read_parquet(TEST_PATH))


if __name__ == "__main__":
    main()
