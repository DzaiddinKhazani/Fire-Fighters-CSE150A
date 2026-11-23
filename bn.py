"""
bn.py -- small discrete Bayesian network toolkit used by fit_models.py and evaluate.py.

  * fit_cpts:   estimate every conditional probability table (CPT) from weighted counts,
                with maximum likelihood ("mle") or Dirichlet / BDeu smoothing ("bdeu").
  * BayesNet.query:        exact inference by variable elimination (any query, any evidence).
  * BayesNet.predict_fire: P(fire = 1 | all features observed) for many rows at once.

Why weighted counts (and not "fit on the subsample, then correct the odds")?
The training set keeps every fire but only NEG_KEEP = 2% of non-fire rows, and each kept
non-fire row has weight 1/NEG_KEEP = 50. Weighted counts are therefore unbiased estimates of
the counts in the FULL 2015-2022 table, so every CPT -- including P(tmax), P(rhmin | ...)
and the other parents of fire -- describes the real data. That matters for queries that sum
variables out (e.g. the Phase 6 static query with weather marginalized out): with unweighted
counts, fire-weather would look 50x more common than it is. The predicted P(fire) needs no
separate prior correction afterwards.
"""
import pickle
import string
from functools import lru_cache

import numpy as np
import pandas as pd


def parents_from_edges(edges, variables):
    pa = {v: [] for v in variables}
    for a, b in edges:
        pa[b].append(a)
    return pa


def topological_order(parents):
    order, done = [], set()
    while len(order) < len(parents):
        for v, ps in parents.items():
            if v not in done and all(p in done for p in ps):
                order.append(v)
                done.add(v)
    return order


def fit_cpts(df, parents, states, estimator="mle", ess=10.0, weight_col="weight"):
    """
    One CPT per variable, as a numpy array with axes (parent_1, ..., parent_k, variable)
    whose last axis sums to 1.

    mle : P(x | pa) = N(x, pa) / N(pa)                      (weighted counts N)
    bdeu: P(x | pa) = (N(x, pa) + a) / (N(pa) + r * a),   a = ess / (r * q)
          r = #states of x, q = #parent configurations. ess = equivalent sample size.
    dirichlet: P(x | pa) = (N(x, pa) + ess * P(x)) / (N(pa) + ess)
          Dirichlet prior centred on the variable's overall distribution P(x): each parent
          configuration gets `ess` pseudo-observations that look like the average row.
    A parent configuration never seen in training gets a uniform row under mle and (almost)
    under bdeu -- for fire that means P(fire) = 0.5 -- but gets P(x) under dirichlet.
    The expert DAG has 343 such configurations for fire's 5 parents, which is why the
    dirichlet estimator matters there (see results/sanity_queries.md).
    """
    w = df[weight_col].to_numpy(np.float64) if weight_col else np.ones(len(df))
    cpts = {}
    for v, ps in parents.items():
        axes = ps + [v]
        shape = tuple(len(states[a]) for a in axes)
        codes = [codes_of(df[a], states[a]) for a in axes]
        flat = np.ravel_multi_index(codes, shape)
        counts = np.bincount(flat, weights=w, minlength=int(np.prod(shape))).reshape(shape)
        if estimator == "bdeu":
            r, q = shape[-1], int(np.prod(shape[:-1]))
            counts = counts + ess / (r * q)
        elif estimator == "dirichlet":
            marginal = counts.reshape(-1, shape[-1]).sum(axis=0)
            counts = counts + ess * marginal / marginal.sum()
        elif estimator != "mle":
            raise ValueError(estimator)
        totals = counts.sum(axis=-1, keepdims=True)
        cpt = np.divide(counts, totals, out=np.full(shape, 1.0 / shape[-1]), where=totals > 0)
        cpts[v] = cpt
    return cpts


def codes_of(col, state_list):
    """Integer state index for each row (works for categorical, string or 0/1 columns)."""
    if isinstance(col.dtype, pd.CategoricalDtype) and list(col.cat.categories) == list(state_list):
        return col.cat.codes.to_numpy()
    lookup = {s: i for i, s in enumerate(state_list)}
    out = col.map(lookup)
    if out.isna().any():
        raise ValueError(f"{col.name}: value outside {state_list}")
    return out.to_numpy(np.int64)


class BayesNet:
    def __init__(self, parents, states, cpts, name=""):
        self.parents, self.states, self.cpts, self.name = parents, states, cpts, name
        self.vars = topological_order(parents)
        self._query = lru_cache(maxsize=50_000)(self._query_uncached)

    # ---------------------------------------------------------------- exact inference
    def query(self, targets, evidence=None):
        """
        P(targets | evidence) by variable elimination.
        targets: list of variable names; evidence: {var: state_name}.
        Returns a pandas Series (one target) or a DataFrame-like Series with a MultiIndex.

        Each CPT is a factor. Evidence variables are fixed by slicing their factors;
        every other non-target variable is summed out. np.einsum(..., optimize="greedy")
        picks an elimination order and multiplies/sums the factors pairwise -- this is
        sum-product variable elimination. Results are cached (lru_cache), so repeated
        queries -- e.g. the same evidence on thousands of test rows -- are computed once.
        """
        targets = tuple(targets)
        ev = tuple(sorted((evidence or {}).items()))
        return self._query(targets, ev)

    def _query_uncached(self, targets, ev):
        ev = dict(ev)
        letters = {v: string.ascii_letters[i] for i, v in enumerate(self.vars)}
        operands, subs = [], []
        for v in self.vars:
            axes = self.parents[v] + [v]
            f = self.cpts[v]
            index = tuple(self.states[a].index(ev[a]) if a in ev else slice(None) for a in axes)
            f = f[index]
            kept = [a for a in axes if a not in ev]
            operands.append(f)
            subs.append("".join(letters[a] for a in kept))
        out = "".join(letters[t] for t in targets)
        table = np.einsum(",".join(subs) + "->" + out, *operands, optimize="greedy")
        table = table / table.sum()
        idx = pd.MultiIndex.from_product([self.states[t] for t in targets], names=list(targets))
        s = pd.Series(table.ravel(), index=idx)
        return s.droplevel(list(range(1, len(targets)))) if len(targets) == 1 else s

    # ---------------------------------------------------------------- fast prediction
    def predict_fire(self, df, target="fire"):
        """
        P(target = 1 | every other variable observed), vectorized over rows.

        With all features observed, variable elimination has nothing left to sum out except
        the target: P(fire | x) is proportional to the product of the CPTs that mention fire
        (fire's own CPT and its children's CPTs); every other factor cancels. This computes
        exactly that product for all rows at once, on the unique feature combinations only
        (the cache), then maps the answers back to the rows.
        """
        feats = [v for v in self.vars if v != target]
        codes = np.column_stack([codes_of(df[v], self.states[v]) for v in feats])
        uniq, inverse = np.unique(codes, axis=0, return_inverse=True)
        col = {v: uniq[:, i] for i, v in enumerate(feats)}
        n_t = len(self.states[target])
        logp = np.zeros((len(uniq), n_t))
        for v in self.vars:
            axes = self.parents[v] + [v]
            if target not in axes:
                continue
            for t in range(n_t):
                index = tuple(np.full(len(uniq), t) if a == target else col[a] for a in axes)
                with np.errstate(divide="ignore"):
                    logp[:, t] += np.log(self.cpts[v][index])
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        p /= p.sum(axis=1, keepdims=True)
        return p[inverse.ravel(), self.states[target].index(1)]

    # ---------------------------------------------------------------- persistence
    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump({"name": self.name, "parents": self.parents,
                         "states": self.states, "cpts": self.cpts}, fh)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        return cls(d["parents"], d["states"], d["cpts"], d["name"])
