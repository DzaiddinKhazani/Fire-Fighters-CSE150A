"""
structure_search.py -- H. score-based structure learning (greedy hill climbing with BIC).

Start from an empty graph. At every step try every single-edge change
(add an edge, remove an edge, reverse an edge), keep the one that raises the BIC score
the most while keeping the graph acyclic, and stop when no change helps.

BIC(G) = sum over variables X of  [ loglik(X | Pa(X)) - (log N / 2) * #free parameters ]
  loglik(X | Pa) = sum_{x, pa} N(x, pa) * log( N(x, pa) / N(pa) )     (weighted counts)
  #free parameters = (r_X - 1) * q_Pa                                  (table size)

The score splits into one term per variable ("family"), so a single edge change only
needs the one or two affected families re-scored; family scores are cached.

Weighted counts (weight 50 per kept non-fire row) make N equal to the full 2015-2022 table,
so the penalty log(N)/2 is the one the full data would get.
"""
import itertools

import networkx as nx
import numpy as np


def _codes(df, cols, states):
    return {c: df[c].cat.codes.to_numpy() if hasattr(df[c], "cat")
            else np.searchsorted(states[c], df[c].to_numpy()) for c in cols}


class BICScorer:
    def __init__(self, df, variables, states, weight_col="weight"):
        self.vars, self.states = variables, states
        self.w = df[weight_col].to_numpy(np.float64)
        self.N = self.w.sum()
        self.codes = {}
        for v in variables:
            col = df[v]
            if hasattr(col, "cat"):
                lookup = {s: i for i, s in enumerate(states[v])}
                self.codes[v] = np.array([lookup[c] for c in col.astype(object)], dtype=np.int64)
            else:
                self.codes[v] = np.asarray(col, dtype=np.int64)
        self.cache = {}

    def family(self, x, parents):
        key = (x, frozenset(parents))
        if key in self.cache:
            return self.cache[key]
        ps = sorted(parents)
        axes = ps + [x]
        shape = tuple(len(self.states[a]) for a in axes)
        flat = np.ravel_multi_index([self.codes[a] for a in axes], shape)
        n = np.bincount(flat, weights=self.w, minlength=int(np.prod(shape))).reshape(-1, shape[-1])
        npa = n.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ll = np.nansum(np.where(n > 0, n * np.log(n / npa), 0.0))
        k = (shape[-1] - 1) * int(np.prod(shape[:-1]))
        s = ll - 0.5 * np.log(self.N) * k
        self.cache[key] = s
        return s


def hill_climb(df, variables, states, max_parents=4, max_iter=500, verbose=True):
    sc = BICScorer(df, variables, states)
    parents = {v: set() for v in variables}
    g = nx.DiGraph()
    g.add_nodes_from(variables)
    score = {v: sc.family(v, ()) for v in variables}
    history = []
    for it in range(max_iter):
        best, best_op = 1e-9, None
        for a, b in itertools.permutations(variables, 2):
            if g.has_edge(a, b):
                # remove a -> b
                d = sc.family(b, parents[b] - {a}) - score[b]
                if d > best:
                    best, best_op = d, ("remove", a, b)
                # reverse a -> b  into  b -> a
                if len(parents[a]) < max_parents:
                    g.remove_edge(a, b); g.add_edge(b, a)
                    ok = nx.is_directed_acyclic_graph(g)
                    g.remove_edge(b, a); g.add_edge(a, b)
                    if ok:
                        d = (sc.family(b, parents[b] - {a}) - score[b]
                             + sc.family(a, parents[a] | {b}) - score[a])
                        if d > best:
                            best, best_op = d, ("reverse", a, b)
            elif not g.has_edge(b, a) and len(parents[b]) < max_parents:
                # add a -> b (only if no path b ~> a, which would close a cycle)
                if not nx.has_path(g, b, a):
                    d = sc.family(b, parents[b] | {a}) - score[b]
                    if d > best:
                        best, best_op = d, ("add", a, b)
        if best_op is None:
            break
        op, a, b = best_op
        if op == "add":
            g.add_edge(a, b); parents[b].add(a)
        elif op == "remove":
            g.remove_edge(a, b); parents[b].discard(a)
        else:
            g.remove_edge(a, b); g.add_edge(b, a)
            parents[b].discard(a); parents[a].add(b)
        for v in (a, b):
            score[v] = sc.family(v, parents[v])
        history.append((op, a, b, round(best, 1)))
        if verbose:
            print(f"  step {it + 1:2d}: {op:7s} {a} -> {b}   (BIC +{best:,.1f})")
    return sorted(g.edges()), history, sum(score.values())
