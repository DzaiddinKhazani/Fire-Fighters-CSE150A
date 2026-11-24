"""
independence_tests.py -- check the conditional independences a DAG implies (Phase 5).

Usage:
    python independence_tests.py                 # expert DAG (EXPERT_EDGES)
    python independence_tests.py --edges NAME    # any edge list defined in config.py

Outputs: results/independence_tests_<structure>.md and .csv

What is tested
--------------
1. Local Markov property of `fire`. Fire has no children, so the DAG claims
       fire  _|_  X  |  parents(fire)      for every feature X that is not a parent.
   These are the edges the DAG deliberately leaves out.
2. Reference tests: for each parent P,  fire _|_ P | (other parents).  These edges ARE
   in the DAG, so they should fail; their effect size is the yardstick for (1).
3. Weather and root-node claims (expert DAG only):
       wind _|_ tmax | elev,   wind _|_ elev,   wind _|_ burn_hist,   elev _|_ burn_hist.

Method
------
Stratified G-test (likelihood-ratio test of conditional independence) on the FULL
2015-2022 training data (13.7M cell-days, unweighted). The subsampled training file is
not used: its 1/0.02 weights would inflate G fiftyfold.

    G   = 2 * sum_{x,y,z} N(x,y,z) * ln[ N(x,y,z) N(z) / (N(x,z) N(y,z)) ]
    df  = sum_z (rows_z - 1)(cols_z - 1), counting only levels observed in stratum z
    CMI_adj = max(G - E0[G], 0) / (2N)   (bias-corrected conditional mutual information, nats)


Rows are not independent: cells on the same day share weather, and static variables
(elev, slope, vegetation) repeat every day. Tests between static variables therefore use
one row per cell-year (values on Jan 1). The cell-day p-values are still optimistic,
which is another reason to read the CMI column rather than p alone.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2

import config
from config import DISCRETE_FEATURES, RESULTS_DIR, TARGET, TRAIN_YEARS
from discretize_and_split import discretize, load

ALPHA = getattr(config, "INDEPENDENCE_ALPHA", 0.05)
VARS = DISCRETE_FEATURES + [TARGET]
STATIC = ["elev", "slope", "veg_type", "veg_cover", "veg_height", "burn_hist"]


# Counting: one joint contingency table over all variables, built a year at a time
# (counts add up across years, so the full 13.7M-row table never sits in memory)
def codes(df, v):
    return (df[v].cat.codes if hasattr(df[v], "cat") else df[v]).to_numpy().astype(np.int64)


def joint_counts(years):
    cards, day_table, cy_table = None, None, None
    for y in years:
        d = discretize(load([y]))
        if cards is None:
            cards = {v: (len(d[v].cat.categories) if hasattr(d[v], "cat") else 2) for v in VARS}
            day_table = np.zeros([cards[v] for v in VARS], dtype=np.int64)
            cy_table = np.zeros([cards[v] for v in STATIC], dtype=np.int64)
        idx = np.ravel_multi_index([codes(d, v) for v in VARS], day_table.shape)
        day_table += np.bincount(idx, minlength=day_table.size).reshape(day_table.shape)

        first = d[d["date"] == d.groupby("cell_id")["date"].transform("min")]   # Jan 1 per cell
        idx = np.ravel_multi_index([codes(first, v) for v in STATIC], cy_table.shape)
        cy_table += np.bincount(idx, minlength=cy_table.size).reshape(cy_table.shape)
        print(f"  {y}: {len(d):,} cell-days, {int(d[TARGET].sum())} fires")
    return day_table, VARS, cy_table, STATIC


def marginal(table, axes_names, keep):
    """Sum out everything except `keep`, returned in the order given by `keep`."""
    drop = tuple(i for i, n in enumerate(axes_names) if n not in keep)
    m = table.sum(axis=drop)
    remaining = [n for n in axes_names if n in keep]
    return np.transpose(m, [remaining.index(k) for k in keep])


# Stratified G-test of X _|_ Y | Z
def g_test(table, axes_names, x, y, z=(), n_sims=0):
    m = marginal(table, axes_names, [x, y, *z]).astype(float)
    nx_, ny_ = m.shape[:2]
    m = m.reshape(nx_, ny_, -1)                
    n_z = m.sum(axis=(0, 1))
    n_xz = m.sum(axis=1)                           
    n_yz = m.sum(axis=0)               
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = m * n_z[None, None, :] / (n_xz[:, None, :] * n_yz[None, :, :])
        terms = np.where(m > 0, m * np.log(ratio), 0.0)
    g = 2.0 * terms.sum()
    df = int((((n_xz > 0).sum(0) - 1).clip(0) * ((n_yz > 0).sum(0) - 1).clip(0)).sum())
    n = m.sum()
    out = dict(G=g, df=df, log10_p=chi2.logsf(g, df) / np.log(10) if df > 0 else 0.0,
               G_null_mean=float(df), N=int(n), strata_used=int((n_z > 0).sum()), p_mc=np.nan)
    if x == TARGET and n_sims:
        out["p_mc"], out["G_null_mean"] = monte_carlo_p(m, g, n_sims)
    out["CMI_adj"] = max(g - out["G_null_mean"], 0.0) / (2 * n)
    return out


def monte_carlo_p(m, g_obs, n_sims, seed=config.SEED):
    """Exact-margins null for fire (axis 0, binary) vs Y (axis 1) within strata (axis 2).

    Under H0 the fires in stratum z are a random draw (without replacement) from that
    stratum's cell-days, so fire counts per Y level are multivariate hypergeometric with
    colours n(y, z) and sample size n(fire=1, z). Strata without fires add 0 to G."""
    rng = np.random.default_rng(seed)
    n_yz, n_fz, n_z = m.sum(axis=0), m.sum(axis=1), m.sum(axis=(0, 1))
    g_sim = np.zeros(n_sims)
    for z in np.flatnonzero(m[1].sum(axis=0) > 0):
        colours = n_yz[:, z].astype(np.int64)
        if (colours > 0).sum() < 2:
            continue
        f1 = rng.multivariate_hypergeometric(colours, int(n_fz[1, z]), size=n_sims)   # (S, Y)
        for cnt, n_f in ((f1, n_fz[1, z]), (colours[None, :] - f1, n_fz[0, z])):
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(cnt > 0, cnt * np.log(cnt * n_z[z] / (n_f * colours[None, :])), 0.0)
            g_sim += 2.0 * t.sum(axis=1)
    return (1 + np.sum(g_sim >= g_obs - 1e-9)) / (n_sims + 1), float(g_sim.mean())


def fmt_p(log10_p):
    return "1" if log10_p > -1e-9 else (f"{10 ** log10_p:.3g}" if log10_p > -300
                                        else f"1e{log10_p:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default="EXPERT_EDGES",
                    help="name of an edge list in config.py (default EXPERT_EDGES)")
    ap.add_argument("--sims", type=int, default=2000, help="Monte Carlo replicates")
    args = ap.parse_args()
    edges = getattr(config, args.edges)
    parents = sorted({a for a, b in edges if b == TARGET})
    others = [v for v in DISCRETE_FEATURES if v not in parents]
    name = args.edges.lower().replace("_edges", "")

    print(f"Counting {list(TRAIN_YEARS)[0]}-{list(TRAIN_YEARS)[-1]} (full, unweighted)...")
    day, day_axes, cy, cy_axes = joint_counts(TRAIN_YEARS)

    rows = []
    for x in others:                                   # 1. omitted edges into fire
        rows.append(dict(group="omitted edge into fire", claim=f"fire ⊥ {x} | parents",
                         given=", ".join(parents), rows="cell-day",
                         in_dag=False, **g_test(day, day_axes, TARGET, x, parents, args.sims)))
    for p in parents:                                  # 2. included edges (reference)
        rest = [q for q in parents if q != p]
        rows.append(dict(group="edge in DAG (reference)", claim=f"fire ⊥ {p} | other parents",
                         given=", ".join(rest), rows="cell-day",
                         in_dag=True, **g_test(day, day_axes, TARGET, p, rest, args.sims)))
    if args.edges == "EXPERT_EDGES":                   # 3. weather / root-node claims
        extra = [("wind", "tmax", ["elev"], "cell-day", day, day_axes),
                 ("wind", "elev", [], "cell-day", day, day_axes),
                 ("wind", "burn_hist", [], "cell-day", day, day_axes),
                 ("elev", "burn_hist", [], "cell-year", cy, cy_axes)]
        for x, y, z, unit, t, ax in extra:
            rows.append(dict(group="weather / root-node claim",
                             claim=f"{x} ⊥ {y}" + (f" | {', '.join(z)}" if z else ""),
                             given=", ".join(z) or "-", rows=unit, in_dag=False,
                             **g_test(t, ax, x, y, z)))

    res = pd.DataFrame(rows)
    # Bonferroni over every test in the table
    m = len(res)                                       
    alpha_b = ALPHA / m
    # p used for decisions: Monte Carlo where computed (fire tests), chi-square otherwise
    res["p_used"] = np.where(res["p_mc"].notna(), res["p_mc"], 10 ** res["log10_p"])
    ref = res[res.in_dag & (res.p_used < alpha_b)]
    weakest = ref["CMI_adj"].min() if len(ref) else res.loc[res.in_dag, "CMI_adj"].max()
    weakest_name = (ref if len(ref) else res[res.in_dag]).sort_values("CMI_adj").iloc[0]["claim"]
    res["CMI_vs_weakest_edge"] = res["CMI_adj"] / weakest
    res["p_chi2"] = res["log10_p"].map(fmt_p)
    res["p_mc_str"] = res["p_mc"].map(lambda v: "-" if np.isnan(v) else f"{v:.3g}")

    def verdict(r):
        sig = r.p_used < alpha_b
        if r.in_dag:
            return "edge supported" if sig else "weak: not significant given the other parents"
        if not sig:
            return "independence holds"
        return ("rejected: dependence as large as a DAG edge" if r.CMI_vs_weakest_edge >= 0.5
                else "rejected, but tiny effect")
    res["verdict"] = res.apply(verdict, axis=1)

    out = RESULTS_DIR / f"independence_tests_{name}.csv"
    res.drop(columns=["log10_p", "p_mc_str"]).to_csv(out, index=False)

    cols = ["claim", "rows", "G", "df", "G_null_mean", "p_chi2", "p_mc", "CMI_adj", "CMI_vs_weakest_edge", "verdict"]
    show = res.copy()
    show["p_mc"] = show["p_mc_str"]
    show["G"] = show["G"].map(lambda v: f"{v:,.1f}")
    show["G_null_mean"] = show["G_null_mean"].map(lambda v: f"{v:,.1f}")
    show["CMI_adj"] = show["CMI_adj"].map(lambda v: f"{v:.2e}")
    show["CMI_vs_weakest_edge"] = show["CMI_vs_weakest_edge"].map(lambda v: f"{v:.2f}")
    n_day = int(day.sum()); n_fire = int(marginal(day, day_axes, [TARGET])[1])
    md = [f"# Conditional independence tests: {name} DAG",
          "",
          f"Stratified G-tests on the full {list(TRAIN_YEARS)[0]}-{list(TRAIN_YEARS)[-1]} training data "
          f"({n_day:,} cell-days, {n_fire:,} fires, unweighted). Parents of fire: {', '.join(parents)}.",
          f"Bonferroni threshold: alpha = {ALPHA} / {m} tests = {alpha_b:.2g}. Decisions use `p_mc` "
          f"(exact-margins Monte Carlo, {args.sims} replicates) where shown, else `p_chi2`.",
          f"`G_null_mean` = mean of G under independence (simulated where p_mc is shown, else df). "
          f"`CMI_adj` = (G - G_null_mean) / 2N, the bias-corrected conditional mutual information in "
          f"nats. `CMI_vs_weakest_edge` divides by the "
          f"CMI_adj of the weakest significant edge into fire ({weakest_name}, {weakest:.2e}).",
          "A rejected claim is called 'tiny' when it carries < 0.5x the information of that edge.",
          "Cell-day rows are not independent (shared daily weather, repeated static values), "
          "so the root-node p-values are badly optimistic; read CMI_adj for those.", ""]
    for g in res["group"].unique():
        md += [f"## {g}", "", show.loc[res.group == g, cols].to_markdown(index=False), ""]
    (RESULTS_DIR / f"independence_tests_{name}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
