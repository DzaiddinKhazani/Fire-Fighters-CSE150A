import json
import subprocess

import networkx as nx
import numpy as np
import pandas as pd

from config import (DISCRETE_FEATURES, EXPERT_EDGES, FIGURES_DIR, LATENT, LATENT_CARDS,
                    LATENT_DAG_EDGES, LATENT_EDGES, NAIVE_BAYES_EDGES, NEG_KEEP,
                    RESULTS_DIR, TAN_ROOT, TARGET, TRAIN_PATH)

GROUP_COLOR = {   # fill colours by variable group, shared by every diagram
    "weather": "#FDE2C2", "terrain": "#D9EAD3", "fuel": "#CFE2F3",
    "history": "#EADCF8", "target": "#F4CCCC", "latent": "#FFFFFF",
}
GROUP = {"tmax": "weather", "rhmin": "weather", "wind": "weather",
         "elev": "terrain", "slope": "terrain",
         "veg_type": "fuel", "veg_cover": "fuel", "veg_height": "fuel",
         "burn_hist": "history", TARGET: "target", LATENT: "latent"}

# Tree Augmented Naive Bayes (Chow-Liu algorithm)

# Compute I(A; B | C)(A = variable 1, B = variable 2, C = fire)
def conditional_mi(df, a, b, c, w):
    t = pd.DataFrame({"a": df[a].cat.codes, "b": df[b].cat.codes, "c": df[c].astype(int), "w": w})
    p_abc = t.groupby(["a", "b", "c"])["w"].sum() / t["w"].sum()
    p_ac = p_abc.groupby(level=["a", "c"]).sum()
    p_bc = p_abc.groupby(level=["b", "c"]).sum()
    p_c = p_abc.groupby(level="c").sum()
    idx = p_abc.index
    num = p_abc.values * p_c.reindex(idx.get_level_values("c")).values
    den = (p_ac.reindex(pd.MultiIndex.from_arrays([idx.get_level_values("a"), idx.get_level_values("c")])).values
           * p_bc.reindex(pd.MultiIndex.from_arrays([idx.get_level_values("b"), idx.get_level_values("c")])).values)
    return float(np.sum(p_abc.values * np.log(num / den)))

# Learn the TAN structure:
#   1. score every feature pair by I(A; B | fire)
#   2. keep the maximum spanning tree over those scores
#   3. point the tree's edges away from root
#   4. add the Naive Bayes edges fire -> every feature
def learn_tan(train):
    w = train["weight"].to_numpy()
    g = nx.Graph()
    for i, a in enumerate(DISCRETE_FEATURES):
        for b in DISCRETE_FEATURES[i + 1:]:
            g.add_edge(a, b, weight=conditional_mi(train, a, b, TARGET, w))
    tree = nx.maximum_spanning_tree(g)
    directed = list(nx.bfs_edges(tree, TAN_ROOT)) # orient away from the root
    weights = {f"{a}->{b}": round(g[a][b]["weight"], 5) for a, b in directed}
    return NAIVE_BAYES_EDGES + directed, directed, weights


# Draw diagrams using DOT 
def draw(name, title, edges, highlight=(), labels=None, rankdir="TB"):
    nodes = sorted({n for e in edges for n in e})
    lines = [f'digraph "{name}" {{', f'  rankdir={rankdir}; labelloc=t; fontsize=20;',
             f'  label="{title}"; fontname="Helvetica";',
             '  node [shape=ellipse, style=filled, fontname="Helvetica", fontsize=13];',
             '  edge [color="#555555", arrowsize=0.8];']
    for n in nodes:
        extra = ', style="filled,dashed", penwidth=2' if n == LATENT else ""
        shape = ', shape=doubleoctagon' if n == TARGET else ""
        lines.append(f'  "{n}" [fillcolor="{GROUP_COLOR[GROUP[n]]}"{extra}{shape}];')
    if name in ("expert", "latent"):   # keep layers readable: drivers on top, fire at bottom
        lines.append('  { rank=same; "elev"; "wind"; "burn_hist"; }')
    for a, b in edges:
        style = ' [color="#C0392B", penwidth=2]' if (a, b) in highlight else ""
        lab = f' [label="{labels[(a, b)]}", fontsize=10]' if labels and (a, b) in labels else ""
        lines.append(f'  "{a}" -> "{b}"{style or lab};')
    lines.append(
        '  subgraph cluster_legend { label="Legend"; fontsize=12; style=rounded; '
        + " ".join(f'"_{k}" [label="{k}", fillcolor="{c}", shape=box, fontsize=10];'
                   for k, c in GROUP_COLOR.items() if k in {GROUP[n] for n in nodes})
        + " }")
    lines.append("}")
    dot_path = FIGURES_DIR / f"dag_{name}.dot"
    dot_path.write_text("\n".join(lines))
    subprocess.run(["dot", "-Tpng", "-Gdpi=130", str(dot_path), "-o",
                    str(FIGURES_DIR / f"dag_{name}.png")], check=True)


def main():
    # learn TAN from training data only
    train = pd.read_parquet(TRAIN_PATH)
    tan_edges, tree_edges, tan_w = learn_tan(train)

    structures = {
        "naive_bayes": {"edges": NAIVE_BAYES_EDGES},
        "tan": {"edges": tan_edges, "tree_edges": tree_edges, "root": TAN_ROOT,
                "cond_mutual_info_nats": tan_w,
                "note": f"CMI weighted by 1/{NEG_KEEP} for kept non-fire rows"},
        "expert": {"edges": [list(e) for e in EXPERT_EDGES]},   # justifications: comments in config.py
    }
    for k in LATENT_CARDS:
        structures[f"latent_k{k}"] = {
            "edges": [list(e) for e in LATENT_DAG_EDGES], "latent": LATENT,
            "latent_states": k}

    # check that there are no loops and that all models include all features
    for name, s in structures.items():
        g = nx.DiGraph([tuple(e) for e in s["edges"]])
        assert nx.is_directed_acyclic_graph(g), f"{name} has a cycle"
        assert set(DISCRETE_FEATURES + [TARGET]) <= set(g.nodes), f"{name} is missing a variable"
        s["n_edges"] = g.number_of_edges()
        s["fire_parents"] = sorted(g.predecessors(TARGET))
    (RESULTS_DIR / "structures.json").write_text(json.dumps(structures, indent=2))

    draw("naive_bayes", "A. Naive Bayes", NAIVE_BAYES_EDGES)
    draw("tan", "B. Tree-augmented Naive Bayes (Chow-Liu tree in red)", tan_edges,
         highlight=set(tree_edges))
    draw("expert", "C. Expert DAG", EXPERT_EDGES)
    draw("latent", 'D. Expert DAG + latent danger node (2 / 3 / 4 states, dashed)',
         [tuple(e) for e in LATENT_DAG_EDGES],
         highlight=set(LATENT_EDGES))

    print("TAN tree (parent -> child, I(X;Y|fire) in nats):")
    for e, v in tan_w.items():
        print(f"  {e:28s} {v:.4f}")
    for name, s in structures.items():
        print(f"{name:12s} {s['n_edges']:2d} edges, fire parents: {s['fire_parents']}")

if __name__ == "__main__":
    main()
