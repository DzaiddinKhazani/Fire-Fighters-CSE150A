"""
hillclimb_stability.py -- how stable is the hill-climbing (BIC) structure H?

Reruns structure_search.hill_climb on the training data
  * with the max-parents limit set to 2, 3, 4 (the value used for H) and 5,
  * on two halves of the training period (2015-2018 and 2019-2022),
  * on N_BOOT bootstrap resamples of the training rows,
and records the number of edges, fire's parents, and how many edges match H.

Output: results/hillclimb_stability.md
"""
import json

import pandas as pd

from config import (DISCRETE_FEATURES, HILLCLIMB_MAX_PARENTS, RESULTS_DIR, SPLIT_INFO_PATH,
                    TARGET, TRAIN_PATH)
from structure_search import hill_climb

N_BOOT = 10


def main():
    train = pd.read_parquet(TRAIN_PATH)
    states = json.loads(SPLIT_INFO_PATH.read_text())["states"]
    states[TARGET] = [0, 1]
    variables = DISCRETE_FEATURES + [TARGET]
    ref = {tuple(e) for e in json.loads((RESULTS_DIR / "structures.json").read_text())["hillclimb"]["edges"]}

    def run(label, df, max_parents=HILLCLIMB_MAX_PARENTS):
        edges, steps, _ = hill_climb(df, variables, states, max_parents=max_parents, verbose=False)
        e = {tuple(x) for x in edges}
        # an edge counts as shared if H has it in either direction
        shared = sum((a, b) in ref or (b, a) in ref for a, b in e)
        return {"run": label, "moves": len(steps), "edges": len(e),
                "edges shared with H (either direction)": shared,
                "fire parents": ", ".join(sorted(a for a, b in e if b == TARGET)) or "-",
                "fire children": ", ".join(sorted(b for a, b in e if a == TARGET)) or "-"}

    rows = [run(f"max parents = {k}", train, k) for k in (2, 3, 4, 5)]
    year = pd.to_datetime(train["date"]).dt.year
    rows.append(run("2015-2018 only", train[year <= 2018]))
    rows.append(run("2019-2022 only", train[year >= 2019]))
    for i in range(N_BOOT):
        rows.append(run(f"bootstrap {i + 1}", train.sample(frac=1, replace=True, random_state=i)))
    df = pd.DataFrame(rows)

    boot = df[df.run.str.startswith("bootstrap")]
    text = "\n".join([
        "# Stability of the hill-climbing structure (H)\n",
        f"H itself: {len(ref)} edges, max parents {HILLCLIMB_MAX_PARENTS}. "
        f"Fire's parents were identical in {int((df['fire parents'] == df.loc[2, 'fire parents']).sum())} "
        f"of {len(df)} runs; in the {len(boot)} bootstrap runs, "
        f"{boot['edges shared with H (either direction)'].mean():.1f} of "
        f"{boot.edges.mean():.1f} edges on average matched H.\n",
        df.to_markdown(index=False) + "\n",
    ])
    (RESULTS_DIR / "hillclimb_stability.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
