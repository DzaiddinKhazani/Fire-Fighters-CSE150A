# Fire-Fighters-CSE150A

SQLite file, tif files and county shp file are not tracked due to size so download that on your own device. It's in the google drive.

## Fitting and evaluation (todo phases 4-5, models A-C and E-H)

```bash
pip install -r requirements.txt
python define_structures.py   # A-H edge lists + diagrams -> results/structures.json, results/figures/dag_*.png
python fit_models.py   # fits A-C x {mle, bdeu, dirichlet} -> results/models/, results/sanity_queries.md
python evaluate.py     # test-set predictions, baselines, metrics + bootstrap CIs -> results/metrics.md, figures
```

- Experimental structures (`config.py`): E fuel-structure DAG, F expert DAG without weather -> fire,
  G fuel + burn history only. Each changes one idea relative to the expert DAG (C);
  `results/metrics.md` has a table of their differences from C.
- H hill climbing (`structure_search.py`): greedy add/remove/reverse edge search scored by BIC,
  max 4 parents per node, learned from the training years only.

- `bn.py`: CPT fitting from weighted counts (the `weight` column undoes the 2% non-fire
  subsampling, so no separate prior correction is needed) and exact inference by
  variable elimination (`BayesNet.query`).
- `dirichlet` estimator: Dirichlet prior centred on each variable's marginal. The expert DAG has
  343 never-seen combinations of fire's 5 parents; MLE/BDeu give those P(fire) = 0.5, which
  doubles the model's overall fire rate. See the docstring in `bn.py`.
- Bootstrap CIs resample whole test days (cells on the same day share weather).
- Model D (latent weather regime, EM): `python fit_latent.py` (fits K = 2, 3, 4 and evaluates)
  -> `results/latent_em.md`, `results/models/latent_k*.pkl`, `results/figures/em_convergence.png`.
- Conditional-independence tests on the expert DAG: `python independence_tests.py`
  -> `results/independence_tests_expert.md`.
- Report source: `report/report.tex` (figures in `report/figures/`).
- Not done yet: Phase 6 (FHSZ comparison).
