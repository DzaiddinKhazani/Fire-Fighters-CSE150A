# Fire-Fighters-CSE150A

SQLite file, tif files and county shp file are not tracked due to size so download that on your own device. It's in the google drive.

## Fitting and evaluation (todo phases 4-5, models A-C)

```bash
pip install -r requirements.txt
python fit_models.py   # fits A-C x {mle, bdeu, dirichlet} -> results/models/, results/sanity_queries.md
python evaluate.py     # test-set predictions, baselines, metrics + bootstrap CIs -> results/metrics.md, figures
```

- `bn.py`: CPT fitting from weighted counts (the `weight` column undoes the 2% non-fire
  subsampling, so no separate prior correction is needed) and exact inference by
  variable elimination (`BayesNet.query`).
- `dirichlet` estimator: Dirichlet prior centred on each variable's marginal. The expert DAG has
  343 never-seen combinations of fire's 5 parents; MLE/BDeu give those P(fire) = 0.5, which
  doubles the model's overall fire rate. See the docstring in `bn.py`.
- Bootstrap CIs resample whole test days (cells on the same day share weather).
- Not done yet: model D (latent node, EM), conditional-independence tests, Phase 6.
