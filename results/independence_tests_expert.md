# Conditional independence tests: expert DAG

Stratified G-tests on the full 2015-2022 training data (13,669,116 cell-days, 2,861 fires, unweighted). Parents of fire: burn_hist, rhmin, slope, tmax, veg_type.
Bonferroni threshold: alpha = 0.05 / 13 tests = 0.0038. Decisions use `p_mc` (exact-margins Monte Carlo, 2000 replicates) where shown, else `p_chi2`.
`G_null_mean` = mean of G under independence (simulated where p_mc is shown, else df). `CMI_adj` = (G - G_null_mean) / 2N, the bias-corrected conditional mutual information in nats. `CMI_vs_weakest_edge` divides by the CMI_adj of the weakest significant edge into fire (fire ⊥ slope | other parents, 2.91e-06).
A rejected claim is called 'tiny' when it carries < 0.5x the information of that edge.
Cell-day rows are not independent (shared daily weather, repeated static values), so the root-node p-values are badly optimistic; read CMI_adj for those.

## omitted edge into fire

| claim                       | rows     |      G |   df |   G_null_mean |   p_chi2 |   p_mc |   CMI_adj |   CMI_vs_weakest_edge | verdict                                     |
|:----------------------------|:---------|-------:|-----:|--------------:|---------:|-------:|----------:|----------------------:|:--------------------------------------------|
| fire ⊥ wind | parents       | cell-day | 1077.4 | 1090 |        1031.8 |    0.602 | 0.097  |  1.67e-06 |                  0.57 | independence holds                          |
| fire ⊥ elev | parents       | cell-day |  969.4 |  917 |         812.6 |    0.112 | 0.0005 |  5.74e-06 |                  1.97 | rejected: dependence as large as a DAG edge |
| fire ⊥ veg_cover | parents  | cell-day |  713   |  697 |         602.3 |    0.329 | 0.0005 |  4.05e-06 |                  1.39 | rejected: dependence as large as a DAG edge |
| fire ⊥ veg_height | parents | cell-day |  828.7 |  905 |         736.1 |    0.967 | 0.004  |  3.39e-06 |                  1.16 | independence holds                          |

## edge in DAG (reference)

| claim                            | rows     |      G |   df |   G_null_mean | p_chi2   |   p_mc |   CMI_adj |   CMI_vs_weakest_edge | verdict                                       |
|:---------------------------------|:---------|-------:|-----:|--------------:|:---------|-------:|----------:|----------------------:|:----------------------------------------------|
| fire ⊥ burn_hist | other parents | cell-day | 3822.8 |  387 |         389.7 | 1e-inf   | 0.0005 |  0.000126 |                 43.08 | edge supported                                |
| fire ⊥ rhmin | other parents     | cell-day |  446.3 |  393 |         392.4 | 0.0326   | 0.014  |  1.97e-06 |                  0.68 | weak: not significant given the other parents |
| fire ⊥ slope | other parents     | cell-day |  502.3 |  442 |         422.7 | 0.0246   | 0.001  |  2.91e-06 |                  1    | edge supported                                |
| fire ⊥ tmax | other parents      | cell-day | 1012.6 |  414 |         397.5 | 5.04e-52 | 0.0005 |  2.25e-05 |                  7.72 | edge supported                                |
| fire ⊥ veg_type | other parents  | cell-day |  394.4 |  358 |         333.3 | 0.0902   | 0.0075 |  2.23e-06 |                  0.77 | weak: not significant given the other parents |

## weather / root-node claim

| claim              | rows      |               G |   df |   G_null_mean | p_chi2   | p_mc   |   CMI_adj |   CMI_vs_weakest_edge | verdict                                     |
|:-------------------|:----------|----------------:|-----:|--------------:|:---------|:-------|----------:|----------------------:|:--------------------------------------------|
| wind ⊥ tmax | elev | cell-day  |      1.5591e+06 |   36 |            36 | 1e-inf   | -      |   0.057   |              19565.9  | rejected: dependence as large as a DAG edge |
| wind ⊥ elev        | cell-day  | 385883          |    9 |             9 | 1e-inf   | -      |   0.0141  |               4842.63 | rejected: dependence as large as a DAG edge |
| wind ⊥ burn_hist   | cell-day  |  92435.6        |    9 |             9 | 1e-inf   | -      |   0.00338 |               1159.93 | rejected: dependence as large as a DAG edge |
| elev ⊥ burn_hist   | cell-year |   1759.7        |    9 |             9 | 1e-inf   | -      |   0.0234  |               8024.64 | rejected: dependence as large as a DAG edge |
