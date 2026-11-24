# Model D: expert DAG + latent weather regime, fitted with EM

Weighted EM on 275,616 training rows, several random starts per state count (`starts`), random initial CPTs, stop at relative log-likelihood change < 1e-09 or 400 iterations. Marginal-centred Dirichlet smoothing (ESS = 10) in every M-step (MAP-EM). 
BIC uses N = full training cell-days and counts only the CPTs that involve the latent node (the rest are identical across models).

## Convergence

|   states |   starts |   iterations | converged   |   train_loglik |   spread_across_starts |   latent_params |           BIC |
|---------:|---------:|-------------:|:------------|---------------:|-----------------------:|----------------:|--------------:|
|        2 |        3 |          264 | True        | -129,499,676.1 |              499,192.0 |             255 | 259,003,542.1 |
|        3 |        3 |          400 | False       | -129,164,314.3 |              141,581.1 |             383 | 258,334,921.5 |
|        4 |        2 |          400 | False       | -129,022,732.9 |               25,118.6 |             511 | 258,053,861.9 |

`spread_across_starts` = best minus worst final log-likelihood over the random starts (0 means every start reached the same optimum). `converged` = False means the 400-iteration cap was hit while the log-likelihood was still rising slowly.

## Test set (2023-2024), 95% day-level bootstrap CIs

| model                 | log_loss                         | brier                            | pr_auc                           | roc_auc              |
|:----------------------|:---------------------------------|:---------------------------------|:---------------------------------|:---------------------|
| naive_bayes_dirichlet | 1.446e-03 [1.330e-03, 1.577e-03] | 1.565e-04 [1.416e-04, 1.725e-04] | 7.534e-04 [6.309e-04, 9.689e-04] | 0.815 [0.798, 0.831] |
| expert_dirichlet      | 1.428e-03 [1.305e-03, 1.561e-03] | 1.564e-04 [1.415e-04, 1.724e-04] | 7.734e-04 [6.598e-04, 9.640e-04] | 0.814 [0.796, 0.832] |
| latent_k2             | 1.424e-03 [1.305e-03, 1.554e-03] | 1.564e-04 [1.415e-04, 1.725e-04] | 6.529e-04 [5.552e-04, 7.873e-04] | 0.812 [0.796, 0.828] |
| latent_k3             | 1.420e-03 [1.302e-03, 1.553e-03] | 1.564e-04 [1.415e-04, 1.724e-04] | 6.970e-04 [5.870e-04, 8.728e-04] | 0.816 [0.799, 0.832] |
| latent_k4             | 1.421e-03 [1.302e-03, 1.554e-03] | 1.564e-04 [1.415e-04, 1.725e-04] | 7.000e-04 [5.863e-04, 8.989e-04] | 0.816 [0.800, 0.831] |

## Difference from the expert DAG (C, dirichlet), paired bootstrap

Better = negative for log_loss/brier, positive for pr_auc/roc_auc.

| model     | log_loss                           | brier                              | pr_auc                              | roc_auc                |
|:----------|:-----------------------------------|:-----------------------------------|:------------------------------------|:-----------------------|
| latent_k2 | -3.469e-06 [-1.624e-05, 9.493e-06] | 3.703e-09 [-4.970e-09, 1.356e-08]  | -1.205e-04 [-2.234e-04, -4.995e-05] | -0.002 [-0.013, 0.009] |
| latent_k3 | -7.766e-06 [-1.962e-05, 5.025e-06] | -2.199e-10 [-8.340e-09, 8.887e-09] | -7.640e-05 [-1.699e-04, 1.077e-05]  | 0.002 [-0.009, 0.012]  |
| latent_k4 | -6.928e-06 [-1.997e-05, 5.496e-06] | 3.590e-09 [-4.032e-09, 1.189e-08]  | -7.338e-05 [-1.670e-04, 2.213e-05]  | 0.002 [-0.010, 0.013]  |

Mean predicted P(fire) on the test set, latent_k3: 0.000187 (observed test rate 0.000156).

## What the hidden states look like (2 states)

Responsibility-weighted shares over the training data (weights undo the subsampling).

|                          |   state 0 |   state 1 |
|:-------------------------|----------:|----------:|
| share of cell-days       |     0.539 |     0.461 |
| fire rate (per 10k)      |     2.680 |     1.416 |
| P(tmax in hot/extreme)   |     0.513 |     0.087 |
| P(rhmin in crit_dry/dry) |     0.383 |     0.268 |
| P(wind in strong)        |     0.025 |     0.224 |
| P(tmax in cool)          |     0.110 |     0.529 |
| P(rhmin in humid)        |     0.319 |     0.526 |
| share in Sep-Dec         |     0.328 |     0.342 |
| share in Jun-Aug         |     0.391 |     0.090 |
| share in Jan-May         |     0.281 |     0.568 |

## What the hidden states look like (3 states)

Responsibility-weighted shares over the training data (weights undo the subsampling).

|                          |   state 0 |   state 1 |   state 2 |
|:-------------------------|----------:|----------:|----------:|
| share of cell-days       |     0.268 |     0.294 |     0.438 |
| fire rate (per 10k)      |     1.089 |     1.661 |     3.006 |
| P(tmax in hot/extreme)   |     0.079 |     0.275 |     0.490 |
| P(rhmin in crit_dry/dry) |     0.101 |     0.559 |     0.317 |
| P(wind in strong)        |     0.367 |     0.036 |     0.017 |
| P(tmax in cool)          |     0.732 |     0.184 |     0.120 |
| P(rhmin in humid)        |     0.692 |     0.195 |     0.392 |
| share in Sep-Dec         |     0.320 |     0.356 |     0.330 |
| share in Jun-Aug         |     0.083 |     0.201 |     0.389 |
| share in Jan-May         |     0.597 |     0.444 |     0.280 |

## What the hidden states look like (4 states)

Responsibility-weighted shares over the training data (weights undo the subsampling).

|                          |   state 0 |   state 1 |   state 2 |   state 3 |
|:-------------------------|----------:|----------:|----------:|----------:|
| share of cell-days       |     0.154 |     0.258 |     0.274 |     0.314 |
| fire rate (per 10k)      |     0.605 |     1.732 |     2.157 |     3.081 |
| P(tmax in hot/extreme)   |     0.028 |     0.144 |     0.246 |     0.663 |
| P(rhmin in crit_dry/dry) |     0.049 |     0.342 |     0.368 |     0.426 |
| P(wind in strong)        |     0.471 |     0.105 |     0.062 |     0.000 |
| P(tmax in cool)          |     0.929 |     0.275 |     0.291 |     0.028 |
| P(rhmin in humid)        |     0.814 |     0.332 |     0.393 |     0.305 |
| share in Sep-Dec         |     0.296 |     0.369 |     0.358 |     0.306 |
| share in Jun-Aug         |     0.035 |     0.177 |     0.176 |     0.487 |
| share in Jan-May         |     0.669 |     0.455 |     0.466 |     0.207 |
