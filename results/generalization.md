# Generalization: training vs test fit (dirichlet estimator)

Gain = % reduction in log-loss vs a constant predictor (the period's own fire rate). 2,861 training fires. A big train-to-test drop = overfitting.

| structure   | fire parents                                      | fire-CPT rows   | fires per fire-CPT row   |   train gain % |   test gain % |   drop |
|:------------|:--------------------------------------------------|:----------------|:-------------------------|---------------:|--------------:|-------:|
| naive_bayes | (none: fire is a root)                            | -               | -                        |           6.43 |          5.31 |  -1.12 |
| tan         | (none: fire is a root)                            | -               | -                        |           5.7  |          3.08 |  -2.62 |
| expert      | burn_hist, rhmin, slope, tmax, veg_type           | 1024            | 2.8                      |           9.9  |          6.54 |  -3.37 |
| fuel        | burn_hist, rhmin, veg_cover, veg_height, veg_type | 1600            | 1.8                      |           8.81 |          5.53 |  -3.29 |
| no_weather  | burn_hist, slope, veg_type                        | 64              | 44.7                     |           7.73 |          6.04 |  -1.69 |
| fuel_only   | burn_hist, veg_cover, veg_height, veg_type        | 400             | 7.2                      |           8.13 |          6.09 |  -2.05 |
| hillclimb   | burn_hist, tmax                                   | 16              | 178.8                    |           8.22 |          8.08 |  -0.15 |

## Fire CPT of the hill-climbing DAG (H): P(fire = 1 | parents) per 10,000 cell-days

| burn_hist \ tmax   |   cool |   mild |   hot |   extreme |
|:-------------------|-------:|-------:|------:|----------:|
| recent             |   4.52 |   9.18 | 14.74 |     17.03 |
| mid                |   1.4  |   3.74 |  5.26 |      6.17 |
| old                |   0.58 |   1.25 |  2.25 |      2.01 |
| none               |   0.1  |   0.34 |  0.49 |      0.25 |
