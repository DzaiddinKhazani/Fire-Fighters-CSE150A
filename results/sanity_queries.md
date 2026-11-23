# Sanity and diagnostic queries (variable elimination, unobserved variables summed out)

Training fire rate (weighted, = full 2015-2022 table): 0.000210

## Sanity: P(fire) on a hot/dry/windy vs a cool/humid/calm day

Other variables are summed out. `P(fire)` = the model's overall fire rate, which should match the training rate above.

| model | P(fire) | hot/dry/windy | cool/humid/calm | ratio | higher when hot? |
|---|---|---|---|---|---|
| naive_bayes_mle | 0.000210 | 0.000083 | 0.000067 | 1.2x | PASS |
| naive_bayes_bdeu | 0.000210 | 0.000084 | 0.000067 | 1.3x | PASS |
| naive_bayes_dirichlet | 0.000210 | 0.000084 | 0.000068 | 1.2x | PASS |
| tan_mle | 0.000210 | 0.000103 | 0.000059 | 1.8x | PASS |
| tan_bdeu | 0.000210 | 0.000114 | 0.000059 | 1.9x | PASS |
| tan_dirichlet | 0.000210 | 0.000135 | 0.000061 | 2.2x | PASS |
| expert_mle | 0.000472 | 0.002346 | 0.000087 | 26.9x | PASS |
| expert_bdeu | 0.000472 | 0.002347 | 0.000087 | 26.9x | PASS |
| expert_dirichlet | 0.000217 | 0.000423 | 0.000087 | 4.9x | PASS |

## Diagnostic: P(variable | fire = 1) vs P(variable)  (dirichlet models)

`data` columns are the weighted training frequencies, for comparison.

| rhmin    |   data P(.|fire) |   naive_bayes P(.|fire) |   tan P(.|fire) |   expert P(.|fire) |   data P(.) |   expert P(.) |
|:---------|-----------------:|------------------------:|----------------:|-------------------:|------------:|--------------:|
| crit_dry |            0.093 |                   0.093 |           0.093 |              0.150 |       0.117 |         0.119 |
| dry      |            0.178 |                   0.178 |           0.177 |              0.235 |       0.213 |         0.221 |
| moderate |            0.298 |                   0.298 |           0.297 |              0.279 |       0.255 |         0.261 |
| humid    |            0.431 |                   0.431 |           0.432 |              0.335 |       0.414 |         0.399 |

| tmax    |   data P(.|fire) |   naive_bayes P(.|fire) |   tan P(.|fire) |   expert P(.|fire) |   data P(.) |   expert P(.) |
|:--------|-----------------:|------------------------:|----------------:|-------------------:|------------:|--------------:|
| cool    |            0.131 |                   0.132 |           0.134 |              0.131 |       0.303 |         0.303 |
| mild    |            0.416 |                   0.416 |           0.416 |              0.375 |       0.380 |         0.380 |
| hot     |            0.373 |                   0.373 |           0.371 |              0.333 |       0.223 |         0.223 |
| extreme |            0.080 |                   0.080 |           0.079 |              0.162 |       0.093 |         0.093 |

| wind     |   data P(.|fire) |   naive_bayes P(.|fire) |   tan P(.|fire) |   expert P(.|fire) |   data P(.) |   expert P(.) |
|:---------|-----------------:|------------------------:|----------------:|-------------------:|------------:|--------------:|
| calm     |            0.129 |                   0.129 |           0.129 |              0.178 |       0.181 |         0.181 |
| light    |            0.479 |                   0.478 |           0.477 |              0.409 |       0.404 |         0.404 |
| moderate |            0.325 |                   0.325 |           0.325 |              0.300 |       0.299 |         0.299 |
| strong   |            0.068 |                   0.068 |           0.069 |              0.113 |       0.117 |         0.117 |

| veg_type    |   data P(.|fire) |   naive_bayes P(.|fire) |   tan P(.|fire) |   expert P(.|fire) |   data P(.) |   expert P(.) |
|:------------|-----------------:|------------------------:|----------------:|-------------------:|------------:|--------------:|
| nonburnable |            0.422 |                   0.421 |           0.422 |              0.424 |       0.342 |         0.340 |
| shrub       |            0.508 |                   0.508 |           0.508 |              0.491 |       0.582 |         0.579 |
| other_veg   |            0.070 |                   0.070 |           0.070 |              0.084 |       0.071 |         0.078 |
| missing     |            0.000 |                   0.000 |           0.000 |              0.000 |       0.005 |         0.004 |

| burn_hist   |   data P(.|fire) |   naive_bayes P(.|fire) |   tan P(.|fire) |   expert P(.|fire) |   data P(.) |   expert P(.) |
|:------------|-----------------:|------------------------:|----------------:|-------------------:|------------:|--------------:|
| recent      |            0.464 |                   0.463 |           0.459 |              0.447 |       0.103 |         0.103 |
| mid         |            0.329 |                   0.328 |           0.327 |              0.318 |       0.197 |         0.197 |
| old         |            0.142 |                   0.142 |           0.143 |              0.162 |       0.224 |         0.224 |
| none        |            0.065 |                   0.066 |           0.071 |              0.073 |       0.476 |         0.476 |
