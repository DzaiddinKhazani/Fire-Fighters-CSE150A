# Stability of the hill-climbing structure (H)

H itself: 28 edges, max parents 4. Fire's parents were identical in 16 of 16 runs; in the 10 bootstrap runs, 23.9 of 28.0 edges on average matched H.

| run             |   moves |   edges |   edges shared with H (either direction) | fire parents    | fire children   |
|:----------------|--------:|--------:|-----------------------------------------:|:----------------|:----------------|
| max parents = 2 |      18 |      17 |                                       17 | burn_hist, tmax | -               |
| max parents = 3 |      24 |      23 |                                       23 | burn_hist, tmax | -               |
| max parents = 4 |      29 |      28 |                                       28 | burn_hist, tmax | -               |
| max parents = 5 |      35 |      32 |                                       28 | burn_hist, tmax | -               |
| 2015-2018 only  |      28 |      28 |                                       24 | burn_hist, tmax | -               |
| 2019-2022 only  |      28 |      28 |                                       24 | burn_hist, tmax | -               |
| bootstrap 1     |      28 |      28 |                                       23 | burn_hist, tmax | -               |
| bootstrap 2     |      29 |      28 |                                       23 | burn_hist, tmax | -               |
| bootstrap 3     |      28 |      28 |                                       22 | burn_hist, tmax | -               |
| bootstrap 4     |      28 |      28 |                                       28 | burn_hist, tmax | -               |
| bootstrap 5     |      29 |      28 |                                       23 | burn_hist, tmax | -               |
| bootstrap 6     |      28 |      28 |                                       23 | burn_hist, tmax | -               |
| bootstrap 7     |      28 |      28 |                                       23 | burn_hist, tmax | -               |
| bootstrap 8     |      29 |      28 |                                       28 | burn_hist, tmax | -               |
| bootstrap 9     |      28 |      28 |                                       23 | burn_hist, tmax | -               |
| bootstrap 10    |      28 |      28 |                                       23 | burn_hist, tmax | -               |
