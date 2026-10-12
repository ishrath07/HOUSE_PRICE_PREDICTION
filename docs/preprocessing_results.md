# Preprocessing results (Phase 3)

5-fold cross-validation on the training split, same folds as Phase 2.
Lower RMSE (log) is better.

| Model | CV RMSE (log) | Std | Train RMSE (log) | CV MAE ($) | CV R2 | vs Phase 2 linear |
|---|---|---|---|---|---|---|
| Phase 2 baseline: LinearRegression (numeric only) | 0.1258 | 0.0065 | 0.1200 | 15,629 | 0.910 | +0.0% |
| Numeric only + Ridge(alpha=10) | 0.1257 | 0.0062 | 0.1200 | 15,607 | 0.910 | -0.1% |
| Phase 3 pipeline + Ridge(alpha=10) | 0.1100 | 0.0059 | 0.0930 | 13,676 | 0.928 | -12.6% |

## What each decision is worth (ablation)

One change at a time versus the default pipeline, all with Ridge(alpha=10).

| Model | CV RMSE (log) | Std | Train RMSE (log) | CV MAE ($) | CV R2 | vs default |
|---|---|---|---|---|---|---|
| default (flag, log1p, 1% rare, drops on) | 0.1100 | 0.0059 | 0.0930 | 13,676 | 0.928 | +0.0% |
| sparse columns: drop | 0.1100 | 0.0058 | 0.0931 | 13,669 | 0.927 | -0.1% |
| sparse columns: keep | 0.1103 | 0.0052 | 0.0928 | 13,711 | 0.926 | +0.3% |
| no log1p on skewed columns | 0.1124 | 0.0052 | 0.0936 | 13,557 | 0.932 | +2.1% |
| keep every category level | 0.1098 | 0.0061 | 0.0922 | 13,609 | 0.928 | -0.3% |
| rare levels: under 3% | 0.1108 | 0.0061 | 0.0954 | 13,857 | 0.925 | +0.7% |
| keep collinear columns | 0.1102 | 0.0059 | 0.0927 | 13,633 | 0.929 | +0.1% |
| keep raw years and baths | 0.1101 | 0.0062 | 0.0928 | 13,669 | 0.928 | +0.0% |
