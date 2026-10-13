# Model results (Phase 4)

Nested 5-fold cross-validation on the training split. Settings were tuned inside each
training fold only. Same outer folds as Phases 2 and 3.

| Model | Tuned settings | CV RMSE (log) | Std | CV MAE ($) | CV R2 |
|---|---|---|---|---|---|
| Ridge (alpha=10, untuned) | alpha=10 (fixed) | 0.1100 | 0.0059 | 13,676 | 0.928 |
| LinearRegression | no hyperparameters | 0.1190 | 0.0063 | 14,306 | 0.920 |
| Ridge | alpha=31.6 | 0.1099 | 0.0058 | 13,806 | 0.926 |
| Lasso | alpha=0.000681 | 0.1102 | 0.0057 | 13,943 | 0.926 |
| ElasticNet | alpha=0.00681, l1_ratio=0.05 | 0.1100 | 0.0056 | 13,914 | 0.926 |
| RandomForest | min_samples_leaf=1, max_features=0.33, max_depth=None | 0.1276 | 0.0075 | 15,766 | 0.891 |
| GradientBoosting | subsample=0.6, n_estimators=800, min_samples_leaf=1, max_depth=4, learning_rate=0.05 | 0.1138 | 0.0056 | 14,050 | 0.916 |
| **Blend: Ridge + GradientBoosting** (selected) | equal-weight average of Ridge and GradientBoosting | 0.1067 | 0.0059 | 13,067 | 0.929 |

Selected by the one-standard-error rule: **Blend: Ridge + GradientBoosting**.

## Paired comparison against the selected model

Negative `mean_diff` means better than the selected model. `ci95` is the half-width of a
95% interval, so a difference smaller than `ci95` is a statistical tie.

```
                    model  mean_diff   ci95  folds_better verdict
Ridge (alpha=10, untuned)     0.0033 0.0012             0   worse
         LinearRegression     0.0123 0.0011             0   worse
                    Ridge     0.0032 0.0012             0   worse
                    Lasso     0.0035 0.0014             0   worse
               ElasticNet     0.0033 0.0014             0   worse
             RandomForest     0.0209 0.0130             0   worse
         GradientBoosting     0.0071 0.0033             0   worse
```
