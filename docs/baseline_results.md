# Baseline results (Phase 2)

5-fold cross-validation on the training split. Target modelled as log1p(SalePrice).
Every later model has to beat these numbers.

| Model | CV RMSE (log) | Std | Train RMSE (log) | CV MAE ($) | CV R2 |
|---|---|---|---|---|---|
| DummyRegressor (mean) | 0.3906 | 0.0115 | 0.3907 | 54,689 | -0.032 |
| LinearRegression (numeric only) | 0.1258 | 0.0065 | 0.1200 | 15,629 | 0.910 |
