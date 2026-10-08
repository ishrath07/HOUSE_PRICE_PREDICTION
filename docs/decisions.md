# Phase 0: Project decisions

Written before any modelling so choices are deliberate, not accidental.

## Problem
Predict the sale price of a house from its attributes (regression).

## Dataset
Ames Housing (Kaggle "House Prices: Advanced Regression Techniques"), about 1,460 labelled rows and about 80 features.
Chosen over California Housing because it has missing values, many categoricals, skew and outliers, so feature engineering has real work to do.

## Metric
- **Primary:** RMSE on `log(SalePrice)`. Errors become relative (a 10% miss counts the same on a cheap or expensive house), and it matches Kaggle.
- **Secondary:** MAE and R-squared on the original price scale, for interpretability.

## Evaluation protocol
- One train/test split (80/20, `random_state=42`), created once in `data.py`.
- All exploration, tuning and model selection use **training data only**, with 5-fold cross-validation.
- The test set is scored **once**, at the very end (Phase 5).

## Leakage rules
- Fit every imputer, encoder and scaler inside a scikit-learn `Pipeline`, on training folds only.
- Never compute statistics (means, medians, category lists) on the full dataset.

## Baselines to beat
1. `DummyRegressor` (mean prediction).
2. Plain `LinearRegression` with minimal preprocessing.

## Engineering conventions
- Notebooks explore; reusable code lives in `src/house_price/`.
- All paths, seeds and constants come from `config.py`.
- Logging instead of `print`; type hints and docstrings on public functions.
- `ruff` for lint, `pytest` for tests, pinned requirements.

## Definition of done (whole project)
Reproducible with `make setup && make data && make train`; README with results table and figures; tests passing.