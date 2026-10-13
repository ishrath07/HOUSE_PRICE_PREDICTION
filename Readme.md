# House Price Prediction

End-to-end regression project on the Ames Housing dataset, built in an industry-style structure
(pipelines, tests, reproducible config).

> **Status:** Phases 0-4 complete (scaffold, EDA, baseline, features, modelling and validation). Next: Phase 5 (evaluation and diagnostics).

## Quick start

```bash
make setup                 # create venv, install deps, install package
# put train.csv in data/raw/  (see data/README.md)
make data                  # create the fixed train/test split
make test                  # run tests
python -m house_price.train   # Phase 4: tune and compare models (about 10 min, see --help)
```

## Project layout

```
data/raw, data/processed   input data (git-ignored)
notebooks/                 exploration only
src/house_price/           reusable code (config, data, baseline, features, pipeline, models, train; later: evaluate, predict)
models/                    saved model artifacts
reports/figures/           plots for the README
tests/                     pytest
docs/decisions.md          problem, metric, evaluation protocol
```

## Roadmap

- [x] Phase 0: Setup and decisions
- [x] Phase 1: Data and EDA
- [x] Phase 2: Baseline
- [x] Phase 3: Features and preprocessing pipeline
- [x] Phase 4: Modelling and validation
- [ ] Phase 5: Evaluation and diagnostics
- [ ] Phase 6: Packaging (CLI, tests, docs)

## Results

Nested 5-fold cross-validation on the training split (RMSE on log price, lower is better).
Settings are tuned inside each training fold, so these scores are not flattered by the tuning.
The test split stays untouched until Phase 5.

<!-- RESULTS:START -->
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
<!-- RESULTS:END -->

Earlier phases for reference: dummy 0.3906, numeric-only linear baseline 0.1258
(`docs/baseline_results.md`), Phase 3 pipeline with Ridge 0.1100 (`docs/preprocessing_results.md`).
Full tables, per-fold scores and tuning curves: `docs/model_results.md`, `reports/`.
