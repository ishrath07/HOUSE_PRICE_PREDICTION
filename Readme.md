# House Price Prediction

End-to-end regression project on the Ames Housing dataset, built in an industry-style structure
(pipelines, tests, reproducible config).

> **Status:** Phases 0-3 complete (scaffold, EDA, baseline, features and preprocessing). Next: Phase 4 (modelling and validation).

## Quick start

```bash
make setup                 # create venv, install deps, install package
# put train.csv in data/raw/  (see data/README.md)
make data                  # create the fixed train/test split
make test                  # run tests
```

## Project layout

```
data/raw, data/processed   input data (git-ignored)
notebooks/                 exploration only
src/house_price/           reusable code (config, data, baseline, features, pipeline; later: train, evaluate, predict)
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
- [ ] Phase 4: Modelling and validation
- [ ] Phase 5: Evaluation and diagnostics
- [ ] Phase 6: Packaging (CLI, tests, docs)

## Results

5-fold cross-validation on the training split (RMSE on log price, lower is better):

| Model | CV RMSE (log) | CV MAE ($) | CV R2 |
|---|---|---|---|
| Dummy (mean) | 0.3906 | 54,689 | -0.032 |
| Phase 2: LinearRegression, numeric columns only | 0.1258 | 15,629 | 0.910 |
| Phase 3: full preprocessing pipeline + Ridge | 0.1100 | 13,676 | 0.928 |

Ridge on the numeric-only columns scores 0.1257, so the Phase 3 gain comes from the engineered
features and encoders, not from regularization. Details: `docs/baseline_results.md`,
`docs/preprocessing_results.md`.
