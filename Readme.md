# House Price Prediction

End-to-end regression project on the Ames Housing dataset. It estimates a house's sale price
from its recorded characteristics using a leakage-safe scikit-learn pipeline, nested
cross-validation, a single held-out evaluation, and a reusable prediction CLI.

> **Status:** Phases 0-6 complete, including model packaging and CSV / single-record prediction.
> Requires Python 3.12 or newer; exact dependency versions are pinned in `requirements.txt`.

## Quick start

```bash
make setup                       # create .venv and install pinned dependencies/package
# put Kaggle's train.csv in data/raw/ (see data/README.md)
make data                        # create the fixed train/test split
make test                        # run tests
make train                       # tune, compare, fit, and save models/best_model.joblib
make evaluate                    # score the held-out test split; run this only once
make predict ARGS="--input-csv houses.csv --output predictions.csv"
```

For a single complete Ames feature record, save one JSON object to a file and run:

```bash
.venv/Scripts/python -m house_price.predict --record-file house.json  # Windows
# .venv/bin/python -m house_price.predict --record-file house.json    # macOS/Linux
```

The JSON object must include the same raw feature columns used for training. A CSV can
contain an optional `Id` column; the output is a Kaggle-style `Id,SalePrice` file. Input
must not include `SalePrice`. To score one inline JSON object instead, use `--record-json`.

The command-line equivalents work from an activated environment as well:

```bash
python -m house_price.train
python -m house_price.evaluate
python -m house_price.predict --input-csv houses.csv --output predictions.csv
```

## Project layout

```
data/raw, data/processed   input data (git-ignored)
notebooks/                 exploration only
src/house_price/           reusable code and train/evaluate/predict CLIs
models/                    saved model artifacts
reports/figures/           evaluation and diagnostic plots
tests/                     pytest
docs/decisions.md          problem, metric, evaluation protocol
```

## Approach

The target is modelled as `log1p(SalePrice)` and predictions are returned in dollars. Feature
engineering, imputation, encoding, and scaling are fitted inside scikit-learn pipelines.
Nested cross-validation on training rows selects the final model; the held-out 20% test split
is used only once, at the end. The saved artifact contains the complete fitted preprocessing
and model pipeline, so prediction applies the exact same transformations as training.

Run `make lint` for Ruff checks, `make format` to format Python code with Black, and
`make test` for pytest. `make train` performs the full nested-CV workflow; `--quick` is for
development only and does not save a final artifact.

## Roadmap

- [x] Phase 0: Setup and decisions
- [x] Phase 1: Data and EDA
- [x] Phase 2: Baseline
- [x] Phase 3: Features and preprocessing pipeline
- [x] Phase 4: Modelling and validation
- [x] Phase 5: Evaluation and diagnostics
- [x] Phase 6: Packaging (CLI, tests, docs)

## Results

Nested 5-fold cross-validation on the training split (RMSE on log price, lower is better).
Settings are tuned inside each training fold, so these scores are not flattered by the tuning.
The test split was left untouched until Phase 5.

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

## Held-out test (Phase 5)

Scored **once** on the 20% test split after model selection was finished. Outliers were
not dropped from the test rows (EDA decision 12). Diagnostics, coefficients and the
houses the model gets most wrong: `docs/evaluation_results.md`, `reports/figures/13_*.png`–`17_*.png`.

<!-- TEST_RESULTS:START -->
Held-out test scored **once** with the Phase 4 winner (**Blend: Ridge + GradientBoosting**).

| Split | n | RMSE (log) | MAE ($) | R2 |
|---|---|---|---|---|
| cv_train_5fold | 1166 | 0.1067 | 13,067 | 0.929 |
| test | 292 | 0.1236 | 13,798 | 0.930 |
| train_resubstitution | 1166 | 0.0547 | 7,077 | 0.981 |

Test RMSE(log) = **0.1236** (+15.9% vs CV). `train_resubstitution` refits nothing: it is the same frozen model scored on the rows it was trained on, so it is optimistic (especially for the tree member).

Top raw features by permutation importance (test): OverallQual (0.0502), TotalBsmtSF (0.0349), GrLivArea (0.0314), 2ndFlrSF (0.0268), 1stFlrSF (0.0221), YearBuilt (0.0142), OverallCond (0.0133), LotArea (0.0114).
<!-- TEST_RESULTS:END -->

## Evaluation figures

The held-out diagnostics and top test-set feature drivers are shown below. Full explanations,
assumption checks, coefficients, and worst-case errors are in
[`docs/evaluation_results.md`](docs/evaluation_results.md).

![Predicted prices versus actual prices](reports/figures/13_predicted_vs_actual.png)

![Residuals and error distribution](reports/figures/14_residuals.png)

![Linear-member assumption diagnostics](reports/figures/15_assumptions.png)

![Feature drivers and Ridge coefficients](reports/figures/16_feature_drivers.png)

![Error analysis](reports/figures/17_error_analysis.png)
