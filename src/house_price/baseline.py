"""Phase 2: baseline models.

Two deliberately simple models set the floor every later model must beat:

1. DummyRegressor        - always predicts the average (in log space).
2. LinearRegression      - numeric columns only, median imputation, nothing else.

Both train on log1p(SalePrice) (EDA decision 3) and are scored with 5-fold
cross-validation on the training split only. The test set is never touched.

Run:  python -m house_price.baseline
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
    make_column_selector,
)
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from house_price import config, data

logger = logging.getLogger(__name__)

# rmse_log is RMSE of log1p(price) (the primary metric, Kaggle's metric).
# MAE and R2 are computed on the original price scale.
SCORING = {
    "rmse_log": "neg_root_mean_squared_log_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def get_cv() -> KFold:
    """The one cross-validation scheme used for every model in the project."""
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def with_log_target(regressor) -> TransformedTargetRegressor:
    """Train on log1p(SalePrice); predictions come back on the price scale."""
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def build_dummy() -> TransformedTargetRegressor:
    """Baseline 1: predict the mean of the (log) training target for every house."""
    return with_log_target(DummyRegressor(strategy="mean"))


def build_linear(regressor=None) -> TransformedTargetRegressor:
    """Baseline 2: plain LinearRegression with minimal preprocessing.

    `regressor` lets Phase 3 swap in another estimator (e.g. Ridge) behind the same
    numeric-only preprocessing, to separate the effect of the model from the effect of features.

    Numeric columns only (chosen by dtype), missing values filled with the
    median, no scaling, no encoding, no feature engineering. Text columns are
    ignored on purpose: handling them is Phase 3's job.
    """
    numeric_only = ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                make_column_selector(dtype_include=np.number),
            )
        ],
        remainder="drop",
    )
    return with_log_target(
        Pipeline(
            [
                ("preprocess", numeric_only),
                ("model", LinearRegression() if regressor is None else regressor),
            ]
        )
    )


def evaluate(name: str, model, X: pd.DataFrame, y: pd.Series, cv=None) -> dict:
    """Cross-validate one model and return one row of results."""
    scores = cross_validate(
        model, X, y, cv=cv or get_cv(), scoring=SCORING, return_train_score=True
    )
    return {
        "model": name,
        "cv_rmse_log": -scores["test_rmse_log"].mean(),
        "cv_rmse_log_std": scores["test_rmse_log"].std(),
        "train_rmse_log": -scores["train_rmse_log"].mean(),
        "cv_mae": -scores["test_mae"].mean(),
        "cv_r2": scores["test_r2"].mean(),
    }


def run_baselines(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Evaluate both baselines with identical folds and return a results table."""
    cv = get_cv()
    rows = [
        evaluate("DummyRegressor (mean)", build_dummy(), X, y, cv),
        evaluate("LinearRegression (numeric only)", build_linear(), X, y, cv),
    ]
    results = pd.DataFrame(rows)
    dummy_rmse = results.loc[0, "cv_rmse_log"]
    results["rmse_log_vs_dummy_pct"] = (results["cv_rmse_log"] / dummy_rmse - 1) * 100
    return results


def oof_predictions(model, X: pd.DataFrame, y: pd.Series, cv=None) -> pd.Series:
    """Out-of-fold predictions (price scale): each house predicted by a model that never saw it."""
    pred = cross_val_predict(model, X, y, cv=cv or get_cv())
    return pd.Series(pred, index=X.index, name="predicted")


def save_results(results: pd.DataFrame) -> None:
    """Write the numbers later phases must beat to reports/ (CSV) and docs/ (Markdown)."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.REPORTS_DIR / "baseline_results.csv", index=False)

    lines = [
        "# Baseline results (Phase 2)",
        "",
        f"{config.CV_FOLDS}-fold cross-validation on the training split. "
        "Target modelled as log1p(SalePrice).",
        "Every later model has to beat these numbers.",
        "",
        "| Model | CV RMSE (log) | Std | Train RMSE (log) | CV MAE ($) | CV R2 |",
        "|---|---|---|---|---|---|",
    ]
    for r in results.itertuples():
        lines.append(
            f"| {r.model} | {r.cv_rmse_log:.4f} | {r.cv_rmse_log_std:.4f} | "
            f"{r.train_rmse_log:.4f} | {r.cv_mae:,.0f} | {r.cv_r2:.3f} |"
        )
    (config.DOCS_DIR / "baseline_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved baseline results to reports/ and docs/")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X, y = data.load_train()
    logger.info("Training rows: %d | features: %d", len(X), X.shape[1])
    results = run_baselines(X, y)
    print()
    print(results.round(4).to_string(index=False))
    save_results(results)


if __name__ == "__main__":
    main()
