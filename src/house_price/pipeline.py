"""Phase 3: the full preprocessing + model pipeline.

    raw rows -> FeatureEngineer -> GroupMedianImputer -> ColumnTransformer -> regressor

    ColumnTransformer branches:
        ordinal    : fill with "None", OrdinalEncoder, scale
        nominal    : mode-fill, OneHotEncoder (rare levels grouped)
        continuous : median-fill, log1p if skewed, scale
        discrete   : median-fill, scale
        flags      : pass through

The whole chain is one scikit-learn estimator wrapped in TransformedTargetRegressor, so
`fit(X_train, y_price)` learns every statistic from the training rows only and
`predict(X_new)` replays the same steps and returns prices in dollars (EDA decisions 3, 15).

Run:  python -m house_price.pipeline
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from house_price import baseline, config, data
from house_price.features import (
    FeatureEngineer,
    GroupMedianImputer,
    SkewedLog1p,
    column_groups,
    ordinal_categories,
)

logger = logging.getLogger(__name__)


def build_preprocessor(
    sparse_strategy: str = config.SPARSE_STRATEGY,
    drop_collinear: bool = True,
    drop_redundant: bool = True,
    skew_threshold: float | None = config.SKEW_THRESHOLD,
    min_frequency: float | int | None = config.RARE_LEVEL_MIN_FREQUENCY,
) -> Pipeline:
    """Everything that turns raw Ames rows into a numeric matrix (no model yet).

    `skew_threshold=None` turns the log transform off and `min_frequency=None` keeps every
    category level; both exist so Phase 3 can measure what each decision is worth.
    """
    groups = column_groups(sparse_strategy, drop_collinear, drop_redundant)

    ordinal = Pipeline(
        [
            ("fill", SimpleImputer(strategy="constant", fill_value="None")),
            (
                "encode",
                OrdinalEncoder(
                    categories=ordinal_categories(groups["ordinal"]),
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )
    nominal = Pipeline(
        [
            ("fill", SimpleImputer(strategy="most_frequent")),  # Electrical: mode (decision 5)
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )
    continuous = Pipeline(
        [
            ("fill", SimpleImputer(strategy="median")),
            ("log", SkewedLog1p(threshold=skew_threshold)),
            ("scale", StandardScaler()),
        ]
    )
    discrete = Pipeline(
        [("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )

    columns = ColumnTransformer(
        [
            ("ordinal", ordinal, groups["ordinal"]),
            ("nominal", nominal, groups["nominal"]),
            ("continuous", continuous, groups["continuous"]),
            ("discrete", discrete, groups["discrete"]),
            ("flags", "passthrough", groups["flags"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return Pipeline(
        [
            (
                "features",
                FeatureEngineer(
                    sparse_strategy=sparse_strategy,
                    drop_collinear=drop_collinear,
                    drop_redundant=drop_redundant,
                ),
            ),
            ("lot_frontage", GroupMedianImputer(column="LotFrontage", by="Neighborhood")),
            ("columns", columns),
        ]
    )


def build_model(regressor=None, **preprocessor_kwargs):
    """Preprocessing + regressor, trained on log1p(price) and predicting dollars.

    The default regressor is Ridge. After encoding there are about 200 overlapping columns
    (for example HasGarage is exactly "GarageType is None"), which makes plain
    LinearRegression ill-conditioned: its answer then depends on the numerical library
    version. Ridge's penalty makes the problem well-posed. Phase 4 tunes alpha.
    """
    regressor = Ridge(alpha=config.RIDGE_ALPHA) if regressor is None else regressor
    steps = [
        ("preprocess", build_preprocessor(**preprocessor_kwargs)),
        ("model", regressor),
    ]
    return baseline.with_log_target(Pipeline(steps))


def feature_names(fitted_model) -> list[str]:
    """Names of the columns the regressor sees (for coefficients in Phase 5)."""
    pipeline = fitted_model.regressor_
    return list(pipeline.named_steps["preprocess"].named_steps["columns"].get_feature_names_out())


# --------------------------------------------------------------------------
# Phase 3 experiments
# --------------------------------------------------------------------------


def run_experiments(X: pd.DataFrame, y: pd.Series, experiments: dict[str, dict]) -> pd.DataFrame:
    """Cross-validate several pipeline variants on identical folds.

    `experiments` maps a name to {"regressor": ..., **preprocessor options}.
    """
    cv = baseline.get_cv()
    rows = []
    for name, options in experiments.items():
        options = dict(options)
        model = build_model(options.pop("regressor", None), **options)
        rows.append(baseline.evaluate(name, model, X, y, cv))
    return pd.DataFrame(rows)


def phase3_experiments() -> dict[str, dict]:
    """The Phase 3 pipeline with the default Ridge regressor."""
    return {f"Phase 3 pipeline + Ridge(alpha={config.RIDGE_ALPHA:g})": {}}


def ablation_experiments() -> dict[str, dict]:
    """One change at a time versus the default pipeline (Ridge regressor throughout)."""
    ridge = {"regressor": Ridge(alpha=config.RIDGE_ALPHA)}
    return {
        "default (flag, log1p, 1% rare, drops on)": {**ridge},
        "sparse columns: drop": {**ridge, "sparse_strategy": "drop"},
        "sparse columns: keep": {**ridge, "sparse_strategy": "keep"},
        "no log1p on skewed columns": {**ridge, "skew_threshold": None},
        "keep every category level": {**ridge, "min_frequency": None},
        "rare levels: under 3%": {**ridge, "min_frequency": 0.03},
        "keep collinear columns": {**ridge, "drop_collinear": False},
        "keep raw years and baths": {**ridge, "drop_redundant": False},
    }


def compare_to_baseline(results: pd.DataFrame, baseline_row: pd.Series) -> pd.DataFrame:
    """Add the percent change in RMSE(log) versus the baseline row."""
    out = results.copy()
    out["rmse_log_vs_baseline_pct"] = (out["cv_rmse_log"] / baseline_row.cv_rmse_log - 1) * 100
    return out


def save_results(results: pd.DataFrame, ablation: pd.DataFrame) -> None:
    """Write reports/preprocessing_results.csv and docs/preprocessing_results.md."""
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.REPORTS_DIR / "preprocessing_results.csv", index=False)
    ablation.to_csv(config.REPORTS_DIR / "preprocessing_ablation.csv", index=False)

    def table(df: pd.DataFrame, pct_col: str) -> list[str]:
        lines = [
            f"| Model | CV RMSE (log) | Std | Train RMSE (log) | CV MAE ($) | CV R2 | {pct_col} |",
            "|---|---|---|---|---|---|---|",
        ]
        pct = "rmse_log_vs_baseline_pct" if "rmse_log_vs_baseline_pct" in df else "vs_default_pct"
        for r in df.itertuples():
            lines.append(
                f"| {r.model} | {r.cv_rmse_log:.4f} | {r.cv_rmse_log_std:.4f} | "
                f"{r.train_rmse_log:.4f} | {r.cv_mae:,.0f} | {r.cv_r2:.3f} | "
                f"{getattr(r, pct):+.1f}% |"
            )
        return lines

    text = [
        "# Preprocessing results (Phase 3)",
        "",
        f"{config.CV_FOLDS}-fold cross-validation on the training split, same folds as Phase 2.",
        "Lower RMSE (log) is better.",
        "",
        *table(results, "vs Phase 2 linear"),
        "",
        "## What each decision is worth (ablation)",
        "",
        "One change at a time versus the default pipeline, all with "
        f"Ridge(alpha={config.RIDGE_ALPHA:g}).",
        "",
        *table(ablation, "vs default"),
        "",
    ]
    (config.DOCS_DIR / "preprocessing_results.md").write_text("\n".join(text), encoding="utf-8")
    logger.info("Saved preprocessing results to reports/ and docs/")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X, y = data.load_train()
    logger.info("Training rows: %d | raw features: %d", len(X), X.shape[1])

    cv = baseline.get_cv()
    base = baseline.evaluate(
        "Phase 2 baseline: LinearRegression (numeric only)", baseline.build_linear(), X, y, cv
    )
    ridge = Ridge(alpha=config.RIDGE_ALPHA)
    numeric_ridge = baseline.evaluate(
        f"Numeric only + Ridge(alpha={config.RIDGE_ALPHA:g})",
        baseline.build_linear(ridge),
        X,
        y,
        cv,
    )
    rows = [base, numeric_ridge] + run_experiments(X, y, phase3_experiments()).to_dict("records")
    results = compare_to_baseline(pd.DataFrame(rows), pd.Series(base))

    ablation = run_experiments(X, y, ablation_experiments())
    default_rmse = ablation.loc[0, "cv_rmse_log"]
    ablation["vs_default_pct"] = (ablation["cv_rmse_log"] / default_rmse - 1) * 100

    print()
    print(results.round(4).to_string(index=False))
    print()
    print(ablation.round(4).to_string(index=False))
    save_results(results, ablation)


if __name__ == "__main__":
    main()
