"""Phase 5: score the frozen model on the held-out test split once, then diagnose it.

The test split was created in Phase 0 and has not been opened by EDA, features, or
tuning. This module loads `models/best_model.joblib`, predicts test prices exactly
once, and builds the tables the notebook plots: residuals, linear-assumption checks
on the Ridge member, feature drivers, and the houses the model gets most wrong.

Run:  python -m house_price.evaluate
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge

from house_price import config, data, train

logger = logging.getLogger(__name__)

README_START, README_END = "<!-- TEST_RESULTS:START -->", "<!-- TEST_RESULTS:END -->"

# Raw columns that help explain a miss. Missing ones are skipped.
CONTEXT_COLUMNS = [
    "Neighborhood",
    "OverallQual",
    "OverallCond",
    "GrLivArea",
    "SaleCondition",
    "SaleType",
    "YearBuilt",
    "GarageCars",
    "TotalBsmtSF",
    "ExterQual",
    "KitchenQual",
]
UNUSUAL_SALES = {"Abnorml", "Family", "Alloca", "AdjLand"}
VIF_HIGH = 10.0  # textbook cutoff; Ridge is there because many columns exceed it


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_model(path=None):
    """The Phase 4 winner, already refit on the full training split."""
    path = path or config.MODELS_DIR / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m house_price.train` (Phase 4) first."
        )
    model = joblib.load(path)
    logger.info("Loaded %s from %s", type(model).__name__, path.name)
    return model


def inner_estimator(model):
    """The estimator that sees log-price: a Pipeline or a VotingRegressor."""
    return getattr(model, "regressor_", model)


def member_pipelines(model) -> dict:
    """Fitted member pipelines of a blend, or {'model': pipeline} for a single model."""
    inner = inner_estimator(model)
    named = getattr(inner, "named_estimators_", None)
    if named:
        return dict(named)
    return {"model": inner}


def linear_member(model):
    """The Ridge/Lasso/ElasticNet/LinearRegression pipeline, or None for a pure tree."""
    for pipe in member_pipelines(model).values():
        step = getattr(pipe, "named_steps", {}).get("model")
        if isinstance(step, (Ridge, LinearRegression)) or type(step).__name__ in {
            "Lasso",
            "ElasticNet",
        }:
            return pipe
    return None


def tree_member(model):
    """The tree-model pipeline, or None if the winner is purely linear."""
    for pipe in member_pipelines(model).values():
        step = getattr(pipe, "named_steps", {}).get("model")
        if type(step).__name__ in {"GradientBoostingRegressor", "RandomForestRegressor"}:
            return pipe
    return None


def member_feature_names(member_pipeline) -> list[str]:
    columns = member_pipeline.named_steps["preprocess"].named_steps["columns"]
    return list(columns.get_feature_names_out())


def short_name(name: str) -> str:
    """Strip the ColumnTransformer prefix: 'continuous__GrLivArea' -> 'GrLivArea'."""
    return name.split("__", 1)[-1]


# --------------------------------------------------------------------------
# Scoring (the one official test evaluation)
# --------------------------------------------------------------------------


def prediction_frame(X: pd.DataFrame, y: pd.Series, pred) -> pd.DataFrame:
    """One row per house: actual, predicted, residuals, plus the context columns."""
    y = pd.Series(y, index=X.index, dtype=float)
    pred = pd.Series(np.asarray(pred, float), index=X.index)
    residual_log = np.log1p(y) - np.log1p(pred)
    out = pd.DataFrame(
        {
            "actual": y,
            "predicted": pred,
            "residual": y - pred,
            "abs_error": (y - pred).abs(),
            "pct_error": 100.0 * (y - pred) / y,
            "residual_log": residual_log,
            "abs_log_error": residual_log.abs(),
        },
        index=X.index,
    )
    for col in CONTEXT_COLUMNS:
        if col in X.columns:
            out[col] = X[col].to_numpy()
    return out


def member_dollar_predictions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Each blend member's dollar predictions (empty for a single model)."""
    inner = inner_estimator(model)
    named = getattr(inner, "named_estimators_", None)
    if not named:
        return pd.DataFrame(index=X.index)
    return pd.DataFrame(
        {name: np.expm1(est.predict(X)) for name, est in named.items()}, index=X.index
    )


def score_split(model, X: pd.DataFrame, y: pd.Series, name: str) -> dict:
    """RMSE(log) / MAE / R2 for one split. Does not fit anything."""
    pred = np.asarray(model.predict(X), float)
    metrics = train.fold_metrics(y, pred)
    metrics["n"] = int(len(y))
    metrics["split"] = name
    return metrics, pred


# --------------------------------------------------------------------------
# Linear-assumption checks (Ridge member, training rows)
# --------------------------------------------------------------------------


def design_matrix(member_pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    names = member_feature_names(member_pipeline)
    matrix = np.asarray(member_pipeline.named_steps["preprocess"].transform(X), float)
    return matrix, names


def variance_inflation_factors(matrix: np.ndarray, names: list[str]) -> pd.Series:
    """VIF for each column of the design matrix (no extra dependency).

    Columns are standardized, then VIF_j = (R^{-1})_{jj}. A constant column
    (zero variance) is reported as inf. High VIF is expected with one-hot columns
    that sum to 1: that is why Phase 4 kept Ridge instead of plain least squares.
    """
    x = np.asarray(matrix, float)
    n, p = x.shape
    std = x.std(axis=0, ddof=0)
    keep = std > 1e-12
    z = np.zeros((n, p))
    z[:, keep] = (x[:, keep] - x[:, keep].mean(axis=0)) / std[keep]
    # Tiny jitter so duplicated one-hot columns (exact collinearity) invert to a huge
    # finite VIF instead of the small numbers np.linalg.pinv would return.
    corr = z.T @ z / max(n, 1) + np.eye(p) * 1e-8
    inv = np.linalg.pinv(corr)
    vif = np.diag(inv).copy()
    vif[~keep] = np.inf
    return pd.Series(vif, index=names, name="vif")


def cooks_distance_ridge(
    matrix: np.ndarray, residuals: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Ridge Cook's distance and leverage (hat diagonal).

    The intercept is unpenalized, matching sklearn's Ridge(fit_intercept=True).
    Threshold commonly used: 4 / n.
    """
    x = np.asarray(matrix, float)
    e = np.asarray(residuals, float)
    n, p = x.shape
    xa = np.hstack([np.ones((n, 1)), x])
    penalty = np.diag([0.0] + [float(alpha)] * p)
    a_inv = np.linalg.pinv(xa.T @ xa + penalty)
    leverage = np.einsum("ij,ji->i", xa @ a_inv, xa.T)
    mse = float(np.mean(e**2))
    p_params = p + 1
    safe = np.clip(1.0 - leverage, 1e-12, None)
    cooks = (e**2 / (p_params * mse)) * (leverage / safe**2)
    return cooks, leverage


def residual_pattern_checks(residual_log: np.ndarray, predicted: np.ndarray) -> dict:
    """Cheap tests you can actually run without a statsmodels OLS object.

    - Shapiro-Wilk on a subsample: are log-residuals roughly normal?
    - Spearman |residual| vs predicted: does spread grow with price (hetero)?
    """
    r = np.asarray(residual_log, float)
    p = np.asarray(predicted, float)
    sample = r if len(r) <= 500 else r[np.linspace(0, len(r) - 1, 500).astype(int)]
    shap_stat, shap_p = stats.shapiro(sample)
    rho, rho_p = stats.spearmanr(p, np.abs(r))
    return {
        "shapiro_stat": float(shap_stat),
        "shapiro_p": float(shap_p),
        "hetero_spearman": float(rho),
        "hetero_p": float(rho_p),
        "residual_mean": float(np.mean(r)),
        "residual_std": float(np.std(r, ddof=1)),
    }


# --------------------------------------------------------------------------
# Feature drivers
# --------------------------------------------------------------------------


def ridge_coefficients(member_pipeline) -> pd.DataFrame:
    """Log-price coefficients of the linear member, plus an approximate % effect.

    Numeric columns are standardized, so a coefficient is the change in log-price
    for a 1-standard-deviation move. One-hot columns are 0/1. The % column is
    (exp(coef) - 1) * 100: a multiplicative change in price, holding the rest fixed.
    All dummy levels are kept (no drop-first), so neighborhood effects are relative
    to the intercept, not to a dropped reference level.
    """
    model = member_pipeline.named_steps["model"]
    names = member_feature_names(member_pipeline)
    coef = np.asarray(model.coef_, float)
    out = pd.DataFrame(
        {
            "feature": names,
            "short": [short_name(n) for n in names],
            "group": [n.split("__", 1)[0] if "__" in n else "other" for n in names],
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "pct_effect": 100.0 * (np.expm1(coef)),
        }
    )
    return out.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def tree_importances(member_pipeline) -> pd.DataFrame:
    model = member_pipeline.named_steps["model"]
    names = member_feature_names(member_pipeline)
    imp = np.asarray(model.feature_importances_, float)
    out = pd.DataFrame(
        {
            "feature": names,
            "short": [short_name(n) for n in names],
            "importance": imp,
        }
    )
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def permutation_drivers(model, X: pd.DataFrame, y: pd.Series, n_repeats: int) -> pd.DataFrame:
    """How much test RMSE(log) gets worse when each *raw* column is shuffled.

    This is the model-agnostic story for a blend: it lives in the original feature
    space, so OverallQual is one number rather than a mix of processed columns.
    The model is already frozen; shuffling test columns does not change selection.
    """
    result = permutation_importance(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_log_error",
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=config.N_JOBS,
    )
    # sklearn reports importance as a *drop in the score*. The score is negative
    # RMSE, so a useful column has a positive importance (RMSE got worse).
    out = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------


def miss_reason(row: pd.Series) -> str:
    """A short, human tag for why this house is a hard case (not a model bug)."""
    tags = []
    sale = str(row.get("SaleCondition", ""))
    if sale in UNUSUAL_SALES:
        tags.append(f"unusual sale ({sale})")
    elif sale == "Partial":
        tags.append("new-home / partial sale")
    sale_type = str(row.get("SaleType", ""))
    if sale_type in {"New", "COD", "CWD", "Con", "ConLD", "ConLI", "ConLw", "Oth"}:
        if sale_type != "WD":
            tags.append(f"sale type {sale_type}")
    qual = row.get("OverallQual")
    if pd.notna(qual) and qual <= 4:
        tags.append(f"low quality ({int(qual)})")
    elif pd.notna(qual) and qual >= 9:
        tags.append(f"very high quality ({int(qual)})")
    area = row.get("GrLivArea")
    if pd.notna(area) and area >= 3000:
        tags.append(f"very large living area ({int(area)} sqft)")
    pct = row.get("pct_error")
    if pd.notna(pct) and pct < -20:
        tags.append("over-predicted")
    elif pd.notna(pct) and pct > 20:
        tags.append("under-predicted")
    return "; ".join(tags) if tags else "no obvious flag (idiosyncratic price)"


def worst_errors(frame: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """The houses with the largest |log residual|, tagged with a why."""
    cols = [
        c
        for c in [
            "actual",
            "predicted",
            "residual",
            "pct_error",
            "residual_log",
            *CONTEXT_COLUMNS,
        ]
        if c in frame.columns
    ]
    out = frame.loc[frame["abs_log_error"].nlargest(n).index, cols].copy()
    out["why"] = out.apply(miss_reason, axis=1)
    return out


def error_by_group(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Mean |log error| and count for each level of a categorical (or a quality rating)."""
    if column not in frame.columns:
        return pd.DataFrame()
    grouped = frame.groupby(column, observed=True, dropna=False)
    out = grouped.agg(
        n=("abs_log_error", "size"),
        mean_abs_log_error=("abs_log_error", "mean"),
        mean_pct_error=("pct_error", "mean"),
    )
    return out.sort_values("mean_abs_log_error", ascending=False)


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------


def _metrics_row(metrics: dict) -> pd.Series:
    return pd.Series(
        {
            "split": metrics["split"],
            "n": metrics["n"],
            "rmse_log": metrics["rmse_log"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
        }
    )


def run(n_repeats: int | None = None, save: bool = True, model_path=None) -> dict:
    """Score the frozen model on the test split and compute every diagnostic table."""
    n_repeats = config.PERMUTATION_REPEATS if n_repeats is None else n_repeats
    model = load_model(model_path)
    X_test, y_test = data.load_test()
    X_train, y_train = data.load_train()

    test_metrics, test_pred = score_split(model, X_test, y_test, "test")
    train_metrics, train_pred = score_split(model, X_train, y_train, "train_resubstitution")
    logger.info(
        "TEST (once): n=%d  RMSE(log)=%.4f  MAE=$%.0f  R2=%.3f",
        test_metrics["n"],
        test_metrics["rmse_log"],
        test_metrics["mae"],
        test_metrics["r2"],
    )

    test_frame = prediction_frame(X_test, y_test, test_pred)
    train_frame = prediction_frame(X_train, y_train, train_pred)
    members = member_dollar_predictions(model, X_test)
    for col in members.columns:
        test_frame[f"pred_{col}"] = members[col].to_numpy()

    checks = residual_pattern_checks(
        test_frame["residual_log"].to_numpy(), test_frame["predicted"].to_numpy()
    )
    ridge_pipe = linear_member(model)
    tree_pipe = tree_member(model)

    coefs = ridge_coefficients(ridge_pipe) if ridge_pipe is not None else pd.DataFrame()
    importances = tree_importances(tree_pipe) if tree_pipe is not None else pd.DataFrame()
    logger.info(
        "Permutation importance on %d test columns x %d repeats", X_test.shape[1], n_repeats
    )
    drivers = permutation_drivers(model, X_test, y_test, n_repeats=n_repeats)

    vif = pd.DataFrame()
    influence = pd.DataFrame()
    if ridge_pipe is not None:
        matrix, names = design_matrix(ridge_pipe, X_train)
        vif = variance_inflation_factors(matrix, names).sort_values(ascending=False).rename("vif")
        vif = vif.rename_axis("feature").reset_index()
        intercept = float(ridge_pipe.named_steps["model"].intercept_)
        # Training residuals on the log scale the Ridge member actually fit.
        ridge_log_pred = ridge_pipe.predict(X_train)
        ridge_resid = np.log1p(y_train.to_numpy()) - ridge_log_pred
        alpha = float(getattr(ridge_pipe.named_steps["model"], "alpha", 0.0))
        cooks, leverage = cooks_distance_ridge(matrix, ridge_resid, alpha)
        influence = pd.DataFrame(
            {
                "cooks_distance": cooks,
                "leverage": leverage,
                "ridge_log_residual": ridge_resid,
                "actual": y_train.to_numpy(),
                "predicted_dollars": np.expm1(ridge_log_pred),
            },
            index=X_train.index,
        )
        for col in CONTEXT_COLUMNS:
            if col in X_train.columns:
                influence[col] = X_train[col].to_numpy()
        influence["flag"] = influence["cooks_distance"] > (4.0 / len(influence))
        logger.info(
            "Ridge intercept=%.3f  VIF>%.0f: %d/%d columns  Cook's > 4/n: %d/%d rows",
            intercept,
            VIF_HIGH,
            int((vif["vif"] > VIF_HIGH).sum()),
            len(vif),
            int(influence["flag"].sum()),
            len(influence),
        )

    worst = worst_errors(test_frame)
    by_sale = error_by_group(test_frame, "SaleCondition")
    by_qual = error_by_group(test_frame, "OverallQual")
    by_nbhd = error_by_group(test_frame, "Neighborhood")

    saved_cv = train.load_saved()
    winner = saved_cv["winner"] if saved_cv else "unknown"
    cv_rmse = (
        float(saved_cv["summary"].set_index("model").loc[winner, "cv_rmse_log"])
        if saved_cv
        else np.nan
    )

    metrics_table = pd.DataFrame(
        [
            {
                "split": "cv_train_5fold",
                "n": len(X_train),
                "rmse_log": cv_rmse,
                "mae": (
                    float(saved_cv["summary"].set_index("model").loc[winner, "cv_mae"])
                    if saved_cv
                    else np.nan
                ),
                "r2": (
                    float(saved_cv["summary"].set_index("model").loc[winner, "cv_r2"])
                    if saved_cv
                    else np.nan
                ),
            },
            _metrics_row(test_metrics),
            _metrics_row(train_metrics),
        ]
    )

    out = {
        "model": model,
        "winner": winner,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "metrics": metrics_table,
        "test": test_metrics,
        "train_resubstitution": train_metrics,
        "cv_rmse_log": cv_rmse,
        "test_frame": test_frame,
        "train_frame": train_frame,
        "checks": checks,
        "coefficients": coefs,
        "tree_importances": importances,
        "drivers": drivers,
        "vif": vif,
        "influence": influence,
        "worst": worst,
        "by_sale": by_sale,
        "by_qual": by_qual,
        "by_neighborhood": by_nbhd,
    }
    if save:
        save_results(out)
    return out


# --------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------


def results_markdown(out: dict) -> str:
    test, cv = out["test"], out["cv_rmse_log"]
    delta = "" if not np.isfinite(cv) else f" ({(test['rmse_log'] / cv - 1) * 100:+.1f}% vs CV)"
    lines = [
        "| Split | n | RMSE (log) | MAE ($) | R2 |",
        "|---|---|---|---|---|",
    ]
    for r in out["metrics"].itertuples():
        rmse = "—" if not np.isfinite(r.rmse_log) else f"{r.rmse_log:.4f}"
        mae = "—" if not np.isfinite(r.mae) else f"{r.mae:,.0f}"
        r2 = "—" if not np.isfinite(r.r2) else f"{r.r2:.3f}"
        lines.append(f"| {r.split} | {r.n} | {rmse} | {mae} | {r2} |")
    top = out["drivers"].head(8)
    driver_lines = ", ".join(
        f"{r.feature} ({r.importance:.4f})" for r in top.itertuples()
    )
    return (
        f"Held-out test scored **once** with the Phase 4 winner (**{out['winner']}**).\n\n"
        + "\n".join(lines)
        + f"\n\nTest RMSE(log) = **{test['rmse_log']:.4f}**{delta}. "
        + "`train_resubstitution` refits nothing: it is the same frozen model scored on the "
        + "rows it was trained on, so it is optimistic (especially for the tree member).\n\n"
        + f"Top raw features by permutation importance (test): {driver_lines}."
    )


def update_readme(block: str, path=None) -> bool:
    path = path or config.PROJECT_ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        logger.warning(
            "README has no %s / %s markers; test table not updated", README_START, README_END
        )
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(f"{head}{README_START}\n{block}\n{README_END}{tail}", encoding="utf-8")
    return True


def save_results(out: dict) -> None:
    config.ensure_dirs()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    out["metrics"].to_csv(config.REPORTS_DIR / "test_metrics.csv", index=False)
    out["test_frame"].to_csv(config.REPORTS_DIR / "test_predictions.csv")
    out["worst"].to_csv(config.REPORTS_DIR / "test_worst_errors.csv")
    out["drivers"].to_csv(config.REPORTS_DIR / "feature_drivers.csv", index=False)
    if len(out["coefficients"]):
        out["coefficients"].to_csv(config.REPORTS_DIR / "ridge_coefficients.csv", index=False)
    if len(out["tree_importances"]):
        out["tree_importances"].to_csv(config.REPORTS_DIR / "tree_importances.csv", index=False)
    if len(out["vif"]):
        out["vif"].to_csv(config.REPORTS_DIR / "vif.csv", index=False)
    if len(out["influence"]):
        out["influence"].to_csv(config.REPORTS_DIR / "cooks_distance.csv")
    for name, table in [
        ("error_by_sale_condition", out["by_sale"]),
        ("error_by_overall_qual", out["by_qual"]),
        ("error_by_neighborhood", out["by_neighborhood"]),
    ]:
        if len(table):
            table.to_csv(config.REPORTS_DIR / f"{name}.csv")

    (config.REPORTS_DIR / "test_score.json").write_text(
        json.dumps(
            {
                "winner": out["winner"],
                "n_test": out["test"]["n"],
                "rmse_log": out["test"]["rmse_log"],
                "mae": out["test"]["mae"],
                "r2": out["test"]["r2"],
                "cv_rmse_log": out["cv_rmse_log"] if np.isfinite(out["cv_rmse_log"]) else None,
                "checks": out["checks"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    block = results_markdown(out)
    n_vif = int((out["vif"]["vif"] > VIF_HIGH).sum()) if len(out["vif"]) else 0
    n_cooks = int(out["influence"]["flag"].sum()) if len(out["influence"]) else 0
    worst_cols = ["actual", "predicted", "pct_error", "SaleCondition", "Neighborhood", "why"]
    worst_md = out["worst"][[c for c in worst_cols if c in out["worst"]]].round(1)
    (config.DOCS_DIR / "evaluation_results.md").write_text(
        "# Evaluation results (Phase 5)\n\n"
        f"{block}\n\n"
        "## Linear-assumption checks (Ridge member, training rows)\n\n"
        f"- Columns with VIF > {VIF_HIGH:g}: **{n_vif}**. Expected with one-hot neighborhoods; "
        "Ridge shrinks the overlapping coefficients rather than dropping them.\n"
        f"- Training rows with Cook's distance > 4/n: **{n_cooks}**. These are influential, "
        "not automatically errors; they were not dropped.\n"
        f"- Test log-residuals: mean {out['checks']['residual_mean']:.4f}, "
        f"std {out['checks']['residual_std']:.4f}. "
        f"Shapiro p={out['checks']['shapiro_p']:.3g}; "
        f"Spearman |residual| vs predicted ρ={out['checks']['hetero_spearman']:.3f} "
        f"(p={out['checks']['hetero_p']:.3g}).\n\n"
        "## Houses the model gets most wrong (test)\n\n"
        f"```\n{worst_md.to_string()}\n```\n",
        encoding="utf-8",
    )
    update_readme(block)
    logger.info("Saved test score, diagnostic tables and README block")


def load_saved() -> dict | None:
    """Reload what `save_results` wrote, or None if Phase 5 has not been run yet."""
    needed = ["test_metrics.csv", "test_predictions.csv", "test_score.json", "feature_drivers.csv"]
    if not all((config.REPORTS_DIR / f).exists() for f in needed):
        return None
    score = json.loads((config.REPORTS_DIR / "test_score.json").read_text(encoding="utf-8"))
    coef_path = config.REPORTS_DIR / "ridge_coefficients.csv"
    tree_path = config.REPORTS_DIR / "tree_importances.csv"
    vif_path = config.REPORTS_DIR / "vif.csv"
    cook_path = config.REPORTS_DIR / "cooks_distance.csv"
    return {
        "metrics": pd.read_csv(config.REPORTS_DIR / "test_metrics.csv"),
        "test_frame": pd.read_csv(config.REPORTS_DIR / "test_predictions.csv", index_col=0),
        "drivers": pd.read_csv(config.REPORTS_DIR / "feature_drivers.csv"),
        "worst": pd.read_csv(config.REPORTS_DIR / "test_worst_errors.csv", index_col=0),
        "coefficients": pd.read_csv(coef_path) if coef_path.exists() else pd.DataFrame(),
        "tree_importances": pd.read_csv(tree_path) if tree_path.exists() else pd.DataFrame(),
        "vif": pd.read_csv(vif_path) if vif_path.exists() else pd.DataFrame(),
        "influence": pd.read_csv(cook_path, index_col=0) if cook_path.exists() else pd.DataFrame(),
        "winner": score["winner"],
        "test": {
            "split": "test",
            "n": score["n_test"],
            "rmse_log": score["rmse_log"],
            "mae": score["mae"],
            "r2": score["r2"],
        },
        "cv_rmse_log": score.get("cv_rmse_log") if score.get("cv_rmse_log") is not None else np.nan,
        "checks": score.get("checks", {}),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Phase 5: score the held-out test split once")
    parser.add_argument(
        "--repeats",
        type=int,
        default=config.PERMUTATION_REPEATS,
        help="permutation-importance repeats (default from config)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    out = run(n_repeats=args.repeats, save=True)
    print()
    print(out["metrics"].round(4).to_string(index=False))
    print(f"\nWinner: {out['winner']}")
    print("Top permutation drivers:")
    print(out["drivers"].head(8).round(4).to_string(index=False))
    print("\nWorst test errors:")
    cols = [c for c in ["actual", "predicted", "pct_error", "SaleCondition", "why"]
            if c in out["worst"]]  # fmt: skip
    print(out["worst"][cols].round(1).to_string())


if __name__ == "__main__":
    main()
