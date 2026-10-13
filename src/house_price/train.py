"""Phase 4: honest model comparison with nested cross-validation.

Why nested? If we pick the best alpha with cross-validation and then quote that same
cross-validation score, the number is a little too good: the settings were chosen because
they scored well on those very folds. Nested cross-validation fixes this with two loops:

    outer loop (5 folds): hold out one fold to score the model
        inner loop (3-5 folds, training rows only): GridSearchCV / RandomizedSearchCV picks settings
        refit the winner on the outer training rows, score it once on the held-out fold

The outer folds are the same ones used in Phase 2 and 3, so every score in the project is
directly comparable. The test split is never loaded here.

Run:  python -m house_price.train            (about 10 minutes; use --n-jobs 2 or more to speed up)
      python -m house_price.train --quick    (tiny searches, seconds, nothing is saved)
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from house_price import baseline, config, data, models

logger = logging.getLogger(__name__)

REFERENCE_NAME = f"Ridge (alpha={config.RIDGE_ALPHA:g}, untuned)"  # the Phase 3 model
README_START, README_END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


@dataclass
class NestedResult:
    """Everything one nested cross-validation run produces for one model."""

    name: str
    folds: pd.DataFrame  # one row per outer fold: rmse_log, mae, r2, train_rmse_log, inner_rmse_log
    oof_log: pd.Series  # out-of-fold predictions on the log1p scale, indexed like X
    fold_of_row: pd.Series  # which outer fold scored each row
    settings: list[dict]  # winning settings chosen inside each outer fold
    seconds: float = 0.0


def fold_metrics(y_true, y_pred) -> dict:
    """RMSE on log price (primary), plus MAE and R2 on the price scale."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    with np.errstate(all="ignore"):
        rmse_log = float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))
    if not np.isfinite(y_pred).all() or not np.isfinite(rmse_log):
        return {"rmse_log": np.nan, "mae": np.nan, "r2": np.nan}  # unstable model, not a score
    return {
        "rmse_log": rmse_log,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_nested(
    spec: models.ModelSpec, X, y, outer=None, n_jobs=config.N_JOBS
) -> NestedResult:
    """Nested cross-validation for one candidate (see the module docstring)."""
    outer = outer or baseline.get_cv()
    start = time.time()
    rows, settings = [], []
    oof = pd.Series(np.nan, index=X.index, name=spec.name)
    fold_of_row = pd.Series(-1, index=X.index, name="fold")

    for k, (train_idx, valid_idx) in enumerate(outer.split(X), start=1):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]

        estimator = models.build_search(spec, n_jobs=n_jobs)
        with np.errstate(all="ignore"):
            estimator.fit(X_tr, y_tr)
            pred_va, pred_tr = estimator.predict(X_va), estimator.predict(X_tr)

        row = fold_metrics(y_va, pred_va)
        row["train_rmse_log"] = fold_metrics(y_tr, pred_tr)["rmse_log"]
        row["inner_rmse_log"] = -getattr(estimator, "best_score_", np.nan)
        rows.append({"fold": k, **row})
        settings.append(models.best_settings(estimator))
        with np.errstate(all="ignore"):
            oof.iloc[valid_idx] = np.log1p(pred_va)
        fold_of_row.iloc[valid_idx] = k
        logger.info("%s fold %d/%d: RMSE(log)=%.4f  settings=%s",
                    spec.name, k, outer.get_n_splits(), row["rmse_log"], settings[-1])  # fmt: skip

    seconds = time.time() - start
    return NestedResult(spec.name, pd.DataFrame(rows), oof, fold_of_row, settings, seconds)


def blend_result(name: str, members: list[NestedResult], y: pd.Series) -> NestedResult:
    """Average the members' out-of-fold predictions (on the log scale) and score per fold.

    No refitting: each member's prediction for a row already came from a model that never
    saw that row, so averaging them is still an honest out-of-fold estimate.
    """
    oof_log = sum(m.oof_log for m in members) / len(members)
    fold_of_row = members[0].fold_of_row
    rows = []
    for k in sorted(fold_of_row.unique()):
        mask = fold_of_row == k
        row = fold_metrics(y[mask], np.expm1(oof_log[mask]))
        rows.append({"fold": int(k), **row, "train_rmse_log": np.nan, "inner_rmse_log": np.nan})
    return NestedResult(name, pd.DataFrame(rows), oof_log.rename(name), fold_of_row, [])


def summarize(results: list[NestedResult]) -> pd.DataFrame:
    """One row per model, in the same layout as the Phase 2 and 3 tables."""
    rows = []
    for r in results:
        f = r.folds
        rows.append(
            {
                "model": r.name,
                "cv_rmse_log": f["rmse_log"].mean(),
                "cv_rmse_log_std": f["rmse_log"].std(ddof=0),
                "train_rmse_log": f["train_rmse_log"].mean(),
                # The inner score uses smaller training sets, so it is not comparable to the rest.
                "inner_rmse_log": f["inner_rmse_log"].mean(),
                "cv_mae": f["mae"].mean(),
                "cv_r2": f["r2"].mean(),
                "seconds": round(r.seconds, 1),
            }
        )
    out = pd.DataFrame(rows).drop(columns=["inner_rmse_log"])
    ref = out.loc[out["model"] == REFERENCE_NAME, "cv_rmse_log"]
    if len(ref):
        out["vs_phase3_pct"] = (out["cv_rmse_log"] / ref.iloc[0] - 1) * 100
    return out


def paired_comparison(results: list[NestedResult], reference: str) -> pd.DataFrame:
    """Is each model really different from `reference`, or is it fold-to-fold noise?

    All models are scored on the same folds, so we compare them fold by fold. If the 95%
    interval of the mean difference includes 0, the models are statistically tied.
    """
    by_name = {r.name: r.folds["rmse_log"].to_numpy() for r in results}
    ref = by_name[reference]
    k = len(ref)
    t_crit = stats.t.ppf(0.975, k - 1)
    rows = []
    for name, scores in by_name.items():
        if name == reference:
            continue
        diff = scores - ref  # negative = better than the reference
        if not np.isfinite(diff).all():
            rows.append({"model": name, "mean_diff": np.nan, "ci95": np.nan,
                         "folds_better": 0, "verdict": "unstable"})  # fmt: skip
            continue
        se = diff.std(ddof=1) / np.sqrt(k)
        half = t_crit * se
        verdict = "tie" if abs(diff.mean()) <= half else ("better" if diff.mean() < 0 else "worse")
        rows.append({"model": name, "mean_diff": diff.mean(), "ci95": half,
                     "folds_better": int((diff < 0).sum()), "verdict": verdict})  # fmt: skip
    return pd.DataFrame(rows)


def select_best(summary: pd.DataFrame, results: list[NestedResult], specs: dict) -> str:
    """One-standard-error rule: of the models within one standard error of the best score,
    take the simplest. A tie in accuracy goes to the model that is easier to explain and less
    likely to overfit.
    """
    scored = summary[summary["model"].isin(specs) & summary["cv_rmse_log"].notna()]
    best = scored.loc[scored["cv_rmse_log"].idxmin()]
    best_folds = next(r for r in results if r.name == best["model"]).folds["rmse_log"]
    se = best_folds.std(ddof=1) / np.sqrt(len(best_folds))
    eligible = scored[scored["cv_rmse_log"] <= best["cv_rmse_log"] + se]
    return min(eligible["model"], key=lambda name: specs[name].complexity)


def tuning_table(name: str, fitted_search) -> pd.DataFrame:
    """Every setting a search tried and its inner-CV score."""
    if not hasattr(fitted_search, "cv_results_"):
        return pd.DataFrame()
    cv = pd.DataFrame(fitted_search.cv_results_)
    out = pd.DataFrame(
        [{k.removeprefix(models.PARAM_PREFIX): v for k, v in p.items()} for p in cv["params"]]
    )
    out.insert(0, "model", name)
    out["inner_rmse_log"] = -cv["mean_test_score"].to_numpy()
    out["inner_rmse_log_std"] = cv["std_test_score"].to_numpy()
    out["rank"] = cv["rank_test_score"].to_numpy()
    return out


def grid_edge_warnings(spec: models.ModelSpec, settings: dict) -> list[str]:
    """Warn when a winning value sits at the edge of a numeric grid (the best may lie outside)."""
    if spec.search != "grid":
        return []
    out = []
    for name, values in spec.grid.items():
        numeric = [v for v in values if isinstance(v, (int, float))]
        if len(numeric) == len(values) and len(values) > 2 and name in settings:
            if settings[name] in (min(values), max(values)):
                out.append(f"{spec.name}: best {name}={settings[name]:.3g} is at the grid edge")
    return out


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------


def run_all(quick=False, n_jobs=config.N_JOBS, only=None, save=True) -> dict:
    """Nested CV for every candidate, selection, final fits, and (optionally) saved artifacts."""
    X, y = data.load_train()
    specs = models.get_specs(quick)
    if only:
        specs = {k: v for k, v in specs.items() if k in only}
    outer = baseline.get_cv()
    logger.info("Training rows: %d | outer folds: %d | quick=%s", len(X), outer.n_splits, quick)

    reference = models.ModelSpec(
        REFERENCE_NAME, lambda: Ridge(alpha=config.RIDGE_ALPHA), complexity=1
    )
    results = [evaluate_nested(reference, X, y, outer, n_jobs)]
    for spec in specs.values():
        results.append(evaluate_nested(spec, X, y, outer, n_jobs))
    by_name = {r.name: r for r in results}

    candidates = dict(specs)
    has_blend = all(m in by_name for m in models.BLEND_MEMBERS)
    if has_blend:
        members = [by_name[m] for m in models.BLEND_MEMBERS]
        results.append(blend_result(models.BLEND_NAME, members, y))
        # The blend is the most complex candidate: it needs both of its members.
        top = max(s.complexity for s in specs.values()) + 1
        candidates[models.BLEND_NAME] = models.ModelSpec(
            models.BLEND_NAME, lambda: None, family="blend", complexity=top
        )

    summary = summarize(results)
    winner = select_best(summary, results, candidates)
    paired = paired_comparison(results, reference=winner)

    # Final fits on the full training split: the saved model, plus the tuning curves.
    finals, final_settings, tuning, warnings_ = {}, {}, [], []
    for name, spec in specs.items():
        search = models.build_search(spec, n_jobs=n_jobs)
        with np.errstate(all="ignore"):
            search.fit(X, y)
        finals[name] = getattr(search, "best_estimator_", search)
        final_settings[name] = models.best_settings(search)
        tuning.append(tuning_table(name, search))
        warnings_ += grid_edge_warnings(spec, final_settings[name])
    if has_blend:
        finals[models.BLEND_NAME] = models.build_blend(specs, final_settings).fit(X, y)
        final_settings[models.BLEND_NAME] = {m: final_settings[m] for m in models.BLEND_MEMBERS}
    tables = [t for t in tuning if len(t)]
    tuning = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()

    out = {
        "results": results, "summary": summary, "paired": paired, "winner": winner,
        "settings": final_settings, "finals": finals, "tuning": tuning,
        "grid_warnings": warnings_, "specs": candidates,
    }  # fmt: skip
    if save:
        save_results(out)
    return out


# --------------------------------------------------------------------------
# Saving and loading
# --------------------------------------------------------------------------


def settings_text(name: str, settings: dict, specs: dict) -> str:
    """Short human-readable settings for the results table."""
    if name == REFERENCE_NAME:
        return f"alpha={config.RIDGE_ALPHA:g} (fixed)"
    if name == models.BLEND_NAME:
        return "equal-weight average of " + " and ".join(models.BLEND_MEMBERS)
    chosen = settings.get(name, {})
    if not chosen:
        return "no hyperparameters"
    parts = [f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in chosen.items()]
    return ", ".join(parts)


def results_markdown(out: dict) -> str:
    """The results table for the README and docs: model, settings, CV RMSE and its std."""
    lines = [
        "| Model | Tuned settings | CV RMSE (log) | Std | CV MAE ($) | CV R2 |",
        "|---|---|---|---|---|---|",
    ]
    for r in out["summary"].itertuples():
        label = f"**{r.model}** (selected)" if r.model == out["winner"] else r.model
        lines.append(
            f"| {label} | {settings_text(r.model, out['settings'], out['specs'])} | "
            f"{r.cv_rmse_log:.4f} | {r.cv_rmse_log_std:.4f} | {r.cv_mae:,.0f} | {r.cv_r2:.3f} |"
        )
    return "\n".join(lines)


def update_readme(table_md: str, path=None) -> bool:
    """Replace the table between the RESULTS markers in the README. Returns False if absent."""
    path = path or config.PROJECT_ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        logger.warning("README has no %s / %s markers; table not updated", README_START, README_END)
        return False
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    path.write_text(f"{head}{README_START}\n{table_md}\n{README_END}{tail}", encoding="utf-8")
    return True


def save_results(out: dict) -> None:
    """Write the model, tables, tuning curves and README table."""
    config.ensure_dirs()
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    out["summary"].to_csv(config.REPORTS_DIR / "model_results.csv", index=False)
    out["paired"].to_csv(config.REPORTS_DIR / "model_paired_comparison.csv", index=False)
    pd.concat([r.folds.assign(model=r.name) for r in out["results"]]).to_csv(
        config.REPORTS_DIR / "model_cv_folds.csv", index=False
    )
    out["tuning"].to_csv(config.REPORTS_DIR / "tuning_results.csv", index=False)
    pd.DataFrame({r.name: np.expm1(r.oof_log) for r in out["results"]}).to_csv(
        config.REPORTS_DIR / "model_oof_predictions.csv"
    )
    (config.REPORTS_DIR / "best_params.json").write_text(
        json.dumps({"selected": out["winner"], "settings": out["settings"]}, indent=2),
        encoding="utf-8",
    )
    joblib.dump(out["finals"][out["winner"]], config.MODELS_DIR / "best_model.joblib")

    table = results_markdown(out)
    paired = out["paired"].round(4).to_string(index=False)
    (config.DOCS_DIR / "model_results.md").write_text(
        "# Model results (Phase 4)\n\n"
        "Nested 5-fold cross-validation on the training split. Settings were tuned inside each\n"
        "training fold only. Same outer folds as Phases 2 and 3.\n\n"
        f"{table}\n\n"
        f"Selected by the one-standard-error rule: **{out['winner']}**.\n\n"
        "## Paired comparison against the selected model\n\n"
        "Negative `mean_diff` means better than the selected model. `ci95` is the half-width of a\n"
        "95% interval, so a difference smaller than `ci95` is a statistical tie.\n\n"
        f"```\n{paired}\n```\n",
        encoding="utf-8",
    )
    update_readme(table)
    logger.info("Saved model, tables and README results (selected: %s)", out["winner"])


def load_saved() -> dict | None:
    """Reload what `save_results` wrote, or None if Phase 4 has not been run yet."""
    needed = ["model_results.csv", "model_cv_folds.csv", "tuning_results.csv", "best_params.json",
              "model_paired_comparison.csv", "model_oof_predictions.csv"]  # fmt: skip
    if not all((config.REPORTS_DIR / f).exists() for f in needed):
        return None
    params = json.loads((config.REPORTS_DIR / "best_params.json").read_text(encoding="utf-8"))
    return {
        "summary": pd.read_csv(config.REPORTS_DIR / "model_results.csv"),
        "paired": pd.read_csv(config.REPORTS_DIR / "model_paired_comparison.csv"),
        "folds": pd.read_csv(config.REPORTS_DIR / "model_cv_folds.csv"),
        "tuning": pd.read_csv(config.REPORTS_DIR / "tuning_results.csv"),
        "oof": pd.read_csv(config.REPORTS_DIR / "model_oof_predictions.csv", index_col=0),
        "winner": params["selected"],
        "settings": params["settings"],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Phase 4: nested-CV model comparison")
    parser.add_argument("--quick", action="store_true", help="tiny searches, nothing is saved")
    parser.add_argument("--n-jobs", type=int, default=config.N_JOBS, help="parallel workers")
    parser.add_argument("--models", default="", help="comma-separated subset, e.g. Ridge,Lasso")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    only = [m.strip() for m in args.models.split(",") if m.strip()] or None
    out = run_all(quick=args.quick, n_jobs=args.n_jobs, only=only, save=not args.quick and not only)

    print()
    print(out["summary"].round(4).to_string(index=False))
    print(f"\nSelected (one-standard-error rule): {out['winner']}")
    print(out["paired"].round(4).to_string(index=False))
    for warning in out["grid_warnings"]:
        print("WARNING:", warning)
    if args.quick or only:
        print("\n(quick or partial run: nothing was saved)")


if __name__ == "__main__":
    main()
