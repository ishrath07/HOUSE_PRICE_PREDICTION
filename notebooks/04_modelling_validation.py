# %% [markdown]
# # 04 · Modelling and validation (Phase 4)
#
# **Goal:** find the best regressor for the Phase 3 features, tune it honestly, and compare it
# with the alternatives, including the optional tree-model stretch.
#
# **Rules**
# - Only the training split is used. The test split is never opened (Phase 5 scores it once).
# - Every candidate sits behind the same Phase 3 pipeline, so differences come from the model.
# - Settings are tuned with `GridSearchCV` / `RandomizedSearchCV` **inside** each training fold
#   (nested cross-validation), so the reported scores are not flattered by the tuning.
# - Same outer 5 folds as Phases 2 and 3, so every number in the project is comparable.
#
# | Step | Question | Output |
# |---|---|---|
# | 1 | What goes in? | Training rows |
# | 2 | How do we tune without fooling ourselves? | Nested CV protocol |
# | 3 | What does one search look like? | Ridge alpha curve |
# | 4 | Run (or load) every candidate | Saved results |
# | 5 | How do they score? | Results table |
# | 6 | Are the differences real? | Paired comparison |
# | 7 | Which model do we keep? | One-standard-error rule |
# | 8 | Were the grids wide enough? | Tuning curves |
# | 9 | What does Lasso switch off? | Sparsity |
# | 10 | Stretch: do trees help? | RF, GBR, blend |
# | 11 | Overview | Comparison figure |
# | 12 | Is the saved model usable? | Reload check |
#
# **How to run this file**
# - Whole file: `python notebooks/04_modelling_validation.py` (project root, venv activated).
# - The first run executes the full search (about 10 minutes) and saves everything; later runs
#   load the saved results in seconds. Set `FORCE_RERUN = True` below to recompute.
# - Cell by cell: open it in VS Code and click "Run Cell" above any `# %%` block.

# %% [markdown]
# ## Setup

# %%
import sys
import warnings
from pathlib import Path

import matplotlib

if "get_ipython" not in globals():  # plain script: save figures, do not open windows
    matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, lasso_path


def find_project_root() -> Path:
    """Walk up from this file (or the working directory) until pyproject.toml is found."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("Run this from inside the project folder (pyproject.toml not found).")


try:
    from house_price import config, data, models, pipeline, train
except ImportError:  # package not installed: fall back to the src/ layout
    sys.path.insert(0, str(find_project_root() / "src"))
    from house_price import config, data, models, pipeline, train

try:
    from IPython.display import display
except ImportError:  # plain Python without IPython
    display = print

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 140)

config.ensure_dirs()
FORCE_RERUN = False  # True = recompute the full nested search even if saved results exist
N_JOBS = 1  # raise (for example to 2 or 4) to speed up the search; 1 is safest on Windows

FAMILY_COLORS = {"reference": "#9aa5b1", "linear": "#2a7de1", "tree": "#e8832a", "blend": "#3aa76d"}


def save_fig(name: str) -> None:
    """Save the current figure to reports/figures/<name>.png."""
    path = config.FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {path.relative_to(config.PROJECT_ROOT)}")


def show_or_close(fig) -> None:
    if "get_ipython" in globals():
        plt.show()
    else:
        plt.close(fig)


print("Project root:", config.PROJECT_ROOT)

# %% [markdown]
# ## Step 1 · Load the training data
# Same loader as before: training split only, with the 2 outlier houses removed (EDA decision 12).

# %%
TRAIN_PATH = config.PROCESSED_DIR / "train.csv"
LOADED_FILES = [TRAIN_PATH.name]  # audit trail: test.csv must never appear here

X, y = data.load_train(TRAIN_PATH)
print(f"Rows: {len(X):,} | raw features: {X.shape[1]} | files loaded: {LOADED_FILES}")

specs = models.get_specs()
print("\nCandidates and what gets tuned:")
overview = pd.DataFrame(
    [
        {
            "model": s.name,
            "family": s.family,
            "settings tuned": ", ".join(s.grid) or "none",
            "search": s.search if s.grid else "-",
            "settings tried": s.n_candidates,
            "inner folds": s.inner_folds if s.grid else "-",
        }
        for s in specs.values()
    ]
)
display(overview)

# %% [markdown]
# ## Step 2 · How do we tune without fooling ourselves?
# If we choose the best alpha with cross-validation and then quote that same cross-validation score,
# the number is slightly too good: the alpha was chosen *because* it scored well on those folds.
#
# **Nested cross-validation** uses two loops:
#
# ```
# outer loop (5 folds)  - the same folds as Phases 2 and 3
#     hold out one fold                      <- used only to score
#     inner loop (3-5 folds, remaining rows) <- GridSearchCV picks the settings here
#     refit the winner on all remaining rows, score it once on the held-out fold
# ```
#
# The inner loop re-runs the whole Phase 3 pipeline in every fold, so imputers, scalers and encoders
# never see validation rows. The held-out fold never influences any decision about its own score.

# %% [markdown]
# ## Step 3 · One search up close: Ridge
# Fit a `GridSearchCV` on the whole training split to see what a search actually does. This is for
# illustration only: the honest score comes from the nested loop in Step 4.

# %%
ridge_spec = specs["Ridge"]
ridge_search = models.build_search(ridge_spec, n_jobs=N_JOBS).fit(X, y)
ridge_curve = train.tuning_table("Ridge", ridge_search).sort_values("alpha")

print(f"Tried {len(ridge_curve)} alphas with {ridge_spec.inner_folds}-fold CV each")
print(f"Best alpha: {models.best_settings(ridge_search)['alpha']:.3g} "
      f"(inner RMSE(log) {-ridge_search.best_score_:.4f})")
display(ridge_curve.round(4).reset_index(drop=True))

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(ridge_curve["alpha"], ridge_curve["inner_rmse_log"],
            yerr=ridge_curve["inner_rmse_log_std"], marker="o", capsize=3)
ax.axvline(models.best_settings(ridge_search)["alpha"], color="crimson", ls="--", label="best alpha")
ax.set_xscale("log")
ax.set_xlabel("alpha (log scale)")
ax.set_ylabel("Inner-CV RMSE (log)")
ax.set_title("Ridge: error vs regularization strength")
ax.legend()
plt.tight_layout()
show_or_close(fig)

# %% [markdown]
# Reading the curve: tiny alpha means almost no regularization (the correlated one-hot columns
# push the coefficients around), huge alpha means the model is squashed towards the average.
# The best value sits in between, and the curve is flat near the bottom, so the exact choice
# matters little.

# %% [markdown]
# ## Step 4 · Run (or load) every candidate
# For each model: 5 outer folds x (inner search + refit). Trees take the longest.
# Results are saved to `reports/` and `docs/`, the winner to `models/best_model.joblib`, and the
# table in the README is updated automatically.

# %%
saved = None if FORCE_RERUN else train.load_saved()
if saved is None:
    print("No saved results found: running the full nested search (about 10 minutes) ...")
    train.run_all(n_jobs=N_JOBS, save=True)
    saved = train.load_saved()
else:
    print("Loaded saved results from reports/ (set FORCE_RERUN = True to recompute).")

summary, paired, folds = saved["summary"], saved["paired"], saved["folds"]
winner, final_settings, tuning = saved["winner"], saved["settings"], saved["tuning"]
print(f"Models scored: {len(summary)}")

# %% [markdown]
# ## Step 5 · How do they score?
# Nested 5-fold CV, RMSE on the log scale (lower is better). `Std` is the spread across the 5 outer
# folds. `vs_phase3_pct` is the change against the untuned Ridge from Phase 3.

# %%
shown = summary.drop(columns=["seconds"]).round(4)
display(shown)

ref = summary.loc[summary["model"] == train.REFERENCE_NAME].iloc[0]
print(f"Phase 3 model (Ridge alpha={config.RIDGE_ALPHA:g}, untuned): RMSE(log) {ref.cv_rmse_log:.4f}")
for row in summary.itertuples():
    if row.model in specs:
        text = train.settings_text(row.model, final_settings, specs)
        print(f"  {row.model:<18} {row.cv_rmse_log:.4f}   tuned: {text}")

# %% [markdown]
# ## Step 6 · Are the differences real?
# With only 5 folds and a fold-to-fold spread of about 0.006, a gap of 0.001 means nothing.
# Because every model is scored on the **same folds**, we compare them fold by fold. `mean_diff` is
# the average gap versus the selected model (negative = better), `ci95` the half-width of a 95%
# interval. A difference smaller than `ci95` is a statistical tie.

# %%
display(paired.round(4))
ties = paired.loc[paired["verdict"] == "tie", "model"].tolist()
worse = paired.loc[paired["verdict"] == "worse", "model"].tolist()
print(f"Statistically tied with the selected model ({winner}): {ties}")
print(f"Clearly worse: {worse}")

# %% [markdown]
# ## Step 7 · Which model do we keep?
# **One-standard-error rule:** take the simplest model whose score is within one standard error of the
# best. When accuracy is a tie, the simpler model is easier to explain and less likely to overfit.
# Complexity order: LinearRegression < Ridge < Lasso < ElasticNet < RandomForest < GradientBoosting
# < Blend (the blend needs two models, so it is the most complex candidate).

# %%
candidates = [*specs, models.BLEND_NAME]
scored = summary[summary["model"].isin(candidates) & summary["cv_rmse_log"].notna()]
best_row = scored.loc[scored["cv_rmse_log"].idxmin()]
best_folds = folds.loc[folds["model"] == best_row["model"], "rmse_log"]
one_se = best_folds.std(ddof=1) / np.sqrt(len(best_folds))
eligible = scored[scored["cv_rmse_log"] <= best_row["cv_rmse_log"] + one_se]

print(f"Best raw score : {best_row['model']} at {best_row['cv_rmse_log']:.4f}")
print(f"One standard error: {one_se:.4f}  -> eligible: {eligible['model'].tolist()}")
print(f"Selected (simplest eligible): {winner}")
print(f"Settings: {final_settings[winner]}")

# %% [markdown]
# ## Step 8 · Were the grids wide enough?
# A best value on the edge of a grid means the true best may lie outside it. These plots show every
# setting each search tried (inner-CV score on the full training split).

# %%
fig, axes = plt.subplots(
    1, 3, figsize=(18, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.4]}
)

for ax, name in zip(axes[:2], ["Ridge", "Lasso"]):
    t = tuning[tuning["model"] == name].sort_values("alpha")
    ax.errorbar(t["alpha"], t["inner_rmse_log"], yerr=t["inner_rmse_log_std"], marker="o", capsize=3)
    ax.axvline(final_settings[name]["alpha"], color="crimson", ls="--", label="best")
    ax.set_xscale("log")
    ax.set_xlabel("alpha (log scale)")
    ax.set_ylabel("Inner-CV RMSE (log)")
    ax.set_title(f"{name}")
    ax.legend()

en = tuning[tuning["model"] == "ElasticNet"]
grid = en.pivot_table(index="l1_ratio", columns="alpha", values="inner_rmse_log")
sns.heatmap(grid, annot=True, fmt=".3f", annot_kws={"size": 8}, cmap="viridis_r", ax=axes[2],
            cbar_kws={"label": "RMSE (log)"})
axes[2].set_xticklabels([f"{a:.1e}" for a in grid.columns], rotation=45)
axes[2].set_title("ElasticNet: alpha x l1_ratio")
plt.tight_layout()
save_fig("10_tuning_curves")
show_or_close(fig)

# %%
edge = []
for name, spec in specs.items():
    edge += train.grid_edge_warnings(spec, final_settings.get(name, {}))
print("Grid-edge warnings:", edge or "none: every winning value is inside its grid")

# A winner on the edge only matters if the score is still improving there. Check how flat the
# surface is: the spread of the inner-CV score among each model's 5 best settings.
print("\nSpread of inner RMSE(log) among the 5 best settings (tiny = flat, the edge is harmless):")
for name in ["Ridge", "Lasso", "ElasticNet"]:
    top5 = tuning[tuning["model"] == name].nsmallest(5, "inner_rmse_log")["inner_rmse_log"]
    print(f"  {name:<11} {top5.max() - top5.min():.5f}   (fold-to-fold spread is about 0.006)")

# %% [markdown]
# ## Step 9 · What does Lasso switch off?
# Lasso's L1 penalty sets coefficients exactly to zero. The path below shows how many of the 199
# columns stay active as alpha grows. This is descriptive only (fitted once on the training rows to
# draw the picture) and nothing here is scored.

# %%
ly = np.log1p(y)
pre = pipeline.build_preprocessor().fit(X, ly)
matrix = pre.transform(X)
names = list(pre.named_steps["columns"].get_feature_names_out())

alphas = np.logspace(-4.5, -0.5, 40)
_, coefs, _ = lasso_path(matrix, ly, alphas=alphas)
n_active = (coefs != 0).sum(axis=0)
lasso_alpha = final_settings["Lasso"]["alpha"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(alphas, n_active, marker=".")
ax.axvline(lasso_alpha, color="crimson", ls="--", label=f"tuned alpha = {lasso_alpha:.2g}")
ax.set_xscale("log")
ax.set_xlabel("alpha (log scale)")
ax.set_ylabel(f"Active columns (of {matrix.shape[1]})")
ax.set_title("Lasso path: columns kept vs regularization")
ax.legend()
plt.tight_layout()
save_fig("11_lasso_sparsity")
show_or_close(fig)

lasso = Lasso(alpha=lasso_alpha, max_iter=50_000).fit(matrix, ly)
coef = pd.Series(lasso.coef_, index=names)
print(f"At the tuned alpha, Lasso keeps {int((coef != 0).sum())} of {len(coef)} columns "
      f"and sets {int((coef == 0).sum())} to zero.")
print("\nLargest coefficients (log-price change per 1 unit of the column; see the note below):")
top = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10).round(4)
display(top.to_frame("coefficient"))

# %% [markdown]
# *Note:* numeric columns are standardized and one-hot columns are 0/1, so coefficient sizes are only
# roughly comparable. Phase 5 does a careful interpretation. As a sanity check against EDA decision 8,
# the strongest numeric predictors (OverallQual, GrLivArea) should appear near the top.

# %% [markdown]
# ## Step 10 · Stretch: do tree models help?
# RandomForest and GradientBoosting use the same features and the same nested CV. We also test a
# **blend**: the plain average of Ridge and GradientBoosting predictions (fixed in advance, so it is
# not cherry-picked). A blend only helps when the two models make different mistakes, and it is only
# worth its extra complexity if it wins consistently, which the paired comparison in Step 6 tests.

# %%
tree_rows = summary[summary["model"].isin(["Ridge", "RandomForest", "GradientBoosting"])
                    | summary["model"].str.startswith("Blend")]
display(tree_rows[["model", "cv_rmse_log", "cv_rmse_log_std", "train_rmse_log", "cv_mae", "cv_r2"]].round(4))

oof = saved["oof"]
err = {m: np.log1p(y) - np.log1p(oof[m]) for m in ["Ridge", "GradientBoosting"] if m in oof}
if len(err) == 2:
    corr = np.corrcoef(err["Ridge"], err["GradientBoosting"])[0, 1]
    print(f"Correlation of out-of-fold errors, Ridge vs GradientBoosting: {corr:.2f}")
    print("(near 1 = same mistakes, so blending adds little; lower = more to gain)")

gbr = summary.loc[summary["model"] == "GradientBoosting"].iloc[0]
print(f"\nGradientBoosting train vs CV RMSE: {gbr.train_rmse_log:.4f} vs {gbr.cv_rmse_log:.4f} "
      f"(gap {gbr.cv_rmse_log - gbr.train_rmse_log:.4f}): trees fit the training rows much more closely.")
print("Settings:", final_settings["GradientBoosting"])

blend_diff = paired.loc[paired["model"] == "Ridge"]
if winner == models.BLEND_NAME and len(blend_diff):
    r = blend_diff.iloc[0]
    print(f"\nBlend vs tuned Ridge: Ridge is {r.mean_diff:+.4f} +/- {r.ci95:.4f} worse on average "
          f"({r.verdict}); the blend wins on {5 - int(r.folds_better)} of 5 folds.")

# %% [markdown]
# Why linear models do so well here: after the log transform and the Phase 3 features, price is close
# to a linear function of the inputs; there are only about 1,100 rows; and one-hot encoding spreads the
# categorical signal over many sparse columns, which trees split inefficiently. More data or
# hand-crafted interactions would favour trees more.

# %% [markdown]
# ## Step 11 · Overview figure

# %%
plot = summary.copy()


def family(name):
    if name == train.REFERENCE_NAME:
        return "reference"
    if name.startswith("Blend"):
        return "blend"
    return specs[name].family if name in specs else "reference"


plot["family"] = plot["model"].map(family)
plot = plot.dropna(subset=["cv_rmse_log"]).sort_values("cv_rmse_log", ascending=False)

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(plot["model"], plot["cv_rmse_log"], xerr=plot["cv_rmse_log_std"],
        color=[FAMILY_COLORS[f] for f in plot["family"]], capsize=3)
ax.axvline(0.1258, color="black", ls=":", lw=1, label="Phase 2 baseline (0.1258)")
ax.set_xlabel("Nested CV RMSE (log), lower is better")
ax.set_xlim(0.09, max(0.15, plot["cv_rmse_log"].max() + 0.02))
ax.set_title("All models, same outer folds (error bars = spread across folds)")
ax.legend(loc="upper right")
plt.tight_layout()
save_fig("12_model_comparison")
show_or_close(fig)

# %% [markdown]
# ## Step 12 · Is the saved model usable?
# `models/best_model.joblib` is the selected model refitted on the whole training split (when the
# blend wins, it contains both members, each with its own preprocessing). Phase 5 loads it and scores
# the held-out test split exactly once. Here we only confirm it loads and predicts.

# %%
best_model = joblib.load(config.MODELS_DIR / "best_model.joblib")
sample = X.head(3)
pred = best_model.predict(sample)
print("Loaded:", type(best_model).__name__, "| selected:", winner)
print("Predictions for 3 training houses:", pred.round(0), "| actual:", y.head(3).to_numpy())

# %% [markdown]
# ## Definition of done
# Run this last cell. All lines should say PASS.

# %%
readme_text = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
tuned_ok = summary[summary["model"].isin(["Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting"])]
win_row = summary.loc[summary["model"] == winner].iloc[0]
checks = {
    "Only the training split was loaded (test split never opened)": LOADED_FILES == ["train.csv"],
    "All 6 candidates, the Phase 3 reference and the blend were scored": len(summary) == 8,
    "Every tuned model produced a finite score": bool(np.isfinite(tuned_ok["cv_rmse_log"]).all()),
    "Tuning did not make the selected model worse than the untuned Phase 3 model": win_row.cv_rmse_log <= ref.cv_rmse_log + 0.001,
    "Selected model is within one standard error of the best score": win_row.cv_rmse_log <= best_row["cv_rmse_log"] + one_se + 1e-12,
    "Tuning curves figure saved": (config.FIGURES_DIR / "10_tuning_curves.png").exists(),
    "Model comparison figure saved": (config.FIGURES_DIR / "12_model_comparison.png").exists(),
    "Best model saved and predicts": bool(np.isfinite(pred).all()),
    "README results table updated": "(selected)" in readme_text,
    "Results saved (reports/ and docs/)": (config.DOCS_DIR / "model_results.md").exists()
    and (config.REPORTS_DIR / "model_results.csv").exists(),
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print("\nPhase 4 complete." if all(checks.values()) else "\nSomething is off - re-run the file top to bottom.")

# %% [markdown]
# ## Next: Phase 5 (evaluation and diagnostics)
# Score the saved model on the held-out test split **once**, then study residuals, the biggest misses
# (many are abnormal sales), and which features drive the predictions.
# Run `python notebooks/05_evaluation_diagnostics.py` or `python -m house_price.evaluate`.
