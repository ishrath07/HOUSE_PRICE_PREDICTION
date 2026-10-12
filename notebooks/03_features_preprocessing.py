# %% [markdown]
# # 03 · Features and preprocessing pipeline (Phase 3)
#
# **Goal:** turn the raw Ames table into a clean numeric matrix using every decision from
# `docs/eda_decisions.md`, inside one scikit-learn pipeline, and prove it beats the Phase 2 baseline.
#
# **Rules**
# - Only the training split is used. The test split is never opened.
# - Every statistic (medians, modes, skew, scaling, category lists) is learned from training rows only.
# - Same 5-fold cross-validation as Phase 2, so scores are directly comparable.
#
# | Step | Question | Output |
# |---|---|---|
# | 1 | What goes in? | Raw training rows |
# | 2 | What does feature engineering add and remove? | Before/after columns |
# | 3 | Which columns get a log transform? | Skew table |
# | 4 | What does the model finally see? | Matrix shape, column names |
# | 5 | Is anything leaking? | Leakage checks |
# | 6 | Does it beat the baseline? | Results table |
# | 7 | Which decisions mattered? | Ablation table |
# | 8 | Where did the errors go? | Diagnostics |
# | 9 | Save the numbers | Reports |
#
# **How to run this file**
# - Whole file: `python notebooks/03_features_preprocessing.py` (project root, venv activated).
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from sklearn.linear_model import Ridge


def find_project_root() -> Path:
    """Walk up from this file (or the working directory) until pyproject.toml is found."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("Run this from inside the project folder (pyproject.toml not found).")


try:
    from house_price import baseline, config, data, features, pipeline
except ImportError:  # package not installed: fall back to the src/ layout
    sys.path.insert(0, str(find_project_root() / "src"))
    from house_price import baseline, config, data, features, pipeline

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


def save_fig(name: str) -> None:
    """Save the current figure to reports/figures/<name>.png."""
    path = config.FIGURES_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {path.relative_to(config.PROJECT_ROOT)}")


print("Project root:", config.PROJECT_ROOT)

# %% [markdown]
# ## Step 1 · Load the training data
# Same loader as Phase 2: training split only, with the 2 outlier houses removed (EDA decision 12).

# %%
TRAIN_PATH = config.PROCESSED_DIR / "train.csv"
LOADED_FILES = [TRAIN_PATH.name]  # audit trail: test.csv must never appear here

X, y = data.load_train(TRAIN_PATH)
print(f"Rows: {len(X):,} | raw features: {X.shape[1]} | files loaded: {LOADED_FILES}")
print(f"Columns with missing values: {int((X.isna().sum() > 0).sum())}")
print(f"Total missing cells        : {int(X.isna().sum().sum()):,}")

# %% [markdown]
# ## Step 2 · What feature engineering adds and removes
# `FeatureEngineer` is the first pipeline step. It applies fixed rules (no learning), so it cannot leak.
#
# | Rule | EDA decision |
# |---|---|
# | NA means "absent" -> category `None`; `MasVnrArea` NA -> 0 | 4 |
# | `MSSubClass`, `MoSold` become text categories | 2 |
# | `TotalSF`, `TotalBathrooms`, `HouseAge`, `YearsSinceRemodel`, `Has*` flags are created | 14 |
# | Weaker column of each correlated pair is dropped | 9 |
# | PoolQC / MiscFeature / Alley (over 90% missing) become flags | 6 |

# %%
engineer = features.FeatureEngineer().fit(X)
X_eng = engineer.transform(X)

added = [c for c in X_eng.columns if c not in X.columns]
removed = [c for c in X.columns if c not in X_eng.columns]
print(f"Columns in : {X.shape[1]}  ->  out: {X_eng.shape[1]}")
print(f"\nAdded   ({len(added)}): {added}")
print(f"\nRemoved ({len(removed)}): {removed}")

# %%
show = ["GrLivArea", "TotalBsmtSF", "TotalSF", "TotalBathrooms", "HouseAge", "YearsSinceRemodel"]
flags = ["HasPool", "HasGarage", "HasFireplace", "HasBasement", "HasAlley", "HasMiscFeature"]
print("Engineered numbers for the first 5 houses:")
display(X_eng.loc[X.index[:5], show + flags])
print("\nFlag prevalence (share of houses that have the feature):")
print(X_eng[flags].mean().round(3).to_string())

# %% [markdown]
# ## Step 3 · Which columns get a log transform?
# `SkewedLog1p` measures skew on the **training rows**, remembers which continuous columns have
# `abs(skew) > 0.75` (EDA decision 7), and applies `log1p` to exactly those, everywhere.

# %%
model = pipeline.build_model().fit(X, y)
pre = model.regressor_.named_steps["preprocess"]
log_step = pre.named_steps["columns"].named_transformers_["continuous"].named_steps["log"]
cont_cols = features.column_groups()["continuous"]

skew_table = pd.DataFrame({"column": cont_cols, "skew_before": log_step.skews_.round(2)})
skew_table["log1p_applied"] = log_step.skewed_mask_
after = []
for col, logged in zip(cont_cols, log_step.skewed_mask_):
    values = X_eng[col].dropna().clip(lower=0)
    after.append(skew(np.log1p(values) if logged else values))
skew_table["skew_after"] = np.round(after, 2)
display(skew_table.sort_values("skew_before", ascending=False).reset_index(drop=True))
print(f"\n{int(log_step.skewed_mask_.sum())} of {len(cont_cols)} continuous columns were log-transformed.")

# %% [markdown]
# ## Step 4 · What the model finally sees
# Four branches feed one matrix:
#
# | Branch | Columns | Treatment |
# |---|---|---|
# | ordinal | quality and condition ratings | fill "None", order Po < Fa < TA < Gd < Ex, scale |
# | nominal | Neighborhood, Exterior, ... | mode-fill, one-hot, levels under 1% grouped as "infrequent" |
# | continuous | areas | median-fill, `log1p` if skewed, scale |
# | discrete / flags | counts, years, 0/1 flags | median-fill and scale / pass through |

# %%
matrix = pre.transform(X)
names = pipeline.feature_names(model)
branch_counts = pd.Series([n.split("__")[0] for n in names]).value_counts()

print(f"Matrix shape       : {matrix.shape[0]:,} rows x {matrix.shape[1]} columns")
print(f"Missing values left: {int(np.isnan(matrix).sum())}")
print("\nColumns per branch:")
print(branch_counts.to_string())
print("\nSample of column names:", names[:3], "...", names[30:33], "...", names[-3:])

# %%
# The one-hot step groups rare levels. Show it for Neighborhood.
hood_cols = [n for n in names if n.startswith("nominal__Neighborhood_")]
print(f"Neighborhood: {X['Neighborhood'].nunique()} raw levels -> {len(hood_cols)} one-hot columns")
print("Grouped into 'infrequent':", "nominal__Neighborhood_infrequent_sklearn" in names)

# %% [markdown]
# ## Step 5 · Leakage checks
# The pipeline must only learn from the rows it is fitted on. Two direct checks:
# 1. Transforming very different new data must not change what was learned.
# 2. The scaler's mean must equal the mean of the training rows, nothing else.

# %%
fitted_on = X.iloc[:900]
held_out = X.iloc[900:]
check_model = pipeline.build_model().fit(fitted_on, y.iloc[:900])
check_pre = check_model.regressor_.named_steps["preprocess"]

before = check_pre.transform(fitted_on)
distorted = held_out.copy()
distorted["GrLivArea"] = distorted["GrLivArea"] * 50  # absurd new data
check_pre.transform(distorted)
after_new_data = check_pre.transform(fitted_on)
no_drift = np.array_equal(before, after_new_data)

scaler = check_pre.named_steps["columns"].named_transformers_["discrete"].named_steps["scale"]
train_means = check_pre.named_steps["features"].transform(fitted_on)[
    features.column_groups()["discrete"]
].mean()
scaler_matches_train = bool(np.allclose(scaler.mean_, train_means.to_numpy()))

print(f"Transforming new data changed nothing that was learned : {no_drift}")
print(f"Scaler means equal the fitting rows' means             : {scaler_matches_train}")
print("Cross-validation refits the whole pipeline inside every fold, so validation rows never")
print("influence an imputer, scaler, encoder or skew decision.")

# %% [markdown]
# ## Step 6 · Does it beat the baseline?
# Three rows on identical folds:
# 1. The Phase 2 baseline (plain LinearRegression, numeric columns only).
# 2. The same numeric-only input with **Ridge**. This isolates the effect of regularization.
# 3. The Phase 3 pipeline with Ridge. The gap from row 2 to row 3 is the value of the new features.
#
# Why Ridge and not plain LinearRegression for the Phase 3 pipeline? After encoding there are about
# 200 overlapping columns (for example `HasGarage` is exactly "GarageType is None"). Plain least squares
# is ill-conditioned there and its answer changes with the numerical library version. Ridge's penalty
# makes the problem well-posed. Alpha is a placeholder here; Phase 4 tunes it.

# %%
cv = baseline.get_cv()
ridge = Ridge(alpha=config.RIDGE_ALPHA)

base_row = baseline.evaluate("Phase 2 baseline: LinearRegression (numeric only)", baseline.build_linear(), X, y, cv)
num_ridge_row = baseline.evaluate(
    f"Numeric only + Ridge(alpha={config.RIDGE_ALPHA:g})", baseline.build_linear(ridge), X, y, cv
)
phase3_rows = pipeline.run_experiments(X, y, pipeline.phase3_experiments()).to_dict("records")

results = pipeline.compare_to_baseline(pd.DataFrame([base_row, num_ridge_row, *phase3_rows]), pd.Series(base_row))
display(results.round(4))

b, n, p3 = results.iloc[0], results.iloc[1], results.iloc[2]
print(f"Baseline RMSE (log)          : {b.cv_rmse_log:.4f}")
print(f"Numeric only + Ridge         : {n.cv_rmse_log:.4f}   (regularization alone: {n.rmse_log_vs_baseline_pct:+.1f}%)")
print(f"Phase 3 pipeline + Ridge     : {p3.cv_rmse_log:.4f}   (features + encoding: {p3.rmse_log_vs_baseline_pct:+.1f}%)")
print(f"Typical error now about {(np.exp(p3.cv_rmse_log) - 1) * 100:.0f}% of the price, or ${p3.cv_mae:,.0f} on average")
print(f"Train vs CV RMSE: {p3.train_rmse_log:.4f} vs {p3.cv_rmse_log:.4f} (gap {p3.cv_rmse_log - p3.train_rmse_log:.4f})")

# %% [markdown]
# ## Step 7 · Which decisions mattered? (ablation)
# Change one thing at a time against the default pipeline. The fold-to-fold standard deviation is
# about 0.006, so differences of a few tenths of a percent are noise, not evidence.

# %%
ablation = pipeline.run_experiments(X, y, pipeline.ablation_experiments())
default_rmse = ablation.loc[0, "cv_rmse_log"]
ablation["vs_default_pct"] = (ablation["cv_rmse_log"] / default_rmse - 1) * 100
display(ablation.round(4))

noise = ablation.loc[0, "cv_rmse_log_std"] / default_rmse * 100
mattered = ablation[ablation["vs_default_pct"].abs() > noise].iloc[1:]
print(f"Fold noise is about {noise:.1f}% of the score.")
print("Changes larger than the noise:", mattered["model"].tolist() or "none")

# %% [markdown]
# ## Step 8 · Where did the errors go?
# Out-of-fold predictions for the baseline and the Phase 3 model, split by price band.

# %%
oof_base = baseline.oof_predictions(baseline.build_linear(), X, y, cv)
oof_p3 = baseline.oof_predictions(pipeline.build_model(), X, y, cv)
err_base = (np.log1p(y) - np.log1p(oof_base)).abs()
err_p3 = (np.log1p(y) - np.log1p(oof_p3)).abs()

bands = pd.qcut(y, 4, labels=["cheapest 25%", "25-50%", "50-75%", "most expensive 25%"])
by_band = pd.DataFrame(
    {
        "baseline": err_base.groupby(bands, observed=True).mean(),
        "phase 3": err_p3.groupby(bands, observed=True).mean(),
    }
).round(4)
by_band["change"] = ((by_band["phase 3"] / by_band["baseline"] - 1) * 100).round(1).astype(str) + "%"
print("Mean absolute log error by price band (lower is better):")
display(by_band)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

by_band[["baseline", "phase 3"]].plot.bar(ax=axes[0], rot=15, color=["#9aa5b1", "#2a7de1"])
axes[0].set_ylabel("Mean absolute log error")
axes[0].set_xlabel("")
axes[0].set_title("Error by price band: baseline vs Phase 3")

resid = np.log1p(y) - np.log1p(oof_p3)
axes[1].scatter(np.log1p(oof_p3), resid, s=14, alpha=0.5)
axes[1].axhline(0, color="crimson", lw=1.5)
axes[1].set_xlabel("Predicted log1p(SalePrice)")
axes[1].set_ylabel("Error on log scale (actual - predicted)")
axes[1].set_title("Phase 3 residuals (out-of-fold)")

plt.tight_layout()
save_fig("09_phase3_diagnostics")
if "get_ipython" in globals():
    plt.show()
else:
    plt.close(fig)

# %%
cols = [c for c in ["Neighborhood", "OverallQual", "GrLivArea", "SaleCondition"] if c in X.columns]
worst_idx = resid.abs().sort_values(ascending=False).index[:8]
worst = X.loc[worst_idx, cols].assign(
    actual=y.loc[worst_idx], predicted=oof_p3.loc[worst_idx].round(0), log_error=resid.loc[worst_idx].round(3)
)
print("The 8 houses the Phase 3 model gets most wrong:")
display(worst)

# %% [markdown]
# ## Step 9 · Save the numbers
# Writes `reports/preprocessing_results.csv`, `reports/preprocessing_ablation.csv` and
# `docs/preprocessing_results.md`. Phase 4 starts from these numbers.

# %%
pipeline.save_results(results, ablation)
print("Targets for Phase 4 (5-fold CV on the training split):")
print(f"  - Must beat the Phase 3 pipeline: RMSE(log) < {p3.cv_rmse_log:.4f}  |  MAE < ${p3.cv_mae:,.0f}  |  R2 > {p3.cv_r2:.3f}")

# %% [markdown]
# ## Definition of done
# Run this last cell. All lines should say PASS.

# %%
checks = {
    "Only the training split was loaded (test split never opened)": LOADED_FILES == ["train.csv"],
    "Feature engineering added the 10 engineered columns": {"TotalSF", "TotalBathrooms", "HouseAge", "YearsSinceRemodel", "HasPool", "HasGarage", "HasFireplace", "HasBasement", "HasAlley", "HasMiscFeature"} <= set(added),
    "Preprocessed matrix has no missing values": int(np.isnan(matrix).sum()) == 0,
    "Leakage check 1: new data changes nothing that was learned": no_drift,
    "Leakage check 2: scaler statistics come from the fitting rows": scaler_matches_train,
    "Phase 3 pipeline beats the Phase 2 baseline": p3.cv_rmse_log < b.cv_rmse_log,
    "Gain comes from features, not just regularization": p3.cv_rmse_log < n.cv_rmse_log - 0.005,
    "Results saved (reports/ and docs/)": all(
        path.exists()
        for path in [
            config.REPORTS_DIR / "preprocessing_results.csv",
            config.REPORTS_DIR / "preprocessing_ablation.csv",
            config.DOCS_DIR / "preprocessing_results.md",
        ]
    ),
    "Diagnostics figure saved": (config.FIGURES_DIR / "09_phase3_diagnostics.png").exists(),
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print("\nPhase 3 complete." if all(checks.values()) else "\nSomething is off - re-run the file top to bottom.")

# %% [markdown]
# ## Next: Phase 4 (modelling and validation)
# The pipeline is fixed. Phase 4 swaps in better regressors (Ridge, Lasso, ElasticNet, then tree ensembles),
# tunes them with cross-validated search, and compares everything on the same folds.
