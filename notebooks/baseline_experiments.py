# %% [markdown]
# # 02 · Baseline models (Phase 2)
#
# **Goal:** set the floor. Two deliberately simple models whose scores every later model has to beat.
#
# 1. **DummyRegressor**: predicts the same average price for every house.
# 2. **LinearRegression**: numeric columns only, missing values filled with the median, nothing else.
#
# **Rules (from `docs/decisions.md` and `docs/eda_decisions.md`)**
# - Only the training split is used. The test split is never opened.
# - The target is modelled as `log1p(SalePrice)` (EDA decision 3).
# - The 2 outlier houses are removed from the training data before cross-validation (EDA decision 12).
# - Both models are scored with the same 5-fold cross-validation.
# - Primary metric: RMSE on the log scale. Secondary: MAE and R-squared on the price scale.
#
# | Step | Question | Output |
# |---|---|---|
# | 1 | What data do the baselines see? | Clean training rows |
# | 2 | How are models scored? | CV protocol |
# | 3 | What does "no skill" look like? | Dummy score |
# | 4 | What does a plain linear model reach? | Linear score |
# | 5 | How do they compare? | Results table |
# | 6 | Where does the linear model fail? | Diagnostics |
# | 7 | What must later models beat? | Saved results |
#
# **How to run this file**
# - Whole file: `python notebooks/02_baseline_experiments.py` (from the project root, venv activated).
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


def find_project_root() -> Path:
    """Walk up from this file (or the working directory) until pyproject.toml is found."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("Run this from inside the project folder (pyproject.toml not found).")


try:
    from house_price import baseline, config, data
except ImportError:  # package not installed: fall back to the src/ layout
    sys.path.insert(0, str(find_project_root() / "src"))
    from house_price import baseline, config, data

try:
    from IPython.display import display
except ImportError:  # plain Python without IPython
    display = print

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 100)
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
# `data.load_train()` reads only `data/processed/train.csv` and drops the outlier rule from EDA decision 12
# (`GrLivArea > 4000` and `SalePrice < 300,000`).
#
# **Output:** the clean training rows, and confirmation that no test data was loaded.

# %%
TRAIN_PATH = config.PROCESSED_DIR / "train.csv"
LOADED_FILES = [TRAIN_PATH.name]  # audit trail: test.csv must never appear here

X_all, _ = data.load_train(TRAIN_PATH, remove_outliers=False)
X, y = data.load_train(TRAIN_PATH)  # outliers removed
n_dropped = len(X_all) - len(X)

print(f"Rows before outlier removal : {len(X_all):,}")
print(f"Outlier rows removed        : {n_dropped}")
print(f"Rows used for the baselines : {len(X):,} x {X.shape[1]} features")
print(f"Files loaded                : {LOADED_FILES}")
print(f"\nColumn types: {X.select_dtypes('number').shape[1]} numeric, "
      f"{X.select_dtypes(exclude='number').shape[1]} text")
print(f"Target {config.TARGET}: min ${y.min():,.0f} | median ${y.median():,.0f} | max ${y.max():,.0f}")

# %% [markdown]
# ## Step 2 · Fix the scoring protocol
# One cross-validation scheme is shared by every model in the project. Using the same folds means differences between models come from the models, not from lucky splits.
#
# | Metric | Scale | Why |
# |---|---|---|
# | **RMSE (log)** | log price | Primary. A 10% miss counts the same on a cheap or an expensive house (and matches Kaggle). |
# | MAE | dollars | Easy to read: average error in dollars. |
# | R-squared | price | Share of price variation explained. 0 = no better than predicting the average. |

# %%
cv = baseline.get_cv()
print(f"Cross-validation : {cv.n_splits}-fold, shuffle={cv.shuffle}, random_state={cv.random_state}")
print(f"Seed (config)    : {config.RANDOM_STATE}")
print("Scorers          :", baseline.SCORING)

# %% [markdown]
# ## Step 3 · Baseline 1: DummyRegressor
# Predicts the mean of the training target for every house, ignoring all features. Because the target is `log1p(SalePrice)`, the mean is taken in log space, then converted back to dollars.
# Any model that cannot beat this has learned nothing.

# %%
dummy = baseline.build_dummy().fit(X, y)
constant = dummy.predict(X.head(1))[0]
print(f"The dummy predicts ${constant:,.0f} for every single house.")

dummy_row = baseline.evaluate("DummyRegressor (mean)", baseline.build_dummy(), X, y)
display(pd.DataFrame([dummy_row]).round(4))

# %% [markdown]
# ## Step 4 · Baseline 2: LinearRegression with minimal preprocessing
# - Keeps **numeric columns only** (chosen by dtype).
# - Fills missing values with the **median** (learned on the training folds only).
# - No scaling, no encoding of text columns, no engineered features, no regularization.
#
# The text columns (neighborhood, quality ratings, ...) are ignored on purpose. Using them is Phase 3's job, so this score shows how far numbers alone go.

# %%
linear = baseline.build_linear().fit(X, y)
used = list(linear.regressor_.named_steps["preprocess"].transformers_[0][2])
ignored = [c for c in X.columns if c not in used]

print(f"Used    : {len(used)} numeric columns")
print(f"Ignored : {len(ignored)} text columns")

gaps = X[used].isna().sum()
gaps = gaps[gaps > 0]
print("\nNumeric columns with missing values (filled with the median):")
print(gaps.to_string())

display(pd.DataFrame({"used (numeric)": pd.Series(used), "ignored (text)": pd.Series(ignored)}))

# %%
linear_row = baseline.evaluate("LinearRegression (numeric only)", baseline.build_linear(), X, y)
display(pd.DataFrame([linear_row]).round(4))

# %% [markdown]
# ## Step 5 · Compare
# Both models on identical folds.

# %%
results = baseline.run_baselines(X, y)
display(results.round(4))

d, l = results.iloc[0], results.iloc[1]
typical_error = (np.exp(l.cv_rmse_log) - 1) * 100
gap = l.cv_rmse_log - l.train_rmse_log

print(f"Dummy RMSE (log)  : {d.cv_rmse_log:.4f}")
print(f"Linear RMSE (log) : {l.cv_rmse_log:.4f}  ({-l.rmse_log_vs_dummy_pct:.1f}% lower than the dummy)")
print(f"Linear typical error: about {typical_error:.0f}% of the price, or ${l.cv_mae:,.0f} on average")
print(f"Linear train vs CV RMSE: {l.train_rmse_log:.4f} vs {l.cv_rmse_log:.4f} (gap {gap:.4f})")
if gap < 0.02:
    print("-> Small gap: the model is limited by what it can see (features), not by overfitting.")
else:
    print("-> Noticeable gap: the model is starting to overfit the training folds.")

# %% [markdown]
# ## Step 6 · Diagnostics: where does the linear model fail?
# Out-of-fold predictions: each house is predicted by a model that never saw it, so these errors are honest.

# %%
oof = baseline.oof_predictions(baseline.build_linear(), X, y)
log_error = np.log1p(y) - np.log1p(oof)  # positive = model predicted too low

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y, oof, s=14, alpha=0.5)
lims = [min(y.min(), oof.min()), max(y.max(), oof.max())]
axes[0].plot(lims, lims, color="crimson", lw=1.5, label="perfect prediction")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("Actual SalePrice ($, log scale)")
axes[0].set_ylabel("Predicted SalePrice ($, log scale)")
axes[0].set_title("Linear baseline: actual vs predicted (out-of-fold)")
axes[0].legend()

axes[1].scatter(np.log1p(oof), log_error, s=14, alpha=0.5)
axes[1].axhline(0, color="crimson", lw=1.5)
axes[1].set_xlabel("Predicted log1p(SalePrice)")
axes[1].set_ylabel("Error on log scale (actual - predicted)")
axes[1].set_title("Residuals: random scatter around 0 is the goal")

plt.tight_layout()
save_fig("08_baseline_diagnostics")
if "get_ipython" in globals():
    plt.show()  # inline in Jupyter / VS Code interactive window
else:
    plt.close(fig)  # script mode: the figure is saved to reports/figures/

# %%
by_band = (
    log_error.abs()
    .groupby(pd.qcut(y, 4, labels=["cheapest 25%", "25-50%", "50-75%", "most expensive 25%"]), observed=True)
    .mean()
    .round(4)
    .rename("mean absolute log error")
)
print("Error by price band (higher = worse):")
display(by_band.to_frame())

cols = [c for c in ["Neighborhood", "OverallQual", "GrLivArea", "YearBuilt", "SaleCondition"] if c in X.columns]
worst_idx = log_error.abs().sort_values(ascending=False).index[:8]
worst = X.loc[worst_idx, cols].assign(
    actual=y.loc[worst_idx], predicted=oof.loc[worst_idx].round(0), log_error=log_error.loc[worst_idx].round(3)
)
print("\nThe 8 houses the linear baseline gets most wrong:")
display(worst)

# %% [markdown]
# ## Step 7 · Save the numbers later models must beat
# Writes `reports/baseline_results.csv` and `docs/baseline_results.md` (a table you can paste into the README).

# %%
baseline.save_results(results)

print("Targets for Phase 3 and Phase 4 (5-fold CV on the training split):")
print(f"  - Must beat the dummy   : RMSE(log) < {d.cv_rmse_log:.4f}")
print(f"  - Must beat the baseline: RMSE(log) < {l.cv_rmse_log:.4f}  |  MAE < ${l.cv_mae:,.0f}  |  R2 > {l.cv_r2:.3f}")

# %% [markdown]
# ## Definition of done
# Run this last cell. All lines should say PASS.

# %%
checks = {
    "Only the training split was loaded (test split never opened)": LOADED_FILES == ["train.csv"],
    "2 outlier rows removed, matching EDA decision 12": n_dropped == 2,
    "Linear baseline beats the dummy": l.cv_rmse_log < d.cv_rmse_log,
    "Results CSV saved (reports/baseline_results.csv)": (config.REPORTS_DIR / "baseline_results.csv").exists(),
    "Results table saved (docs/baseline_results.md)": (config.DOCS_DIR / "baseline_results.md").exists(),
    "Diagnostics figure saved": (config.FIGURES_DIR / "08_baseline_diagnostics.png").exists(),
}
for name, ok in checks.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print("\nPhase 2 complete." if all(checks.values()) else "\nSomething is off - re-run the notebook top to bottom.")

# %% [markdown]
# ## Next: Phase 3 (features and preprocessing pipeline)
# Build the `ColumnTransformer` from `docs/eda_decisions.md`: imputers, ordinal and one-hot encoders, log transforms for skewed columns, engineered features.
# The linear model's job in Phase 3 is to beat the numbers saved above.
