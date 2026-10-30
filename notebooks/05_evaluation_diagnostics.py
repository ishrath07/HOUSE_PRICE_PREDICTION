# %% [markdown]
# # 05 · Evaluation and diagnostics (Phase 5)
#
# **Goal:** score the Phase 4 winner on the held-out test split **once**, then explain the
# model: residual plots, linear-assumption checks, the features that drive price, and the
# houses it gets most wrong.
#
# **Rules**
# - The model in `models/best_model.joblib` is frozen. Nothing is retuned here.
# - The test split is opened for the first time. Outliers are **not** dropped from it
#   (EDA decision 12 applied to training rows only).
# - Diagnostics that describe the *fit* (VIF, Cook's distance, coefficients) use the
#   training rows. Diagnostics that describe *generalization* (residuals, permutation
#   importance, worst misses) use the test rows.
#
# | Step | Question | Output |
# |---|---|---|
# | 1 | What is being scored? | Frozen blend + test rows |
# | 2 | How good is it on unseen houses? | Test RMSE / MAE / R2, once |
# | 3 | Do predictions track actual prices? | Predicted vs actual |
# | 4 | What do the errors look like? | Residuals and error distribution |
# | 5 | Do linear assumptions hold for Ridge? | Residual pattern, VIF, Cook's |
# | 6 | What drives price? | Permutation + Ridge coefficients |
# | 7 | Which houses are most wrong, and why? | Error analysis |
# | 8 | Save | reports/, docs/, README |
#
# **How to run this file**
# - Whole file: `python notebooks/05_evaluation_diagnostics.py` (project root, venv activated).
# - First run scores the test set and computes permutation importance (about a minute).
#   Later runs load the saved tables. Set `FORCE_RERUN = True` below to recompute.
# - Cell by cell: open it in VS Code and click "Run Cell" above any `# %%` block.
# - Equivalent CLI: `python -m house_price.evaluate`

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
from scipy import stats


def find_project_root() -> Path:
    """Walk up from this file (or the working directory) until pyproject.toml is found."""
    start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").exists():
            return folder
    raise RuntimeError("Run this from inside the project folder (pyproject.toml not found).")


try:
    from house_price import config, evaluate, train
except ImportError:
    sys.path.insert(0, str(find_project_root() / "src"))
    from house_price import config, evaluate, train

try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 140)

config.ensure_dirs()
FORCE_RERUN = False  # True = score the test split again even if saved results exist


def save_fig(name: str) -> None:
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
# ## Step 1 · Load the frozen model and open the test split
# Phase 4 selected **Blend: Ridge + GradientBoosting** by the one-standard-error rule
# (nested CV RMSE(log) 0.1067). The blend is a frozen `VotingRegressor` behind
# `TransformedTargetRegressor`: each member has its own copy of the Phase 3 pipeline,
# predictions are averaged on the log scale, then inverted to dollars.
#
# This is the first file in the project that reads `data/processed/test.csv`.

# %%
LOADED_FILES = []
saved = None if FORCE_RERUN else evaluate.load_saved()
if saved is None:
    print("Scoring the held-out test split (and computing diagnostics) ...")
    out = evaluate.run(save=True)
    saved = evaluate.load_saved()
    LOADED_FILES = ["test.csv", "train.csv"]  # train is loaded only for VIF / Cook / coefficients
else:
    print("Loaded saved Phase 5 tables from reports/ (set FORCE_RERUN = True to recompute).")
    out = saved
    LOADED_FILES = ["reports/"]  # test.csv is not re-opened on a cached run

phase4 = train.load_saved()
winner = out["winner"] if out.get("winner") not in (None, "unknown") else (
    phase4["winner"] if phase4 else "unknown"
)
print(f"Selected model: {winner}")
print(f"Test houses: {int(out['test']['n'])}")
print("Files involved:", LOADED_FILES)

# Cached runs do not keep the fitted model. Load it when we need member details.
model = evaluate.load_model()
print("Estimator:", type(model).__name__, "->", type(evaluate.inner_estimator(model)).__name__)
print("Members:", list(evaluate.member_pipelines(model)))

# %% [markdown]
# ## Step 2 · Score once
# Primary metric is still RMSE on log price (same as every earlier phase). MAE and R2
# are on the dollar scale. The CV number is *not* a competitor: it was computed on
# training folds. The test number is the honest generalization estimate.

# %%
metrics = out["metrics"].copy()
display(metrics.round(4))

test_rmse = out["test"]["rmse_log"]
cv_rmse = out.get("cv_rmse_log", np.nan)
print(f"Held-out test RMSE(log): {test_rmse:.4f}   MAE: ${out['test']['mae']:,.0f}   R2: {out['test']['r2']:.3f}")
if np.isfinite(cv_rmse):
    print(f"Nested CV RMSE(log):     {cv_rmse:.4f}   change vs CV: {(test_rmse / cv_rmse - 1) * 100:+.1f}%")
    print("A small gap is normal (different houses). A large jump would mean the CV estimate was optimistic.")

print("\nFor context: dummy 0.3906, Phase 2 linear 0.1258, Phase 3 Ridge 0.1100, Phase 4 blend CV 0.1067.")

# %% [markdown]
# ## Step 3 · Predicted vs actual
# Points on the diagonal are perfect. Systematic bowing (cheap houses over-predicted,
# expensive ones under-predicted) would mean the log-linear story is breaking down.

# %%
frame = out["test_frame"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

lo = min(frame["actual"].min(), frame["predicted"].min())
hi = max(frame["actual"].max(), frame["predicted"].max())
axes[0].scatter(frame["actual"], frame["predicted"], s=18, alpha=0.55, c="#2a7de1", edgecolors="none")
axes[0].plot([lo, hi], [lo, hi], color="crimson", lw=1.4, label="perfect")
axes[0].set_xlabel("Actual sale price ($)")
axes[0].set_ylabel("Predicted sale price ($)")
axes[0].set_title("Test: predicted vs actual (dollars)")
axes[0].legend(loc="upper left")

axes[1].scatter(np.log1p(frame["actual"]), np.log1p(frame["predicted"]),
                s=18, alpha=0.55, c="#2a7de1", edgecolors="none")
lims = [np.log1p(lo), np.log1p(hi)]
axes[1].plot(lims, lims, color="crimson", lw=1.4)
axes[1].set_xlabel("Actual log1p(SalePrice)")
axes[1].set_ylabel("Predicted log1p(SalePrice)")
axes[1].set_title("Test: predicted vs actual (log scale)")

plt.tight_layout()
save_fig("13_predicted_vs_actual")
show_or_close(fig)

# %% [markdown]
# ## Step 4 · Residual plot and error distribution
# Residuals are `log1p(actual) - log1p(predicted)`: a value of 0.10 is roughly a 10.5%
# under-prediction. On a well-specified log-price model the cloud is a horizontal band
# around zero, and the histogram is roughly bell-shaped.

# %%
resid = frame["residual_log"]
pred = frame["predicted"]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
axes[0].scatter(np.log1p(pred), resid, s=18, alpha=0.55, c="#2a7de1", edgecolors="none")
axes[0].axhline(0, color="crimson", lw=1.4)
axes[0].set_xlabel("Predicted log1p(SalePrice)")
axes[0].set_ylabel("Residual (log)")
axes[0].set_title("Residuals vs predicted")

sns.histplot(resid, bins=25, kde=True, ax=axes[1], color="#2a7de1")
axes[1].axvline(0, color="crimson", lw=1.4)
axes[1].set_xlabel("Residual (log)")
axes[1].set_title("Error distribution")

stats.probplot(resid, dist="norm", plot=axes[2])
axes[2].set_title("Normal Q-Q of log residuals")
axes[2].get_lines()[0].set_markerfacecolor("#2a7de1")
axes[2].get_lines()[0].set_markeredgecolor("#2a7de1")
axes[2].get_lines()[0].set_markersize(4)

plt.tight_layout()
save_fig("14_residuals")
show_or_close(fig)

checks = out["checks"]
print(
    f"Log-residual mean {checks['residual_mean']:.4f} (should be ~0), "
    f"std {checks['residual_std']:.4f}."
)
print(
    f"Shapiro-Wilk p={checks['shapiro_p']:.3g} on a 500-row subsample "
    f"(small p = not perfectly Gaussian; tails matter more than the p-value)."
)
print(
    f"Spearman |residual| vs predicted: ρ={checks['hetero_spearman']:.3f} "
    f"(p={checks['hetero_p']:.3g}). Near 0 = spread does not grow with price."
)
print(
    f"Median |% error|: {frame['pct_error'].abs().median():.1f}%   "
    f"within 10% of actual: {(frame['pct_error'].abs() <= 10).mean() * 100:.0f}% of test houses."
)

# %% [markdown]
# ## Step 5 · Linear-assumption checks you can actually test
# The *blend* is not a linear model. The checks below apply to its **Ridge member**,
# which is the part that has coefficients, a design matrix, and leverage.
#
# | Assumption | What we look at | What "good" looks like |
# |---|---|---|
# | Linearity / no leftover pattern | Residual vs predicted (Step 4) | Horizontal band, no curve |
# | Homoscedasticity | Spearman of \|residual\| vs predicted | ρ near 0 |
# | Normal-ish errors | Histogram + Q-Q | Bell shape; tails can be a bit heavy |
# | Multicollinearity | VIF on the training design matrix | High VIF is expected; Ridge handles it |
# | Influential points | Cook's distance on training rows | Flag D_i > 4/n, do not auto-delete |
#
# VIF and Cook's distance are computed on **training** rows because they describe the
# fit, not generalization.

# %%
vif = out["vif"]
influence = out["influence"]

if len(vif):
    n_high = int((vif["vif"] > evaluate.VIF_HIGH).sum())
    print(f"Columns with VIF > {evaluate.VIF_HIGH:g}: {n_high} of {len(vif)}")
    print("Highest VIF (one-hot neighborhoods and overlapping quality ratings typically dominate):")
    top_vif = vif.head(12).copy()
    top_vif["short"] = top_vif["feature"].map(evaluate.short_name)
    display(top_vif.round(1))
else:
    print("No linear member: VIF skipped.")

if len(influence):
    n_flag = int(influence["flag"].sum())
    thresh = 4.0 / len(influence)
    print(f"\nCook's distance > 4/n ({thresh:.4f}): {n_flag} of {len(influence)} training houses.")
    flagged = influence.loc[influence["flag"]].nlargest(8, "cooks_distance")
    cols = [c for c in ["cooks_distance", "leverage", "actual", "Neighborhood",
                        "OverallQual", "GrLivArea", "SaleCondition"] if c in flagged.columns]
    display(flagged[cols].round(3))
    print("These points pull the Ridge fit. They were not dropped: EDA only removed the two")
    print("huge-and-cheap houses, and Phase 4's regularization already shrinks their effect.")
else:
    print("No linear member: Cook's distance skipped.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

if len(vif):
    shown = vif.head(20)
    labels = [evaluate.short_name(f) for f in shown["feature"]]
    axes[0].barh(labels[::-1], shown["vif"].iloc[::-1], color="#9aa5b1")
    axes[0].axvline(evaluate.VIF_HIGH, color="crimson", ls="--", label=f"VIF = {evaluate.VIF_HIGH:g}")
    axes[0].set_xlabel("VIF")
    axes[0].set_title("Highest VIFs (training design matrix)")
    axes[0].legend()
else:
    axes[0].text(0.5, 0.5, "VIF not available", ha="center")
    axes[0].set_axis_off()

if len(influence):
    axes[1].scatter(influence["leverage"], influence["cooks_distance"],
                    s=14, alpha=0.45, c="#e8832a", edgecolors="none")
    axes[1].axhline(4.0 / len(influence), color="crimson", ls="--", label="4/n")
    axes[1].set_xlabel("Leverage (hat diagonal)")
    axes[1].set_ylabel("Cook's distance")
    axes[1].set_title("Influence vs leverage (Ridge, training)")
    axes[1].legend()
else:
    axes[1].text(0.5, 0.5, "Cook's distance not available", ha="center")
    axes[1].set_axis_off()

plt.tight_layout()
save_fig("15_assumptions")
show_or_close(fig)

# %% [markdown]
# ## Step 6 · What drives price?
# Three views, on purpose:
#
# 1. **Permutation importance** on the blend, in the original column space. Shuffle one
#    raw column on the test set and watch RMSE(log) get worse. This is the recruiter-friendly
#    answer: "the model relies on living area, overall quality, neighborhood, ..."
# 2. **Ridge coefficients** on the processed columns. Sign and size, with the caveat that
#    numeric columns are standardized (1 unit = 1 SD) and every dummy level is kept.
# 3. **GradientBoosting importances** on the same processed columns (how often a column
#    is used for a split, not a signed effect).
#
# EDA decision 8 said OverallQual, GrLivArea, GarageCars, TotalBsmtSF should show up near
# the top. If they do, the model agrees with the data; if they do not, something is off.

# %%
drivers = out["drivers"]
print("Permutation importance (drop in test RMSE(log) when the raw column is shuffled):")
display(drivers.head(15).round(4))

coefs = out["coefficients"]
if len(coefs):
    print("\nLargest |Ridge coefficients| (log-price per 1 unit of the processed column):")
    display(coefs.head(12)[["short", "group", "coefficient", "pct_effect"]].round(3))
    print("pct_effect ≈ (exp(coef) - 1) × 100: a multiplicative change in price, holding the rest fixed.")

trees = out["tree_importances"]
if len(trees):
    print("\nGradientBoosting importances (processed columns):")
    display(trees.head(12)[["short", "importance"]].round(4))

eda_expected = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "Neighborhood"]
perm_names = drivers["feature"].head(20).tolist()
print("\nEDA decision 8 sanity check (is the name in the top 20 permutation drivers)?")
for name in eda_expected:
    hit = name in perm_names
    rank = perm_names.index(name) + 1 if hit else None
    print(f"  {name:<16} {'yes, rank ' + str(rank) if hit else 'not in top 20'}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

top_d = drivers.head(12).iloc[::-1]
axes[0].barh(top_d["feature"], top_d["importance"], xerr=top_d["importance_std"],
             color="#3aa76d", capsize=3)
axes[0].set_xlabel("Increase in test RMSE(log) when shuffled")
axes[0].set_title("Blend: permutation importance (raw columns)")

if len(coefs):
    top_c = coefs.head(12).iloc[::-1]
    colors = ["#2a7de1" if v >= 0 else "#d64545" for v in top_c["coefficient"]]
    axes[1].barh(top_c["short"], top_c["coefficient"], color=colors)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Ridge coefficient (log-price)")
    axes[1].set_title("Ridge member: largest |coefficients|")
else:
    axes[1].text(0.5, 0.5, "No linear member", ha="center")
    axes[1].set_axis_off()

plt.tight_layout()
save_fig("16_feature_drivers")
show_or_close(fig)

# %% [markdown]
# Reading the Ridge signs (typical Ames story, confirm against the table above):
# living area and overall quality raise price; house age lowers it; Stone Brook /
# Crawford neighborhoods sell at a premium; abnormal sales sell at a discount.
# Coefficient sizes are only roughly comparable across column types (standardized
# numbers vs 0/1 dummies), which is why permutation importance leads this section.

# %% [markdown]
# ## Step 7 · Which houses does it get most wrong, and why?
# The table is sorted by |log residual|. The `why` tag is a checklist, not a proof:
# abnormal / family / partial sales, new-home contracts, very low or very high quality,
# and unusually large living area are the usual Ames failure modes. A miss with
# **no obvious flag** is an idiosyncratic price the features cannot see (seller
# motivation, unrecorded remodel, buyer taste).

# %%
worst = out["worst"]
show_cols = [c for c in [
    "actual", "predicted", "pct_error", "residual_log", "Neighborhood",
    "OverallQual", "GrLivArea", "SaleCondition", "SaleType", "why",
] if c in worst.columns]
display(worst[show_cols].round(2))

print("\nMean |log error| by SaleCondition (unusual sales should be worse):")
by_sale_path = config.REPORTS_DIR / "error_by_sale_condition.csv"
if by_sale_path.exists():
    display(pd.read_csv(by_sale_path).round(4))
elif "SaleCondition" in frame.columns:
    display(evaluate.error_by_group(frame, "SaleCondition").round(4))

print("\nMean |log error| by OverallQual:")
by_qual_path = config.REPORTS_DIR / "error_by_overall_qual.csv"
if by_qual_path.exists():
    display(pd.read_csv(by_qual_path).round(4))
elif "OverallQual" in frame.columns:
    display(evaluate.error_by_group(frame, "OverallQual").round(4))

member_cols = [c for c in frame.columns if c.startswith("pred_")]
if member_cols and "SaleCondition" in frame.columns:
    print("\nWhen the blend misses, which member was closer?")
    ridge_col = next((c for c in member_cols if "ridge" in c), None)
    tree_col = next((c for c in member_cols if "gradient" in c or "forest" in c), None)
    if ridge_col and tree_col:
        ridge_err = (np.log1p(frame["actual"]) - np.log1p(frame[ridge_col])).abs()
        tree_err = (np.log1p(frame["actual"]) - np.log1p(frame[tree_col])).abs()
        print(f"  Ridge closer on {(ridge_err < tree_err).mean() * 100:.0f}% of test houses")
        print(f"  GradientBoosting closer on {(tree_err < ridge_err).mean() * 100:.0f}% of test houses")
        print("  The blend exists because they are not always wrong on the same house.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

if "SaleCondition" in frame.columns:
    order = frame.groupby("SaleCondition")["abs_log_error"].median().sort_values().index
    sns.boxplot(data=frame, x="SaleCondition", y="abs_log_error", order=order,
                ax=axes[0], color="#2a7de1")
    axes[0].set_ylabel("|log residual|")
    axes[0].set_title("Test error by sale condition")
    axes[0].tick_params(axis="x", rotation=25)
else:
    axes[0].set_axis_off()

axes[1].scatter(frame["GrLivArea"], frame["residual_log"], s=18, alpha=0.55,
                c="#e8832a", edgecolors="none")
axes[1].axhline(0, color="crimson", lw=1.4)
axes[1].set_xlabel("GrLivArea (sqft)")
axes[1].set_ylabel("Residual (log)")
axes[1].set_title("Test residuals vs living area")

plt.tight_layout()
save_fig("17_error_analysis")
show_or_close(fig)

# %% [markdown]
# ## Step 8 · What this means
# The official number is the **test RMSE(log)**. Everything else is explanation:
# the blend is a regularized linear model averaged with a tree model; size, quality
# and neighborhood dominate; the remaining misses cluster on unusual sales rather
# than on a broken size-price relationship. Phase 6 packages this frozen model
# behind a CLI so a new house can be scored without reopening the notebooks.

# %% [markdown]
# ## Definition of done
# Run this last cell. All lines should say PASS.

# %%
readme_text = (config.PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
checks_done = {
    "Held-out test split was scored (n > 0)": int(out["test"]["n"]) > 0,
    "Test RMSE(log) is finite and beats the dummy (0.3906)": bool(
        np.isfinite(test_rmse) and test_rmse < 0.3906
    ),
    "Test RMSE(log) is in the same ballpark as nested CV": bool(
        not np.isfinite(cv_rmse) or test_rmse < cv_rmse + 0.05
    ),
    "Predicted-vs-actual figure saved": (config.FIGURES_DIR / "13_predicted_vs_actual.png").exists(),
    "Residual figure saved": (config.FIGURES_DIR / "14_residuals.png").exists(),
    "Assumption figure saved": (config.FIGURES_DIR / "15_assumptions.png").exists(),
    "Feature-driver figure saved": (config.FIGURES_DIR / "16_feature_drivers.png").exists(),
    "Error-analysis figure saved": (config.FIGURES_DIR / "17_error_analysis.png").exists(),
    "Worst-error table has a why column": "why" in out["worst"].columns and len(out["worst"]) > 0,
    "Permutation drivers were computed": len(out["drivers"]) > 0,
    "README test block updated": evaluate.README_START in readme_text and "RMSE (log)" in readme_text,
    "Results saved (reports/ and docs/)": (config.DOCS_DIR / "evaluation_results.md").exists()
    and (config.REPORTS_DIR / "test_metrics.csv").exists()
    and (config.REPORTS_DIR / "test_score.json").exists(),
}
for name, ok in checks_done.items():
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
print("\nPhase 5 complete." if all(checks_done.values()) else "\nSomething is off - re-run the file top to bottom.")

# %% [markdown]
# ## Next: Phase 6 (packaging)
# Wrap the frozen model in a small CLI (`predict`), add a usage example to the README,
# and make sure `make setup && make data && make train && make evaluate` reproduces
# the numbers from a clean checkout.
