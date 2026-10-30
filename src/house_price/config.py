"""Central configuration: paths, seeds, and project-wide constants.

Every other module imports from here, so nothing is hard-coded elsewhere.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

RAW_TRAIN_FILE = RAW_DIR / "train.csv"  # Kaggle Ames file with SalePrice

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42

# --- Problem definition ----------------------------------------------------
TARGET = "SalePrice"
ID_COLUMN = "Id"
TEST_SIZE = 0.2  # held-out test set, touched ONCE at the very end
CV_FOLDS = 5

# --- Outlier rule (EDA decision 12) ----------------------------------------
# Very large houses that sold cheaply break the size-price relationship.
# Applied to TRAINING data only, never to the test set.
OUTLIER_MIN_GRLIVAREA = 4000
OUTLIER_MAX_SALEPRICE = 300_000

# --- Preprocessing (Phase 3) -----------------------------------------------
SKEW_THRESHOLD = 0.75  # EDA decision 7: log1p continuous columns with abs(skew) above this
RARE_LEVEL_MIN_FREQUENCY = 0.01  # EDA decision 10: levels under 1% of rows become 'infrequent'
SPARSE_STRATEGY = "flag"  # EDA decision 6: "flag" | "drop" | "keep" for PoolQC/MiscFeature/Alley

RIDGE_ALPHA = 10.0  # placeholder regularization strength used in Phase 3; Phase 4 tunes it

# --- Modelling (Phase 4) ----------------------------------------------------
INNER_CV_SEED = RANDOM_STATE + 1  # inner (tuning) folds differ from the outer (scoring) folds
N_JOBS = 1  # parallel workers for searches; raise it with `--n-jobs` on the command line

# --- Evaluation (Phase 5) ---------------------------------------------------
PERMUTATION_REPEATS = 8  # repeats for permutation importance on the frozen model

# --- Metrics ---------------------------------------------------------------
# Primary: RMSE on log(SalePrice) (Kaggle's metric). Also report MAE and R^2.
PRIMARY_METRIC = "rmse_log"


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for path in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)
