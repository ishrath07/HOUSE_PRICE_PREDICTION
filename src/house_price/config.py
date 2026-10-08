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
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_TRAIN_FILE = RAW_DIR / "train.csv"  # Kaggle Ames file with SalePrice

# --- Reproducibility -------------------------------------------------------
RANDOM_STATE = 42

# --- Problem definition ----------------------------------------------------
TARGET = "SalePrice"
ID_COLUMN = "Id"
TEST_SIZE = 0.2  # held-out test set, touched ONCE at the very end
CV_FOLDS = 5

# --- Metrics ---------------------------------------------------------------
# Primary: RMSE on log(SalePrice) (Kaggle's metric). Also report MAE and R^2.
PRIMARY_METRIC = "rmse_log"


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for path in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)