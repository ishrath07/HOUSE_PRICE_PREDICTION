"""Data loading and the one-time train/test split."""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from house_price import config

logger = logging.getLogger(__name__)


def load_raw(path=config.RAW_TRAIN_FILE) -> pd.DataFrame:
    """Load the raw Ames training CSV. Raw data is never modified on disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download 'train.csv' from Kaggle's "
            "'House Prices: Advanced Regression Techniques' into data/raw/ "
            "(see data/README.md)."
        )
    df = pd.read_csv(path)
    logger.info("Loaded %s rows x %s columns from %s", *df.shape, path.name)
    return df


def split_data(df: pd.DataFrame):
    """Split into train/test ONCE with a fixed seed.

    Returns X_train, X_test, y_train, y_test. The test set must not be used
    for EDA, tuning, or model selection.
    """
    X = df.drop(columns=[config.TARGET, config.ID_COLUMN], errors="ignore")
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )


def drop_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove houses with a huge living area but a low price (EDA decision 12).

    Use on TRAINING data only. The original index is kept so dropped rows can
    still be traced back to reports/outliers.csv.
    """
    is_outlier = (df["GrLivArea"] > config.OUTLIER_MIN_GRLIVAREA) & (
        df[config.TARGET] < config.OUTLIER_MAX_SALEPRICE
    )
    logger.info("Dropping %d outlier row(s) from the training data", int(is_outlier.sum()))
    return df.loc[~is_outlier]


def load_train(path=None, remove_outliers: bool = True):
    """Load the processed TRAINING split and return (X, y).

    The test split is never read here. Outliers are removed by default so every
    model is cross-validated on the same cleaned training rows.
    """
    path = path or config.PROCESSED_DIR / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put Kaggle's train.csv in data/raw/ and run "
            "`python -m house_price.data` to create the train/test split."
        )
    df = pd.read_csv(path).drop(columns=[config.ID_COLUMN], errors="ignore")
    if remove_outliers:
        df = drop_outliers(df)
    return df.drop(columns=[config.TARGET]), df[config.TARGET]


def load_test(path=None):
    """Load the held-out TEST split and return (X, y).

    Outliers are never dropped here (EDA decision 12). Call this only from Phase 5:
    the test split is scored once, after model selection is finished.
    """
    path = path or config.PROCESSED_DIR / "test.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Put Kaggle's train.csv in data/raw/ and run "
            "`python -m house_price.data` to create the train/test split."
        )
    df = pd.read_csv(path).drop(columns=[config.ID_COLUMN], errors="ignore")
    logger.info("Loaded test split: %d rows (outliers kept)", len(df))
    return df.drop(columns=[config.TARGET]), df[config.TARGET]


def make_split_files() -> None:
    """Write the split to data/processed/ so every script uses the same rows."""
    config.ensure_dirs()
    X_train, X_test, y_train, y_test = split_data(load_raw())
    X_train.assign(**{config.TARGET: y_train}).to_csv(
        config.PROCESSED_DIR / "train.csv", index=False
    )
    X_test.assign(**{config.TARGET: y_test}).to_csv(
        config.PROCESSED_DIR / "test.csv", index=False
    )
    logger.info("Saved train (%s) and test (%s) splits", len(X_train), len(X_test))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    make_split_files()