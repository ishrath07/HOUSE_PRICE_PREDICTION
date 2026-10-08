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