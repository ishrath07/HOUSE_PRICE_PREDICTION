import pandas as pd
import pytest

from house_price import config, data


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            config.ID_COLUMN: [1, 2, 3, 4],
            "GrLivArea": [1500, 4676, 5642, 4500],
            config.TARGET: [200_000, 184_750, 160_000, 450_000],
        }
    )


def test_drop_outliers_removes_only_big_and_cheap_houses():
    out = data.drop_outliers(_frame())
    # rows 1 and 2 are huge but cheap; row 3 is huge but expensive, so it stays
    assert list(out.index) == [0, 3]


def test_load_train_returns_features_and_target_without_outliers(tmp_path):
    path = tmp_path / "train.csv"
    _frame().to_csv(path, index=False)
    X, y = data.load_train(path)
    assert len(X) == len(y) == 2
    assert config.TARGET not in X.columns
    assert config.ID_COLUMN not in X.columns


def test_load_train_can_keep_outliers(tmp_path):
    path = tmp_path / "train.csv"
    _frame().to_csv(path, index=False)
    X, _ = data.load_train(path, remove_outliers=False)
    assert len(X) == 4


def test_load_train_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="house_price.data"):
        data.load_train(tmp_path / "nope.csv")