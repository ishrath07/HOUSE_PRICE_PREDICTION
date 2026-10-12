"""Shared test fixtures: a small fake Ames-like dataset with the real column names."""

import numpy as np
import pandas as pd
import pytest

from house_price import config
from house_price.features import NA_MEANS_NONE, NOMINAL_COLUMNS, ORDINAL_ORDERS

_NEIGHBORHOODS = ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"]
_NUMERIC_COUNTS = ["BedroomAbvGr", "KitchenAbvGr", "Fireplaces", "GarageCars", "FullBath",
                   "HalfBath", "BsmtFullBath", "BsmtHalfBath", "TotRmsAbvGrd"]  # fmt: skip
_AREAS = [
    "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea", "WoodDeckSF",
    "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal",
]  # fmt: skip


def make_ames_like(n: int = 200, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Random data with every raw Ames column the pipeline expects, including the NA patterns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(index=range(n))

    for col in NOMINAL_COLUMNS:
        X[col] = rng.choice(["a", "b", "c", "d"], n)
    X["Neighborhood"] = rng.choice(_NEIGHBORHOODS, n)
    X["MSSubClass"] = rng.choice([20, 60, 120], n)
    X["MoSold"] = rng.integers(1, 13, n)
    X["Electrical"] = rng.choice(["SBrkr", "FuseA"], n)

    for col, order in ORDINAL_ORDERS.items():
        X[col] = rng.choice([v for v in order if v != "None"], n)
    for col in ["Alley", "PoolQC", "MiscFeature"]:
        X[col] = np.where(rng.random(n) < 0.95, None, "x")

    for col in _AREAS:
        X[col] = np.round(rng.gamma(2.0, 400, n))
    X["PoolArea"] = np.where(rng.random(n) < 0.95, 0, 300)
    X["MiscVal"] = np.where(rng.random(n) < 0.95, 0, 500)
    for col in _NUMERIC_COUNTS:
        X[col] = rng.integers(0, 4, n)
    X["OverallQual"] = rng.integers(1, 11, n)
    X["OverallCond"] = rng.integers(1, 10, n)
    X["YearBuilt"] = rng.integers(1900, 2008, n)
    X["YearRemodAdd"] = np.maximum(X["YearBuilt"], rng.integers(1950, 2010, n))
    X["GarageYrBlt"] = X["YearBuilt"].astype(float)
    X["YrSold"] = rng.integers(2006, 2011, n)
    X["LotFrontage"] = np.round(rng.normal(70, 20, n))

    # Missing values in the same places the real data has them (EDA decision 4 and 5).
    for col in NA_MEANS_NONE:
        if col in X:
            X.loc[rng.random(n) < 0.1, col] = np.nan
    X.loc[rng.random(n) < 0.15, "LotFrontage"] = np.nan
    X.loc[rng.random(n) < 0.05, "MasVnrArea"] = np.nan
    X.loc[rng.random(n) < 0.05, "MasVnrType"] = np.nan
    X.loc[0, "Electrical"] = np.nan
    X.loc[rng.random(n) < 0.1, "GarageYrBlt"] = np.nan

    y = pd.Series(
        np.exp(10.5 + 0.0003 * X["GrLivArea"] + 0.1 * X["OverallQual"] + rng.normal(0, 0.05, n)),
        name=config.TARGET,
    )
    return X, y


@pytest.fixture
def ames():
    return make_ames_like(200, seed=0)


@pytest.fixture
def ames_split():
    """Two non-overlapping chunks: the first plays 'train', the second plays 'test'."""
    X, y = make_ames_like(300, seed=1)
    return X.iloc[:200], y.iloc[:200], X.iloc[200:], y.iloc[200:]
