import itertools

import numpy as np
import pandas as pd
import pytest

from house_price import features
from house_price.features import (
    FeatureEngineer,
    GroupMedianImputer,
    SkewedLog1p,
    column_groups,
)


def _row(**overrides) -> pd.DataFrame:
    """One hand-made house with values chosen so the engineered numbers are easy to check."""
    row = {
        "TotalBsmtSF": 800, "1stFlrSF": 900, "2ndFlrSF": 500,
        "FullBath": 2, "HalfBath": 1, "BsmtFullBath": 1, "BsmtHalfBath": 0,
        "YrSold": 2010, "YearBuilt": 1990, "YearRemodAdd": 2000,
        "PoolArea": 0, "GarageArea": 400, "Fireplaces": 1,
        "MSSubClass": 60, "MoSold": 7, "MasVnrType": np.nan, "MasVnrArea": np.nan,
        "Alley": np.nan, "MiscFeature": "Shed", "PoolQC": np.nan, "BsmtQual": np.nan,
        "TotRmsAbvGrd": 6, "GarageYrBlt": 1990.0,
    }  # fmt: skip
    row.update(overrides)
    return pd.DataFrame([row])


def test_engineered_numbers_are_correct():
    out = FeatureEngineer().fit_transform(_row())
    assert out.loc[0, "TotalSF"] == 800 + 900 + 500
    assert out.loc[0, "TotalBathrooms"] == 2 + 0.5 * 1 + 1 + 0.5 * 0
    assert out.loc[0, "HouseAge"] == 20
    assert out.loc[0, "YearsSinceRemodel"] == 10
    flags = ["HasPool", "HasGarage", "HasFireplace", "HasBasement"]
    assert out.loc[0, flags].tolist() == [0, 1, 1, 1]
    assert out.loc[0, ["HasAlley", "HasMiscFeature"]].tolist() == [0, 1]


def test_na_means_none_and_veneer_area_is_zero():
    out = FeatureEngineer().fit_transform(_row())
    assert out.loc[0, "BsmtQual"] == "None"
    assert out.loc[0, "MasVnrType"] == "None"
    assert out.loc[0, "MasVnrArea"] == 0


def test_numeric_categories_become_text():
    out = FeatureEngineer().fit_transform(_row())
    assert out.loc[0, "MSSubClass"] == "60"
    assert out.loc[0, "MoSold"] == "7"


def test_houses_sold_before_completion_get_age_zero():
    out = FeatureEngineer().fit_transform(_row(YrSold=2006, YearBuilt=2007, YearRemodAdd=2007))
    assert out.loc[0, "HouseAge"] == 0
    assert out.loc[0, "YearsSinceRemodel"] == 0


def test_feature_engineer_does_not_modify_its_input():
    raw = _row()
    before = raw.copy()
    FeatureEngineer().fit_transform(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_drop_switches_remove_the_documented_columns():
    cols = set(FeatureEngineer().fit_transform(_row()).columns)
    assert not cols & set(features.COLLINEAR_DROP + features.REDUNDANT_DROP)
    kept = set(FeatureEngineer(drop_collinear=False, drop_redundant=False).fit_transform(_row()))
    assert set(features.COLLINEAR_DROP + features.REDUNDANT_DROP) <= kept


@pytest.mark.parametrize(
    "strategy, sparse_left, flags",
    [("flag", set(), {"HasAlley", "HasMiscFeature"}), ("drop", set(), set()),
     ("keep", set(features.SPARSE_COLUMNS), set())],
)  # fmt: skip
def test_sparse_strategies(strategy, sparse_left, flags):
    cols = set(FeatureEngineer(sparse_strategy=strategy).fit_transform(_row()).columns)
    assert cols & set(features.SPARSE_COLUMNS) == sparse_left
    assert cols & {"HasAlley", "HasMiscFeature"} == flags


def test_unknown_sparse_strategy_is_rejected():
    with pytest.raises(ValueError, match="sparse_strategy"):
        FeatureEngineer(sparse_strategy="nope").fit(_row())
    with pytest.raises(ValueError, match="sparse_strategy"):
        column_groups("nope")


def test_missing_required_columns_give_a_clear_error():
    with pytest.raises(ValueError, match="TotalBsmtSF"):
        FeatureEngineer().fit(_row().drop(columns=["TotalBsmtSF"]))


@pytest.mark.parametrize(
    "strategy, collinear, redundant",
    list(itertools.product(features.SPARSE_STRATEGIES, [True, False], [True, False])),
)
def test_every_engineered_column_is_routed_to_exactly_one_branch(
    ames, strategy, collinear, redundant
):
    """Guards against a column being silently ignored (or counted twice) by the pipeline."""
    X, _ = ames
    out = FeatureEngineer(strategy, collinear, redundant).fit_transform(X)
    groups = column_groups(strategy, collinear, redundant)
    routed = [c for cols in groups.values() for c in cols]
    assert len(routed) == len(set(routed)), "a column appears in two branches"
    assert set(routed) == set(out.columns)


def test_group_median_imputer_learns_from_training_rows_only():
    train = pd.DataFrame({"Neighborhood": ["A", "A", "B"], "LotFrontage": [10.0, 30.0, 50.0]})
    other = pd.DataFrame({"Neighborhood": ["A", "B"], "LotFrontage": [1000.0, 1000.0]})
    imp = GroupMedianImputer().fit(train)
    test = pd.DataFrame({"Neighborhood": ["A", "B"], "LotFrontage": [np.nan, np.nan]})
    assert imp.transform(test)["LotFrontage"].tolist() == [20.0, 50.0]
    imp.transform(other)  # transforming other data must not change what was learned
    assert imp.transform(test)["LotFrontage"].tolist() == [20.0, 50.0]


def test_group_median_imputer_falls_back_to_the_overall_median():
    train = pd.DataFrame({"Neighborhood": ["A", "A", "B"], "LotFrontage": [10.0, 30.0, 50.0]})
    test = pd.DataFrame({"Neighborhood": ["Z", None], "LotFrontage": [np.nan, np.nan]})
    out = GroupMedianImputer().fit(train).transform(test)
    assert out["LotFrontage"].tolist() == [30.0, 30.0]


def test_skewed_log1p_transforms_only_skewed_columns():
    rng = np.random.default_rng(0)
    skewed = rng.exponential(100, 500)
    symmetric = rng.normal(50, 1, 500)
    X = np.column_stack([skewed, symmetric])
    t = SkewedLog1p(threshold=0.75).fit(X)
    assert t.skewed_mask_.tolist() == [True, False]
    out = t.transform(X)
    np.testing.assert_allclose(out[:, 0], np.log1p(skewed))
    np.testing.assert_allclose(out[:, 1], symmetric)


def test_skewed_log1p_reuses_the_choice_made_in_fit():
    rng = np.random.default_rng(0)
    t = SkewedLog1p().fit(np.column_stack([rng.exponential(100, 300)]))
    new = np.column_stack([rng.normal(50, 1, 20)])  # looks symmetric, but fit said "log it"
    np.testing.assert_allclose(t.transform(new), np.log1p(new))


def test_skewed_log1p_can_be_switched_off_and_never_returns_nan():
    X = np.array([[0.0], [0.0], [0.0], [500.0]])
    np.testing.assert_allclose(SkewedLog1p(threshold=None).fit_transform(X), X)
    assert np.isfinite(SkewedLog1p().fit_transform(np.array([[-5.0], [0.0], [9.0], [400.0]]))).all()
