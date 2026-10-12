"""Phase 3: custom feature-engineering transformers and the column schema.

Everything here follows the scikit-learn transformer API (fit / transform), so it can sit
inside a Pipeline and be refitted on the training folds only. That is what prevents leakage
(EDA decision 15).

Two kinds of transformer:
- Stateless (FeatureEngineer): fixed rules that need nothing learned from data.
- Stateful (GroupMedianImputer, SkewedLog1p): learn a statistic in fit() from the
  training rows only, then reuse it unchanged in transform().
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# --------------------------------------------------------------------------
# Column schema (from docs/eda_decisions.md and data_description.txt)
# --------------------------------------------------------------------------

# EDA decision 4: NA means "this feature is absent", so the category is "None".
NA_MEANS_NONE = [
    "Alley", "Fence", "MiscFeature", "PoolQC", "FireplaceQu",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
]  # fmt: skip

# EDA decision 6: columns that are missing in more than 90% of rows.
SPARSE_COLUMNS = ["PoolQC", "MiscFeature", "Alley"]

# EDA decision 9: weaker member of each highly correlated pair (checked on training data).
COLLINEAR_DROP = ["GarageArea", "1stFlrSF", "TotRmsAbvGrd", "GarageYrBlt"]

# EDA decision 14: raw columns replaced by an engineered one that carries the same information.
#   Full/half baths (above and below ground) -> TotalBathrooms
#   YearBuilt, YearRemodAdd                  -> HouseAge, YearsSinceRemodel
REDUNDANT_DROP = [
    "FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath", "YearBuilt", "YearRemodAdd",
]  # fmt: skip

_QUALITY = ["None", "Po", "Fa", "TA", "Gd", "Ex"]

# EDA decision 2: ordered columns, each listed from worst to best.
ORDINAL_ORDERS = {
    "ExterQual": _QUALITY,
    "ExterCond": _QUALITY,
    "BsmtQual": _QUALITY,
    "BsmtCond": _QUALITY,
    "HeatingQC": _QUALITY,
    "KitchenQual": _QUALITY,
    "FireplaceQu": _QUALITY,
    "GarageQual": _QUALITY,
    "GarageCond": _QUALITY,
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "LandSlope": ["Gtl", "Mod", "Sev"],
    "LotShape": ["IR3", "IR2", "IR1", "Reg"],
    "PavedDrive": ["N", "P", "Y"],
    "CentralAir": ["N", "Y"],
    "Street": ["Grvl", "Pave"],
}
_POOL_ORDER = _QUALITY  # used only when sparse_strategy="keep"

# Nominal (unordered) text columns. MSSubClass and MoSold are numbers that are really
# categories (EDA decision 2), so FeatureEngineer casts them to text.
NOMINAL_COLUMNS = [
    "MSSubClass", "MSZoning", "LandContour", "Utilities", "LotConfig", "Neighborhood",
    "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
    "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "Electrical",
    "GarageType", "Fence", "SaleType", "SaleCondition", "MoSold",
]  # fmt: skip

# Continuous measurements (areas): skewed ones get log1p inside SkewedLog1p.
CONTINUOUS_COLUMNS = [
    "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "TotalSF",
]  # fmt: skip

# Counts, ratings and years: scaled but not log-transformed.
DISCRETE_COLUMNS = [
    "OverallQual", "OverallCond", "BedroomAbvGr", "KitchenAbvGr", "Fireplaces",
    "GarageCars", "YrSold", "TotalBathrooms", "HouseAge", "YearsSinceRemodel",
]  # fmt: skip

# 0/1 columns created by FeatureEngineer: passed through untouched.
FLAG_COLUMNS = ["HasPool", "HasGarage", "HasFireplace", "HasBasement"]
SPARSE_FLAG_COLUMNS = ["HasAlley", "HasMiscFeature"]  # only with sparse_strategy="flag"

SPARSE_STRATEGIES = ("flag", "drop", "keep")


def column_groups(
    sparse_strategy: str = "flag", drop_collinear: bool = True, drop_redundant: bool = True
) -> dict[str, list[str]]:
    """Which engineered column belongs to which preprocessing branch.

    The lists are fixed by the schema, not discovered from the data, so the pipeline
    behaves identically on every fold, at prediction time, and in tests. The arguments
    mirror FeatureEngineer: a column that is not dropped is routed to a branch here.
    """
    _check_strategy(sparse_strategy)
    ordinal = list(ORDINAL_ORDERS)
    nominal = list(NOMINAL_COLUMNS)
    continuous = list(CONTINUOUS_COLUMNS)
    discrete = list(DISCRETE_COLUMNS)
    flags = list(FLAG_COLUMNS)
    if sparse_strategy == "flag":
        flags += SPARSE_FLAG_COLUMNS
    elif sparse_strategy == "keep":
        ordinal.append("PoolQC")
        nominal += ["MiscFeature", "Alley"]
    if not drop_collinear:
        continuous += ["GarageArea", "1stFlrSF"]
        discrete += ["TotRmsAbvGrd", "GarageYrBlt"]
    if not drop_redundant:
        discrete += REDUNDANT_DROP
    return {
        "ordinal": ordinal,
        "nominal": nominal,
        "continuous": continuous,
        "discrete": discrete,
        "flags": flags,
    }


def ordinal_categories(columns: list[str]) -> list[list[str]]:
    """Category order for each ordinal column, in the order of `columns`."""
    orders = {**ORDINAL_ORDERS, "PoolQC": _POOL_ORDER}
    return [orders[c] for c in columns]


def _check_strategy(strategy: str) -> None:
    if strategy not in SPARSE_STRATEGIES:
        raise ValueError(f"sparse_strategy must be one of {SPARSE_STRATEGIES}, got {strategy!r}")


# --------------------------------------------------------------------------
# Stateless transformer: cleaning rules and engineered features
# --------------------------------------------------------------------------


class FeatureEngineer(TransformerMixin, BaseEstimator):
    """Apply the cleaning rules and create the new features from the EDA decisions.

    Input and output are DataFrames. Nothing is learned from the data, so there is
    nothing to leak: fit() only records the column names it saw.

    Parameters
    ----------
    sparse_strategy : {"flag", "drop", "keep"}
        What to do with PoolQC, MiscFeature and Alley (over 90% missing, EDA decision 6).
        "flag" replaces them with HasAlley / HasMiscFeature (HasPool is always created),
        "drop" removes them, "keep" leaves them for the encoders.
    drop_collinear : bool
        Drop the weaker member of each correlated pair (EDA decision 9).
    drop_redundant : bool
        Drop raw bath and year columns that TotalBathrooms / HouseAge / YearsSinceRemodel
        replace (EDA decision 14).
    """

    def __init__(self, sparse_strategy="flag", drop_collinear=True, drop_redundant=True):
        self.sparse_strategy = sparse_strategy
        self.drop_collinear = drop_collinear
        self.drop_redundant = drop_redundant

    _REQUIRED = [
        "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "FullBath", "HalfBath", "BsmtFullBath",
        "BsmtHalfBath", "YrSold", "YearBuilt", "YearRemodAdd", "PoolArea", "GarageArea",
        "Fireplaces", "MSSubClass", "MoSold", "MasVnrType", "MasVnrArea",
    ]  # fmt: skip

    def fit(self, X: pd.DataFrame, y=None):
        _check_strategy(self.sparse_strategy)
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FeatureEngineer needs a pandas DataFrame (it works by column name).")
        missing = [c for c in self._REQUIRED if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.feature_names_out_ = np.array(self._transform(X.head(2)).columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        return self._transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Flags that rely on NA / zero values must be built BEFORE those values are filled.
        if self.sparse_strategy == "flag":
            df["HasAlley"] = df["Alley"].notna().astype(int)
            df["HasMiscFeature"] = df["MiscFeature"].notna().astype(int)
        df["HasPool"] = (df["PoolArea"] > 0).astype(int)
        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)

        # EDA decision 14: combined size, bathroom count and ages.
        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
        df["TotalBathrooms"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )
        # A house can be sold before it is finished, so clip negative ages to 0.
        df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
        df["YearsSinceRemodel"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)

        # EDA decision 4: NA means "absent" for these columns; veneer area is 0 when absent.
        for col in NA_MEANS_NONE + ["MasVnrType"]:
            if col in df:
                df[col] = df[col].fillna("None")
        df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

        # EDA decision 2: numbers that are really categories become text.
        df["MSSubClass"] = df["MSSubClass"].astype(str)
        df["MoSold"] = df["MoSold"].astype(str)

        drop = []
        if self.sparse_strategy in ("flag", "drop"):
            drop += SPARSE_COLUMNS
        if self.drop_collinear:
            drop += COLLINEAR_DROP
        if self.drop_redundant:
            drop += REDUNDANT_DROP
        return df.drop(columns=[c for c in drop if c in df.columns])


# --------------------------------------------------------------------------
# Stateful transformers (learn from the training rows only)
# --------------------------------------------------------------------------


class GroupMedianImputer(TransformerMixin, BaseEstimator):
    """Fill missing values of `column` with the median of its `by` group (EDA decision 5).

    Example: a missing LotFrontage becomes the median LotFrontage of that Neighborhood.
    fit() learns the medians from the training rows only. A group never seen in fit (or a
    group whose values were all missing) falls back to the overall training median.
    """

    def __init__(self, column="LotFrontage", by="Neighborhood"):
        self.column = column
        self.by = by

    def fit(self, X: pd.DataFrame, y=None):
        self.group_medians_ = X.groupby(self.by)[self.column].median().dropna()
        self.global_median_ = float(X[self.column].median())
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "group_medians_")
        df = X.copy()
        fill = df[self.by].map(self.group_medians_).fillna(self.global_median_)
        df[self.column] = df[self.column].fillna(fill)
        return df

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        return self.feature_names_in_


class SkewedLog1p(TransformerMixin, BaseEstimator):
    """Apply log1p to the columns whose skew is above a threshold (EDA decision 7).

    fit() measures the skew of every column on the training rows and remembers which
    ones to transform; transform() applies exactly that choice to any data. Input must be
    free of missing values (put an imputer first). Negative values are clipped to 0.
    """

    def __init__(self, threshold=0.75):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = self._as_array(X)
        self.n_features_in_ = X.shape[1]
        self.skews_ = np.asarray(skew(X, axis=0, nan_policy="omit"), dtype=float)
        # `threshold=None` switches the log transform off (used for ablation studies).
        if self.threshold is None:
            self.skewed_mask_ = np.zeros(X.shape[1], dtype=bool)
        else:
            self.skewed_mask_ = np.abs(np.nan_to_num(self.skews_)) > self.threshold
        return self

    def transform(self, X):
        check_is_fitted(self, "skewed_mask_")
        X = self._as_array(X).copy()
        mask = self.skewed_mask_
        X[:, mask] = np.log1p(np.clip(X[:, mask], 0, None))
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "skewed_mask_")
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)

    @staticmethod
    def _as_array(X) -> np.ndarray:
        return np.asarray(X, dtype=float)
