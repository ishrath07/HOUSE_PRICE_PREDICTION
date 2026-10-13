"""Phase 4: the model candidates and their hyperparameter search spaces.

Every candidate sits behind the same Phase 3 preprocessing pipeline, so a score difference
comes from the regressor and its settings, not from the features.

    LinearRegression   no hyperparameters: the unregularized reference
    Ridge              L2 penalty, shrinks correlated coefficients together
    Lasso              L1 penalty, sets some coefficients exactly to zero
    ElasticNet         mix of both (alpha and l1_ratio)
    RandomForest       }  optional stretch: tree models as a comparison
    GradientBoosting   }
    Blend              plain average of Ridge and GradientBoosting predictions (log scale)

A "spec" says how to build a candidate and which settings to search. `build_search` turns a
spec into a scikit-learn estimator: a GridSearchCV / RandomizedSearchCV wrapped around the
full pipeline (or the plain pipeline when there is nothing to tune).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

from house_price import baseline, config, pipeline

# Parameter path from the search object down to the regressor:
# TransformedTargetRegressor -> regressor (Pipeline) -> "model" step.
PARAM_PREFIX = "regressor__model__"

# The blend averages these two models. Chosen in advance (a linear and a tree model make
# different kinds of mistakes), not picked by looking at the results.
BLEND_MEMBERS = ("Ridge", "GradientBoosting")
BLEND_NAME = "Blend: " + " + ".join(BLEND_MEMBERS)

# Same metric as every earlier phase: RMSE of log1p(price), computed on dollar predictions.
SEARCH_SCORING = "neg_root_mean_squared_log_error"


@dataclass(frozen=True)
class ModelSpec:
    """How to build one candidate and what to tune."""

    name: str
    make: Callable[[], object]  # returns a fresh, unfitted regressor
    grid: dict[str, list] = field(default_factory=dict)  # empty = nothing to tune
    search: str = "grid"  # "grid" = try every combination, "random" = sample n_iter of them
    n_iter: int | None = None
    inner_folds: int = 5
    family: str = "linear"  # "linear" or "tree"
    complexity: int = 0  # lower = simpler; breaks ties in the one-standard-error rule

    @property
    def n_candidates(self) -> int:
        """How many settings the search tries."""
        total = int(np.prod([len(v) for v in self.grid.values()])) if self.grid else 1
        return min(total, self.n_iter) if self.search == "random" and self.n_iter else total


def get_specs(quick: bool = False) -> dict[str, ModelSpec]:
    """All candidates, simplest first. `quick=True` shrinks every search for fast smoke runs."""
    if quick:
        return _quick_specs()
    return {
        "LinearRegression": ModelSpec(
            "LinearRegression", LinearRegression, complexity=0
        ),
        "Ridge": ModelSpec(
            "Ridge", Ridge,
            grid={"alpha": list(np.logspace(-0.5, 2.5, 13))},
            complexity=1,
        ),
        "Lasso": ModelSpec(
            "Lasso", lambda: Lasso(max_iter=50_000),
            grid={"alpha": list(np.logspace(-4, -2, 13))},
            complexity=2,
        ),
        "ElasticNet": ModelSpec(
            "ElasticNet", lambda: ElasticNet(max_iter=50_000),
            grid={
                "alpha": list(np.logspace(-3.5, -1.5, 7)),
                "l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.9],
            },
            complexity=3,
        ),
        "RandomForest": ModelSpec(
            "RandomForest",
            lambda: RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_STATE),
            grid={
                "max_depth": [8, 12, None],
                "min_samples_leaf": [1, 2, 4],
                "max_features": [0.2, 0.33, 0.5],
            },
            search="random", n_iter=8, inner_folds=3, family="tree", complexity=4,
        ),
        "GradientBoosting": ModelSpec(
            "GradientBoosting",
            lambda: GradientBoostingRegressor(random_state=config.RANDOM_STATE),
            grid={
                "n_estimators": [300, 500, 800],
                "learning_rate": [0.02, 0.05, 0.1],
                "max_depth": [2, 3, 4],
                "subsample": [0.6, 0.8],
                "min_samples_leaf": [1, 3, 10],
            },
            search="random", n_iter=8, inner_folds=3, family="tree", complexity=5,
        ),
    }  # fmt: skip


def _quick_specs() -> dict[str, ModelSpec]:
    """Tiny searches: same code path, seconds instead of minutes (used by tests and --quick)."""
    return {
        "LinearRegression": ModelSpec("LinearRegression", LinearRegression, complexity=0),
        "Ridge": ModelSpec(
            "Ridge", Ridge, grid={"alpha": [1.0, 10.0, 100.0]}, inner_folds=3, complexity=1
        ),
        "Lasso": ModelSpec(
            "Lasso", lambda: Lasso(max_iter=10_000),
            grid={"alpha": [1e-3, 1e-2]}, inner_folds=3, complexity=2,
        ),
        "ElasticNet": ModelSpec(
            "ElasticNet", lambda: ElasticNet(max_iter=10_000),
            grid={"alpha": [1e-3, 1e-2], "l1_ratio": [0.2, 0.8]}, inner_folds=3, complexity=3,
        ),
        "RandomForest": ModelSpec(
            "RandomForest",
            lambda: RandomForestRegressor(n_estimators=20, random_state=config.RANDOM_STATE),
            grid={"max_depth": [4, 8], "max_features": [0.3, 0.5]},
            search="random", n_iter=2, inner_folds=3, family="tree", complexity=4,
        ),
        "GradientBoosting": ModelSpec(
            "GradientBoosting",
            lambda: GradientBoostingRegressor(n_estimators=20, random_state=config.RANDOM_STATE),
            grid={"max_depth": [2, 3], "learning_rate": [0.1, 0.2]},
            search="random", n_iter=2, inner_folds=3, family="tree", complexity=5,
        ),
    }  # fmt: skip


def inner_cv(spec: ModelSpec) -> KFold:
    """The folds used for tuning. Their seed differs from the outer scoring folds."""
    return KFold(spec.inner_folds, shuffle=True, random_state=config.INNER_CV_SEED)


def build_search(spec: ModelSpec, n_jobs: int = config.N_JOBS):
    """An unfitted estimator for this candidate.

    With a grid: a search object whose `fit` tries every setting with inner cross-validation
    (re-running the whole preprocessing inside each inner fold, so nothing leaks), then refits
    the winner on all the rows it was given. Without a grid: just the pipeline.
    """
    model = pipeline.build_model(spec.make())
    if not spec.grid:
        return model
    param_space = {PARAM_PREFIX + name: values for name, values in spec.grid.items()}
    common = {"scoring": SEARCH_SCORING, "cv": inner_cv(spec), "n_jobs": n_jobs, "refit": True}
    if spec.search == "random":
        return RandomizedSearchCV(
            model, param_space, n_iter=spec.n_iter, random_state=config.RANDOM_STATE, **common
        )
    return GridSearchCV(model, param_space, **common)


def best_settings(fitted_search) -> dict:
    """The winning settings of a fitted search, without the long parameter prefix."""
    params = getattr(fitted_search, "best_params_", {})
    return {name.removeprefix(PARAM_PREFIX): _plain(value) for name, value in params.items()}


def _plain(value):
    """Turn numpy scalars into plain Python numbers so settings print and save cleanly."""
    return value.item() if isinstance(value, np.generic) else value


def build_with_settings(spec: ModelSpec, settings: dict):
    """An unfitted pipeline built with the given settings (no search)."""
    regressor = spec.make().set_params(**settings)
    return pipeline.build_model(regressor)


def build_blend(specs: dict[str, ModelSpec], settings: dict[str, dict]):
    """The blend as one estimator: both members, each with its own preprocessing, averaged.

    The average is taken on the log1p scale (the scale the members are trained on), which is
    exactly how the nested cross-validation scored the blend.
    """
    members = []
    for name in BLEND_MEMBERS:
        regressor = specs[name].make().set_params(**settings[name])
        steps = [("preprocess", pipeline.build_preprocessor()), ("model", regressor)]
        members.append((name.lower(), Pipeline(steps)))
    return baseline.with_log_target(VotingRegressor(members))
