import json

import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from house_price import baseline, models


@pytest.mark.parametrize("quick", [False, True])
def test_every_search_parameter_exists_on_its_regressor(quick):
    for spec in models.get_specs(quick).values():
        params = spec.make().get_params()
        assert set(spec.grid) <= set(params), spec.name


def test_candidates_cover_the_phase_4_brief_and_are_ordered_simplest_first():
    specs = models.get_specs()
    assert list(specs) == [
        "LinearRegression", "Ridge", "Lasso", "ElasticNet", "RandomForest", "GradientBoosting",
    ]  # fmt: skip
    complexity = [s.complexity for s in specs.values()]
    assert complexity == sorted(complexity) and len(set(complexity)) == len(complexity)
    assert {s.family for s in specs.values()} == {"linear", "tree"}


def test_number_of_candidates_per_search():
    specs = models.get_specs()
    assert specs["LinearRegression"].n_candidates == 1
    assert specs["Ridge"].n_candidates == 13
    assert specs["ElasticNet"].n_candidates == 7 * 5  # grid: every combination
    assert specs["RandomForest"].n_candidates == 8  # random: only n_iter of them


def test_a_model_without_hyperparameters_is_just_the_pipeline():
    model = models.build_search(models.get_specs(True)["LinearRegression"])
    assert isinstance(model, TransformedTargetRegressor)


def test_models_with_a_grid_get_the_right_kind_of_search():
    specs = models.get_specs(True)
    assert isinstance(models.build_search(specs["Ridge"]), GridSearchCV)
    assert isinstance(models.build_search(specs["GradientBoosting"]), RandomizedSearchCV)


def test_tuning_folds_use_a_different_seed_from_the_scoring_folds():
    spec = models.get_specs()["Ridge"]
    assert models.inner_cv(spec).random_state != baseline.get_cv().random_state
    assert models.inner_cv(spec).get_n_splits() == spec.inner_folds


def test_search_picks_a_value_from_the_grid_and_the_winner_predicts(ames):
    X, y = ames
    spec = models.get_specs(True)["Ridge"]
    search = models.build_search(spec).fit(X, y)
    settings = models.best_settings(search)
    assert set(settings) == {"alpha"}
    assert settings["alpha"] in spec.grid["alpha"]
    pred = search.predict(X)
    assert np.isfinite(pred).all() and (pred > 1000).all()  # dollars


def test_best_settings_are_plain_python_values_that_can_be_saved(ames):
    X, y = ames
    search = models.build_search(models.get_specs(True)["ElasticNet"]).fit(X, y)
    settings = models.best_settings(search)
    assert json.loads(json.dumps(settings)) == settings
    assert (
        models.best_settings(models.build_search(models.get_specs(True)["LinearRegression"])) == {}
    )


def test_build_with_settings_applies_them():
    spec = models.get_specs(True)["Ridge"]
    model = models.build_with_settings(spec, {"alpha": 7.5})
    assert model.regressor.steps[-1][1].alpha == 7.5


def test_blend_model_averages_its_members_on_the_log_scale(ames):
    X, y = ames
    specs = models.get_specs(True)
    settings = {
        "Ridge": {"alpha": 10.0},
        "GradientBoosting": {"max_depth": 2, "learning_rate": 0.1},
    }
    blend = models.build_blend(specs, settings).fit(X, y)
    ridge = models.build_with_settings(specs["Ridge"], settings["Ridge"]).fit(X, y)
    gbr = models.build_with_settings(specs["GradientBoosting"], settings["GradientBoosting"]).fit(
        X, y
    )

    expected = np.expm1((np.log1p(ridge.predict(X)) + np.log1p(gbr.predict(X))) / 2)
    np.testing.assert_allclose(blend.predict(X), expected, rtol=1e-6)
    assert models.BLEND_NAME == "Blend: Ridge + GradientBoosting"
