import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import Lasso, Ridge

from house_price import config, data, pipeline
from house_price.features import column_groups


def _preprocess(model):
    return model.regressor_.named_steps["preprocess"]


def test_pipeline_predicts_finite_positive_prices_in_dollars(ames):
    X, y = ames
    pred = pipeline.build_model().fit(X, y).predict(X)
    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()
    assert (pred > 1000).all()  # dollars, not log units


def test_preprocessed_matrix_has_no_missing_values(ames):
    X, y = ames
    model = pipeline.build_model().fit(X, y)
    matrix = _preprocess(model).transform(X)
    assert np.isfinite(matrix).all()


@pytest.mark.parametrize("strategy", ["flag", "drop", "keep"])
def test_every_sparse_strategy_fits_and_predicts(ames, strategy):
    X, y = ames
    pred = pipeline.build_model(sparse_strategy=strategy).fit(X, y).predict(X)
    assert np.isfinite(pred).all()


def test_unseen_categories_and_odd_values_do_not_crash(ames_split):
    X_train, y_train, X_test, _ = ames_split
    model = pipeline.build_model().fit(X_train, y_train)
    odd = X_test.head(3).copy()
    odd["Neighborhood"] = "BrandNewTown"  # never seen in training
    odd["Exterior1st"] = "Unobtainium"
    odd["MSSubClass"] = 999
    odd["KitchenQual"] = "Zz"  # unknown rating
    odd["LotFrontage"] = np.nan
    assert np.isfinite(model.predict(odd)).all()


def test_single_row_prediction_works(ames):
    X, y = ames
    model = pipeline.build_model().fit(X, y)
    assert model.predict(X.iloc[[5]]).shape == (1,)


def test_no_leakage_transforming_new_data_changes_nothing_that_was_learned(ames_split):
    """The pipeline may only learn from the rows it was fitted on."""
    X_train, y_train, X_test, _ = ames_split
    model = pipeline.build_model().fit(X_train, y_train)
    pre = _preprocess(model)
    before = pre.transform(X_train)

    shifted = X_test.copy()
    shifted["GrLivArea"] = shifted["GrLivArea"] * 50  # wildly different new data
    pre.transform(shifted)

    np.testing.assert_array_equal(before, pre.transform(X_train))


def test_scaler_statistics_come_from_the_training_rows(ames_split):
    X_train, y_train, X_test, _ = ames_split
    pre = _preprocess(pipeline.build_model().fit(X_train, y_train))
    scaler = pre.named_steps["columns"].named_transformers_["discrete"].named_steps["scale"]
    train_engineered = pre.named_steps["features"].transform(X_train)
    expected = train_engineered[column_groups()["discrete"]].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected)


def test_fitting_on_different_data_gives_different_statistics(ames_split):
    X_train, y_train, X_test, y_test = ames_split
    a = _preprocess(pipeline.build_model().fit(X_train, y_train))
    b = _preprocess(pipeline.build_model().fit(X_test, y_test))
    mean_a = a.named_steps["columns"].named_transformers_["discrete"].named_steps["scale"].mean_
    mean_b = b.named_steps["columns"].named_transformers_["discrete"].named_steps["scale"].mean_
    assert not np.allclose(mean_a, mean_b)  # each pipeline learned from its own rows only


def test_feature_names_match_the_matrix_and_are_unique(ames):
    X, y = ames
    model = pipeline.build_model().fit(X, y)
    names = pipeline.feature_names(model)
    assert len(names) == _preprocess(model).transform(X).shape[1]
    assert len(names) == len(set(names))
    assert "ordinal__KitchenQual" in names
    assert "continuous__TotalSF" in names
    assert "flags__HasPool" in names
    assert any(n.startswith("nominal__Neighborhood_") for n in names)


def test_ordinal_encoding_keeps_the_quality_order(ames):
    X, y = ames
    pre = _preprocess(pipeline.build_model().fit(X, y))
    enc = pre.named_steps["columns"].named_transformers_["ordinal"].named_steps["encode"]
    names = column_groups()["ordinal"]  # the encoder sees columns in this order
    cats = enc.categories_[names.index("KitchenQual")].tolist()
    assert cats == ["None", "Po", "Fa", "TA", "Gd", "Ex"]


def test_rare_levels_are_grouped_instead_of_getting_their_own_column(ames):
    X, y = ames
    X = X.copy()
    X.loc[X.index[0], "Neighborhood"] = "OneOffHood"  # a single house: far under 1%
    names = pipeline.feature_names(pipeline.build_model().fit(X, y))
    assert "nominal__Neighborhood_OneOffHood" not in names
    assert "nominal__Neighborhood_infrequent_sklearn" in names


def test_model_survives_clone_and_save_roundtrip(ames, tmp_path):
    X, y = ames
    model = pipeline.build_model().fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict(X), model.predict(X))
    assert np.isfinite(clone(model).fit(X, y).predict(X)).all()  # unfitted copy refits cleanly


def test_default_regressor_is_ridge_and_another_can_be_plugged_in(ames):
    X, y = ames
    default = pipeline.build_model().fit(X, y)
    assert isinstance(default.regressor_.named_steps["model"], Ridge)
    assert default.regressor_.named_steps["model"].alpha == config.RIDGE_ALPHA
    other = pipeline.build_model(Lasso(alpha=0.001)).fit(X, y)
    assert isinstance(other.regressor_.named_steps["model"], Lasso)


def test_run_experiments_returns_one_row_per_experiment(ames):
    X, y = ames
    out = pipeline.run_experiments(
        X, y, {"a": {}, "b": {"regressor": Ridge(alpha=10), "sparse_strategy": "drop"}}
    )
    assert list(out["model"]) == ["a", "b"]
    assert {"cv_rmse_log", "cv_mae", "cv_r2"} <= set(out.columns)
    assert np.isfinite(out["cv_rmse_log"]).all()


def test_save_results_writes_csv_and_markdown(ames, tmp_path, monkeypatch):
    X, y = ames
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path / "docs")
    res = pipeline.run_experiments(X, y, {"alpha": {}})
    res["rmse_log_vs_baseline_pct"] = 0.0
    abl = pipeline.run_experiments(X, y, {"beta": {}})
    abl["vs_default_pct"] = 0.0
    pipeline.save_results(res, abl)
    assert (tmp_path / "reports" / "preprocessing_results.csv").exists()
    assert (tmp_path / "reports" / "preprocessing_ablation.csv").exists()
    md = (tmp_path / "docs" / "preprocessing_results.md").read_text(encoding="utf-8")
    assert "alpha" in md and "beta" in md and "ablation" in md


# --- Integration: only runs when the real Ames split exists locally ----------------------

_TRAIN = config.PROCESSED_DIR / "train.csv"


@pytest.mark.skipif(not _TRAIN.exists(), reason="data/processed/train.csv not created yet")
def test_pipeline_runs_on_the_real_training_data():
    X, y = data.load_train()
    model = pipeline.build_model(Ridge(alpha=10)).fit(X, y)
    matrix = _preprocess(model).transform(X)
    assert matrix.shape[0] == len(X)
    assert np.isfinite(matrix).all()
    assert pd.Series(model.predict(X)).between(10_000, 1_000_000).all()
