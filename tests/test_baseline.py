import numpy as np
import pandas as pd
import pytest

from house_price import baseline, config


@pytest.fixture
def toy():
    """Small dataset: price depends on Area and Qual; one text column, one column with gaps."""
    rng = np.random.default_rng(0)
    n = 200
    area = rng.uniform(500, 3000, n)
    qual = rng.integers(1, 11, n)
    X = pd.DataFrame(
        {
            "Area": area,
            "Qual": qual,
            "Gappy": np.where(rng.random(n) < 0.2, np.nan, rng.normal(size=n)),
            "Style": rng.choice(["a", "b"], n),
        }
    )
    y = pd.Series(
        np.exp(10 + 0.0004 * area + 0.08 * qual + rng.normal(0, 0.05, n)), name=config.TARGET
    )
    return X, y


def test_dummy_predicts_the_mean_of_the_log_target(toy):
    X, y = toy
    pred = baseline.build_dummy().fit(X, y).predict(X)
    expected = np.expm1(np.log1p(y).mean())
    np.testing.assert_allclose(pred, expected)


def test_linear_baseline_handles_text_and_missing_values(toy):
    X, y = toy
    pred = baseline.build_linear().fit(X, y).predict(X)
    assert np.isfinite(pred).all()
    assert (pred > 0).all()


def test_linear_baseline_uses_only_numeric_columns(toy):
    X, y = toy
    model = baseline.build_linear().fit(X, y)
    used = model.regressor_.named_steps["preprocess"].transformers_[0][2]
    assert sorted(used) == ["Area", "Gappy", "Qual"]


def test_run_baselines_returns_both_models_and_linear_wins(toy):
    X, y = toy
    results = baseline.run_baselines(X, y)
    assert list(results["model"]) == [
        "DummyRegressor (mean)",
        "LinearRegression (numeric only)",
    ]
    assert results.loc[1, "cv_rmse_log"] < results.loc[0, "cv_rmse_log"]
    assert results.loc[0, "rmse_log_vs_dummy_pct"] == 0


def test_cross_validation_is_deterministic(toy):
    X, y = toy
    a = baseline.run_baselines(X, y)
    b = baseline.run_baselines(X, y)
    pd.testing.assert_frame_equal(a, b)


def test_oof_predictions_cover_every_row_on_the_price_scale(toy):
    X, y = toy
    pred = baseline.oof_predictions(baseline.build_linear(), X, y)
    assert pred.index.equals(X.index)
    assert np.isfinite(pred).all()
    assert (pred > 1000).all()  # price scale (dollars), not log scale


def test_save_results_writes_csv_and_markdown(toy, tmp_path, monkeypatch):
    X, y = toy
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "DOCS_DIR", tmp_path / "docs")
    baseline.save_results(baseline.run_baselines(X, y))
    assert (tmp_path / "reports" / "baseline_results.csv").exists()
    md = (tmp_path / "docs" / "baseline_results.md").read_text(encoding="utf-8")
    assert "DummyRegressor" in md and "LinearRegression" in md