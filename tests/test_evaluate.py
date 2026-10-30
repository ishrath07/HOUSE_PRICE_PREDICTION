import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from house_price import config, data, evaluate, models, pipeline, train

README = (
    f"# Title\n\nintro\n\n{train.README_START}\nold table\n{train.README_END}\n\n"
    f"## Test\n{evaluate.README_START}\nold test table\n{evaluate.README_END}\n"
)


def test_load_test_keeps_the_huge_cheap_houses_that_training_drops(tmp_path):
    path = tmp_path / "test.csv"
    pd.DataFrame(
        {
            config.ID_COLUMN: [1, 2, 3, 4],
            "GrLivArea": [1500, 4676, 5642, 4500],
            config.TARGET: [200_000, 184_750, 160_000, 450_000],
        }
    ).to_csv(path, index=False)
    X, y = data.load_test(path)
    assert len(X) == len(y) == 4
    assert config.TARGET not in X.columns
    assert config.ID_COLUMN not in X.columns


def test_load_test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="house_price.data"):
        data.load_test(tmp_path / "nope.csv")


def test_prediction_frame_uses_log_residuals_and_keeps_context(ames):
    X, y = ames
    pred = y.to_numpy() * 1.1
    frame = evaluate.prediction_frame(X, y, pred)
    assert (frame["residual_log"] < 0).all()  # over-prediction
    assert frame["abs_error"].iloc[0] == pytest.approx(abs(y.iloc[0] - pred[0]))
    assert "OverallQual" in frame.columns and "GrLivArea" in frame.columns


def test_vif_is_high_for_duplicated_columns_and_near_one_for_independent_ones():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = rng.normal(size=200)
    x = np.column_stack([a, b, a])  # column 2 copies column 0
    vif = evaluate.variance_inflation_factors(x, ["a", "b", "a_copy"])
    assert vif["a_copy"] > 100
    assert vif["b"] == pytest.approx(1.0, abs=0.15)


def test_cooks_distance_flags_an_obvious_outlier():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(80, 3))
    residual = rng.normal(scale=0.05, size=80)
    residual[0] = 5.0  # one wild residual
    cooks, leverage = evaluate.cooks_distance_ridge(x, residual, alpha=1.0)
    assert cooks.argmax() == 0
    assert cooks[0] > np.median(cooks) * 10
    assert leverage.shape == (80,) and (leverage >= 0).all()


def test_ridge_coefficients_have_a_percent_effect_and_are_sorted(ames):
    X, y = ames
    model = pipeline.build_model(Ridge(alpha=10.0)).fit(X, y)
    coefs = evaluate.ridge_coefficients(evaluate.linear_member(model))
    assert coefs["abs_coefficient"].is_monotonic_decreasing
    assert "GrLivArea" in set(coefs["short"])
    row = coefs.iloc[0]
    assert row["pct_effect"] == pytest.approx(100.0 * np.expm1(row["coefficient"]))


def test_blend_unwraps_into_a_linear_and_a_tree_member(ames):
    X, y = ames
    specs = models.get_specs(True)
    model = models.build_blend(
        specs, {"Ridge": {"alpha": 10.0}, "GradientBoosting": {}}
    ).fit(X, y)
    assert set(evaluate.member_pipelines(model)) == {"ridge", "gradientboosting"}
    assert evaluate.linear_member(model) is not None
    assert evaluate.tree_member(model) is not None
    dollars = evaluate.member_dollar_predictions(model, X.head(5))
    assert set(dollars.columns) == {"ridge", "gradientboosting"}
    assert (dollars > 1000).all().all()


def test_miss_reason_tags_abnormal_sales_and_leaves_ordinary_ones_alone():
    abnormal = pd.Series(
        {
            "SaleCondition": "Abnorml",
            "SaleType": "WD",
            "OverallQual": 5,
            "GrLivArea": 1500,
            "pct_error": -8,
        }
    )
    ordinary = pd.Series(
        {
            "SaleCondition": "Normal",
            "SaleType": "WD",
            "OverallQual": 5,
            "GrLivArea": 1500,
            "pct_error": -3,
        }
    )
    assert "unusual sale" in evaluate.miss_reason(abnormal)
    assert evaluate.miss_reason(ordinary).startswith("no obvious flag")


def test_worst_errors_are_the_largest_log_misses_and_carry_a_why(ames):
    X, y = ames
    pred = y.to_numpy().copy()
    pred[3] = y.iloc[3] * 3  # spectacular miss
    frame = evaluate.prediction_frame(X, y, pred)
    worst = evaluate.worst_errors(frame, n=5)
    assert worst.index[0] == 3
    assert "why" in worst.columns


@pytest.fixture
def sandbox(ames_split, tmp_path, monkeypatch):
    X_train, y_train, X_test, y_test = ames_split
    monkeypatch.setattr(data, "load_train", lambda *a, **k: (X_train, y_train))
    monkeypatch.setattr(data, "load_test", lambda *a, **k: (X_test, y_test))
    monkeypatch.setattr(
        train,
        "load_saved",
        lambda: {
            "winner": "Ridge",
            "summary": pd.DataFrame(
                {"model": ["Ridge"], "cv_rmse_log": [0.11], "cv_mae": [12000.0], "cv_r2": [0.9]}
            ),
        },
    )
    for name in ["RAW_DIR", "PROCESSED_DIR", "MODELS_DIR",
                 "REPORTS_DIR", "FIGURES_DIR", "DOCS_DIR"]:  # fmt: skip
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    config.ensure_dirs()
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    model = pipeline.build_model(Ridge(alpha=10.0)).fit(X_train, y_train)
    joblib.dump(model, tmp_path / "models_dir" / "best_model.joblib")
    return tmp_path, X_test, y_test, model


def test_run_scores_the_test_split_once_and_saves_reloadable_tables(sandbox):
    tmp_path, X_test, y_test, model = sandbox
    out = evaluate.run(n_repeats=1, save=True)

    assert out["winner"] == "Ridge"
    assert out["test"]["n"] == len(X_test)
    assert np.isfinite(out["test"]["rmse_log"])
    expected = train.fold_metrics(y_test, model.predict(X_test))
    assert out["test"]["rmse_log"] == pytest.approx(expected["rmse_log"])
    assert len(out["drivers"]) == X_test.shape[1]
    assert len(out["coefficients"]) > 0 and len(out["vif"]) > 0
    assert "why" in out["worst"].columns

    for name in ["test_metrics.csv", "test_predictions.csv", "test_score.json",
                 "feature_drivers.csv", "ridge_coefficients.csv", "vif.csv"]:  # fmt: skip
        assert (tmp_path / "reports_dir" / name).exists(), name
    assert (tmp_path / "docs_dir" / "evaluation_results.md").exists()

    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "old test table" not in readme and "RMSE (log)" in readme

    loaded = evaluate.load_saved()
    assert loaded["test"]["rmse_log"] == pytest.approx(out["test"]["rmse_log"])
    assert loaded["winner"] == "Ridge"
    score = json.loads((tmp_path / "reports_dir" / "test_score.json").read_text())
    assert score["n_test"] == len(X_test)


def test_load_saved_returns_none_before_phase_5_has_run(sandbox):
    tmp_path, *_ = sandbox
    for leftover in (tmp_path / "reports_dir").glob("test_*"):
        leftover.unlink()
    assert evaluate.load_saved() is None


def test_command_line_prints_the_test_score(sandbox, capsys):
    evaluate.main(["--repeats", "1"])
    printed = capsys.readouterr().out
    assert "Winner: Ridge" in printed
    assert "rmse_log" in printed or "test" in printed
