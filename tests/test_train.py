import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from house_price import config, data, models, train

README = f"# Title\n\nintro\n\n{train.README_START}\nold table\n{train.README_END}\n\nfooter\n"


def _ridge_spec(grid=None, folds=3):
    return models.ModelSpec(
        "Ridge", Ridge, grid={"alpha": [1.0, 10.0, 100.0]} if grid is None else grid,
        inner_folds=folds, complexity=1,
    )  # fmt: skip


def _fake_result(name, scores):
    """A NestedResult built from known per-fold scores (no model fitting)."""
    scores = np.asarray(scores, float)
    n = len(scores)
    folds = pd.DataFrame(
        {
            "fold": range(1, n + 1),
            "rmse_log": scores,
            "mae": 1.0,
            "r2": 0.9,
            "train_rmse_log": 0.0,
            "inner_rmse_log": 0.0,
        }  # fmt: skip
    )
    idx = pd.RangeIndex(n)
    return train.NestedResult(
        name, folds, pd.Series(0.0, index=idx), pd.Series(idx + 1, index=idx), []
    )


# --- metrics -------------------------------------------------------------------------------


def test_fold_metrics_are_zero_error_for_perfect_predictions():
    m = train.fold_metrics([100_000, 200_000], [100_000, 200_000])
    assert m["rmse_log"] == pytest.approx(0) and m["r2"] == pytest.approx(1) and m["mae"] == 0


def test_fold_metrics_rmse_is_on_the_log_scale_and_mae_on_the_price_scale():
    m = train.fold_metrics([100.0, 200.0], [110.0, 200.0])
    assert m["rmse_log"] == pytest.approx(np.sqrt((np.log1p(100.0) - np.log1p(110.0)) ** 2 / 2))
    assert m["mae"] == pytest.approx(5.0)


@pytest.mark.parametrize("bad", [np.inf, -5.0, np.nan])
def test_fold_metrics_flag_unstable_predictions_as_nan_instead_of_a_score(bad):
    m = train.fold_metrics([100.0, 200.0], [100.0, bad])
    assert all(np.isnan(v) for v in m.values())


# --- nested cross-validation ---------------------------------------------------------------


def test_nested_cv_scores_every_row_once_and_picks_settings_inside_each_fold(ames):
    X, y = ames
    res = train.evaluate_nested(_ridge_spec(), X, y, KFold(5, shuffle=True, random_state=0))
    assert len(res.folds) == 5 and len(res.settings) == 5
    assert res.oof_log.notna().all()
    assert sorted(res.fold_of_row.unique()) == [1, 2, 3, 4, 5]
    assert all(s["alpha"] in (1.0, 10.0, 100.0) for s in res.settings)
    assert res.folds["rmse_log"].between(0, 1).all()


def test_nested_cv_is_deterministic(ames):
    X, y = ames
    a = train.evaluate_nested(_ridge_spec(), X, y)
    b = train.evaluate_nested(_ridge_spec(), X, y)
    pd.testing.assert_frame_equal(a.folds, b.folds)
    assert a.settings == b.settings


def test_no_fit_ever_sees_the_held_out_fold(ames):
    """The tuning loop must only use the outer training rows (the key nested-CV guarantee)."""

    class SpyRidge(Ridge):
        sizes: list[int] = []

        def fit(self, X, y, sample_weight=None):
            SpyRidge.sizes.append(len(X))
            return super().fit(X, y, sample_weight)

    X, y = ames
    SpyRidge.sizes.clear()
    spec = models.ModelSpec("spy", SpyRidge, grid={"alpha": [1.0, 10.0]}, inner_folds=3)
    train.evaluate_nested(spec, X, y, KFold(5, shuffle=True, random_state=0))
    outer_train_rows = len(X) * 4 // 5
    assert SpyRidge.sizes and max(SpyRidge.sizes) <= outer_train_rows < len(X)


def test_untuned_model_goes_through_the_same_harness(ames):
    X, y = ames
    res = train.evaluate_nested(_ridge_spec(grid={}), X, y)
    assert res.settings == [{}] * 5
    assert res.folds["inner_rmse_log"].isna().all()


# --- blend, summary, paired comparison, selection ---------------------------------------------


def test_blend_averages_out_of_fold_predictions_on_the_log_scale(ames):
    X, y = ames
    a = train.evaluate_nested(_ridge_spec(), X, y)
    b = train.evaluate_nested(_ridge_spec(grid={"alpha": [50.0]}), X, y)
    blend = train.blend_result("blend", [a, b], y)
    np.testing.assert_allclose(blend.oof_log, (a.oof_log + b.oof_log) / 2)
    assert len(blend.folds) == 5 and blend.folds["rmse_log"].notna().all()


def test_summary_reports_mean_and_std_across_folds_and_change_vs_the_phase_3_model():
    ref = _fake_result(train.REFERENCE_NAME, [0.10, 0.12, 0.11, 0.13, 0.09])
    better = _fake_result("Ridge", [0.09, 0.11, 0.10, 0.12, 0.08])
    out = train.summarize([ref, better]).set_index("model")
    assert out.loc["Ridge", "cv_rmse_log"] == pytest.approx(0.10)
    assert out.loc["Ridge", "cv_rmse_log_std"] == pytest.approx(
        np.std([0.09, 0.11, 0.10, 0.12, 0.08])
    )
    assert out.loc["Ridge", "vs_phase3_pct"] == pytest.approx((0.10 / 0.11 - 1) * 100)
    assert out.loc[train.REFERENCE_NAME, "vs_phase3_pct"] == 0


def test_paired_comparison_separates_real_differences_from_noise():
    ref = _fake_result("ref", [0.10, 0.12, 0.11, 0.13, 0.09])
    consistently_worse = _fake_result("worse", [0.12, 0.14, 0.13, 0.15, 0.11])
    noise = _fake_result("noise", [0.101, 0.119, 0.112, 0.128, 0.092])
    broken = _fake_result("broken", [0.1, 0.1, np.nan, 0.1, 0.1])
    table = train.paired_comparison([ref, consistently_worse, noise, broken], "ref").set_index(
        "model"
    )
    assert table.loc["worse", "verdict"] == "worse" and table.loc["worse", "folds_better"] == 0
    assert table.loc["noise", "verdict"] == "tie"
    assert table.loc["broken", "verdict"] == "unstable"
    assert "ref" not in table.index


def test_one_standard_error_rule_prefers_the_simpler_model_when_scores_tie():
    simple = _fake_result("Ridge", [0.101, 0.121, 0.111, 0.131, 0.091])
    complex_ = _fake_result(
        "GradientBoosting", [0.100, 0.120, 0.110, 0.130, 0.090]
    )  # best by a hair
    terrible = _fake_result("LinearRegression", [0.2] * 5)
    specs = {s.name: s for s in [
        models.ModelSpec("LinearRegression", Ridge, complexity=0),
        models.ModelSpec("Ridge", Ridge, complexity=1),
        models.ModelSpec("GradientBoosting", Ridge, complexity=5),
    ]}  # fmt: skip
    results = [simple, complex_, terrible]
    assert train.select_best(train.summarize(results), results, specs) == "Ridge"


def test_selection_keeps_a_clearly_better_complex_model():
    simple = _fake_result("Ridge", [0.15, 0.16, 0.15, 0.16, 0.15])
    complex_ = _fake_result("GradientBoosting", [0.10, 0.11, 0.10, 0.11, 0.10])
    specs = {
        "Ridge": models.ModelSpec("Ridge", Ridge, complexity=1),
        "GradientBoosting": models.ModelSpec("GradientBoosting", Ridge, complexity=5),
    }
    results = [simple, complex_]
    assert train.select_best(train.summarize(results), results, specs) == "GradientBoosting"


# --- tuning table and grid edge warning -----------------------------------------------------------


def test_tuning_table_lists_every_candidate_with_its_rank(ames):
    X, y = ames
    spec = models.get_specs(True)["Ridge"]
    search = models.build_search(spec).fit(X, y)
    table = train.tuning_table("Ridge", search)
    assert len(table) == 3 and set(table["alpha"]) == {1.0, 10.0, 100.0}
    assert table.loc[table["rank"] == 1, "alpha"].iloc[0] == models.best_settings(search)["alpha"]
    assert train.tuning_table(
        "x", models.build_search(models.get_specs(True)["LinearRegression"])
    ).empty


def test_grid_edge_warning_fires_only_at_the_edge_of_a_numeric_grid():
    spec = models.get_specs()["Ridge"]
    lo, hi = min(spec.grid["alpha"]), max(spec.grid["alpha"])
    middle = spec.grid["alpha"][6]
    assert train.grid_edge_warnings(spec, {"alpha": lo}) and train.grid_edge_warnings(
        spec, {"alpha": hi}
    )
    assert train.grid_edge_warnings(spec, {"alpha": middle}) == []
    assert train.grid_edge_warnings(models.get_specs()["RandomForest"], {"max_depth": 8}) == []


# --- README table, saving, loading ---------------------------------------------------------------


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(README, encoding="utf-8")
    assert train.update_readme("| new | table |", path) is True
    text = path.read_text(encoding="utf-8")
    assert "| new | table |" in text and "old table" not in text
    assert text.startswith("# Title\n\nintro") and text.endswith("footer\n")
    assert train.update_readme("| again |", path) is True  # safe to run repeatedly


def test_update_readme_leaves_a_file_without_markers_alone(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("# no markers\n", encoding="utf-8")
    assert train.update_readme("table", path) is False
    assert path.read_text(encoding="utf-8") == "# no markers\n"


@pytest.fixture
def sandbox(ames, tmp_path, monkeypatch):
    """Point every output folder at a temp dir and feed run_all the fake data."""
    X, y = ames
    monkeypatch.setattr(data, "load_train", lambda *a, **k: (X, y))
    for name in [
        "RAW_DIR",
        "PROCESSED_DIR",
        "MODELS_DIR",
        "REPORTS_DIR",
        "FIGURES_DIR",
        "DOCS_DIR",
    ]:
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    return tmp_path


def test_run_all_saves_the_model_tables_and_readme_and_everything_reloads(sandbox, ames):
    out = train.run_all(quick=True, only=["Ridge", "GradientBoosting"], save=True)

    assert out["winner"] in {"Ridge", "GradientBoosting", models.BLEND_NAME}
    assert [r.name for r in out["results"]] == [
        train.REFERENCE_NAME,
        "Ridge",
        "GradientBoosting",
        models.BLEND_NAME,
    ]
    assert set(out["settings"][models.BLEND_NAME]) == set(models.BLEND_MEMBERS)  # members' settings
    for f in [
        "model_results.csv",
        "model_cv_folds.csv",
        "tuning_results.csv",
        "best_params.json",
        "model_paired_comparison.csv",
        "model_oof_predictions.csv",
        "experiment_runs.csv",
    ]:
        assert (sandbox / "reports_dir" / f).exists(), f
    assert (sandbox / "docs_dir" / "model_results.md").exists()

    experiment_log = pd.read_csv(sandbox / "reports_dir" / "experiment_runs.csv")
    assert set(experiment_log["model"]) == set(out["summary"]["model"])
    assert experiment_log["selected"].sum() == 1

    saved = joblib.load(sandbox / "models_dir" / "best_model.joblib")
    X, _ = ames
    assert np.isfinite(saved.predict(X)).all()  # the saved winner works on its own

    readme = (sandbox / "README.md").read_text(encoding="utf-8")
    assert "(selected)" in readme and "old table" not in readme and "footer" in readme

    loaded = train.load_saved()
    assert loaded["winner"] == out["winner"]
    assert (
        json.loads((sandbox / "reports_dir" / "best_params.json").read_text())["selected"]
        == out["winner"]
    )
    assert len(loaded["folds"]) == 5 * 4  # four scored entries, five folds each


def test_load_saved_returns_none_before_phase_4_has_run(sandbox):
    assert train.load_saved() is None


def test_run_all_without_save_writes_nothing(sandbox):
    train.run_all(quick=True, only=["Ridge"], save=False)
    assert not (sandbox / "reports_dir").exists() and not (sandbox / "models_dir").exists()


def test_command_line_quick_run_logs_the_selection(sandbox, caplog):
    caplog.set_level("INFO", logger="house_price.train")
    train.main(["--quick", "--models", "Ridge"])
    assert "Selected (one-standard-error rule): Ridge" in caplog.text
    assert "no final model artifact was saved" in caplog.text


def test_the_blend_can_be_selected_and_the_saved_model_is_the_selected_one(sandbox, ames):
    out = train.run_all(quick=True, only=["Ridge", "GradientBoosting"], save=True)
    saved = joblib.load(sandbox / "models_dir" / "best_model.joblib")
    X, _ = ames
    np.testing.assert_allclose(saved.predict(X), out["finals"][out["winner"]].predict(X))
    assert models.BLEND_NAME in out["specs"]  # the blend is a real candidate in the selection
