import numpy as np
import pandas as pd
import pytest

from house_price import config, pipeline, predict


@pytest.fixture
def fitted_model(ames):
    X, y = ames
    return pipeline.build_model().fit(X, y)


def test_predict_frame_returns_one_finite_price_per_record_and_preserves_ids(ames, fitted_model):
    X, _ = ames
    records = X.iloc[:3].copy()
    records.insert(0, config.ID_COLUMN, [101, 102, 103])
    records.loc[records.index[0], "Neighborhood"] = "UnseenNeighborhood"

    scored = predict.predict_frame(records, fitted_model)

    assert scored.shape == (3, 2)
    assert scored[config.ID_COLUMN].tolist() == [101, 102, 103]
    assert np.isfinite(scored[config.TARGET]).all()
    assert (scored[config.TARGET] > 0).all()


def test_predict_record_returns_a_dollar_price(ames, fitted_model):
    X, _ = ames
    price = predict.predict_record(X.iloc[0].to_dict(), fitted_model)

    assert np.isfinite(price) and price > 0


def test_predict_frame_rejects_missing_required_features(ames, fitted_model):
    X, _ = ames

    with pytest.raises(ValueError, match="missing required house features"):
        predict.predict_frame(X.iloc[:1].drop(columns=["GrLivArea"]), fitted_model)


def test_predict_frame_rejects_target_column_to_prevent_leakage(ames, fitted_model):
    X, y = ames
    records = X.iloc[:1].copy()
    records[config.TARGET] = y.iloc[:1]

    with pytest.raises(ValueError, match="must not contain SalePrice"):
        predict.predict_frame(records, fitted_model)


def test_predict_csv_writes_reloadable_submission(tmp_path, ames, fitted_model):
    X, _ = ames
    records = X.iloc[:4].copy()
    records.insert(0, config.ID_COLUMN, [11, 12, 13, 14])
    input_path = tmp_path / "houses.csv"
    output_path = tmp_path / "out" / "predictions.csv"
    records.to_csv(input_path, index=False)

    scored = predict.predict_csv(input_path, output_path, model=fitted_model)
    saved = pd.read_csv(output_path)

    assert scored.shape == (4, 2)
    pd.testing.assert_frame_equal(saved, scored, check_dtype=False)


def test_predict_csv_does_not_overwrite_the_input(tmp_path):
    path = tmp_path / "houses.csv"
    path.write_text("Id\n1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite"):
        predict.predict_csv(path, path, model=object())


def test_preprocessed_output_has_no_missing_or_infinite_values(ames, fitted_model):
    X, _ = ames
    preprocessor = fitted_model.regressor_.named_steps["preprocess"]
    matrix = preprocessor.transform(X)

    assert np.isfinite(matrix).all()
