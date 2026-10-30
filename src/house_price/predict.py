"""Score a CSV or one JSON record with the saved house-price pipeline.

Examples
--------
    python -m house_price.predict --input-csv houses.csv --output predictions.csv
    python -m house_price.predict --record-json '{"Neighborhood":"NAmes", ...}'
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from house_price import config

logger = logging.getLogger(__name__)


def load_model(path: Path | None = None) -> Any:
    """Load the fitted pipeline artifact written by ``house_price.train``."""
    model_path = path or config.MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run `python -m house_price.train` first.")
    model = joblib.load(model_path)
    logger.info("Loaded model artifact from %s", model_path)
    return model


def required_columns(model: Any) -> list[str]:
    """Get the raw input schema stored in the fitted preprocessing pipeline."""
    estimator = getattr(model, "regressor_", model)
    members = getattr(estimator, "named_estimators_", None)
    pipelines = members.values() if members else [estimator]
    for pipeline in pipelines:
        steps = getattr(pipeline, "named_steps", {})
        preprocessor = steps.get("preprocess")
        feature_engineer = getattr(preprocessor, "named_steps", {}).get("features")
        columns = getattr(feature_engineer, "feature_names_in_", None)
        if columns is not None:
            return list(columns)
    raise ValueError("The model artifact does not contain a fitted raw-feature schema.")


def _features_for_model(records: pd.DataFrame, model: Any) -> pd.DataFrame:
    if records.empty:
        raise ValueError("Provide at least one house to score.")
    if config.TARGET in records.columns:
        raise ValueError(
            f"Input must not contain {config.TARGET}; remove the target before prediction."
        )
    missing = sorted(set(required_columns(model)) - set(records.columns))
    if missing:
        raise ValueError(f"Input is missing required house features: {missing}")
    return records.drop(columns=[config.ID_COLUMN], errors="ignore")


def predict_frame(records: pd.DataFrame, model: Any) -> pd.DataFrame:
    """Return Kaggle-style ID and SalePrice columns for the supplied house rows."""
    features = _features_for_model(records, model)
    predictions = np.asarray(model.predict(features), dtype=float)
    if predictions.shape != (len(records),):
        raise ValueError(
            f"Expected {len(records)} predictions, received shape {predictions.shape}."
        )
    if not np.isfinite(predictions).all():
        raise ValueError("The model returned a non-finite price prediction.")

    result = pd.DataFrame(index=records.index)
    if config.ID_COLUMN in records.columns:
        result[config.ID_COLUMN] = records[config.ID_COLUMN]
    result[config.TARGET] = predictions
    return result


def predict_record(record: Mapping[str, object], model: Any) -> float:
    """Score one raw house record and return its predicted sale price in dollars."""
    prediction = predict_frame(pd.DataFrame([dict(record)]), model)
    return float(prediction[config.TARGET].iloc[0])


def predict_csv(
    input_path: Path, output_path: Path | None = None, model: Any | None = None
) -> pd.DataFrame:
    """Score a CSV, save a Kaggle-style submission, and return the output frame."""
    input_path = Path(input_path)
    output_path = output_path or input_path.with_name(f"{input_path.stem}_predictions.csv")
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The prediction output path must not overwrite the input CSV.")

    records = pd.read_csv(input_path)
    scored = predict_frame(records, model if model is not None else load_model())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    logger.info("Scored %d houses; wrote predictions to %s", len(scored), output_path)
    return scored


def _read_record(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("A JSON record file must contain one object.")
    return record


def main(argv: list[str] | None = None) -> None:
    """Command-line interface for CSV and single-record prediction."""
    parser = argparse.ArgumentParser(description="Predict house sale prices with the saved model")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-csv", type=Path, help="CSV containing raw house features")
    source.add_argument("--record-json", help="one raw house record as a JSON object")
    source.add_argument("--record-file", type=Path, help="path to a JSON object for one house")
    parser.add_argument("--output", type=Path, help="CSV output path (defaults beside input)")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="model artifact (default: models/best_model.joblib)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    model = load_model(args.model)
    if args.input_csv:
        predict_csv(args.input_csv, args.output, model=model)
        return
    if args.output:
        parser.error("--output is only valid with --input-csv.")

    record = json.loads(args.record_json) if args.record_json else _read_record(args.record_file)
    if not isinstance(record, dict):
        parser.error("The JSON record must be one object with feature names as keys.")
    price = predict_record(record, model)
    logger.info("Predicted SalePrice: $%s", f"{price:,.2f}")


if __name__ == "__main__":
    main()
