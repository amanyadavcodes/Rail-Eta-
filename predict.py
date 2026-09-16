"""Run batch predictions with a model created by train.py."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV containing live/model features.")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model-dir", default="models/basic_xgboost")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    with open(model_dir / "feature_schema.json", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema["features_in_order"]

    rows = pd.read_csv(args.input)
    missing_columns = sorted(set(features) - set(rows.columns))
    if missing_columns:
        raise ValueError("Input is missing required columns: " + ", ".join(missing_columns))
    matrix = rows[features].apply(pd.to_numeric, errors="coerce")

    model = xgb.XGBRegressor()
    model.load_model(model_dir / "eta_model.json")
    predictions = np.maximum(model.predict(matrix), 0)
    output = rows.copy()
    output["predicted_eta_minutes"] = predictions.round(1)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output):,} predictions to {args.output}")


if __name__ == "__main__":
    main()