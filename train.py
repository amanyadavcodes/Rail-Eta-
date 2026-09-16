"""Train an executable XGBoost remaining-ETA regression model.

The script deliberately uses only numeric features that can be reconstructed
for a live prediction. It performs a chronological split by service date:

  train: 2024-09-01 through 2024-09-20
  validation: 2024-09-21 through 2024-09-25
  test: 2024-09-26 through 2024-09-30
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

TARGET = "actual_eta_minutes"
DATE_COLUMN = "service_date"

# No future actual timestamp or future delay appears in this list.
FEATURES = [
    "latitude",
    "longitude",
    "station_index",
    "route_station_count",
    "stations_remaining",
    "distance_from_source_km",
    "route_distance_km",
    "distance_remaining_km",
    "previous_segment_distance_km",
    "network_segment_distance_km",
    "segment_train_count",
    "scheduled_segment_minutes",
    "scheduled_remaining_minutes",
    "scheduled_eta_from_prediction_minutes",
    "current_arrival_delay_minutes",
    "current_departure_delay_minutes",
    "previous_arrival_delay_minutes",
    "previous_departure_delay_minutes",
    "actual_segment_minutes",
    "historical_segment_speed_kmph",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure",
    "prediction_hour",
    "prediction_day_of_week",
    "prediction_month",
]


def metrics(y_true: pd.Series, predictions: np.ndarray) -> dict:
    return {
        "mae_minutes": float(mean_absolute_error(y_true, predictions)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
        "rows": int(len(y_true)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/processed/final_eta_dataset.csv",
        help="Prepared ETA CSV.",
    )
    parser.add_argument("--output-dir", default="models/basic_xgboost")
    parser.add_argument("--train-end", default="2024-09-20")
    parser.add_argument("--validation-end", default="2024-09-25")
    parser.add_argument("--test-end", default="2024-09-30")
    parser.add_argument("--estimators", type=int, default=700)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 150 trees for a fast pipeline check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [DATE_COLUMN, TARGET, *FEATURES]
    data = pd.read_csv(data_path, usecols=required, parse_dates=[DATE_COLUMN])
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    train_end = pd.Timestamp(args.train_end)
    validation_end = pd.Timestamp(args.validation_end)
    test_end = pd.Timestamp(args.test_end)
    train_mask = data[DATE_COLUMN].le(train_end)
    validation_mask = data[DATE_COLUMN].gt(train_end) & data[DATE_COLUMN].le(validation_end)
    test_mask = data[DATE_COLUMN].gt(validation_end) & data[DATE_COLUMN].le(test_end)
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("One or more chronological splits are empty.")

    x_train, y_train = data.loc[train_mask, FEATURES], data.loc[train_mask, TARGET]
    x_validation = data.loc[validation_mask, FEATURES]
    y_validation = data.loc[validation_mask, TARGET]
    x_test, y_test = data.loc[test_mask, FEATURES], data.loc[test_mask, TARGET]

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_estimators=150 if args.quick else args.estimators,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        missing=np.nan,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_validation, y_validation)],
        verbose=25,
    )

    validation_predictions = np.maximum(model.predict(x_validation), 0)
    test_predictions = np.maximum(model.predict(x_test), 0)
    scheduled_baseline = np.maximum(
        x_test["scheduled_eta_from_prediction_minutes"].fillna(
            x_train["scheduled_eta_from_prediction_minutes"].median()
        ),
        0,
    )

    result = {
        "model_type": "XGBRegressor",
        "objective": "remaining ETA in minutes",
        "target": TARGET,
        "features": FEATURES,
        "split": {
            "train": f"<= {args.train_end}",
            "validation": f"{args.train_end} < date <= {args.validation_end}",
            "test": f"{args.validation_end} < date <= {args.test_end}",
        },
        "best_iteration": (
            int(model.best_iteration)
            if getattr(model, "best_iteration", None) is not None
            else None
        ),
        "validation": metrics(y_validation, validation_predictions),
        "test": metrics(y_test, test_predictions),
        "scheduled_timetable_baseline_test": metrics(y_test, scheduled_baseline),
    }

    model.save_model(output_dir / "eta_model.json")
    with open(output_dir / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "target": TARGET,
                "features_in_order": FEATURES,
                "missing_value_policy": "Leave unavailable numeric values as NaN.",
                "prediction_unit": "minutes",
            },
            f,
            indent=2,
        )
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()