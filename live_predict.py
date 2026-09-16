"""Minimal RailRadar -> Open-Meteo -> XGBoost ETA prediction."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from dotenv import load_dotenv

RAILRADAR_BASE_URL = "https://api.railradar.in/v1"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "wind_speed_10m",
    "cloud_cover",
    "surface_pressure",
]


def minutes_between(later, earlier):
    if not later or not earlier:
        return np.nan
    return (pd.Timestamp(later) - pd.Timestamp(earlier)).total_seconds() / 60


def get_live_train(train_number: str, journey_date: str | None) -> dict:
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise RuntimeError("API_KEY is missing. Add it to your local .env file.")
    params = {"haltsOnly": "true", "includeCoordinates": "true"}
    if journey_date:
        params["date"] = journey_date
    response = requests.get(
        f"{RAILRADAR_BASE_URL}/trains/{train_number}/live",
        headers={"Authorization": f"Bearer {api_key}"},
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        message = payload.get("error", {}).get("message", "RailRadar request failed.")
        raise RuntimeError(message)
    return payload["data"]


def get_hourly_weather(latitude: float, longitude: float, timestamp) -> dict:
    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(WEATHER_COLUMNS),
            "timezone": "Asia/Kolkata",
            "forecast_days": 1,
        },
        timeout=30,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]
    weather = pd.DataFrame(hourly)
    weather["time"] = pd.to_datetime(weather["time"])
    target_hour = pd.Timestamp(timestamp).tz_localize(None).floor("h")
    closest = (weather["time"] - target_hour).abs().idxmin()
    return weather.loc[closest, WEATHER_COLUMNS].to_dict()


def load_edge(previous_code: str | None, current_code: str, edges_path: Path) -> tuple:
    if not previous_code or not edges_path.exists():
        return np.nan, np.nan
    edges = pd.read_csv(edges_path)
    match = edges[
        (edges["from"].eq(previous_code) & edges["to"].eq(current_code))
        | (edges["from"].eq(current_code) & edges["to"].eq(previous_code))
    ]
    if match.empty:
        return np.nan, np.nan
    distance = float(match.iloc[0]["distance"])
    return (distance if distance > 0 else np.nan, float(match.iloc[0]["ntrains"]))


def make_live_features(live: dict, edges_path: Path) -> tuple[dict, pd.Timestamp]:
    route = sorted(live["route"], key=lambda stop: stop["sequence"])
    departed = [stop for stop in route if stop.get("actualDeparture")]
    if not departed:
        raise RuntimeError("The train has not departed yet.")

    current = departed[-1]
    current_index = route.index(current)
    previous = route[current_index - 1] if current_index else None
    next_stop = route[current_index + 1] if current_index + 1 < len(route) else None
    destination = route[-1]
    if current is destination:
        raise RuntimeError("The train has already reached its destination.")

    prediction_time = pd.Timestamp(current["actualDeparture"])
    live_time = pd.Timestamp(live.get("lastUpdatedAt") or pd.Timestamp.now(tz="Asia/Kolkata"))
    if prediction_time.tzinfo is None:
        prediction_time = prediction_time.tz_localize("Asia/Kolkata")
    if live_time.tzinfo is None:
        live_time = live_time.tz_localize("Asia/Kolkata")

    station_lat, station_lng = current.get("lat"), current.get("lng")
    if station_lat is None or station_lng is None:
        raise RuntimeError("RailRadar did not return coordinates for the latest departed halt.")

    # Weather is requested near the live segment position. The XGBoost location
    # fields stay at the latest station to match historical training.
    weather_lat, weather_lng = float(station_lat), float(station_lng)
    progress = live.get("currentLocation", {}).get("segmentProgress")
    if next_stop and progress is not None and next_stop.get("lat") is not None:
        progress = min(max(float(progress), 0.0), 1.0)
        weather_lat += progress * (float(next_stop["lat"]) - weather_lat)
        weather_lng += progress * (float(next_stop["lng"]) - weather_lng)
    weather = get_hourly_weather(weather_lat, weather_lng, live_time)

    previous_distance = (
        float(current["distance"]) - float(previous["distance"]) if previous else np.nan
    )
    network_distance, train_count = load_edge(
        previous.get("stationCode") if previous else None,
        current["stationCode"],
        edges_path,
    )
    actual_segment = (
        minutes_between(current.get("actualArrival"), previous.get("actualDeparture"))
        if previous
        else np.nan
    )
    segment_speed = (
        previous_distance / (actual_segment / 60)
        if previous_distance > 0 and actual_segment > 0
        else np.nan
    )
    if pd.notna(segment_speed) and segment_speed > 200:
        segment_speed = np.nan

    route_distance = float(live["train"]["distance"])
    current_distance = float(current["distance"])
    departure_delay = current.get("delayDeparture")
    if departure_delay is None:
        departure_delay = live.get("delayMinutes")

    features = {
        "latitude": float(station_lat),
        "longitude": float(station_lng),
        "station_index": current_index + 1,
        "route_station_count": len(route),
        "stations_remaining": len(route) - current_index - 1,
        "distance_from_source_km": current_distance,
        "route_distance_km": route_distance,
        "distance_remaining_km": route_distance - current_distance,
        "previous_segment_distance_km": previous_distance,
        "network_segment_distance_km": network_distance,
        "segment_train_count": train_count,
        "scheduled_segment_minutes": (
            minutes_between(current.get("scheduledArrival"), previous.get("scheduledDeparture"))
            if previous
            else np.nan
        ),
        "scheduled_remaining_minutes": minutes_between(
            destination.get("scheduledArrival"), current.get("scheduledDeparture")
        ),
        "scheduled_eta_from_prediction_minutes": minutes_between(
            destination.get("scheduledArrival"), prediction_time
        ),
        "current_arrival_delay_minutes": current.get("delayArrival"),
        "current_departure_delay_minutes": departure_delay,
        "previous_arrival_delay_minutes": previous.get("delayArrival") if previous else np.nan,
        "previous_departure_delay_minutes": (
            previous.get("delayDeparture") if previous else np.nan
        ),
        "actual_segment_minutes": actual_segment,
        "historical_segment_speed_kmph": segment_speed,
        **weather,
        "prediction_hour": prediction_time.hour,
        "prediction_day_of_week": prediction_time.dayofweek,
        "prediction_month": prediction_time.month,
    }
    print(weather)
    return features, live_time, prediction_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_number", help="5-digit train number, e.g. 12919")
    parser.add_argument("--date", help="Journey start date: YYYY-MM-DD")
    parser.add_argument("--model-dir", default="models/basic_xgboost")
    parser.add_argument("--edges", default="data/raw/IRN_edges.csv")
    args = parser.parse_args()

    load_dotenv()
    train_number = args.train_number.strip().zfill(5)
    model_dir = Path(args.model_dir)
    with open(model_dir / "feature_schema.json", encoding="utf-8") as file:
        features = json.load(file)["features_in_order"]

    live = get_live_train(train_number, args.date)
    values, live_time, prediction_time = make_live_features(live, Path(args.edges))
    matrix = pd.DataFrame([values]).reindex(columns=features)

    model = xgb.XGBRegressor()
    model.load_model(model_dir / "eta_model.json")
    eta_from_departure = max(float(model.predict(matrix)[0]), 0)
    elapsed = max((live_time - prediction_time).total_seconds() / 60, 0)
    remaining_eta = max(eta_from_departure - elapsed, 0)
    arrival = live_time + pd.Timedelta(minutes=remaining_eta)

    print(f"Train {train_number} — predicted arrival in {round(remaining_eta)} minutes")
    print(f"Predicted destination arrival: {arrival.strftime('%Y-%m-%d %H:%M %Z')}")


if __name__ == "__main__":
    main()