# Final ETA dataset: data dictionary

## Row grain

One row is one historical prediction point immediately after a train's actual
departure from a non-terminal station.

## Target

- `actual_eta_minutes`: minutes from `prediction_timestamp` to the actual
  arrival at the final station of that train's route.

## Identifiers and route position

- `train_number`, `train_name`, `service_date`
- `station_code`, `station_name`, `zone`
- `station_index`, `route_station_count`, `stations_remaining`
- `previous_station_code`, `next_station_code`
- `distance_from_source_km`, `route_distance_km`,
  `distance_remaining_km`

## Segment and schedule features

- `previous_segment_distance_km`: route-distance difference for the completed
  segment.
- `network_segment_distance_km`, `segment_train_count`: values from the
  undirected IRN edge table.
- `scheduled_arrival_timestamp`, `scheduled_departure_timestamp`
- `prediction_timestamp`: scheduled departure plus the recorded departure
  delay.
- `scheduled_segment_minutes`, `scheduled_remaining_minutes`
- `scheduled_eta_from_prediction_minutes`: destination's scheduled arrival
  minus the actual prediction timestamp. It can be negative when the train is
  already later than the destination's scheduled arrival.

## Delay and completed-segment features

- `current_arrival_delay_minutes`, `current_departure_delay_minutes`
- `previous_arrival_delay_minutes`, `previous_departure_delay_minutes`
- `actual_segment_minutes`: actual arrival at the current station minus actual
  departure from the previous station; nonpositive source inconsistencies are
  left empty.
- `historical_segment_speed_kmph`: completed segment distance divided by
  completed segment time. Inconsistent derived values above 200 km/h are left
  empty, not imputed.

## Coordinates and weather

- `latitude`, `longitude`, `coordinate_source`
- `weather_hour`
- `temperature_2m`, `relative_humidity_2m`, `precipitation`, `rain`
- `weather_code`, `wind_speed_10m`, `cloud_cover`, `surface_pressure`
- `weather_source`

Weather is matched to the current station and the hour containing
`prediction_timestamp`. Open-Meteo requests use a 0.1-degree grid and
`Asia/Kolkata` local time. Exact station codes without a verified coordinate
retain empty coordinate and weather fields.

## Calendar features

- `prediction_hour`
- `prediction_day_of_week`: Monday = 0 through Sunday = 6
- `prediction_month`

## Important training notes

- Read `train_number` as a string to preserve leading zeroes.
- Use a chronological split based on `service_date`, not a random row split.
- Do not treat identifiers, timestamps, or source-label columns as numeric
  features without explicit encoding.
- Empty values are genuine unavailable/invalid-source values; no synthetic
  imputations were written into the CSV.