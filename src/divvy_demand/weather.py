"""
Weather ETL — daily Chicago weather from Open-Meteo.

Two tables are written, same schema:
    processed/weather/historical   actual observations (archive API)
    processed/weather/forecast     what the forecast model predicted (historical-forecast API)

Use `forecast` for test/inference features since only a forecast is available
at prediction time; `historical` is for EDA and forecast-quality checks.
"""
import logging
import requests
import pandas as pd
from divvy_demand.infra.s3_client import S3Client

logger = logging.getLogger(__name__)

CHICAGO_LAT = 41.8827
CHICAGO_LON = -87.6233
START_DATE = "2020-01-01"
END_DATE = "2026-02-28"

# Open-Meteo daily variable -> output column name
VAR_MAP = {
    # original four (names unchanged for backward compatibility)
    "temperature_2m_min":        "temp_min_c",
    "rain_sum":                  "rain_sum_mm",
    "snowfall_sum":              "snowfall_sum_cm",
    "wind_speed_10m_max":        "wind_speed_max_kmh",
    # temperature
    "temperature_2m_max":        "temp_max_c",
    "temperature_2m_mean":       "temp_mean_c",
    "apparent_temperature_max":  "apparent_temp_max_c",
    "apparent_temperature_min":  "apparent_temp_min_c",
    # precipitation
    "precipitation_sum":         "precip_sum_mm",
    "precipitation_hours":       "precip_hours",
    "weather_code":              "weather_code",       # WMO code, see WMO_GROUP below
    # wind
    "wind_gusts_10m_max":        "wind_gust_max_kmh",
    # sun — deterministic, zero forecast error
    "daylight_duration":         "daylight_s",
    "sunshine_duration":         "sunshine_s",
    "sunrise":                   "sunrise",
    "sunset":                    "sunset",
    "shortwave_radiation_sum":   "radiation_mj_m2",
}
DAILY_VARS = ",".join(VAR_MAP)

# Coarse grouping of WMO weather codes into an ordinal severity (0 = clear ... 4 = thunderstorm)
WMO_GROUP = {
    0: 0, 1: 0, 2: 0, 3: 0,                       # clear / cloudy
    45: 1, 48: 1,                                 # fog
    51: 2, 53: 2, 55: 2, 56: 2, 57: 2,            # drizzle
    61: 2, 63: 2, 65: 2, 66: 2, 67: 2,            # rain
    80: 2, 81: 2, 82: 2,                          # showers
    71: 3, 73: 3, 75: 3, 77: 3, 85: 3, 86: 3,     # snow
    95: 4, 96: 4, 99: 4,                          # thunderstorm
}


def _parse_response(data: dict) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(data["time"])})
    for var, col in VAR_MAP.items():
        df[col] = data[var]

    # Derived: hours instead of seconds, local clock-time of sunrise/sunset
    df["daylight_hours"] = df["daylight_s"] / 3600.0
    df["sunshine_hours"] = df["sunshine_s"] / 3600.0
    for c in ["sunrise", "sunset"]:
        t = pd.to_datetime(df[c])
        df[f"{c}_hour"] = t.dt.hour + t.dt.minute / 60.0
    df["weather_severity"] = df["weather_code"].map(WMO_GROUP).fillna(2).astype(int)
    df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]

    return df.drop(columns=["daylight_s", "sunshine_s", "sunrise", "sunset"])


def _fetch(base_url: str, start: str, end: str) -> pd.DataFrame:
    url = (
        f"{base_url}?latitude={CHICAGO_LAT}&longitude={CHICAGO_LON}"
        f"&start_date={start}&end_date={end}"
        f"&daily={DAILY_VARS}&timezone=America%2FChicago"
    )
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return _parse_response(resp.json()["daily"])


def fetch_historical(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Fetch actual observed daily weather from Open-Meteo archive API."""
    df = _fetch("https://archive-api.open-meteo.com/v1/archive", start, end)
    logger.info(f"Fetched historical weather: {len(df)} days x {df.shape[1] - 1} vars ({start} → {end})")
    return df


def fetch_forecast_historical(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Fetch historical forecast weather from Open-Meteo historical-forecast API.

    This returns what the forecast model predicted for each day — the right
    feature to use for training since at inference time only a forecast is available.
    """
    df = _fetch("https://historical-forecast-api.open-meteo.com/v1/forecast", start, end)
    logger.info(f"Fetched forecast-historical weather: {len(df)} days x {df.shape[1] - 1} vars ({start} → {end})")
    return df


def build_weather(s3: S3Client, start: str = START_DATE, end: str = END_DATE) -> None:
    """Fetch both historical and forecast-historical weather and write to S3."""
    historical = fetch_historical(start, end)
    s3.write_processed(historical, "weather/historical")

    forecast = fetch_forecast_historical(start, end)
    s3.write_processed(forecast, "weather/forecast")
