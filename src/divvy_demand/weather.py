import logging
import requests
import pandas as pd
from divvy_demand.infra.s3_client import S3Client

logger = logging.getLogger(__name__)

CHICAGO_LAT = 41.8827
CHICAGO_LON = -87.6233
START_DATE = "2020-01-01"
END_DATE = "2026-02-28"

DAILY_VARS = "temperature_2m_min,rain_sum,snowfall_sum,wind_speed_10m_max"


def _parse_response(data: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(data["time"]),
        "temp_min_c": data["temperature_2m_min"],
        "rain_sum_mm": data["rain_sum"],
        "snowfall_sum_cm": data["snowfall_sum"],
        "wind_speed_max_kmh": data["wind_speed_10m_max"],
    })


def fetch_historical(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """Fetch actual observed daily weather from Open-Meteo archive API."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={CHICAGO_LAT}&longitude={CHICAGO_LON}"
        f"&start_date={start}&end_date={end}"
        f"&daily={DAILY_VARS}&timezone=America%2FChicago"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = _parse_response(resp.json()["daily"])
    logger.info(f"Fetched historical weather: {len(df)} days ({start} → {end})")
    return df


def fetch_forecast_historical(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Fetch historical forecast weather from Open-Meteo historical-forecast API.

    This returns what the forecast model predicted for each day — the right
    feature to use for training since at inference time only a forecast is available.
    """
    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={CHICAGO_LAT}&longitude={CHICAGO_LON}"
        f"&start_date={start}&end_date={end}"
        f"&daily={DAILY_VARS}&timezone=America%2FChicago"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = _parse_response(resp.json()["daily"])
    logger.info(f"Fetched forecast-historical weather: {len(df)} days ({start} → {end})")
    return df


def build_weather(s3: S3Client, start: str = START_DATE, end: str = END_DATE) -> None:
    """Fetch both historical and forecast-historical weather and write to S3."""
    historical = fetch_historical(start, end)
    s3.write_processed(historical, "weather/historical")

    forecast = fetch_forecast_historical(start, end)
    s3.write_processed(forecast, "weather/forecast")
