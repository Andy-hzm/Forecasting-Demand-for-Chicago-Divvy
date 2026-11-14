"""
Ingest raw Divvy trip data from the public S3 bucket into our S3 as Parquet.

Usage:
    python scripts/ingest.py                        # ingest all available months
    python scripts/ingest.py --year 2024            # ingest a specific year
    python scripts/ingest.py --year 2024 --month 6  # ingest a specific month
    python scripts/ingest.py --skip-existing        # skip months already on S3
"""

import argparse
import logging
import requests
import zipfile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv

from divvy_demand.infra.s3_client import S3Client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DIVVY_BASE_URL = "https://divvy-tripdata.s3.amazonaws.com/"
RAW_COLUMNS = [
    "ride_id", "rideable_type", "started_at", "ended_at",
    "start_station_name", "start_station_id",
    "end_station_name", "end_station_id",
    "start_lat", "start_lng", "end_lat", "end_lng",
    "member_casual",
]


def _zip_filename(year: int, month: int | None) -> str:
    if year == 2020 and month is None:
        return "Divvy_Trips_2020_Q1.zip"
    return f"{year}{month:02d}-divvy-tripdata.zip"


def download_month(year: int, month: int | None) -> pd.DataFrame | None:
    filename = _zip_filename(year, month)
    url = f"{DIVVY_BASE_URL}{filename}"
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(
                    f,
                    usecols=lambda c: c in RAW_COLUMNS,
                    dtype={"start_station_id": str, "end_station_id": str},
                )
        logger.info(f"Downloaded {filename}: {len(df):,} rows")
        return df
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            logger.info(f"Not available yet: {filename}")
            return None
        raise


def ingest_month(s3: S3Client, year: int, month: int, skip_existing: bool) -> bool:
    """Download one month, upload to S3 as Parquet. Returns True if written."""
    if skip_existing and s3.raw_exists(year, month):
        logger.info(f"Skipping {year}-{month:02d} (already on S3)")
        return False

    # Q1 2020 is a single quarterly file covering Jan–Mar
    if year == 2020 and month <= 3:
        df = download_month(2020, None)
        if df is None:
            return False
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
        for m in range(1, 4):
            chunk = df[df["started_at"].dt.month == m].copy()
            if not chunk.empty:
                s3.write_raw(chunk, 2020, m)
        return True

    df = download_month(year, month)
    if df is None:
        return False

    s3.write_raw(df, year, month)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, help="Ingest a specific year only")
    parser.add_argument("--month", type=int, help="Ingest a specific month (requires --year)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip months already on S3")
    parser.add_argument("--env", default="prod", choices=["prod", "dev"], help="S3 environment (default: prod)")
    args = parser.parse_args()

    s3 = S3Client(env=args.env)

    # Build list of (year, month) pairs to ingest
    if args.year and args.month:
        jobs = [(args.year, args.month)]
    elif args.year:
        jobs = [(args.year, m) for m in range(1, 13)]
    else:
        # All months from 2020 to Feb 2026
        end = pd.Timestamp("2026-02-01")
        jobs = [
            (y, m)
            for y in range(2020, 2027)
            for m in range(1, 13)
            if pd.Timestamp(y, m, 1) <= end
        ]

    # Skip Q1 2020 individual months — handled as a batch inside ingest_month
    jobs = [(y, m) for y, m in jobs if not (y == 2020 and m <= 3)]
    # Add a single sentinel for Q1 2020 (month=1 triggers the batch download)
    if not args.month and (not args.year or args.year == 2020):
        jobs = [(2020, 1)] + jobs

    written = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(ingest_month, s3, y, m, args.skip_existing): (y, m)
            for y, m in jobs
        }
        for future in as_completed(futures):
            y, m = futures[future]
            try:
                if future.result():
                    written += 1
            except Exception as e:
                logger.error(f"Failed {y}-{m:02d}: {e}")

    logger.info(f"Done. {written} month(s) written to S3.")


if __name__ == "__main__":
    main()
