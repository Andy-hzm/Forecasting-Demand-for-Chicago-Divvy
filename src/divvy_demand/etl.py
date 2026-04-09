import logging
import pandas as pd
from divvy_demand.infra.s3_client import S3Client

logger = logging.getLogger(__name__)

START_YEAR, START_MONTH = 2020, 1
END_YEAR, END_MONTH = 2026, 2


def _iter_months():
    y, m = START_YEAR, START_MONTH
    while (y, m) <= (END_YEAR, END_MONTH):
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def _load_all_raw(s3: S3Client, start: str = None, end: str = None) -> pd.DataFrame:
    start_ts = pd.Timestamp(start) if start else pd.Timestamp(f"{START_YEAR}-{START_MONTH:02d}-01")
    end_ts = pd.Timestamp(end) if end else pd.Timestamp(f"{END_YEAR}-{END_MONTH:02d}-28")

    frames = []
    for year, month in _iter_months():
        month_start = pd.Timestamp(f"{year}-{month:02d}-01")
        month_end = month_start + pd.offsets.MonthEnd(0)
        if month_end < start_ts or month_start > end_ts:
            continue
        df = s3.read_raw(year, month)
        if df.empty:
            logger.warning(f"No data for {year}-{month:02d}, skipping")
            continue
        logger.info(f"Loaded {year}-{month:02d}: {len(df):,} rows")
        frames.append(df)

    if not frames:
        raise RuntimeError("No raw data found on S3")

    df = pd.concat(frames, ignore_index=True)
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["date"] = pd.to_datetime(df["started_at"].dt.date)
    df["is_ebike"] = (df["rideable_type"] == "electric_bike").astype(int)
    df["is_member"] = (df["member_casual"] == "member").astype(int)
    df["is_casual"] = (df["member_casual"] == "casual").astype(int)
    return df


def build_overall(s3: S3Client) -> pd.DataFrame:
    """
    Aggregate trip-level data to daily city-wide counts.

    Columns: date, total_rides, ebike_rides, member_rides, casual_rides
    """
    df = _load_all_raw(s3)

    overall = df.groupby("date").agg(
        total_rides=("ride_id", "nunique"),
        ebike_rides=("is_ebike", "sum"),
        member_rides=("is_member", "sum"),
        casual_rides=("is_casual", "sum"),
    ).reset_index().sort_values("date")

    logger.info(f"Overall: {len(overall)} days, {overall['total_rides'].sum():,} total rides")
    return overall


def build_station(s3: S3Client) -> pd.DataFrame:
    """
    Aggregate trip-level data to daily station-level counts.

    E-bike rides with no dock are labelled 'Outside of Dock'.
    Columns: date, start_station_name, start_station_id, total_rides, ebike_rides,
             member_rides, casual_rides
    """
    df = _load_all_raw(s3)

    # Label dockless e-bike departures
    mask = (df["rideable_type"] == "electric_bike") & df["start_station_name"].isna()
    df.loc[mask, "start_station_name"] = "Outside of Dock"
    df.loc[mask, "start_station_id"] = "outside_of_dock"

    station = df.groupby(["date", "start_station_name", "start_station_id"]).agg(
        total_rides=("ride_id", "nunique"),
        ebike_rides=("is_ebike", "sum"),
        member_rides=("is_member", "sum"),
        casual_rides=("is_casual", "sum"),
    ).reset_index().sort_values(["date", "start_station_name"])

    logger.info(f"Station: {len(station):,} rows, {station['start_station_name'].nunique()} unique stations")
    return station


def build_trip(s3: S3Client, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Aggregate trip-level data to daily origin-destination counts.

    Excludes trips where start or end station is missing (outside of dock).
    Columns: date, start_station_name, start_station_id,
             end_station_name, end_station_id, total_rides, member_rides, casual_rides
    """
    df = _load_all_raw(s3, start=start, end=end)
    if start:
        df = df[df["date"] >= start]
    if end:
        df = df[df["date"] <= end]

    # Drop trips with missing start or end station (outside of dock)
    df = df.dropna(subset=["start_station_name", "end_station_name",
                            "start_station_id", "end_station_id"])

    trip = df.groupby([
        "date",
        "start_station_name", "start_station_id",
        "end_station_name", "end_station_id",
    ]).agg(
        total_rides=("ride_id", "nunique"),
        member_rides=("is_member", "sum"),
        casual_rides=("is_casual", "sum"),
    ).reset_index().sort_values(["date", "start_station_name", "end_station_name"])

    logger.info(f"Trip: {len(trip):,} rows, {trip[['start_station_name','end_station_name']].drop_duplicates().shape[0]:,} unique routes")
    return trip
