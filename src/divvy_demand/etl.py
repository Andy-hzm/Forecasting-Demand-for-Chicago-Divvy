import logging
import numpy as np
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


# Manhattan distance conversion factors at Chicago's latitude (~41.88°)
_KM_PER_DEG_LAT = 111.0
_KM_PER_DEG_LNG = 111.0 * np.cos(np.radians(41.88))  # ≈ 82.7


def build_station(s3: S3Client) -> pd.DataFrame:
    """
    Aggregate trip-level data to daily station-level counts with within-day shape features.

    E-bike rides with no dock are labelled 'Outside of Dock'.

    Columns:
      - date, start_station_name, start_station_id
      - total_rides, ebike_rides, member_rides, casual_rides
      - hour_00_pct ... hour_23_pct                     # 24 hourly fractions, sum to 1
      - frac_round_trips                                 # share of trips ending at same station
      - mh_dist_p10_km, p25, p50, p75, p90               # Manhattan-distance quantiles (km)
    """
    df = _load_all_raw(s3)

    # Label dockless e-bike departures
    mask = (df["rideable_type"] == "electric_bike") & df["start_station_name"].isna()
    df.loc[mask, "start_station_name"] = "Outside of Dock"
    df.loc[mask, "start_station_id"] = "outside_of_dock"

    # Derived columns for shape features
    df["hour"] = df["started_at"].dt.hour
    df["is_round_trip"] = (
        df["start_station_name"].notna()
        & (df["start_station_name"] == df["end_station_name"])
    ).astype(int)
    df["mh_km"] = (
        (df["end_lat"] - df["start_lat"]).abs() * _KM_PER_DEG_LAT
        + (df["end_lng"] - df["start_lng"]).abs() * _KM_PER_DEG_LNG
    )

    group_keys = ["date", "start_station_name", "start_station_id"]

    # 1) Base counts + round-trip fraction
    base = df.groupby(group_keys).agg(
        total_rides=("ride_id", "nunique"),
        ebike_rides=("is_ebike", "sum"),
        member_rides=("is_member", "sum"),
        casual_rides=("is_casual", "sum"),
        frac_round_trips=("is_round_trip", "mean"),
    ).reset_index()

    # 2) Hourly fractions — pivot to wide, normalize per row
    logger.info("Building hourly fractions...")
    hourly = (
        df.groupby(["date", "start_station_name", "hour"])
        .size()
        .rename("n")
        .unstack(level="hour", fill_value=0)
    )
    for h in range(24):
        if h not in hourly.columns:
            hourly[h] = 0
    hourly = hourly[list(range(24))]
    hourly = hourly.div(hourly.sum(axis=1), axis=0)
    hourly.columns = [f"hour_{h:02d}_pct" for h in range(24)]
    hourly = hourly.reset_index()

    # 3) Manhattan-distance quantiles — only over trips with valid end coords
    logger.info("Building Manhattan-distance quantiles...")
    valid = df[df["mh_km"].notna()]
    quantiles = (
        valid.groupby(["date", "start_station_name"])["mh_km"]
        .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
        .unstack()
    )
    quantiles.columns = [f"mh_dist_p{int(q * 100):02d}_km" for q in quantiles.columns]
    quantiles = quantiles.reset_index()

    # Merge
    station = (
        base
        .merge(hourly,    on=["date", "start_station_name"], how="left")
        .merge(quantiles, on=["date", "start_station_name"], how="left")
        .sort_values(["date", "start_station_name"])
        .reset_index(drop=True)
    )

    logger.info(
        f"Station: {len(station):,} rows, "
        f"{station['start_station_name'].nunique()} unique stations, "
        f"{station.shape[1]} columns"
    )
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


def build_station_flow(s3: S3Client) -> pd.DataFrame:
    """
    Daily inbound / outbound / net flow per station.

    net_flow > 0 means more bikes arrived than left (station filling up);
    yesterday's net_flow is a cheap proxy for "bikes available this morning".
    Dockless e-bike ends are dropped (no end station).

    Aggregates month by month so it runs on a laptop (never holds all raw trips
    at once). Trips ending in the next month are counted on their end date.

    Columns: date, start_station_name (canonical, see station_meta), outbound_rides,
             inbound_rides, net_flow, inbound_ebike, inbound_member
    """
    out_parts, in_parts = [], []
    for year, month in _iter_months():
        df = s3.read_raw(year, month)
        if df.empty:
            logger.warning(f"No data for {year}-{month:02d}, skipping")
            continue
        df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
        df["ended_at"]   = pd.to_datetime(df["ended_at"],   errors="coerce")
        df["is_ebike"]   = (df["rideable_type"] == "electric_bike").astype(int)
        df["is_member"]  = (df["member_casual"] == "member").astype(int)

        o = df.dropna(subset=["start_station_name", "started_at"]).copy()
        o["date"] = pd.to_datetime(o["started_at"].dt.date)
        out_parts.append(
            o.groupby(["date", "start_station_name"]).agg(outbound_rides=("ride_id", "nunique")).reset_index()
        )

        e = df.dropna(subset=["end_station_name", "ended_at"]).copy()
        e["date"] = pd.to_datetime(e["ended_at"].dt.date)
        in_parts.append(
            e.groupby(["date", "end_station_name"])
            .agg(inbound_rides=("ride_id", "nunique"),
                 inbound_ebike=("is_ebike", "sum"),
                 inbound_member=("is_member", "sum"))
            .reset_index()
            .rename(columns={"end_station_name": "start_station_name"})
        )
        logger.info(f"Flow {year}-{month:02d}: {len(df):,} trips")

    if not out_parts:
        raise RuntimeError("No raw data found on S3")

    # Month boundaries: a (date, station) can appear in two monthly parts — sum them
    outbound = pd.concat(out_parts).groupby(["date", "start_station_name"], as_index=False).sum()
    inbound  = pd.concat(in_parts).groupby(["date", "start_station_name"], as_index=False).sum()

    flow = outbound.merge(inbound, on=["date", "start_station_name"], how="outer")
    for c in ["outbound_rides", "inbound_rides", "inbound_ebike", "inbound_member"]:
        flow[c] = flow[c].fillna(0).astype(int)

    # Merge station aliases (needs processed/station_meta; silently skipped if absent)
    meta = s3.read_processed("station_meta")
    if len(meta) and "canonical_name" in meta:
        alias = dict(zip(meta["start_station_name"], meta["canonical_name"]))
        flow["start_station_name"] = flow["start_station_name"].map(alias).fillna(flow["start_station_name"])
        flow = flow.groupby(["date", "start_station_name"], as_index=False).sum()
        logger.info("Flow: station aliases merged")
    flow["net_flow"] = flow["inbound_rides"] - flow["outbound_rides"]
    flow = flow.sort_values(["start_station_name", "date"]).reset_index(drop=True)

    logger.info(f"Flow: {len(flow):,} rows, {flow['start_station_name'].nunique()} stations")
    return flow[["date", "start_station_name", "outbound_rides", "inbound_rides", "net_flow",
                 "inbound_ebike", "inbound_member"]]


def build_station_spatial(s3: S3Client, k: int = 5, radius_km: float = 1.0, meta: pd.DataFrame = None) -> pd.DataFrame:
    """
    Spatial-lag features: what happened *yesterday* at nearby stations and city-wide.

    All columns are already shifted by one day (``_lag1``) so they are safe to merge
    onto a row for date t without further shifting. Rolling windows are computed on
    the lagged series (window ends at t-1).

    Columns: date, start_station_name,
             knn{k}_rides_lag1, knn{k}_rides_roll7_lag1,       # mean over k nearest stations
             r{radius}km_rides_lag1,                           # sum over stations within radius
             city_rides_lag1, city_rides_roll7_lag1, city_rides_roll28_lag1,
             n_active_stations_lag1                            # stations with >=1 ride yesterday
    Rows cover each station's own [first_seen, last_seen] window only. knn mean is
    NaN on days when none of the k neighbours existed yet. Station names are the
    *canonical* names from station_meta (aliases merged) — canonicalize your station
    frame with features.canonicalize_stations() before merging.
    Requires processed/station and processed/station_meta (for coordinates);
    pass `meta` to skip the S3 read (e.g. right after building it).
    """
    from sklearn.neighbors import BallTree

    station = s3.read_processed("station")
    station["date"] = pd.to_datetime(station["date"])
    station = station[station["start_station_name"] != "Outside of Dock"]

    meta = s3.read_processed("station_meta") if meta is None else meta
    if "canonical_name" in meta:
        from divvy_demand.station_meta import collapse_meta
        alias = dict(zip(meta["start_station_name"], meta["canonical_name"]))
        station["start_station_name"] = station["start_station_name"].map(alias).fillna(station["start_station_name"])
        meta = collapse_meta(meta)
    meta = meta.dropna(subset=["lat", "lng"])
    meta = meta[meta["start_station_name"].isin(station["start_station_name"].unique())]
    names = meta["start_station_name"].tolist()
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Dense (date x station) ride matrix, 0 where a station had no departures
    rides = (
        station.pivot_table(index="date", columns="start_station_name",
                            values="total_rides", aggfunc="sum", fill_value=0)
        .reindex(pd.date_range(station["date"].min(), station["date"].max(), freq="D"), fill_value=0)
        .reindex(columns=names, fill_value=0)
    )
    rides.index.name = "date"
    R = rides.to_numpy(dtype="float64")                              # (T, S)

    # Neighbour structure
    coords_rad = np.radians(meta[["lat", "lng"]].to_numpy())
    tree = BallTree(coords_rad, metric="haversine")
    _, knn_idx = tree.query(coords_rad, k=k + 1)                     # first column is self
    knn_idx = knn_idx[:, 1:]
    radius_idx = tree.query_radius(coords_rad, r=radius_km / 6371.0)

    # A neighbour that had not opened yet (or had closed) must not count as "0 rides":
    # mask it out of the window so the mean is over stations that actually existed.
    dates = rides.index.to_numpy()
    first = pd.to_datetime(meta["first_seen"]).to_numpy()
    last  = pd.to_datetime(meta["last_seen"]).to_numpy()
    alive = (dates[:, None] >= first[None, :]) & (dates[:, None] <= last[None, :])   # (T, S)
    R_alive = np.where(alive, R, np.nan)

    with np.errstate(invalid="ignore"):
        knn_mean = np.nanmean(R_alive[:, knn_idx], axis=2)                 # (T, S), nan if no live neighbour
    radius_sum = np.stack(
        [np.nansum(R_alive[:, np.setdiff1d(nb, [i])], axis=1) for i, nb in enumerate(radius_idx)], axis=1
    )

    city = pd.Series(R.sum(axis=1), index=rides.index)
    n_active = pd.Series((R > 0).sum(axis=1), index=rides.index)

    knn_df = pd.DataFrame(knn_mean, index=rides.index, columns=names)
    rad_df = pd.DataFrame(radius_sum, index=rides.index, columns=names)

    # Shift everything by one day, then roll on the shifted series
    out = pd.concat({
        f"knn{k}_rides_lag1":        knn_df.shift(1),
        f"knn{k}_rides_roll7_lag1":  knn_df.shift(1).rolling(7, min_periods=1).mean(),
        f"r{radius_km:g}km_rides_lag1": rad_df.shift(1),
    }, axis=1)
    out = out.stack(future_stack=True).reset_index()
    out.columns = ["date", "start_station_name"] + [c for c in out.columns[2:]]

    city_df = pd.DataFrame({
        "date":                   rides.index,
        "city_rides_lag1":        city.shift(1).to_numpy(),
        "city_rides_roll7_lag1":  city.shift(1).rolling(7,  min_periods=1).mean().to_numpy(),
        "city_rides_roll28_lag1": city.shift(1).rolling(28, min_periods=1).mean().to_numpy(),
        "n_active_stations_lag1": n_active.shift(1).to_numpy(),
    })
    out = out.merge(city_df, on="date", how="left")
    # Keep only rows inside each station's own active window; first day per station has no lag
    alive_long = pd.DataFrame(alive, index=rides.index, columns=names).stack(future_stack=True)
    alive_long.index.names = ["date", "start_station_name"]
    out = out.set_index(["date", "start_station_name"])
    out = out[alive_long.reindex(out.index).to_numpy()].reset_index()
    out = out.dropna(subset=["city_rides_lag1"]).sort_values(["start_station_name", "date"]).reset_index(drop=True)

    logger.info(f"Spatial: {len(out):,} rows, {len(names)} stations, k={k}, radius={radius_km}km")
    return out
