"""
Station metadata ETL — one row per station name.

processed/station_meta
    start_station_name   str      # join key used everywhere else in the project
    lat, lng             float    # mean of trip start coords (raw data), GBFS fallback
    capacity             float    # dock count from Divvy GBFS, NaN if station no longer in feed
    is_charging          float    # 1/0 e-bike charging station (GBFS), NaN if not in feed
    in_gbfs              int      # 1 if matched to the live GBFS feed
    first_seen, last_seen datetime  # first / last date with a departure in processed/station
    n_active_days        int
    canonical_name       str      # see build_station_aliases(); == start_station_name for most rows

GBFS is a live snapshot, so capacity reflects today, not history — fine as a
static station attribute, don't read it as a time series.
"""
import logging

import numpy as np
import pandas as pd
import requests

from divvy_demand.infra.s3_client import S3Client

logger = logging.getLogger(__name__)

GBFS_STATION_INFO_URL = "https://gbfs.lyft.com/gbfs/2.3/chi/en/station_information.json"


def fetch_gbfs_station_info() -> pd.DataFrame:
    resp = requests.get(GBFS_STATION_INFO_URL, timeout=30)
    resp.raise_for_status()
    stations = resp.json()["data"]["stations"]
    df = pd.DataFrame([{
        "start_station_name": s["name"],
        "gbfs_station_id":    s["station_id"],
        "gbfs_short_name":    s.get("short_name"),
        "gbfs_lat":           s["lat"],
        "gbfs_lng":           s["lon"],
        "capacity":           s.get("capacity"),
        "is_charging":        int(bool(s.get("is_charging", False))),
    } for s in stations])
    # A handful of names repeat in the feed (virtual + physical); keep the larger dock
    df = df.sort_values("capacity", ascending=False).drop_duplicates("start_station_name")
    logger.info(f"GBFS: {len(df)} stations, median capacity {df['capacity'].median():.0f}")
    return df


SAMPLE_MONTHS = (3, 6, 9)   # one month per season is enough to see every station that existed


def _coords_from_raw(s3: S3Client, sample_months: tuple = SAMPLE_MONTHS) -> pd.DataFrame:
    """
    Median start lat/lng per station name. Reads only `sample_months` of each year
    (pass None to scan everything) — stations that only existed outside the sampled
    months fall back to GBFS coordinates.
    """
    from divvy_demand.etl import _iter_months
    frames = []
    for year, month in _iter_months():
        if sample_months and month not in sample_months:
            continue
        raw = s3.read_raw(year, month)
        if raw.empty:
            continue
        frames.append(
            raw[["start_station_name", "start_lat", "start_lng"]]
            .dropna()
            .groupby("start_station_name")
            .median()
            .reset_index()
        )
    per_month = pd.concat(frames, ignore_index=True)
    coords = (
        per_month.groupby("start_station_name")[["start_lat", "start_lng"]]
        .median()
        .rename(columns={"start_lat": "lat", "start_lng": "lng"})
        .reset_index()
    )
    logger.info(f"Raw coords: {len(coords)} stations from {len(frames)} sampled months")
    return coords


def build_station_meta(s3: S3Client, coords_from_raw: bool = True) -> pd.DataFrame:
    """
    coords_from_raw=True reads a sample of raw months (3/6/9 of each year, ~1-2 min
    locally) to get trip-derived coordinates for stations that have since closed or
    been renamed in GBFS. Set False to use GBFS coordinates only (seconds).
    """
    station = s3.read_processed("station")
    station["date"] = pd.to_datetime(station["date"])
    station = station[station["start_station_name"] != "Outside of Dock"]

    activity = station.groupby("start_station_name").agg(
        first_seen=("date", "min"),
        last_seen=("date", "max"),
        n_active_days=("date", "nunique"),
    ).reset_index()

    gbfs = fetch_gbfs_station_info()
    meta = activity.merge(gbfs, on="start_station_name", how="left")
    meta["in_gbfs"] = meta["gbfs_station_id"].notna().astype(int)

    if coords_from_raw:
        coords = _coords_from_raw(s3)
        meta = meta.merge(coords, on="start_station_name", how="left")
        meta["lat"] = meta["lat"].fillna(meta["gbfs_lat"])
        meta["lng"] = meta["lng"].fillna(meta["gbfs_lng"])
    else:
        meta["lat"] = meta["gbfs_lat"]
        meta["lng"] = meta["gbfs_lng"]

    missing = meta["lat"].isna().sum()
    logger.info(
        f"Station meta: {len(meta)} stations, {meta['in_gbfs'].sum()} in GBFS, "
        f"{missing} without coords"
    )
    cols = ["start_station_name", "lat", "lng", "capacity", "is_charging", "in_gbfs",
            "gbfs_station_id", "gbfs_short_name", "first_seen", "last_seen", "n_active_days"]
    meta = meta[cols].sort_values("start_station_name").reset_index(drop=True)

    aliases = build_station_aliases(meta)
    meta = meta.merge(aliases[["start_station_name", "canonical_name"]], on="start_station_name", how="left")
    return meta


# ---------------------------------------------------------------------------
# Station aliases — the same physical station under several names
# ---------------------------------------------------------------------------

import re

# Names that are clearly not public stations (ops / test / event parking) — never merged by coords
_JUNK_RE = re.compile(r"test|\bhq\b|warehouse|\bwh\b|\boh\b|parking|corral|scooter|bonfire|hastings|\bmtv\b|charging", re.I)
ALIAS_RADIUS_M = 25          # coordinate-based rename: same spot within this radius ...
ALIAS_MAX_OVERLAP_DAYS = 7   # ... and the two names were (almost) never active at the same time


def normalize_station_name(name: str) -> str:
    """Collapse the known naming variants Divvy has used for the same station."""
    n = re.sub(r"^Public Rack - ", "", name)
    n = re.sub(r"\s*\(temp\)\s*$", "", n, flags=re.I)
    n = re.sub(r"\s*-\s*charging$", "", n, flags=re.I)
    n = n.replace("*", "").replace("/", " & ")
    n = re.sub(r"\s+", " ", n).strip()
    return n.casefold()


def build_station_aliases(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Map every station name to a canonical name.

    Two names are aliases when (a) they normalize to the same string, or (b) they sit
    within ALIAS_RADIUS_M of each other and their active windows overlap by at most
    ALIAS_MAX_OVERLAP_DAYS (a rename, not two neighbouring racks). Junk/ops names are
    never merged by coords. The canonical name is the alias with the most active days.

    Returns: start_station_name, canonical_name, reason ('self' | 'name' | 'coords')
    """
    from sklearn.neighbors import BallTree

    m = meta.reset_index(drop=True).copy()
    n = len(m)
    parent = list(range(n))
    reason = ["self"] * n

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j, why):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
            reason[j] = why

    # (a) name normalization
    m["_norm"] = m["start_station_name"].map(normalize_station_name)
    for idx in m.groupby("_norm").indices.values():
        for j in idx[1:]:
            union(idx[0], j, "name")

    # (b) coordinate renames
    ok = m["lat"].notna() & m["lng"].notna() & ~m["start_station_name"].str.contains(_JUNK_RE)
    sub = m[ok]
    xy = np.radians(sub[["lat", "lng"]].to_numpy())
    nbrs = BallTree(xy, metric="haversine").query_radius(xy, r=ALIAS_RADIUS_M / 1000 / 6371.0)
    first = pd.to_datetime(sub["first_seen"]).to_numpy()
    last = pd.to_datetime(sub["last_seen"]).to_numpy()
    pos = sub.index.to_numpy()
    for a, nb in enumerate(nbrs):
        for b in nb:
            if b <= a:
                continue
            overlap = (min(last[a], last[b]) - max(first[a], first[b])) / np.timedelta64(1, "D")
            if overlap <= ALIAS_MAX_OVERLAP_DAYS:
                union(pos[a], pos[b], "coords")

    m["_root"] = [find(i) for i in range(n)]
    canon = (
        m.sort_values(["n_active_days", "in_gbfs"], ascending=False)
        .groupby("_root")["start_station_name"].first()
    )
    m["canonical_name"] = m["_root"].map(canon)
    m["reason"] = reason
    m.loc[m["canonical_name"] == m["start_station_name"], "reason"] = "self"
    out = m[["start_station_name", "canonical_name", "reason"]].sort_values(["canonical_name", "start_station_name"])
    n_groups = int((out.groupby("canonical_name").size() > 1).sum())
    logger.info(f"Aliases: {len(out)} names -> {out['canonical_name'].nunique()} canonical "
                f"({n_groups} merged groups, {(out['reason'] == 'coords').sum()} names merged by coords)")
    return out.reset_index(drop=True)


def collapse_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """
    One row per canonical station: coordinates / capacity from the canonical alias
    (fallback: any alias that has them), active window = union of the aliases.
    Output is keyed by `start_station_name` (= canonical) so it drops into the
    same merges as the raw meta.
    """
    if "canonical_name" not in meta:
        return meta
    m = meta.copy()
    m["_is_canon"] = (m["start_station_name"] == m["canonical_name"]).astype(int)
    m = m.sort_values(["canonical_name", "_is_canon", "n_active_days"], ascending=[True, False, False])
    g = m.groupby("canonical_name", sort=False)
    out = pd.DataFrame({
        "start_station_name": g["canonical_name"].first(),
        "lat":          g["lat"].first(),
        "lng":          g["lng"].first(),
        "capacity":     g["capacity"].first(),
        "is_charging":  g["is_charging"].first(),
        "in_gbfs":      g["in_gbfs"].max(),
        "first_seen":   g["first_seen"].min(),
        "last_seen":    g["last_seen"].max(),
        "n_active_days": g["n_active_days"].sum(),
        "n_aliases":    g.size(),
    }).reset_index(drop=True)
    return out
