"""
Model-side feature helpers — pure pandas, no S3.

Everything here is leakage-safe by construction: anything derived from the
target is shifted so a row for date t only sees t-1 and earlier. Use these in
the forecast notebooks after the station / weather / events merges.

    from divvy_demand.features import (
        canonicalize_stations, add_fourier_calendar, add_station_dow_profile,
        add_next_day_weather, add_event_distance_interactions, add_university_session,
        haversine_km,
    )
"""
import re
import numpy as np
import pandas as pd


def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, [lat1, lng1, lat2, lng2])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Station aliases
# ---------------------------------------------------------------------------

_SUM_RE  = re.compile(r"_rides$|^net_flow$|^inbound_|^outbound_")
_WMEAN_RE = re.compile(r"_pct$|^frac_|^mh_dist_")


def canonicalize_stations(
    df: pd.DataFrame,
    meta: pd.DataFrame,
    name_col: str = "start_station_name",
    date_col: str = "date",
    weight_col: str = "total_rides",
) -> pd.DataFrame:
    """
    Rename stations to their canonical name (station_meta.canonical_name) and merge
    rows that now collide on (date, station):
      count columns (*_rides, net_flow, inbound_*/outbound_*)   -> sum
      shape columns (hour_*_pct, frac_*, mh_dist_*)              -> mean weighted by total_rides
      everything else                                            -> value from the busiest alias
    Fixes histories split by renames ('Lake Shore Dr' -> 'DuSable Lake Shore Dr',
    'X' -> 'Public Rack - X', 'X' vs 'X*', '/' vs '&').
    """
    alias = dict(zip(meta["start_station_name"], meta["canonical_name"]))
    out = df.copy()
    out[name_col] = out[name_col].map(alias).fillna(out[name_col])

    keys = [date_col, name_col] if date_col in out else [name_col]
    dup = out.duplicated(keys, keep=False)
    if not dup.any():
        return out

    clash, clean = out[dup].copy(), out[~dup]
    w = clash[weight_col].astype(float) if weight_col in clash else pd.Series(1.0, index=clash.index)
    agg, wmean_cols = {}, []
    for c in clash.columns:
        if c in keys:
            continue
        if _SUM_RE.search(c):
            agg[c] = "sum"
        elif _WMEAN_RE.search(c) and pd.api.types.is_numeric_dtype(clash[c]):
            wmean_cols.append(c)
        else:
            agg[c] = "first"
    # busiest alias first so "first" picks it
    clash = clash.assign(_w=w).sort_values(keys + ["_w"], ascending=[True] * len(keys) + [False])
    merged = clash.groupby(keys, sort=False).agg(agg)
    if wmean_cols:
        gk = [clash[k] for k in keys]
        for c in wmean_cols:
            # NaN-aware: a row with no value for this column contributes neither value nor weight
            w_c = clash["_w"].where(clash[c].notna(), 0.0)
            num = (clash[c].fillna(0) * w_c).groupby(gk, sort=False).sum()
            den = w_c.groupby(gk, sort=False).sum().replace(0, np.nan)
            merged[c] = num / den
    merged = merged.reset_index()[out.columns]
    return pd.concat([clean, merged], ignore_index=True).sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def add_fourier_calendar(df: pd.DataFrame, date_col: str = "date", harmonics: int = 2) -> pd.DataFrame:
    """
    Smooth seasonality encodings — no Dec→Jan cliff like integer month / week_of_year.
      doy_sin_k / doy_cos_k   k-th annual harmonic (k = 1..harmonics)
      dow_sin / dow_cos       weekly cycle
      days_since_start        linear trend (days since min date in df)
    """
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    doy = d.dt.dayofyear.to_numpy()
    year_len = np.where(d.dt.is_leap_year, 366.0, 365.0)
    for k in range(1, harmonics + 1):
        ang = 2 * np.pi * k * doy / year_len
        out[f"doy_sin_{k}"] = np.sin(ang)
        out[f"doy_cos_{k}"] = np.cos(ang)
    ang = 2 * np.pi * d.dt.weekday.to_numpy() / 7.0
    out["dow_sin"] = np.sin(ang)
    out["dow_cos"] = np.cos(ang)
    out["days_since_start"] = (d - d.min()).dt.days
    return out


# ---------------------------------------------------------------------------
# Station x weekday profile (target encoding, shifted)
# ---------------------------------------------------------------------------

def add_station_dow_profile(
    df: pd.DataFrame,
    target: str = "total_rides",
    station_col: str = "start_station_name",
    date_col: str = "date",
    window_weeks: int = 52,
    min_weeks: int = 4,
) -> pd.DataFrame:
    """
    Rolling mean of `target` for the same station on the same weekday, over the
    previous `window_weeks` occurrences (shifted by one occurrence → no leakage).
      {target}_dow_profile        rolling mean
      {target}_dow_profile_ratio  dow_profile / station rolling 7-day mean (shape, not level)
    Expects one row per station-date; sorts internally.
    """
    out = df.sort_values([station_col, date_col]).copy()
    out["_dow"] = pd.to_datetime(out[date_col]).dt.weekday
    grp = out.groupby([station_col, "_dow"], sort=False)[target]
    out[f"{target}_dow_profile"] = grp.transform(
        lambda s: s.shift(1).rolling(window_weeks, min_periods=min_weeks).mean()
    )
    level = out.groupby(station_col, sort=False)[target].transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean()
    )
    out[f"{target}_dow_profile_ratio"] = out[f"{target}_dow_profile"] / level.replace(0, np.nan)
    return out.drop(columns="_dow")


# ---------------------------------------------------------------------------
# Weather: tomorrow's forecast as context for today
# ---------------------------------------------------------------------------

def add_next_day_weather(
    df: pd.DataFrame,
    weather: pd.DataFrame,
    cols: list,
    date_col: str = "date",
    suffix: str = "_next",
) -> pd.DataFrame:
    """
    Merge weather for date t+1 onto row t (use the *forecast* table so it is
    available at inference time). People check tomorrow's forecast before
    deciding to bike-commute today.
    """
    w = weather[[date_col] + cols].copy()
    w[date_col] = pd.to_datetime(w[date_col]) - pd.Timedelta(days=1)
    w = w.rename(columns={c: f"{c}{suffix}" for c in cols})
    return df.merge(w, on=date_col, how="left")


# ---------------------------------------------------------------------------
# Events x distance interactions
# ---------------------------------------------------------------------------

def add_event_distance_interactions(
    df: pd.DataFrame,
    venues: dict,
    lat_col: str = "start_lat",
    lng_col: str = "start_lng",
    decay_km: float = 1.5,
    flag_prefix: str = "event_",
) -> pd.DataFrame:
    """
    For each venue slug in `venues` ({slug: (lat, lng)}) with a daily flag column
    `{flag_prefix}{slug}` already in df, add
      evdist_{slug} = flag * exp(-dist_km / decay_km)
    so the model gets the "game nearby today" signal explicitly instead of having
    to learn flag x distance on its own. Also adds `evdist_max` across venues.
    Works for sports venues and city-event locations alike.
    """
    out = df.copy()
    cols = []
    for slug, (lat, lng) in venues.items():
        flag = f"{flag_prefix}{slug}"
        if flag not in out:
            continue
        d = haversine_km(out[lat_col], out[lng_col], lat, lng)
        out[f"evdist_{slug}"] = out[flag].fillna(0) * np.exp(-d / decay_km)
        cols.append(f"evdist_{slug}")
    if cols:
        out["evdist_max"] = out[cols].max(axis=1)
    return out


# ---------------------------------------------------------------------------
# University term in session
# ---------------------------------------------------------------------------

def add_university_session(
    df: pd.DataFrame,
    terms: pd.DataFrame,
    lat_col: str = "start_lat",
    lng_col: str = "start_lng",
    date_col: str = "date",
    decay_km: float = 1.0,
) -> pd.DataFrame:
    """
    `terms` is processed/events/academic (school, start, end, lat, lng, is_remote).
    Adds
      univ_in_session        1 if the station's nearest school is in session on date
      univ_in_session_wt     in_session * exp(-dist_to_nearest_school / decay_km)
      univ_term_week         weeks since term start (0 = first week, NaN if not in session)
      univ_is_finals         1 if within the last 7 days of a term
    """
    out = df.copy()
    schools = terms.groupby("school")[["lat", "lng"]].first()
    dists = np.stack([
        haversine_km(out[lat_col], out[lng_col], r.lat, r.lng) for r in schools.itertuples()
    ], axis=1)
    nearest = schools.index.to_numpy()[dists.argmin(axis=1)]
    out["_school"] = nearest
    out["univ_dist_km"] = dists.min(axis=1)

    t = terms.copy()
    t["start"] = pd.to_datetime(t["start"])
    t["end"] = pd.to_datetime(t["end"])
    # Expand terms to one row per (school, date)
    days = []
    for r in t.itertuples():
        rng = pd.date_range(r.start, r.end, freq="D")
        days.append(pd.DataFrame({
            "_school": r.school,
            date_col: rng,
            "univ_term_week": (rng - r.start).days // 7,
            "univ_is_finals": ((r.end - rng).days < 7).astype(int),
            "univ_is_remote": int(getattr(r, "is_remote", 0) or 0),
        }))
    cal = pd.concat(days, ignore_index=True).drop_duplicates(["_school", date_col])

    out[date_col] = pd.to_datetime(out[date_col])
    out = out.merge(cal, on=["_school", date_col], how="left")
    out["univ_in_session"] = out["univ_term_week"].notna().astype(int)
    out["univ_is_finals"] = out["univ_is_finals"].fillna(0).astype(int)
    out["univ_is_remote"] = out["univ_is_remote"].fillna(0).astype(int)
    out["univ_in_session_wt"] = out["univ_in_session"] * np.exp(-out["univ_dist_km"] / decay_km)
    return out.drop(columns="_school")
