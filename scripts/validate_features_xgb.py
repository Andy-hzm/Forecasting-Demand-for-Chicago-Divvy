"""
Feature ablation with the notebook-10 XGBoost Poisson model.

Reproduces notebook 10's features / split / hyper-parameters as the baseline, then
adds the new feature groups (one at a time, and all together) and reports Val / Test
metrics side by side plus the top gain-importances of the new columns.

Usage:
    python scripts/validate_features_xgb.py                       # all configs
    python scripts/validate_features_xgb.py --configs baseline all
    python scripts/validate_features_xgb.py --no-canon            # keep raw station names everywhere
    python scripts/validate_features_xgb.py --out reports/feature_ablation.csv

Groups:
    holidays   is_holiday / is_major_holiday / eve / after / bridge / holiday week
    city       16 Chicago event flags + n_city_events + distance-weighted evdist_* interactions
    weather    15 extra Open-Meteo vars (temp max/mean, apparent, precip hours, sun, gusts, ...)
    sports     attendance / day-game / playoff per venue + evdist_* for the 4 venues
    spatial    knn5 / 1km-radius / city-wide lagged rides, n_active_stations
    flow       yesterday's inbound / net_flow / inbound e-bike share
    meta       dock capacity, is_charging
    calendar   Fourier day-of-year & day-of-week, days_since_start, station x weekday profile
    academic   nearest university in session / term week / finals / remote, distance-weighted
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from divvy_demand.features import (
    add_event_distance_interactions, add_fourier_calendar, add_station_dow_profile,
    add_university_session, canonicalize_stations, haversine_km,
)
from divvy_demand.infra.s3_client import S3Client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ablation")

# ----------------------------------------------------------------------------- metrics (notebook 10)

def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)) * 100

def mase(y_true, y_pred, y_train):
    return mean_absolute_error(y_true, y_pred) / np.mean(np.abs(np.diff(y_train)))

def metrics_dict(y_true, y_pred, y_tr):
    return {
        "R2":    r2_score(y_true, y_pred),
        "RMSE":  mean_squared_error(y_true, y_pred) ** 0.5,
        "MAE":   mean_absolute_error(y_true, y_pred),
        "sMAPE": smape(np.asarray(y_true), y_pred),
        "MASE":  mase(np.asarray(y_true), y_pred, np.asarray(y_tr)),
    }

# ----------------------------------------------------------------------------- data

VENUES = {"wrigley": (41.9484, -87.6553), "grf": (41.8299, -87.6338),
          "uc": (41.8807, -87.6742), "soldier": (41.8623, -87.6167)}
UNIVERSITIES = [(41.8708, -87.6505), (41.9243, -87.6555), (41.7886, -87.5987), (41.9988, -87.6580),
                (41.8345, -87.6275), (42.0565, -87.6753), (41.8717, -87.6242)]
BASE_WEATHER = ["temp_min_c", "rain_sum_mm", "snowfall_sum_cm", "wind_speed_max_kmh"]


def load_tables(s3):
    t = {}
    t["station"] = s3.read_processed("station")
    t["meta"] = s3.read_processed("station_meta")
    for k in ["weather/historical", "weather/forecast", "events/sports_daily", "events/holidays",
              "events/city", "events/academic", "station_spatial", "station_flow"]:
        t[k] = s3.read_processed(k)
    for k, v in t.items():
        if "date" in v:
            v["date"] = pd.to_datetime(v["date"])
    return t


def build_frame(t, canon: bool):
    station = t["station"][t["station"]["start_station_name"] != "Outside of Dock"].copy()
    meta = t["meta"]
    if canon:
        station = canonicalize_stations(station, meta)
        from divvy_demand.station_meta import collapse_meta
        meta = collapse_meta(meta)

    # notebook 10: stations active on > 95 % of days
    n_days = station["date"].nunique()
    active = station.groupby("start_station_name")["date"].nunique().loc[lambda s: s > int(0.95 * n_days)].index
    df = station[station["start_station_name"].isin(active)].copy()
    log.info(f"Active stations: {len(active)}  rows: {len(df):,}  canon={canon}")

    # ---- weather: ALL columns from historical; test rows get forecast later
    wcols = [c for c in t["weather/historical"].columns if c != "date"]
    df = df.merge(t["weather/historical"][["date"] + wcols], on="date", how="left")
    df = df.sort_values(["start_station_name", "date"]).reset_index(drop=True)

    # ---- sports daily (contains the 4 binary event_* flags used by the baseline)
    df = df.merge(t["events/sports_daily"], on="date", how="left")
    sports_cols = [c for c in t["events/sports_daily"].columns if c != "date"]
    df[sports_cols] = df[sports_cols].fillna(0)

    # ---- holidays + city events (date-level)
    hol = t["events/holidays"].drop(columns=["holiday_name"])
    df = df.merge(hol, on="date", how="left")
    df[[c for c in hol.columns if c != "date"]] = df[[c for c in hol.columns if c != "date"]].fillna(0)

    # ---- coords / capacity
    df = df.merge(meta[["start_station_name", "lat", "lng", "capacity", "is_charging"]], on="start_station_name", how="left")
    df = df.rename(columns={"lat": "start_lat", "lng": "start_lng"})
    df["capacity"] = df["capacity"].fillna(df["capacity"].median())
    df["is_charging"] = df["is_charging"].fillna(0)

    # ---- baseline static distances
    for key, (lat, lng) in VENUES.items():
        df[f"log_dist_{key}"] = np.log1p(haversine_km(df["start_lat"], df["start_lng"], lat, lng))
    uni = np.stack([haversine_km(df["start_lat"].values, df["start_lng"].values, la, ln) for la, ln in UNIVERSITIES])
    df["log_dist_univ"] = np.log1p(uni.min(axis=0))

    # ---- baseline calendar
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    for d in range(7):
        df[f"dow_{d}"] = (df["weekday"] == d).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # ---- baseline lags / rolls / props / shape
    grp = df.groupby("start_station_name")["total_rides"]
    for lag in [1, 2, 7, 30, 60, 90, 365]:
        df[f"y_lag{lag}"] = grp.shift(lag)
    df["rolling_avg_7"] = grp.shift(1).transform(lambda x: x.rolling(7, min_periods=1).mean())
    df["rolling_avg_30"] = grp.shift(1).transform(lambda x: x.rolling(30, min_periods=1).mean())
    df["rolling_std_7"] = grp.shift(1).transform(lambda x: x.rolling(7, min_periods=2).std())
    df["rolling_std_30"] = grp.shift(1).transform(lambda x: x.rolling(30, min_periods=2).std())
    safe = df["total_rides"].replace(0, np.nan)
    df["ebike_prop"] = df["ebike_rides"] / safe
    df["member_prop"] = df["member_rides"] / safe
    for col in ["ebike_prop", "member_prop"]:
        df[f"{col}_roll365_lag1"] = df.groupby("start_station_name")[col].transform(
            lambda x: x.shift(1).rolling(365, min_periods=30).mean())
    shape_base = [f"hour_{h:02d}_pct" for h in range(24)] + ["frac_round_trips"] + [f"mh_dist_p{q:02d}_km" for q in [10, 25, 50, 75, 90]]
    for col in shape_base:
        df[f"{col}_roll30_lag1"] = df.groupby("start_station_name")[col].transform(
            lambda x: x.shift(1).rolling(30, min_periods=7).mean())

    # ---- NEW: spatial (already lagged) + flow (lag here)
    df = df.merge(t["station_spatial"], on=["date", "start_station_name"], how="left")
    flow = t["station_flow"].sort_values(["start_station_name", "date"]).copy()
    g = flow.groupby("start_station_name")
    flow["inbound_lag1"] = g["inbound_rides"].shift(1)
    flow["net_flow_lag1"] = g["net_flow"].shift(1)
    flow["net_flow_roll7_lag1"] = g["net_flow"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    flow["inbound_ebike_share_lag1"] = (flow["inbound_ebike"] / flow["inbound_rides"].replace(0, np.nan)).groupby(flow["start_station_name"]).shift(1)
    df = df.merge(flow[["date", "start_station_name", "inbound_lag1", "net_flow_lag1", "net_flow_roll7_lag1", "inbound_ebike_share_lag1"]],
                  on=["date", "start_station_name"], how="left")

    # ---- NEW: calendar Fourier + station x dow profile
    df = add_fourier_calendar(df)
    df = add_station_dow_profile(df)

    # ---- NEW: distance-weighted event interactions (sports venues + city event locations)
    df = add_event_distance_interactions(df, VENUES, decay_km=1.5)
    city = t["events/city"]
    city_loc = city.groupby("event")[["lat", "lng"]].first()
    df = add_event_distance_interactions(df, {e: (r.lat, r.lng) for e, r in city_loc.iterrows()}, decay_km=1.5, flag_prefix="event_")
    df = df.rename(columns={"evdist_max": "evdist_max_all"})

    # ---- NEW: academic
    df = add_university_session(df, t["events/academic"], decay_km=1.0)

    # ---- station dummies, lag drop (notebook 10)
    dummies = pd.get_dummies(df["start_station_name"], prefix="stn")
    df = pd.concat([df, dummies], axis=1)
    df = df.dropna(subset=["y_lag1", "y_lag7", "y_lag365"]).reset_index(drop=True)

    groups = {
        "holidays": ["is_holiday", "is_major_holiday", "is_holiday_eve", "is_holiday_after", "is_bridge_day", "is_holiday_week"],
        "city":     [c for c in hol.columns if c.startswith("event_")] + ["n_city_events", "city_event_any"]
                    + [f"evdist_{e}" for e in city_loc.index if f"evdist_{e}" in df],
        "weather":  [c for c in wcols if c not in BASE_WEATHER],
        "sports":   [c for c in sports_cols if not c.startswith("event_")] + [f"evdist_{v}" for v in VENUES],
        "spatial":  [c for c in t["station_spatial"].columns if c not in ("date", "start_station_name")],
        "flow":     ["inbound_lag1", "net_flow_lag1", "net_flow_roll7_lag1", "inbound_ebike_share_lag1"],
        "meta":     ["capacity", "is_charging"],
        "calendar": ["doy_sin_1", "doy_cos_1", "doy_sin_2", "doy_cos_2", "dow_sin", "dow_cos", "days_since_start",
                     "total_rides_dow_profile", "total_rides_dow_profile_ratio"],
        "academic": ["univ_in_session", "univ_in_session_wt", "univ_term_week", "univ_is_finals", "univ_is_remote", "univ_dist_km"],
    }
    baseline = (["day", "month", "week_of_year", "is_weekend"] + BASE_WEATHER
                + [f"y_lag{l}" for l in [1, 2, 7, 30, 60, 90, 365]]
                + ["rolling_avg_7", "rolling_avg_30", "rolling_std_7", "rolling_std_30",
                   "ebike_prop_roll365_lag1", "member_prop_roll365_lag1"]
                + [f"dow_{d}" for d in range(7)]
                + [f"event_{v}" for v in VENUES]
                + [f"log_dist_{v}" for v in VENUES] + ["log_dist_univ"]
                + [f"{c}_roll30_lag1" for c in shape_base]
                + dummies.columns.tolist())
    return df, baseline, groups, wcols


def split(df, weather_fcst, wcols):
    max_date = df["date"].max()
    test_start = max_date - pd.DateOffset(years=1) + pd.Timedelta(days=1)
    val_start = max_date - pd.DateOffset(years=2) + pd.Timedelta(days=1)
    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start].drop(columns=wcols).merge(weather_fcst[["date"] + wcols], on="date", how="left")
    log.info(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}  (test from {test_start.date()})")
    return train, val, test


def fit_eval(name, feats, train, val, test, seed=42):
    t0 = time.time()
    model = XGBRegressor(
        n_estimators=2000, learning_rate=0.1, max_depth=10, subsample=0.8, colsample_bytree=0.8,
        objective="count:poisson", reg_lambda=5, random_state=seed, n_jobs=-1,
        early_stopping_rounds=30, eval_metric="poisson-nloglik",
    )
    model.fit(train[feats], train["total_rides"], eval_set=[(val[feats], val["total_rides"])], verbose=False)
    pv, pt = model.predict(val[feats]), model.predict(test[feats])
    row = {"config": name, "n_features": len(feats), "best_iter": model.best_iteration, "fit_s": round(time.time() - t0)}
    for split_name, y, p in [("val", val["total_rides"], pv), ("test", test["total_rides"], pt)]:
        for k, v in metrics_dict(y, p, train["total_rides"]).items():
            row[f"{split_name}_{k}"] = v
    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    return row, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="prod")
    ap.add_argument("--configs", nargs="*", default=None, help="subset of: baseline <group>... all")
    ap.add_argument("--no-canon", action="store_true", help="do not merge station aliases")
    ap.add_argument("--out", default="reports/feature_ablation.csv")
    args = ap.parse_args()

    s3 = S3Client(env=args.env)
    t = load_tables(s3)
    df, baseline, groups, wcols = build_frame(t, canon=not args.no_canon)
    train, val, test = split(df, t["weather/forecast"], wcols)

    configs = {"baseline": baseline}
    for g, cols in groups.items():
        configs[f"+{g}"] = baseline + cols
    configs["all"] = baseline + [c for cols in groups.values() for c in cols]
    if args.configs:
        want = {c if c in ("baseline", "all") else f"+{c}" for c in args.configs}
        configs = {k: v for k, v in configs.items() if k in want}

    rows, importances = [], {}
    for name, feats in configs.items():
        feats = list(dict.fromkeys(f for f in feats if f in df.columns))
        row, imp = fit_eval(name, feats, train, val, test)
        rows.append(row)
        importances[name] = imp
        log.info(f"{name:12s} feats={row['n_features']:4d} iters={row['best_iter']:4d} "
                 f"val R2={row['val_R2']:.4f} sMAPE={row['val_sMAPE']:.2f} | "
                 f"test R2={row['test_R2']:.4f} sMAPE={row['test_sMAPE']:.2f} MASE={row['test_MASE']:.4f}  ({row['fit_s']}s)")

    res = pd.DataFrame(rows).set_index("config")
    base = res.loc["baseline"] if "baseline" in res.index else None
    if base is not None:
        res["d_test_R2"] = res["test_R2"] - base["test_R2"]
        res["d_test_sMAPE"] = res["test_sMAPE"] - base["test_sMAPE"]
        res["d_test_MASE"] = res["test_MASE"] - base["test_MASE"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res.to_csv(args.out)
    pd.set_option("display.width", 200)
    print("\n" + res.round(4).to_string())

    if "all" in importances:
        new_cols = {c for cols in groups.values() for c in cols}
        imp_all = importances["all"]
        print("\nTop 25 NEW features by gain (config 'all'):")
        print(imp_all[imp_all.index.isin(new_cols)].head(25).round(1).to_string())
        print("\nTop 10 features overall (config 'all'):")
        print(imp_all.head(10).round(1).to_string())
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
