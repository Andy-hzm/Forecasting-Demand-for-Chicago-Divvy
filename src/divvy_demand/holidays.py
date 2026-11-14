"""
Holiday & city-event calendar ETL — no external API, built from the `holidays`
package plus the curated list in calendars/chicago_events.py.

processed/events/holidays   one row per calendar date
    date
    is_holiday            1 on any US federal or Illinois state holiday (incl. Pulaski Day)
    is_major_holiday      1 on the big ones where the city empties out (see MAJOR_HOLIDAYS)
    holiday_name          str or ''
    is_holiday_eve        1 on the day before a major holiday
    is_holiday_after      1 on the day after a major holiday
    is_bridge_day         1 on a weekday squeezed between a major holiday and a weekend
    is_holiday_week       1 Dec 24 – Jan 1 (Christmas/New Year lull)
    n_city_events         number of curated city events that day
    city_event_any        1 if n_city_events > 0
    event_{slug}          1 per event slug (chicago_marathon, lollapalooza, ...)

processed/events/city       long table, one row per event-day
    date, event, location, lat, lng, note
"""
import logging

import holidays as holidays_lib
import pandas as pd

from divvy_demand.infra.s3_client import S3Client
from divvy_demand.calendars.chicago_events import CHICAGO_EVENTS

logger = logging.getLogger(__name__)

START = "2020-01-01"
END   = "2026-12-31"

MAJOR_HOLIDAYS = {
    "New Year's Day", "Martin Luther King Jr. Day", "Washington's Birthday",
    "Memorial Day", "Juneteenth National Independence Day", "Independence Day",
    "Labor Day", "Thanksgiving Day", "Christmas Day",
}


def _base_name(name: str) -> str:
    # holidays returns e.g. "Independence Day (observed)" — strip the suffix
    return name.split(" (")[0]


def build_holiday_calendar(start: str = START, end: str = END) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    years = range(dates[0].year, dates[-1].year + 1)
    us_il = holidays_lib.US(subdiv="IL", years=years, observed=True)

    df = pd.DataFrame({"date": dates})
    df["holiday_name"] = df["date"].map(lambda d: us_il.get(d.date(), ""))
    df["is_holiday"] = (df["holiday_name"] != "").astype(int)
    df["is_major_holiday"] = df["holiday_name"].map(
        lambda n: int(_base_name(n) in MAJOR_HOLIDAYS) if n else 0
    )

    major = df["is_major_holiday"].to_numpy().astype(bool)
    df["is_holiday_eve"]   = pd.Series(major).shift(-1, fill_value=False).astype(int).to_numpy()
    df["is_holiday_after"] = pd.Series(major).shift(1,  fill_value=False).astype(int).to_numpy()

    # Bridge day: weekday with a major holiday on one side and the weekend on the other
    wd = df["date"].dt.weekday.to_numpy()
    prev_major = pd.Series(major).shift(1,  fill_value=False).to_numpy()
    next_major = pd.Series(major).shift(-1, fill_value=False).to_numpy()
    df["is_bridge_day"] = (
        ((wd == 4) & prev_major) | ((wd == 0) & next_major)
    ).astype(int)

    md = df["date"].dt.month * 100 + df["date"].dt.day
    df["is_holiday_week"] = ((md >= 1224) | (md <= 101)).astype(int)
    return df


def build_city_events() -> pd.DataFrame:
    if not CHICAGO_EVENTS:
        logger.warning("CHICAGO_EVENTS is empty — calendars/chicago_events.py not populated yet")
        return pd.DataFrame(columns=["date", "event", "location", "lat", "lng", "note"])
    ev = pd.DataFrame(CHICAGO_EVENTS)
    ev["date"] = pd.to_datetime(ev["date"])
    ev = ev.drop_duplicates(["date", "event"]).sort_values(["date", "event"]).reset_index(drop=True)
    return ev[["date", "event", "location", "lat", "lng", "note"]]


def build_holidays(s3: S3Client, start: str = START, end: str = END) -> pd.DataFrame:
    cal = build_holiday_calendar(start, end)
    city = build_city_events()

    if len(city):
        wide = (
            city.assign(flag=1)
            .pivot_table(index="date", columns="event", values="flag", aggfunc="max", fill_value=0)
        )
        wide.columns = [f"event_{c}" for c in wide.columns]
        wide["n_city_events"] = wide.sum(axis=1)
        cal = cal.merge(wide.reset_index(), on="date", how="left")
        event_cols = list(wide.columns)
        cal[event_cols] = cal[event_cols].fillna(0).astype(int)
    else:
        cal["n_city_events"] = 0
    cal["city_event_any"] = (cal["n_city_events"] > 0).astype(int)

    s3.write_processed(cal, "events/holidays")
    s3.write_processed(city, "events/city")
    logger.info(
        f"Holidays: {len(cal)} days, {cal['is_holiday'].sum()} holidays "
        f"({cal['is_major_holiday'].sum()} major), {len(city)} city event-days"
    )
    return cal
