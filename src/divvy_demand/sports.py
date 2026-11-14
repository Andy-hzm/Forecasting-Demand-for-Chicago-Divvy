"""
Sports schedule ETL — Cubs, White Sox, Bears, Bulls, Blackhawks home games.

Two tables:

processed/events/sports        one row per home game
    date           datetime
    venue          str         # 'Wrigley Field', 'United Center', ...
    venue_lat      float
    venue_lng      float
    sport          str         # 'MLB' | 'NFL' | 'NBA' | 'NHL'
    team           str         # 'Cubs' | 'Bears' | ...
    opponent       str
    game_type      str         # 'REG' | 'POST' | 'PRE'
    attendance     float       # nullable (MLB only; postseason via MLB Stats API boxscore)
    start_hour     float       # local start hour, nullable (MLB day/night only → 13.0 / 19.0)
    is_day_game    float       # 1/0, nullable when start time unknown

processed/events/sports_daily  one row per date (full calendar, zeros explicit), per-venue wide:
    event_{slug}, n_games_{slug}, playoff_{slug}, daygame_{slug}, attendance_{slug}
    plus event_any, n_games_total.
    slug ∈ {wrigley, grf, uc, soldier}
"""
import logging
from typing import Iterable, List

import pandas as pd
import requests

from divvy_demand.infra.s3_client import S3Client

logger = logging.getLogger(__name__)

VENUES = {
    "Wrigley Field":         {"lat": 41.9484, "lng": -87.6553, "slug": "wrigley"},
    "Guaranteed Rate Field": {"lat": 41.8299, "lng": -87.6338, "slug": "grf"},
    "United Center":         {"lat": 41.8807, "lng": -87.6742, "slug": "uc"},
    "Soldier Field":         {"lat": 41.8623, "lng": -87.6167, "slug": "soldier"},
}
VENUE_SLUGS = [v["slug"] for v in VENUES.values()]

OUTPUT_COLS = [
    "date", "venue", "venue_lat", "venue_lng",
    "sport", "team", "opponent", "game_type", "attendance",
    "start_hour", "is_day_game",
]

# A game starting before this local hour counts as a day game
DAY_GAME_CUTOFF_HOUR = 17


def _finish(df: pd.DataFrame, venue: str, sport: str, team: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "date":        df["date"].dt.normalize(),
        "venue":       venue,
        "venue_lat":   VENUES[venue]["lat"],
        "venue_lng":   VENUES[venue]["lng"],
        "sport":       sport,
        "team":        team,
        "opponent":    df["opponent"],
        "game_type":   df["game_type"],
        "attendance":  df["attendance"].astype("Float64") if "attendance" in df else pd.NA,
        "start_hour":  df["start_hour"].astype("Float64") if "start_hour" in df else pd.NA,
    })
    out["is_day_game"] = (out["start_hour"] < DAY_GAME_CUTOFF_HOUR).astype("Float64")
    out.loc[out["start_hour"].isna(), "is_day_game"] = pd.NA
    logger.info(f"  {sport} {team}: {len(out)} home games ({out['date'].min().date()} → {out['date'].max().date()})")
    return out[OUTPUT_COLS]


# ---------------------------------------------------------------------------
# MLB — Cubs / White Sox via pybaseball.schedule_and_record
# ---------------------------------------------------------------------------

def _patch_pybaseball():
    """Patch pybaseball's make_numeric — it crashes on 'Unknown' attendance under CoW + Arrow strings."""
    import pybaseball.team_results as tr

    def _safe(data):
        for col in ["R", "RA", "Inn", "Rank", "Attendance"]:
            # Attendance arrives as "40,072" — strip thousands separators before coercing
            data[col] = pd.to_numeric(
                data[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
        return data
    tr.make_numeric = _safe


def fetch_mlb_home_games(team_code: str, team_name: str, venue: str, seasons: Iterable[int]) -> pd.DataFrame:
    from pybaseball import schedule_and_record
    _patch_pybaseball()

    frames = []
    for yr in seasons:
        sched = schedule_and_record(yr, team_code)
        sched = sched[sched["Home_Away"] == "Home"].copy()
        sched["season"] = yr
        frames.append(sched)
    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(
        df["Date"].str.replace(r" \(\d+\)", "", regex=True) + " " + df["season"].astype(str),
        format="%A, %b %d %Y",
        errors="coerce",
    )
    df = df.dropna(subset=["date"])

    # Baseball-Reference only gives D/N, not a clock time — map to representative hours
    df["start_hour"] = df["D/N"].map({"D": 13.0, "N": 19.0})
    df["opponent"]   = df["Opp"]
    df["game_type"]  = "REG"
    df["attendance"] = df["Attendance"]
    return _finish(df, venue, "MLB", team_name)


# ---------------------------------------------------------------------------
# MLB postseason — MLB Stats API (pybaseball's schedule_and_record is regular season only)
# ---------------------------------------------------------------------------

MLB_TEAM_IDS = {"Cubs": 112, "White Sox": 145}
MLB_POSTSEASON_TYPES = "F,D,L,W"   # wild card, division series, LCS, world series


def _mlb_boxscore_attendance(game_pk: int) -> float:
    try:
        resp = requests.get(f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=30)
        resp.raise_for_status()
        for item in resp.json().get("info", []):
            if item.get("label") in ("Att", "Attendance"):
                return float(item["value"].replace(",", "").rstrip("."))
    except Exception as e:  # attendance is nice-to-have; never fail the ETL over it
        logger.warning(f"  boxscore {game_pk}: {e}")
    return float("nan")


def fetch_mlb_postseason_home_games(team_name: str, venue: str, seasons: Iterable[int]) -> pd.DataFrame:
    team_id = MLB_TEAM_IDS[team_name]
    rows = []
    for yr in seasons:
        url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&teamId={team_id}"
               f"&season={yr}&gameType={MLB_POSTSEASON_TYPES}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        for day in resp.json().get("dates", []):
            for g in day["games"]:
                if g["teams"]["home"]["team"]["id"] != team_id:
                    continue
                if g["status"].get("detailedState") not in ("Final", "Completed Early", "Game Over"):
                    continue
                rows.append({
                    "date":       pd.to_datetime(g["officialDate"]),
                    "opponent":   g["teams"]["away"]["team"]["name"],
                    "game_type":  "POST",
                    "start_hour": 13.0 if g.get("dayNight") == "day" else 19.0,
                    "attendance": _mlb_boxscore_attendance(g["gamePk"]),
                })
    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLS)
    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "opponent", "start_hour"])
    return _finish(df, venue, "MLB", f"{team_name}")


# ---------------------------------------------------------------------------
# NFL — Bears via nfl_data_py
# ---------------------------------------------------------------------------

def fetch_bears_home_games(seasons: Iterable[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    sched = nfl.import_schedules(list(seasons))
    home  = sched[sched["home_team"] == "CHI"].copy()
    home["date"] = pd.to_datetime(home["gameday"])

    # nflverse gametime is US/Eastern — shift to Chicago local
    t = pd.to_datetime(home["gametime"], format="%H:%M", errors="coerce")
    home["start_hour"] = (t.dt.hour + t.dt.minute / 60.0) - 1.0
    home["opponent"]   = home["away_team"]
    home["attendance"] = pd.NA
    return _finish(home, "Soldier Field", "NFL", "Bears")


# ---------------------------------------------------------------------------
# NBA — Bulls via nba_api
# ---------------------------------------------------------------------------

def fetch_bulls_home_games(date_from: str = "01/01/2021") -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamefinder

    BULLS_TEAM_ID = 1610612741
    games = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=BULLS_TEAM_ID,
        date_from_nullable=date_from,
    ).get_data_frames()[0]

    home = games[games["MATCHUP"].str.contains(r"vs\.", regex=True)].copy()
    home["date"] = pd.to_datetime(home["GAME_DATE"])
    home["opponent"] = home["MATCHUP"].str.split(" vs. ").str[1]

    # SEASON_ID prefix: 1=preseason, 2=regular, 3=allstar, 4=playoff, 5=playin
    type_map = {"1": "PRE", "2": "REG", "4": "POST", "5": "POST"}
    home["game_type"] = home["SEASON_ID"].astype(str).str[0].map(type_map).fillna("REG")
    home["attendance"] = pd.NA
    home["start_hour"] = pd.NA   # LeagueGameFinder has no tip-off time; Bulls home games are ~all evening
    return _finish(home, "United Center", "NBA", "Bulls")


# ---------------------------------------------------------------------------
# NHL — Blackhawks via NHL public API
# ---------------------------------------------------------------------------

NHL_TYPE_MAP = {1: "PRE", 2: "REG", 3: "POST"}


def fetch_hawks_home_games(seasons: Iterable[str]) -> pd.DataFrame:
    """seasons: iterable of season strings like '20232024'."""
    rows = []
    for s in seasons:
        url = f"https://api-web.nhle.com/v1/club-schedule-season/CHI/{s}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        for g in resp.json().get("games", []):
            if g["homeTeam"]["abbrev"] != "CHI":
                continue
            start_utc = pd.to_datetime(g.get("startTimeUTC"), utc=True, errors="coerce")
            local = start_utc.tz_convert("America/Chicago") if pd.notna(start_utc) else pd.NaT
            rows.append({
                "date":       pd.to_datetime(g["gameDate"]),
                "opponent":   g["awayTeam"]["abbrev"],
                "game_type":  NHL_TYPE_MAP.get(g.get("gameType"), "REG"),
                "start_hour": (local.hour + local.minute / 60.0) if pd.notna(local) else pd.NA,
            })
    df = pd.DataFrame(rows)
    df["attendance"] = pd.NA
    return _finish(df, "United Center", "NHL", "Blackhawks")


# ---------------------------------------------------------------------------
# Daily wide table
# ---------------------------------------------------------------------------

def build_sports_daily(events: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Collapse game-level rows into one row per calendar date with per-venue columns.
    Dates with no game are present with zeros so a left-merge needs no fillna.
    """
    ev = events.copy()
    ev["slug"]       = ev["venue"].map({k: v["slug"] for k, v in VENUES.items()})
    ev["is_playoff"] = (ev["game_type"] == "POST").astype(int)
    ev["is_day"]     = ev["is_day_game"].fillna(0).astype(int)
    ev["att"]        = ev["attendance"].fillna(0).astype(float)

    agg = ev.groupby(["date", "slug"]).agg(
        n_games=("team", "size"),
        playoff=("is_playoff", "max"),
        daygame=("is_day", "max"),
        attendance=("att", "sum"),
    )
    wide = agg.unstack("slug")
    wide.columns = [f"{stat}_{slug}" for stat, slug in wide.columns]

    cal = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    daily = cal.merge(wide.reset_index(), on="date", how="left").fillna(0)

    for slug in VENUE_SLUGS:
        for stat in ["n_games", "playoff", "daygame", "attendance"]:
            col = f"{stat}_{slug}"
            if col not in daily:
                daily[col] = 0
        daily[f"event_{slug}"] = (daily[f"n_games_{slug}"] > 0).astype(int)

    daily["n_games_total"] = daily[[f"n_games_{s}" for s in VENUE_SLUGS]].sum(axis=1)
    daily["event_any"]     = (daily["n_games_total"] > 0).astype(int)

    int_cols = [c for c in daily.columns if c.split("_")[0] in ("n", "playoff", "daygame", "event")]
    daily[int_cols] = daily[int_cols].astype(int)

    ordered = ["date", "event_any", "n_games_total"] + [
        f"{stat}_{slug}" for slug in VENUE_SLUGS
        for stat in ["event", "n_games", "playoff", "daygame", "attendance"]
    ]
    return daily[ordered]


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

DEFAULT_MLB_NFL_NBA_SEASONS: List[int] = list(range(2021, 2026))
DEFAULT_NHL_SEASONS: List[str] = [
    "20202021", "20212022", "20222023", "20232024", "20242025", "20252026",
]
DAILY_START = "2020-01-01"
DAILY_END   = "2026-12-31"


def build_sports_events(
    s3: S3Client,
    mlb_nfl_nba_seasons: Iterable[int] = None,
    nhl_seasons: Iterable[str] = None,
) -> None:
    """Fetch all five teams, concat, dedupe, write events/sports and events/sports_daily."""
    mlb_nfl_nba_seasons = mlb_nfl_nba_seasons or DEFAULT_MLB_NFL_NBA_SEASONS
    nhl_seasons         = nhl_seasons         or DEFAULT_NHL_SEASONS

    logger.info("Fetching sports schedules...")
    parts = [
        fetch_mlb_home_games("CHC", "Cubs",      "Wrigley Field",         mlb_nfl_nba_seasons),
        fetch_mlb_home_games("CHW", "White Sox", "Guaranteed Rate Field", mlb_nfl_nba_seasons),
        fetch_mlb_postseason_home_games("Cubs",      "Wrigley Field",         mlb_nfl_nba_seasons),
        fetch_mlb_postseason_home_games("White Sox", "Guaranteed Rate Field", mlb_nfl_nba_seasons),
        fetch_bears_home_games(mlb_nfl_nba_seasons),
        fetch_bulls_home_games(date_from=f"01/01/{min(mlb_nfl_nba_seasons)}"),
        fetch_hawks_home_games(nhl_seasons),
    ]
    events = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["date", "venue", "team", "opponent"])
        .sort_values(["date", "venue", "team"])
        .reset_index(drop=True)
    )
    s3.write_processed(events, "events/sports")
    logger.info(f"Wrote {len(events):,} game rows  ({events['date'].min().date()} → {events['date'].max().date()})")

    daily = build_sports_daily(events, DAILY_START, DAILY_END)
    s3.write_processed(daily, "events/sports_daily")
    logger.info(f"Wrote {len(daily):,} daily rows, {daily['event_any'].sum()} event days")
