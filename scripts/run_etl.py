"""
Run the ETL to produce daily ride counts from raw trip data on S3.

Usage:
    python scripts/run_etl.py                                    # run all pipelines on prod
    python scripts/run_etl.py --env dev                          # run on dev
    python scripts/run_etl.py --level overall                    # city-wide only
    python scripts/run_etl.py --level station                    # station-level only
    python scripts/run_etl.py --level weather                    # weather only
    python scripts/run_etl.py --level weather --start 2024-01-01 --end 2024-03-31
    python scripts/run_etl.py --level sports                     # sports schedules + daily wide table
    python scripts/run_etl.py --level holidays                   # US/IL holidays + curated city events
    python scripts/run_etl.py --level academic                   # university term calendar
    python scripts/run_etl.py --level station_meta               # GBFS capacity + coords (loads raw)
    python scripts/run_etl.py --level station_flow               # inbound / net flow per station (loads raw)
    python scripts/run_etl.py --level station_spatial            # kNN / city spatial lags (needs station_meta)

Cheap, no-raw levels (run locally):  weather, sports, holidays, academic, station_spatial
Raw-scanning levels (run on EC2):    overall, station, trip, station_meta, station_flow
"""

import argparse
import logging

from dotenv import load_dotenv

from divvy_demand.etl import build_overall, build_station, build_trip, build_station_flow, build_station_spatial
from divvy_demand.weather import build_weather
from divvy_demand.sports import build_sports_events
from divvy_demand.holidays import build_holidays
from divvy_demand.academic import build_academic
from divvy_demand.station_meta import build_station_meta
from divvy_demand.infra.s3_client import S3Client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LEVELS = ["overall", "station", "trip", "weather", "sports", "holidays", "academic",
          "station_meta", "station_flow", "station_spatial"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="prod", choices=["prod", "dev"])
    parser.add_argument("--level", choices=LEVELS, help="Run one pipeline only")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (weather/trip)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (weather/trip)")
    parser.add_argument("--year", type=int, default=None, help="Shorthand for --start YYYY-01-01 --end YYYY-12-31 (trip only)")
    parser.add_argument("--knn", type=int, default=5, help="k for station_spatial")
    parser.add_argument("--radius-km", type=float, default=1.0, help="radius for station_spatial")
    parser.add_argument("--no-raw-coords", action="store_true", help="station_meta: use GBFS coords only (skip raw scan)")
    args = parser.parse_args()

    s3 = S3Client(env=args.env)
    run = lambda lvl: args.level in (None, lvl)

    if run("overall"):
        logger.info("Building overall (city-wide) counts...")
        s3.write_processed(build_overall(s3), "overall")

    if run("station"):
        logger.info("Building station-level counts...")
        s3.write_processed(build_station(s3), "station")

    if run("trip"):
        logger.info("Building trip O-D counts...")
        start = f"{args.year}-01-01" if args.year else args.start
        end = f"{args.year}-12-31" if args.year else args.end
        df = build_trip(s3, start=start, end=end)
        key = f"trip/{args.year}" if args.year else "trip/all"
        s3.write_processed(df, key)

    if run("weather"):
        logger.info("Fetching weather data...")
        kwargs = {}
        if args.start:
            kwargs["start"] = args.start
        if args.end:
            kwargs["end"] = args.end
        build_weather(s3, **kwargs)

    if run("sports"):
        logger.info("Fetching sports schedules...")
        build_sports_events(s3)

    if run("holidays"):
        logger.info("Building holiday / city-event calendar...")
        build_holidays(s3)

    if run("academic"):
        logger.info("Building academic term calendar...")
        build_academic(s3)

    if run("station_meta"):
        logger.info("Building station metadata (GBFS + coords)...")
        s3.write_processed(build_station_meta(s3, coords_from_raw=not args.no_raw_coords), "station_meta")

    if run("station_flow"):
        logger.info("Building station inbound / net flow...")
        s3.write_processed(build_station_flow(s3), "station_flow")

    if run("station_spatial"):
        logger.info("Building station spatial lags...")
        s3.write_processed(build_station_spatial(s3, k=args.knn, radius_km=args.radius_km), "station_spatial")

    logger.info("ETL complete.")


if __name__ == "__main__":
    main()
