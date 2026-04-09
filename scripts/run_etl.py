"""
Run the ETL to produce daily ride counts from raw trip data on S3.

Usage:
    python scripts/run_etl.py                                    # run all pipelines on prod
    python scripts/run_etl.py --env dev                          # run on dev
    python scripts/run_etl.py --level overall                    # city-wide only
    python scripts/run_etl.py --level station                    # station-level only
    python scripts/run_etl.py --level weather                    # weather only
    python scripts/run_etl.py --level weather --start 2024-01-01 --end 2024-03-31
"""

import argparse
import logging

from dotenv import load_dotenv

from divvy_demand.etl import build_overall, build_station, build_trip
from divvy_demand.weather import build_weather
from divvy_demand.infra.s3_client import S3Client

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="prod", choices=["prod", "dev"])
    parser.add_argument("--level", choices=["overall", "station", "trip", "weather"], help="Run one pipeline only")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (weather/trip)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (weather/trip)")
    parser.add_argument("--year", type=int, default=None, help="Shorthand for --start YYYY-01-01 --end YYYY-12-31 (trip only)")
    args = parser.parse_args()

    s3 = S3Client(env=args.env)

    if args.level in (None, "overall"):
        logger.info("Building overall (city-wide) counts...")
        df = build_overall(s3)
        s3.write_processed(df, "overall")

    if args.level in (None, "station"):
        logger.info("Building station-level counts...")
        df = build_station(s3)
        s3.write_processed(df, "station")

    if args.level in (None, "trip"):
        logger.info("Building trip O-D counts...")
        start = f"{args.year}-01-01" if args.year else args.start
        end = f"{args.year}-12-31" if args.year else args.end
        df = build_trip(s3, start=start, end=end)
        key = f"trip/{args.year}" if args.year else "trip/all"
        s3.write_processed(df, key)

    if args.level in (None, "weather"):
        logger.info("Fetching weather data...")
        kwargs = {}
        if args.start:
            kwargs["start"] = args.start
        if args.end:
            kwargs["end"] = args.end
        build_weather(s3, **kwargs)

    logger.info("ETL complete.")


if __name__ == "__main__":
    main()
