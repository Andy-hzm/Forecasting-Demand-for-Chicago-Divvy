import os
import logging
from io import BytesIO

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Client:
    """
    Read/write Parquet files on S3 for the Divvy project.

    Layout:
        s3://{bucket}/divvy/{env}/
        ├── raw/ridership/year={YYYY}/month={MM}/data.parquet   ← trip-level
        ├── processed/overall/data.parquet                      ← city-wide daily counts
        └── processed/station/data.parquet                      ← station-level daily counts

    env defaults to 'prod'. Use env='dev' for experiments.
    """

    def __init__(self, bucket: str = None, region: str = None, env: str = "prod"):
        self.bucket = bucket or os.environ["S3_BUCKET"]
        self.prefix = f"divvy/{env}"
        self._s3 = boto3.client(
            "s3",
            region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
        )

    # ------------------------------------------------------------------
    # Raw ridership
    # ------------------------------------------------------------------

    def write_raw(self, df: pd.DataFrame, year: int, month: int) -> None:
        key = f"{self.prefix}/raw/ridership/year={year}/month={month:02d}/data.parquet"
        self._write_parquet(df, key)
        logger.info(f"Wrote {len(df)} rows → s3://{self.bucket}/{key}")

    def read_raw(self, year: int, month: int) -> pd.DataFrame:
        key = f"{self.prefix}/raw/ridership/year={year}/month={month:02d}/data.parquet"
        return self._read_parquet(key)

    def raw_exists(self, year: int, month: int) -> bool:
        key = f"{self.prefix}/raw/ridership/year={year}/month={month:02d}/data.parquet"
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    # ------------------------------------------------------------------
    # Processed data
    # ------------------------------------------------------------------

    def write_processed(self, df: pd.DataFrame, level: str) -> None:
        """level: 'overall' or 'station'"""
        key = f"{self.prefix}/processed/{level}/data.parquet"
        self._write_parquet(df, key)
        logger.info(f"Wrote {len(df)} rows → s3://{self.bucket}/{key}")

    def read_processed(self, level: str) -> pd.DataFrame:
        """level: 'overall' or 'station'"""
        key = f"{self.prefix}/processed/{level}/data.parquet"
        return self._read_parquet(key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_parquet(self, df: pd.DataFrame, key: str) -> None:
        buf = BytesIO()
        df.to_parquet(buf, index=False, compression="snappy")
        buf.seek(0)
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=buf.read())

    def _read_parquet(self, key: str) -> pd.DataFrame:
        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=key)
            return pd.read_parquet(BytesIO(obj["Body"].read()))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return pd.DataFrame()
            raise
