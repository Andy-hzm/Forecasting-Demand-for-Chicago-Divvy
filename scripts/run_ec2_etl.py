"""
Run the ETL on EC2 via SSM. Code is synced via S3 (no GitHub/SSH needed).

Usage:
    python scripts/run_ec2_etl.py                        # run both pipelines, prod
    python scripts/run_ec2_etl.py --env dev              # run on dev
    python scripts/run_ec2_etl.py --level overall        # city-wide only
    python scripts/run_ec2_etl.py --level station        # station-level only
    python scripts/run_ec2_etl.py --setup                # first-time instance setup
    python scripts/run_ec2_etl.py --no-stop              # keep instance running after job
"""

import argparse
import logging
import os
import tempfile
import zipfile

import boto3
from dotenv import load_dotenv

from divvy_demand.infra.ec2_runner import EC2Runner

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_S3_KEY = "divvy/code/code.zip"


def upload_code(bucket: str) -> None:
    """Zip local src/ and scripts/ and upload to S3."""
    logger.info("Uploading code to S3...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for base in ["src", "scripts"]:
                for dirpath, _, filenames in os.walk(os.path.join(ROOT, base)):
                    for fname in filenames:
                        if fname.endswith(".pyc"):
                            continue
                        fpath = os.path.join(dirpath, fname)
                        zf.write(fpath, os.path.relpath(fpath, ROOT))
            zf.write(os.path.join(ROOT, "pyproject.toml"), "pyproject.toml")
        tmp_path = tmp.name

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    s3.upload_file(tmp_path, bucket, CODE_S3_KEY)
    os.unlink(tmp_path)
    logger.info(f"Code uploaded to s3://{bucket}/{CODE_S3_KEY}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="prod", choices=["prod", "dev"])
    parser.add_argument("--level", choices=["overall", "station"])
    parser.add_argument("--setup", action="store_true", help="Run first-time setup on the instance")
    parser.add_argument("--no-stop", action="store_true", help="Keep instance running after job")
    args = parser.parse_args()

    bucket = os.environ["S3_BUCKET"]
    runner = EC2Runner()

    runner.start()

    if args.setup:
        runner.setup()

    # Sync code from S3 to EC2
    upload_code(bucket)
    runner.run(
        f"aws s3 cp s3://{bucket}/{CODE_S3_KEY} /tmp/code.zip && "
        f"unzip -o /tmp/code.zip -d {runner.REPO_DIR} && "
        f"{runner.REPO_DIR}/.venv/bin/pip install -q {runner.REPO_DIR}"
    )

    # Pass credentials as env vars, run ETL
    env_vars = (
        f"S3_BUCKET='{os.environ['S3_BUCKET']}' "
        f"AWS_ACCESS_KEY_ID='{os.environ['AWS_ACCESS_KEY_ID']}' "
        f"AWS_SECRET_ACCESS_KEY='{os.environ['AWS_SECRET_ACCESS_KEY']}' "
        f"AWS_REGION='{os.environ.get('AWS_REGION', 'us-east-1')}' "
    )
    cmd = (
        f"cd {runner.REPO_DIR} && "
        f"{env_vars}"
        f"{runner.REPO_DIR}/.venv/bin/python scripts/run_etl.py --env {args.env}"
    )
    if args.level:
        cmd += f" --level {args.level}"

    runner.run(cmd)

    if not args.no_stop:
        runner.stop()


if __name__ == "__main__":
    main()
