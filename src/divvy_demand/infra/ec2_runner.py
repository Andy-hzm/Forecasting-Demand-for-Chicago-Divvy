import os
import time
import logging

import boto3

logger = logging.getLogger(__name__)


class EC2Runner:
    """
    Run shell commands on an EC2 instance via AWS SSM. No SSH needed.

    The instance must have:
        - AmazonSSMManagedInstanceCore IAM policy
        - SSM Agent running (pre-installed on Amazon Linux 2023)

    Usage:
        runner = EC2Runner(instance_id='i-xxxx')
        runner.start()
        runner.setup()                         # first time only
        runner.run("python scripts/ingest.py")
        runner.stop()
    """

    REPO_URL = "https://github.com/Andy-hzm/Forecasting-Demand-for-Chicago-Divvy.git"
    REPO_DIR = "/home/ec2-user/divvy"

    def __init__(self, instance_id: str = None, region: str = None):
        self.instance_id = instance_id or os.environ["EC2_INSTANCE_ID"]
        region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._ec2 = boto3.client("ec2", region_name=region)
        self._ssm = boto3.client("ssm", region_name=region)

    def start(self) -> None:
        logger.info(f"Starting {self.instance_id}...")
        self._ec2.start_instances(InstanceIds=[self.instance_id])
        self._ec2.get_waiter("instance_running").wait(InstanceIds=[self.instance_id])
        logger.info("Instance running — waiting for SSM agent...")
        self._wait_for_ssm()
        logger.info("Ready")

    def stop(self) -> None:
        self._ec2.stop_instances(InstanceIds=[self.instance_id])
        logger.info(f"Stop signal sent to {self.instance_id}")

    def setup(self) -> None:
        """First-time setup: install Python, clone repo, install dependencies."""
        command = (
            "sudo dnf install -y python3.11 python3.11-pip git && "
            f"(git clone {self.REPO_URL} {self.REPO_DIR} 2>/dev/null || git -C {self.REPO_DIR} pull) && "
            f"python3.11 -m venv --clear {self.REPO_DIR}/.venv && "
            f"{self.REPO_DIR}/.venv/bin/pip install -q --upgrade pip setuptools wheel && "
            f"{self.REPO_DIR}/.venv/bin/pip install -q -e {self.REPO_DIR}"
        )
        logger.info("Running setup...")
        self.run(command)
        logger.info("Setup complete")

    def run(self, commands: list | str, timeout: int = 3600) -> None:
        if isinstance(commands, str):
            commands = [commands]
        response = self._ssm.send_command(
            InstanceIds=[self.instance_id],
            DocumentName="AWS-RunShellScript",
            Parameters={"commands": commands},
            TimeoutSeconds=timeout,
        )
        self._stream_output(response["Command"]["CommandId"], timeout)

    def _stream_output(self, command_id: str, timeout: int) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(5)
            result = self._ssm.get_command_invocation(
                CommandId=command_id,
                InstanceId=self.instance_id,
            )
            status = result["Status"]
            for line in result.get("StandardOutputContent", "").splitlines():
                logger.info(f"[EC2] {line}")
            for line in result.get("StandardErrorContent", "").splitlines():
                logger.warning(f"[EC2:err] {line}")
            if status in ("Success", "Failed", "Cancelled", "TimedOut"):
                if status != "Success":
                    raise RuntimeError(f"SSM command {status}")
                return
        raise TimeoutError(f"Command did not finish within {timeout}s")

    def _wait_for_ssm(self, retries: int = 20, pause: int = 10) -> None:
        for _ in range(retries):
            try:
                info = self._ssm.describe_instance_information(
                    Filters=[{"Key": "InstanceIds", "Values": [self.instance_id]}]
                )
                if info["InstanceInformationList"]:
                    return
            except Exception:
                pass
            time.sleep(pause)
        raise TimeoutError("SSM agent not reachable")
