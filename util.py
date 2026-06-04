import os
import sys
import pandas as pd
import subprocess
import logging
from pathlib import Path
from urllib.parse import urlparse, unquote
from PIL import Image
from tqdm import tqdm
import ast

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    _s3_region = (
        os.environ.get("S3_REGION_NAME")
        or os.environ.get("AWS_DEFAULT_REGION")
        or "ap-south-1"
    )
    S3_CLIENT = boto3.client("s3", region_name=_s3_region)
except Exception as e:
    S3_CLIENT = None
    print(f"Warning: boto3 or S3 client setup failed ({e})")


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_s3_url(s3_url: str):
    """
    Parses an S3 URL and returns (bucket_name, key, filename).
    Handles URL-encoded paths and '+' as space.
    """
    parsed = urlparse(s3_url)
    bucket = parsed.netloc.split(".")[0]
    key = unquote(parsed.path.lstrip("/")).replace("+", " ")
    filename = Path(key).name
    return bucket, key, filename


def generate_presigned_s3_url(
    s3_url: str,
    bucket_name: str | None = None,
    expires_in: int = 3600,
) -> str | None:
    """
    Return a time-limited HTTPS URL for direct browser access to an S3 object.
    """
    if not s3_url or S3_CLIENT is None:
        return None
    try:
        parsed_bucket, key, _ = parse_s3_url(s3_url)
        bucket = bucket_name or parsed_bucket
        presigned_url = S3_CLIENT.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        return presigned_url
    except (ClientError, NoCredentialsError) as e:
        logger.error(f"Failed to presign S3 URL ({s3_url}): {e}")
        return None


def fetch_latest_data():
    logger.info("Fetching latest data...")

    try:
        subprocess.run(
            [sys.executable, "connect.py"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Data fetch successful")

    except subprocess.CalledProcessError as e:
        logger.error(e.stderr.strip() if e.stderr else "No stderr output")
        raise RuntimeError(
            f"Data fetch failed!"
        ) from e


def setup_df(valid_users: list[str], form_names: list[str], project_name: str):
    logger.info("Setting up dataframe...")
    data_df = pd.read_csv("projectapp_datapoint.csv") # one row per filled form
    data_df = data_df.rename(columns={"id": "datapoint_id"})

    forms_df = pd.read_csv("projectapp_dataset.csv") # one row per form type
    forms_df = forms_df.rename(columns={"id": "dataset_id", "name": "form_name"})

    users_df = pd.read_csv("auth_user.csv") # one row per user
    users_df = users_df.rename(columns={"id": "owner_id"})

    projects_df = pd.read_csv("projectapp_project.csv") # one row per project
    projects_df = projects_df.rename(columns={"id": "project_id", "name": "project_name"})

    # merging
    data_df = data_df.merge(forms_df[["dataset_id", "form_name", "project_id"]], on="dataset_id", how="left")
    data_df = data_df.merge(users_df[["owner_id", "username"]], on="owner_id", how="left")
    data_df = data_df.merge(projects_df[["project_id", "project_name"]], on="project_id", how="left")

    data_df = data_df[data_df["username"].isin(valid_users)]

    data_df = data_df[data_df["project_name"] == project_name]

    data_df = data_df[data_df["form_name"].isin(form_names)]

    # removing rows where is_deleted is True
    data_df = data_df[~data_df["is_deleted"]]

    # safely parse python literals
    data_df["data"] = data_df["data"].apply(ast.literal_eval)

    return data_df
