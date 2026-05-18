import pandas as pd
import subprocess
import logging
from pathlib import Path
from urllib.parse import urlparse, unquote
from PIL import Image
from tqdm import tqdm
import ast

try:
    from google.cloud import storage

    GCS_CLIENT = storage.Client()
except Exception as e:
    GCS_CLIENT = None
    print(f"Warning: GCS client setup failed ({e})")

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    S3_CLIENT = boto3.client("s3")
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


def upload_to_gcp(local_path: Path, gcp_prefix: Path, bucket_name: str) -> str:
    """
    Uploads a local file to GCS.
    The uploaded file name is based on the stem of the local file name (matching S3 stem).
    
    Args:
        local_path (Path): Local file to upload.
        gcp_prefix (Path): Folder prefix (e.g., Path("images/2025/")).
        bucket_name (str): Target GCS bucket.

    Returns:
        str: GCS URL of the uploaded file, or None on failure.
    """
    try:
        if not local_path.exists():
            logger.error(f"Local file not found: {local_path}")
            return None

        gcp_filename = local_path.name
        destination_blob_name = str((gcp_prefix / gcp_filename).as_posix())

        # Initialize GCS client and upload
        bucket = GCS_CLIENT.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        blob.upload_from_filename(str(local_path))

        return f"gs://{bucket_name}/{destination_blob_name}"

    except Exception as e:
        logger.error(f"Failed to upload {local_path} to GCP: {e}")
        return None


def fetch_latest_data():
    logger.info("Fetching latest data...")

    try:
        subprocess.run(
            ["python", "connect.py"],
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


def ensure_s3_image_cached(
    s3_url: str,
    cache_dir: Path,
    bucket_name: str,
) -> Path | None:
    """
    Ensure an S3 object is present on local disk (download if missing).
    Returns the local file path, or None on failure.
    """
    if not s3_url or S3_CLIENT is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    _, _, filename = parse_s3_url(s3_url)
    local_path = cache_dir / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    return download_image_from_s3(s3_url, local_path, bucket_name)


def download_image_from_s3(s3_url: str, save_path: Path, bucket_name: str):
    """
    Downloads an image from an S3 URL, validates image file and saves it locally.
    Returns the local save path if successful, else None.
    """
    try:
        _, key, _ = parse_s3_url(s3_url)

        S3_CLIENT.download_file(bucket_name, key, str(save_path.resolve()))

        # Validate image
        with Image.open(save_path) as img:
            img.verify()

        return save_path
    except (ClientError, NoCredentialsError, FileNotFoundError) as e:
        logger.error(f"Failed to download from S3 ({s3_url}): {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in download_image_from_s3: {e}")
        return None


def push_images_to_gcp(df: pd.DataFrame, gcp_prefix: Path, gcs_bucket_name: str, s3_bucket_name: str) -> pd.DataFrame:
    """
    Copy images from s3 to GCP, add gcp_url column in df.
    Expects column with name s3_url in the df.
    """
    updated_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        if row["s3_url"] is None:
            updated_rows.append({**row.to_dict(), "gcp_url": None})
            continue
        gcp_url = copy_image_from_s3_to_gcp(
            s3_url=row["s3_url"],
            gcp_prefix=gcp_prefix,
            gcs_bucket_name=gcs_bucket_name,
            s3_bucket_name=s3_bucket_name,
        )
        updated_rows.append({**row.to_dict(), "gcp_url": gcp_url})
    return pd.DataFrame(updated_rows)


def copy_image_from_s3_to_gcp(
    s3_url: str,
    gcp_prefix: Path,
    gcs_bucket_name: str,
    s3_bucket_name: str,
    tmp_dir: Path = Path("/tmp"),
):
    """
    Download an image from S3 and upload it to GCP.

    Returns:
        gcp_url if successful, else None
    """
    _, _, s3_filename = parse_s3_url(s3_url)
    local_path = tmp_dir / s3_filename

    downloaded_path = download_image_from_s3(
        s3_url=s3_url,
        save_path=local_path,
        bucket_name=s3_bucket_name,
    )

    if not downloaded_path:
        return None

    gcp_url = upload_to_gcp(
        local_path=downloaded_path,
        gcp_prefix=gcp_prefix,
        bucket_name=gcs_bucket_name,
    )

    if downloaded_path.exists():
        downloaded_path.unlink()
    
    return gcp_url


def get_gcp_url_from_s3(s3_url: str, gcp_prefix: Path, gcs_bucket_name: str) -> str:
    """
    Returns what the GCP URL *would be* for a given S3 URL.
    Purely deterministic. No downloads/uploads.
    """
    _, _, s3_filename = parse_s3_url(s3_url)
    return f"gs://{gcs_bucket_name}/{(gcp_prefix / s3_filename).as_posix()}"