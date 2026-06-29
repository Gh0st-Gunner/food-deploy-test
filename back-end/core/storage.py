import os
from pathlib import Path

from core.settings import get_settings

settings = get_settings()

# Use local file storage when S3 is not available (local dev mode)
_storage_mode = None
_local_storage_dir = None


def _get_storage_mode():
    global _storage_mode, _local_storage_dir
    if _storage_mode is not None:
        return _storage_mode

    # Check if S3 is configured by trying to connect
    s3_endpoint = settings.s3_endpoint
    if s3_endpoint and not s3_endpoint.startswith("http://localhost"):
        # Likely a real S3 endpoint, try to use it
        try:
            import boto3
            client = boto3.client(
                "s3",
                endpoint_url=s3_endpoint,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=settings.s3_secret_key,
            )
            try:
                client.head_bucket(Bucket=settings.s3_bucket)
            except Exception:
                try:
                    client.create_bucket(Bucket=settings.s3_bucket)
                except Exception:
                    pass
            _storage_mode = "s3"
            return "s3"
        except Exception:
            pass

    # Fallback to local file storage
    _local_storage_dir = Path("storage")
    (_local_storage_dir / "images").mkdir(parents=True, exist_ok=True)
    (_local_storage_dir / "results").mkdir(parents=True, exist_ok=True)
    _storage_mode = "local"
    return "local"


def _get_s3_client():
    import boto3
    from botocore.config import Config as BotoConfig

    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        region_name=settings.s3_region,
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except Exception:
        client.create_bucket(Bucket=settings.s3_bucket)
    return client


def upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload bytes and return the key."""
    mode = _get_storage_mode()
    if mode == "local":
        filepath = _local_storage_dir / key
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_bytes(data)
        return key

    # S3 mode
    client = _get_s3_client()
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    return key


def download_bytes(key: str) -> bytes:
    """Download bytes by key."""
    mode = _get_storage_mode()
    if mode == "local":
        filepath = _local_storage_dir / key
        return filepath.read_bytes()

    # S3 mode
    client = _get_s3_client()
    response = client.get_object(Bucket=settings.s3_bucket, Key=key)
    return response["Body"].read()


def upload_image(job_id: str, image_bytes: bytes, content_type: str = "image/jpeg") -> str:
    """Upload an original image and return the key."""
    key = f"images/{job_id}/original.jpg"
    return upload_bytes(key, image_bytes, content_type)


def upload_result_image(job_id: str, image_type: str, image_bytes: bytes) -> str:
    """Upload a result image (overlay, depth map) and return the key."""
    key = f"results/{job_id}/{image_type}.png"
    return upload_bytes(key, image_bytes, "image/png")


def get_presigned_url(key: str, expires_in: int = 3600) -> str:
    """Generate a URL for accessing the stored object."""
    mode = _get_storage_mode()
    if mode == "local":
        # Return a relative API path for local dev
        return f"/api/v1/files/{key}"

    # S3 mode
    client = _get_s3_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.s3_bucket, "Key": key},
        ExpiresIn=expires_in,
    )