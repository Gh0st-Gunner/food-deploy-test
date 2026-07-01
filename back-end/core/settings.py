from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 10800

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # SMTP for Email Flow
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "noreply@munchin.com"

    # PostgreSQL (use sqlite:///vnfood.db for local dev without PostgreSQL)
    database_url: str = "sqlite:///vnfood.db"

    # S3 / MinIO
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "vn-food-images"
    s3_region: str = "us-east-1"

    # Celery (use "memory://" for local dev without Redis)
    celery_broker_url: str = "memory://"
    celery_result_backend: str = "cache+memory://"

    # API Keys
    usda_api_key: str = ""
    fatsecret_client_id: str = ""
    fatsecret_client_secret: str = ""
    ollama_token: str = ""
    ollama_host: str = "http://localhost:11434"
    gemini_api_key: str = ""

    # Model paths
    models_dir: str = "models"

    # Model IDs (HuggingFace)
    grounding_dino_model: str = "IDEA-Research/grounding-dino-tiny"
    sam2_model: str = "facebook/sam2.1-hiera-small"
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"

    # Defaults
    default_box_threshold: float = 0.3
    default_text_threshold: float = 0.25
    plate_diameter_cm: float = 25.0

    # Worker config
    worker_preload_enabled: bool = True

    # Cache TTL (seconds)
    nutrition_cache_ttl: int = 3600
    class_mapping_cache_ttl: int = 86400

    model_config = {
        "env_file": ".env",
        "env_prefix": "VNFOOD_",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()