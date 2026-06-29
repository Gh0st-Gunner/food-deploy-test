import redis

from core.settings import get_settings
from core.database import get_session

settings = get_settings()

_redis_client = None


def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
        )
    return _redis_client


def get_db():
    """FastAPI dependency that yields a database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()