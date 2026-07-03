import json
import time
from core.settings import get_settings

settings = get_settings()

_redis_client = None
_cache_mode = None  # "redis" or "memory"
_memory_cache = {}  # Fallback in-memory cache
_last_redis_try_time = 0.0


def _get_cache_mode():
    global _cache_mode, _redis_client
    if _cache_mode is not None:
        return _cache_mode

    _try_connect_redis()
    return _cache_mode or "memory"



def _try_connect_redis():
    """Attempt to connect to Redis. Call this only from a background thread or worker."""
    global _cache_mode, _redis_client
    try:
        import redis
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password or None,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        client.ping()
        _redis_client = client
        _cache_mode = "redis"
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception(f"Redis connection failed to {settings.redis_host}:{settings.redis_port}")
        _cache_mode = "memory"


def get_redis():
    """Get Redis client (raises if Redis is not available)."""
    mode = _get_cache_mode()
    if mode == "redis":
        return _redis_client
    raise RuntimeError("Redis is not available")


def _cache_key(prefix: str, *args) -> str:
    return f"{prefix}:{':'.join(str(a) for a in args)}"


def get_cached(key: str) -> dict | list | None:
    try:
        mode = _get_cache_mode()
        if mode == "redis":
            data = _redis_client.get(key)
            if data:
                return json.loads(data)
            return None
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Redis get failed for key {key}, falling back to memory: {e}")

    # In-memory fallback
    entry = _memory_cache.get(key)
    if entry and entry["expires"] > time.time():
        return entry["value"]
    if entry:
        del _memory_cache[key]
    return None


def set_cached(key: str, value: dict | list, ttl: int = 3600):
    try:
        mode = _get_cache_mode()
        if mode == "redis":
            _redis_client.setex(key, ttl, json.dumps(value, default=str))
            return
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Redis set failed for key {key}, falling back to memory: {e}")

    # In-memory fallback
    now = time.time()
    if len(_memory_cache) >= 1000:
        expired_keys = [k for k, v in _memory_cache.items() if v["expires"] <= now]
        for k in expired_keys:
            del _memory_cache[k]
        
        # If still too large, evict the oldest entry (FIFO)
        if len(_memory_cache) >= 1000:
            oldest_key = next(iter(_memory_cache))
            del _memory_cache[oldest_key]
            
    _memory_cache[key] = {"value": value, "expires": now + ttl}


def delete_cached(key: str):
    try:
        mode = _get_cache_mode()
        if mode == "redis":
            _redis_client.delete(key)
            return
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Redis delete failed for key {key}, falling back to memory: {e}")

    _memory_cache.pop(key, None)


# --- Nutrition-specific caches ---

def get_usda_cached(query: str) -> dict | None:
    return get_cached(_cache_key("usda", query))


def set_usda_cached(query: str, data: dict):
    set_cached(_cache_key("usda", query), data, ttl=settings.nutrition_cache_ttl)


def get_usda_nutrients_cached(fdc_id: str) -> dict | None:
    return get_cached(_cache_key("usda_nutrients", fdc_id))


def set_usda_nutrients_cached(fdc_id: str, data: dict):
    set_cached(_cache_key("usda_nutrients", fdc_id), data, ttl=settings.nutrition_cache_ttl)


def get_fatsecret_cached(query: str) -> dict | None:
    return get_cached(_cache_key("fatsecret", query))


def set_fatsecret_cached(query: str, data: dict):
    set_cached(_cache_key("fatsecret", query), data, ttl=settings.nutrition_cache_ttl)


def get_class_mapping_cached(class_name: str) -> dict | None:
    return get_cached(_cache_key("class_map", class_name))


def set_class_mapping_cached(class_name: str, data: dict):
    set_cached(_cache_key("class_map", class_name), data, ttl=settings.class_mapping_cache_ttl)


def get_ingredient_nutrition_cached(label: str) -> dict | None:
    return get_cached(_cache_key("ing_nutrition", label))


def set_ingredient_nutrition_cached(label: str, data: dict):
    set_cached(_cache_key("ing_nutrition", label), data, ttl=settings.nutrition_cache_ttl)