import logging

logger = logging.getLogger(__name__)

def check_rate_limit(redis_client, key: str, max_requests: int, window_seconds: int) -> bool:
    """
    Returns True if the request is allowed (within limit), False if rate-limited.
    """
    try:
        current = redis_client.get(key)
        if current is not None and int(current) >= max_requests:
            return False
        
        # Increment and set expiry
        pipe = redis_client.pipeline()
        pipe.incr(key)
        # If it was a new key, we set the expiration
        # Note: incr returns the new value, we check the first response in execute
        pipe.ttl(key)
        res = pipe.execute()
        
        new_val = res[0]
        ttl_val = res[1]
        
        # If key is new or doesn't have an expiry set, set the expiration window
        if new_val == 1 or ttl_val == -1:
            redis_client.expire(key, window_seconds)
            
        return True
    except Exception as e:
        # In case Redis is down or experiencing issues, we log the failure 
        # and fail-safe (allow the request so users aren't locked out)
        logger.error(f"Redis rate limiting failed for key {key}: {e}", exc_info=True)
        return True
