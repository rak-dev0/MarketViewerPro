import time
from typing import Any, Optional
from functools import lru_cache

from ..config import get_settings
from ..storage.cache_sqlite import SQLiteCache

settings = get_settings()

_MEMORY_CACHE: dict[str, tuple[Any, float]] = {}


@lru_cache
def get_sqlite_cache() -> SQLiteCache:
    url = settings.CACHE_DB_URL
    if url.startswith("sqlite:///"):
        path = url.replace("sqlite:///", "")
    else:
        # minimal: support only sqlite path for now
        path = "./storage/cache.db"
    return SQLiteCache(db_path=path)


def cache_key_price(ticker: str, range_: str, interval: str) -> str:
    return f"price:{ticker.upper()}:{range_}:{interval}"


def get_cached(key: str) -> Optional[tuple[Any, bool]]:
    """
    Returns (value, is_stale) if cache usable, else None.
    is_stale=True means: older than TTL but younger than max_age.
    """
    ttl = settings.CACHE_TTL_SECONDS
    max_age = settings.CACHE_MAX_AGE_SECONDS
    now = time.time()

    # In-memory
    if key in _MEMORY_CACHE:
        value, ts = _MEMORY_CACHE[key]
        age = now - ts
        if age <= max_age:
            return value, age > ttl

    # SQLite
    sqlite_cache = get_sqlite_cache()
    row = sqlite_cache.get(key)
    if not row:
        return None

    value, ts = row
    age = now - ts
    if age > max_age:
        return None

    _MEMORY_CACHE[key] = (value, ts)
    return value, age > ttl


def set_cached(key: str, value: Any) -> None:
    now = time.time()
    _MEMORY_CACHE[key] = (value, now)
    sqlite_cache = get_sqlite_cache()
    sqlite_cache.set(key, value)

def delete_cached(key: str) -> None:
    """
    Remove a broken cache entry from memory and SQLite.
    """
    _MEMORY_CACHE.pop(key, None)
    sqlite_cache = get_sqlite_cache()
    try:
        sqlite_cache.delete(key)
    except Exception:
        # Cache delete failures should never break the app.
        pass
