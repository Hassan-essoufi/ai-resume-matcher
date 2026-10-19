"""Cache for LLM and embedding results.

Same text + same model = same answer, so we never pay the API twice for it.
Keys are SHA-256 hashes, so no raw text is stored in the keys.

RedisCache: shared by every API worker/container, with automatic expiry (TTL).
NullCache : no caching (unit tests only).

A cache is only an optimization: if Redis is down, calls go straight to OpenAI
instead of failing.
"""
import hashlib
import json
import logging

import redis

log = logging.getLogger(__name__)


def make_key(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


class RedisCache:
    def __init__(self, url: str, ttl_days: int = 30, prefix: str = "rm:", client=None):
        if client is None:
            client = redis.Redis.from_url(url, socket_timeout=2, socket_connect_timeout=2)
        self.r = client
        self.ttl = int(ttl_days * 86400) if ttl_days else None
        self.prefix = prefix

    key = staticmethod(make_key)

    def get(self, key: str):
        try:
            raw = self.r.get(self.prefix + key)
        except Exception as e:
            log.warning("redis get failed: %s", e)
            return None
        return json.loads(raw) if raw else None

    def set(self, key: str, value) -> None:
        try:
            self.r.set(self.prefix + key, json.dumps(value), ex=self.ttl)
        except Exception as e:
            log.warning("redis set failed: %s", e)

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False


class NullCache:
    key = staticmethod(make_key)

    def get(self, key):
        return None

    def set(self, key, value):
        pass

    def ping(self) -> bool:
        return True


def make_cache(settings):
    if settings.redis_url:
        cache = RedisCache(settings.redis_url, settings.cache_ttl_days)
        log.info("cache: redis %s", "reachable" if cache.ping() else "NOT reachable, calls will not be cached")
        return cache
    log.warning("REDIS_URL not set: LLM and embedding calls are not cached")
    return NullCache()
