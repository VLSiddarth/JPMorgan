# src/data/repository/redis_cache.py

"""
Redis Cache Layer
-----------------

Production-ready caching utility used for:
- Market data (latest quotes, recent OHLCV)
- Signals (latest signal snapshot)
- Portfolio states
- Macro indicators

Implements:
- TTL-based caching
- JSON serialization
- Consistent key prefixing for namespaces
- Safe error handling
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional
from datetime import timedelta

import redis

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Thin enterprise-grade Redis cache wrapper.

    Supports:
    - get / set
    - TTL expiry
    - Namespaced keys
    - JSON serialization
    """

    def __init__(
        self,
        uri: str,
        prefix: str = "jpm_cache",
        default_ttl_sec: int = 3600,
    ) -> None:
        """
        Args:
            uri: redis://localhost:6379/0
            prefix: key namespace prefix
            default_ttl_sec: default expiry time (seconds)
        """
        try:
            self.client = redis.Redis.from_url(uri, decode_responses=True)
            self.prefix = prefix
            self.default_ttl = default_ttl_sec
            # Test connection
            self.client.ping()
            logger.info("RedisCache initialized with prefix '%s'", prefix)
        except Exception as e:
            logger.exception("Failed to initialize RedisCache: %s", e)
            raise

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _format_key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def _serialize(self, value: Any) -> str:
        try:
            return json.dumps(value, default=str)
        except Exception as e:
            logger.error("JSON serialization error: %s", e)
            raise

    def _deserialize(self, value: Optional[str]) -> Any:
        if value is None:
            return None
        try:
            return json.loads(value)
        except Exception:
            # Fallback: return raw text
            logger.warning("JSON deserialization failed; returning raw value")
            return value

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def set(
        self,
        key: str,
        value: Any,
        ttl_sec: Optional[int] = None,
    ) -> None:
        """
        Set key to value with optional TTL.

        Args:
            key: logical identifier
            value: JSON-serializable object
            ttl_sec: TTL override (seconds)
        """
        ttl = ttl_sec or self.default_ttl
        redis_key = self._format_key(key)
        try:
            payload = self._serialize(value)
            self.client.set(redis_key, payload, ex=ttl)
            logger.debug("Redis SET %s (ttl=%s)", redis_key, ttl)
        except Exception as e:
            logger.exception("Failed to SET key '%s': %s", key, e)
            raise

    def get(self, key: str) -> Any:
        """
        Get a value by key.

        Returns:
            Deserialized JSON or None
        """
        redis_key = self._format_key(key)
        try:
            raw = self.client.get(redis_key)
            return self._deserialize(raw)
        except Exception as e:
            logger.exception("Failed to GET key '%s': %s", key, e)
            return None

    def delete(self, key: str) -> None:
        """Delete a key from Redis."""
        redis_key = self._format_key(key)
        try:
            self.client.delete(redis_key)
            logger.debug("Redis DEL %s", redis_key)
        except Exception as e:
            logger.exception("Failed to DELETE key '%s': %s", key, e)

    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        redis_key = self._format_key(key)
        try:
            return bool(self.client.exists(redis_key))
        except Exception as e:
            logger.exception("Failed to EXISTS key '%s': %s", key, e)
            return False

    def expire(self, key: str, ttl_sec: int) -> None:
        """Manually update TTL for a key."""
        redis_key = self._format_key(key)
        try:
            self.client.expire(redis_key, ttl_sec)
        except Exception as e:
            logger.exception("Failed to EXPIRE key '%s': %s", key, e)

    def flush_namespace(self) -> None:
        """Delete all keys under namespace prefix."""
        pattern = f"{self.prefix}:*"
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
            logger.info("Flushed Redis namespace '%s' (%d keys)", self.prefix, len(keys))
        except Exception:
            logger.exception("Failed to flush namespace")
